"""按群聊维度切换系统提示词的可复用工具。

用法：
    set_group_prompt(group_id, "secret_mode_prompt", 600)   # 在该群生效 600 秒
    set_group_prompt(group_id, "secret_mode_prompt", None)  # 持久生效（写入缓存文件）
    reset_group_prompt(group_id)                            # 恢复默认 sys_prompt
    get_active_prompt_name(group_id)                        # 当前生效的提示词变量名

永久切换（duration_seconds=None）会写入 data/group_modes.json，
服务重启时自动加载各群的最后持久 mode；限时切换不持久化。
"""

import json
import time
from pathlib import Path

from . import system_prompt as _system_prompt

DEFAULT_PROMPT_VAR = "sys_prompt"

# group_id -> (system_prompt.py 变量名, 过期时刻 monotonic 秒；None = 持久)
_overrides: dict[int, tuple[str, float | None]] = {}

_MODE_STORE_PATH = Path("data/group_modes.json")


def get_prompt_text(prompt_var_name: str) -> str:
    """按变量名读取 system_prompt.py 中的提示词文本，变量名非法时报错。"""
    value = getattr(_system_prompt, prompt_var_name, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"system_prompt.py 中不存在字符串变量 {prompt_var_name!r}")
    return value


def _load_forever_overrides() -> dict[int, str]:
    """启动时从缓存文件加载各群的持久 mode；文件缺失/损坏/变量名非法的条目跳过。"""
    try:
        raw = json.loads(_MODE_STORE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    loaded: dict[int, str] = {}
    for key, name in raw.items():
        try:
            get_prompt_text(name)  # 校验变量名仍存在且为非空字符串
            loaded[int(key)] = name
        except (ValueError, TypeError):
            continue
    return loaded


def _save_forever_overrides() -> None:
    """把持久覆盖写入缓存文件（临时文件 + replace，避免写一半损坏）。"""
    _MODE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _MODE_STORE_PATH.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {str(g): n for g, n in _forever_overrides.items()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(_MODE_STORE_PATH)


_forever_overrides = _load_forever_overrides()
for _gid, _name in _forever_overrides.items():
    _overrides[_gid] = (_name, None)


def set_group_prompt(
    group_id: int,
    prompt_var_name: str,
    duration_seconds: float | None,
) -> None:
    """仅在指定群聊切换系统提示词；duration_seconds=None 表示持久生效并写入缓存文件。"""
    get_prompt_text(prompt_var_name)  # 提前校验变量名
    gid = int(group_id)
    expires_at = None if duration_seconds is None else time.monotonic() + duration_seconds
    _overrides[gid] = (prompt_var_name, expires_at)
    if expires_at is None:
        _forever_overrides[gid] = prompt_var_name
        _save_forever_overrides()
    elif gid in _forever_overrides:
        # 限时切换覆盖了原持久 mode：到期后回到默认而非旧持久 mode
        _forever_overrides.pop(gid, None)
        _save_forever_overrides()


def reset_group_prompt(group_id: int) -> None:
    """清除指定群聊的提示词覆盖（含持久缓存），恢复默认 sys_prompt。"""
    gid = int(group_id)
    _overrides.pop(gid, None)
    if _forever_overrides.pop(gid, None) is not None:
        _save_forever_overrides()


def get_active_prompt_name(group_id: int) -> str:
    """返回该群聊当前生效的提示词变量名（无覆盖或已过期则返回默认）。"""
    entry = _overrides.get(int(group_id))
    if entry is None:
        return DEFAULT_PROMPT_VAR
    prompt_var_name, expires_at = entry
    if expires_at is not None and time.monotonic() >= expires_at:
        _overrides.pop(int(group_id), None)
        return DEFAULT_PROMPT_VAR
    return prompt_var_name
