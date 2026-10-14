"""按群聊维度临时切换系统提示词的可复用工具。

用法：
    set_group_prompt(group_id, "secret_mode_prompt", 600)   # 在该群生效 600 秒
    set_group_prompt(group_id, "secret_mode_prompt", None)  # 本次服务启动期间持久生效
    reset_group_prompt(group_id)                            # 恢复默认 sys_prompt
    get_active_prompt_name(group_id)                        # 当前生效的提示词变量名
"""

import time

from . import system_prompt as _system_prompt

DEFAULT_PROMPT_VAR = "sys_prompt"

# group_id -> (system_prompt.py 变量名, 过期时刻 monotonic 秒；None = 服务期间持久)
_overrides: dict[int, tuple[str, float | None]] = {}


def get_prompt_text(prompt_var_name: str) -> str:
    """按变量名读取 system_prompt.py 中的提示词文本，变量名非法时报错。"""
    value = getattr(_system_prompt, prompt_var_name, None)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"system_prompt.py 中不存在字符串变量 {prompt_var_name!r}")
    return value


def set_group_prompt(
    group_id: int,
    prompt_var_name: str,
    duration_seconds: float | None,
) -> None:
    """仅在指定群聊切换系统提示词；duration_seconds=None 表示服务运行期间持久生效。"""
    get_prompt_text(prompt_var_name)  # 提前校验变量名
    expires_at = (
        None if duration_seconds is None else time.monotonic() + duration_seconds
    )
    _overrides[int(group_id)] = (prompt_var_name, expires_at)


def reset_group_prompt(group_id: int) -> None:
    """清除指定群聊的提示词覆盖，恢复默认 sys_prompt。"""
    _overrides.pop(int(group_id), None)


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
