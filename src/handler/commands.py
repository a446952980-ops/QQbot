"""群聊 “/” 指令：完全匹配触发，便于扩展。

新增一个指令只需三步：
1. 在 GroupCommand 中加一个成员（成员值即触发文本）；
2. 写一个处理函数 `async def _handle_xxx(arg, *, group_id, websocket) -> bool`
   （arg 为第二个空格分隔的参数，无参数时为 None；参数不合法返回 False 表示不触发）；
3. 在 _COMMAND_HANDLERS 中登记（管理员指令需同时加入 _ADMIN_COMMANDS）。
"""

from collections.abc import Awaitable, Callable
from enum import Enum

import config
from websockets.asyncio.client import ClientConnection

from ..action import interaction
from ..agent import group_prompt

SECRET_MODE_DEFAULT_MINUTES = 10


# ---- 群级服务开关（/shutdown 关闭、/startup 恢复；服务重启后默认全部开启） ----
_DISABLED_GROUPS: set[int] = set()


def is_group_enabled(group_id: int) -> bool:
    """该群是否处于正常处理消息的状态。"""
    return int(group_id) not in _DISABLED_GROUPS


def set_group_enabled(group_id: int, enabled: bool) -> None:
    if enabled:
        _DISABLED_GROUPS.discard(int(group_id))
    else:
        _DISABLED_GROUPS.add(int(group_id))


class GroupCommand(str, Enum):
    """群聊指令；成员值即触发的消息文本（完全匹配，大小写敏感）。"""

    SECRET_MODE = "/secret_mode"
    MODE_RESET = "/mode_reset"
    SHUTDOWN = "/shutdown"
    STARTUP = "/startup"


async def _handle_secret_mode(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/secret_mode、/secret_mode -N（N 为正整数分钟）、/secret_mode -forever"""
    minutes: int | None  # None 表示 -forever（服务运行期间持久）
    if arg is None:
        minutes = SECRET_MODE_DEFAULT_MINUTES
    elif arg == "-forever":
        minutes = None
    elif arg.startswith("-") and arg[1:].isdigit() and int(arg[1:]) > 0:
        minutes = int(arg[1:])
    else:
        return False  # 传参错误：不触发，按普通消息处理
    if minutes is None:
        group_prompt.set_group_prompt(group_id, "secret_mode_prompt", None)
        duration_desc = ""
    else:
        group_prompt.set_group_prompt(group_id, "secret_mode_prompt", minutes * 60)
        duration_desc = f"{minutes} 分钟"
    await interaction.send_group_msg(
        group_id, f"secret_mode 已在本群开启（{duration_desc}）", websocket
    )
    return True


async def _handle_mode_reset(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/mode_reset：不接收参数，恢复本群默认 sys_prompt"""
    if arg is not None:
        return False  # 带了任何参数都不触发
    group_prompt.reset_group_prompt(group_id)
    await interaction.send_group_msg(group_id, "本群已恢复默认模式", websocket)
    return True


async def _handle_shutdown(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/shutdown：仅在本群停止处理消息（仍接收并记日志），/startup 可恢复"""
    if arg is not None:
        return False  # 带了任何参数都不触发
    set_group_enabled(group_id, False)
    await interaction.send_group_msg(
        group_id, "睡觉喽", websocket
    )
    return True


async def _handle_startup(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/startup：恢复本群的消息处理"""
    if arg is not None:
        return False  # 带了任何参数都不触发
    set_group_enabled(group_id, True)
    await interaction.send_group_msg(group_id, "醒了", websocket)
    return True


_COMMAND_HANDLERS: dict[GroupCommand, Callable[..., Awaitable[bool]]] = {
    GroupCommand.SECRET_MODE: _handle_secret_mode,
    GroupCommand.MODE_RESET: _handle_mode_reset,
    GroupCommand.SHUTDOWN: _handle_shutdown,
    GroupCommand.STARTUP: _handle_startup,
}

# 仅管理员（config.ADMIN_QQ_IDS）可触发的指令
_ADMIN_COMMANDS: set[GroupCommand] = {GroupCommand.SHUTDOWN, GroupCommand.STARTUP}


async def try_handle_group_command(
    text: str,
    *,
    group_id: int,
    websocket: ClientConnection,
    sender_id: int | str | None = None,
) -> bool:
    """对以 “/” 开头的群消息做完全匹配分发；命中并执行成功返回 True，否则 False。"""
    stripped = text.strip()
    if not stripped.startswith("/"):
        return False
    tokens = stripped.split()
    if len(tokens) > 2:  # 至多 “指令 + 一个参数”，多余 token 一律不触发
        return False
    try:
        command = GroupCommand(tokens[0])
    except ValueError:
        return False  # 未注册的 “/” 消息按普通消息处理
    if command in _ADMIN_COMMANDS:
        try:
            is_admin = int(sender_id) in config.ADMIN_QQ_IDS
        except (TypeError, ValueError):
            is_admin = False
        if not is_admin:
            return False  # 非管理员不触发，按普通消息处理
    arg = tokens[1] if len(tokens) == 2 else None
    return await _COMMAND_HANDLERS[command](arg, group_id=group_id, websocket=websocket)
