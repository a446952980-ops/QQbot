"""群聊 “/” 指令：完全匹配触发，便于扩展。

新增一个指令只需三步：
1. 在 GroupCommand 中加一个成员（成员值即触发文本）；
2. 写一个处理函数 `async def _handle_xxx(arg, *, group_id, websocket) -> bool`
   （arg 为第二个空格分隔的参数，无参数时为 None；参数不合法返回 False 表示不触发）；
3. 在 _COMMAND_HANDLERS 中登记（管理员指令需同时加入 _ADMIN_COMMANDS）。

mode 切换类指令（_MODE_COMMANDS）额外支持可选参数 `--empty`：
切换或重置时清空该群除 background 外的上下文（处理函数需接受 empty 关键字参数）。
"""

from collections.abc import Awaitable, Callable
from enum import Enum

import config
from websockets.asyncio.client import ClientConnection

from ..action import interaction
from ..agent import conversation_history, group_prompt
from ..agent.init_llm import cancel_group_invocations, default_group_agent

TIMED_MODE_DEFAULT_MINUTES = 10


# ---- 群级服务开关（/shutdown 关闭、/startup 恢复；服务重启后默认全部开启） ----
_DISABLED_GROUPS: set[int] = set()
# 分级关闭：仅关闭被 @ 回复（/shutdown -at）或仅关闭随机插话（/shutdown -rd）
_AT_DISABLED_GROUPS: set[int] = set()
_RD_DISABLED_GROUPS: set[int] = set()


def is_group_enabled(group_id: int) -> bool:
    """该群是否处于正常处理消息的状态。"""
    return int(group_id) not in _DISABLED_GROUPS


def set_group_enabled(group_id: int, enabled: bool) -> None:
    if enabled:
        _DISABLED_GROUPS.discard(int(group_id))
    else:
        _DISABLED_GROUPS.add(int(group_id))


def is_at_reply_enabled(group_id: int) -> bool:
    """该群被 @ 时的回复逻辑是否开启（/shutdown -at 关闭）。"""
    return int(group_id) not in _AT_DISABLED_GROUPS


def is_interjection_enabled(group_id: int) -> bool:
    """该群随机插话是否开启（/shutdown -rd 关闭）。"""
    return int(group_id) not in _RD_DISABLED_GROUPS


class GroupCommand(str, Enum):
    """群聊指令；成员值即触发的消息文本（完全匹配，大小写敏感）。"""

    SECRET_MODE = "/secret_mode"
    SMART_MODE = "/smart_mode"
    MODE_RESET = "/mode_reset"
    EMPTY = "/empty"
    STOP = "/stop"
    THINKING = "/thinking"
    COMPRESS = "/compress"
    SHUTDOWN = "/shutdown"
    STARTUP = "/startup"


def _apply_timed_mode(
    group_id: int, prompt_var_name: str, arg: str | None
) -> str | None:
    """定时模式共用的参数解析与提示词切换。

    无参数 = 默认 10 分钟；`-N` = N 分钟（N 为正整数）；`-forever` = 本次服务启动期间持久。
    返回生效时长描述；参数不合法返回 None（调用方视为不触发）。
    """
    minutes: int | None  # None 表示 -forever（服务运行期间持久）
    if arg is None:
        minutes = TIMED_MODE_DEFAULT_MINUTES
    elif arg == "-forever":
        minutes = None
    elif arg.startswith("-") and arg[1:].isdigit() and int(arg[1:]) > 0:
        minutes = int(arg[1:])
    else:
        return None
    group_prompt.set_group_prompt(
        group_id, prompt_var_name, None if minutes is None else minutes * 60
    )
    return "Unlimited" if minutes is None else f"{minutes} 分钟"


async def _handle_secret_mode(
    arg: str | None, *, group_id: int, websocket: ClientConnection, empty: bool = False
) -> bool:
    """/secret_mode [-N | -forever] [--empty]：N 为正整数分钟"""
    duration_desc = _apply_timed_mode(group_id, "secret_mode_prompt", arg)
    if duration_desc is None:
        return False  # 传参错误：不触发，按普通消息处理
    if empty:
        conversation_history.clear_group_turns(group_id)
    suffix = "，上下文已清空" if empty else ""
    await interaction.send_group_msg(
        group_id, f"secret_mode 已在本群开启（{duration_desc}）{suffix}", websocket=websocket
    )
    return True


async def _handle_smart_mode(
    arg: str | None, *, group_id: int, websocket: ClientConnection, empty: bool = False
) -> bool:
    """/smart_mode [-N | -forever] [--empty]：生效时间逻辑同 /secret_mode"""
    duration_desc = _apply_timed_mode(group_id, "smart_mode_prompt", arg)
    if duration_desc is None:
        return False  # 传参错误：不触发，按普通消息处理
    if empty:
        conversation_history.clear_group_turns(group_id)
    suffix = "，上下文已清空" if empty else ""
    await interaction.send_group_msg(
        group_id, f"smart_mode 已在本群开启（{duration_desc}）{suffix}", websocket=websocket
    )
    return True


async def _handle_mode_reset(
    arg: str | None, *, group_id: int, websocket: ClientConnection, empty: bool = False
) -> bool:
    """/mode_reset [--empty]：恢复本群默认 sys_prompt"""
    if arg is not None:
        return False  # 除 --empty 外带参数都不触发
    group_prompt.reset_group_prompt(group_id)
    if empty:
        conversation_history.clear_group_turns(group_id)
    suffix = "，上下文已清空" if empty else ""
    await interaction.send_group_msg(group_id, f"本群已恢复默认模式{suffix}", websocket=websocket)
    return True


async def _handle_empty(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/empty：清空该群除 background 外的上下文；/empty -a：清空该群全部上下文"""
    if arg is None:
        conversation_history.clear_group_turns(group_id)
        await interaction.send_group_msg(
            group_id, "本群上下文已清空（background 保留）", websocket=websocket
        )
        return True
    if arg == "-a":
        conversation_history.clear_group_all(group_id)
        await interaction.send_group_msg(group_id, "本群全部上下文已清空", websocket=websocket)
        return True
    return False  # 传参错误：不触发，按普通消息处理


async def _handle_thinking(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/thinking：切换主模型思考模式（全局，默认关闭）；开启后思维链写入 data/thinking.log"""
    if arg is not None:
        return False  # 带了任何参数都不触发
    enabled = not default_group_agent.is_thinking_enabled()
    default_group_agent.set_thinking_enabled(enabled)
    text = (
        "thinking on"
        if enabled
        else "thinking off"
    )
    await interaction.send_group_msg(group_id, text, websocket=websocket)
    return True


async def _handle_stop(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/stop：强行终止本群所有进行中的模型调用（不改变本群的消息处理开关）"""
    if arg is not None:
        return False  # 带了任何参数都不触发
    count = cancel_group_invocations(group_id)
    text = (
        f"强行终止 {count}"
        if count
        else "0"
    )
    await interaction.send_group_msg(group_id, text, websocket=websocket)
    return True


async def _handle_compress(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/compress：主动触发一次对话压缩（立即压缩本群当前上下文，无参数）"""
    if arg is not None:
        return False  # 带参数不触发
    summary = await default_group_agent.compress_group_now(int(group_id))
    if summary is None:
        await interaction.send_group_msg(
            group_id, "没有可压缩的内容/正在压缩中", websocket=websocket
        )
    else:
        await interaction.send_group_msg(
            group_id, "已压缩本群上下文", websocket=websocket
        )
    return True


async def _handle_shutdown(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/shutdown：仅在本群停止处理消息（仍接收并记日志），/startup 可恢复；
    可选 -at 仅关闭被 @ 回复、-rd 仅关闭随机插话（可组合，任意顺序，无重复）；
    不带参数为整体关闭。同时强行终止本群进行中的模型调用"""
    at = rd = False
    if arg is not None:
        tokens = arg.split()
        if (
            not tokens
            or any(token not in ("-at", "-rd") for token in tokens)
            or len(set(tokens)) != len(tokens)
        ):
            return False  # 传参错误：不触发，按普通消息处理
        at = "-at" in tokens
        rd = "-rd" in tokens
    cancelled = cancel_group_invocations(group_id)
    suffix = f"（{cancelled}）" if cancelled else ""
    if not at and not rd:
        set_group_enabled(group_id, False)
        text = f"睡觉喽{suffix}"
    else:
        if at:
            _AT_DISABLED_GROUPS.add(int(group_id))
        if rd:
            _RD_DISABLED_GROUPS.add(int(group_id))
        scope = "、".join(
            label for label, on in (("at", at), ("rd", rd)) if on
        )
        text = f"睡觉喽({suffix}{scope})"
    await interaction.send_group_msg(group_id, text, websocket=websocket)
    return True


async def _handle_startup(
    arg: str | None, *, group_id: int, websocket: ClientConnection
) -> bool:
    """/startup：恢复本群的消息处理（含被 -at / -rd 分级关闭的部分）"""
    if arg is not None:
        return False  # 带了任何参数都不触发
    set_group_enabled(group_id, True)
    _AT_DISABLED_GROUPS.discard(int(group_id))
    _RD_DISABLED_GROUPS.discard(int(group_id))
    await interaction.send_group_msg(group_id, "醒了", websocket=websocket)
    return True


_COMMAND_HANDLERS: dict[GroupCommand, Callable[..., Awaitable[bool]]] = {
    GroupCommand.SECRET_MODE: _handle_secret_mode,
    GroupCommand.SMART_MODE: _handle_smart_mode,
    GroupCommand.MODE_RESET: _handle_mode_reset,
    GroupCommand.EMPTY: _handle_empty,
    GroupCommand.STOP: _handle_stop,
    GroupCommand.THINKING: _handle_thinking,
    GroupCommand.COMPRESS: _handle_compress,
    GroupCommand.SHUTDOWN: _handle_shutdown,
    GroupCommand.STARTUP: _handle_startup,
}

# 仅管理员（config.ADMIN_QQ_IDS）可触发的指令
_ADMIN_COMMANDS: set[GroupCommand] = {GroupCommand.SHUTDOWN, GroupCommand.STARTUP}

# mode 切换类指令：额外支持可选参数 --empty（切换/重置时清空该群除 background 外的上下文）
_MODE_COMMANDS: set[GroupCommand] = {
    GroupCommand.SECRET_MODE,
    GroupCommand.SMART_MODE,
    GroupCommand.MODE_RESET,
}


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
    if len(tokens) > 3:  # 至多 “指令 + 参数 + --empty”，多余 token 一律不触发
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
    empty = False
    rest = tokens[1:]
    if "--empty" in rest:
        if rest.count("--empty") > 1:
            return False  # --empty 至多出现一次
        empty = True
        rest = [token for token in rest if token != "--empty"]
    if len(rest) > 1 and command is not GroupCommand.SHUTDOWN:
        return False  # /shutdown 除外：支持 “-at -rd” 组合传参
    arg = rest[0] if rest else None
    if command is GroupCommand.SHUTDOWN and len(rest) > 1:
        arg = " ".join(rest)
    if empty and command not in _MODE_COMMANDS:
        return False  # --empty 仅 mode 切换类指令支持
    if command in _MODE_COMMANDS:
        return await _COMMAND_HANDLERS[command](
            arg, group_id=group_id, websocket=websocket, empty=empty
        )
    return await _COMMAND_HANDLERS[command](arg, group_id=group_id, websocket=websocket)
