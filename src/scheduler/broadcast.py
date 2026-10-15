"""Scheduled group tasks: sign-in at 00:00; proactive messages at 08:00 and 19:00 (local)."""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Literal

import config
from websockets.asyncio.client import ClientConnection

from ..action import interaction
from ..agent.init_llm import default_group_agent, extract_final_assistant_text
from ..handler.commands import is_group_enabled
from ..handler.message_handler import _RATE_LIMIT_REPLY_TEXT, _is_rate_limit_error
from ..qq_ws_context import current_qq_websocket


async def _run_broadcast_for_group(
    websocket: ClientConnection,
    *,
    group_id: int,
    slot: Literal["morning", "evening"],
) -> None:
    result = await default_group_agent.ainvoke_scheduled_broadcast(
        group_id=group_id,
        group_name="",
        slot=slot,
    )
    body = extract_final_assistant_text(result.get("messages", []))
    for segment in interaction.split_message_segments(body):
        await interaction.send_group_msg(group_id, segment, websocket)


async def run_scheduled_broadcast(
    websocket: ClientConnection,
    slot: Literal["morning", "evening"],
) -> None:
    if not config.PROACTIVE_GROUP_IDS:
        return
    token = current_qq_websocket.set(websocket)
    try:
        for gid in config.PROACTIVE_GROUP_IDS:
            if not is_group_enabled(gid):
                continue  # 该群已被管理员 /shutdown
            try:
                await _run_broadcast_for_group(websocket, group_id=gid, slot=slot)
            except Exception as e:
                print(f"定时群发失败 group_id={gid} slot={slot}: {e}")
                if _is_rate_limit_error(e):
                    try:
                        await interaction.send_group_msg(
                            gid, _RATE_LIMIT_REPLY_TEXT, websocket
                        )
                    except Exception as send_error:
                        print(f"发送限流提示失败 group_id={gid}: {send_error}")
    finally:
        current_qq_websocket.reset(token)


async def run_daily_group_sign(websocket: ClientConnection) -> None:
    """每天 0:00 对配置的群调用 NapCat set_group_sign。"""
    if not config.PROACTIVE_GROUP_IDS:
        return
    token = current_qq_websocket.set(websocket)
    try:
        for gid in config.PROACTIVE_GROUP_IDS:
            if not is_group_enabled(gid):
                continue  # 该群已被管理员 /shutdown
            try:
                await interaction.set_group_sign(gid, websocket)
            except Exception as e:
                print(f"群打卡失败 group_id={gid}: {e}")
    finally:
        current_qq_websocket.reset(token)


async def proactive_scheduler_loop(websocket: ClientConnection) -> None:
    """Wake periodically; fire sign-in, morning/evening once per calendar day in a short window after :00."""
    last_date: date | None = None
    fired: dict[str, bool] = {"sign": False, "morning": False, "evening": False}
    while True:
        await asyncio.sleep(15)
        if not config.PROACTIVE_GROUP_IDS:
            continue
        now = datetime.now().astimezone()
        today = now.date()
        if today != last_date:
            last_date = today
            fired["sign"] = False
            fired["morning"] = False
            fired["evening"] = False
        # 00:00–00:01: 群打卡
        if now.hour == 0 and now.minute <= 1 and not fired["sign"]:
            fired["sign"] = True
            await run_daily_group_sign(websocket)
        # 08:00–08:01 / 19:00–19:01: 定时问候
        elif now.hour == 8 and now.minute <= 1 and not fired["morning"]:
            fired["morning"] = True
            await run_scheduled_broadcast(websocket, "morning")
        elif now.hour == 19 and now.minute <= 1 and not fired["evening"]:
            fired["evening"] = True
            await run_scheduled_broadcast(websocket, "evening")