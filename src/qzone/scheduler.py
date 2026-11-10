"""定时说说调度：每天在活跃时段内随机挑 N 个时间点（间隔不少于最小间隔），计划写入
data/qzone_plan.json，重启后沿用；错过 30 分钟以内的时间点补发，超过则跳过。"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from websockets.asyncio.client import ClientConnection

import config

from ..qq_ws_context import current_qq_websocket
from .events import pick_event
from .writer import write_and_post

PLAN_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "qzone_plan.json"
_TICK_SECONDS = 30
_LATE_TOLERANCE = timedelta(minutes=30)
_PLAN_LEAD = timedelta(minutes=5)


def _at(day: date, hhmm: str) -> datetime:
    hour, _, minute = hhmm.partition(":")
    naive = datetime.combine(day, datetime.min.time()).replace(
        hour=int(hour), minute=int(minute or 0)
    )
    return naive.astimezone()


def make_plan(day: date, now: datetime, count: int) -> list[datetime]:
    """在 [max(开始, 现在+5分钟), 结束] 内随机取 count 个时间点，两两间隔 ≥ 最小间隔；
    时段不够时自动减少条数。"""
    start = max(_at(day, config.QZONE_AUTO_START), now + _PLAN_LEAD)
    end = _at(day, config.QZONE_AUTO_END)
    gap = timedelta(hours=config.QZONE_AUTO_MIN_GAP_HOURS)
    span = (end - start).total_seconds()
    for n in range(count, 0, -1):
        free = span - (n - 1) * gap.total_seconds()
        if free < 0:
            continue
        # 在压缩后的区间里取 n 个随机点，再按序加回间隔：均匀随机且必然满足最小间隔
        offsets = sorted(random.uniform(0, free) for _ in range(n))
        return [
            start + timedelta(seconds=off + i * gap.total_seconds())
            for i, off in enumerate(offsets)
        ]
    return []


def _load_plan() -> dict:
    try:
        data = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_plan(plan: dict) -> None:
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PLAN_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PLAN_PATH)


def _ensure_today_plan(now: datetime) -> dict:
    plan = _load_plan()
    today = now.date().isoformat()
    if plan.get("date") == today and isinstance(plan.get("slots"), list):
        return plan
    times = make_plan(now.date(), now, config.QZONE_AUTO_DAILY)
    plan = {
        "date": today,
        "slots": [{"at": t.isoformat(timespec="seconds"), "status": "pending"} for t in times],
    }
    _save_plan(plan)
    print(f"[qzone] 今日说说计划: {[t.strftime('%H:%M') for t in times]}")
    return plan


async def _fire(slot: dict, websocket: ClientConnection) -> None:
    event = await pick_event()
    slot["event"] = event.kind
    try:
        ok = await write_and_post(event, websocket)
        slot["status"] = "done" if ok else "skipped"
    except asyncio.CancelledError:
        raise
    except Exception as e:
        slot["status"] = "failed"
        print(f"[qzone] 定时说说失败: {e}")


async def qzone_scheduler_loop(websocket: ClientConnection) -> None:
    """随 NapCat 连接启动；按当日计划到点发布说说。"""
    if not config.QZONE_AUTO_ENABLED:
        return
    token = current_qq_websocket.set(websocket)
    try:
        while True:
            now = datetime.now().astimezone()
            try:
                plan = _ensure_today_plan(now)
                changed = False
                for slot in plan["slots"]:
                    if slot.get("status") != "pending":
                        continue
                    at = datetime.fromisoformat(slot["at"])
                    if at > now:
                        continue
                    if now - at > _LATE_TOLERANCE:
                        slot["status"] = "missed"
                    else:
                        # 先落盘为 running，避免发布中途重启后重复发
                        slot["status"] = "running"
                        _save_plan(plan)
                        await _fire(slot, websocket)
                    changed = True
                    break  # 一次只处理一个时间点
                if changed:
                    _save_plan(plan)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[qzone] 调度出错: {e}")
            await asyncio.sleep(_TICK_SECONDS)
    finally:
        current_qq_websocket.reset(token)
