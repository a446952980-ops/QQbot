"""Serialize outbound NapCat WebSocket events and enforce a minimum send interval."""

from __future__ import annotations

import asyncio
from asyncio import Future, Queue, QueueEmpty, Task

import config
from websockets.asyncio.client import ClientConnection

_QueuedEvent = tuple[ClientConnection, str, Future[None]]
_outbound_events: Queue[_QueuedEvent] = Queue()
_sender_task: Task[None] | None = None
_last_sent_at: float | None = None


def _ensure_sender_task() -> None:
    global _sender_task
    if _sender_task is None or _sender_task.done():
        _sender_task = asyncio.create_task(_drain_outbound_events())


async def send_ws_event(websocket: ClientConnection, payload: str) -> None:
    """Queue a raw WebSocket event and wait until it has been sent or failed."""
    future: Future[None] = asyncio.get_running_loop().create_future()
    _outbound_events.put_nowait((websocket, payload, future))
    _ensure_sender_task()
    await future


async def _drain_outbound_events() -> None:
    global _last_sent_at, _sender_task
    current_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                websocket, payload, future = _outbound_events.get_nowait()
            except QueueEmpty:
                return

            try:
                if _last_sent_at is not None:
                    delay = config.WS_SEND_INTERVAL_SECONDS - (loop.time() - _last_sent_at)
                    if delay > 0:
                        await asyncio.sleep(delay)
                await websocket.send(payload)
            except asyncio.CancelledError:
                if not future.done():
                    future.cancel()
                raise
            except Exception as error:
                if not future.done():
                    future.set_exception(error)
            else:
                _last_sent_at = loop.time()
                if not future.done():
                    future.set_result(None)
            finally:
                _outbound_events.task_done()
    finally:
        if _sender_task is current_task:
            _sender_task = None
        if not _outbound_events.empty():
            _ensure_sender_task()
