"""Request/response correlation for NapCat WebSocket API actions."""

from __future__ import annotations

import asyncio
import json
import uuid
from asyncio import Future

from websockets.asyncio.client import ClientConnection

from .ws_sender import send_ws_event

_REQUEST_TIMEOUT_SECONDS = 15
_pending_requests: dict[str, tuple[ClientConnection, Future[dict]]] = {}


async def request_napcat_api(
    websocket: ClientConnection,
    action: str,
    params: dict,
) -> dict | None:
    """Send an API action with an echo ID and wait for its matching response."""
    echo = uuid.uuid4().hex
    future: Future[dict] = asyncio.get_running_loop().create_future()
    _pending_requests[echo] = (websocket, future)
    try:
        await send_ws_event(
            websocket,
            json.dumps(
                {"action": action, "params": params, "echo": echo},
                ensure_ascii=False,
            ),
        )
        return await asyncio.wait_for(future, timeout=_REQUEST_TIMEOUT_SECONDS)
    except TimeoutError:
        return None
    finally:
        _pending_requests.pop(echo, None)


def resolve_napcat_api_response(event: dict) -> bool:
    """Resolve a pending API request; return whether the event was its response."""
    echo = event.get("echo")
    if echo is None:
        return False
    pending = _pending_requests.pop(str(echo), None)
    if pending is None:
        return False
    _, future = pending
    if not future.done():
        future.set_result(event)
    return True


def fail_pending_napcat_api_requests(websocket: ClientConnection) -> None:
    """Unblock API callers when their WebSocket connection is closing."""
    for echo, (pending_websocket, future) in tuple(_pending_requests.items()):
        if pending_websocket is websocket:
            _pending_requests.pop(echo, None)
            if not future.done():
                future.set_result(
                    {
                        "status": "failed",
                        "retcode": -1,
                        "message": "NapCat WebSocket disconnected",
                    }
                )
