import json
import random

import config
from websockets.asyncio.client import ClientConnection

from ..qq_ws_context import current_qq_websocket
from .napcat_api import request_napcat_api
from .ws_sender import send_ws_event


def _resolve_ws(websocket: ClientConnection | None) -> ClientConnection:
    if websocket is not None:
        return websocket
    ws = current_qq_websocket.get()
    if ws is None:
        raise RuntimeError("未绑定 QQ WebSocket：请在 handle_message 入口设置 current_qq_websocket")
    return ws


async def get_msg(message_id: int | str, websocket: ClientConnection | None = None) -> dict | None:
    """Fetch a message by ID through NapCat's get_msg API."""
    ws = _resolve_ws(websocket)
    try:
        response = await request_napcat_api(
            ws, "get_msg", {"message_id": message_id}
        )
    except Exception as error:
        print(f"查询引用消息失败 message_id={message_id}: {error}")
        return None
    if not response or response.get("status") != "ok" or response.get("retcode") != 0:
        print(f"查询引用消息失败 message_id={message_id}: {response}")
        return None
    data = response.get("data")
    return data if isinstance(data, dict) else None


async def set_msg_emoji_like(msg_id: int, websocket: ClientConnection | None = None):
    """群中向消息贴表情（随机）"""
    emoji_id_lst = [171, 75, 10024, 128147, 128522]
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "set_msg_emoji_like",
                "params": {
                    "message_id": msg_id,
                    "emoji_id": emoji_id_lst[random.randint(0, len(emoji_id_lst) - 1)],
                    "set": True
                }
            }
        )
    )
    print(f"给消息 {msg_id} 添加了表情")


async def set_msg_emoji_like2(msg_id: int, websocket: ClientConnection | None = None):
    """群中向消息贴表情（鲸鱼）"""
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "set_msg_emoji_like",
                "params": {
                    "message_id": msg_id,
                    "emoji_id": 128051,
                    "set": True
                }
            }
        )
    )
    print(f"给消息 {msg_id} 添加了表情")


async def set_group_sign(
    group_id: int, websocket: ClientConnection | None = None
) -> None:
    """NapCat 群打卡（set_group_sign）。"""
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {"action": "send_group_sign", "params": {"group_id": group_id}}
        )
    )
    print(f"群 {group_id} 已请求 send_group_sign")


async def send_group_msg(
    group_id: int, text: str, websocket: ClientConnection | None = None
):
    """
    发送群聊文本消息
    """
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "send_group_msg",
                "params": {
                    "group_id": group_id,
                    "message": [
                        {
                            "type": "text",
                            "data": {
                                "text": text
                            }
                        },
                    ]
                }
            }
        )
    )
    print(f"在 {group_id} 发送了消息")


async def send_group_ai_record(
    group_id: int,
    text: str,
    character: str | None = None,
    websocket: ClientConnection | None = None,
):
    """Send an AI-generated voice record to a QQ group (NapCat send_group_ai_record)."""
    voice = character or config.AI_VOICE_CHARACTER
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "send_group_ai_record",
                "params": {
                    "group_id": group_id,
                    "character": voice,
                    "text": text,
                },
            }
        )
    )
    print(f"在 {group_id} 发送了 AI 语音")


async def send_group_msg_reply(
    group_id: int, msg_id: int, text: str, websocket: ClientConnection | None = None
):
    """
    文本回复群聊指定消息
    """
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "send_group_msg",
                "params": {
                    "group_id": group_id,
                    "message": [
                        {
                            "type": "reply",
                            "data": {
                                "id": str(msg_id)
                            }
                        },
                        {
                            "type": "text",
                            "data": {
                                "text": text
                            }
                        }
                    ]
                }
            }
        )
    )
    print(f"回复了 {msg_id} 消息")


async def send_group_msg_image(
    group_id: int,
    file: str,
    url: str,
    sub_type: int = 1,
    websocket: ClientConnection | None = None,
):
    """Send a single image to a QQ group (NapCat send_group_msg image segment). `sub_type` 1 is required for expression-style images."""
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "send_group_msg",
                "params": {
                    "group_id": group_id,
                    "message": [
                        {
                            "type": "image",
                            "data": {
                                "file": file,
                                "url": url,
                                "sub_type": sub_type,
                            },
                        }
                    ],
                },
            }
        )
    )
    print(f"在 {group_id} 发送了图片 {file}")
