import base64
import json
import random
import re
from collections.abc import Sequence
from contextvars import ContextVar
from pathlib import Path

import config
from websockets.asyncio.client import ClientConnection

from ..qq_ws_context import current_qq_websocket
from .napcat_api import request_napcat_api
from .ws_sender import send_ws_event

# 最近一次群消息事件里上报的机器人自身 QQ（合并转发自定义节点显示用）
_bot_self_id: ContextVar[str | None] = ContextVar("bot_self_id", default=None)


def set_bot_self_id(self_id: str) -> None:
    """记录机器人自身 QQ（message 事件入口调用）。"""
    _bot_self_id.set(self_id)


def _resolve_ws(websocket: ClientConnection | None) -> ClientConnection:
    if websocket is not None:
        return websocket
    ws = current_qq_websocket.get()
    if ws is None:
        raise RuntimeError("未绑定 QQ WebSocket：请在 handle_message 入口设置 current_qq_websocket")
    return ws


def split_message_segments(text: str) -> list[str]:
    """按单/多换行把文本拆成非空段落，供分段排队发送。"""
    return [seg.strip() for seg in re.split(r"\n+", text) if seg.strip()]


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


async def get_forward_msg(
    forward_id: int | str, websocket: ClientConnection | None = None
) -> list | None:
    """通过 NapCat get_forward_msg API 获取合并转发消息的子消息列表（失败返回 None）。

    同时传 message_id 与 id 以兼容不同 NapCat 版本的参数名。
    """
    ws = _resolve_ws(websocket)
    try:
        response = await request_napcat_api(
            ws,
            "get_forward_msg",
            {"message_id": str(forward_id), "id": str(forward_id)},
        )
    except Exception as error:
        print(f"查询合并转发消息失败 id={forward_id}: {error}")
        return None
    if not response or response.get("status") != "ok" or response.get("retcode") != 0:
        print(f"查询合并转发消息失败 id={forward_id}: {response}")
        return None
    data = response.get("data")
    messages = data.get("messages") if isinstance(data, dict) else data
    return messages if isinstance(messages, list) else None


async def fetch_custom_faces(
    count: int = 48, websocket: ClientConnection | None = None
) -> list[str]:
    """Fetch custom-face image URLs from NapCat's fetch_custom_face API."""
    ws = _resolve_ws(websocket)
    response = await request_napcat_api(
        ws, "fetch_custom_face", {"count": count}
    )
    if not response or response.get("status") != "ok" or response.get("retcode") != 0:
        raise RuntimeError(
            f"获取自定义表情失败: {response.get('message', 'NapCat 请求失败') if response else '请求超时'}"
        )
    data = response.get("data")
    if not isinstance(data, list):
        raise RuntimeError("获取自定义表情失败: NapCat 返回的数据不是列表")
    return [url for url in data if isinstance(url, str) and url.startswith(("http://", "https://"))]


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
    group_id: int,
    text: str,
    at: str | None = None,
    websocket: ClientConnection | None = None,
):
    """
    发送群聊文本消息；at 传 QQ 号或 "all" 时在文本前插入 at 消息段
    """
    ws = _resolve_ws(websocket)
    if at is not None and not isinstance(at, (str, int)):
        # 防御：误把 websocket 等对象按位置传进 at 时忽略，避免拼出无效 at 段
        print(f"send_group_msg 收到非法 at 参数（已忽略）: {type(at).__name__}")
        at = None
    message = []
    if at is not None:
        message.append({"type": "at", "data": {"qq": str(at)}})
    message.append({"type": "text", "data": {"text": text}})
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "send_group_msg",
                "params": {
                    "group_id": group_id,
                    "message": message,
                },
            }
        ),
    )
    print(f"在 {group_id} 发送了消息")


async def group_poke(
    group_id: int,
    user_id: int,
    websocket: ClientConnection | None = None,
):
    """
    发送群戳一戳（ws 事件 group_poke）
    """
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "group_poke",
                "params": {"group_id": str(group_id), "user_id": str(user_id)},
            }
        ),
    )
    print(f"已戳群 {group_id} 的 {user_id}")


async def send_qzone_msg(
    content: str,
    images: list[str] | None = None,
    ugc_right: int = 1,
    target_uins: list[int] | None = None,
    websocket: ClientConnection | None = None,
):
    """
    发表 QQ 空间说说（ws 事件 send_qzone_msg）
    """
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "send_qzone_msg",
                "params": {
                    "content": content,
                    "images": list(images) if images else [],
                    "ugc_right": ugc_right,
                    "target_uins": list(target_uins) if target_uins else [],
                },
            }
        )
    )
    print(f"已发表说说: {content[:50]}")


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


async def send_group_forward_msg(
    group_id: int,
    texts: Sequence[str],
    websocket: ClientConnection | None = None,
):
    """
    以合并转发消息（自定义节点）发送多条文本到群聊
    （NapCat ws 事件 send_forward_msg；节点 user_id 优先用机器人自身 QQ）
    """
    ws = _resolve_ws(websocket)
    bot_id = _bot_self_id.get()
    nodes = [
        {
            "type": "node",
            "data": {
                "user_id": str(bot_id) if bot_id else "10000",
                "nickname": "思考过程",
                "content": [{"type": "text", "data": {"text": text}}],
            },
        }
        for text in texts
    ]
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "send_forward_msg",
                "params": {"group_id": str(group_id), "message": nodes},
            }
        ),
    )
    print(f"已在 {group_id} 发送合并转发消息（{len(nodes)} 个节点）")


async def send_group_msg_record(
    group_id: int, file: str, websocket: ClientConnection | None = None
):
    """Send a voice record message to a QQ group (NapCat send_group_msg record segment).
    `file` 支持 http(s) URL / file:/// / base64:// / 本地路径。
    本地路径直接内联为 base64://：NapCat 可能运行在另一进程/容器中，其文件系统
    未必能访问本机路径（file:// 会报 Failed to open input: No such file or directory）。"""
    if not file.startswith(("http://", "https://", "file://", "base64://")):
        file = "base64://" + base64.b64encode(Path(file).read_bytes()).decode()
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
                            "type": "record",
                            "data": {"file": file},
                        }
                    ],
                },
            }
        ),
    )
    print(f"在 {group_id} 发送了语音 {file}")


async def send_group_msg_image(
    group_id: int,
    file: str,
    url: str | Sequence[str],
    sub_type: int = 1,
    websocket: ClientConnection | None = None,
    at_user: int | None = None,
):
    """Send one or more images to a QQ group (NapCat send_group_msg image segments).
    `sub_type` 1 is required for expression-style images. `at_user` 可选：发送时先 @ 某成员。"""
    urls = [url] if isinstance(url, str) else list(url)
    message: list[dict] = []
    if at_user is not None:
        message.append({"type": "at", "data": {"qq": str(at_user)}})
    message.extend(
        {"type": "image", "data": {"file": file, "url": u, "sub_type": sub_type}}
        for u in urls
    )
    ws = _resolve_ws(websocket)
    await send_ws_event(
        ws,
        json.dumps(
            {
                "action": "send_group_msg",
                "params": {
                    "group_id": group_id,
                    "message": message,
                },
            }
        )
    )
    print(f"在 {group_id} 发送了图片 {file} x{len(urls)}")
