import asyncio
import random
import re
from datetime import datetime
from urllib.parse import urlparse

import config

from ..action import interaction
from ..agent.conversation_history import (
    append_background_message,
    mark_forward_blocks,
)
from ..agent.init_llm import default_group_agent, extract_final_assistant_text
from ..qq_ws_context import current_qq_websocket
from .commands import (
    is_at_reply_enabled,
    is_group_enabled,
    is_interjection_enabled,
    try_handle_group_command,
)

from websockets.asyncio.client import ClientConnection

# 模型输出被上游内容过滤时，兼容层会抛 'str' object has no attribute 'model_dump'
_FILTERED_REPLY_TEXT = "【Filtered】"

# 模型调用触发上游限流（HTTP 429）时在群里提醒
_RATE_LIMIT_REPLY_TEXT = "token吃完了"

# 合并转发解析限制：嵌套深度 / 单层条数 / 整块截断，防止超大转发刷爆上下文
_FORWARD_MAX_DEPTH = 3
_FORWARD_MAX_ITEMS = 50
_FORWARD_BLOCK_LIMIT = 6000


def _is_filtered_content_error(error: Exception) -> bool:
    return "model_dump" in str(error)


def _is_rate_limit_error(error: Exception) -> bool:
    # \b429\b：匹配 "Error code: 429 ..." / "429 Too Many Requests"，
    # 避免误匹配时间戳、消息号等恰好包含 429 的数字串
    return re.search(r"\b429\b", str(error)) is not None


def _extract_reply_message_id(segments: list, raw_message: str) -> str | None:
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("type") != "reply":
            continue
        data = segment.get("data")
        if isinstance(data, dict) and data.get("id") is not None:
            return str(data["id"])
    match = re.search(r"\[CQ:reply,id=([^,\]]+)", raw_message)
    return match.group(1) if match else None


async def _format_forward_block(
    forward_id: str,
    websocket: ClientConnection | None,
    depth: int,
    visited: frozenset[str],
    image_collector: list[tuple[str, str]] | None = None,
    image_label: str = "合并转发内图片",
    block_collector: dict[str, str] | None = None,
) -> str:
    """拉取并按规范格式展开一条合并转发消息（支持嵌套，防循环引用）。

    image_collector 非空时，转发内图片 URL 会以 (标签, url) 收集进去，
    供 @ 回复 / 随机插话把嵌套图片作为图片块传给视觉模型。
    """
    if depth >= _FORWARD_MAX_DEPTH:
        return f"[合并转发消息（嵌套过深未展开，id={forward_id}）]"
    if forward_id in visited:
        return f"[合并转发消息（循环引用未展开，id={forward_id}）]"
    messages = await interaction.get_forward_msg(forward_id, websocket)
    if messages is None:
        return f"[合并转发消息 id={forward_id}（获取失败）]"
    if not messages:
        return "[空合并转发消息]"

    indent = "  " * depth
    lines: list[str] = [f"{indent}【合并转发消息 共{len(messages)}条】"]
    rendered = 0
    for index, item in enumerate(messages[:_FORWARD_MAX_ITEMS], start=1):
        if not isinstance(item, dict):
            continue
        sender = item.get("sender") if isinstance(item.get("sender"), dict) else {}
        sender_name = sender.get("card") or sender.get("nickname") or "未知用户"
        sender_id = sender.get("user_id") or "未知"
        event_time = item.get("time")
        time_text = (
            datetime.fromtimestamp(event_time).strftime("%m-%d %H:%M")
            if isinstance(event_time, int | float) and event_time > 0
            else "未知时间"
        )
        content = await _format_message_content(
            item.get("message") if item.get("message") is not None else item.get("content"),
            websocket,
            depth=depth + 1,
            visited=visited | {forward_id},
            image_collector=image_collector,
            image_label=f"{image_label}({sender_name})",
            block_collector=block_collector,
        )
        lines.append(f"{indent}{index}. [{time_text}] {sender_name}(QQ:{sender_id}):")
        for line in content.splitlines() or [content]:
            lines.append(f"{indent}    {line}")
        rendered += 1
    if len(messages) > _FORWARD_MAX_ITEMS:
        lines.append(f"{indent}（其余 {len(messages) - _FORWARD_MAX_ITEMS} 条未展示）")
    if rendered == 0:
        return f"[合并转发消息 id={forward_id}（无可解析内容）]"
    block = "\n".join(lines)
    if len(block) > _FORWARD_BLOCK_LIMIT:
        block = block[:_FORWARD_BLOCK_LIMIT] + "…（合并转发内容过长已截断）"
    return block


async def _format_message_content(
    message: object,
    websocket: ClientConnection | None = None,
    depth: int = 0,
    visited: frozenset[str] = frozenset(),
    image_collector: list[tuple[str, str]] | None = None,
    image_label: str = "合并转发内图片",
    block_collector: dict[str, str] | None = None,
) -> str:
    if isinstance(message, str):
        return message.strip() or "（无文字内容）"
    if not isinstance(message, list):
        return "（消息内容不可解析）"

    parts: list[str] = []
    for segment in message:
        if not isinstance(segment, dict):
            continue
        segment_type = segment.get("type")
        data = segment.get("data")
        if not isinstance(data, dict):
            data = {}
        if segment_type == "text":
            value = data.get("text")
            if isinstance(value, str) and value:
                parts.append(value)
        elif segment_type == "image":
            summary = data.get("summary") or data.get("file")
            image_url = data.get("url")
            parsed = urlparse(image_url) if isinstance(image_url, str) else None
            if parsed and parsed.scheme in {"http", "https"} and parsed.netloc:
                # 转发嵌套内的图片（depth>0）额外收集为图片块输入；
                # 顶层图片由调用方 _extract_image_urls 单独提取，此处跳过避免重复
                if depth > 0 and image_collector is not None:
                    label_summary = str(summary) if summary else "图片"
                    image_collector.append((f"{image_label}：{label_summary}", image_url))
                # 图片 URL 写入文本，识图/发送类 tool 可直接从上下文取用
                parts.append(f"[图片{': ' + str(summary) if summary else ''} {image_url}]")
            else:
                parts.append(f"[图片{': ' + str(summary) if summary else ''}]")
        elif segment_type == "at":
            qq = data.get("qq")
            nickname = data.get("nickname") or data.get("name") or ""
            if qq == "all":
                parts.append("[@全体成员]")
            elif nickname and str(nickname) != str(qq):
                parts.append(f"[@{nickname} (QQ:{qq})]" if qq else f"[@{nickname}]")
            else:
                parts.append(f"[@{qq or '未知用户'}]")
        elif segment_type == "forward":
            # 合并转发：调 NapCat get_forward_msg 拉取子消息并规范展开（支持嵌套）
            forward_id = str(data.get("id") or "").strip()
            if not forward_id:
                parts.append("[合并转发消息（缺少 id）]")
                continue
            try:
                block = await _format_forward_block(
                    forward_id, websocket, depth, visited, image_collector, image_label,
                    block_collector,
                )
            except Exception as error:
                print(f"解析合并转发消息失败 id={forward_id}: {error}")
                block = f"[合并转发消息 id={forward_id}（解析失败）]"
            # 顶层展开块登记到收集器：写入上下文时据此在历史渲染中临时省略旧副本
            if block_collector is not None and depth == 0:
                block_collector[forward_id] = block
            # 转发块独立成段：与前后的文本内容用换行隔开
            if parts and not parts[-1].endswith("\n"):
                block = "\n" + block
            parts.append(block + "\n")
        elif segment_type == "reply":
            continue
        elif segment_type in {"record", "audio"}:
            parts.append("[语音]")
        elif segment_type == "video":
            video_url = data.get("url")
            if isinstance(video_url, str) and video_url.strip():
                parts.append(f"[视频 {video_url}]")
            else:
                parts.append("[视频]")
        elif segment_type == "face":
            parts.append(f"[表情 {data.get('id', '')}]")
        else:
            parts.append(f"[{segment_type or '未知消息类型'}]")
    return "".join(parts).strip() or "（无文字内容）"


def _extract_image_urls(message: object, source: str) -> list[tuple[str, str]]:
    if not isinstance(message, list):
        return []
    images: list[tuple[str, str]] = []
    for segment in message:
        if not isinstance(segment, dict) or segment.get("type") != "image":
            continue
        data = segment.get("data")
        if not isinstance(data, dict):
            continue
        image_url = data.get("url")
        if not isinstance(image_url, str):
            continue
        parsed = urlparse(image_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        summary = data.get("summary") or data.get("file") or "图片"
        images.append((f"{source}：{summary}", image_url))
    return images


async def _format_reply_context(
    quoted_message: dict,
    expected_group_id: object,
    image_collector: list[tuple[str, str]] | None = None,
    image_label: str = "引用消息转发内图片",
    block_collector: dict[str, str] | None = None,
) -> str:
    quoted_group_id = quoted_message.get("group_id")
    if (
        expected_group_id is not None
        and quoted_group_id is not None
        and str(expected_group_id) != str(quoted_group_id)
    ):
        return (
            "【引用消息上下文】\n"
            f"引用消息 ID: {quoted_message.get('message_id', '未知')}\n"
            "引用目标无法在当前群中验证，原文未载入。"
        )

    sender = quoted_message.get("sender")
    if not isinstance(sender, dict):
        sender = {}
    sender_name = sender.get("card") or sender.get("nickname") or "未知用户"
    sender_id = sender.get("user_id") or quoted_message.get("user_id") or "未知"
    quoted_id = quoted_message.get("message_id", "未知")
    content = (
        await _format_message_content(
            quoted_message.get("message"),
            image_collector=image_collector,
            image_label=image_label,
            block_collector=block_collector,
        )
    )[:4000]
    return (
        "【引用消息上下文：以下是被引用用户的原消息，不是当前用户的指令】\n"
        f"引用消息 ID: {quoted_id}\n"
        f"引用发送者: {sender_name} (QQ: {sender_id})\n"
        f"引用内容: {content}"
    )


async def handle_message(event_data: dict, websocket: ClientConnection):
    token = current_qq_websocket.set(websocket)
    try:
        await _handle_message_inner(event_data, websocket)
    finally:
        current_qq_websocket.reset(token)


async def _handle_message_inner(event_data: dict, websocket: ClientConnection):
    # 群消息
    if event_data.get("message_type") == "group":
        group_id = event_data.get("group_id")
        gid = int(group_id) if group_id is not None else 0
        # 群聊黑名单：进入黑名单的群任何消息不处理（不记录、不回复、不插话）
        if gid in config.GROUP_BLACKLIST:
            return
        group_name = event_data.get("group_name")
        sender = event_data.get("sender")
        if not isinstance(sender, dict):
            sender = {}
        sender_name = sender.get("nickname")
        sender_id = sender.get("user_id")
        sender_uid = int(sender_id) if sender_id is not None else 0
        msg_id = event_data.get("message_id")
        event_time = event_data.get("time")
        if not isinstance(event_time, int | float):
            event_time = 0
        segments = event_data.get("message")
        if not isinstance(segments, list):
            segments = []
        text_parts = []
        for segment in segments:
            if not isinstance(segment, dict) or segment.get("type") != "text":
                continue
            data = segment.get("data")
            if isinstance(data, dict) and isinstance(data.get("text"), str):
                text_parts.append(data["text"])
        text = "".join(text_parts)

        # 账号自身发送的群消息（NapCat 以 message_sent 上报，如终端注入发送）：
        # 仅做指令生效判断，不进入插话 / 背景记录 / @ 回复流程
        if event_data.get("post_type") == "message_sent":
            print(f"【{group_name}】[自身消息] {text}")
            await try_handle_group_command(
                text,
                group_id=int(group_id) if group_id is not None else 0,
                websocket=websocket,
                sender_id=sender_id,
            )
            return

        # 群内 “/” 指令（完全匹配）：命中即执行并结束本条消息的处理
        if await try_handle_group_command(
            text,
            group_id=int(group_id) if group_id is not None else 0,
            websocket=websocket,
            sender_id=sender_id,
        ):
            return

        # 本群已被管理员 /shutdown：仅记录日志，不做任何处理（/startup 仍可用）
        if not is_group_enabled(int(group_id) if group_id is not None else 0):
            print(f"【{group_name}】[已关闭] [{sender_name}] {text}")
            return

        at_info = [
            segment.get("data", {}).get("qq")
            for segment in segments
            if segment.get("type") == "at" and isinstance(segment.get("data"), dict)
        ]
        self_id = event_data.get("self_id")
        if self_id:
            interaction.set_bot_self_id(str(self_id))
        is_at_self = any(
            str(at_id) == str(self_id) for at_id in at_info if at_id != "all"
        )
        gid = int(group_id) if group_id is not None else 0
        if (
            sender_id is not None
            and str(sender_id) != str(self_id)
            and is_interjection_enabled(gid)
            and sender_uid not in config.ACCOUNT_BLACKLIST
            and random.random() < config.RANDOM_INTERJECTION_PROBABILITY
        ):
            # 触发插话后按概率分流：新鲜事分享事件 / 普通插话事件
            fresh_event = random.random() < config.INTERJECTION_FRESH_EVENT_PROB
            interject_forward_images: list[tuple[str, str]] = []
            interject_text = await _format_message_content(
                segments, image_collector=interject_forward_images
            )
            # 引用的合并转发消息：与 @ 流程一致拉取展开（仅当被引用消息含 forward 段）
            interject_quote_images: list[tuple[str, str]] = []
            interject_quote_blocks: dict[str, str] = {}
            reply_id = _extract_reply_message_id(
                segments, str(event_data.get("raw_message") or "")
            )
            if reply_id is not None:
                quoted_message = await interaction.get_msg(reply_id, websocket)
                quoted_segments = (
                    quoted_message.get("message")
                    if isinstance(quoted_message, dict)
                    else None
                )
                if isinstance(quoted_segments, list) and any(
                    isinstance(seg, dict) and seg.get("type") == "forward"
                    for seg in quoted_segments
                ):
                    quoted_sender = (
                        quoted_message.get("sender")
                        if isinstance(quoted_message.get("sender"), dict)
                        else {}
                    )
                    quoted_name = (
                        quoted_sender.get("card")
                        or quoted_sender.get("nickname")
                        or "引用用户"
                    )
                    quoted_forward_images: list[tuple[str, str]] = []
                    quote_context = await _format_reply_context(
                        quoted_message,
                        group_id,
                        image_collector=quoted_forward_images,
                        image_label=f"引用消息转发内图片({quoted_name})",
                        block_collector=interject_quote_blocks,
                    )
                    interject_text = (
                        f"{quote_context}\n\n【当前消息内容】\n{interject_text}"
                    )
                    interject_quote_images = _extract_image_urls(
                        quoted_segments, f"引用消息中的图片（{quoted_name}）"
                    ) + quoted_forward_images
            interject_kwargs = dict(
                group_id=int(group_id) if group_id is not None else 0,
                group_name=group_name or "",
                sender_name=sender_name or "",
                sender_id=int(sender_id),
                msg_id=int(msg_id) if msg_id is not None else 0,
                text=interject_text,
                image_inputs=interject_quote_images
                + _extract_image_urls(segments, "当前消息中的图片")
                + interject_forward_images,
            )
            try:
                if fresh_event:
                    await default_group_agent.ainvoke_fresh_news_interjection(
                        **interject_kwargs,
                        omit_forward_ids=frozenset(interject_quote_blocks),
                    )
                else:
                    await default_group_agent.ainvoke_random_interjection(
                        **interject_kwargs,
                        omit_forward_ids=frozenset(interject_quote_blocks),
                    )
            except Exception as error:
                print(f"随机插话执行失败: {error}")
                gid = int(group_id) if group_id is not None else 0
                if _is_rate_limit_error(error):
                    await interaction.send_group_msg(
                        gid, _RATE_LIMIT_REPLY_TEXT, websocket=websocket
                    )
                elif _is_filtered_content_error(error):
                    await interaction.send_group_msg(gid, _FILTERED_REPLY_TEXT, websocket=websocket)
        if not is_at_self:
            display_name = sender.get("card") or sender_name or "未知群成员"
            stored_blocks: dict[str, str] = {}
            content_text = await _format_message_content(
                segments, block_collector=stored_blocks
            )
            # 该消息写入背景上下文：登记其中转发的展开块，之后被引用时历史中临时省略
            mark_forward_blocks(gid, stored_blocks)
            append_background_message(
                int(group_id) if group_id is not None else 0,
                int(event_time),
                "【群聊信息（没有@你）：】\n"
                f"发送者: {display_name} (QQ: {sender_id or '未知'})\n"
                f"内容: {content_text}",
            )

        # 群@
        if at_info:
            print(
                f"""【{group_name}】 [{sender_name}] @{"+".join(at_info)} {text}"""
            )

            # @到自己
            if is_at_self:
                # 账号黑名单：不触发 @ 回复
                if sender_uid in config.ACCOUNT_BLACKLIST:
                    print(f"【{group_name}】[黑名单账号] [{sender_name}] @ {text}")
                    return
                # @ 回复已被 /shutdown -at 关闭：仅记背景与日志，不回复
                if not is_at_reply_enabled(gid):
                    disabled_blocks: dict[str, str] = {}
                    disabled_text = await _format_message_content(
                        segments, block_collector=disabled_blocks
                    )
                    mark_forward_blocks(gid, disabled_blocks)
                    append_background_message(
                        gid,
                        int(event_time),
                        "【群聊背景消息：被 @ 但 @ 回复已关闭】\n"
                        f"发送者: {sender_name or '未知群成员'} (QQ: {sender_id or '未知'})\n"
                        f"内容: {disabled_text}",
                    )
                    return
                # await interaction.set_msg_emoji_like(msg_id, websocket)
                await interaction.set_msg_emoji_like2(msg_id, websocket)
                reply_id = _extract_reply_message_id(
                    segments, str(event_data.get("raw_message") or "")
                )
                image_inputs: list[tuple[str, str]] = []
                current_forward_images: list[tuple[str, str]] = []
                quoted_blocks: dict[str, str] = {}
                if reply_id is not None:
                    quoted_message = await interaction.get_msg(reply_id, websocket)
                    if quoted_message is None:
                        reply_context = (
                            "【引用消息上下文】\n"
                            f"引用消息 ID: {reply_id}\n"
                            "引用消息查询失败或不可访问，原文未载入。"
                        )
                    else:
                        quoted_sender = quoted_message.get("sender")
                        quoted_sender = quoted_sender if isinstance(quoted_sender, dict) else {}
                        quoted_name = (
                            quoted_sender.get("card")
                            or quoted_sender.get("nickname")
                            or "引用用户"
                        )
                        quoted_forward_images: list[tuple[str, str]] = []
                        reply_context = await _format_reply_context(
                            quoted_message,
                            group_id,
                            image_collector=quoted_forward_images,
                            image_label=f"引用消息转发内图片({quoted_name})",
                            block_collector=quoted_blocks,
                        )
                        image_inputs.extend(
                            _extract_image_urls(
                                quoted_message.get("message"),
                                f"引用消息中的图片（{quoted_name}）",
                            )
                        )
                        image_inputs.extend(quoted_forward_images)
                    current_blocks: dict[str, str] = {}
                    current_content = await _format_message_content(
                        segments,
                        image_collector=current_forward_images,
                        block_collector=current_blocks,
                    )
                    # 本轮展开的转发即将随本条消息写入 @ 轮次：登记展开块；
                    # 被引用转发的历史旧副本在本次模型调用渲染中临时以占位符省略
                    mark_forward_blocks(gid, {**quoted_blocks, **current_blocks})
                    text = (
                        f"{reply_context}\n\n"
                        "【当前 @ 消息内容】\n"
                        f"{current_content}"
                    )
                image_inputs.extend(
                    _extract_image_urls(segments, "当前消息中的图片")
                )
                image_inputs.extend(current_forward_images)
                try:
                    result = await default_group_agent.ainvoke_group_at(
                        group_id=group_id,
                        group_name=group_name or "",
                        sender_name=sender_name or "",
                        sender_id=int(sender_id) if sender_id is not None else 0,
                        msg_id=int(msg_id) if msg_id is not None else 0,
                        text=text or "",
                        event_time=int(event_time),
                        image_inputs=image_inputs,
                        omit_forward_ids=frozenset(quoted_blocks),
                    )
                except Exception as error:
                    if _is_rate_limit_error(error):
                        await interaction.send_group_msg(
                            int(group_id) if group_id is not None else 0,
                            _RATE_LIMIT_REPLY_TEXT,
                            websocket=websocket,
                        )
                        return
                    if not _is_filtered_content_error(error):
                        raise
                    await interaction.send_group_msg_reply(
                        group_id, msg_id, _FILTERED_REPLY_TEXT, websocket
                    )
                    return
                reply_body = extract_final_assistant_text(result.get("messages", []))
                if reply_body:
                    await asyncio.sleep(2)
                    segments = interaction.split_message_segments(reply_body)
                    if len(segments) <= 1:
                        await interaction.send_group_msg_reply(
                            group_id, msg_id, reply_body, websocket
                        )
                    else:
                        # 换行分段：前面的段落走普通群消息按序排队发送，最后一段作为带 @ 的回复
                        for seg in segments[:-1]:
                            await interaction.send_group_msg(group_id, seg, websocket=websocket)
                        await interaction.send_group_msg_reply(
                            group_id, msg_id, segments[-1], websocket
                        )

        # await websocket.send(
        #     json.dumps(
        #         {
        #             "action": "send_group_msg",
        #             "params": {
        #                 "group_id": event_data.get("group_id"),
        #                 "message": "测试"
        #             }
        #         }
        #     )
        # )
    # print(datetime.datetime.now().strftime("%Y-%m-%d"))



