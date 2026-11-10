"""生成并发布一条说说：人设 + 事件素材 → 模型写稿（可用已注册的检索/图片工具）→
代码校验（长度、隐私、与近期说说相似度）→ 配图（模型找的图，没有则用表情）→ 发布。"""

from __future__ import annotations

import json
import random
import re

import httpx
import numpy as np
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.messages import SystemMessage
from langchain_core.messages import HumanMessage
from websockets.asyncio.client import ClientConnection

import config

from ..action import interaction
from langchain_core.messages import AIMessage
from ..agent.face_enum import list_face_enum_keys, load_face_entries, resolve_face
from ..agent.group_prompt import DEFAULT_PROMPT_VAR, get_prompt_text
from ..agent.init_llm import _build_chat_model
from ..agent.tools import _qzone_quota_consume, _qzone_quota_left, tools_lst
from ..meme.embed import embed_texts
from .events import Event, group_review_material
from .posts import recent_posts, record_post

# 写说说时可用的工具：仅限当前已注册到 tools_lst 的检索与图片类工具
_ALLOWED_TOOLS = ("search", "search_images", "analyze_images", "scrape_website")
_MAX_IMAGES = 2
_MAX_CHARS = 120
_SIMILARITY_LIMIT = 0.85

_PERSONA_BLOCK = re.compile(r"<(\w*persona\w*)>(.*?)</\1>", re.DOTALL)
_QZONE_RULES = re.compile(r"<qzone_rules>(.*?)</qzone_rules>", re.DOTALL)
_QQ_NUMBER = re.compile(r"\d{6,}")
_URL = re.compile(r"https?://|www\.", re.IGNORECASE)

_TASK_RULES = f"""
<qzone_task>
你现在不是在群聊里，而是在给自己的 QQ 空间写一条说说。
- 正文 1～3 句、不超过 {_MAX_CHARS} 字，像随手发的朋友圈；口吻、语气完全按上面的人设。
- 不出现群名、群友昵称、QQ 号、@、链接；不照搬任何人的原话。
- 不编造现实中可核查的具体经历（某天去了某地、见了某人）；人设里的日常小设定可以写。
- 尽量配图：先想好说说内容，再用 search_images 找贴切的图，并用 analyze_images 确认图的内容确实合适，
  最多 {_MAX_IMAGES} 张；实在找不到合适的就不配，系统会配一张表情。
- 最终只输出一个 JSON 对象，不要其他文字：
  {{"content": "说说正文", "images": ["确认过的图片URL"], "mood": "表情情绪"}}
  mood 从这些里选一个最贴近这条说说情绪的：{", ".join(list_face_enum_keys())}
</qzone_task>
"""

_agent = None


def _persona_prompt() -> str:
    """从默认提示词中抽取人设与说说规则（人设只维护一处）；没有标签时退回整段提示词。"""
    text = get_prompt_text(DEFAULT_PROMPT_VAR)
    blocks = [m.group(0) for m in _PERSONA_BLOCK.finditer(text)]
    rules = _QZONE_RULES.search(text)
    if rules:
        blocks.append(rules.group(0))
    return ("\n\n".join(blocks) if blocks else text) + "\n" + _TASK_RULES


def _get_agent():
    global _agent
    if _agent is None:
        tools = [t for t in tools_lst if t.name in _ALLOWED_TOOLS]
        _agent = create_agent(
            model=_build_chat_model(config.DEFAULT_MODEL, thinking=False),
            tools=tools,
            middleware=[ToolCallLimitMiddleware(run_limit=8)],
            system_prompt=SystemMessage(content=[{"type": "text", "text": _persona_prompt()}]),
        )
    return _agent


def _human_prompt(event: Event, feedback: str) -> str:
    from datetime import datetime

    now = datetime.now().astimezone()
    weekday = "一二三四五六日"[now.weekday()]
    parts = [
        f"当前时间：{now:%Y-%m-%d %H:%M}，星期{weekday}",
        f"这次的说说类型：{event.title}",
        event.instruction,
    ]
    if event.material:
        parts.append("素材：\n" + event.material)
    recent = recent_posts(10)
    if recent:
        parts.append(
            "你最近发过的说说（换个话题和说法，别重复）：\n"
            + "\n".join(f"- {p['content']}" for p in recent)
        )
    if feedback:
        parts.append(f"上一稿没通过：{feedback}。请重写。")
    return "\n\n".join(parts)


def _final_text(messages: list) -> str:
    """最后一条无工具调用的模型输出原文（不做 markdown 清洗，JSON 可能包在代码块里）。"""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            if isinstance(content, str) and content.strip():
                return content
    return ""


def _parse_draft(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("content"), str):
        return None
    images = data.get("images")
    data["images"] = [u for u in images if isinstance(u, str)] if isinstance(images, list) else []
    data["mood"] = data.get("mood") if isinstance(data.get("mood"), str) else ""
    data["content"] = data["content"].strip()
    return data


async def _check_draft(content: str, banned_names: set[str]) -> str:
    """返回不通过的原因；通过返回空串。"""
    if not content:
        return "正文为空"
    if len(content) > _MAX_CHARS:
        return f"超过 {_MAX_CHARS} 字"
    if "@" in content or _URL.search(content) or _QQ_NUMBER.search(content):
        return "包含 @、链接或疑似 QQ 号"
    hit = next((n for n in banned_names if len(n) >= 2 and n in content), None)
    if hit:
        return "包含群友昵称"
    recent = [p["content"] for p in recent_posts(10)]
    if recent:
        try:
            matrix = await embed_texts([content, *recent])
            sims = matrix[1:] @ matrix[0]
            if float(np.max(sims)) > _SIMILARITY_LIMIT:
                return "和最近的说说太像"
        except Exception as e:
            print(f"[qzone] 相似度检查跳过: {e}")
    return ""


async def _image_ok(url: str) -> bool:
    """确认 URL 可访问且确实是图片（避免盗链/失效图导致说说发图失败）。"""
    if not url.startswith(("http://", "https://")):
        return False
    try:
        async with httpx.AsyncClient(timeout=8, follow_redirects=True) as client:
            async with client.stream("GET", url, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                return resp.status_code == 200 and resp.headers.get(
                    "content-type", ""
                ).startswith("image/")
    except Exception:
        return False


async def _fallback_face(mood: str, websocket: ClientConnection) -> list[str]:
    """没有配图时用表情：优先按情绪取内置表情，其次随机收藏表情。"""
    face = resolve_face(mood) if mood else None
    if face is not None:
        return [face[1]]
    try:
        faces = await interaction.fetch_custom_faces(count=48, websocket=websocket)
    except Exception as e:
        print(f"[qzone] 获取收藏表情失败: {e}")
        faces = []
    if faces:
        return [random.choice(faces)]
    return [random.choice(list(load_face_entries().values()))[1]]


async def write_and_post(event: Event, websocket: ClientConnection) -> bool:
    """生成并发布一条说说；成功返回 True。最多写两稿，都不通过则放弃。"""
    if _qzone_quota_left() <= 0:
        print("[qzone] 今日说说额度已用完，跳过")
        return False
    _, banned_names = group_review_material()
    feedback = ""
    draft: dict | None = None
    for _ in range(2):
        result = await _get_agent().ainvoke(
            {"messages": [HumanMessage(content=_human_prompt(event, feedback))]}
        )
        draft = _parse_draft(_final_text(result.get("messages", [])))
        if draft is None:
            feedback = "没有按要求只输出 JSON"
            continue
        feedback = await _check_draft(draft["content"], banned_names)
        if not feedback:
            break
        print(f"[qzone] 说说草稿未通过（{feedback}）: {draft['content']}")
    if draft is None or feedback:
        return False

    images = [u for u in draft["images"][:_MAX_IMAGES] if await _image_ok(u)]
    if not images:
        images = await _fallback_face(draft["mood"], websocket)
    if _qzone_quota_left() <= 0:
        return False
    await interaction.send_qzone_msg(draft["content"], images, websocket=websocket)
    _qzone_quota_consume()
    record_post(draft["content"], images, source="auto", event=event.kind)
    print(f"[qzone] 定时说说已发布（{event.title}）: {draft['content']}")
    return True
