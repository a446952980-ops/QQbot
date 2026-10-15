from typing import Literal, Sequence
from urllib.parse import unquote, urlparse
from pathlib import PurePosixPath
import random

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import config
from tavily import AsyncTavilyClient
from ..action import interaction
from .face_enum import list_face_enum_keys, resolve_face

tavily_client = AsyncTavilyClient(config.TAVILY_KEY)

# 供 analyze_images 使用的视觉模型；未配置 VISION_MODEL 时退回默认模型
_vision_llm = ChatOpenAI(
    model=config.VISION_MODEL or config.DEFAULT_MODEL,
    temperature=0.1,
    max_tokens=4096,
    timeout=120,
    api_key=config.AI_KEY,
    base_url=config.AI_BASE_URL,
    extra_body={"thinking": {"type": "disabled"}},
)

_DEFAULT_VISION_PROMPT = "请分别描述每张图片的内容，包括主要对象、出现的文字和关键细节。"

_FACE_ENUM_DOC = ", ".join(list_face_enum_keys()) or "(none configured)"

_EMOTION_FACE_TOOL_DESCRIPTION = (
    "REQUIRED whenever Nika's affect clearly shifts or is strongly colored — not optional. "
    "You MUST call this tool if there is real emotional movement: joy/warmth (happy), grief or "
    "heaviness (sad, crying), shock (surprised), not understanding (confused), anger "
    "(extremelyangry), being drawn in or wonder-struck (captivated), etc. If words, actions in "
    "parentheses, or tone imply such a feeling, you must send the matching face before or "
    "alongside finishing your turn (same model step is fine). Pick the closest enum if none "
    "fits perfectly. Pass only lowercase enum without .png. Valid keys: "
    + _FACE_ENUM_DOC
    + ". Plain non-reply image; never pre-announce with a separate status line. The sticker "
    "adds a visible beat; it does not replace your final text reply."
)


@tool
async def qq_send_group_msg(group_id: int, text: str) -> str:
    """
    Send exactly one standalone plain-text message to a QQ group (not a reply). This tool is strictly for pre-final messages, never for sending the final answer or its正文.
    When preparing a long response, you MUST send its preceding sentences or paragraphs through separate tool calls, in their intended order, leaving only the final sentence or paragraph to be delivered directly by the model's final response. Never send the final paragraph through this tool.
    Each call sends one message of 1–2 short sentences. For @-replies, the final response MUST contain only new text; never repeat, quote, concatenate, or paraphrase anything already sent through this tool. The same rule applies to pre-search/crawl status messages.
    """
    try:
        await interaction.send_group_msg(group_id, text)
    except RuntimeError as e:
        return f"Failed: {e}"
    return "Plain group message sent."


@tool
async def qq_send_group_ai_record(
    group_id: int, text: str, character: str | None = None
) -> str:
    """
    Send an AI voice message to a QQ group (synthesized speech from text).
    `text` is what will be spoken. `character` is the NapCat voice profile id;
    omit it to use the default from server configuration (e.g. lucy-voice-female1).
    This is not a reply to a specific message; it posts a new voice record in the group.
    """
    try:
        await interaction.send_group_ai_record(group_id, text, character)
    except RuntimeError as e:
        return f"Failed: {e}"
    return "AI voice record sent to the group."


@tool(description=_EMOTION_FACE_TOOL_DESCRIPTION)
async def qq_send_group_emotion_face(group_id: int, face: str) -> str:
    """Required on clear affect; sends emotion-face image (NapCat, sub_type 1)."""
    resolved = resolve_face(face)
    if resolved is None:
        keys = ", ".join(list_face_enum_keys()) or "none"
        return f"Unknown face enum {face!r}. Valid: {keys}"
    fname, url = resolved
    try:
        await interaction.send_group_msg_image(group_id, fname, url, sub_type=1)
    except RuntimeError as e:
        return f"Failed: {e}"
    return f"Emotion face {fname} sent."


@tool
async def qq_send_group_emotion_face2(group_id: int) -> str:
    """Send one randomly selected custom QQ face fetched live from NapCat."""
    try:
        faces = await interaction.fetch_custom_faces(count=48)
        if not faces:
            return "Failed: NapCat returned no custom faces."
        url = random.choice(faces)
        filename = PurePosixPath(unquote(urlparse(url).path)).name or "custom_face.png"
        await interaction.send_group_msg_image(
            group_id, filename, url, sub_type=1
        )
    except RuntimeError as e:
        return f"Failed: {e}"
    return f"Custom emotion face {filename} sent."


@tool
async def qq_send_group_msg_image(group_id: int, url: str) -> str:
    """
    根据已知的图片 URL 直接发送图片到群聊。
    常用于群友要求找图时：先结合 search / crawl_website 联网检索拿到相关图片的 URL，
    再用本工具把找到的图片直接发到群里。url 必须是可公网访问的 http(s) 图片直链。
    """
    try:
        await interaction.send_group_msg_image(group_id, "pic.png", url, sub_type=0)
    except RuntimeError as e:
        return f"Failed: {e}"
    return "Image sent to the group."


@tool
async def analyze_images(urls: list[str], prompt: str | None = None) -> str:
    """
    识别一张或多张图片（传入图片 URL），由 VISION_MODEL 分析后返回文字描述。
    默认提示词为「描述图片内容」；需要特定识别目标（如提取图中文字、辨认物体/人物、
    结合联网搜索结果判断图片出处等）时可用 prompt 自定义。
    适用场景：群里给出图片链接，或 search / crawl_website 联网检索得到图片 URL
    后需要了解图片内容时。仅返回文字描述，不会发送任何内容到群聊。
    """
    if not urls:
        return "Failed: 未提供图片 URL。"
    content: list[dict[str, object]] = [
        {"type": "text", "text": (prompt or "").strip() or _DEFAULT_VISION_PROMPT}
    ]
    for url in urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    try:
        response = await _vision_llm.ainvoke([HumanMessage(content=content)])
    except Exception as e:
        return f"Failed: {e}"
    text = response.content
    if isinstance(text, list):  # 兼容多段返回，仅取文本部分
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    return str(text) or "Failed: 视觉模型未返回内容。"


@tool
async def search(
    query: str,
    topic: Literal["general", "news", "finance"] = None,
    time_range: Literal["day", "week", "month", "year"] = None,
    include_domains: Sequence[str] = None,
    exclude_domains: Sequence[str] = None,
) -> dict:
    """
    Search the web using Tavily and return relevant results with content snippets.
    Use for real-time information, current events, or questions needing up-to-date
    web sources. Prefer `crawl_website` when deep multi-page exploration is needed.

    Args:
        query: Search query, e.g. "latest AI research papers 2025".
        topic: "news" for current events; "general" for broad searches; "finance" for markets.
        time_range: Filter by recency: "day", "week", "month", or "year".
        include_domains: Optional allowlist of domains (max 300).
        exclude_domains: Optional blocklist of domains (max 150).
    """
    try:
        response = await tavily_client.search(
            query,
            "advanced",
            topic,
            time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )
        return response
    except Exception as e:
        return {"error": str(e)}


@tool
async def crawl_website(url: str, instructions: str) -> dict:
    """
    Crawl a site from a root URL and extract content across pages per natural-language
    instructions. Use when information is scattered across a site. Prefer `search`
    for quick real-time queries.

    Args:
        url: Root URL to start from, e.g. "https://docs.tavily.com".
        instructions: What to look for, e.g. "Find Python SDK authentication pages".
    """
    try:
        response = await tavily_client.crawl(
            url, instructions=instructions, max_breadth=20
        )
        return response
    except Exception as e:
        return {"error": str(e)}


tools_lst = [
    qq_send_group_msg,
    qq_send_group_ai_record,
    qq_send_group_emotion_face2,
    qq_send_group_msg_image,
    search,
    crawl_website,
    analyze_images,
]

# Random interjections only need tools that visibly post to the group.
interjection_tools_lst = [
    qq_send_group_msg,
    qq_send_group_ai_record,
    qq_send_group_emotion_face2,
]
