from typing import Literal, Sequence
from urllib.parse import unquote, urlparse
from pathlib import PurePosixPath
from contextvars import ContextVar
from datetime import datetime
import asyncio
import json
import random

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import httpx
import config
from tavily import AsyncTavilyClient
from ..action import interaction
from ..qq_ws_context import current_qq_websocket
from ..tts.client import synthesize_to_mp3 as _tts_synthesize_to_mp3
from . import aliyun_image_gen, volces_image_gen

# 生图平台按 config.IMAGE_GEN_PLATFORM 分发（aliyun / volc），同签名可互换
_IMAGE_GENERATOR = (
    volces_image_gen.generate_images
    if config.IMAGE_GEN_PLATFORM == "volc"
    else aliyun_image_gen.generate_images
)
from .face_enum import list_face_enum_keys, resolve_face

tavily_client = AsyncTavilyClient(config.TAVILY_KEY)

# 当前 @ 消息上下文（由 init_llm.ainvoke_group_at 设置）：
# generate_image 自动取参考图、群号与 @ 目标，无需模型传参
current_group_id: ContextVar[int | None] = ContextVar("current_group_id", default=None)
current_sender_id: ContextVar[int | None] = ContextVar("current_sender_id", default=None)
current_image_urls: ContextVar[tuple[str, ...]] = ContextVar("current_image_urls", default=())

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
    常用于群友要求找图时：先结合 search / search_images / crawl_website /
    scrape_website 联网检索拿到相关图片的 URL（可用 analyze_images 先确认内容），
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
    适用场景：群里给出图片链接，或 search / search_images / crawl_website /
    scrape_website 联网检索得到图片 URL 后需要了解图片内容时。
    仅返回文字描述，不会发送任何内容到群聊。
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
    web sources. Prefer `crawl_website` when deep multi-page exploration is needed;
    use `search_images` when looking for pictures instead of text.

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
    for quick real-time queries. If this fails on a single page, try `scrape_website`.
    Extracted content may contain image URLs: use `analyze_images` to identify them,
    or `qq_send_group_msg_image` to post a found image to the group.

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


@tool
async def qq_send_group_msg_voice(group_id: int, text: str) -> str:
    """
    把一段文字合成为语音，并以 QQ 语音消息发送到群聊。
    适合念一段话、讲故事、温柔读消息等场景；语音在后台流式合成（mp3）
    后自动发送，无需上传文件。
    text 会被完整朗读：只传要说的内容，不要混入表情代码、图片链接或 markdown。

    Args:
        group_id: QQ group ID.
        text: 要朗读的文字。
    """
    text = (text or "").strip()
    if not text:
        return "Failed: 未提供要朗读的文字。"
    try:
        mp3_path = await _tts_synthesize_to_mp3(text)
    except Exception as e:
        return f"Failed: 语音合成失败 {e}"
    try:
        await interaction.send_group_msg_record(group_id, mp3_path)
    except Exception as e:
        return f"Failed: 语音发送失败 {e}"
    return "OK"


_IMAGE_TOOL_COMMON = """后台生图，不阻塞当前对话，仅支持群聊被 @ 场景。
当前 @ 消息及其引用消息里的图片（最多 3 张）自动作为参考图传入，无需传参；
仅需 prompt 描述要生成的图片（涉及参考图时用 "Image 1 / Image 2" 指代）。
调用后任务立即提交、图片尚未生成：完成后所有生成图片会自动发到本群并
@ 发起请求的用户；失败只记录终端日志。
收到本工具返回后，必须告知用户图片正在生成中、稍后自动送达，不得说已生成完成。"""

if config.IMAGE_GEN_PLATFORM == "volc":
    _GENERATE_IMAGE_DOC = f"""使用火山方舟 Seedream 生图，不依赖参考图也可纯文生图。
{_IMAGE_TOOL_COMMON}
prompt 可直接描述想要的画面比例（如 "竖版 3:4"）；涉及时效性、事实或
需要查证的信息时，需在 prompt 中特别要求，平台会自动联网搜索辅助生图。

    Args:
        prompt: 生图内容描述，可包含画面细节与比例。"""
else:
    _GENERATE_IMAGE_DOC = f"""使用阿里云 qwen-mt-image-2.0 生图。
{_IMAGE_TOOL_COMMON}

    Args:
        prompt: 生图内容描述，例如在有多张参考图时传入 "让 Image 1 的人物穿上 Image 2 的黑裙,"。"""


async def _generate_image(prompt: str) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return "Failed: 未提供生图 prompt。"
    group_id = current_group_id.get()
    websocket = current_qq_websocket.get()
    if group_id is None or websocket is None:
        return "Failed: 生图仅支持在被 @ 的群聊回复中使用。"
    sender_id = current_sender_id.get()
    reference_urls = current_image_urls.get()
    asyncio.create_task(
        _run_image_generation(group_id, sender_id, prompt, reference_urls, websocket)
    )
    return (
        "生图任务已提交，正在后台生成（尚未完成）。请告知用户图片生成中，"
        "完成后会@用户发过去。"
    )


_generate_image.__doc__ = _GENERATE_IMAGE_DOC
generate_image = tool(_generate_image)


async def _run_image_generation(
    group_id: int,
    sender_id: int | None,
    prompt: str,
    reference_urls: tuple[str, ...],
    websocket,
) -> None:
    try:
        urls = await _IMAGE_GENERATOR(prompt, reference_urls)
    except Exception as error:  # 生图报错只记终端日志，不打扰群聊
        print(f"生图任务失败 group_id={group_id} prompt={prompt!r}: {error}")
        return
    try:
        await interaction.send_group_msg_image(
            group_id, "pic.png", urls, websocket=websocket, at_user=sender_id
        )
    except Exception as error:
        print(f"生图结果发送失败 group_id={group_id}: {error}")


async def _serper_post(endpoint: str, payload: dict) -> dict:
    headers = {"X-API-KEY": config.SERPER_API_KEY, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


@tool
async def search_images(
    query: str, gl: str = "cn", time_range: Literal["day", "week", "month"] | None = None
) -> str:
    """
    联网搜索图片（Serper Google 图片接口），返回与 query 相关的图片列表
    （imageUrl / 标题 / 来源页面）。time_range 可限定时效：day（1 天内）、
    week（1 周内）、month（1 月内）。
    常与 `analyze_images` 联用：先搜图，再对感兴趣的 imageUrl 做内容识别；
    找到想要的图后用 `qq_send_group_msg_image` 直接发送到群聊。

    Args:
        query: Search query, e.g. "deepseek logo".
        gl: 国家/地区参数，影响结果偏好，默认 "cn"。
        time_range: 时效过滤，不传则不限。
    """
    payload: dict = {"q": query, "gl": gl}
    tbs = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m"}.get(time_range)
    if tbs:
        payload["tbs"] = tbs
    else:
        payload["tbs"] = "qdr:w"
    try:
        data = await _serper_post("https://google.serper.dev/images", payload)
    except Exception as e:
        return f"Failed: {e}"
    images = data.get("images", [])
    if not images:
        return "Failed: 未搜到相关图片。"
    lines = []
    for idx, item in enumerate(images[:10], 1):
        lines.append(
            f"{idx}. title: {item.get('title', '')}\n"
            f"   imageUrl: {item.get('imageUrl', '')}\n"
            f"   source: {item.get('source', '')} {item.get('link', '')}"
        )
    return "\n".join(lines)


@tool
async def scrape_website(url: str) -> str:
    """
    抓取单个网页并返回 Markdown 正文（Serper scrape 接口，含 includeMarkdown）。
    与 `crawl_website` 的区别：本工具面向单个页面，对 Google 已收录的普通网页
    成功率更高；一个失败时可换另一个重试。
    抓到的内容若包含图片 URL，可用 `analyze_images` 识别图片内容，
    或用 `qq_send_group_msg_image` 把图片发到群聊。

    Args:
        url: 要抓取的网页 URL，e.g. "https://example.com/post".
    """
    try:
        data = await _serper_post(
            "https://scrape.serper.dev", {"url": url, "includeMarkdown": True}
        )
    except Exception as e:
        return f"Failed: {e}"
    markdown = data.get("markdown", "")
    if not markdown:
        return f"Failed: 未获取到网页正文（返回字段：{', '.join(data.keys())}）。"
    return markdown


tools_lst = [
    qq_send_group_msg,
    # qq_send_group_ai_record,
    qq_send_group_emotion_face2,
    qq_send_group_msg_image,
    qq_send_group_msg_voice,
    search,
    crawl_website,
    scrape_website,
    search_images,
    analyze_images,
    generate_image,
]

# Random interjections only need tools that visibly post to the group.
interjection_tools_lst = [
    qq_send_group_msg,
    qq_send_group_msg_voice,
    # qq_send_group_ai_record,
    qq_send_group_emotion_face2,
]


# ---- 工具调用日志（终端输入 tool 回车切换开关） ----
TOOL_LOG_PATH = "data/tool_calls.log"
_tool_log_enabled = False


def toggle_tool_logging() -> bool:
    """切换工具调用日志开关，返回切换后的状态。"""
    global _tool_log_enabled
    _tool_log_enabled = not _tool_log_enabled
    return _tool_log_enabled


def _log_tool_call(tool_name: str, args: dict, result: object) -> None:
    def _fmt(value: object) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return repr(value)

    with open(TOOL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"tool={tool_name}\n"
            f"args: {_fmt(args)}\n"
            f"result: {_fmt(result)}\n\n"
        )


def _wrap_with_tool_logging(tool_obj):
    """给工具协程包一层日志（仅开关开启时写文件）。"""
    original = tool_obj.coroutine
    if original is None:
        return tool_obj

    async def logged(**kwargs):
        result = await original(**kwargs)
        if _tool_log_enabled:
            try:
                _log_tool_call(tool_obj.name, kwargs, result)
            except Exception as e:
                print(f"记录工具调用日志出错: {e}")
        return result

    tool_obj.coroutine = logged
    return tool_obj


tools_lst = [_wrap_with_tool_logging(t) for t in tools_lst]
interjection_tools_lst = [_wrap_with_tool_logging(t) for t in interjection_tools_lst]
