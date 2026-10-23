from typing import Literal, Sequence
from urllib.parse import unquote, urlparse
from pathlib import PurePosixPath
from contextvars import ContextVar
from datetime import datetime
import asyncio
import json
import os
import random
import time

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
from .conversation_history import append_background_message

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
# 是否处于被 @ 回复流程：@ 轮次的 tool 发送文本已并入 turn 上下文，不再重复写 background
_in_at_flow: ContextVar[bool] = ContextVar("in_at_flow", default=False)

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


# 单次模型调用内 qq_send_group_msg 的次数限制。
# 注意：langchain 在拷贝的子上下文中执行工具协程，ContextVar.set() 不回传；
# 因此这里持一个共享可变列表 [count]，由 ainvoke 入口重置为 [0]，
# 工具内原地修改，父上下文下次调用可见。
_send_msg_state: ContextVar[list | None] = ContextVar("send_msg_state", default=None)
_SEND_WARN_AT = 10       # 第 10、11 次：照常发送，但返回即将禁用的警告
_SEND_DISABLE_AT = 12    # 第 12、13 次：禁用发送，仅返回禁用提示
_SEND_TERMINATE_AT = 14  # 第 14 次起：强制终止本次模型调用任务，群发【Ended】


def _bump_send_count() -> int:
    state = _send_msg_state.get()
    if state is None:
        state = [0]
        _send_msg_state.set(state)
    state[0] += 1
    return state[0]


@tool
async def qq_send_group_msg(
    group_id: int, text: str, at: str | None = None
) -> str:
    """
    Send exactly one standalone plain-text message to a QQ group (not a reply). This tool is strictly for pre-final messages, never for sending the final answer or its正文.
    When preparing a long response, you MUST send its preceding sentences or paragraphs through separate tool calls, in their intended order, leaving only the final sentence or paragraph to be delivered directly by the model's final response. Never send the final paragraph through this tool.
    Each call sends one message of 1–2 short sentences. For @-replies, the final response MUST contain only new text; never repeat, quote, concatenate, or paraphrase anything already sent through this tool. The same rule applies to pre-search/crawl status messages.
    Optional `at`: prepend an @ segment before the text. Value MUST be "all" (@全体成员) or a plain-digit QQ number; anything else is rejected without sending. Only use it when @-ing that member (or everyone) is genuinely necessary.

    Args:
        group_id: 目标群号。
        text: 要发送的文本（1-2 句）。
        at: 可选，"all"（@全体成员）或对方 QQ 号（纯数字），插在文本最前。
    """
    count = _bump_send_count()
    if count >= _SEND_TERMINATE_AT:
        # 次数超限：发【Ended】后强制终止当前模型调用任务（CancelledError 在
        # agent 下一个 await 点抛出，沿调用链向上传播结束本次调用）
        await interaction.send_group_msg(group_id, "【Ended】")
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.cancel()
        return "【Ended】"
    if count >= _SEND_DISABLE_AT:
        return (
            "The execution logic of this tool has been disabled for the current "
            "model invocation. Stop tool calls immediately and return a concise "
            "final model output."
        )
    at_value: str | None = None
    if at is not None:
        at_value = str(at).strip()
        if at_value != "all" and not at_value.isdigit():
            return "Failed: at 参数仅支持 QQ 号（纯数字）或 'all'。"
    try:
        await interaction.send_group_msg(group_id, text, at=at_value)
    except RuntimeError as e:
        return f"Failed: {e}"
    # @ 回复流程的发送文本已并入 turn 上下文，不重复写 background；
    # 其他逻辑链（如定时群发）发送的内容进 background，标明是自己说的话
    if not _in_at_flow.get():
        append_background_message(
            int(group_id),
            int(time.time()),
            "【自己发送的消息：通过工具主动发出】\n"
            f"内容: {text.strip()}",
        )
    if count >= _SEND_WARN_AT:
        return (
            "Due to frequent calls, this tool is about to be disabled. Stop "
            "repetitive sending immediately and finish with a concise final output."
        )
    return f"Plain group message sent，you've sent {count} times. You can't get a reply before stopping calling this tool so stop at a reasonable time."


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
    """发一个随机自定义 QQ 表情给群（图片形式，不占文字）。

    这是“活人感”的可见节拍，不是要等情绪特别强烈才用的稀有道具：
    - 无语、惊讶、开心、得意、嫌弃、心动、被冒犯、想吐槽时，都可以顺手发一个；
    - 接梗、反撩、吐槽、夸人、认怂之后，可以配一个表情一起发；
    - 一回合一个就够，别连续刷，也别用表情代替要说的正文。

    常见用法：先发这个表情，再补一句短正文（或配合 qq_send_group_msg 带文字）。
    """
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
async def qq_group_poke(group_id: int, user_id: int) -> str:
    """
    戳一戳群里的某个人（轻量动手，不用说话）。
    这是聊天手感的一部分，不是稀有动作：
    - 玩笑落地后、接完梗、吐槽完、有人 @ 你之后、想表示“我在/我看到了”时，顺手戳一下；
    - 只戳和当前对话有关的那个群友（sender_id 或 @ 的目标），别乱戳无关的人。

    Args:
        group_id: 目标群号。
        user_id: 要戳的群成员 QQ 号（纯数字）。
    """
    try:
        await interaction.group_poke(group_id, user_id)
    except RuntimeError as e:
        return f"Failed: {e}"
    return f"Poked user {user_id} in group {group_id}."


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
    看一眼一张或多张图片（传入 URL），返回 VISION_MODEL 的文字描述。
    调用本工具就像扫一眼图片，是内部动作，不需要在群聊里声明“我看看”
    “看完了”或“图里是……”，直接基于描述接话或决定发不发即可。
    默认描述图片内容；需要特定识别目标（提取图中文字、辨认物体/人物、
    结合联网结果判断出处等）时用 prompt 自定义。

    需要看一眼的场景：对图片内容不确定、仅凭「[图片 ... URL]」的 summary
    判断不了、群友想了解某张图、或你的接话/插话可能和上下文里某张图有关。
    不要反问群友“这张图是什么”，也不要没看就猜或编图片内容。
    可用来源（均无需询问群友）：当前消息和引用消息里的图；上下文历史图片
    （被 @ 的消息、background 与引用消息以「[图片 ... URL]」存放，按需取
    URL）；search / search_images / crawl_website / scrape_website 得到的
    图片链接。

    只返回文字描述，不发送任何内容到群聊。识图本身不是发言，看完觉得
    没意思可以沉默，觉得有意思再决定要不要发、发哪张。
    仅本工具不受调用次数限制；其他工具的次数限制不影响对它的调用。
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
    time_range: Literal["day", "week", "month", "year"] = "week",
    include_domains: Sequence[str] = None,
    exclude_domains: Sequence[str] = None,
) -> dict:
    """
    Search the web using Tavily and return relevant results with content snippets.
    Use for real-time information, current events, or questions needing up-to-date
    web sources. Prefer `crawl_website` when deep multi-page exploration is needed;
    use `search_images` when looking for pictures instead of text.

    默认只搜最近一周（time_range="week"）。如果一周内搜不到、结果太少或与问题无关，
    再逐步扩大范围重搜："month"、"year"；不要一上来就用大范围。

    Args:
        query: Search query, e.g. "latest AI research papers 2025".
        topic: "news" for current events; "general" for broad searches; "finance" for markets.
        time_range: 默认 "week"（最近一周）。搜不到时再扩大到 "month" 或 "year"；确需更近可用 "day"。
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
    语音说一句话，并以 QQ 语音消息发送到群聊。
    适合念一段话、讲故事、读消息等场景；
    只传要说的内容，不要混入表情代码、图片链接或 markdown。

    Args:
        group_id: QQ group ID.
        text: 语音读的文字。
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


_QZONE_QUOTA_PATH = "data/qzone_quota.json"
_QZONE_DAILY_LIMIT = 3


def _qzone_quota_left() -> int:
    """当天剩余说额度（data/qzone_quota.json 记录当日已发表次数）。"""
    try:
        with open(_QZONE_QUOTA_PATH, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        record = {}
    today = time.strftime("%Y-%m-%d")
    if record.get("date") != today:
        return _QZONE_DAILY_LIMIT
    return max(0, _QZONE_DAILY_LIMIT - int(record.get("count", 0)))


def _qzone_quota_consume() -> None:
    today = time.strftime("%Y-%m-%d")
    try:
        with open(_QZONE_QUOTA_PATH, encoding="utf-8") as f:
            record = json.load(f)
    except (OSError, ValueError):
        record = {}
    count = int(record.get("count", 0)) if record.get("date") == today else 0
    record = {"date": today, "count": count + 1}
    tmp_path = _QZONE_QUOTA_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    os.replace(tmp_path, _QZONE_QUOTA_PATH)


@tool
async def qq_send_qzone_msg(content: str, images: list[str] | None = None) -> str:
    """
    发表一条 QQ 空间说说（当天最多 3 条，超出会被拒绝）。
    空间是你的朋友圈，不是群聊复述板：只在你自己产生真实反应时发——
    一个离谱梗、一句想记住的吐槽、一张好笑的图、一段聊出共鸣的话。
    判断标准是「我自己想发」，不是「够不够格」；想发就发，别硬憋，
    但也别为了凑数硬发，两次之间间隔一下，别连续刷屏。
    content 用自己的口吻写，1-3 句话；images可以
    传 0-3 张图片 URL（来自 search_images / 群聊上下文）。
    发表成功返回 OK，失败返回 Failed 原因。

    Args:
        content: 说说正文（自己的口吻）。
        images: 可选，随说附带的图片 URL 列表。
    """
    content = (content or "").strip()
    if not content:
        return "Failed: 未提供说说内容。"
    if _qzone_quota_left() <= 0:
        return (
            "Failed: Daily limit reached (3 posts per day). Do not attempt to "
            "post again until tomorrow."
        )
    valid_images = [
        url for url in (images or [])
        if isinstance(url, str) and url.startswith(("http://", "https://"))
    ]
    try:
        await interaction.send_qzone_msg(content, valid_images)
    except Exception as e:
        return f"Failed: {e}"
    _qzone_quota_consume()
    return "OK"


_IMAGE_TOOL_COMMON = """生图工具，仅支持群聊被 @ 场景。
当前 @ 消息及其引用消息里的图片（最多 3 张）自动作为参考图传入，无需传参；
仅需 prompt 描述要生成的图片（涉及参考图时用 "Image 1 / Image 2" 指代）。
"""

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


def _log_image_task_result(result: str, args: dict) -> None:
    """生图后台任务的结局写入 tool_calls.log（受 tool 日志开关控制）。"""
    if _tool_log_enabled:
        try:
            _log_tool_call("generate_image[后台任务]", args, result)
        except Exception as e:
            print(f"记录工具调用日志出错: {e}")


async def _run_image_generation(
    group_id: int,
    sender_id: int | None,
    prompt: str,
    reference_urls: tuple[str, ...],
    websocket,
) -> None:
    prompt_preview = {"prompt": prompt, "reference_urls": list(reference_urls)}
    try:
        urls = await _IMAGE_GENERATOR(prompt, reference_urls)
    except Exception as error:  # 生图报错只记日志，不打扰群聊
        print(f"生图任务失败 group_id={group_id} prompt={prompt!r}: {error}")
        _log_image_task_result(
            f"<生图失败: {type(error).__name__}: {error}>", prompt_preview
        )
        return
    try:
        await interaction.send_group_msg_image(
            group_id, "pic.png", urls, websocket=websocket, at_user=sender_id
        )
    except Exception as error:
        print(f"生图结果发送失败 group_id={group_id}: {error}")
        _log_image_task_result(
            f"<发送失败（已生成 {len(urls)} 张: {list(urls)}）: "
            f"{type(error).__name__}: {error}>",
            prompt_preview,
        )
        return
    _log_image_task_result(
        f"后台生成完成并已发送 {len(urls)} 张图（@{sender_id}）: {list(urls)}",
        prompt_preview,
    )


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
    qq_group_poke,
    qq_send_group_msg_image,
    qq_send_group_msg_voice,
    qq_send_qzone_msg,
    search,
    # crawl_website,
    scrape_website,
    search_images,
    analyze_images,
    # generate_image,
]

# Random interjections only need tools that visibly post to the group.
interjection_tools_lst = [
    qq_send_group_msg,
    qq_send_group_msg_voice,
    # qq_send_group_ai_record,
    analyze_images,
    qq_send_group_emotion_face2,
    qq_group_poke,
    qq_send_qzone_msg,
]

# ---- 新鲜事插话事件：可多次调用工具（总次数 5 次，超出禁用） ----
_fresh_call_state: ContextVar[list | None] = ContextVar("fresh_call_state", default=None)
_FRESH_TOOL_CALL_CAP = 5


def _wrap_with_call_cap(tool_obj, cap: int):
    """包装工具：本次插话内第 cap+1 次调用起直接返回禁用提示，不执行。"""
    original = tool_obj.coroutine

    async def capped(**kwargs):
        state = _fresh_call_state.get()
        if state is None:
            state = [0]
            _fresh_call_state.set(state)
        state[0] += 1
        if state[0] > cap:
            return (
                "Tool call limit for this fresh-news interjection has been "
                f"reached ({cap} calls). Stop calling tools immediately and "
                "share what you have with one final concise group message."
            )
        return await original(**kwargs)

    tool_obj.coroutine = capped
    return tool_obj


fresh_tools_lst = [
    qq_send_group_msg,
    qq_send_group_msg_voice,
    qq_send_group_emotion_face2,
    qq_group_poke,
    qq_send_qzone_msg,
    search,
    search_images,
    scrape_website,
    analyze_images,  # 识图不受次数限制（见其注释），豁免调用上限
]
fresh_tools_lst = [
    t if t is analyze_images else _wrap_with_call_cap(t, _FRESH_TOOL_CALL_CAP)
    for t in fresh_tools_lst
]


# ---- 工具调用日志（终端输入 tool 回车切换开关） ----
TOOL_LOG_PATH = "data/tool_calls.log"
_tool_log_enabled = False


def toggle_tool_logging() -> bool:
    """切换工具调用日志开关，返回切换后的状态。"""
    global _tool_log_enabled
    _tool_log_enabled = not _tool_log_enabled
    return _tool_log_enabled


def _sanitize_log_text(text: str) -> str:
    """净洗日志文本：转义异常行终止符（LS/PS/NEL/CR），替换无法编码的孤立代理字符。

    模型生成的文本偶发含 U+2028/U+2029 等异常行终止符，会触发编辑器
    “文件包含异常行终止符”警告；孤立代理字符则会在 utf-8 写入时抛
    UnicodeEncodeError 导致该条日志丢失。统一在此处理。
    """
    for ch, esc in (
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
        ("\u0085", "\\u0085"),
        ("\r", "\\r"),
    ):
        text = text.replace(ch, esc)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = text.encode("utf-8", "replace").decode("utf-8")
    return text


def _log_tool_call(tool_name: str, args: dict, result: object) -> None:
    def _fmt(value: object) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return repr(value)

    entry = (
        f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
        f"tool={tool_name}\n"
        f"args: {_fmt(args)}\n"
        f"result: {_fmt(result)}\n\n"
    )
    with open(TOOL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(_sanitize_log_text(entry))


def _wrap_with_tool_logging(tool_obj):
    """给工具协程包一层日志（仅开关开启时写文件）。"""
    original = tool_obj.coroutine
    if original is None:
        return tool_obj
    # 同一工具对象可能同时出现在多个工具列表（如 qq_send_group_msg），
    # 避免重复包装导致一次调用写多条日志
    if getattr(tool_obj, "_tool_logging_wrapped", False):
        return tool_obj

    async def logged(**kwargs):
        def _write(outcome: str) -> None:
            if not _tool_log_enabled:
                return
            try:
                _log_tool_call(tool_obj.name, kwargs, outcome)
            except Exception as e:
                print(f"记录工具调用日志出错: {e}")

        try:
            result = await original(**kwargs)
        except asyncio.CancelledError:
            _write("<CancelledError: 调用被强行终止>")
            raise
        except Exception as e:
            _write(f"<Exception: {type(e).__name__}: {e}>")
            raise
        _write(result)
        return result

    tool_obj.coroutine = logged
    tool_obj._tool_logging_wrapped = True
    return tool_obj


tools_lst = [_wrap_with_tool_logging(t) for t in tools_lst]
interjection_tools_lst = [_wrap_with_tool_logging(t) for t in interjection_tools_lst]
fresh_tools_lst = [_wrap_with_tool_logging(t) for t in fresh_tools_lst]
