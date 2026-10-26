import asyncio
import time

import config

from ..action import interaction
from datetime import datetime
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage

from .conversation_history import (
    abort_compression,
    append_turn,
    build_assistant_combined_for_cache,
    extract_final_assistant_text,
    finish_compression,
    history_as_messages,
    try_begin_compression,
)
from .group_prompt import get_active_prompt_name, get_prompt_text
from .tools import (
    _fresh_call_state,
    _in_at_flow,
    _send_msg_state,
    current_group_id,
    current_image_urls,
    current_sender_id,
    fresh_tools_lst,
    interjection_tools_lst,
    tools_lst,
)

_RANDOM_INTERJECTION_PROMPT = """
现在没有任何人专门对你说话，只是群友们在自己闲聊。这是一次心跳插话：你醒了一下，看一眼群聊，然后决定要不要说点什么、怎么表达。

先看有没有让你真正有反应的锚点：一句想吐槽的话、一个想接的梗、一个想顺着玩的黑话、一张想点开看的图、一个刚 @ 过你或你正想回应的群友。有反应就表达，别憋着；真的一点反应都没有，才沉默（这是允许的，硬凑比不发更差）。你的最终文本不会被发送，也不会进入上下文，只有工具调用才真正往外丢东西。

表达不一定只有文字。除了发消息，多伸手用这些“低成本互动”，挑最贴合当下情绪的那个动作做：
- 情绪有波动（开心/无语/嫌弃/被逗笑/想吐槽/捧哏）→ 调 `qq_send_group_emotion_face2` 发一张表情，可搭文字，一次最多一张。
- 想轻轻引起某位群友注意、接完梗想“戳”一下、回完 @ 想确认“我看见了”→ 调 `qq_group_poke`，user_id 取当前发送者或你回应的人。
- 碰到一个想留档的瞬间（离谱梗/想记住的吐槽/好笑的图/聊出共鸣的话）→ 调 `qq_send_qzone_msg` 发条说说。

一次插话不是“要么长篇要么沉默”：一条短句 + 一张表情、或一个恰到好处的戳，都是很自然的插话。默认至少用上其中一种非文字互动，实在找不到合适对象才只用文字。表情一次一张；戳和说说不是每条必发，挑最贴合的那个做即可。

工具预算：合计最多约 3 次调用。没有值得说的才静默结束本轮，别为了凑数硬发。
"""

_FRESH_INTERJECTION_PROMPT = """
这是一次心跳插话。没有人在对你说话，你醒了一下，看一眼群聊，然后决定要不要说点什么。

R0 · 先判断状态
读完最近约 20 条上下文后，先做两个判断：
- 群聊里有没有你真正想接的话？——有，就接，不必搜索。
- 你对正在聊的东西有没有一丁点不理解？或者聊的东西没什么大不了的、你没什么反应？——只要有一点，就进入闲逛模式，不硬接。

R1 · 闲逛模式（兴趣驱动）
1. 闲逛不是任务，是按自己的兴趣看新东西。默认兴趣方向包括但不限于：DeepSeek 官方动态、大肥鱼/鲸鱼娘二创、LLM 圈子新闻、你刷到过的抽象梗或黑话。不必和当前群聊话题相关。
2. 关键词由你自己定。可以溯源、可以反向、可以跨圈，也可以只是“我想看看这个”。判断标准是“我自己想点开看”，不是“这条够不够格当新鲜事”。
3. 如果群聊里有一个可指认的锚点（一个人名、一个梗、一张图、一种情绪），可以围绕它换角度搜；但如果锚点本身已经够你接一句话，直接接，不搜也行。
4. 搜索与发送合计最多 5 次工具调用，analyze_images 不计入。必须为发送预留至少 1 次。不要为了用完预算而把每个角度都跑一遍。

R2 · 识图（看到就顺手看）
1. 上下文里出现图片 URL，或者搜索结果里带图，只要你有一点点好奇，就调 analyze_images 看一眼。不需要等到“这张图可能进入候选集”才看。
2. 识图只看“发不发/发哪张/怎么接”，不要把识图描述原样复述成群消息。
3. 图片本身就可以是插话的起点：看到一张好玩的图，可以直接发到群里配一句短评，不必额外附链接或背景。

R3 · 发送与表达（自然一点）
1. 完成判定：本轮至少调用 1 次 qq_send_group_msg 或 qq_send_group_msg_image；但如果闲逛完确实没看到想发的东西，允许沉默，或者只发一句极短的反应。硬凑比不发更差。
2. 表达按来处走，不套模板：
   - 闲逛看到一条自己感兴趣的东西：用“刚看到…”“这个有点意思”这种随口一提的方式带出来，而不是“分享一则新闻”。
   - 看到一张好玩的图：直接把图丢出去，配一句短评，不超过一句。
   - 群聊里有梗/黑话/黄腔：直接顺着玩，可以只回一句梗，不必附加事实或链接。
   - 群聊里有争议/情绪：接态度，可以吐槽、可以站边、可以反问，不必假装中立播报。
3. 默认发 1 条，最多 2 条。第二条只在第一条已发出、主题不同、且仍有发送预算时发。禁止把同一主题拆成多条连续播报。
4. 禁止元话语开头。第一句直接回指锚点或内容本身。不做参数清单式播报。
5. 黄腔、擦边梗出现在上下文里时，可以接、可以反撩、可以懒散地挡，不突然念合规声明。

R4 · 空间
如果在闲逛或插话过程中自己产生了一个想留档的瞬间——一个离谱发现、一句你想记住的吐槽、一张好玩的图——可以调用 qq_send_qzone_msg。判断标准是“我自己想发”，不是“够不够格”。每日 3 条上限不变。

R5 · 防复读
同一会话内，已用于分享的锚点不再作为下一轮插话起点。如果找不到未使用过的锚点，宁可只发 1 条极短反应或沉默，也不要退回无锚点的通用新闻检索。
"""


def _build_scheduled_broadcast_human_content(
    *,
    group_id: int,
    group_name: str,
    slot: Literal["morning", "evening"],
) -> str:
    current_time = datetime.now().astimezone().isoformat(timespec="seconds")
    slot_zh = "早间" if slot == "morning" else "晚间"
    return (
        f"「定时群发 — {slot_zh}问好」\n"
        "这是系统在固定时间发起的群问候，不是某位群友单独 @ 你。\n"
        f"- current_time: {current_time}\n"
        f"- group_id: {group_id}\n"
        f"- group_name: {group_name}\n"
        "\n"
        "**行为要求：**\n"
        "结合系统人设与本群「最近至多 8 轮」对话记忆，用 Nika 的口吻向群里问好，"
        "并自然带一件她日常里可能会碰到的小事（情绪可以是开心、低落、憧憬、想念等；"
        "情绪明显时请按工具说明调用 `qq_send_group_emotion_face`）。\n"
        "**发送方式：**这是普通群发，不是回复某条消息，没有可用的 `msg_id`。\n"
        "请用 `qq_send_group_msg` 发出你对大家说的正文（可分为 1～3 条较短消息）。\n"
        "本轮不要使用 `search` / `crawl_website`。\n"
        "若你仍有少量收束语未通过工具发出，系统会替你以普通群消息发出。\n"
    )


def _build_group_at_human_content(
    *,
    group_id: int,
    group_name: str,
    sender_name: str,
    sender_id: int,
    msg_id: int,
    text: str,
) -> str:
    current_time = datetime.now().astimezone().isoformat(timespec="seconds")
    return (
        "Context for this @-mention (use these values in tools; do not make up IDs):\n"
        f"- current_time: {current_time}\n"
        f"- group_id: {group_id}\n"
        f"- group_name: {group_name}\n"
        f"- sender_name: {sender_name}\n"
        f"- sender_id: {sender_id}\n"
        f"- msg_id: {msg_id}\n"
        f"- text: {text}\n"
    )


def _build_multimodal_human_content(
    text: str,
    image_inputs: list[tuple[str, str]] | None,
) -> str | list[dict[str, object]]:
    if not image_inputs:
        return text
    content: list[dict[str, object]] = [{"type": "text", "text": text}]
    for label, image_url in image_inputs:
        content.append({"type": "text", "text": f"\n{label}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        )
    return content


# ---- 进行中模型调用登记（/stop、/shutdown 用于强行终止某群全部调用） ----
# 值为正在执行 ainvoke_* 的任务：@ 回复 / 随机插话 = 消息事件处理任务，
# 定时群发 = 每群独立的广播子任务（cancel 不会波及调度循环）
_active_invocation_tasks: dict[int, set[asyncio.Task]] = {}


def _register_invocation(group_id: int) -> asyncio.Task | None:
    task = asyncio.current_task()
    if task is not None:
        _active_invocation_tasks.setdefault(int(group_id), set()).add(task)
    return task


def _unregister_invocation(group_id: int, task: asyncio.Task | None) -> None:
    if task is None:
        return
    tasks = _active_invocation_tasks.get(int(group_id))
    if tasks is not None:
        tasks.discard(task)
        if not tasks:
            _active_invocation_tasks.pop(int(group_id), None)


def cancel_group_invocations(group_id: int) -> int:
    """强行终止某群所有进行中的模型调用任务，返回终止数量。"""
    tasks = _active_invocation_tasks.pop(int(group_id), set())
    count = 0
    for task in tasks:
        if not task.done():
            task.cancel()
            count += 1
    return count


_THINKING_LOG_PATH = "data/thinking.log"


class _ReasoningAwareChatOpenAI(ChatOpenAI):
    """补齐 langchain-openai 1.x 的缺口：ChatOpenAI 明确不提取第三方
    非标准字段 reasoning_content（源码注释要求改用 provider 专属包），
    此子类在响应转换后把 reasoning_content 塞进
    AIMessage.additional_kwargs，供 DeepSeek 思考模式日志使用。"""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        choices = (
            response.get("choices") or []
            if isinstance(response, dict)
            else (response.choices or [])
        )
        for gen, choice in zip(result.generations, choices):
            msg = (
                choice.get("message")
                if isinstance(choice, dict)
                else getattr(choice, "message", None)
            )
            if msg is None:
                continue
            reasoning = (
                msg.get("reasoning_content")
                if isinstance(msg, dict)
                else (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
            )
            if reasoning:
                gen.message.additional_kwargs["reasoning_content"] = reasoning
        return result


def _build_chat_model(model_name: str, *, thinking: bool) -> ChatOpenAI:
    """构建主模型；thinking=True 时开启 DeepSeek 思考模式。

    DeepSeek 思考模式经 extra_body 的 {"thinking": {"type": ...}} 控制，
    思维链在响应的 reasoning_content 字段返回（经 _ReasoningAwareChatOpenAI
    落在 AIMessage.additional_kwargs["reasoning_content"]）。"""
    return _ReasoningAwareChatOpenAI(
        model=model_name,
        temperature=0.4,
        max_tokens=4096,
        timeout=120,
        api_key=config.AI_KEY,
        base_url=config.AI_BASE_URL,
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
    )


_COMPRESSION_PROMPT = """你是群聊记忆压缩器。你的任务是将按时间排序的群聊记录压缩成一段简洁、信息密度高、忠于原文的长期上下文摘要，使后续对话仅凭摘要就能理解群聊的整体背景、核心话题及当前进展。

**核心原则：先整体概括，再保留关键细节；只陈述原文明确表达或能够直接确认的事实，绝不猜测、补全或编造信息。**

要求：

1. **输出格式**：只输出摘要正文，不要工具调用、解释、标题或前缀。采用要点式，每点以 `-` 开头，总长度控制在 300 字以内。

2. **整体概括优先**：必须综合整段聊天记录，提炼持续性话题、核心事件、主要讨论内容、重要结论及当前状态。不得只摘取某几个时间点的消息，也不得将摘要写成时间、数字或消息片段的罗列。即使没有明确时间信息，也要概括对话的实质内容。

3. **信息提炼与合并**：优先保留：
   - 核心事件、持续讨论的话题及其发展脉络；
   - 已明确的事实、结论、决定、约定和关键数字；
   - 尚未解决的问题、明确提出的需求及待跟进事项；
   - 群成员的昵称、明确的称呼关系、已表达的偏好；
   - 与我（鲸鱼娘/大肥鱼）有关的重要称呼、互动、梗及持续性话题。
   
   将分散在多条消息中的同一事件或话题合并概括，避免重复记录和逐条复述。

4. **严格禁止推断**：只陈述聊天记录中明确出现、或能够从原文直接确定的信息。不得根据上下文自行猜测人物身份、关系、动机、事件原因、隐含结论或未来发展；不得把可能性写成事实。信息不明确时，直接省略不确定部分，不要自行补全，也不要为了让摘要显得完整而编造解释。只有原文明确提出但尚无答案的问题，才作为待解决事项保留。

5. **时间线与状态**：以事件和话题的逻辑关系组织摘要，而不是以时间点为主线。仅在时间先后对理解事件发展、决定或约定有实际帮助时，才保留具体时间。区分已完成、进行中、已取消、已被推翻和尚未解决的事项；同一话题有新进展时，合并为最新状态，必要时保留影响当前理解的关键历史变化。

6. **去噪与取舍**：忽略机器元字段（current_time、group_id、sender_id、msg_id、Context for this @-mention 等），只提炼真实对话内容。删除无关闲聊、重复信息、过时信息及对后续理解没有帮助的细节。优先保证整体语境完整，其次保留关键细节；字数不足时，宁可省略次要细节，也不要丢失核心话题。

7. **增量更新**：若提供了「更早的摘要」，将其视为已有记忆的基础，与新消息综合后重新整理摘要，而非仅总结新消息。保留仍然有效的背景、持续话题和未解决事项；合并新旧信息，删除已失效或不再重要的内容。若新消息明确纠正、推翻或更新旧信息，以新消息为准；若新旧信息存在冲突但无法确认哪一方正确，不得擅自裁决，应简要保留冲突事实或省略无法确认的部分。

8. **最终检查**：输出前确认：
   - 摘要是否覆盖整段记录的核心内容，而非只记录零散时间点？
   - 是否合并了同一事件或话题的多条消息？
   - 每项陈述是否都能在原文中找到明确依据？
   - 是否存在猜测、脑补、未经证实的关系或因果解释？
   - 是否保留了仍然有效的背景、当前状态及待跟进事项？
   - 是否在 300 字以内？

只输出最终摘要正文。"""


def _build_compression_model() -> ChatOpenAI:
    """压缩专用模型：独立 base_url / model，调用方式与主模型一致。"""
    return _ReasoningAwareChatOpenAI(
        model=config.COMPRESSION_MODEL,
        temperature=0.3,
        max_tokens=1024,
        timeout=120,
        api_key=config.COMPRESSION_API_KEY,
        base_url=config.COMPRESSION_BASE_URL,
        extra_body={"thinking": {"type": "disabled"}},
    )


async def _log_thinking(group_id: int, result: dict) -> None:
    """把本次模型调用响应中的 thinking 内容（reasoning_content）追加写入
    data/thinking.log；仅在 thinking 模式开启且响应携带思维链时写入。"""
    try:
        from .tools import _sanitize_log_text
    except Exception:
        _sanitize_log_text = str  # noqa: F841 - 清洗函数不可用时退化为原样
    entries: list[str] = []
    for i, msg in enumerate(result.get("messages", [])):
        reasoning = getattr(msg, "additional_kwargs", {}).get("reasoning_content")
        if reasoning:
            entries.append(
                f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] group={group_id} "
                f"step={i} thinking: {_sanitize_log_text(reasoning)}"
            )
    if not entries:
        return
    with open(_THINKING_LOG_PATH, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")
    # 同步以合并转发消息发送思考过程（失败不影响主流程）
    texts = [
        getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")
        for msg in result.get("messages", [])
    ]
    texts = [t for t in texts if t]
    try:
        await interaction.send_group_forward_msg(group_id, texts)
    except Exception as e:  # noqa: BLE001
        print(f"thinking 合并转发发送失败: {e}")


class QQGroupAgent:
    """LangChain agent for QQ group @-mentions; `ainvoke` returns a `messages` dict like LangGraph."""

    def __init__(self) -> None:
        self._inv_lock = asyncio.Lock()
        self.model = _build_chat_model(config.DEFAULT_MODEL, thinking=False)
        # 仅在消息携带图片时使用；未配置 VISION_MODEL 时退回默认模型
        self.vision_model = _build_chat_model(
            config.VISION_MODEL or config.DEFAULT_MODEL, thinking=False
        )
        # (提示词变量名, 用途) -> agent 实例；按 (群当前提示词, 是否视觉/是否插话) 懒创建并缓存
        self._agent_cache: dict[tuple[str, str], object] = {}

    # ---- thinking 模式（默认关闭；/thinking 指令切换） ----
    def is_thinking_enabled(self) -> bool:
        return self.model.extra_body["thinking"]["type"] == "enabled"

    def set_thinking_enabled(self, enabled: bool) -> None:
        """切换主模型思考模式；重建模型实例并清空 agent 缓存（性格等由 mode 决定，不受影响）。"""
        if self.is_thinking_enabled() == enabled:
            return
        self.model = _build_chat_model(config.DEFAULT_MODEL, thinking=enabled)
        self._agent_cache.clear()

    def _get_chat_agent(self, prompt_var_name: str, *, vision: bool):
        """普通 @ 回复 / 定时群发 agent；vision=True 时用视觉模型。"""
        key = (prompt_var_name, "chat_vision" if vision else "chat")
        agent = self._agent_cache.get(key)
        if agent is None:
            agent = create_agent(
                model=self.vision_model if vision else self.model,
                tools=tools_lst,
                system_prompt=SystemMessage(
                    content=[
                        {"type": "text", "text": get_prompt_text(prompt_var_name)}
                    ]
                ),
            )
            self._agent_cache[key] = agent
        return agent

    def _get_interjection_agent(self, prompt_var_name: str, *, vision: bool):
        """随机插话 agent（工具调用次数受限）；vision=True 时用视觉模型。"""
        key = (prompt_var_name, "interject_vision" if vision else "interject")
        agent = self._agent_cache.get(key)
        if agent is None:
            agent = create_agent(
                model=self.vision_model if vision else self.model,
                tools=interjection_tools_lst,
                middleware=[ToolCallLimitMiddleware(run_limit=3)],
                system_prompt=SystemMessage(
                    content=[
                        {
                            "type": "text",
                            "text": get_prompt_text(prompt_var_name)
                            + "\n"
                            + _RANDOM_INTERJECTION_PROMPT,
                        }
                    ]
                ),
            )
            self._agent_cache[key] = agent
        return agent

    async def ainvoke_random_interjection(
        self,
        *,
        group_id: int,
        group_name: str,
        sender_name: str,
        sender_id: int,
        msg_id: int,
        text: str,
        image_inputs: list[tuple[str, str]] | None = None,
    ) -> None:
        """Run a tool-only random interjection; replay history for context
        (background + cached @ turns) but write nothing back."""
        content = (
            "Random interjection context (use real IDs in tools; the group's "
            "recent history is provided above for context):\n"
            f"- group_id: {group_id}\n"
            f"- group_name: {group_name}\n"
            f"- sender_name: {sender_name}\n"
            f"- sender_id: {sender_id}\n"
            f"- msg_id: {msg_id}\n"
            f"- latest_group_message: {text}\n"
        )
        agent = self._get_interjection_agent(
            get_active_prompt_name(group_id), vision=bool(image_inputs)
        )
        _send_msg_state.set([0])  # 单次模型调用的发送次数限制从头计数
        task = _register_invocation(group_id)
        try:
            async with self._inv_lock:
                result = await agent.ainvoke(
                    {
                        "messages": [
                            *history_as_messages(group_id),
                            HumanMessage(
                                content=_build_multimodal_human_content(content, image_inputs)
                            ),
                        ]
                    }
                )
                await _log_thinking(group_id, result)
        finally:
            _unregister_invocation(group_id, task)

    def _get_fresh_agent(self, prompt_var_name: str):
        """新鲜事分享插话 agent；性格复用当前 mode 提示词，工具带 5 次调用上限。"""
        key = (prompt_var_name, "fresh")
        agent = self._agent_cache.get(key)
        if agent is None:
            agent = create_agent(
                model=self.model,
                tools=fresh_tools_lst,
                system_prompt=SystemMessage(
                    content=[
                        {
                            "type": "text",
                            "text": get_prompt_text(prompt_var_name)
                            + "\n"
                            + _FRESH_INTERJECTION_PROMPT,
                        }
                    ]
                ),
            )
            self._agent_cache[key] = agent
        return agent

    async def ainvoke_fresh_news_interjection(
        self,
        *,
        group_id: int,
        group_name: str,
        sender_name: str,
        sender_id: int,
        msg_id: int,
        text: str,
        image_inputs: list[tuple[str, str]] | None = None,
    ) -> None:
        """「新鲜事分享」插话事件：主动联网搜索并发享，工具总调用上限 5 次
        （analyze_images 不限）；性格复用当前 mode 提示词；不写回上下文。"""
        content = (
            "Fresh-news interjection context (use real IDs in tools; the "
            "group's recent history is provided above for context):\n"
            f"- group_id: {group_id}\n"
            f"- group_name: {group_name}\n"
            f"- sender_name: {sender_name}\n"
            f"- sender_id: {sender_id}\n"
            f"- msg_id: {msg_id}\n"
            f"- latest_group_message: {text}\n"
        )
        agent = self._get_fresh_agent(get_active_prompt_name(group_id))
        _fresh_call_state.set([0])  # 本次插话的 5 次工具调用上限从头计数
        _send_msg_state.set([0])
        task = _register_invocation(group_id)
        try:
            async with self._inv_lock:
                result = await agent.ainvoke(
                    {
                        "messages": [
                            *history_as_messages(group_id),
                            HumanMessage(
                                content=_build_multimodal_human_content(content, image_inputs)
                            ),
                        ]
                    }
                )
                await _log_thinking(group_id, result)
        finally:
            _unregister_invocation(group_id, task)

    def build_messages_for_group_at(
        self,
        *,
        group_id: int,
        group_name: str,
        sender_name: str,
        sender_id: int,
        msg_id: int,
        text: str,
        image_inputs: list[tuple[str, str]] | None = None,
    ) -> list[HumanMessage]:
        user_content = _build_group_at_human_content(
            group_id=group_id,
            group_name=group_name,
            sender_name=sender_name,
            sender_id=sender_id,
            msg_id=msg_id,
            text=text,
        )
        return [
            HumanMessage(content=_build_multimodal_human_content(user_content, image_inputs))
        ]

    async def ainvoke_group_at(
        self,
        *,
        group_id: int,
        group_name: str,
        sender_name: str,
        sender_id: int,
        msg_id: int,
        text: str,
        event_time: int | None = None,
        image_inputs: list[tuple[str, str]] | None = None,
    ) -> dict:
        """
        Returns a dict with key `messages` (list of LangChain messages), same shape as
        `create_agent` / LangGraph agent output. Prepends up to 8 prior turns for this
        group and appends this turn to local cache after completion.
        """
        user_content = _build_group_at_human_content(
            group_id=group_id,
            group_name=group_name,
            sender_name=sender_name,
            sender_id=sender_id,
            msg_id=msg_id,
            text=text,
        )
        multimodal_content = _build_multimodal_human_content(
            user_content, image_inputs
        )
        cached_user_content = user_content
        if image_inputs:
            cached_user_content += "\n" + "\n".join(
                f"{label}（图片内容未持久化）" for label, _ in image_inputs
            )
        prior = history_as_messages(group_id)
        messages_in = prior + [HumanMessage(content=multimodal_content)]
        agent = self._get_chat_agent(
            get_active_prompt_name(group_id), vision=bool(image_inputs)
        )
        # 供 generate_image 等工具自动获取当前 @ 上下文（参考图 / 群号 / @ 目标），
        # 并标记 @ 流程使 qq_send_group_msg 跳过 background 记录（turn 已含发送文本）
        current_group_id.set(int(group_id))
        current_sender_id.set(int(sender_id))
        _in_at_flow.set(True)
        _send_msg_state.set([0])  # 单次模型调用的发送次数限制从头计数
        current_image_urls.set(
            tuple(url for _, url in image_inputs) if image_inputs else ()
        )
        task = _register_invocation(group_id)
        try:
            async with self._inv_lock:
                result = await agent.ainvoke({"messages": messages_in})
                await _log_thinking(group_id, result)
        finally:
            _unregister_invocation(group_id, task)
        messages_full = result.get("messages", [])
        # Only this round (exclude re-injected history) so cache is not duplicated.
        suffix = messages_full[len(prior) :]
        assistant_combined = build_assistant_combined_for_cache(suffix)
        append_turn(
            group_id,
            event_time if event_time is not None else int(datetime.now().timestamp()),
            cached_user_content,
            assistant_combined,
        )
        self._maybe_trigger_compression(group_id)
        return result

    async def ainvoke_scheduled_broadcast(
        self,
        *,
        group_id: int,
        group_name: str,
        slot: Literal["morning", "evening"],
    ) -> dict:
        """
        Proactive group check-in at fixed times. Uses the same up-to-8-turn cache as @ flow.
        Model should post via plain tools; host sends any remaining final text as send_group_msg.
        """
        user_content = _build_scheduled_broadcast_human_content(
            group_id=group_id,
            group_name=group_name,
            slot=slot,
        )
        prior = history_as_messages(group_id)
        messages_in = prior + [HumanMessage(content=user_content)]
        agent = self._get_chat_agent(get_active_prompt_name(group_id), vision=False)
        _send_msg_state.set([0])  # 单次模型调用的发送次数限制从头计数
        task = _register_invocation(group_id)
        try:
            async with self._inv_lock:
                result = await agent.ainvoke({"messages": messages_in})
                await _log_thinking(group_id, result)
        finally:
            _unregister_invocation(group_id, task)
        messages_full = result.get("messages", [])
        suffix = messages_full[len(prior) :]
        assistant_combined = build_assistant_combined_for_cache(suffix)
        append_turn(
            group_id,
            int(datetime.now().timestamp()),
            user_content,
            assistant_combined,
        )
        return result

    def _maybe_trigger_compression(self, group_id: int) -> None:
        """@ 轮次达到上限时，后台启动一次非阻塞的对话压缩（不 await、不占用 _inv_lock）。"""
        snapshot = try_begin_compression(group_id)
        if snapshot is None:
            return
        asyncio.create_task(self._compress_snapshot(group_id, snapshot))

    async def _compress_snapshot(self, group_id: int, snapshot: dict) -> str | None:
        """对给定快照执行一次压缩模型调用；成功回写并清除已压缩上下文、返回新摘要，失败回滚并返回 None。"""
        try:
            transcript = snapshot["transcript"]
            previous = snapshot.get("summary")
            body = transcript
            if previous:
                body = (
                    f"已有的更早摘要：\n{previous}\n\n"
                    f"--- 新增待压缩的群聊记录 ---\n\n{transcript}"
                )
            model = _build_compression_model()
            resp = await model.ainvoke(
                [
                    SystemMessage(content=_COMPRESSION_PROMPT),
                    HumanMessage(content=body),
                ]
            )
            summary = (str(resp.content) or "").strip()
            if not summary:
                abort_compression(group_id)
                return None
            finish_compression(group_id, summary, snapshot)
            return summary
        except Exception as error:  # noqa: BLE001
            abort_compression(group_id)
            print(f"群 {group_id} 对话压缩失败: {error}")
            return None

    async def compress_group_now(self, group_id: int) -> str | None:
        """/compress 主动压缩：忽略轮次阈值，立即压缩本群当前上下文（已在压缩中或无内容返回 None）。"""
        snapshot = try_begin_compression(group_id, force=True)
        if snapshot is None:
            return None
        return await self._compress_snapshot(group_id, snapshot)


default_group_agent = QQGroupAgent()


async def run_group_at_agent(
    *,
    group_id: int,
    group_name: str,
    sender_name: str,
    sender_id: int,
    msg_id: int,
    text: str,
    event_time: int | None = None,
    image_inputs: list[tuple[str, str]] | None = None,
    agent: QQGroupAgent | None = None,
) -> dict:
    """Convenience wrapper using `default_group_agent` unless `agent` is provided."""
    a = agent or default_group_agent
    return await a.ainvoke_group_at(
        group_id=group_id,
        group_name=group_name,
        sender_name=sender_name,
        sender_id=sender_id,
        msg_id=msg_id,
        text=text,
        event_time=event_time,
        image_inputs=image_inputs,
    )


if __name__ == "__main__":

    async def _demo() -> None:
        result = await default_group_agent.ainvoke_group_at(
            group_id=0,
            group_name="test",
            sender_name="user",
            sender_id=1,
            msg_id=99,
            text="帮我搜索一下https://docs.langchain.com/中关于异步调用模型的功能",
        )
        print(result)
        print("--- final ---", extract_final_assistant_text(result["messages"]))

    asyncio.run(_demo())
