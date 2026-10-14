import asyncio
import config
from datetime import datetime
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage

from .conversation_history import (
    append_turn,
    build_assistant_combined_for_cache,
    extract_final_assistant_text,
    history_as_messages,
)
from .group_prompt import get_active_prompt_name, get_prompt_text
from .tools import interjection_tools_lst, tools_lst

_RANDOM_INTERJECTION_PROMPT = """
现在没有任何人专门对你说话，只是群友们在自己闲聊，所以不要以对方在向自己说话的方式回复，而是要结合历史信息发送结合当前情境下适当的消息。
这是一次随机群聊插话。必须执行 1 或 2 次工具调用，然后结束：只能使用可见的群发送工具。
`qq_send_group_emotion_face2` 最多调用一次。工具调用才会发送到群里；你的最终文本不会被发送，
也不会存储到对话上下文，所以不要在最终文本中写任何消息内容。不要使用搜索或网页抓取。
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


class QQGroupAgent:
    """LangChain agent for QQ group @-mentions; `ainvoke` returns a `messages` dict like LangGraph."""

    def __init__(self) -> None:
        self._inv_lock = asyncio.Lock()
        self.model = ChatOpenAI(
            model=config.DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=4096,
            timeout=120,
            api_key=config.AI_KEY,
            base_url=config.AI_BASE_URL,
            extra_body={"thinking": {"type": "disabled"}},
        )
        # 仅在消息携带图片时使用；未配置 VISION_MODEL 时退回默认模型
        self.vision_model = ChatOpenAI(
            model=config.VISION_MODEL or config.DEFAULT_MODEL,
            temperature=0.1,
            max_tokens=4096,
            timeout=120,
            api_key=config.AI_KEY,
            base_url=config.AI_BASE_URL,
            extra_body={"thinking": {"type": "disabled"}},
        )
        # (提示词变量名, 用途) -> agent 实例；按 (群当前提示词, 是否视觉/是否插话) 懒创建并缓存
        self._agent_cache: dict[tuple[str, str], object] = {}

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
                middleware=[ToolCallLimitMiddleware(run_limit=2)],
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
        """Run a tool-only random interjection without reading or writing history."""
        content = (
            "Random interjection context (use real IDs in tools):\n"
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
        async with self._inv_lock:
            await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=_build_multimodal_human_content(content, image_inputs)
                        )
                    ]
                }
            )

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
        async with self._inv_lock:
            result = await agent.ainvoke({"messages": messages_in})
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
        async with self._inv_lock:
            result = await agent.ainvoke({"messages": messages_in})
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
