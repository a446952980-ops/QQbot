import asyncio
import config
from datetime import datetime
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage

from .conversation_history import (
    append_turn,
    build_assistant_combined_for_cache,
    extract_final_assistant_text,
    history_as_messages,
)
from .tools import tools_lst

from .system_prompt import sys_prompt


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
        )
        self.agent = create_agent(
            model=self.model,
            tools=tools_lst,
            system_prompt=SystemMessage(
                content=[{"type": "text", "text": sys_prompt}]
            ),
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
    ) -> list[HumanMessage]:
        return [
            HumanMessage(
                content=_build_group_at_human_content(
                    group_id=group_id,
                    group_name=group_name,
                    sender_name=sender_name,
                    sender_id=sender_id,
                    msg_id=msg_id,
                    text=text,
                )
            )
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
        prior = history_as_messages(group_id)
        messages_in = prior + [HumanMessage(content=user_content)]
        async with self._inv_lock:
            result = await self.agent.ainvoke({"messages": messages_in})
        messages_full = result.get("messages", [])
        # Only this round (exclude re-injected history) so cache is not duplicated.
        suffix = messages_full[len(prior) :]
        assistant_combined = build_assistant_combined_for_cache(suffix)
        append_turn(group_id, user_content, assistant_combined)
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
        async with self._inv_lock:
            result = await self.agent.ainvoke({"messages": messages_in})
        messages_full = result.get("messages", [])
        suffix = messages_full[len(prior) :]
        assistant_combined = build_assistant_combined_for_cache(suffix)
        append_turn(group_id, user_content, assistant_combined)
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
