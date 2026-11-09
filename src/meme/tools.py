"""供 agent 使用的梗工具：查询与记录（均不发送任何群消息）。"""

from __future__ import annotations

from langchain.tools import tool

from . import matcher, store


def _format_meme(meme: store.Meme) -> str:
    scope = "公共词库" if meme.scope == store.GLOBAL_SCOPE else "本群词条"
    parts = [f"「{meme.term}」（{scope}）", f"释义：{meme.meaning}"]
    if meme.origin:
        parts.append(f"出处：{meme.origin}")
    if meme.examples:
        parts.append("例句：" + " / ".join(meme.examples[:3]))
    if meme.offense:
        parts.append("注意：带冒犯/粗口")
    return "\n".join(parts)


@tool
async def lookup_meme(group_id: int, term: str) -> str:
    """
    查询一个网络梗/黑话/缩写的含义。看到不理解、可能是梗的词时调用。
    依次查：梗词库（本群词条优先）→ 拼音缩写猜测 → 联网搜索。
    联网结果仅供参考；若你据此确认了含义，且它是本群会反复用到的梗，
    可以调用 remember_meme 记入本群词库。本工具不会向群里发送任何内容。

    Args:
        group_id: 当前 QQ 群号。
        term: 要查的词，如 "尊嘟假嘟"、"xswl"。
    """
    term = " ".join((term or "").split())
    if not term:
        return "Failed: 未提供要查询的词。"
    meme = store.resolve(term, group_id)
    if meme is not None:
        return _format_meme(meme)
    parts: list[str] = []
    if matcher.abbreviation_candidates(term, group_id) == [term.lower()]:
        guesses = await matcher.guess_abbreviations([term.lower()])
        if guesses.get(term.lower()):
            parts.append("缩写猜测：" + " / ".join(guesses[term.lower()]))
    from ..agent.tools import _serper_post  # 延迟导入，避免循环依赖

    try:
        data = await _serper_post(
            "https://google.serper.dev/search",
            {"q": f"{term} 梗 是什么意思", "gl": "cn", "hl": "zh-cn", "num": 6},
        )
        for item in (data.get("organic") or [])[:5]:
            snippet = item.get("snippet") or ""
            if snippet:
                parts.append(f"- {item.get('title', '')}：{snippet}")
    except Exception as e:
        parts.append(f"联网搜索失败: {e}")
    if not parts:
        return f"词库未收录「{term}」，联网也没有找到相关解释。"
    return f"词库未收录「{term}」。\n" + "\n".join(parts)


@tool
async def remember_meme(group_id: int, term: str, meaning: str) -> str:
    """
    把一个已确认含义的梗记入本群词库（只对本群生效），之后群里再出现时会自动提示含义。
    仅在含义明确时调用；meaning 用一句话说清意思和用法，不超过 80 字。
    本工具不会向群里发送任何内容。

    Args:
        group_id: 当前 QQ 群号。
        term: 梗本身，2～20 字。
        meaning: 一句话释义。
    """
    term = " ".join((term or "").split())
    meaning = " ".join((meaning or "").split())
    if not 2 <= len(term) <= 20:
        return "Failed: term 需为 2～20 字。"
    if not 1 <= len(meaning) <= 120:
        return "Failed: meaning 需为 1～120 字。"
    store.add_group_meme(group_id, term, meaning, source="agent", operator="agent")
    return f"已记入本群词库：「{term}」"


meme_tools_lst = [lookup_meme, remember_meme]
