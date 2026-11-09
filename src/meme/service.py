"""把识别到的梗整理成上下文提示（只生成文本，不发送任何消息）。

- annotate_message：背景消息 / 随机插话用，仅做本地精确匹配，结果随消息写入 history；
- build_at_context：@ 回复用，精确匹配 + 缩写猜测 + 语义兜底；
  hints 写入本轮用户内容（随 turn 进入 history），suggestion（可选化用）只给本轮模型看。
同一个梗已在本群 history 中注释过就不再重复注释。
"""

from __future__ import annotations

import re

import config

from ..agent.conversation_history import history_texts
from . import embed, matcher
from .store import Meme, resolve

HINT_HEADER = "【涉及的梗（仅帮助理解；除非被问，别主动解释）】"
SUGGEST_HEADER = "【可选：本轮如果自然，可以化用下面这个梗；不合适就别用，别硬套】"
_MAX_HINTS = 3
_TERM_MARK = re.compile(r"「([^「」\n]{1,40})」")
_NOISE = re.compile(r"\[[^\[\]]*\]|https?://\S+")
_SELF_PREFIX = "【自己发送的消息"

# 每群自上次提供“可选化用”以来的 @ 次数（仅内存，用于控制频率）
_since_suggest: dict[int, int] = {}


def _clean(text: str) -> str:
    return " ".join(_NOISE.sub(" ", text or "").split())


def _hint_line(meme: Meme, note: str = "") -> str:
    line = f"「{meme.term}」{meme.short_meaning()}"
    if meme.offense:
        line += "（带冒犯/粗口，别模仿）"
    return line + note


def _annotated_terms(group_id: int) -> tuple[set[str], list[str], list[str]]:
    """(history 中已注释过的梗, 群友侧文本, 机器人侧文本)"""
    user_side, assistant_side = history_texts(group_id)
    peers = [t for t in user_side if not t.startswith(_SELF_PREFIX)]
    own = assistant_side + [t for t in user_side if t.startswith(_SELF_PREFIX)]
    annotated = {m.group(1) for t in peers for m in _TERM_MARK.finditer(t)}
    return annotated, peers, own


def _block(lines: list[str]) -> str:
    return HINT_HEADER + "\n" + "\n".join(lines) if lines else ""


def annotate_message(group_id: int, text: str) -> str:
    """本地精确匹配，返回提示块（无命中返回空串）。"""
    try:
        annotated, _, _ = _annotated_terms(group_id)
        lines = [
            _hint_line(m)
            for m in matcher.find_memes(_clean(text), group_id)
            if m.term not in annotated
        ]
        return _block(lines[:_MAX_HINTS])
    except Exception as e:
        print(f"[meme] 注释失败: {e}")
        return ""


async def build_at_context(
    group_id: int, current_text: str, quoted_text: str = ""
) -> tuple[str, str]:
    """返回 (hints, suggestion)，均可能为空串；内部异常不外抛。"""
    try:
        return await _build_at_context(group_id, current_text, quoted_text)
    except Exception as e:
        print(f"[meme] @ 上下文构建失败: {e}")
        return "", ""


async def _build_at_context(
    group_id: int, current_text: str, quoted_text: str
) -> tuple[str, str]:
    annotated, peers, own = _annotated_terms(group_id)
    current = _clean(current_text)
    found: list[Meme] = []
    for text in (current, _clean(quoted_text)):
        for meme in matcher.find_memes(text, group_id):
            if meme not in found:
                found.append(meme)
    lines = [_hint_line(m) for m in found if m.term not in annotated]

    abbr = await matcher.guess_abbreviations(
        matcher.abbreviation_candidates(current, group_id)
    )
    for word, trans in abbr.items():
        if f"「{word}」" not in "".join(peers):
            lines.append(f"「{word}」可能是：{' / '.join(trans)}（缩写猜测，结合语境判断）")

    query_vec = None
    if not found and not abbr and 4 <= len(current) <= 200:
        query_vec = await embed.embed_query(current)
        if query_vec is not None:
            for meme, _ in await embed.search(
                query_vec, group_id, top_k=2, threshold=config.MEME_SEMANTIC_THRESHOLD
            ):
                if meme.term not in annotated:
                    lines.append(_hint_line(meme, "（语义相近，未必相关）"))

    hints = _block(lines[:_MAX_HINTS])
    suggestion = await _pick_suggestion(group_id, found, annotated, peers, own, query_vec)
    return hints, suggestion


async def _pick_suggestion(
    group_id: int,
    found: list[Meme],
    annotated: set[str],
    peers: list[str],
    own: list[str],
    query_vec,
) -> str:
    every = config.MEME_SUGGEST_EVERY
    if every <= 0:
        return ""
    count = _since_suggest.get(group_id, 0) + 1
    _since_suggest[group_id] = count
    if count < every:
        return ""
    own_text = "\n".join(own)

    def usable(meme: Meme | None) -> bool:
        return meme is not None and not meme.offense and meme.term not in own_text

    # 优先：本群群友最近用过的梗（当前消息 > history 中由近及远）
    recent_terms: list[str] = [m.term for m in found]
    for text in reversed(peers):
        recent_terms.extend(m.group(1) for m in _TERM_MARK.finditer(text))
    candidate: Meme | None = None
    for term in recent_terms:
        meme = resolve(term, group_id)
        if usable(meme):
            candidate = meme
            break
    # 其次：与当前话题语义相近的梗
    if candidate is None and query_vec is not None:
        for meme, _ in await embed.search(
            query_vec, group_id, top_k=5, threshold=config.MEME_SEMANTIC_THRESHOLD
        ):
            if usable(meme):
                candidate = meme
                break
    if candidate is None:
        return ""
    _since_suggest[group_id] = 0
    return f"{SUGGEST_HEADER}\n「{candidate.term}」{candidate.short_meaning()}"
