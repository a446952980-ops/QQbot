"""消息中的梗识别：词条精确匹配 + 拼音首字母缩写（nbnhhsh）。"""

from __future__ import annotations

import re

import httpx

from . import store
from .store import Meme

# 两字词条大量与日常用词重合（给力、套路、钓鱼……），仅这些类型或群内词条才按两字匹配
_SHORT_TERM_TYPES = frozenset({"谐音", "缩写"})
_ASCII_WORD = re.compile(r"[a-z0-9]")

_NBNHHSH_URL = "https://lab.magiconch.com/api/nbnhhsh/guess"
_ABBR_RE = re.compile(r"(?<![A-Za-z])[A-Za-z]{2,6}(?![A-Za-z])")
_abbr_cache: dict[str, tuple[str, ...]] = {}
_ABBR_CACHE_MAX = 2000


def _short_term_allowed(meme: Meme) -> bool:
    return meme.scope != store.GLOBAL_SCOPE or meme.type in _SHORT_TERM_TYPES


def find_memes(text: str, group_id: int | None) -> list[Meme]:
    """最长优先、不重叠地匹配该群可见词条，按出现顺序返回（去重）。"""
    if not text:
        return []
    lowered = text.lower()
    max_len = min(store.max_term_len(), len(lowered))
    found: list[Meme] = []
    seen: set[int] = set()
    i = 0
    while i < len(lowered):
        matched = 0
        for length in range(min(max_len, len(lowered) - i), 1, -1):
            piece = lowered[i : i + length]
            meme = store.resolve(piece, group_id)
            if meme is None:
                continue
            if length <= 2 and not _short_term_allowed(meme):
                continue
            # 字母/数字开头或结尾的词条要求词边界，避免 "abc" 命中 "abcd"
            if _ASCII_WORD.match(piece[0]) and i > 0 and _ASCII_WORD.match(lowered[i - 1]):
                continue
            end = i + length
            if _ASCII_WORD.match(piece[-1]) and end < len(lowered) and _ASCII_WORD.match(lowered[end]):
                continue
            if meme.id not in seen:
                seen.add(meme.id)
                found.append(meme)
            matched = length
            break
        i += matched or 1
    return found


def _looks_like_abbr(word: str) -> bool:
    """拼音首字母缩写多为辅音串；排除含较多元音的英文单词与单字母重复（hhh）。"""
    lowered = word.lower()
    if len(set(lowered)) == 1:
        return False
    vowels = sum(ch in "aeiou" for ch in lowered)
    return vowels / len(lowered) <= 0.25


def abbreviation_candidates(text: str, group_id: int | None) -> list[str]:
    words: list[str] = []
    for match in _ABBR_RE.finditer(text or ""):
        word = match.group(0).lower()
        if word in words or store.resolve(word, group_id) is not None:
            continue
        if _looks_like_abbr(word):
            words.append(word)
    return words[:3]


async def guess_abbreviations(words: list[str], *, timeout: float = 3.0) -> dict[str, tuple[str, ...]]:
    """查询 nbnhhsh（能不能好好说话），返回 {缩写: 前几个候选}；失败时返回已缓存部分。"""
    result = {w: _abbr_cache[w] for w in words if _abbr_cache.get(w)}
    pending = [w for w in words if w not in _abbr_cache]
    if not pending:
        return result
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(_NBNHHSH_URL, json={"text": ",".join(pending)})
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        print(f"[meme] nbnhhsh 查询失败: {e}")
        return result
    for item in data if isinstance(data, list) else []:
        name = str(item.get("name") or "").lower()
        trans = tuple(str(t) for t in (item.get("trans") or [])[:3])
        if name not in pending:
            continue
        if len(_abbr_cache) >= _ABBR_CACHE_MAX:
            _abbr_cache.clear()
        _abbr_cache[name] = trans
        if trans:
            result[name] = trans
    return result
