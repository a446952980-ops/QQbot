"""消息中的梗识别：词条精确匹配 + 拼音首字母缩写（nbnhhsh）。"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx
from pypinyin import Style, lazy_pinyin

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


@dataclass(frozen=True)
class Match:
    meme: Meme
    surface: str  # 消息里的原样写法
    homophone: bool = False  # 读音相同、写法不同（谐音变体）


# ---- 谐音：模糊拼音索引 ----
_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)
_CJK = re.compile(r"[\u4e00-\u9fff]")
_MIN_PINYIN_LEN = 3  # 两字谐音误报太多，只做 3 字及以上
_FUZZY_INITIALS = (("zh", "z"), ("ch", "c"), ("sh", "s"), ("n", "l"))
_pinyin_index: dict[tuple[str, ...], list[Meme]] = {}
_pinyin_lengths: list[int] = []
_pinyin_version = -1


def _fuzzy_syllable(syllable: str) -> str:
    """平翘舌、n/l、前后鼻音视为相同（覆盖常见方言谐音）。"""
    for src, dst in _FUZZY_INITIALS:
        if syllable.startswith(src):
            syllable = dst + syllable[len(src):]
            break
    if syllable.endswith("ng") and len(syllable) > 2:
        syllable = syllable[:-1]
    return syllable


def _tokens(text: str) -> list[str]:
    """逐字转模糊拼音（非汉字按小写字符原样保留），与原文逐字对齐。"""
    return [
        _fuzzy_syllable(p)
        for p in lazy_pinyin(text, style=Style.NORMAL, errors=lambda x: list(x.lower()))
    ]


def _ensure_pinyin_index() -> None:
    global _pinyin_index, _pinyin_lengths, _pinyin_version
    if _pinyin_version == store.version():
        return
    index: dict[tuple[str, ...], list[Meme]] = {}
    for meme in store.all_active_memes():
        for surface in (meme.term, *meme.aliases):
            compact = _PUNCT.sub("", store.normalize(surface))
            if len(compact) < _MIN_PINYIN_LEN or len(_CJK.findall(compact)) < 2:
                continue
            key = tuple(_tokens(compact))
            if len(key) == len(compact):
                index.setdefault(key, []).append(meme)
    _pinyin_index = index
    _pinyin_lengths = sorted({len(k) for k in index}, reverse=True)
    _pinyin_version = store.version()


def _visible(memes: list[Meme], group_id: int | None) -> Meme | None:
    for meme in memes:
        resolved = store.resolve(meme.term, group_id)
        if resolved is not None and resolved.id == meme.id:
            return meme
    return None


def find_matches(text: str, group_id: int | None) -> list[Match]:
    """识别消息中的梗：整句即词条 > 最长优先精确匹配 > 谐音（模糊拼音）匹配。"""
    if not text:
        return []
    lowered = store.normalize(text)
    found: list[Match] = []
    seen: set[int] = set()

    def add(meme: Meme, surface: str, homophone: bool = False) -> None:
        if meme.id not in seen:
            seen.add(meme.id)
            found.append(Match(meme, surface, homophone))

    # 1) 整条消息就是一个词条（单独出现的短梗，如“典”“急了”），不受两字限制
    whole = _PUNCT.sub("", lowered)
    if whole and len(whole) <= 8:
        meme = store.resolve(whole, group_id)
        if meme is not None:
            add(meme, whole)
            return found

    # 2) 精确匹配（含别名），最长优先、不重叠
    max_len = min(store.max_term_len(), len(lowered))
    covered = [False] * len(lowered)
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
            add(meme, piece)
            for k in range(i, end):
                covered[k] = True
            matched = length
            break
        i += matched or 1

    # 3) 谐音：在未被精确命中的连续片段上按模糊拼音匹配
    _ensure_pinyin_index()
    if _pinyin_index:
        for segment in re.split(r"[\s\W_]+", "".join(
            ch if not covered[k] else " " for k, ch in enumerate(lowered)
        )):
            if len(segment) < _MIN_PINYIN_LEN:
                continue
            tokens = _tokens(segment)
            if len(tokens) != len(segment):
                continue
            j = 0
            while j < len(segment):
                step = 1
                for length in _pinyin_lengths:
                    if j + length > len(segment):
                        continue
                    memes = _pinyin_index.get(tuple(tokens[j : j + length]))
                    if not memes:
                        continue
                    meme = _visible(memes, group_id)
                    if meme is None:
                        continue
                    surface = segment[j : j + length]
                    add(meme, surface, homophone=surface != store.normalize(meme.term))
                    step = length
                    break
                j += step
    return found


def find_memes(text: str, group_id: int | None) -> list[Meme]:
    return [m.meme for m in find_matches(text, group_id)]


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
