"""热榜补充新梗：每天拉一次 B站、百度热搜，让模型挑出其中的网络梗/流行语，
再走与群内学习相同的查证流程（只接受公开流行的梗，写入公共词库）。不发送任何消息。"""

from __future__ import annotations

import asyncio
from datetime import datetime

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

import config

from . import learn, matcher, store

_HOT_GROUP_ID = 0  # 热榜候选在候选表中的占位群号
_RUN_AFTER_HOUR = 10
_CHECK_INTERVAL = 600
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

_PICK_PROMPT = """下面是今天的热搜标题。请从中挑出“网络梗、流行语、新造的网络说法”（例如谐音梗、缩写、新句式、出圈台词），
不要挑新闻事件本身、人名、作品名、品牌名、赛事名。term 必须原样出现在某条标题里，长度 2～10 字。
最多挑 10 个，拿不准的不挑，没有就输出 []。
只输出 JSON 数组：[{"term": "原样说法", "title": "出现的标题"}]"""


def _walk_words(node, out: list[str]) -> None:
    if isinstance(node, dict):
        for key in ("keyword", "show_name", "word", "query"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value.strip())
                break
        for value in node.values():
            _walk_words(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_words(value, out)


async def fetch_hot_titles() -> list[str]:
    """B站热搜 + 百度实时热搜标题（去重）；单个源失败不影响其他源。"""
    sources = (
        "https://s.search.bilibili.com/main/hotword?limit=50",
        "https://api.bilibili.com/x/web-interface/wbi/search/square?limit=50",
        "https://top.baidu.com/api/board?platform=wise&tab=realtime",
    )
    titles: list[str] = []
    async with httpx.AsyncClient(timeout=10, headers={"User-Agent": _UA}) as client:
        for url in sources:
            try:
                response = await client.get(url)
                response.raise_for_status()
                _walk_words(response.json(), titles)
            except Exception as e:
                print(f"[meme] 热榜获取失败 {url}: {e}")
    seen: set[str] = set()
    return [t for t in titles if not (t in seen or seen.add(t))][:120]


async def run_once() -> list[str]:
    """拉热榜→挑梗→查证，返回本次新学到的词。"""
    titles = await fetch_hot_titles()
    if not titles:
        return []
    response = await learn._model().ainvoke(
        [
            SystemMessage(content=_PICK_PROMPT),
            HumanMessage(content="\n".join(f"- {t}" for t in titles)),
        ]
    )
    items = learn._parse_json(learn._content(response), "[")
    learned: list[str] = []
    for item in (items if isinstance(items, list) else [])[:10]:
        if not isinstance(item, dict):
            continue
        term = " ".join(str(item.get("term") or "").split())
        title = str(item.get("title") or "")
        if not 2 <= len(term) <= 10 or term not in title:
            continue
        if store.is_handled(term, _HOT_GROUP_ID) or matcher.find_matches(term, None):
            continue
        store.observe_candidate(
            _HOT_GROUP_ID, term, kind="热榜", speaker="hot", sample=title,
            need_count=1, need_speakers=1,
        )
        status = await learn.verify(
            _HOT_GROUP_ID,
            term,
            public_only=True,
            quota="meme_verify_hot",
            quota_limit=config.MEME_HOT_DAILY,
        )
        if status == "learned":
            learned.append(term)
    return learned


async def hot_loop() -> None:
    """每天 10 点后运行一次（重启当天不重复）。"""
    if not (config.MEME_LEARN_ENABLED and config.MEME_HOT_DAILY > 0):
        return
    while True:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        if now.hour >= _RUN_AFTER_HOUR and store.get_meta("hot_last_run") != today:
            store.set_meta("hot_last_run", today)
            try:
                learned = await run_once()
                print(f"[meme] 热榜新梗: {learned or '无'}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[meme] 热榜任务失败: {e}")
        await asyncio.sleep(_CHECK_INTERVAL)
