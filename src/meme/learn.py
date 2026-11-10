"""群聊黑话/新梗自动学习（参考 SuInk/Diana PR #1269 的流程）。

1. 收集：每群攒够一批消息（默认 20 条，或 ≥8 条且距首条超过 15 分钟），用便宜模型
   批量标出“看不懂、疑似有梗义”的说法——重点是谐音、在上下文里突兀的词、单独成句的短语；
2. 计数：候选按群记录次数、发言人、最近 3 句原话；
3. 查证：谐音/突兀/单独成句类出现 2 次即查，其余需 ≥2 人且 ≥3 次；
   群友 @ 机器人问“X 是什么梗/啥意思”时立即查。查证 = 联网搜索一次 + 不带工具的模型
   对照原话与搜索结果判断；
4. 入库：公开流行的梗写入公共词库，只在本群流通的黑话写入本群词库；
   未查实则门槛翻倍（最多到 48 次）。人工 /meme 添加或屏蔽的词条不会被覆盖。
所有步骤只读写词库，不发送任何群消息。
"""

from __future__ import annotations

import asyncio
import json
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage

import config

from . import matcher, store

_BATCH_SIZE = 20
_BATCH_MIN = 8
_BATCH_MAX_WAIT = 15 * 60
_MAX_NEED = 48
_QUICK_KINDS = frozenset({"谐音", "突兀", "单独成句"})

_EXTRACT_PROMPT = """你在帮一个 QQ 群机器人发现它可能看不懂的网络梗和群内黑话。
下面是一批按编号排列的群聊消息（发言人用代号表示）。请标出其中所有网络梗、流行语、群内黑话——即使你自己知道它的意思也要标，因为要把它收进词库；以及“字面意思解释不通、疑似另有梗义”的说法。

重点关注：
1. 谐音：用字奇怪、字面不通，但读起来像别的词、英文或方言（如“栓Q”“蚌埠住了”“尊嘟假嘟”“蓝瘦香菇”）；
2. 突兀：放在这句话的上下文里接不上、显得很突兀的词或短语；
3. 单独成句：整条消息就是一个短词/短语，单独看不明白在说什么（如单独一句“典”“急了”“包的”“绷”）；
4. 拼音或英文首字母缩写（如“xswl”“yyds”）。

不要标：人名、昵称、称呼；没有梗义的日常常用词；没有梗义的错别字；通用技术名词、产品型号；明显只是一次性的玩笑。

要求：term 必须原样出现在对应编号的消息里；长度 1～8 个字（只有“单独成句”才允许 1 个字）；每条消息最多 3 个，总数最多 10 个；拿不准是不是梗的宁可不标。
只输出 JSON 数组，不要其他文字，没有就输出 []：
[{"msg": 消息编号, "term": "原样说法", "kind": "谐音|突兀|单独成句|缩写|黑话"}]"""

_VERIFY_PROMPT = """你是网络用语核查员。根据群聊原话和联网搜索结果，判断一个说法是不是网络梗/流行语/群内黑话，并给出通用释义。

规则：
- 只核实核心网络用语：如果它是某个梗的变体或复合说法，term 写核心梗本身（必须在原话中出现）。
- 释义只写这个说法的通用意思（不超过 40 字），不要夹带群里这几句话的具体情境。
- 谐音梗必须写清楚它谐的是哪个词/哪句话/哪个读音。
- 搜索结果能证实它是公开流行的网络用语 → is_meme=true，public=true，source 填最可靠的一条链接。
- 搜不到公开释义，但几句原话里用法一致、意思明确（群内黑话、简称）→ is_meme=true，public=false。
- 普通词、错别字、人名、一次性玩笑、或无法确定 → is_meme=false。宁可保守。
- aliases 填常见的其他写法（如谐音变体），没有就 []；offense 表示是否带辱骂、色情或歧视色彩。
只输出一个 JSON 对象，不要其他文字：
{"is_meme": true, "public": true, "term": "", "meaning": "", "aliases": [], "type": "谐音|缩写|黑话|引用|现象|俗语", "offense": false, "source": ""}"""

_HOT_STRICT_NOTE = """
（热榜模式，从严判断）只接受近几年在网上出圈、有明确梗义的网络流行语。以下一律 is_meme=false：
普通词汇、成语俗语、方言称呼、行业术语、新闻用语、商品或节目名，即使它们有百科释义。
term 必须与给定的词完全一致，不要改写或截取。"""

_ASK_RE = re.compile(
    r"([\w\u4e00-\u9fff]{1,12})\s*(?:是什么梗|是啥梗|什么梗|是什么意思|啥意思|什么意思)"
)
_NOISE = re.compile(r"\[[^\[\]]*\]|https?://\S+")

_buffers: dict[int, list[tuple[str, str]]] = {}
_buffer_started: dict[int, float] = {}
_running: set[str] = set()
_tasks: set[asyncio.Task] = set()
_llm = None


def _model():
    global _llm
    if _llm is None:
        from ..agent.init_llm import _build_chat_model  # 延迟导入，避免循环依赖

        _llm = _build_chat_model(config.DEFAULT_MODEL, thinking=False)
    return _llm


def _spawn(key: str, coro) -> None:
    """同一 key 同时只跑一个后台任务。"""
    if key in _running:
        coro.close()
        return
    _running.add(key)

    async def _run() -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[meme] 自动学习出错（{key}）: {e}")
        finally:
            _running.discard(key)

    task = asyncio.create_task(_run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def _clean(text: str) -> str:
    return " ".join(_NOISE.sub(" ", text or "").split())


def _is_known(term: str, group_id: int) -> bool:
    """词库已有（含本群屏蔽标记，人工结果优先）即视为已处理。"""
    return store.is_handled(term, group_id)


_ASK_PREFIXES = (
    "请问一下", "请问", "问一下", "问下", "有人知道", "有谁知道", "谁知道", "你们知道",
    "你知道", "知道", "大家", "所以", "那个", "这个",
)


def _asked_term(raw: str) -> str:
    term = raw.strip()
    changed = True
    while changed:
        changed = False
        for prefix in _ASK_PREFIXES:
            if term.startswith(prefix) and len(term) > len(prefix):
                term = term[len(prefix):]
                changed = True
    return re.sub(r"是$", "", term)


def observe(group_id: int, speaker: str, text: str, *, at_bot: bool = False) -> None:
    """记录一条群友消息；攒够一批后后台提取候选。@ 机器人问梗时立即查证。"""
    if not config.MEME_LEARN_ENABLED:
        return
    text = _clean(text)
    if not text or text.startswith("/"):
        return
    group_id = int(group_id)
    if at_bot:
        for m in _ASK_RE.finditer(text):
            term = _asked_term(m.group(1))
            if term and not _is_known(term, group_id) and not matcher.find_matches(term, group_id):
                store.observe_candidate(
                    group_id, term, kind="询问", speaker=speaker, sample=text,
                    need_count=1, need_speakers=1,
                )
                _spawn(f"verify:{group_id}:{term}", verify(group_id, term))
    buffer = _buffers.setdefault(group_id, [])
    if not buffer:
        _buffer_started[group_id] = time.time()
    buffer.append((speaker, text[:200]))
    waited = time.time() - _buffer_started.get(group_id, time.time())
    if len(buffer) >= _BATCH_SIZE or (len(buffer) >= _BATCH_MIN and waited >= _BATCH_MAX_WAIT):
        batch = list(buffer)
        buffer.clear()
        _spawn(f"extract:{group_id}:{time.time()}", _extract(group_id, batch))


def _parse_json(text: str, opener: str):
    closer = "]" if opener == "[" else "}"
    start, end = text.find(opener), text.rfind(closer)
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except ValueError:
        return None


def _content(message) -> str:
    content = message.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content or "")


async def _extract(group_id: int, batch: list[tuple[str, str]]) -> None:
    if not store.take_daily_quota("meme_extract", config.MEME_EXTRACT_DAILY):
        return
    codes: dict[str, str] = {}
    lines = []
    for i, (speaker, text) in enumerate(batch, 1):
        code = codes.setdefault(speaker, f"群友{chr(ord('A') + len(codes) % 26)}")
        lines.append(f"[{i}] {code}: {text}")
    response = await _model().ainvoke(
        [SystemMessage(content=_EXTRACT_PROMPT), HumanMessage(content="\n".join(lines))]
    )
    items = _parse_json(_content(response), "[")
    if not isinstance(items, list):
        return
    for item in items[:10]:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("msg")) - 1
        except (TypeError, ValueError):
            continue
        term = " ".join(str(item.get("term") or "").split())
        kind = str(item.get("kind") or "黑话")
        if not 0 <= idx < len(batch) or not term:
            continue
        speaker, text = batch[idx]
        if store.normalize(term) not in store.normalize(text):
            continue  # 必须原样出现
        standalone = store.normalize(term) == store.normalize(re.sub(r"[\s\W_]+", "", text))
        if len(term) > 8 or (len(term) < 2 and not standalone):
            continue
        if standalone:
            kind = "单独成句"
        if _is_known(term, group_id) or matcher.find_matches(term, group_id):
            continue
        quick = kind in _QUICK_KINDS
        cand = store.observe_candidate(
            group_id,
            term,
            kind=kind,
            speaker=speaker,
            sample=text,
            need_count=2 if quick else 3,
            need_speakers=1 if quick else 2,
        )
        if (
            cand["status"] == "pending"
            and cand["count"] >= cand["need_count"]
            and len(cand["speakers"]) >= cand["need_speakers"]
        ):
            _spawn(f"verify:{group_id}:{term}", verify(group_id, term))


async def _search(term: str, kind: str) -> list[dict]:
    from ..agent.tools import _serper_post  # 延迟导入，避免循环依赖

    query = f"{term} {'谐音梗' if kind == '谐音' else '梗'} 是什么意思"
    data = await _serper_post(
        "https://google.serper.dev/search",
        {"q": query, "gl": "cn", "hl": "zh-cn", "num": 8},
    )
    return [
        {"title": r.get("title", ""), "snippet": r.get("snippet", ""), "link": r.get("link", "")}
        for r in (data.get("organic") or [])[:8]
    ]


async def verify(
    group_id: int,
    term: str,
    *,
    public_only: bool = False,
    quota: str = "meme_verify",
    quota_limit: int | None = None,
) -> str:
    """查证一个候选，返回结果状态（learned / pending / dead / known / skipped）。

    public_only=True（热榜来源）时只接受能证实为公开流行的梗，写入公共词库。
    """
    cand = store.get_candidate(group_id, term)
    if cand is None or cand["status"] != "pending":
        return "skipped"
    if _is_known(term, group_id):
        cand["status"] = "known"
        store.save_candidate(cand)
        return "skipped"
    if not config.SERPER_API_KEY:
        return "skipped"
    if not store.take_daily_quota(
        quota, config.MEME_VERIFY_DAILY if quota_limit is None else quota_limit
    ):
        return "skipped"  # 保持 pending，下次出现时再尝试
    results = await _search(term, cand.get("kind") or "")
    body = {
        "term": term,
        "类型线索": cand.get("kind"),
        "群里原话": cand["samples"],
        "搜索结果": results,
    }
    response = await _model().ainvoke(
        [
            SystemMessage(content=_VERIFY_PROMPT + (_HOT_STRICT_NOTE if public_only else "")),
            HumanMessage(content=json.dumps(body, ensure_ascii=False)),
        ]
    )
    verdict = _parse_json(_content(response), "{")
    learned = None
    if (
        isinstance(verdict, dict)
        and verdict.get("is_meme")
        and verdict.get("meaning")
        and (verdict.get("public") or not public_only)
    ):
        core = " ".join(str(verdict.get("term") or term).split())
        if public_only or not any(
            store.normalize(core) in store.normalize(s) for s in cand["samples"]
        ):
            core = term
        # 查证期间若有人工添加或屏蔽，以人工结果为准
        if not _is_known(core, group_id):
            public = bool(verdict.get("public"))
            source = str(verdict.get("source") or "")
            learned = store.add_learned_meme(
                store.GLOBAL_SCOPE if public else str(group_id),
                core,
                str(verdict["meaning"])[:80],
                aliases=[str(a) for a in verdict.get("aliases") or [] if isinstance(a, str)],
                type_=str(verdict.get("type") or "") or None,
                offense=bool(verdict.get("offense")),
                origin=source or None,
                examples=[] if public else cand["samples"],
                detail=(
                    f"{verdict['meaning']} | 群 {group_id}，{len(cand['speakers'])} 人 "
                    f"{cand['count']} 次 | {source or '群内用法'}"
                ),
            )
    if learned is not None:
        cand["status"] = "learned"
    else:
        cand["need_count"] = cand["count"] * 2 if cand["count"] else 2
        cand["status"] = "dead" if cand["need_count"] > _MAX_NEED else "pending"
    store.save_candidate(cand)
    print(f"[meme] 查证「{term}」→ {cand['status']}")
    return cand["status"]
