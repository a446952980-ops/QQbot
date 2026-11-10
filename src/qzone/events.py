"""说说事件类型：按当前时间与素材可用性加权抽取，并准备素材。"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import chinese_calendar

import config

from ..agent.conversation_history import known_group_ids, user_texts_since
from ..meme import hot as meme_hot
from ..meme import matcher as meme_matcher
from .posts import recent_posts

_REVIEW_MIN_SNIPPETS = 5
_REVIEW_MAX_SNIPPETS = 30
_REVIEW_WINDOW_SECONDS = 12 * 3600
_REVIEW_MAX_MEMES = 6
_HOT_MAX_TITLES = 40

_BASE_WEIGHTS = {"daily": 3.0, "group_review": 3.0, "hot_news": 2.0, "date_event": 3.0}

_NOISE = re.compile(r"\[[^\[\]]*\]|https?://\S+")
_SENDER = re.compile(r"发送者:\s*(.+?)\s*\(QQ")


@dataclass
class Event:
    kind: str
    title: str
    instruction: str
    material: str = ""


def _time_of_day(now: datetime) -> str:
    h = now.hour
    if h < 11:
        return "上午"
    if h < 14:
        return "中午"
    if h < 18:
        return "下午"
    if h < 21:
        return "傍晚到晚上"
    return "深夜"


def _date_facts(today: date) -> list[str]:
    facts: list[str] = []
    on_holiday, name = chinese_calendar.get_holiday_detail(today)
    if name:
        zh = next((h.chinese for h in chinese_calendar.Holiday if h.value == name), name)
        facts.append(f"今天是{zh}假期" if on_holiday else f"今天因{zh}调休要上班")
    for day, term in chinese_calendar.get_solar_terms(today - timedelta(days=1), today):
        facts.append(f"{'今天' if day == today else '昨天'}是节气{term}")
    if today.weekday() == 4:
        facts.append("今天周五")
    elif today.weekday() >= 5 and not name:
        facts.append("今天是周末")
    if today.day == 1:
        facts.append(f"今天是{today.month}月第一天")
    elif (today + timedelta(days=1)).day == 1:
        facts.append(f"今天是{today.month}月最后一天")
    return facts


def group_review_material(now_ts: int | None = None) -> tuple[str, set[str]]:
    """近 12 小时各群群友消息（匿名化：只留内容，不留发送者与群号），以及出现过的昵称。"""
    since = int(now_ts or time.time()) - _REVIEW_WINDOW_SECONDS
    snippets: list[tuple[int, str]] = []
    names: set[str] = set()
    from ..handler.commands import is_group_enabled  # 延迟导入，避免启动时循环依赖

    for gid in known_group_ids():
        if gid in config.GROUP_BLACKLIST or not is_group_enabled(gid):
            continue
        for text in user_texts_since(gid, since):
            for m in _SENDER.finditer(text):
                names.add(m.group(1).strip())
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("内容:"):
                    body = line[3:]
                elif line.startswith("- text:"):
                    body = line[7:]
                else:
                    continue
                body = " ".join(_NOISE.sub(" ", body).split())
                if len(body) >= 2:
                    snippets.append((gid, body[:80]))
    names.discard("")
    snippets = snippets[-_REVIEW_MAX_SNIPPETS:]
    if len(snippets) < _REVIEW_MIN_SNIPPETS:
        return "", names
    material = "\n".join(f"- {s}" for _, s in snippets)
    notes = _meme_notes(snippets)
    if notes:
        material += (
            "\n\n【素材里出现的梗（仅帮助你看懂这些聊天，不要以梗为主题，也不要解释梗）】\n"
            + "\n".join(notes)
        )
    return material, names


def _meme_notes(snippets: list[tuple[int, str]]) -> list[str]:
    """用梗词库识别素材中的梗（含谐音写法），附上简短释义。"""
    notes: list[str] = []
    seen: set[int] = set()
    for gid, text in snippets:
        try:
            matches = meme_matcher.find_matches(text, gid)
        except Exception as e:
            print(f"[qzone] 素材梗识别失败: {e}")
            return notes
        for m in matches:
            if m.meme.id in seen:
                continue
            seen.add(m.meme.id)
            line = f"「{m.meme.term}」{m.meme.short_meaning(60)}"
            if m.homophone:
                line += f"（素材里写作「{m.surface}」）"
            notes.append(line)
            if len(notes) >= _REVIEW_MAX_MEMES:
                return notes
    return notes


def _build(
    kind: str,
    now: datetime,
    review: str,
    facts: list[str],
    hot_titles: list[str] | None = None,
) -> Event:
    tod = _time_of_day(now)
    if kind == "daily":
        return Event(
            kind,
            "日常碎碎念",
            f"现在是{tod}。随手发一条自己此刻的日常状态或念头，"
            "可以从人设里的小习惯、喜好、懒散的一面出发，像真人随手发的朋友圈。",
        )
    if kind == "group_review":
        return Event(
            kind,
            "群聊回顾",
            "下面是最近群里聊过的内容（已匿名）。挑一个你真有感触的话题或瞬间，"
            "写成自己的感想；不要复述原话，不提群名、昵称或任何能认出是谁的信息。",
            review,
        )
    if kind == "hot_news" and hot_titles:
        return Event(
            kind,
            "热点新鲜事",
            "下面是今天 B站和百度的热搜。挑一件你真正感兴趣的（科技、AI、游戏、网络文化、"
            "生活趣闻优先，避开政治与社会敏感话题），用自己的口吻说一句看法或吐槽，"
            "不写成新闻摘要；需要了解来龙去脉时再用 search 查（time_range 用 day）。",
            "\n".join(f"- {t}" for t in hot_titles[:_HOT_MAX_TITLES]),
        )
    if kind == "hot_news":
        return Event(
            kind,
            "热点新鲜事",
            "先用 search 查今天的新鲜事（topic 用 news，time_range 用 day），"
            "挑一件你真正感兴趣的（科技、AI、网络文化、生活趣闻优先，避开政治与社会敏感话题），"
            "用自己的口吻说一句看法或吐槽，不写成新闻摘要。",
        )
    return Event(
        kind,
        "日期事件",
        f"现在是{tod}。今天的日期信息：{'；'.join(facts)}。"
        "围绕这个日期写一条自己的状态或感想，不要写成祝福模板。",
    )


async def pick_event(now: datetime | None = None) -> Event:
    """按权重抽取事件：不可用的事件排除；今天已用过的排除；最近两天用过的权重减半。"""
    now = now or datetime.now().astimezone()
    review, _ = group_review_material(int(now.timestamp()))
    facts = _date_facts(now.date())
    available = {"daily", "hot_news"}
    if review:
        available.add("group_review")
    if facts:
        available.add("date_event")

    today_start = datetime.combine(now.date(), datetime.min.time()).astimezone().timestamp()
    two_days_ago = today_start - 2 * 86400
    used_today: set[str] = set()
    used_recent: set[str] = set()
    for post in recent_posts(20):
        if post.get("source") != "auto":
            continue
        ts = post.get("time", 0)
        if ts >= today_start:
            used_today.add(post.get("event", ""))
        elif ts >= two_days_ago:
            used_recent.add(post.get("event", ""))

    candidates = [k for k in available if k not in used_today] or sorted(available)
    weights = [
        _BASE_WEIGHTS[k] * (0.5 if k in used_recent else 1.0) for k in candidates
    ]
    kind = random.choices(candidates, weights=weights, k=1)[0]
    hot_titles: list[str] = []
    if kind == "hot_news":
        try:
            hot_titles = await meme_hot.get_hot_titles()
        except Exception as e:
            print(f"[qzone] 热榜获取失败，改用搜索: {e}")
    return _build(kind, now, review, facts, hot_titles)
