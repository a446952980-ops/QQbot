"""Per-group @ turns and recent non-@ group context; persisted under project ``data/``."""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

MAX_TURNS = 8
MAX_BACKGROUND_MESSAGES = 10

_PLAIN_POST_TOOLS = frozenset(
    {"qq_send_group_msg", "qq_send_group_ai_record", "qq_send_group_emotion_face"}
)

_DSML_TOOL_CALLS_RE = re.compile(
    r"<｜DSML｜tool_calls>.*?</｜DSML｜tool_calls>", re.DOTALL
)

_lock = threading.Lock()
_turn_store: dict[int, deque[tuple[int, str, str]]] = defaultdict(
    lambda: deque(maxlen=MAX_TURNS)
)
_background_store: dict[int, deque[tuple[int, str]]] = defaultdict(
    lambda: deque(maxlen=MAX_BACKGROUND_MESSAGES)
)
# 已压缩的历史摘要（每群一条，代表该群更早的对话记忆）
_summary_store: dict[int, str] = {}
# 正在后台压缩中的群，避免重复触发
_compressing: set[int] = set()
# 每群已展开写入上下文（background / @ 轮次）的合并转发：id -> 渲染块文本。
# 当前消息再次引用同一转发时，历史中的旧展开副本仅在本次模型调用渲染时
# 临时替换为占位符（存储不动），避免同一段转发内容在上下文里重复出现。
# dict 充当有序集合，超限丢最早记录。
_forward_blocks: dict[int, dict[str, str]] = {}
_FORWARD_SEEN_KEEP = 200
_FORWARD_OMIT_PLACEHOLDER = "【这是你收到的最新一次消息引用的转发内容，已省略】"


def mark_forward_blocks(group_id: int, blocks: dict[str, str]) -> None:
    """登记已展开写入该群上下文的合并转发块（内存态，随上下文清空而清空）。"""
    try:
        gid = int(group_id)
    except (TypeError, ValueError):
        return
    clean = {fid: text for fid, text in blocks.items() if fid and text}
    if not clean:
        return
    known = _forward_blocks.setdefault(gid, {})
    known.update(clean)
    if len(known) > _FORWARD_SEEN_KEEP:
        for fid in list(known)[: len(known) - _FORWARD_SEEN_KEEP // 2]:
            known.pop(fid, None)


def _persist_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent.parent.parent
    data_dir = root / "data"
    return data_dir, data_dir / "conversation_history.json"


def _load_persisted() -> None:
    _, path = _persist_paths()
    if not path.is_file():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(raw, dict):
        return
    loaded_turns: dict[int, deque[tuple[int, str, str]]] = {}
    loaded_background: dict[int, deque[tuple[int, str]]] = {}
    loaded_summary: dict[int, str] = {}
    for key, value in raw.items():
        try:
            gid = int(key)
        except (TypeError, ValueError):
            continue
        raw_turns = value.get("turns", []) if isinstance(value, dict) else value
        raw_background = value.get("background", []) if isinstance(value, dict) else []
        raw_summary = value.get("summary", "") if isinstance(value, dict) else ""
        if not isinstance(raw_turns, list):
            continue
        turns: list[tuple[int, str, str]] = []
        for index, item in enumerate(raw_turns):
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                timestamp, user, assistant = item[0], item[1], item[2]
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                timestamp, user, assistant = index, item[0], item[1]
            else:
                continue
            if isinstance(timestamp, int | float) and isinstance(user, str) and isinstance(assistant, str):
                turns.append((int(timestamp), user, assistant))
        if turns:
            loaded_turns[gid] = deque(turns[-MAX_TURNS:], maxlen=MAX_TURNS)
        if isinstance(raw_background, list):
            background: list[tuple[int, str]] = []
            for item in raw_background:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    timestamp, content = item[0], item[1]
                    if isinstance(timestamp, int | float) and isinstance(content, str):
                        background.append((int(timestamp), content))
            if background:
                loaded_background[gid] = deque(
                    background[-MAX_BACKGROUND_MESSAGES:], maxlen=MAX_BACKGROUND_MESSAGES
                )
        if isinstance(raw_summary, str) and raw_summary.strip():
            loaded_summary[gid] = raw_summary.strip()
    _turn_store.update(loaded_turns)
    _background_store.update(loaded_background)
    _summary_store.update(loaded_summary)


def _save_persisted() -> None:
    data_dir, path = _persist_paths()
    data_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict] = {}
    for gid in set(_turn_store) | set(_background_store) | set(_summary_store):
        turns = _turn_store.get(gid)
        background = _background_store.get(gid)
        summary = _summary_store.get(gid)
        if not (turns or background or summary):
            continue
        entry: dict = {
            "turns": [[timestamp, user, assistant] for timestamp, user, assistant in turns or ()],
            "background": [[timestamp, content] for timestamp, content in background or ()],
        }
        if summary:
            entry["summary"] = summary
        payload[str(gid)] = entry
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


_load_persisted()


def _substitute_forward_blocks(message: BaseMessage, blocks: list[str]) -> BaseMessage:
    """把消息文本中已登记的转发展开块替换为占位符；未命中原样返回。"""
    content = message.content
    if not isinstance(content, str):
        return message
    text = content
    for block in blocks:
        if block in text:
            text = text.replace(block, _FORWARD_OMIT_PLACEHOLDER)
    if text == content:
        return message
    if isinstance(message, AIMessage):
        return AIMessage(content=text)
    return HumanMessage(content=text)


def history_as_messages(
    group_id: int, *, omit_forward_ids: frozenset[str] = frozenset()
) -> list[HumanMessage | AIMessage]:
    """Replay @ turns and up to 10 non-@ group messages in event-time order.

    omit_forward_ids：当前消息新引用的转发 id。这些转发此前写入上下文的
    展开副本在本次渲染中临时替换为占位符（当前消息内保留完整展开）。
    """
    with _lock:
        turns = _turn_store.get(group_id)
        background = _background_store.get(group_id)
        summary = _summary_store.get(group_id)
        if not turns and not background and not summary:
            return []
        omit_blocks: list[str] = []
        if omit_forward_ids:
            known = _forward_blocks.get(group_id, {})
            omit_blocks = [known[fid] for fid in omit_forward_ids if fid in known]
        events: list[tuple[int, int, BaseMessage]] = []
        for timestamp, content in background or ():
            events.append((timestamp, 0, HumanMessage(content=content)))
        for timestamp, user_content, assistant_content in turns or ():
            events.append((timestamp, 1, HumanMessage(content=user_content)))
            events.append(
                (timestamp, 2, AIMessage(content=_clean_cached_assistant_content(assistant_content)))
            )
        messages = [message for _, _, message in sorted(events, key=lambda item: item[:2])]
        if omit_blocks:
            messages = [_substitute_forward_blocks(msg, omit_blocks) for msg in messages]
        if summary:
            messages = [HumanMessage(content=f"【更早的群聊记忆摘要】\n{summary}")] + messages
        return messages


def _clean_cached_assistant_content(content: str) -> str:
    """Strip internal transcript labels from older persisted assistant turns."""
    content = _DSML_TOOL_CALLS_RE.sub("", content)
    lines: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(("[Same turn —", "[同回合")):
            continue
        if stripped.startswith("[emotion_face]"):
            continue
        if stripped.startswith("[plain]"):
            line = line.replace("[plain]", "", 1).lstrip()
        elif stripped.startswith("[voice]"):
            line = line.replace("[voice]", "", 1).lstrip()
        lines.append(line)
    return "\n".join(lines).strip()


def append_turn(
    group_id: int, timestamp: int, user_content: str, assistant_combined: str
) -> None:
    with _lock:
        _turn_store[group_id].append((timestamp, user_content, assistant_combined))
        try:
            _save_persisted()
        except OSError:
            pass


def clear_group_turns(group_id: int) -> None:
    """清空该群的 @ 轮次上下文，保留 background（供 mode 切换类指令使用）。"""
    with _lock:
        _forward_blocks.pop(int(group_id), None)  # 转发内容可能原在轮次里，清掉防止占位符遮蔽
        turns = _turn_store.get(group_id)
        if turns:
            turns.clear()
            try:
                _save_persisted()
            except OSError:
                pass


def clear_group_all(group_id: int) -> None:
    """清空该群全部上下文（turns + background）。"""
    with _lock:
        _forward_blocks.pop(int(group_id), None)
        turns = _turn_store.get(group_id)
        background = _background_store.get(group_id)
        if not turns and not background:
            return
        if turns:
            turns.clear()
        if background:
            background.clear()
        try:
            _save_persisted()
        except OSError:
            pass


def append_background_message(group_id: int, timestamp: int, content: str) -> None:
    """Store a non-@ group message as model context without treating it as a command."""
    with _lock:
        _background_store[group_id].append((timestamp, content))
        try:
            _save_persisted()
        except OSError:
            pass


def _tool_call_name(tc: object) -> str:
    if isinstance(tc, dict):
        return str(tc.get("name") or "")
    return str(getattr(tc, "name", None) or "")


def _tool_call_args(tc: object) -> dict:
    raw: object
    if isinstance(tc, dict):
        raw = tc.get("args")
    else:
        raw = getattr(tc, "args", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def extract_plain_tool_texts_from_messages(messages: list[BaseMessage]) -> list[str]:
    """Collect spoken text sent to the group via message tools, in chronological order."""
    lines: list[str] = []
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        for tc in getattr(msg, "tool_calls", None) or []:
            name = _tool_call_name(tc)
            if name not in _PLAIN_POST_TOOLS:
                continue
            args = _tool_call_args(tc)
            if name == "qq_send_group_emotion_face":
                continue
            text = args.get("text")
            if isinstance(text, str) and text.strip():
                lines.append(text.strip())
    return lines


def strip_markdown_for_qq(text: str) -> str:
    """Remove common Markdown so final replies are plain text in QQ."""
    if not text or not text.strip():
        return text
    t = _DSML_TOOL_CALLS_RE.sub("", text)
    t = re.sub(r"```[\s\S]*?```", "", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)
    t = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"~~(.+?)~~", r"\1", t)
    while "**" in t:
        nt = re.sub(r"\*\*(.+?)\*\*", r"\1", t, flags=re.DOTALL)
        if nt == t:
            t = t.replace("**", "")
            break
        t = nt
    while "__" in t:
        nt = re.sub(r"__(.+?)__", r"\1", t, flags=re.DOTALL)
        if nt == t:
            t = t.replace("__", "")
            break
        t = nt
    t = re.sub(r"(?m)^#{1,6}\s+", "", t)
    t = re.sub(r"(?m)^\s*>\s?", "", t)
    t = re.sub(r"(?m)^\s*(\*{3,}|-{3,}|_{3,})\s*$", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_final_assistant_text(messages: list[BaseMessage]) -> str:
    """Last AIMessage with no tool calls; used as the reply body for the @ message."""
    raw = ""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        content = msg.content
        if content is None:
            continue
        if isinstance(content, str):
            if content.strip():
                raw = content
                break
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            joined = "".join(parts).strip()
            if joined:
                raw = joined
                break
        else:
            s = str(content).strip()
            if s:
                raw = s
                break
    return strip_markdown_for_qq(raw.strip()) if raw else ""


def build_assistant_combined_for_cache(messages: list[BaseMessage]) -> str:
    """
    One assistant string for this turn (stored and replayed as a single AIMessage).

    Concatenates, in chronological order: text from `qq_send_group_msg` /
    `qq_send_group_ai_record` args and emotion-face markers from tool_calls across
    the turn, then the last no-tool assistant text (the @-reply body). Matches
    how the agent is instructed to speak in bursts then end with a short reply
    tail — replay sees the full virtual utterance in one block.
    """
    plain_lines = extract_plain_tool_texts_from_messages(messages)
    final = extract_final_assistant_text(messages).strip()
    parts = list(plain_lines)
    if final:
        parts.append(final)
    return "\n\n".join(parts) if parts else "(no assistant output)"


# ---- 对话压缩（后台 summarize 旧轮次，释放上下文，同时保留长期记忆） ----

def get_summary(group_id: int) -> str | None:
    with _lock:
        return _summary_store.get(group_id)


def _render_transcript(
    background: list[tuple[int, str]],
    turns: list[tuple[int, str, str]],
) -> str:
    """把 background 与 @ turns 按事件时间合并排序，渲染成可读的时序文本，供压缩模型使用。"""
    events: list[tuple[int, int, str]] = []
    for ts, content in background:
        events.append((ts, 0, f"[群聊（未@我）] {content}"))
    for ts, user, assistant in turns:
        events.append((ts, 1, f"[@我的消息] {user}"))
        events.append((ts, 2, f"[我的回复] {_clean_cached_assistant_content(assistant)}"))
    events.sort(key=lambda item: (item[0], item[1]))
    lines: list[str] = []
    for idx, (ts, _, text) in enumerate(events, 1):
        try:
            stamp = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
        except (ValueError, OSError, OverflowError):
            stamp = str(ts)
        lines.append(f"[{idx} · {stamp}] {text}")
    return "\n\n".join(lines)


def try_begin_compression(group_id: int, *, force: bool = False) -> dict | None:
    """当某群 @ 轮次达到上限（或 force=True）且当前不在压缩中时，原子地标记压缩中并返回
    一份时间一致的快照（含原始 turns/background 引用与渲染好的转录文本）。"""
    with _lock:
        if group_id in _compressing:
            return None
        turns = list(_turn_store.get(group_id) or ())
        background = list(_background_store.get(group_id) or ())
        prev_summary = _summary_store.get(group_id)
        if not (turns or background or prev_summary):
            return None
        if not force and len(turns) < MAX_TURNS:
            return None
        _compressing.add(group_id)
    return {
        "turns": turns,
        "background": background,
        "summary": prev_summary,
        "transcript": _render_transcript(background, turns),
    }


def finish_compression(group_id: int, summary: str, snapshot: dict) -> None:
    """压缩完成后回写：把新摘要存入 summary_store，并仅清除快照里已有的 turns/background
    （压缩期间新增的条目按对象身份保留），最后落盘。"""
    with _lock:
        _compressing.discard(group_id)
        if summary:
            _summary_store[group_id] = summary
        current_turns = _turn_store.get(group_id)
        if current_turns is not None:
            old_ids = {id(t) for t in snapshot.get("turns") or ()}
            _turn_store[group_id] = deque(
                (t for t in current_turns if id(t) not in old_ids), maxlen=MAX_TURNS
            )
        current_bg = _background_store.get(group_id)
        if current_bg is not None:
            old_ids = {id(b) for b in snapshot.get("background") or ()}
            _background_store[group_id] = deque(
                (b for b in current_bg if id(b) not in old_ids),
                maxlen=MAX_BACKGROUND_MESSAGES,
            )
        try:
            _save_persisted()
        except OSError:
            pass


def abort_compression(group_id: int) -> None:
    """压缩失败时解除标记，保留原上下文，等待下次达到上限再重试。"""
    with _lock:
        _compressing.discard(group_id)
