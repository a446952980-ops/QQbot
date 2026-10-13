"""Per-group @ turns and recent non-@ group context; persisted under project ``data/``."""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict, deque
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
    for key, value in raw.items():
        try:
            gid = int(key)
        except (TypeError, ValueError):
            continue
        raw_turns = value.get("turns", []) if isinstance(value, dict) else value
        raw_background = value.get("background", []) if isinstance(value, dict) else []
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
    _turn_store.update(loaded_turns)
    _background_store.update(loaded_background)


def _save_persisted() -> None:
    data_dir, path = _persist_paths()
    data_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, dict[str, list[list[object]]]] = {}
    for gid in set(_turn_store) | set(_background_store):
        turns = _turn_store.get(gid)
        background = _background_store.get(gid)
        if turns or background:
            payload[str(gid)] = {
                "turns": [[timestamp, user, assistant] for timestamp, user, assistant in turns or ()],
                "background": [[timestamp, content] for timestamp, content in background or ()],
            }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


_load_persisted()


def history_as_messages(group_id: int) -> list[HumanMessage | AIMessage]:
    """Replay @ turns and up to 10 non-@ group messages in event-time order."""
    with _lock:
        turns = _turn_store.get(group_id)
        background = _background_store.get(group_id)
        if not turns and not background:
            return []
        events: list[tuple[int, int, BaseMessage]] = []
        for timestamp, content in background or ():
            events.append((timestamp, 0, HumanMessage(content=content)))
        for timestamp, user_content, assistant_content in turns or ():
            events.append((timestamp, 1, HumanMessage(content=user_content)))
            events.append(
                (timestamp, 2, AIMessage(content=_clean_cached_assistant_content(assistant_content)))
            )
        return [message for _, _, message in sorted(events, key=lambda item: item[:2])]


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
