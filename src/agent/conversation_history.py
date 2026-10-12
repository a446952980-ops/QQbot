"""Per-group last N turns (user + combined assistant); persisted under project ``data/``."""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict, deque
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

MAX_TURNS = 8

_PLAIN_POST_TOOLS = frozenset(
    {"qq_send_group_msg", "qq_send_group_ai_record", "qq_send_group_emotion_face"}
)

_lock = threading.Lock()
_store: dict[int, deque[tuple[str, str]]] = defaultdict(lambda: deque(maxlen=MAX_TURNS))


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
    loaded: dict[int, deque[tuple[str, str]]] = {}
    for key, value in raw.items():
        try:
            gid = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, list):
            continue
        turns: list[tuple[str, str]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                u, a = item[0], item[1]
                if isinstance(u, str) and isinstance(a, str):
                    turns.append((u, a))
        if turns:
            loaded[gid] = deque(turns[-MAX_TURNS:], maxlen=MAX_TURNS)
    for gid, dq in loaded.items():
        _store[gid] = dq


def _save_persisted() -> None:
    data_dir, path = _persist_paths()
    data_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, list[list[str]]] = {}
    for gid, dq in _store.items():
        if dq:
            payload[str(gid)] = [[u, a] for u, a in dq]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


_load_persisted()


def history_as_messages(group_id: int) -> list[HumanMessage | AIMessage]:
    """Prior turns as alternating HumanMessage / AIMessage for agent context."""
    with _lock:
        dq = _store.get(group_id)
        if not dq:
            return []
        out: list[HumanMessage | AIMessage] = []
        for user_content, assistant_content in dq:
            out.append(HumanMessage(content=user_content))
            out.append(AIMessage(content=_clean_cached_assistant_content(assistant_content)))
        return out


def _clean_cached_assistant_content(content: str) -> str:
    """Strip internal transcript labels from older persisted assistant turns."""
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


def append_turn(group_id: int, user_content: str, assistant_combined: str) -> None:
    with _lock:
        _store[group_id].append((user_content, assistant_combined))
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
    t = text
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
