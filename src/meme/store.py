"""梗词库存储：data/memes.db（SQLite），增删记录写入 data/meme.log。

词条按 scope 区分：
- "global"：导入的公共词库（如 CHIME）；
- 群号字符串：某群通过 /meme -add 或 remember_meme 添加的群内词条。
群内 meaning 为空串的记录是“屏蔽标记”：/meme -del 删除公共词条时写入，
仅在该群屏蔽同名公共词条，不影响其他群。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

GLOBAL_SCOPE = "global"

_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_PATH = _DATA_DIR / "memes.db"
LOG_PATH = _DATA_DIR / "meme.log"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    term       TEXT    NOT NULL,
    meaning    TEXT    NOT NULL,
    origin     TEXT,
    examples   TEXT    NOT NULL DEFAULT '[]',
    aliases    TEXT    NOT NULL DEFAULT '[]',
    type       TEXT,
    offense    INTEGER NOT NULL DEFAULT 0,
    scope      TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    created_at INTEGER NOT NULL,
    UNIQUE (term, scope)
);
CREATE TABLE IF NOT EXISTS meme_vec (
    meme_id INTEGER PRIMARY KEY REFERENCES memes(id) ON DELETE CASCADE,
    vec     BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- 自动学习候选：群里出现、词库未收录的疑似梗
CREATE TABLE IF NOT EXISTS slang_candidates (
    group_id      INTEGER NOT NULL,
    term          TEXT    NOT NULL,
    kind          TEXT,
    count         INTEGER NOT NULL DEFAULT 0,
    speakers      TEXT    NOT NULL DEFAULT '[]',
    samples       TEXT    NOT NULL DEFAULT '[]',
    need_count    INTEGER NOT NULL,
    need_speakers INTEGER NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pending',
    updated_at    INTEGER NOT NULL,
    PRIMARY KEY (group_id, term)
);
"""


@dataclass(frozen=True)
class Meme:
    id: int
    term: str
    meaning: str
    origin: str | None
    examples: tuple[str, ...]
    aliases: tuple[str, ...]
    type: str | None
    offense: bool
    scope: str
    source: str

    @property
    def disabled(self) -> bool:
        return not self.meaning

    def short_meaning(self, limit: int = 80) -> str:
        """注入用的精简释义：取首句，超长截断。"""
        text = " ".join(self.meaning.split())
        end = text.find("。")
        if 0 < end < limit:
            return text[: end + 1]
        return text if len(text) <= limit else text[:limit] + "…"


_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
# 内存索引：normalized term -> {scope: Meme}；变更后整体重建（词条量级为千条，开销可忽略）
_by_term: dict[str, dict[str, Meme]] = {}
_max_term_len = 0
_version = 0  # 每次词条变更 +1，供向量索引判断是否需要刷新


def normalize(term: str) -> str:
    """统一全角/半角与大小写、压缩空白。"""
    return " ".join(unicodedata.normalize("NFKC", term).split()).lower()


def _connection() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(_SCHEMA)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(memes)")}
        if "aliases" not in columns:  # 旧库迁移
            conn.execute("ALTER TABLE memes ADD COLUMN aliases TEXT NOT NULL DEFAULT '[]'")
            conn.commit()
        _conn = conn
        _reload_index()
    return _conn


def _row_to_meme(row: sqlite3.Row) -> Meme:
    try:
        examples = tuple(str(x) for x in json.loads(row["examples"] or "[]"))
    except (TypeError, ValueError):
        examples = ()
    try:
        aliases = tuple(str(x) for x in json.loads(row["aliases"] or "[]"))
    except (TypeError, ValueError):
        aliases = ()
    return Meme(
        id=row["id"],
        term=row["term"],
        meaning=row["meaning"],
        origin=row["origin"],
        examples=examples,
        aliases=aliases,
        type=row["type"],
        offense=bool(row["offense"]),
        scope=row["scope"],
        source=row["source"],
    )


def _reload_index() -> None:
    global _by_term, _max_term_len, _version
    assert _conn is not None
    by_term: dict[str, dict[str, Meme]] = {}
    memes = [_row_to_meme(row) for row in _conn.execute("SELECT * FROM memes")]
    for meme in memes:
        by_term.setdefault(normalize(meme.term), {})[meme.scope] = meme
    # 别名（变体写法）后索引：不覆盖同 scope 下的正式词条
    for meme in memes:
        for alias in meme.aliases:
            by_term.setdefault(normalize(alias), {}).setdefault(meme.scope, meme)
    _by_term = by_term
    _max_term_len = max((len(t) for t in by_term), default=0)
    _version += 1


def ensure_loaded() -> None:
    with _lock:
        _connection()


def version() -> int:
    ensure_loaded()
    return _version


def max_term_len() -> int:
    ensure_loaded()
    return _max_term_len


def resolve(term: str, group_id: int | None) -> Meme | None:
    """按“本群词条 > 公共词条”解析；本群屏蔽标记命中时返回 None。"""
    ensure_loaded()
    scopes = _by_term.get(normalize(term))
    if not scopes:
        return None
    if group_id is not None:
        local = scopes.get(str(int(group_id)))
        if local is not None:
            return None if local.disabled else local
    return scopes.get(GLOBAL_SCOPE)


def is_handled(term: str, group_id: int) -> bool:
    """该词在本群是否已有结论：本群词条（含屏蔽标记）或公共词条存在。"""
    ensure_loaded()
    scopes = _by_term.get(normalize(term))
    if not scopes:
        return False
    return str(int(group_id)) in scopes or GLOBAL_SCOPE in scopes


def has_term(term: str) -> bool:
    ensure_loaded()
    return normalize(term) in _by_term


def visible_memes(group_id: int | None) -> list[Meme]:
    """该群可见的全部有效词条（已应用群内覆盖与屏蔽）。"""
    ensure_loaded()
    result: list[Meme] = []
    seen: set[int] = set()
    for term in _by_term:
        meme = resolve(term, group_id)
        if meme is not None and meme.id not in seen:
            seen.add(meme.id)
            result.append(meme)
    return result


def all_active_memes() -> list[Meme]:
    """所有 meaning 非空的词条（向量索引用）。"""
    ensure_loaded()
    unique = {m.id: m for scopes in _by_term.values() for m in scopes.values()}
    return [m for m in unique.values() if not m.disabled]


def write_log(action: str, scope: str, term: str, detail: str = "", operator: str = "") -> None:
    """记录词库增删（data/meme.log）。"""
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    line = json.dumps(
        {
            "time": ts,
            "action": action,
            "scope": scope,
            "term": term,
            "detail": detail,
            "operator": operator,
        },
        ensure_ascii=False,
    )
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        print(f"写入 meme.log 失败: {e}")


def upsert_global(entries: list[dict], source: str) -> tuple[int, int]:
    """批量写入公共词条（导入脚本用），返回 (新增数, 更新数)。"""
    with _lock:
        conn = _connection()
        added = updated = 0
        now = int(time.time())
        for e in entries:
            term = " ".join(str(e["term"]).split())
            if not term or not e.get("meaning"):
                continue
            exists = conn.execute(
                "SELECT id, meaning FROM memes WHERE term = ? AND scope = ?",
                (term, GLOBAL_SCOPE),
            ).fetchone()
            values = (
                str(e["meaning"]).strip(),
                e.get("origin"),
                json.dumps(list(e.get("examples") or []), ensure_ascii=False),
                e.get("type"),
                1 if e.get("offense") else 0,
            )
            if exists is None:
                conn.execute(
                    "INSERT INTO memes (term, meaning, origin, examples, type, offense,"
                    " scope, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (term, *values, GLOBAL_SCOPE, source, now),
                )
                added += 1
            else:
                if exists["meaning"] != values[0]:
                    conn.execute("DELETE FROM meme_vec WHERE meme_id = ?", (exists["id"],))
                conn.execute(
                    "UPDATE memes SET meaning = ?, origin = ?, examples = ?, type = ?,"
                    " offense = ?, source = ? WHERE id = ?",
                    (*values, source, exists["id"]),
                )
                updated += 1
        conn.commit()
        _reload_index()
        return added, updated


def add_group_meme(
    group_id: int, term: str, meaning: str, *, source: str, operator: str = ""
) -> Meme:
    """新增或覆盖本群词条，并记录日志。"""
    term = " ".join(term.split())
    meaning = " ".join(meaning.split())
    scope = str(int(group_id))
    with _lock:
        conn = _connection()
        old = conn.execute(
            "SELECT id FROM memes WHERE term = ? AND scope = ?", (term, scope)
        ).fetchone()
        if old is not None:
            conn.execute("DELETE FROM memes WHERE id = ?", (old["id"],))
        conn.execute(
            "INSERT INTO memes (term, meaning, origin, examples, type, offense,"
            " scope, source, created_at) VALUES (?, ?, NULL, '[]', NULL, 0, ?, ?, ?)",
            (term, meaning, scope, source, int(time.time())),
        )
        conn.commit()
        _reload_index()
    write_log("update" if old else "add", scope, term, meaning, operator or source)
    meme = resolve(term, group_id)
    assert meme is not None
    return meme


def delete_group_meme(group_id: int, term: str, *, operator: str = "") -> str | None:
    """在本群删除词条。

    本群词条直接删除；仅公共词条存在时写入屏蔽标记（只影响本群）。
    返回 "deleted" / "blocked"，词条不存在或已屏蔽时返回 None。
    """
    term = " ".join(term.split())
    scope = str(int(group_id))
    with _lock:
        conn = _connection()
        scopes = _by_term.get(normalize(term), {})
        local = scopes.get(scope)
        global_meme = scopes.get(GLOBAL_SCOPE)
        if local is not None and not local.disabled:
            if global_meme is not None:
                # 删掉本群覆盖后会露出同名公共词条，直接改为屏蔽标记
                conn.execute(
                    "UPDATE memes SET meaning = '', source = 'block' WHERE id = ?",
                    (local.id,),
                )
                conn.execute("DELETE FROM meme_vec WHERE meme_id = ?", (local.id,))
            else:
                conn.execute("DELETE FROM memes WHERE id = ?", (local.id,))
            result = "deleted"
        elif local is None and global_meme is not None:
            conn.execute(
                "INSERT INTO memes (term, meaning, origin, examples, type, offense,"
                " scope, source, created_at) VALUES (?, '', NULL, '[]', NULL, 0, ?, 'block', ?)",
                (global_meme.term, scope, int(time.time())),
            )
            result = "blocked"
        else:
            return None
        conn.commit()
        _reload_index()
    write_log("delete" if result == "deleted" else "block", scope, term, "", operator)
    return result


# ---- 向量存取（embed.py 使用） ----


def get_meta(key: str) -> str | None:
    with _lock:
        row = _connection().execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None


def set_meta(key: str, value: str) -> None:
    with _lock:
        conn = _connection()
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()


def clear_vectors() -> None:
    with _lock:
        conn = _connection()
        conn.execute("DELETE FROM meme_vec")
        conn.commit()


def load_vectors() -> list[tuple[int, bytes]]:
    with _lock:
        return [
            (row["meme_id"], row["vec"])
            for row in _connection().execute("SELECT meme_id, vec FROM meme_vec")
        ]


def save_vectors(items: list[tuple[int, bytes]]) -> None:
    with _lock:
        conn = _connection()
        conn.executemany(
            "INSERT OR REPLACE INTO meme_vec (meme_id, vec) SELECT ?, ?"
            " WHERE EXISTS (SELECT 1 FROM memes WHERE id = ?)",
            [(mid, vec, mid) for mid, vec in items],
        )
        conn.commit()


# ---- 自动学习（learn.py 使用） ----


def add_learned_meme(
    scope: str,
    term: str,
    meaning: str,
    *,
    aliases: list[str],
    type_: str | None,
    offense: bool,
    origin: str | None,
    examples: list[str],
    detail: str,
) -> Meme | None:
    """写入自动学到的词条；该 scope 下已有同名词条（含人工添加或屏蔽）时不覆盖，返回 None。"""
    term = " ".join(term.split())
    with _lock:
        conn = _connection()
        if conn.execute(
            "SELECT 1 FROM memes WHERE term = ? AND scope = ?", (term, scope)
        ).fetchone():
            return None
        conn.execute(
            "INSERT INTO memes (term, meaning, origin, examples, aliases, type, offense,"
            " scope, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'auto', ?)",
            (
                term,
                " ".join(meaning.split()),
                origin,
                json.dumps(examples[:3], ensure_ascii=False),
                json.dumps([a for a in aliases if a and a != term][:5], ensure_ascii=False),
                type_,
                1 if offense else 0,
                scope,
                int(time.time()),
            ),
        )
        conn.commit()
        _reload_index()
    write_log("auto_learn", scope, term, detail, "auto")
    return _by_term[normalize(term)][scope]


def get_candidate(group_id: int, term: str) -> dict | None:
    with _lock:
        row = _connection().execute(
            "SELECT * FROM slang_candidates WHERE group_id = ? AND term = ?",
            (int(group_id), term),
        ).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["speakers"] = json.loads(item["speakers"])
    item["samples"] = json.loads(item["samples"])
    return item


def observe_candidate(
    group_id: int,
    term: str,
    *,
    kind: str,
    speaker: str,
    sample: str,
    need_count: int,
    need_speakers: int,
) -> dict:
    """记录一次出现：次数 +1、发言人去重（最多 8 个）、保留最近 3 句原话。"""
    now = int(time.time())
    with _lock:
        item = get_candidate(group_id, term)
        if item is None:
            item = {
                "group_id": int(group_id),
                "term": term,
                "kind": kind,
                "count": 0,
                "speakers": [],
                "samples": [],
                "need_count": need_count,
                "need_speakers": need_speakers,
                "status": "pending",
            }
        item["count"] += 1
        if speaker and speaker not in item["speakers"]:
            item["speakers"] = (item["speakers"] + [speaker])[-8:]
        if sample and sample not in item["samples"]:
            item["samples"] = (item["samples"] + [sample[:120]])[-3:]
        save_candidate(item, now=now)
        return item


def save_candidate(item: dict, *, now: int | None = None) -> None:
    with _lock:
        conn = _connection()
        conn.execute(
            "INSERT OR REPLACE INTO slang_candidates (group_id, term, kind, count, speakers,"
            " samples, need_count, need_speakers, status, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                item["group_id"],
                item["term"],
                item.get("kind"),
                item["count"],
                json.dumps(item["speakers"], ensure_ascii=False),
                json.dumps(item["samples"], ensure_ascii=False),
                item["need_count"],
                item["need_speakers"],
                item["status"],
                now or int(time.time()),
            ),
        )
        conn.commit()


def take_daily_quota(name: str, limit: int) -> bool:
    """按自然日计数的额度；未超过 limit 时占用一次并返回 True。"""
    today = time.strftime("%Y-%m-%d")
    key = f"quota:{name}"
    with _lock:
        raw = get_meta(key)
        try:
            day, used = (raw or "").split("|")
            used_n = int(used) if day == today else 0
        except ValueError:
            used_n = 0
        if used_n >= limit:
            return False
        set_meta(key, f"{today}|{used_n + 1}")
        return True
