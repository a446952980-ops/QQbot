"""已发说说记录（data/qzone_posts.jsonl），供生成时去重与事件类型轮换。"""

from __future__ import annotations

import json
import time
from pathlib import Path

POSTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "qzone_posts.jsonl"


def record_post(content: str, images: list[str], *, source: str, event: str = "") -> None:
    """记录一条已发出的说说（source: auto=定时发布 / agent=聊天中模型自发）。"""
    entry = {
        "time": int(time.time()),
        "source": source,
        "event": event,
        "content": content,
        "images": list(images),
    }
    try:
        POSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(POSTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[qzone] 记录说说失败: {e}")


def recent_posts(limit: int = 10) -> list[dict]:
    """最近 limit 条说说，旧→新。"""
    try:
        lines = POSTS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    posts: list[dict] = []
    for line in lines[-limit * 3 :]:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict) and isinstance(item.get("content"), str):
            posts.append(item)
    return posts[-limit:]
