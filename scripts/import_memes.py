"""导入 CHIME 梗数据集到公共词库并计算向量。

CHIME: https://github.com/yuboxie/chime （MIT License，1458 条中文短语梗，EMNLP 2025）

用法（项目根目录）：
    uv run python scripts/import_memes.py                 # 从 GitHub 下载 chime_full.json
    uv run python scripts/import_memes.py path/to/chime_full.json
    uv run python scripts/import_memes.py --vectors-only  # 只补算缺失向量
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from src.meme import embed, store  # noqa: E402

CHIME_URL = "https://raw.githubusercontent.com/yuboxie/chime/main/data/chime_full.json"


def _load_chime(path: str | None) -> list[dict]:
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        response = httpx.get(CHIME_URL, timeout=60, follow_redirects=True)
        response.raise_for_status()
        raw = response.json()
    entries = []
    for item in raw:
        entries.append(
            {
                "term": item["meme"],
                "meaning": item["meaning"],
                "origin": item.get("origin"),
                "examples": item.get("examples") or [],
                "type": item.get("type_cn"),
                "offense": bool(item.get("profanity") or item.get("offense")),
            }
        )
    return entries


async def main(argv: list[str]) -> None:
    if "--vectors-only" not in argv:
        path = next((a for a in argv if not a.startswith("--")), None)
        entries = _load_chime(path)
        added, updated = store.upsert_global(entries, source="chime")
        store.write_log(
            "import", store.GLOBAL_SCOPE, "*", f"chime: 新增 {added}，更新 {updated}", "script"
        )
        print(f"导入完成：新增 {added}，更新 {updated}")
    done = await embed.sync_vectors(verbose=True)
    print(f"向量补算完成：{done} 条")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
