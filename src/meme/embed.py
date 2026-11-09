"""梗词条向量检索：DashScope OpenAI 兼容 embeddings 接口 + 内存 numpy 矩阵。

向量空间与模型/维度绑定：meta 表记录 embedding_model / embedding_dim，
与当前配置不一致时清空全部向量并在后台重建。
"""

from __future__ import annotations

import asyncio

import httpx
import numpy as np

import config

from . import store
from .store import Meme

_BATCH_SIZE = 10  # DashScope 兼容接口单次 input 条数上限
_QUERY_TIMEOUT = 5.0

_matrix: np.ndarray | None = None  # (N, D) 已归一化
_ids: list[int] = []
_index_version = -1
_sync_task: asyncio.Task | None = None


def meme_embedding_text(meme: Meme) -> str:
    return f"{meme.term}：{meme.meaning}"


async def embed_texts(texts: list[str], *, timeout: float = 30.0) -> np.ndarray:
    """返回 (len(texts), D) 的归一化 float32 矩阵。"""
    if not config.DASHSCOPE_API_KEY:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY")
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            response = await client.post(
                f"{config.MEME_EMBEDDING_BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {config.DASHSCOPE_API_KEY}"},
                json={
                    "model": config.MEME_EMBEDDING_MODEL,
                    "input": batch,
                    "dimensions": config.MEME_EMBEDDING_DIM,
                },
            )
            response.raise_for_status()
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            vectors.extend(d["embedding"] for d in data)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _check_space() -> None:
    """模型或维度变更时清空旧向量（不同向量空间不可混用）。"""
    model = store.get_meta("embedding_model")
    dim = store.get_meta("embedding_dim")
    current_dim = str(config.MEME_EMBEDDING_DIM)
    if model != config.MEME_EMBEDDING_MODEL or dim != current_dim:
        if model is not None or dim is not None:
            print(
                f"[meme] embedding 空间变更 {model}/{dim} -> "
                f"{config.MEME_EMBEDDING_MODEL}/{current_dim}，重建向量"
            )
        store.clear_vectors()
        store.set_meta("embedding_model", config.MEME_EMBEDDING_MODEL)
        store.set_meta("embedding_dim", current_dim)


async def sync_vectors(*, verbose: bool = False) -> int:
    """为缺少向量的有效词条补算向量，返回新增数量。"""
    _check_space()
    have = {mid for mid, _ in store.load_vectors()}
    missing = [m for m in store.all_active_memes() if m.id not in have]
    done = 0
    step = _BATCH_SIZE * 5
    for start in range(0, len(missing), step):
        chunk = missing[start : start + step]
        matrix = await embed_texts([meme_embedding_text(m) for m in chunk])
        store.save_vectors([(m.id, matrix[i].tobytes()) for i, m in enumerate(chunk)])
        done += len(chunk)
        if verbose:
            print(f"[meme] 向量 {done}/{len(missing)}")
    _invalidate()
    return done


def _invalidate() -> None:
    global _index_version
    _index_version = -1


def _load_index() -> None:
    global _matrix, _ids, _index_version
    rows = store.load_vectors()
    dim = config.MEME_EMBEDDING_DIM
    rows = [(mid, vec) for mid, vec in rows if len(vec) == dim * 4]
    if rows:
        _ids = [mid for mid, _ in rows]
        _matrix = np.frombuffer(b"".join(vec for _, vec in rows), dtype=np.float32).reshape(
            len(rows), dim
        )
    else:
        _ids, _matrix = [], None
    _index_version = store.version()


def _schedule_sync() -> None:
    """词条有变更且缺向量时，后台补算（同一时间最多一个任务）。"""
    global _sync_task
    if _sync_task is not None and not _sync_task.done():
        return

    async def _run() -> None:
        try:
            await sync_vectors()
        except Exception as e:
            print(f"[meme] 向量同步失败: {e}")

    _sync_task = asyncio.create_task(_run())


async def search(
    query_vec: np.ndarray, group_id: int | None, *, top_k: int = 3, threshold: float
) -> list[tuple[Meme, float]]:
    """在该群可见词条中按余弦相似度检索。"""
    if store.version() != _index_version:
        _load_index()
        if len(_ids) < len(store.all_active_memes()):
            _schedule_sync()
    if _matrix is None:
        return []
    scores = _matrix @ query_vec
    order = np.argsort(-scores)
    by_id = {m.id: m for m in store.visible_memes(group_id)}
    results: list[tuple[Meme, float]] = []
    for idx in order:
        score = float(scores[idx])
        if score < threshold:
            break
        meme = by_id.get(_ids[idx])
        if meme is None:
            continue  # 其他群的词条或本群已屏蔽
        results.append((meme, score))
        if len(results) >= top_k:
            break
    return results


async def embed_query(text: str) -> np.ndarray | None:
    try:
        return (await embed_texts([text], timeout=_QUERY_TIMEOUT))[0]
    except Exception as e:
        print(f"[meme] 查询向量失败: {e}")
        return None
