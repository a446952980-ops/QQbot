"""MusicAPI（Suno 风格）persona 音乐生成适配。

仅适配 ``task_type: persona_music``（用虚拟歌手音色演唱自定义歌词）：
- POST /api/v1/sonic/create  → 返回 task_id（credits 20）
- GET  /api/v1/sonic/task/{task_id} → 轮询，data[] 为歌曲数组（state: pending/running/succeeded/failed）

参考文档：
- https://docs.musicapi.ai/concat-music （create 统一端点，task_type 决定操作）
- https://docs.musicapi.ai/get-sonic-music （GET 轮询端点）
"""

import asyncio
import time

import httpx

import config

# 当前文档列出的合法模型。Suno 已把退役 id 映射到 v6，传旧值也不会被拒。
VALID_MV = {
    "sonic-v6",
    "sonic-v6-wild",
    "sonic-v6-mini",
    "sonic-v5-5",
    "sonic-v5",
    "sonic-v4-5-plus",
    "sonic-v4-5",
    "sonic-v4",
    "sonic-v3-5",
}


def _endpoint(path: str) -> str:
    return f"{config.MUSIC_API_BASE_URL}{path}"


def _auth_headers() -> dict:
    if not config.MUSIC_API_KEY:
        raise RuntimeError("音乐生成未配置：需要环境变量 MUSIC_API_KEY")
    return {
        "Authorization": f"Bearer {config.MUSIC_API_KEY}",
        "Content-Type": "application/json",
    }


async def create_persona_music(payload: dict) -> str:
    """创建 persona_music 任务，返回 task_id。"""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            _endpoint("/api/v1/sonic/create"),
            headers=_auth_headers(),
            json=payload,
        )
    if response.status_code != 200:
        raise RuntimeError(
            f"音乐创建失败 HTTP {response.status_code}: {response.text[:300]}"
        )
    data = response.json()
    task_id = data.get("task_id")
    if not task_id:
        raise RuntimeError(f"音乐创建未返回 task_id: {response.text[:300]}")
    return str(task_id)


async def get_task(task_id: str) -> dict | None:
    """查询音乐任务结果。

    HTTP 202 / body 中 type=not_ready 表示任务尚未生成完毕，返回 None 继续轮询；
    其余非 200 才视为真实错误抛异常。
    """
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(
            _endpoint(f"/api/v1/sonic/task/{task_id}"),
            headers=_auth_headers(),
        )
    if response.status_code == 202:
        return None
    if response.status_code != 200:
        raise RuntimeError(
            f"音乐结果查询失败 HTTP {response.status_code}: {response.text[:300]}"
        )
    data = response.json()
    if data.get("type") == "not_ready":
        return None
    return data


async def poll_task_until_done(
    task_id: str, interval: int = 20, timeout: int = 600
) -> list[dict]:
    """轮询直到 task 中所有 clip 都处于终态（succeeded/failed），返回 data 列表。

    文档建议轮询间隔 15-25s。生成通常约两分钟，这里默认最多等 10 分钟。
    HTTP 202（not_ready）视为未完成，继续等待，不算错误。
    """
    deadline = time.time() + timeout
    while True:
        resp = await get_task(task_id)
        items = (resp or {}).get("data") or []
        if items and all(
            it.get("state") in ("succeeded", "failed") for it in items
        ):
            return items
        if time.time() > deadline:
            raise TimeoutError(
                f"音乐生成超时（>{timeout}s 未完成），task_id={task_id}"
            )
        await asyncio.sleep(interval)