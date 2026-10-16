"""火山引擎流式 TTS → 本地 MP3 语音文件。

流式接收 MP3 帧并拼接，存到 data/tts/ 供 QQ 语音发送（base64:// 内联），
NapCat 自行完成 mp3 → silk 转换。相比本地转 AMR：由服务端编码，
无 ffmpeg 依赖，也无 PCM 位深假设问题。
三要素 APPID / Access Token / 音色均从 config 环境变量加载。
"""

import json
import uuid
from pathlib import Path

import websockets

import config
from .protocols import EventType, MsgType, full_client_request, receive_message

_ENDPOINT = "wss://openspeech.bytedance.com/api/v3/tts/unidirectional/stream"
_MP3_DIR = Path("data/tts")


async def synthesize_to_mp3(text: str) -> str:
    """流式合成文本为 MP3，返回本地 MP3 文件路径。

    Raises:
        ValueError: 文本为空。
        RuntimeError: TTS 未配置 / 合成失败 / 无音频。
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("TTS 文本为空")
    if (
        not config.TTS_APP_ID
        or not config.TTS_ACCESS_TOKEN
        or not config.TTS_VOICE_TYPE
    ):
        raise RuntimeError(
            "TTS 未配置：需要环境变量 TTS_APP_ID / TTS_ACCESS_TOKEN / TTS_VOICE_TYPE"
        )

    headers = {
        "X-Api-App-Key": config.TTS_APP_ID,
        "X-Api-Access-Key": config.TTS_ACCESS_TOKEN,
        "X-Api-Resource-Id": "seed-tts-2.0",
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    request = {
        "user": {"uid": str(uuid.uuid4())},
        "req_params": {
            "speaker": config.TTS_VOICE_TYPE,
            "audio_params": {"format": "mp3", "sample_rate": 24000},
            "text": text,
            "additions": json.dumps({"disable_markdown_filter": False}),
        },
    }

    # 边收 MP3 帧边拼接，流结束后落盘（mp3 流式分片可直接顺序拼接）
    audio = bytearray()
    async with websockets.connect(
        _ENDPOINT, additional_headers=headers, max_size=10 * 1024 * 1024
    ) as ws:
        await full_client_request(ws, json.dumps(request).encode())
        while True:
            msg = await receive_message(ws)
            if msg.type == MsgType.FullServerResponse:
                if msg.event == EventType.SessionFinished:
                    break
                if msg.event in (EventType.TaskFailed, EventType.SessionFailed):
                    raise RuntimeError(f"TTS 合成失败: {msg}")
            elif msg.type == MsgType.AudioOnlyServer:
                audio.extend(msg.payload)
            elif msg.type == MsgType.Error:
                raise RuntimeError(f"TTS 转换失败: {msg}")

    if not audio:
        raise RuntimeError("TTS 未返回音频数据")

    _MP3_DIR.mkdir(parents=True, exist_ok=True)
    mp3_path = _MP3_DIR / f"{uuid.uuid4()}.mp3"
    mp3_path.write_bytes(bytes(audio))
    return str(mp3_path)
