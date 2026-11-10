"""Breeze Blue TTS（https://docs.breezeblue.ai/）→ 本地 MP3 语音文件。

POST {BASE}/v1/text-to-speech/{voice_id}?output_format=mp3，鉴权头 xi-api-key，
响应体即音频二进制。key / voice_id / 模型从 config 环境变量加载。

情绪表达（参考官方 guides）：
- instructions：自然语言表演指令（情境/意图/情绪强度/节奏/身体状态），与文本同语言；
  有指令时 guidance_scale 取 1～3（默认 2），无指令时开 use_speaker_boost 保音色。
- 音频标签：中文用方括号写在文本对应位置（如「[笑]」「[叹气]」），
  官方说明 34 种标签在 breeze-tts-2-multilingual 上可用，故文本带标签时切到该模型。
  不在白名单里的方括号/括号动作描写会被清掉，避免被念出来。
- speed：0.5～2.0，保持音高。
"""

import asyncio
import re
import uuid
from pathlib import Path

import httpx

import config

_MP3_DIR = Path("data/tts")
_MULTILINGUAL_MODEL = "breeze-tts-2-multilingual"

# 简体中文音频标签（官方 Audio tags 表 zh 列），按类别排列
ZH_AUDIO_TAGS: tuple[str, ...] = (
    "笑", "轻笑", "咯咯笑",
    "哭声", "抽泣", "呜咽", "呻吟", "哀号",
    "叹气", "惊呼", "吸气", "呼气", "喘息",
    "耳语", "喊叫", "惨叫", "唱歌", "哼歌", "结巴", "停顿",
    "清嗓", "咳嗽", "吸鼻子",
    "咂嘴", "啧嘴",
    "打哈欠", "打喷嚏", "打嗝", "打饱嗝", "吞咽", "干呕",
    "闷哼", "冷笑", "哼鼻子",
)
_TAG_SET = frozenset(ZH_AUDIO_TAGS)
# 模型常写成括号动作的近义词 → 官方标签
_TAG_ALIASES = {
    "笑声": "笑", "大笑": "笑", "哈哈": "笑", "偷笑": "咯咯笑", "憋笑": "轻笑",
    "微笑": "轻笑", "哭": "哭声", "哭泣": "抽泣", "叹息": "叹气", "叹": "叹气",
    "惊讶": "惊呼", "倒吸一口气": "吸气", "深呼吸": "吸气", "小声": "耳语",
    "悄悄": "耳语", "低语": "耳语", "大喊": "喊叫", "尖叫": "惨叫", "哼": "哼鼻子",
    "哈欠": "打哈欠", "咳": "咳嗽", "啧": "啧嘴", "嗤笑": "冷笑", "不屑": "冷笑",
}
_BRACKET_RE = re.compile(r"[\[【（(]\s*([^\[\]【】（）()]{1,12}?)\s*[\]】）)]")
_MD_RE = re.compile(r"[*_`#>~]+|!?\[[^\]]*\]\([^)]*\)|https?://\S+")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE0F\U0000200D]+"
)


# 人设基调：语音表演指令必须带「懒散/懒洋洋」一类词，缺了就由代码补上
LAZY_KEYWORDS: tuple[str, ...] = (
    "懒散", "懒洋洋", "慵懒", "懒懒", "懒懒散散", "懒得", "提不起劲", "没睡醒",
)
_LAZY_PREFIX = "整体懒洋洋、懒散的语气"


def ensure_lazy_instructions(instructions: str | None) -> str:
    """保证表演指令里有懒散类词汇；没有则在开头补「整体懒洋洋、懒散的语气」。"""
    instructions = (instructions or "").strip()
    if any(k in instructions for k in LAZY_KEYWORDS):
        return instructions
    if not instructions:
        return _LAZY_PREFIX + "，随口说出来。"
    return f"{_LAZY_PREFIX}，{instructions}"


def normalize_text(text: str) -> tuple[str, bool]:
    """清洗朗读文本：去 markdown/链接/emoji；括号动作映射为官方标签，其余括号内容删掉。

    返回 (清洗后文本, 是否含音频标签)。"""
    text = _MD_RE.sub("", text or "")
    text = _EMOJI_RE.sub("", text)
    has_tag = False

    def repl(m: re.Match) -> str:
        nonlocal has_tag
        word = m.group(1).strip()
        tag = word if word in _TAG_SET else _TAG_ALIASES.get(word)
        if tag:
            has_tag = True
            return f"[{tag}]"
        return ""

    text = _BRACKET_RE.sub(repl, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip(), has_tag


async def synthesize_to_mp3(
    text: str,
    voice_id: str | None = None,
    *,
    instructions: str | None = None,
    speed: float | None = None,
    guidance_scale: float | None = None,
) -> str:
    """合成文本为 MP3，返回本地 MP3 文件路径。

    Raises:
        ValueError: 文本为空。
        RuntimeError: 未配置 key / 合成失败 / 无音频。
    """
    text, has_tag = normalize_text(text)
    if not text or not re.sub(r"\[[^\]]*\]", "", text).strip():
        raise ValueError("TTS 文本为空")
    if not config.BREEZE_API_KEY:
        raise RuntimeError("Breeze TTS 未配置：需要环境变量 BREEZE_API_KEY")
    voice_id = (voice_id or config.BREEZE_VOICE_ID).strip()
    instructions = (instructions or "").strip()[:1000]

    body: dict = {"text": text, "language_code": config.BREEZE_LANGUAGE_CODE or "zh"}
    model = config.BREEZE_TTS_MODEL or (_MULTILINGUAL_MODEL if has_tag else "")
    if model:
        body["model_id"] = model
    settings: dict = {}
    if instructions:
        body["instructions"] = instructions
        gs = guidance_scale if guidance_scale is not None else config.BREEZE_GUIDANCE_SCALE
        settings["guidance_scale"] = min(max(float(gs), 1.0), 10.0)
    else:
        # 无表演指令时开 speaker boost，尽量贴近参考音色
        settings["use_speaker_boost"] = True
    if speed is not None:
        settings["speed"] = min(max(float(speed), 0.5), 2.0)
    if config.BREEZE_VOLUME != 1.0:
        settings["volume"] = config.BREEZE_VOLUME
    body["voice_settings"] = settings

    url = f"{config.BREEZE_BASE_URL}/v1/text-to-speech/{voice_id}"
    async with httpx.AsyncClient(timeout=90) as client:
        for attempt in range(2):
            resp = await client.post(
                url,
                params={"output_format": "mp3_24000_128"},
                headers={"xi-api-key": config.BREEZE_API_KEY},
                json=body,
            )
            # 限流 / 容量不足：按 Retry-After 等一次再试（最多等 10 秒）
            if resp.status_code in (429, 503) and attempt == 0:
                try:
                    wait = float(resp.headers.get("retry-after", "2"))
                except ValueError:
                    wait = 2.0
                await asyncio.sleep(min(max(wait, 0.5), 10.0))
                continue
            break
    if resp.status_code != 200:
        raise RuntimeError(f"Breeze TTS 合成失败 HTTP {resp.status_code}: {resp.text[:300]}")
    if resp.headers.get("content-type", "").startswith("application/json"):
        raise RuntimeError(f"Breeze TTS 未返回音频: {resp.text[:300]}")
    audio = resp.content
    if not audio:
        raise RuntimeError("Breeze TTS 未返回音频数据")

    _MP3_DIR.mkdir(parents=True, exist_ok=True)
    mp3_path = _MP3_DIR / f"{uuid.uuid4()}.mp3"
    mp3_path.write_bytes(audio)
    return str(mp3_path)
