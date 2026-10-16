import dotenv
import math
import os

dotenv.load_dotenv()

BOT_TOKEN = (os.getenv("BOT_TOKEN") or "").strip()
WS_PORT = int(os.getenv("WS_PORT"))
NAPCAT_WS_URL = f"{os.getenv('WS_URL').strip()}:{WS_PORT}"

try:
    WS_SEND_INTERVAL_SECONDS = float(os.getenv("WS_SEND_INTERVAL_SECONDS", "2"))
except ValueError:
    WS_SEND_INTERVAL_SECONDS = 2.0
if not math.isfinite(WS_SEND_INTERVAL_SECONDS) or WS_SEND_INTERVAL_SECONDS < 0:
    WS_SEND_INTERVAL_SECONDS = 2.0

try:
    RANDOM_INTERJECTION_PROBABILITY = float(
        os.getenv("RANDOM_INTERJECTION_PROBABILITY", "0")
    )
except ValueError:
    RANDOM_INTERJECTION_PROBABILITY = 0.0
if not math.isfinite(RANDOM_INTERJECTION_PROBABILITY):
    RANDOM_INTERJECTION_PROBABILITY = 0.0
RANDOM_INTERJECTION_PROBABILITY = min(max(RANDOM_INTERJECTION_PROBABILITY, 0.0), 1.0)

# Tavily web search API
TAVILY_KEY = os.getenv("TAVILY_KEY")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")
VISION_MODEL = os.getenv("VISION_MODEL")
AI_KEY = os.getenv("AI_KEY")
AI_BASE_URL = os.getenv("AI_BASE_URL")

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_API_MODEL = os.getenv("DASHSCOPE_API_MODEL")

# NapCat send_group_ai_record voice character id (override via AI_VOICE_CHARACTER)
AI_VOICE_CHARACTER = (os.getenv("AI_VOICE_CHARACTER") or "lucy-voice-female1").strip()

# ---- 火山引擎流式 TTS（src/tts/），三要素从环境变量加载 ----
TTS_APP_ID = (os.getenv("TTS_APP_ID") or "").strip()
TTS_ACCESS_TOKEN = (os.getenv("TTS_ACCESS_TOKEN") or "").strip()
TTS_VOICE_TYPE = (os.getenv("TTS_VOICE_TYPE") or "").strip()

def _parse_int_list(raw: str) -> list[int]:
    ids: list[int] = []
    for part in raw.replace(";", ",").split(","):
        _p = part.strip()
        if not _p:
            continue
        try:
            ids.append(int(_p))
        except ValueError:
            pass
    return ids


# Comma- or semicolon-separated QQ group ids for scheduled proactive messages (08:00 / 19:00 local)
PROACTIVE_GROUP_IDS: list[int] = _parse_int_list((os.getenv("PROACTIVE_GROUP_IDS") or "").strip())

# 管理员 QQ 账号（逗号或分号分隔），可执行 /shutdown、/startup 等管理员指令
ADMIN_QQ_IDS: list[int] = _parse_int_list((os.getenv("ADMIN_QQ_IDS") or "").strip())


def _parse_ws_cmd_tcp_bind(s: str) -> tuple[str, int] | None:
    s = s.strip()
    if not s or ":" not in s:
        return None
    host, _, port_s = s.rpartition(":")
    host = host.strip() or "127.0.0.1"
    try:
        port = int(port_s.strip())
    except ValueError:
        return None
    if not (1 <= port <= 65535):
        return None
    return host, port


# 可选：host:port，本地 TCP 每连接读一行 JSON 后经当前 NapCat WS 发送（便于脚本 echo | nc）
_ws_cmd_tcp_raw = (os.getenv("WS_CMD_TCP") or "").strip()
WS_CMD_TCP_ADDR: tuple[str, int] | None = _parse_ws_cmd_tcp_bind(_ws_cmd_tcp_raw)

# 无 TTY / 服务进程时可设为 0：关闭从 stdin 读行发送
WS_STDIN_CMD = (os.getenv("WS_STDIN_CMD") or "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)
