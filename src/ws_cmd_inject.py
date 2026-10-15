"""经当前 NapCat 客户端 WebSocket 发送 API JSON（与 interaction 里 json.dumps(...) 同格式）。"""

from __future__ import annotations

import asyncio
import json
import sys
from websockets.asyncio.client import ClientConnection

from .action.ws_sender import send_ws_event
from .agent.tools import toggle_tool_logging

_client: ClientConnection | None = None


def register_napcat_client_ws(ws: ClientConnection | None) -> None:
    global _client
    _client = ws


async def send_napcat_api_json_line(line: str) -> tuple[bool, str]:
    """
    发送一行 JSON 文本（已是 NapCat API 负载字符串）。
    返回 (是否成功, 说明)。
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return False, "空行或注释，已忽略"
    try:
        json.loads(s)
    except json.JSONDecodeError as e:
        return False, f"非法 JSON: {e}"
    ws = _client
    if ws is None:
        return False, "当前无已连接的 NapCat WebSocket"
    await send_ws_event(ws, s)
    return True, "已发送"


async def stdin_napcat_api_inject_loop() -> None:
    """从标准输入每次读一行，校验为 JSON 后经当前 WS 发送（须单行完整 JSON）。"""
    print(
        "[ws-cmd] 标准输入：每行粘贴一条 JSON（与代码中 json.dumps({...}) 输出一致），回车发送；"
        "行首 # 为注释；输入 tool 回车切换工具调用日志（data/tool_calls.log）",
        flush=True,
    )
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        if line.strip() == "tool":
            enabled = toggle_tool_logging()
            print(
                f"[tool-log] 工具调用日志已{'开启（写入 data/tool_calls.log）' if enabled else '关闭'}",
                flush=True,
            )
            continue
        ok, msg = await send_napcat_api_json_line(line)
        if ok:
            print(f"[ws-cmd] {msg}", flush=True)
        elif line.strip() and not line.strip().startswith("#"):
            print(f"[ws-cmd] {msg}", flush=True)


async def _tcp_client_task(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
) -> None:
    try:
        raw = await reader.readline()
        line = raw.decode("utf-8", errors="replace")
        ok, msg = await send_napcat_api_json_line(line)
        writer.write((msg + "\n").encode("utf-8", errors="replace"))
        await writer.drain()
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def serve_tcp_napcat_api_inject(host: str, port: int) -> None:
    """监听 TCP：每个连接读取第一行 JSON，经当前 NapCat WS 发送，回写一行状态后关闭。"""
    try:
        server = await asyncio.start_server(_tcp_client_task, host, port)
    except OSError as e:
        print(f"[ws-cmd] TCP 监听失败 {host}:{port}: {e}", flush=True)
        return
    socks = server.sockets or ()
    addrs = ", ".join(str(s.getsockname()) for s in socks)
    print(f"[ws-cmd] TCP 已监听 {addrs}（每连接一行 JSON）", flush=True)
    async with server:
        await server.serve_forever()
