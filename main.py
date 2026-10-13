import asyncio
import json

from websockets.asyncio.client import connect

from src.action.napcat_api import (
    fail_pending_napcat_api_requests,
    resolve_napcat_api_response,
)
from config import (
    BOT_TOKEN,
    NAPCAT_WS_URL,
    PROACTIVE_GROUP_IDS,
    WS_CMD_TCP_ADDR,
    WS_STDIN_CMD,
)
from src.handler.dispatcher import handle_event
from src.scheduler.broadcast import proactive_scheduler_loop
from src.ws_cmd_inject import (
    register_napcat_client_ws,
    serve_tcp_napcat_api_inject,
    stdin_napcat_api_inject_loop,
)


async def connect_to_napcat():
    """连接到 NapCat WebSocket 服务"""
    url = f"{NAPCAT_WS_URL}?access_token={BOT_TOKEN}"
    headers = {}
    if BOT_TOKEN:
        headers["Authorization"] = f"Bearer {BOT_TOKEN}"

    while True:
        try:
            async with connect(url, additional_headers=headers) as websocket:
                print(f"已连接到 NapCat: {NAPCAT_WS_URL}")
                register_napcat_client_ws(websocket)
                event_tasks: set[asyncio.Task] = set()

                def _track_event_task(task: asyncio.Task) -> None:
                    event_tasks.discard(task)
                    if not task.cancelled() and (error := task.exception()) is not None:
                        print(f"处理消息事件时出错: {error}")

                inject_tasks: list[asyncio.Task] = []
                if WS_STDIN_CMD:
                    inject_tasks.append(
                        asyncio.create_task(
                            stdin_napcat_api_inject_loop(),
                            name="stdin-napcat-ws-cmd",
                        )
                    )
                if WS_CMD_TCP_ADDR:
                    host, port = WS_CMD_TCP_ADDR
                    inject_tasks.append(
                        asyncio.create_task(
                            serve_tcp_napcat_api_inject(host, port),
                            name="tcp-napcat-ws-cmd",
                        )
                    )
                sched_task = asyncio.create_task(proactive_scheduler_loop(websocket))
                try:
                    async for message in websocket:
                        try:
                            event_data = json.loads(message)
                            if resolve_napcat_api_response(event_data):
                                continue
                            if event_data.get("post_type") == "message":
                                task = asyncio.create_task(
                                    handle_event(event_data, websocket)
                                )
                                event_tasks.add(task)
                                task.add_done_callback(_track_event_task)
                            else:
                                await handle_event(event_data, websocket)
                        except json.JSONDecodeError as e:
                            print(f"JSON 解析错误: {e}")
                        except Exception as e:
                            print(f"处理事件时出错: {e}")
                finally:
                    fail_pending_napcat_api_requests(websocket)
                    for task in event_tasks:
                        task.cancel()
                    if event_tasks:
                        await asyncio.gather(*event_tasks, return_exceptions=True)
                    for t in inject_tasks:
                        t.cancel()
                    for t in inject_tasks:
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass
                    register_napcat_client_ws(None)
                    sched_task.cancel()
                    try:
                        await sched_task
                    except asyncio.CancelledError:
                        pass
                        
        except Exception as e:
            print(f"连接断开，5秒后重试... 错误: {e}")
            # Linux ECONNREFUSED=111，Windows WSAECONNREFUSED=10061
            if isinstance(e, ConnectionRefusedError) or getattr(e, "errno", None) in (111, 10061):
                print(
                    f"NapCat 正向ws未开启: {NAPCAT_WS_URL}"
                )
            await asyncio.sleep(5)


async def main():
    """主函数"""
    print("正在启动 QQ Bot...")
    print(f"NapCat 地址: {NAPCAT_WS_URL}")
    if PROACTIVE_GROUP_IDS:
        print(
            f"定时任务已启用，群: {PROACTIVE_GROUP_IDS}（每天约 00:00 群打卡；08:00 / 19:00 问候）"
        )
    await connect_to_napcat()


if __name__ == "__main__":
    asyncio.run(main())
