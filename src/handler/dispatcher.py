import json
from .message_handler import handle_message
from websockets.asyncio.client import ClientConnection


async def handle_event(event_data: dict, websocket: ClientConnection):
    """处理接收事件"""
    if event_data.get("post_type"):
        print(f"收到事件: {event_data.get("post_type")}")
    # print(f"收到事件: {json.dumps(event_data, ensure_ascii=False, indent=2)}")


    if event_data.get("post_type") in ("message", "message_sent"):
        print(f"收到消息: {event_data.get('message')}")
        # print(f"收到消息: {json.dumps(event_data, ensure_ascii=False, indent=2)}")
        try:
            await handle_message(event_data, websocket)
        except Exception as e:
            print(f"处理消息时出错: {e}")
        # await websocket.send(
        #     json.dumps(
        #         {
        #             "action": "fetch_custom_face",
        #             "params": {
        #                 "count": 48
        #             }
        #         }
        #     )
        # )
        # await websocket.send(
        #     json.dumps(
        #         {
        #             "action": "send_group_msg",
        #             "params": {
        #                 "group_id": 996587913,
        #                 "message": [
        #                     {
        #                         "type": "image",
        #                         "data": {
        #                             "file": "Confused.png",
        #                             "url": "https://p.qpic.cn/qq_expression/994924990/994924990_0_0_0_2E9E7C076157138B8F40D2F80F70D5EA_0_0/0",
        #                             "sub_type": 1
        #                         }
        #                     }
        #                 ]
        #             }
        #         }
        #     )
        # )
