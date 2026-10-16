"""阿里云（DashScope）生图平台逻辑，供 agent.tools.generate_image 后台调用。

对外仅暴露 `generate_images(prompt, reference_urls)`，返回生成图片的 URL 列表；
参考图与 prompt 是两个通用参数，未来接入其他生图平台时实现相同签名即可无缝替换。
"""

import asyncio
from typing import Sequence

import dashscope
from dashscope import MultiModalConversation

import config

# qwen 生图系列走专用网关（参考 test.py）
dashscope.base_http_api_url = "https://maas.qianwenaiapi.com/api/v1"

_MODEL = config.DASHSCOPE_API_MODEL
_SIZE = "1024*1536"
_MAX_REFERENCE_IMAGES = 3  # 模型支持一至三张输入图片
_OUTPUT_COUNT = 1  # 输出 1 至 6 张图片


def _call_sync(prompt: str, reference_urls: Sequence[str]) -> list[str]:
    content: list[dict] = [{"image": url} for url in reference_urls]
    content.append({"text": prompt})
    response = MultiModalConversation.call(
        api_key=config.DASHSCOPE_API_KEY,
        model=_MODEL,
        messages=[{"role": "user", "content": content}],
        stream=False,
        n=_OUTPUT_COUNT,
        watermark=False,
        negative_prompt=" ",
        prompt_extend=True,
        size=_SIZE,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"生图请求失败 HTTP={response.status_code} code={response.code} "
            f"message={response.message}"
        )
    urls = [
        part["image"]
        for part in response.output.choices[0].message.content
        if isinstance(part, dict) and part.get("image")
    ]
    if not urls:
        raise RuntimeError("生图成功但响应中未包含图片")
    return urls


async def generate_images(
    prompt: str, reference_urls: Sequence[str] = ()
) -> list[str]:
    """后台生成图片，返回生成图片的 URL 列表（阻塞 SDK 调用放到线程池执行）。"""
    return await asyncio.to_thread(
        _call_sync, prompt, list(reference_urls)[:_MAX_REFERENCE_IMAGES]
    )
