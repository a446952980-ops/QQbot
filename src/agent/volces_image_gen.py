"""火山引擎（火山方舟 Ark）Seedream 生图平台适配。

与 aliyun_image_gen.generate_images 保持同签名（prompt + 可选参考图 URL 列表），
由 config.IMAGE_GEN_PLATFORM 动态切换。prompt 可直接描述画面比例；
payload 常驻 web_search 工具，涉及时效性/事实信息时平台会自动联网检索辅助生图。
"""

import httpx

import config

_ARK_ENDPOINT = "https://ark.cn-beijing.volces.com/api/v3/images/generations"


async def generate_images(prompt: str, reference_urls=()) -> list[str]:
    """调用 Ark images/generations，返回生成图片的 URL 列表。"""
    if not config.ARK_API_KEY:
        raise RuntimeError("火山生图未配置：需要环境变量 ARK_API_KEY")
    payload = {
        "model": config.VOLCES_IMAGE_MODEL,
        "prompt": prompt,
        "size": "2K",
        "tools": [{"type": "web_search"}],
        "output_format": "png",
        "watermark": False,
    }
    refs = [u for u in (reference_urls or ()) if u][:3]
    if refs:
        payload["image"] = refs
    headers = {
        "Authorization": f"Bearer {config.ARK_API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(_ARK_ENDPOINT, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(
            f"火山生图失败 HTTP {response.status_code}: {response.text[:300]}"
        )
    data = response.json().get("data") or []
    urls = [item["url"] for item in data if item.get("url")]
    if not urls:
        raise RuntimeError(f"火山生图未返回图片: {response.text[:300]}")
    return urls
