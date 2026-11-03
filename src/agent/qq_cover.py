"""QQ 音乐卡片解析 + Arpeggi AI 翻唱流水线。

组成：
1. ``parse_music_card``：从 QQ 音乐分享卡片 JSON 中提取 songmid / 歌名 / 歌手。
2. ``get_qq_music_url``：按 QQ 音乐 vkey 接口取播放直链（参考 test.py 的构造规则）。
3. ``download_audio``：把直链流式下载到本地临时目录。
4. ``run_cover``：并发提交 Arpeggi 声线转换 + 音轨分离，取「转换后人声」与「伴奏」，
   用 ffmpeg 多级降级合并成最终翻唱音频。

上游文档：
- https://arpeggi.io/docs  voice-conversions / vocal-separations
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import httpx

import config

# ---- QQ 音乐 ----

# 卡片 jumpUrl / musicUrl 里的 songmid（也兼容 prompt 中的裸 mid）
_SONG_MID_RE = re.compile(r"songmid=([0-9A-Za-z]{6,24})")
# 音质 -> (文件名前缀, 后缀)，与 QQ 音乐接口约定一致；当前固定使用 128k mp3
_QUALITY_MAP = {"320": ("M800", "mp3"), "m4a": ("C400", "m4a"), "128": ("M500", "mp3")}
_QUALITY_ORDER = ("128",)  # 固定 128k mp3：后续统一转码，无需高音质

_QQ_HEADERS = {
    "Content-Type": "application/json",
    "Referer": "https://y.qq.com/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


class CoverError(RuntimeError):
    """翻唱流程中的可预期失败（下载失败 / 无版权 / 上游任务失败等）。"""


LogFn = Callable[[str], None]


def _emit(log: LogFn | None, message: str) -> None:
    """关键节点日志：终端输出 + 可选回调（用于写入 tool_calls.log）。"""
    print(message)
    if log is not None:
        try:
            log(message)
        except Exception as error:  # noqa: BLE001
            print(f"写翻唱日志失败: {error}")


def parse_music_card(raw: object) -> dict | None:
    """解析 QQ 音乐分享卡片，返回 {songmid, title, singer, tag}；非音乐卡片返回 None。

    兼容 NapCat 的 json 段：data.data 是字符串（也可能已是 dict）。
    """
    payload = raw
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta")
    music = meta.get("music") if isinstance(meta, dict) else None
    if not isinstance(music, dict):
        return None
    songmid = extract_songmid(music.get("jumpUrl")) or extract_songmid(music.get("musicUrl"))
    if not songmid:
        songmid = extract_songmid(payload.get("prompt"))
    if not songmid:
        return None
    return {
        "songmid": songmid,
        "title": str(music.get("title") or "").strip(),
        "singer": str(music.get("desc") or "").strip(),
        "tag": str(music.get("tag") or "QQ音乐").strip(),
    }


def extract_songmid(text: object) -> str | None:
    """从 URL / 文本中提取 songmid。"""
    if not isinstance(text, str):
        return None
    decoded = unquote(text)
    match = _SONG_MID_RE.search(decoded)
    if match:
        return match.group(1)
    # 兜底：jumpUrl 被解析为 query 参数
    parsed = urlparse(decoded)
    if parsed.query:
        values = parse_qs(parsed.query).get("songmid") or []
        if values and values[0]:
            return values[0]
    return None


def _qq_music_payload(songmid: str, quality: str) -> dict:
    prefix, suffix = _QUALITY_MAP.get(quality.lower(), _QUALITY_MAP["128"])
    return {
        "req_1": {
            "module": "vkey.GetVkeyServer",
            "method": "CgiGetVkey",
            "param": {
                "filename": [f"{prefix}{songmid}{songmid}.{suffix}"],
                "guid": "10000",
                "songmid": [songmid],
                "songtype": [0],
                "uin": "0",
                "loginflag": 1,
                "platform": "20",
            },
        },
        "loginUin": "0",
        "comm": {"uin": "0", "format": "json", "ct": 24, "cv": 0},
    }


async def _fetch_qq_url(
    client: httpx.AsyncClient, songmid: str, quality: str, log: LogFn | None = None
) -> str | None:
    headers = dict(_QQ_HEADERS)
    if config.QQ_MUSIC_COOKIE:
        headers["Cookie"] = config.QQ_MUSIC_COOKIE
    try:
        resp = await client.post(
            "https://u.y.qq.com/cgi-bin/musicu.fcg",
            headers=headers,
            json=_qq_music_payload(songmid, quality),
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        _emit(log, f"QQ 音乐直链接口失败 songmid={songmid} quality={quality}: {error}")
        return None
    req = data.get("req_1") or {}
    body = req.get("data") or {}
    infos = body.get("midurlinfo") or [{}]
    purl = infos[0].get("purl") if isinstance(infos[0], dict) else None
    if not purl:
        _emit(log, f"QQ 音乐 {quality} 音质无直链")
        return None
    sip = (body.get("sip") or [""])[0]
    return f"{sip}{purl}" if sip else purl


async def get_qq_music_url(songmid: str, log: LogFn | None = None) -> str:
    """获取 QQ 音乐播放直链（固定 128k mp3）。"""
    async with httpx.AsyncClient(timeout=15) as client:
        for quality in _QUALITY_ORDER:
            url = await _fetch_qq_url(client, songmid, quality, log)
            if url:
                _emit(log, f"已获取 QQ 音乐直链（{quality}）: {url}")
                return url
    raise CoverError("无法获取 QQ 音乐播放直链（可能无版权、需 VIP Cookie 或歌曲已下架）")


async def download_audio(
    url: str, dest: Path, log: LogFn | None = None, timeout: float = 300
) -> Path:
    """流式下载音频到 dest（先写 .part 再原子替换；超时/取消时清理残留）。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    _emit(log, f"开始下载音频 -> {dest.name}")
    headers = {"Referer": "https://y.qq.com/", "User-Agent": _QQ_HEADERS["User-Agent"]}

    async def _stream() -> None:
        written = 0
        async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
            async with client.stream("GET", url, headers=headers) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        if written // (5 << 20) > (written - len(chunk)) // (5 << 20):
                            _emit(log, f"下载中 {dest.name} … {written // (1 << 20)} MB")

    try:
        await asyncio.wait_for(_stream(), timeout=timeout)
    except asyncio.TimeoutError:
        tmp.unlink(missing_ok=True)
        raise CoverError(f"音频下载超时（>{int(timeout)}s）: {dest.name}")
    except BaseException:
        # 含任务被取消（任务失败时另一条流水线会被 cancel），删除半成品
        tmp.unlink(missing_ok=True)
        raise
    if tmp.stat().st_size <= 0:
        tmp.unlink(missing_ok=True)
        raise CoverError("音频下载失败：文件为空")
    tmp.replace(dest)
    _emit(log, f"下载完成 {dest.name}（{dest.stat().st_size // 1024} KB）")
    return dest


# ---- Arpeggi ----

_RETRY_STATUS = {429, 500, 502, 503, 504}
_RUNNING_STATUS = {"running", "queued", "pending", "created", "processing"}
_POLL_INTERVAL = 5.0
_JOB_TIMEOUT = 900.0
_MAX_RETRIES = 3


def _arpeggi_headers() -> dict:
    if not config.ARPEGGI_API_KEY:
        raise CoverError("翻唱未配置：需要环境变量 ARPEGGI_API_KEY")
    return {"Authorization": f"Bearer {config.ARPEGGI_API_KEY}"}


def _arpeggi_url(path: str) -> str:
    return f"{config.ARPEGGI_BASE_URL}{path}"


async def _arpeggi_upload(
    endpoint: str, field: str, audio: Path, data: dict, log: LogFn | None = None
) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with audio.open("rb") as fh:
                async with httpx.AsyncClient(timeout=300) as client:
                    response = await client.post(
                        _arpeggi_url(endpoint),
                        headers=_arpeggi_headers(),
                        files={field: (audio.name, fh, "application/octet-stream")},
                        data=data,
                    )
        except httpx.HTTPError as error:
            last_error = error
            _emit(log, f"Arpeggi 上传异常 {endpoint}（第 {attempt} 次）：{error}")
        else:
            if response.status_code in _RETRY_STATUS:
                last_error = CoverError(f"HTTP {response.status_code}: {response.text[:300]}")
                _emit(log, f"Arpeggi 上传可重试失败 {endpoint}（第 {attempt} 次）：{last_error}")
            elif response.status_code >= 400:
                raise CoverError(f"Arpeggi 上传失败 HTTP {response.status_code}: {response.text[:300]}")
            else:
                return response.json()
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(2 ** attempt)
    raise CoverError(f"Arpeggi 上传 {endpoint} 连续失败：{last_error}")


async def create_voice_conversion(
    audio: Path, voice_model_id: str, log: LogFn | None = None
) -> int:
    data = await _arpeggi_upload(
        "/voice-conversions", "soundFile", audio, {"voiceModelId": str(voice_model_id)}, log
    )
    job_id = int(data["id"])
    _emit(log, f"[声线转换] 已创建任务 id={job_id}")
    return job_id


async def create_vocal_separation(audio: Path, log: LogFn | None = None) -> int:
    data = await _arpeggi_upload("/vocal-separations", "inputFile", audio, {}, log)
    job_id = int(data["id"])
    _emit(log, f"[音轨分离] 已创建任务 id={job_id}")
    return job_id


async def _arpeggi_get(path: str, log: LogFn | None = None) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.get(_arpeggi_url(path), headers=_arpeggi_headers())
        except httpx.HTTPError as error:
            last_error = error
            _emit(log, f"Arpeggi 查询异常 {path}（第 {attempt} 次）：{error}")
        else:
            if response.status_code in _RETRY_STATUS:
                last_error = CoverError(f"HTTP {response.status_code}: {response.text[:300]}")
                _emit(log, f"Arpeggi 查询可重试失败 {path}（第 {attempt} 次）：{last_error}")
            elif response.status_code >= 400:
                raise CoverError(f"Arpeggi 查询失败 HTTP {response.status_code}: {response.text[:300]}")
            else:
                return response.json()
        if attempt < _MAX_RETRIES:
            await asyncio.sleep(2 ** attempt)
    raise CoverError(f"Arpeggi 查询 {path} 连续失败：{last_error}")


async def wait_job(kind: str, job_id: int, tag: str, log: LogFn | None = None) -> dict:
    """轮询任务至 success；其余终态视为失败，超时抛 CoverError。"""
    prefix = "/voice-conversions/" if kind == "voice" else "/vocal-separations/"
    deadline = asyncio.get_running_loop().time() + _JOB_TIMEOUT
    printed: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        current = await _arpeggi_get(f"{prefix}{job_id}", log)
        status = str(current.get("status") or "").lower()
        if status != printed:
            _emit(log, f"[{tag}] Arpeggi id={job_id} status={status or '-'}")
            printed = status
        if status == "success":
            _emit(log, f"[{tag}] 任务完成 id={job_id}")
            return current
        if status and status not in _RUNNING_STATUS:
            raise CoverError(f"[{tag}] Arpeggi 任务失败 id={job_id} status={status}")
        await asyncio.sleep(_POLL_INTERVAL)
    raise CoverError(f"[{tag}] Arpeggi 任务超时 id={job_id}")


def _pick_backing(done: dict) -> str | None:
    # 优先取 lossy（mp3）伴奏：体积小、下载快，最终合并输出也是 mp3
    for key in ("lossyStemFileUrls", "stemFileUrls"):
        for item in done.get(key) or []:
            if str(item.get("instrument", "")).lower() == "backing" and item.get("url"):
                return str(item["url"]).strip()
    url = done.get("backingAudioFileUrl")
    return str(url).strip() if url else None


async def _voice_pipeline(
    source: Path, workdir: Path, model_id: str, log: LogFn | None = None
) -> Path:
    job_id = await create_voice_conversion(source, model_id, log)
    done = await wait_job("voice", job_id, "声线转换", log)
    # 优先 lossy（mp3）输出，避免下载几十 MB 的 wav
    url = done.get("lossyOutputFileUrl") or done.get("outputFileUrl") or done.get("recombinedAudioFileUrl")
    if not url:
        raise CoverError("[声线转换] 任务成功但没有输出 URL")
    _emit(log, f"[声线转换] 输出音频: {url}")
    suffix = Path(urlparse(str(url)).path).suffix or ".wav"
    return await download_audio(str(url).strip(), workdir / f"converted_vocal{suffix}", log)


async def _separation_pipeline(
    source: Path, workdir: Path, log: LogFn | None = None
) -> Path:
    job_id = await create_vocal_separation(source, log)
    done = await wait_job("separation", job_id, "音轨分离", log)
    url = _pick_backing(done)
    if not url:
        raise CoverError("[音轨分离] 任务成功但没有 backing 伴奏")
    _emit(log, f"[音轨分离] 伴奏音频: {url}")
    suffix = Path(urlparse(url).path).suffix or ".wav"
    return await download_audio(url, workdir / f"backing{suffix}", log)


# ---- ffmpeg 合并 ----


def resolve_ffmpeg() -> tuple[str, str | None]:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return ffmpeg, ffprobe
    try:
        from static_ffmpeg import run as static_run

        ffmpeg_bin, ffprobe_bin = static_run.get_or_fetch_platform_executables_else_raise()
        return str(ffmpeg_bin), str(ffprobe_bin)
    except Exception as error:  # noqa: BLE001
        raise CoverError(f"找不到 ffmpeg/ffprobe：{error}") from error


def _probe_audio(ffprobe: str, path: Path) -> dict:
    try:
        proc = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        print(f"ffprobe 调用失败 {path.name}: {error}")
        return {}
    if proc.returncode != 0:
        return {}
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    try:
        duration = float((payload.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    return {"duration": duration, "has_audio": bool(payload.get("streams"))}


async def _run_ffmpeg(cmd: list[str]) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    return proc.returncode or 0, stderr.decode("utf-8", "replace")


def _audio_codec_args(output: Path) -> list[str]:
    """按输出后缀选编码：mp3 体积小便于 base64 发送，其余落 pcm wav。"""
    if output.suffix.lower() == ".mp3":
        return ["-c:a", "libmp3lame", "-b:a", "192k"]
    return ["-c:a", "pcm_s16le"]


async def merge_audio(
    vocal: Path, backing: Path, output: Path, log: LogFn | None = None
) -> Path:
    """人声 + 伴奏按时间轴叠加；多级降级，任一步产出可校验音频即结束。"""
    ffmpeg, ffprobe = resolve_ffmpeg()
    vocal_info = await asyncio.to_thread(_probe_audio, ffprobe, vocal) if ffprobe else {}
    backing_info = await asyncio.to_thread(_probe_audio, ffprobe, backing) if ffprobe else {}
    if vocal_info and not vocal_info.get("has_audio"):
        raise CoverError("人声轨没有音频流")
    if backing_info and not backing_info.get("has_audio"):
        raise CoverError("伴奏轨没有音频流")

    durations = [
        value for value in (vocal_info.get("duration"), backing_info.get("duration")) if value
    ]
    target = max(durations) if durations else None
    _emit(
        log,
        f"[合并] 开始：人声 {vocal_info.get('duration') or '?'}s / "
        f"伴奏 {backing_info.get('duration') or '?'}s，目标 {target or '?'}s",
    )

    amix = "amix=inputs=2:duration=longest"
    attempts: list[tuple[str, str]] = []
    if target:
        pad = (
            f"[0:a]apad=whole_dur={target:.3f},asetpts=N/SR/TB[a0];"
            f"[1:a]apad=whole_dur={target:.3f},asetpts=N/SR/TB[a1]"
        )
        attempts.append(("apad+amix(normalize=0)", f"{pad};[a0][a1]{amix}:normalize=0"))
        attempts.append(("apad+amix", f"{pad};[a0][a1]{amix}"))
    attempts.append(("amix(longest)", f"[0:a][1:a]{amix}"))
    attempts.append(("amix(shortest)", "[0:a][1:a]amix=inputs=2:duration=shortest"))

    tmp = output.with_name(f"{output.stem}.part{output.suffix}")
    for label, graph in attempts:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(vocal), "-i", str(backing),
            "-filter_complex", graph, *_audio_codec_args(output), str(tmp),
        ]
        try:
            code, stderr = await _run_ffmpeg(cmd)
        except (OSError, subprocess.SubprocessError) as error:
            _emit(log, f"[合并] {label} 执行失败: {error}")
            continue
        if code != 0 or not tmp.exists() or tmp.stat().st_size <= 0:
            tmp.unlink(missing_ok=True)
            _emit(log, f"[合并] {label} 失败: {stderr.strip()[:300]}")
            continue
        if ffprobe and not (await asyncio.to_thread(_probe_audio, ffprobe, tmp)).get("has_audio"):
            tmp.unlink(missing_ok=True)
            _emit(log, f"[合并] {label} 产物无音频流，继续降级")
            continue
        tmp.replace(output)
        _emit(log, f"[合并] {label} 成功 -> {output.name}（{output.stat().st_size // 1024} KB）")
        return output
    raise CoverError("合并失败：所有 ffmpeg 策略均未产出可用音频")


# Arpeggi 两个接口允许的输入格式（官方文档：m4a 不被接受，需要先转码）
_ARPEGGI_FORMATS = {".wav", ".mp3", ".flac"}


async def prepare_upload(source: Path, workdir: Path, log: LogFn | None = None) -> Path:
    """把源音频转成 Arpeggi 可接受的格式（非 wav/mp3/flac 一律转 mp3）。"""
    if source.suffix.lower() in _ARPEGGI_FORMATS:
        return source
    ffmpeg, _ = resolve_ffmpeg()
    target = workdir / "upload.mp3"
    _emit(log, f"源音频 {source.suffix} 不被 Arpeggi 接受，转码为 mp3")
    code, stderr = await _run_ffmpeg([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-vn", "-c:a", "libmp3lame", "-b:a", "192k", str(target),
    ])
    if code != 0 or not target.exists() or target.stat().st_size <= 0:
        raise CoverError(f"源音频转码失败: {stderr.strip()[:300]}")
    _emit(log, f"转码完成 upload.mp3（{target.stat().st_size // 1024} KB）")
    return target


async def run_cover(source: Path, workdir: Path, log: LogFn | None = None) -> Path:
    """并发跑两个上游任务，返回合并后的本地翻唱音频路径。"""
    if not config.ARPEGGI_VOICE_MODEL_ID:
        raise CoverError("翻唱未配置：需要环境变量 ARPEGGI_VOICE_MODEL_ID")
    upload_source = await prepare_upload(source, workdir, log)
    _emit(log, f"开始并发任务：声线转换 + 音轨分离（源文件 {upload_source.name}）")
    voice_task = asyncio.create_task(
        _voice_pipeline(upload_source, workdir, config.ARPEGGI_VOICE_MODEL_ID, log)
    )
    sep_task = asyncio.create_task(_separation_pipeline(upload_source, workdir, log))
    try:
        vocal_path, backing_path = await asyncio.gather(voice_task, sep_task)
    except BaseException:
        # 任一任务失败时取消另一个，避免后台任务继续占用目录（目录随后会被删除）
        for task in (voice_task, sep_task):
            task.cancel()
        await asyncio.gather(voice_task, sep_task, return_exceptions=True)
        raise
    try:
        return await merge_audio(vocal_path, backing_path, workdir / "cover.mp3", log)
    except CoverError as error:
        # mp3 编码器不可用时回退 wav，保证仍能产出翻唱文件
        _emit(log, f"[翻唱] mp3 合并失败，回退 wav: {error}")
        return await merge_audio(vocal_path, backing_path, workdir / "cover.wav", log)
