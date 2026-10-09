from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .models import ProbeResult


class MediaError(RuntimeError):
    pass


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=flags)


def probe(path: Path) -> ProbeResult:
    result = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=index,codec_type",
        "-of", "json", str(path),
    ])
    if result.returncode:
        raise MediaError(f"ffprobe không đọc được file: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
        streams = [int(item["index"]) for item in payload.get("streams", []) if item.get("codec_type") == "audio"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaError("Không xác định được thời lượng hoặc audio stream.") from exc
    if not streams:
        raise MediaError("Video không có audio stream.")
    return ProbeResult(duration=duration, audio_streams=streams)


def extract_chunk(source: Path, destination: Path, start: float, length: float, stream: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(source),
        "-map", f"0:{stream}", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination),
    ])
    if result.returncode or not destination.exists() or destination.stat().st_size <= 44:
        raise MediaError(f"FFmpeg không trích xuất được audio: {result.stderr.strip()}")

