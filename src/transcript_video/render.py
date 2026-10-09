from __future__ import annotations

from .io_utils import atomic_json, atomic_text
from .models import Checkpoint, SegmentResult


def _timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def render_outputs(checkpoint: Checkpoint, job_dir) -> None:
    segments = sorted(checkpoint.segments, key=lambda item: (item.start, item.end))
    txt = "\n".join(segment.text.strip() for segment in segments if segment.text.strip()) + "\n"
    srt_parts: list[str] = []
    for number, segment in enumerate(segments, 1):
        srt_parts.append(f"{number}\n{_timestamp(segment.start)} --> {_timestamp(segment.end)}\n{segment.text.strip()}\n")
    atomic_text(job_dir / "transcript.txt", txt)
    atomic_text(job_dir / "transcript.srt", "\n".join(srt_parts))
    atomic_json(job_dir / "transcript.json", checkpoint.model_dump(mode="json"))


def normalize_ids(segments: list[SegmentResult]) -> None:
    segments.sort(key=lambda item: (item.start, item.end))
    for index, segment in enumerate(segments):
        segment.id = index

