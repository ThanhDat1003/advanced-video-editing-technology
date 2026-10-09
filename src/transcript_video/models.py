from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class WordResult(BaseModel):
    start: float
    end: float
    text: str
    probability: float | None = None


class SegmentResult(BaseModel):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    words: list[WordResult] = Field(default_factory=list)


class SourceFingerprint(BaseModel):
    size: int
    mtime_ns: int
    edge_sha256: str


class TranscriptionConfig(BaseModel):
    model: str
    device: Literal["cuda", "cpu"]
    compute_type: str
    batch_size: int
    beam_size: int = 5
    language: str = "vi"
    task: str = "transcribe"
    chunk_seconds: float = 900.0
    overlap_seconds: float = 2.0
    audio_stream: int | None = None
    glossary_sha256: str | None = None


class Checkpoint(BaseModel):
    schema_version: int = 1
    source: str
    fingerprint: SourceFingerprint
    config: TranscriptionConfig
    duration: float
    completed_chunks: list[int] = Field(default_factory=list)
    segments: list[SegmentResult] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    runtime_versions: dict[str, str] = Field(default_factory=dict)
    processing_seconds: float = 0.0
    complete: bool = False

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()


class ProbeResult(BaseModel):
    duration: float
    audio_streams: list[int]


def job_directory(output: Path, source: Path) -> Path:
    return output / source.stem
