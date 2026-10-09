from __future__ import annotations

import gc
import importlib.metadata
import json
import logging
import math
import re
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

from .io_utils import atomic_json, fingerprint, text_sha256
from .media import extract_chunk, probe
from .models import Checkpoint, SegmentResult, TranscriptionConfig, WordResult, job_directory
from .preflight import check
from .render import normalize_ids, render_outputs

console = Console()
logger = logging.getLogger("transcript_video")


class ResumeError(RuntimeError):
    pass


def _is_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda_error_out_of_memory" in message or "cublas_status_alloc_failed" in message


def _batch_candidates(requested: int) -> list[int]:
    values = [requested]
    for candidate in (requested // 2, 2, 1):
        if 1 <= candidate < requested and candidate not in values:
            values.append(candidate)
    return values


def _load_checkpoint(path: Path) -> Checkpoint:
    return Checkpoint.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    checkpoint.touch()
    atomic_json(path, checkpoint.model_dump(mode="json"))


def _runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("faster-whisper", "ctranslate2", "av", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


_DEDUP_PUNCT_RE = re.compile(r"[\s\.,!?;:\"'…]+")


def _normalize_for_dedup(text: str) -> str:
    return _DEDUP_PUNCT_RE.sub(" ", text.strip().lower()).strip()


def _drop_hallucinated_repeats(segments: list[SegmentResult], min_repeat: int = 3) -> list[SegmentResult]:
    """Drop runs of 3+ consecutive segments with identical text.

    Whisper occasionally "hallucinates" a confident but fabricated phrase
    (typically outro boilerplate) over silence or music, repeating it verbatim
    across many consecutive VAD-detected segments. A real speaker repeating
    the same short phrase 3+ times back-to-back is exceedingly rare, so this
    heuristic is safe to apply broadly.
    """
    filtered: list[SegmentResult] = []
    run: list[SegmentResult] = []
    run_key: str | None = None
    for segment in segments:
        key = _normalize_for_dedup(segment.text)
        if run and key == run_key:
            run.append(segment)
        else:
            if len(run) < min_repeat:
                filtered.extend(run)
            run = [segment]
            run_key = key
    if len(run) < min_repeat:
        filtered.extend(run)
    return filtered


def _convert_segments(raw_segments, offset: float, keep_after: float) -> list[SegmentResult]:
    converted: list[SegmentResult] = []
    for raw in raw_segments:
        words: list[WordResult] = []
        for word in raw.words or []:
            absolute_end = offset + float(word.end)
            if absolute_end <= keep_after:
                continue
            words.append(WordResult(
                start=max(keep_after, offset + float(word.start)),
                end=absolute_end,
                text=str(word.word),
                probability=getattr(word, "probability", None),
            ))
        absolute_start = offset + float(raw.start)
        absolute_end = offset + float(raw.end)
        if absolute_end <= keep_after:
            continue
        text = "".join(word.text for word in words).strip() if words else str(raw.text).strip()
        if not text:
            continue
        converted.append(SegmentResult(
            id=0,
            start=max(keep_after, words[0].start if words else absolute_start),
            end=words[-1].end if words else absolute_end,
            text=text,
            avg_logprob=getattr(raw, "avg_logprob", None),
            no_speech_prob=getattr(raw, "no_speech_prob", None),
            words=words,
        ))
    return converted


def transcribe_file(
    source: Path,
    output: Path,
    glossary_path: Path | None = None,
    audio_stream: int | None = None,
    resume: bool = True,
    overwrite: bool = False,
    model_path: str = "large-v3",
    batch_size: int = 4,
    device: str = "cuda",
    compute_type: str | None = None,
    chunk_seconds: float = 900.0,
    overlap_seconds: float = 2.0,
) -> Path:
    source = source.resolve()
    output = output.resolve()
    for warning in check(source, output, device):
        console.print(f"[yellow]Cảnh báo:[/] {warning}")
    media = probe(source)
    if audio_stream is None:
        audio_stream = media.audio_streams[0]
    if audio_stream not in media.audio_streams:
        raise ValueError(f"Audio stream {audio_stream} không tồn tại. Có: {media.audio_streams}")
    if batch_size < 1:
        raise ValueError("batch-size phải từ 1 trở lên.")
    if overlap_seconds < 0 or chunk_seconds <= overlap_seconds:
        raise ValueError("chunk-seconds phải lớn hơn overlap-seconds.")

    glossary = ""
    if glossary_path:
        glossary = glossary_path.read_text(encoding="utf-8-sig").strip()
    resolved_compute = compute_type or ("float16" if device == "cuda" else "int8")
    config = TranscriptionConfig(
        model=model_path, device=device, compute_type=resolved_compute, batch_size=batch_size,
        audio_stream=audio_stream, chunk_seconds=chunk_seconds, overlap_seconds=overlap_seconds,
        glossary_sha256=text_sha256(glossary) if glossary else None,
    )
    source_fingerprint = fingerprint(source)
    job_dir = job_directory(output, source)
    checkpoint_path = job_dir / "checkpoint.json"
    final_paths = [job_dir / name for name in ("transcript.txt", "transcript.srt", "transcript.json")]
    if any(path.exists() for path in final_paths) and not overwrite:
        raise FileExistsError(f"Kết quả đã tồn tại trong {job_dir}; dùng --overwrite để thay thế.")
    job_dir.mkdir(parents=True, exist_ok=True)
    log_handler = logging.FileHandler(job_dir / "run.log", encoding="utf-8")
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    session_started = time.perf_counter()
    previous_processing_seconds = 0.0

    try:
        if checkpoint_path.exists() and resume and not overwrite:
            checkpoint = _load_checkpoint(checkpoint_path)
            if checkpoint.fingerprint != source_fingerprint:
                raise ResumeError("File nguồn đã thay đổi; checkpoint không còn hợp lệ.")
            if checkpoint.config != config:
                raise ResumeError("Cấu hình hiện tại khác checkpoint; hãy dùng lại cấu hình cũ hoặc --overwrite.")
            previous_processing_seconds = checkpoint.processing_seconds
        else:
            checkpoint = Checkpoint(
                source=str(source), fingerprint=source_fingerprint, config=config, duration=media.duration,
                runtime_versions=_runtime_versions(),
            )
            _save_checkpoint(checkpoint_path, checkpoint)

        console.print(f"Đang nạp model [bold]{model_path}[/] ({device}, {resolved_compute})...")
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        model = WhisperModel(model_path, device=device, compute_type=resolved_compute)
        pipeline = BatchedInferencePipeline(model=model)
        step = chunk_seconds - overlap_seconds
        total_chunks = max(1, math.ceil(max(0.0, media.duration - overlap_seconds) / step))
        context = checkpoint.segments[-3:] if checkpoint.segments else []
        initial_context = " ".join(item.text for item in context)[-800:]

        with Progress(console=console) as progress:
            task = progress.add_task("Đang transcript", total=total_chunks, completed=len(checkpoint.completed_chunks))
            for index in range(total_chunks):
                if index in checkpoint.completed_chunks:
                    continue
                start = index * step
                length = min(chunk_seconds, media.duration - start)
                if length <= 0:
                    break
                chunk_path = job_dir / f"chunk-{index:05d}.wav"
                logger.info("chunk=%s start=%.3f length=%.3f", index, start, length)
                extract_chunk(source, chunk_path, start, length, audio_stream)
                prompt_parts = [part for part in (glossary, initial_context) if part]
                prompt = "\n".join(prompt_parts) or None
                raw_segments = None
                last_oom: BaseException | None = None
                try:
                    for current_batch in _batch_candidates(batch_size):
                        try:
                            segments, _ = pipeline.transcribe(
                                str(chunk_path), language="vi", task="transcribe", beam_size=5,
                                temperature=0.0, word_timestamps=True, vad_filter=True,
                                vad_parameters={"min_silence_duration_ms": 1000},
                                initial_prompt=prompt, batch_size=current_batch,
                            )
                            raw_segments = list(segments)
                            logger.info("chunk=%s batch_size=%s success", index, current_batch)
                            break
                        except RuntimeError as exc:
                            if not _is_oom(exc):
                                raise
                            last_oom = exc
                            logger.warning("chunk=%s batch_size=%s CUDA OOM", index, current_batch)
                            gc.collect()
                    if raw_segments is None:
                        raise RuntimeError(
                            "CUDA hết VRAM ngay cả với batch size 1. Hãy đóng ứng dụng dùng GPU rồi chạy --resume."
                        ) from last_oom
                    keep_after = 0.0 if index == 0 else start + overlap_seconds
                    new_segments = _convert_segments(raw_segments, start, keep_after)
                    before_dedup = len(new_segments)
                    new_segments = _drop_hallucinated_repeats(new_segments)
                    dropped = before_dedup - len(new_segments)
                    if dropped:
                        logger.warning("chunk=%s dropped %s hallucinated repeat segments", index, dropped)
                    checkpoint.segments.extend(new_segments)
                    normalize_ids(checkpoint.segments)
                    checkpoint.completed_chunks.append(index)
                    initial_context = " ".join(item.text for item in checkpoint.segments[-3:])[-800:]
                    checkpoint.processing_seconds = previous_processing_seconds + (time.perf_counter() - session_started)
                    _save_checkpoint(checkpoint_path, checkpoint)
                    progress.advance(task)
                finally:
                    chunk_path.unlink(missing_ok=True)

        normalize_ids(checkpoint.segments)
        checkpoint.segments = _drop_hallucinated_repeats(checkpoint.segments)
        checkpoint.complete = True
        checkpoint.processing_seconds = previous_processing_seconds + (time.perf_counter() - session_started)
        _save_checkpoint(checkpoint_path, checkpoint)
        render_outputs(checkpoint, job_dir)
        console.print(f"[green]Hoàn tất:[/] {job_dir}")
        return job_dir
    except KeyboardInterrupt:
        console.print("\n[yellow]Đã dừng. Checkpoint của các chunk hoàn tất vẫn được giữ để --resume.[/]")
        raise
    finally:
        logger.removeHandler(log_handler)
        log_handler.close()
        for temporary in job_dir.glob("chunk-*.wav"):
            temporary.unlink(missing_ok=True)
