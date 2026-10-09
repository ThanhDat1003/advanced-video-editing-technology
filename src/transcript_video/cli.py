from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from .engine import ResumeError, transcribe_file
from .media import MediaError
from .models import job_directory
from .preflight import PreflightError

app = typer.Typer(help="Transcript video tiếng Việt bằng GPU local.", no_args_is_help=True)
console = Console(stderr=True)

_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".flv", ".webm", ".m4v", ".ts", ".mts", ".m2ts"}


def _expand_sources(sources: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for source in sources:
        if source.is_dir():
            videos = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS)
            if not videos:
                raise typer.BadParameter(f"Không tìm thấy video nào trong thư mục {source}", param_hint="sources")
            expanded.extend(videos)
        else:
            expanded.append(source)
    return expanded


def _enable_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8", errors="replace")


_enable_utf8_console()


@app.callback()
def main() -> None:
    """Transcript video tiếng Việt bằng GPU local."""


@app.command("transcribe")
def transcribe(
    sources: Annotated[list[Path], typer.Argument(help="Một hoặc nhiều video cần transcript.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Thư mục chứa kết quả.")] = Path("transcripts"),
    glossary: Annotated[Path | None, typer.Option("--glossary", help="File UTF-8, mỗi dòng một thuật ngữ.")] = None,
    audio_stream: Annotated[int | None, typer.Option("--audio-stream")] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    skip_existing: Annotated[bool, typer.Option("--skip-existing", help="Bỏ qua video đã có kết quả transcript, không tính là lỗi.")] = False,
    model_path: Annotated[str, typer.Option("--model-path")] = "large-v3",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 4,
    device: Annotated[str, typer.Option("--device", help="cuda hoặc cpu")] = "cuda",
    compute_type: Annotated[str | None, typer.Option("--compute-type", help="Mặc định float16 trên CUDA, int8 trên CPU.")] = None,
) -> None:
    if device not in {"cuda", "cpu"}:
        raise typer.BadParameter("device phải là cuda hoặc cpu", param_hint="--device")
    if skip_existing and overwrite:
        raise typer.BadParameter("--skip-existing và --overwrite không thể dùng cùng lúc", param_hint="--skip-existing")
    sources = _expand_sources(sources)
    failures = 0
    skipped = 0
    for source in sources:
        if skip_existing:
            job_dir = job_directory(output, source)
            final_paths = [job_dir / name for name in ("transcript.txt", "transcript.srt", "transcript.json")]
            if any(path.exists() for path in final_paths):
                skipped += 1
                console.print(f"[yellow]Bỏ qua {source}:[/] đã có kết quả trong {job_dir}")
                continue
        try:
            transcribe_file(
                source, output, glossary, audio_stream, resume, overwrite,
                model_path, batch_size, device, compute_type,
            )
        except KeyboardInterrupt:
            raise typer.Exit(130)
        except (PreflightError, MediaError, ResumeError, FileExistsError, ValueError, OSError, RuntimeError) as exc:
            failures += 1
            console.print(f"[red]Lỗi với {source}:[/] {exc}")
    if skipped:
        console.print(f"[yellow]Đã bỏ qua {skipped} video đã transcript.[/]")
    if failures:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
