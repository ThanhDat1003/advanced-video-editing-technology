from __future__ import annotations

import shutil
import sys
from pathlib import Path

import psutil


class PreflightError(RuntimeError):
    pass


def check(source: Path, output: Path, device: str) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []
    if sys.version_info < (3, 11):
        errors.append("Cần Python 3.11 trở lên.")
    if not source.is_file():
        errors.append(f"Không tìm thấy file: {source}")
    for executable in ("ffmpeg", "ffprobe"):
        if not shutil.which(executable):
            errors.append(f"Không tìm thấy {executable} trong PATH.")
    output.mkdir(parents=True, exist_ok=True)
    try:
        marker = output / ".write-test"
        marker.write_text("ok", encoding="ascii")
        marker.unlink()
    except OSError as exc:
        errors.append(f"Không thể ghi vào thư mục output: {exc}")
    free = psutil.disk_usage(str(output)).free
    if free < 2 * 1024**3:
        errors.append("Ổ đĩa output còn dưới 2 GB.")
    elif free < 15 * 1024**3:
        warnings.append("Ổ đĩa output còn dưới mức khuyến nghị 15 GB.")
    if device == "cuda":
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() < 1:
                errors.append("CTranslate2 không tìm thấy GPU CUDA.")
        except Exception as exc:  # DLL errors are platform-dependent.
            errors.append(f"Không khởi tạo được CTranslate2/CUDA: {exc}")
    if errors:
        raise PreflightError("\n".join(errors))
    return warnings

