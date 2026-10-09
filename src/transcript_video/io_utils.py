from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .models import SourceFingerprint

EDGE_BYTES = 1024 * 1024


def fingerprint(path: Path) -> SourceFingerprint:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(EDGE_BYTES))
        if stat.st_size > EDGE_BYTES:
            handle.seek(max(0, stat.st_size - EDGE_BYTES))
            digest.update(handle.read(EDGE_BYTES))
    return SourceFingerprint(size=stat.st_size, mtime_ns=stat.st_mtime_ns, edge_sha256=digest.hexdigest())


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)

