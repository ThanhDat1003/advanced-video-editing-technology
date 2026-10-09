from pathlib import Path

import pytest
import typer

from transcript_video.cli import _expand_sources
from transcript_video.engine import _batch_candidates, _convert_segments, _drop_hallucinated_repeats
from transcript_video.io_utils import fingerprint
from transcript_video.models import Checkpoint, SegmentResult, SourceFingerprint, TranscriptionConfig
from transcript_video.render import _timestamp, render_outputs


class Word:
    def __init__(self, start, end, word, probability=0.9):
        self.start, self.end, self.word, self.probability = start, end, word, probability


class Segment:
    start, end, text, avg_logprob, no_speech_prob = 0.0, 3.0, " xin chào", -0.1, 0.0
    words = [Word(0.0, 1.0, " xin"), Word(1.0, 3.0, " chào")]


def test_overlap_trims_words_and_uses_absolute_time():
    result = _convert_segments([Segment()], offset=898.0, keep_after=900.0)
    assert len(result) == 1
    assert result[0].text == "chào"
    assert result[0].start == 900.0
    assert result[0].end == 901.0


def test_batch_fallback_is_unique_and_reaches_one():
    assert _batch_candidates(4) == [4, 2, 1]
    assert _batch_candidates(1) == [1]


def test_srt_timestamp_rounding():
    assert _timestamp(3661.234) == "01:01:01,234"


def test_fingerprint_changes_when_edge_changes(tmp_path: Path):
    path = tmp_path / "video.bin"
    path.write_bytes(b"a" * 20)
    first = fingerprint(path)
    path.write_bytes(b"b" + b"a" * 19)
    second = fingerprint(path)
    assert first.edge_sha256 != second.edge_sha256


def _segment(index: int, text: str) -> SegmentResult:
    return SegmentResult(id=index, start=float(index), end=float(index) + 1.0, text=text)


def test_drop_hallucinated_repeats_removes_runs_of_three_or_more():
    segments = [
        _segment(0, "xin chào các bạn"),
        *[_segment(i, "Hãy subscribe kênh") for i in range(1, 6)],
        _segment(6, "cảm ơn đã theo dõi"),
    ]
    result = _drop_hallucinated_repeats(segments)
    assert [segment.text for segment in result] == ["xin chào các bạn", "cảm ơn đã theo dõi"]


def test_drop_hallucinated_repeats_keeps_short_runs():
    segments = [_segment(0, "vâng"), _segment(1, "vâng"), _segment(2, "tiếp tục nhé")]
    result = _drop_hallucinated_repeats(segments)
    assert [segment.text for segment in result] == ["vâng", "vâng", "tiếp tục nhé"]


def test_expand_sources_scans_directory_for_videos(tmp_path: Path):
    folder = tmp_path / "videos"
    folder.mkdir()
    (folder / "a.mp4").write_bytes(b"")
    (folder / "b.mkv").write_bytes(b"")
    (folder / "notes.txt").write_bytes(b"")
    explicit_file = tmp_path / "c.mov"
    explicit_file.write_bytes(b"")

    result = _expand_sources([folder, explicit_file])

    assert result == [folder / "a.mp4", folder / "b.mkv", explicit_file]


def test_expand_sources_rejects_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(typer.BadParameter):
        _expand_sources([empty])


def test_render_outputs_are_consistent(tmp_path: Path):
    checkpoint = Checkpoint(
        source="video.mp4",
        fingerprint=SourceFingerprint(size=1, mtime_ns=2, edge_sha256="abc"),
        config=TranscriptionConfig(model="large-v3", device="cuda", compute_type="float16", batch_size=4),
        duration=10,
        segments=_convert_segments([Segment()], offset=0, keep_after=0),
        complete=True,
    )
    render_outputs(checkpoint, tmp_path)
    assert "xin chào" in (tmp_path / "transcript.txt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:03,000" in (tmp_path / "transcript.srt").read_text(encoding="utf-8")
    assert '"complete": true' in (tmp_path / "transcript.json").read_text(encoding="utf-8")
