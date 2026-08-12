#!/usr/bin/env python3
"""Convert the ModelScope MTEB/Daily-Omni parquet release into official layout."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_DURATION_BUCKETS = (30.0, 60.0)
_LETTER_RE = re.compile(r"^\s*\(?([A-D])\)?\s*[.、:：)\-]")

def _answer_letter(answer: str, candidates: list[str] | None) -> str:
    a = (answer or "").strip()
    if len(a) == 1 and a.upper() in "ABCD":
        return a.upper()
    m = _LETTER_RE.match(a)
    if m:
        return m.group(1)
    for c in candidates or []:
        if c.strip() == a:
            m = _LETTER_RE.match(c.strip())
            if m:
                return m.group(1)
    m = re.search(r"\b([A-D])\b", a.upper())
    return m.group(1) if m else ""

def _probe_duration_bucket(video_path: Path) -> str:
    try:
        import av
    except ImportError:
        return ""
    try:
        with av.open(str(video_path)) as container:
            duration = None
            if container.duration is not None:
                duration = container.duration / av.time_base
            else:
                stream = container.streams.video[0]
                if stream.duration is not None and stream.time_base is not None:
                    duration = float(stream.duration * stream.time_base)
    except Exception:
        return ""
    if not duration:
        return ""
    return f"{int(min(_DURATION_BUCKETS, key=lambda b: abs(b - duration)))}s"

def _blob(cell: Any, field: str) -> bytes | None:
    if not isinstance(cell, dict):
        return None
    value = cell.get(field)
    return value if isinstance(value, (bytes, bytearray)) else None

def convert(src: Path, dst: Path, probe_duration: bool, batch_size: int) -> None:
    import pyarrow.parquet as pq
    shards = sorted(src.glob("data/*.parquet")) or sorted(src.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards found under {src}")
    videos_root = dst / "Videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    qa_rows: list[dict[str, Any]] = []
    seen_media: set[str] = set()
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                video_id = str(row.get("video_id") or "").strip()
                if not video_id:
                    continue
                if video_id not in seen_media:
                    out_dir = videos_root / video_id
                    out_dir.mkdir(parents=True, exist_ok=True)
                    mp4 = out_dir / f"{video_id}_video.mp4"
                    wav = out_dir / f"{video_id}_audio.wav"
                    video_bytes = _blob(row.get("video"), "bytes")
                    audio_bytes = _blob(row.get("audio"), "bytes")
                    if video_bytes and not mp4.exists():
                        mp4.write_bytes(video_bytes)
                    if audio_bytes and not wav.exists():
                        wav.write_bytes(audio_bytes)
                    seen_media.add(video_id)
                question = str(row.get("question") or "").strip()
                candidates = [str(c) for c in (row.get("candidates") or [])]
                letter = _answer_letter(str(row.get("answer") or ""), candidates)
                entry: dict[str, Any] = {
                    "Question": question,
                    "Choice": candidates,
                    "Answer": letter,
                    "video_id": video_id,
                    "Type": "",
                    "video_category": "",
                    "video_duration": "",
                }
                qa_rows.append(entry)
    if probe_duration:
        cache: dict[str, str] = {}
        for entry in qa_rows:
            if entry["video_duration"]:
                continue
            vid = entry["video_id"]
            if vid not in cache:
                cache[vid] = _probe_duration_bucket(videos_root / vid / f"{vid}_video.mp4")
            entry["video_duration"] = cache[vid]
    qa_path = dst / "qa.json"
    qa_path.write_text(json.dumps(qa_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {qa_path} ({len(qa_rows)} rows)", file=sys.stderr)
    print(f"Wrote {videos_root} ({len(seen_media)} videos)", file=sys.stderr)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--probe-duration", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    convert(args.src, args.dst, args.probe_duration, args.batch_size)

if __name__ == "__main__":
    main()
