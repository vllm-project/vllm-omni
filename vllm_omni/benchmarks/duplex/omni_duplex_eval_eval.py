# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Evaluate and summarize generated Omni-DuplexEval artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .omni_duplex_eval_clock import normalize_response_items, validate_clock
from .omni_duplex_eval_dataset import DuplexSample
from .omni_duplex_eval_judge import DuplexJudge
from .omni_duplex_eval_media import extract_jpeg, materialize_media, video_duration
from .omni_duplex_eval_metrics import (
    PROTOCOL_PIN,
    build_content_prompt,
    build_reminder_prompt,
    build_temporal_prompt,
    parse_judge_json,
    reminder_window,
    summarize_pr_results,
    summarize_temporal_results,
    temporal_window,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _text(items: list[dict[str, Any]]) -> str:
    return " ".join(str(item.get("sentence", item.get("text", ""))).strip() for item in items).strip()


def evaluate_sample(
    sample: DuplexSample,
    response_path: Path,
    score_path: Path,
    judge: DuplexJudge,
    *,
    judge_fps: int = 2,
    judge_video_mode: str = "video_url",
    window_size: float = 10.0,
    allow_invalid_clock: bool = False,
) -> dict[str, Any]:
    raw = _read(response_path)
    meta = (
        _read(response_path.with_name(response_path.stem + ".meta.json"))
        if response_path.with_name(response_path.stem + ".meta.json").exists()
        else {}
    )
    validate_clock(meta, allow_invalid=allow_invalid_clock)
    items = normalize_response_items(raw)
    result: dict[str, Any] = {
        "id": sample.id,
        "family": sample.family,
        "task_type": sample.task_type,
        "protocol_pin": PROTOCOL_PIN,
        "response_meta": meta,
        "judge_model": getattr(judge, "model", None),
        "judge_video_mode": judge_video_mode,
    }
    if sample.family == "rtd":
        video_path = materialize_media(sample.video, score_path.parent / ".media", sample.id, ".mp4")
        duration = sample.video_duration or video_duration(video_path)
        temporal_rows = []
        for item in items:
            window = temporal_window(item["start"], item["end"], duration)
            if not window:
                temporal_rows.append(
                    {
                        **item,
                        "sentence_start": item["start"],
                        "sentence_end": item["end"],
                        "sentence_duration": max(0.0, item["end"] - item["start"]),
                        "error": "No valid temporal window",
                        "temporal_score": 0,
                        "is_relevant": 0,
                    }
                )
                continue
            frames = _extract_frames(video_path, _times(*window, fps=judge_fps))
            parsed = parse_judge_json(
                judge.temporal(build_temporal_prompt(*window, item["sentence"], sample.question_text), frames)
            )
            temporal_rows.append(
                {
                    **item,
                    "sentence_start": item["start"],
                    "sentence_end": item["end"],
                    "sentence_duration": max(0.0, item["end"] - item["start"]),
                    "window_start": window[0],
                    "window_end": window[1],
                    "error": None,
                    **parsed,
                }
            )
        content_frames = None
        if judge_video_mode == "frame-sample":
            content_frames = _extract_frames(video_path, _content_frame_times(duration))
        content = parse_judge_json(
            judge.content(
                build_content_prompt(_text(items), sample.question_text, [sample.answer1, sample.answer2]),
                video_path,
                frames=content_frames,
                mode=judge_video_mode,
            )
        )
        if content_frames is not None:
            content["frame_count"] = len(content_frames)
        result.update(
            {
                "temporal": {"sentences": temporal_rows, "summary": summarize_temporal_results(temporal_rows)},
                "content": content,
            }
        )
    else:
        if sample.task_type is None:
            raise ValueError(f"missing proactive-reminder task type for split {sample.split!r}")
        task = sample.task_type
        times = _reminder_times(sample)
        event_rows = []
        if task == "correction" or not times:
            response = _text(items)
            parsed = parse_judge_json(
                judge.chat(build_reminder_prompt(sample.question_text, response, task, sample.answer1))
            )
            event_rows.append({"response_segment": response, **parsed})
        else:
            for start in times:
                lo, hi = reminder_window(start, window_size)
                response = " ".join(item["sentence"] for item in items if lo <= item["start"] <= hi)
                if not response:
                    event_rows.append({"start_time": start, "response_segment": "", "success_score": 0})
                else:
                    parsed = parse_judge_json(
                        judge.chat(build_reminder_prompt(sample.question_text, response, task, sample.answer1))
                    )
                    event_rows.append({"start_time": start, "response_segment": response, **parsed})
        scores = [int(row.get("success_score", 0)) for row in event_rows]
        result["pr"] = {
            "events": event_rows,
            "all_success": bool(scores) and all(score == 1 for score in scores),
            "total_score": int(bool(scores) and all(score == 1 for score in scores)),
        }
    _write(score_path, result)
    return result


def _times(start: float, end: float, *, fps: int) -> list[float]:
    count = max(1, int((end - start) * fps))
    return (
        [start + (end - start) / 2] if count == 1 else [start + i * (end - start) / (count - 1) for i in range(count)]
    )


def _extract_frames(path: str | Path, timestamps: list[float]) -> list[bytes]:
    """Extract only decodable JPEG frames; clips can end on non-keyframes."""
    frames = []
    for timestamp in timestamps:
        try:
            frame = extract_jpeg(path, timestamp=timestamp)
        except (OSError, subprocess.CalledProcessError, ValueError):
            continue
        if frame.startswith(b"\xff\xd8"):
            frames.append(frame)
    return frames


def _content_frame_times(duration: float) -> list[float]:
    """Sample the full video at approximately one frame every three seconds."""
    if duration <= 0:
        return []
    times = []
    timestamp = 0.0
    while timestamp < duration:
        times.append(min(timestamp, max(0.0, duration - 0.01)))
        timestamp += 3.0
    # Qwen2.5-VL has a finite per-request image-token budget. Preserve
    # coverage while bounding long clips to a safely portable frame count.
    if len(times) > 16:
        stride = (len(times) - 1) / 15
        times = [times[round(i * stride)] for i in range(16)]
    return times


def _reminder_times(sample: DuplexSample) -> list[float]:
    values = []
    for value in (sample.reminder1, sample.reminder2):
        if isinstance(value, dict):
            value = value.get("start", value.get("time"))
        try:
            if value is not None:
                values.append(float(value))
        except (TypeError, ValueError):
            pass
    return values


def summarize_scores(score_root: str | Path) -> dict[str, Any]:
    rows = []
    for path in sorted(Path(score_root).rglob("*.json")):
        if not path.name.endswith("_summary.json"):
            row = _read(path)
            row.setdefault("split", path.parent.name)
            rows.append(row)
    rtd_temporal = [
        row["temporal"]["summary"] | {"content_score": row.get("content", {}).get("content_score", 0.0)}
        for row in rows
        if "temporal" in row
    ]
    pr = [{"task_type": row.get("task_type"), **row.get("pr", {})} for row in rows if "pr" in row]
    result = {"protocol_pin": PROTOCOL_PIN, "samples": len(rows)}
    if rtd_temporal:
        by_task: dict[str, list[dict[str, float]]] = {}
        for row in rows:
            if "temporal" in row:
                by_task.setdefault(str(row.get("split", row.get("task_type", "unknown"))), []).append(
                    {
                        "content": float(row.get("content", {}).get("content_score", 0.0)),
                        "temporal": float(row["temporal"]["summary"].get("avg_temporal_score", 0.0)),
                    }
                )
        result["rtd"] = {
            "mean_content_score": sum(item["content_score"] for item in rtd_temporal) / len(rtd_temporal),
            "mean_avg_temporal_score": sum(item["avg_temporal_score"] for item in rtd_temporal) / len(rtd_temporal),
            "by_task": {
                task: {
                    "mean_content_score": sum(item["content"] for item in values) / len(values),
                    "mean_avg_temporal_score": sum(item["temporal"] for item in values) / len(values),
                }
                for task, values in by_task.items()
            },
        }
    if pr:
        result["pr"] = summarize_pr_results(pr)
    return result
