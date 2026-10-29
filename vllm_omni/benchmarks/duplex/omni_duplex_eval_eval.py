# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Evaluate and summarize generated Omni-DuplexEval artifacts."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from .omni_duplex_eval_clock import (
    has_response_chunks,
    normalize_response_chunks,
    normalize_response_items,
    validate_clock,
)
from .omni_duplex_eval_dataset import DuplexSample
from .omni_duplex_eval_judge import DuplexJudge, JudgeResponseError
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

logger = logging.getLogger(__name__)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.part")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _text(items: list[dict[str, Any]]) -> str:
    return " ".join(str(item.get("sentence", item.get("text", ""))).strip() for item in items).strip()


def _gt_references(answers: list[object]) -> list[str]:
    """Mirror the official evaluator's reference handling.

    Drop empty strings (order-preserving, strip-based) then deduplicate while
    keeping the first occurrence, as ``non_empty_strings`` followed by
    ``load_reference_texts`` does in OpenBMB/Omni-DuplexEval. Numbering an
    empty or repeated answer would deviate from the pinned content prompt.
    """
    references = []
    for answer in answers:
        if answer is None:
            continue
        text = str(answer).strip()
        if text:
            references.append(text)
    return list(dict.fromkeys(references))


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
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if score_path.exists():
        # A failed retry must not leave an older score eligible for aggregation.
        # Keep the .previous suffix: summaries and accuracy CI scan *.json.
        score_path.replace(score_path.with_name(f"{score_path.name}.{uuid4().hex}.previous"))
    trace_path = score_path.with_suffix(".judge.jsonl")
    trace_path.write_text("", encoding="utf-8")

    def judge_result(call: Callable[[], str], required: tuple[str, ...]) -> dict[str, Any]:
        try:
            raw_response = call()
        except JudgeResponseError as exc:
            with trace_path.open("a", encoding="utf-8") as trace:
                trace.write(
                    json.dumps(
                        {"raw_response": exc.raw_response, "finish_reason": exc.finish_reason, "error": str(exc)},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            raise
        with trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps({"raw_response": raw_response}, ensure_ascii=False) + "\n")
        return parse_judge_json(raw_response, required=required)

    raw = _read(response_path)
    meta = (
        _read(response_path.with_name(response_path.stem + ".meta.json"))
        if response_path.with_name(response_path.stem + ".meta.json").exists()
        else {}
    )
    validate_clock(meta, allow_invalid=allow_invalid_clock)
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
        items = normalize_response_items(raw)
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
            parsed = judge_result(
                lambda: judge.temporal(build_temporal_prompt(*window, item["sentence"], sample.question_text), frames),
                ("temporal_score", "is_relevant"),
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
        content_prompt = build_content_prompt(
            _text(items), sample.question_text, _gt_references([sample.answer1, sample.answer2])
        )
        try:
            content = judge_result(
                lambda: judge.content(content_prompt, video_path, frames=content_frames, mode=judge_video_mode),
                ("content_score",),
            )
        except Exception as exc:
            # A long clip can overflow the judge model's context window when the
            # whole video is passed through (video_url mode): the judge answers
            # HTTP 400 "Input length (83158) exceeds model's maximum context
            # length (65536)" (#9089).  Degrade that single content call to the
            # bounded frame-sample path instead of failing the whole run.
            if judge_video_mode != "video_url" or not _is_judge_context_length_error(exc):
                raise
            logger.warning(
                "Judge rejected the full video for %s (context length exceeded); "
                "retrying the content prompt in frame-sample mode",
                sample.id,
            )
            content_frames = _extract_frames(video_path, _content_frame_times(duration))
            if not content_frames:
                # Nothing to fall back to: keep the original judge error.
                raise
            content = judge_result(
                lambda: judge.content(content_prompt, video_path, frames=content_frames, mode="frame-sample"),
                ("content_score",),
            )
            content["judge_input_fallback"] = "frame-sample"
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
        chunks = normalize_response_chunks(raw)
        # Chunks keep their own boundary whitespace; legacy sentence items are stripped.
        separator = "" if has_response_chunks(raw) else " "
        event_rows = []
        if task == "correction" or not times:
            response = separator.join(chunk["text"] for chunk in chunks).strip()
            parsed = judge_result(
                lambda: judge.chat(build_reminder_prompt(sample.question_text, response, task, sample.answer1)),
                ("success_score",),
            )
            event_rows.append({"response_segment": response, **parsed})
        else:
            for start in times:
                lo, hi = reminder_window(start, window_size)
                response = separator.join(
                    chunk["text"] for chunk in chunks if lo <= chunk["current_time"] <= hi
                ).strip()
                if not response:
                    event_rows.append({"start_time": start, "response_segment": "", "success_score": 0})
                else:
                    parsed = judge_result(
                        lambda: judge.chat(build_reminder_prompt(sample.question_text, response, task, sample.answer1)),
                        ("success_score",),
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


def _is_judge_context_length_error(exc: BaseException) -> bool:
    """True when the judge rejected the request because its input was too long.

    The judge client raises ``requests.HTTPError`` with the HTTP response
    attached, e.g. HTTP 400
    ``Input length (83158) exceeds model's maximum context length (65536)``.
    """
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) != 400:
        return False
    text = str(getattr(response, "text", "") or "").lower()
    if "context length" in text:
        return True
    # Other OpenAI-compatible judges phrase the same rejection differently.
    return "input length" in text and "exceed" in text


def _extract_frames(path: str | Path, timestamps: list[float]) -> list[bytes]:
    """Extract only decodable JPEG frames; clips can end on non-keyframes."""
    frames = []
    for timestamp in timestamps:
        try:
            frame = extract_jpeg(path, timestamp=timestamp)
        except (OSError, ValueError):
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
