"""Video-MME multiple-choice accuracy scoring for vLLM-Omni bench serve.

Answer extraction ports ``extract_characters_regex`` from the official Video-MME
script (MME-Benchmarks ``eval_your_results.py``, also used by lmms-eval).

**Denominator:** responses where no A–D can be extracted are counted as wrong, which
matches lmms-eval and OmniEvalKit (the numbers this bench is compared against). The
official standalone script instead drops them from ``answered``; ``videomme_parse_failed``
is reported so that view can be recovered.
"""

from __future__ import annotations

import re
from typing import Any

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput

from vllm_omni.benchmarks.data_modules.videomme_dataset import VideoMMESampleRequest

_VALID = frozenset("ABCD")

VIDEOMME_DURATION_KEYS: tuple[str, ...] = ("short", "medium", "long")

#: Report dimension -> ``VideoMMESampleRequest`` attribute (official Video-MME breakdowns).
_DIMENSIONS: dict[str, str] = {
    "duration": "videomme_duration",
    "domain": "videomme_domain",
    "sub_category": "videomme_sub_category",
    "task": "videomme_task_type",
}

# Official prefix list, with the two accidental string concatenations in the upstream
# tuple (missing commas after "The best option is" / "Best answer:") split apart.
_ANSWER_PREFIXES = (
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "Answer:",
    "Option:",
    "The correct answer",
    "The correct option",
)


def extract_characters_regex(text: str | None) -> str | None:
    """Port of official Video-MME / lmms-eval ``extract_characters_regex``."""
    if not text:
        return None
    s = str(text).strip()
    for prefix in _ANSWER_PREFIXES:
        s = s.replace(prefix, "")
    if len(s.split()) > 10 and not re.search(r"[ABCD]", s):
        return None
    match = re.search(r"[ABCD]", s)
    return match.group(0) if match else None


def normalize_gold_answer(gold: str) -> str | None:
    g = (gold or "").strip().upper()
    if len(g) == 1 and g in _VALID:
        return g
    m = re.search(r"([ABCD])\b", g)
    return m.group(1).upper() if m else None


def _bucket(req: VideoMMESampleRequest, dimension: str) -> str:
    value = (getattr(req, _DIMENSIONS[dimension]) or "").strip()
    if dimension == "duration":
        return value.lower()
    return value or "unknown"


def compute_videomme_accuracy_metrics(
    input_requests: list[Any],
    outputs: list[RequestFuncOutput],
    *,
    include_per_item: bool = False,
) -> dict[str, Any] | None:
    """If all requests are :class:`VideoMMESampleRequest`, compute accuracy stats.

    Rows without a gold answer are skipped (``no_gold``). Failed HTTP requests are
    excluded from ``videomme_accuracy`` and counted in
    ``videomme_accuracy_incl_http_fail`` instead.
    """
    if not input_requests or len(input_requests) != len(outputs):
        return None
    if not all(isinstance(r, VideoMMESampleRequest) for r in input_requests):
        return None

    stats: dict[str, dict[str, dict[str, int]]] = {
        dim: ({k: {"correct": 0, "total": 0} for k in VIDEOMME_DURATION_KEYS} if dim == "duration" else {})
        for dim in _DIMENSIONS
    }
    items: list[dict[str, Any]] = []
    correct = evaluated = no_gold = request_failed = parse_failed = 0

    for req, out in zip(input_requests, outputs, strict=True):
        assert isinstance(req, VideoMMESampleRequest)
        gold_raw = (req.videomme_gold_answer or "").strip()
        buckets = {dim: _bucket(req, dim) for dim in _DIMENSIONS}
        row: dict[str, Any] = {
            "request_id": req.request_id,
            "video_id": req.videomme_video_id,
            "question_id": req.videomme_question_id,
            **buckets,
        }

        if not gold_raw:
            no_gold += 1
            items.append({**row, "skipped": True, "reason": "no_gold"})
            continue

        evaluated += 1
        gold_norm = normalize_gold_answer(gold_raw)
        pred = extract_characters_regex(out.generated_text) if out.success else None
        is_correct = bool(pred) and pred.upper() == (gold_norm or gold_raw).upper()

        if not out.success:
            request_failed += 1
        else:
            if pred is None:
                parse_failed += 1
            # Only successful responses enter the per-dimension denominators, mirroring
            # the overall videomme_accuracy definition.
            for dim, key in buckets.items():
                bucket = stats[dim].setdefault(key, {"correct": 0, "total": 0})
                bucket["total"] += 1
                bucket["correct"] += int(is_correct)

        if is_correct:
            correct += 1

        items.append(
            {
                **row,
                "gold": gold_raw,
                "gold_normalized": gold_norm,
                "predicted": pred,
                "correct": is_correct,
                "parse_failed": out.success and pred is None,
                **({"error": (out.error or "")[:500]} if not out.success else {}),
            }
        )

    evaluated_ok = evaluated - request_failed
    result: dict[str, Any] = {
        "videomme_accuracy": (correct / evaluated_ok) if evaluated_ok else None,
        "videomme_accuracy_incl_http_fail": (correct / evaluated) if evaluated else None,
        "videomme_correct": correct,
        "videomme_evaluated": evaluated,
        "videomme_evaluated_ok": evaluated_ok,
        "videomme_no_gold": no_gold,
        "videomme_request_failed": request_failed,
        "videomme_parse_failed": parse_failed,
    }
    for dim, per_bucket in stats.items():
        result[f"videomme_per_{dim}"] = {k: dict(v) for k, v in per_bucket.items()}
        result[f"videomme_per_{dim}_accuracy"] = {
            k: (v["correct"] / v["total"]) if v["total"] else None for k, v in per_bucket.items()
        }
    if include_per_item:
        result["videomme_eval_items"] = items
    return result


def print_videomme_accuracy_summary(metrics: dict[str, Any]) -> None:
    """Pretty-print the Video-MME accuracy block (stdout)."""
    acc = metrics.get("videomme_accuracy")
    if acc is None and not metrics.get("videomme_evaluated", 0):
        return

    print("{s:{c}^{n}}".format(s=" Video-MME accuracy (MCQ) ", n=50, c="="))
    ok = int(metrics.get("videomme_evaluated_ok", 0) or 0)
    if ok and acc is not None:
        print(f"Overall Accuracy: {metrics.get('videomme_correct', 0)}/{ok} = {acc:.2%}")
    else:
        print("Overall Accuracy: 0/0 = N/A (no successful HTTP responses)")
    for label, key in (
        ("Submitted (gold present):", "videomme_evaluated"),
        ("Successful HTTP (denominator):", "videomme_evaluated_ok"),
        ("Correct:", "videomme_correct"),
        ("Skipped (no gold):", "videomme_no_gold"),
        ("HTTP failed:", "videomme_request_failed"),
        ("Parsed OK but no A-D found:", "videomme_parse_failed"),
    ):
        print(f"{label:<40} {metrics.get(key, 0):<10}")

    for dim, title in (
        ("duration", "Duration"),
        ("domain", "Domain"),
        ("sub_category", "Sub Category"),
        ("task", "Task Type"),
    ):
        per_acc = metrics.get(f"videomme_per_{dim}_accuracy") or {}
        if not per_acc:
            continue
        counts = metrics.get(f"videomme_per_{dim}") or {}
        names = VIDEOMME_DURATION_KEYS if dim == "duration" else sorted(per_acc)
        print(f"\n--- Accuracy by {title} ---")
        for name in names:
            st = counts.get(name) or {}
            total = int(st.get("total", 0) or 0)
            value = per_acc.get(name)
            if total and value is not None:
                print(f"{name}: {int(st.get('correct', 0))}/{total} = {value:.2%}")
            else:
                print(f"{name}: 0/0 = N/A")
    print("=" * 50)
