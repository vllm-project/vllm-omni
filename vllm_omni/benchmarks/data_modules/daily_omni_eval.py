"""Daily-Omni multiple-choice accuracy scoring for vLLM-Omni bench serve.

Compares model ``generated_text`` to dataset ``Answer`` (A/B/C/D).

**Alignment with open-source** (`Lliar-liar/Daily-Omni` ``test_model/.../testmodel.py``):

- Answer extraction defaults to the same rules as ``extract_choice_letter`` (strip after an
  ``assistant`` marker, then leading ``A``–``D``, else first ``\\b[A-D]\\b``). Set env
  ``DAILY_OMNI_EXTRACT_MODE=relaxed`` to use the older vLLM-Omni heuristics (last ``answer:``,
  tail scan, etc.).
- Overall accuracy comparable to the official script uses **successful HTTP responses only** as
  the denominator (their ``valid_questions = total - failed`` excludes inference / I/O skips).
  We also report ``daily_omni_accuracy_incl_http_fail`` where each failed request counts as a
  wrong answer in the denominator (stricter throughput-bench view).
- **By video length:** mirrors upstream ``--- Accuracy by Video Duration ---`` for ``30s`` /
  ``60s`` (``qa.json`` ``video_duration``): ``daily_omni_per_duration*`` metrics and a printed block.
- **By video category:** mirrors ``--- Accuracy by Video Category ---`` using ``video_category``
  from ``qa.json`` (``daily_omni_per_category*``; empty category is bucketed as ``unknown``).
- **Correctness:** uses the same ``evaluate_answer`` rule as upstream (truthy extracted letter vs
  raw ``Answer`` string, both ``strip().upper()``). Rows with empty ``Answer`` are skipped
  (``no_gold``), matching missing-field skips in the official loop.
"""

from __future__ import annotations

import os
import re
from typing import Any

from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput

from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniSampleRequest

_VALID = frozenset("ABCD")

# Official ``testmodel.py`` buckets (``qa.json`` ``video_duration``).
DAILY_OMNI_DURATION_KEYS: tuple[str, ...] = ("30s", "60s")


def extract_choice_letter_official(text: str | None) -> str | None:
    """Port of Daily-Omni ``extract_choice_letter`` (first A–D, assistant-tail semantics)."""
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    match = re.search(r"assistant\s*([\s\S]*)$", raw, flags=re.IGNORECASE)
    candidate = match.group(1).strip() if match else raw
    direct = re.match(r"(?i)^\s*([A-D])(?:[\s\.\):：]|$)", candidate)
    if direct:
        return direct.group(1).upper()
    fallback = re.search(r"\b([A-D])\b", candidate.upper())
    if fallback:
        return fallback.group(1)
    return None


def evaluate_answer_official(model_answer: str | None, correct_answer: str) -> bool:
    """Port of Daily-Omni ``evaluate_answer`` (strict string match after strip/upper)."""
    if not model_answer:
        return False
    return model_answer.strip().upper() == (correct_answer or "").strip().upper()


def normalize_gold_answer(gold: str) -> str | None:
    """Best-effort single letter from ``Answer`` (for ``gold_normalized`` in saved items only)."""
    g = (gold or "").strip().upper()
    if len(g) == 1 and g in _VALID:
        return g
    m = re.search(r"([ABCD])\b", g)
    if m:
        return m.group(1).upper()
    return None


def _extract_predicted_choice_relaxed(text: str) -> str | None:
    """Legacy vLLM-Omni heuristics (last ``answer:`` patterns, tail scan)."""
    if not text or not str(text).strip():
        return None
    t = str(text).strip()

    strong_patterns = [
        r"(?i)\*\*answer\*\*\s*[:：]?\s*\(?([ABCD])\)?",
        r"(?i)\banswer\s*[:：]?\s*\(?([ABCD])\)?",
        r"(?i)\bfinal\s+answer\s*[:：]?\s*\(?([ABCD])\)?",
        r"(?i)\bcorrect\s+(?:answer|option)\s*[:：]?\s*\(?([ABCD])\)?",
        r"(?i)\bthe\s+(?:correct\s+)?option\s+(?:is|would\s+be)\s*\(?([ABCD])\)?",
        r"(?i)\bI\s+(?:would\s+)?(?:choose|select|pick)\s*\(?([ABCD])\)?",
    ]
    last_letter: str | None = None
    for pat in strong_patterns:
        for m in re.finditer(pat, t):
            last_letter = m.group(1).upper()
    if last_letter:
        return last_letter

    # Weaker phrases: first match can be spurious; still prefer last occurrence.
    weak_patterns = [
        r"(?i)\boption\s*[:：]?\s*\(?([ABCD])\)?",
        r"(?i)\bchoice\s*[:：]?\s*\(?([ABCD])\)?",
    ]
    for pat in weak_patterns:
        for m in re.finditer(pat, t):
            last_letter = m.group(1).upper()
    if last_letter:
        return last_letter

    paren = list(re.finditer(r"\(([ABCD])\)", t))
    if paren:
        return paren[-1].group(1).upper()

    # First line sometimes is just "B" or "B." — allow if whole output is short
    one_line = t.split("\n", 1)[0].strip()
    if len(t) < 120 and len(one_line) <= 6:
        m0 = re.match(r"^([ABCD])\s*[.:\)]?\s*$", one_line, re.I)
        if m0:
            return m0.group(1).upper()

    # Tail-only: avoids matching echoed "A. ..." option blocks at the start
    tail_len = min(500, len(t))
    tail = t[-tail_len:]
    # ``\b`` after the letter avoids "Because"/"Definitely" false positives
    m = re.search(r"(?:^|[^\w])([ABCD])\b", tail, re.I)
    if m:
        return m.group(1).upper()

    return None


def extract_predicted_choice(text: str | None) -> str | None:
    """Parse model output to A–D (official Daily-Omni rules by default)."""
    if not text or not str(text).strip():
        return None
    mode = os.environ.get("DAILY_OMNI_EXTRACT_MODE", "official").strip().lower()
    if mode in ("relaxed", "heuristic", "legacy"):
        return _extract_predicted_choice_relaxed(str(text))
    return extract_choice_letter_official(text)


def compute_daily_omni_accuracy_metrics(
    input_requests: list[Any],
    outputs: list[RequestFuncOutput],
    *,
    include_per_item: bool = False,
) -> dict[str, Any] | None:
    """If all requests are :class:`DailyOmniSampleRequest`, compute accuracy stats.

    Rows with empty ``Answer`` (after strip) are skipped as ``no_gold``, like upstream missing
    ``correct_answer``.

    **Denominators:** The open-source script excludes items that hit inference / I/O failures
    from ``valid_questions``; we mirror that with ``daily_omni_accuracy`` (= correct /
    successful responses). Failed HTTP requests are also tracked and used in
    ``daily_omni_accuracy_incl_http_fail`` (each failure counts as incorrect in the
    denominator).
    """
    if not input_requests or len(input_requests) != len(outputs):
        return None
    if not all(isinstance(r, DailyOmniSampleRequest) for r in input_requests):
        return None

    # total / correct: all rows with gold (incl. HTTP fail in total)
    # total_ok / correct_ok: successful HTTP only (GitHub-style per-type denominator)
    per_task: dict[str, dict[str, int]] = {}
    per_category: dict[str, dict[str, int]] = {}
    per_duration: dict[str, dict[str, int]] = {
        k: {"correct": 0, "total": 0, "correct_ok": 0, "total_ok": 0} for k in DAILY_OMNI_DURATION_KEYS
    }
    items: list[dict[str, Any]] = []
    correct = 0
    evaluated = 0
    no_gold = 0
    request_failed = 0
    parse_failed = 0  # success but could not extract A–D

    for req, out in zip(input_requests, outputs, strict=True):
        assert isinstance(req, DailyOmniSampleRequest)
        gold_raw = (req.daily_omni_gold_answer or "").strip()
        gold_norm = normalize_gold_answer(req.daily_omni_gold_answer)
        tt = (req.daily_omni_task_type or "unknown").strip() or "unknown"
        dur_key = (req.daily_omni_video_duration or "").strip()
        dur_active = dur_key in per_duration
        cat_key = (req.daily_omni_video_category or "").strip() or "unknown"
        if tt not in per_task:
            per_task[tt] = {"correct": 0, "total": 0, "correct_ok": 0, "total_ok": 0}
        if cat_key not in per_category:
            per_category[cat_key] = {"correct": 0, "total": 0, "correct_ok": 0, "total_ok": 0}

        if not gold_raw:
            no_gold += 1
            items.append(
                {
                    "request_id": req.request_id,
                    "skipped": True,
                    "reason": "no_gold",
                    "task_type": tt,
                    "video_id": req.daily_omni_video_id,
                    "video_duration": dur_key or None,
                    "video_category": cat_key if cat_key != "unknown" else None,
                }
            )
            continue

        if not out.success:
            request_failed += 1
            evaluated += 1
            per_task[tt]["total"] += 1
            per_category[cat_key]["total"] += 1
            if dur_active:
                per_duration[dur_key]["total"] += 1
            # GitHub: failed inference not in valid_questions — do not increment total_ok
            items.append(
                {
                    "request_id": req.request_id,
                    "gold": gold_raw,
                    "gold_normalized": gold_norm,
                    "predicted": None,
                    "correct": False,
                    "task_type": tt,
                    "video_id": req.daily_omni_video_id,
                    "video_duration": dur_key or None,
                    "video_category": cat_key if cat_key != "unknown" else None,
                    "error": (out.error or "")[:500],
                }
            )
            continue

        pred = extract_predicted_choice(out.generated_text)
        evaluated += 1
        per_task[tt]["total"] += 1
        per_task[tt]["total_ok"] += 1
        per_category[cat_key]["total"] += 1
        per_category[cat_key]["total_ok"] += 1
        if dur_active:
            per_duration[dur_key]["total"] += 1
            per_duration[dur_key]["total_ok"] += 1
        if pred is None:
            parse_failed += 1
        is_correct = evaluate_answer_official(pred, req.daily_omni_gold_answer)
        if is_correct:
            correct += 1
            per_task[tt]["correct"] += 1
            per_task[tt]["correct_ok"] += 1
            per_category[cat_key]["correct"] += 1
            per_category[cat_key]["correct_ok"] += 1
            if dur_active:
                per_duration[dur_key]["correct"] += 1
                per_duration[dur_key]["correct_ok"] += 1

        items.append(
            {
                "request_id": req.request_id,
                "gold": gold_raw,
                "gold_normalized": gold_norm,
                "predicted": pred,
                "correct": is_correct,
                "parse_failed": pred is None,
                "task_type": tt,
                "video_id": req.daily_omni_video_id,
                "video_duration": dur_key or None,
                "video_category": cat_key if cat_key != "unknown" else None,
            }
        )

    evaluated_ok = evaluated - request_failed
    accuracy_github = (correct / evaluated_ok) if evaluated_ok else None
    accuracy_incl_fail = (correct / evaluated) if evaluated else None

    per_task_accuracy: dict[str, float | None] = {}
    per_task_accuracy_github: dict[str, float | None] = {}
    for name, st in per_task.items():
        tot = st["total"]
        per_task_accuracy[name] = (st["correct"] / tot) if tot else None
        tok = st["total_ok"]
        per_task_accuracy_github[name] = (st["correct_ok"] / tok) if tok else None

    per_category_accuracy: dict[str, float | None] = {}
    per_category_accuracy_github: dict[str, float | None] = {}
    for name, st in per_category.items():
        tot = st["total"]
        per_category_accuracy[name] = (st["correct"] / tot) if tot else None
        tok = st["total_ok"]
        per_category_accuracy_github[name] = (st["correct_ok"] / tok) if tok else None

    per_duration_accuracy: dict[str, float | None] = {}
    per_duration_accuracy_github: dict[str, float | None] = {}
    for name, st in per_duration.items():
        tot = st["total"]
        per_duration_accuracy[name] = (st["correct"] / tot) if tot else None
        tok = st["total_ok"]
        per_duration_accuracy_github[name] = (st["correct_ok"] / tok) if tok else None

    out: dict[str, Any] = {
        # Comparable to GitHub testmodel.py: correct / successful inferences
        "daily_omni_accuracy": accuracy_github,
        "daily_omni_accuracy_incl_http_fail": accuracy_incl_fail,
        "daily_omni_correct": correct,
        "daily_omni_evaluated": evaluated,
        "daily_omni_evaluated_ok": evaluated_ok,
        "daily_omni_no_gold": no_gold,
        "daily_omni_request_failed": request_failed,
        "daily_omni_parse_failed": parse_failed,
        "daily_omni_per_task": {k: dict(v) for k, v in per_task.items()},
        "daily_omni_per_task_accuracy": per_task_accuracy,
        "daily_omni_per_task_accuracy_github_style": per_task_accuracy_github,
        "daily_omni_per_category": {k: dict(v) for k, v in per_category.items()},
        "daily_omni_per_category_accuracy": per_category_accuracy,
        "daily_omni_per_category_accuracy_github_style": per_category_accuracy_github,
        "daily_omni_per_duration": {k: dict(v) for k, v in per_duration.items()},
        "daily_omni_per_duration_accuracy": per_duration_accuracy,
        "daily_omni_per_duration_accuracy_github_style": per_duration_accuracy_github,
    }
    if include_per_item:
        out["daily_omni_eval_items"] = items
    return out


def print_daily_omni_accuracy_summary(metrics: dict[str, Any]) -> None:
    """Pretty-print accuracy block (stdout)."""
    acc = metrics.get("daily_omni_accuracy")
    acc_fail = metrics.get("daily_omni_accuracy_incl_http_fail")
    if acc is None and acc_fail is None and metrics.get("daily_omni_evaluated", 0) == 0:
        return
    print("{s:{c}^{n}}".format(s=" Daily-Omni accuracy (MCQ) ", n=50, c="="))
    ok = int(metrics.get("daily_omni_evaluated_ok", 0) or 0)
    cor = int(metrics.get("daily_omni_correct", 0) or 0)
    if ok > 0 and acc is not None:
        print(f"Overall Accuracy: {cor}/{ok} = {acc:.2%}")
    elif int(metrics.get("daily_omni_evaluated", 0) or 0) > 0:
        print("Overall Accuracy: 0/0 = N/A (no successful HTTP responses)")
    print(
        "{:<40} {:<10}".format(
            "Submitted (gold present):",
            metrics.get("daily_omni_evaluated", 0),
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Successful HTTP (GitHub denom.):",
            metrics.get("daily_omni_evaluated_ok", 0),
        )
    )
    print("{:<40} {:<10}".format("Correct:", metrics.get("daily_omni_correct", 0)))
    if acc is not None:
        print("{:<40} {:<10.4f}".format("Accuracy (ratio, same as above):", acc))
    if acc_fail is not None and metrics.get("daily_omni_request_failed", 0):
        print(
            "{:<40} {:<10.4f}".format(
                "Accuracy (incl. HTTP as wrong):",
                acc_fail,
            )
        )
    print("{:<40} {:<10}".format("Skipped (no gold):", metrics.get("daily_omni_no_gold", 0)))
    print(
        "{:<40} {:<10}".format(
            "HTTP failed (excl. from GitHub acc.):",
            metrics.get("daily_omni_request_failed", 0),
        )
    )
    print(
        "{:<40} {:<10}".format(
            "Parsed OK but no A–D found:",
            metrics.get("daily_omni_parse_failed", 0),
        )
    )
    pt = metrics.get("daily_omni_per_task") or {}
    pta = metrics.get("daily_omni_per_task_accuracy_github_style") or {}
    if pta:
        print("\n--- Accuracy by QA Type ---")
        for name in sorted(pta.keys()):
            a = pta[name]
            st = pt.get(name) or {}
            tok = int(st.get("total_ok", 0) or 0)
            cok = int(st.get("correct_ok", 0) or 0)
            if tok and a is not None:
                print(f"{name}: {cok}/{tok} = {a:.2%}")
            else:
                print(f"{name}: 0/0 = N/A")

    pc = metrics.get("daily_omni_per_category") or {}
    ptc = metrics.get("daily_omni_per_category_accuracy_github_style") or {}
    if ptc:
        print("\n--- Accuracy by Video Category ---")
        for name in sorted(ptc.keys()):
            a = ptc[name]
            st = pc.get(name) or {}
            tok = int(st.get("total_ok", 0) or 0)
            cok = int(st.get("correct_ok", 0) or 0)
            if tok and a is not None:
                print(f"{name}: {cok}/{tok} = {a:.2%}")
            else:
                print(f"{name}: 0/0 = N/A")

    pdf = metrics.get("daily_omni_per_duration_accuracy_github_style") or {}
    if pdf:
        print("\n--- Accuracy by Video Duration ---")
        for name in DAILY_OMNI_DURATION_KEYS:
            a = pdf.get(name)
            st = (metrics.get("daily_omni_per_duration") or {}).get(name) or {}
            tok = int(st.get("total_ok", 0) or 0)
            cor = int(st.get("correct_ok", 0) or 0)
            if tok and a is not None:
                print(f"{name} Duration: {cor}/{tok} = {a:.2%}")
            else:
                print(f"{name} Duration: 0/0 = N/A")
    print("=" * 50)
