#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Accuracy (and light perf) checks for Qwen3-Omni via ``vllm bench serve --omni``.

The standalone CLI uses small ``--num-prompts`` / ``--max-concurrency`` defaults suitable for
L4-style smoke runs against an already-running server. The pytest wrappers in
``tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py`` may still require larger GPUs (currently
H100 / MI325) because they launch the live Omni server inside the test.

1. **Daily-Omni** — MCQ accuracy fields in the saved JSON (``daily_omni_accuracy``, …); by default the
   run **fails** if accuracy is strictly below **0.69** (``--min-daily-omni-accuracy`` / ``ACC_BENCH_MIN_DAILY_OMNI_ACCURACY``).
2. **Seed-TTS** — ``seed-tts-eval``-style metrics when ``--seed-tts-wer-eval`` is used
   (WER / SIM / UTMOS keys from :func:`compute_seed_tts_wer_metrics`).

Prerequisites
-------------
* A running Omni OpenAI-compatible server (same machine or reachable host), e.g.::

    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8000

  On L4 you may need a smaller checkpoint, quantization, or tighter engine flags; this script
  only drives the **client** benchmark.

* ``vllm`` CLI from **vLLM-Omni** (so ``bench serve`` registers ``daily-omni`` / ``seed-tts``).

* **Daily-Omni** — if local ``qa.json`` + ``Videos/`` are not both provided (CLI or matching env),
  the client passes ``--dataset-path`` with a Hub id (default ``liarliar/Daily-Omni``). The **child**
  ``vllm bench serve`` process then loads QA via ``datasets.load_dataset`` (needs ``pip install datasets``,
  network or HF cache). Without ``--daily-omni-video-dir``, the benchmark **lazily** downloads and
  extracts ``Videos.tar`` from the Hub (``huggingface_hub``) on first multimodal request. Override
  the dataset repo with ``--daily-omni-repo`` or ``VLLM_DAILY_OMNI_REPO``; override the tar repo
  with ``VLLM_DAILY_OMNI_MEDIA_REPO`` if needed.

* **Seed-TTS** optional extras for WER/SIM/UTMOS::

    pip install 'vllm-omni[seed-tts-eval]'

Examples
--------
Pytest (same checks; needs a running server)::

    pytest -sv tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py

Smoke on localhost (server already up)::

    python tests/e2e/accuracy/qwen3_omni/run_qwen_omni_acc_benchmark.py \\
        --model Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --daily-omni-qa-json ./qa.json \\
        --daily-omni-video-dir ./Videos \\
        --seed-tts-dataset-path ./seed-tts-eval

Skip one suite, tighten gates::

    python tests/e2e/accuracy/qwen3_omni/run_qwen_omni_acc_benchmark.py \\
        --skip-daily-omni \\
        --max-seed-tts-mean-wer 0.35 \\
        --min-seed-tts-mean-sim 0.75
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.e2e.accuracy.qwen3_omni.qwen3_omni_acc_bench_core import (
    build_serve_common_argv,
    daily_omni_bench_argv,
    find_vllm_cli,
    load_benchmark_result,
    run_vllm_bench_subprocess,
    seed_tts_bench_argv,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _repo_root() -> Path:
    return _REPO_ROOT


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "results" / "qwen_omni_acc"


def _validate_daily_omni(result: dict[str, Any], *, min_accuracy: float | None) -> list[str]:
    errs: list[str] = []
    acc = result.get("daily_omni_accuracy")
    if acc is None:
        errs.append("Missing daily_omni_accuracy (wrong dataset or no gold-evaluated rows).")
        return errs
    ev = int(result.get("daily_omni_evaluated_ok", 0) or 0)
    if ev <= 0:
        errs.append("daily_omni_evaluated_ok is 0; no successful MCQ rows to score.")
    if min_accuracy is not None and float(acc) + 1e-12 < float(min_accuracy):
        errs.append(f"daily_omni_accuracy={acc:.6f} < --min-daily-omni-accuracy={min_accuracy}")
    return errs


def _validate_seed_tts(
    result: dict[str, Any],
    *,
    max_mean_wer: float | None,
    min_mean_sim: float | None,
    min_mean_utmos: float | None,
) -> list[str]:
    errs: list[str] = []
    setup = result.get("seed_tts_eval_setup_error")
    if setup:
        errs.append(f"Seed-TTS eval setup failed: {setup}")
        return errs
    n = int(result.get("seed_tts_content_evaluated", 0) or 0)
    if n <= 0:
        errs.append("seed_tts_content_evaluated is 0 (enable --seed-tts-wer-eval and check PCM capture).")
    mean_wer = result.get("seed_tts_content_error_mean")
    if mean_wer is not None and max_mean_wer is not None and float(mean_wer) > float(max_mean_wer) + 1e-12:
        errs.append(f"seed_tts_content_error_mean (WER)={mean_wer:.6f} > --max-seed-tts-mean-wer={max_mean_wer}")
    sim_m = result.get("seed_tts_sim_mean")
    if sim_m is not None and min_mean_sim is not None and float(sim_m) + 1e-12 < float(min_mean_sim):
        errs.append(f"seed_tts_sim_mean={sim_m:.6f} < --min-seed-tts-mean-sim={min_mean_sim}")
    ut_m = result.get("seed_tts_utmos_mean")
    if ut_m is not None and min_mean_utmos is not None and float(ut_m) + 1e-12 < float(min_mean_utmos):
        errs.append(f"seed_tts_utmos_mean={ut_m:.6f} < --min-seed-tts-mean-utmos={min_mean_utmos}")
    return errs


def sync_dataset_env_from_ns(ns: argparse.Namespace) -> None:
    """Mirror CLI path flags into env vars read by ``daily_omni_bench_argv`` / ``seed_tts_bench_argv``."""
    repo = getattr(ns, "daily_omni_repo", None)
    if repo is not None and str(repo).strip():
        os.environ["VLLM_DAILY_OMNI_REPO"] = str(repo).strip()
    if ns.daily_omni_qa_json is not None:
        os.environ["VLLM_DAILY_OMNI_QA_JSON"] = str(Path(ns.daily_omni_qa_json).expanduser().resolve())
    if ns.daily_omni_video_dir is not None:
        os.environ["VLLM_DAILY_OMNI_VIDEO_DIR"] = str(Path(ns.daily_omni_video_dir).expanduser().resolve())
    if ns.seed_tts_dataset_path is not None:
        # ``--seed-tts-dataset-path`` accepts either a local directory or a
        # Hugging Face repo id. Only resolve to an absolute filesystem path
        # when the value actually exists locally; otherwise preserve the repo
        # string verbatim so downstream code can pass it to snapshot_download.
        raw = str(ns.seed_tts_dataset_path).strip()
        p = Path(raw).expanduser()
        os.environ["VLLM_SEED_TTS_DATASET_PATH"] = str(p.resolve()) if p.exists() and p.is_dir() else raw
    if ns.seed_tts_root is not None:
        os.environ["SEED_TTS_ROOT"] = str(Path(ns.seed_tts_root).expanduser().resolve())


@contextlib.contextmanager
def _preserve_benchmark_dataset_env() -> Any:
    """Save/restore dataset-related env vars so benchmark tests don't leak state."""
    keys = (
        "VLLM_DAILY_OMNI_REPO",
        "VLLM_DAILY_OMNI_QA_JSON",
        "VLLM_DAILY_OMNI_VIDEO_DIR",
        "VLLM_SEED_TTS_DATASET_PATH",
        "SEED_TTS_ROOT",
    )
    original = {k: os.environ.get(k) for k in keys}
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_common_args(ns: argparse.Namespace, *, result_filename: str) -> list[str]:
    return build_serve_common_argv(
        host=ns.host,
        port=ns.port,
        model=ns.model,
        num_prompts=ns.num_prompts,
        max_concurrency=ns.max_concurrency,
        num_warmups=ns.num_warmups,
        percentile_metrics=ns.percentile_metrics,
        result_dir=ns.result_dir,
        result_filename=result_filename,
        ready_check_timeout_sec=ns.ready_check_timeout_sec,
    )


def run_daily_omni(ns: argparse.Namespace, vllm: str) -> Path:
    ns.result_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_filename = f"qwen_omni_acc_daily_omni_{tag}.json"
    extra = json.loads(ns.daily_extra_body_json)
    argv = (
        _build_common_args(ns, result_filename=result_filename)
        + daily_omni_bench_argv()
        + [
            "--daily-omni-input-mode",
            ns.daily_omni_input_mode,
            "--extra-body",
            json.dumps(extra, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    if ns.daily_omni_save_eval_items:
        argv.append("--daily-omni-save-eval-items")
    print("\n$", vllm, *argv, "\n", flush=True)
    run_vllm_bench_subprocess(vllm, argv)
    out = Path(ns.result_dir) / result_filename
    if not out.is_file():
        raise FileNotFoundError(f"Expected result JSON at {out}")
    return out


def run_seed_tts(ns: argparse.Namespace, vllm: str) -> Path:
    ns.result_dir.mkdir(parents=True, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_filename = f"qwen_omni_acc_seed_tts_{tag}.json"
    extra = json.loads(ns.seed_extra_body_json)
    argv = (
        _build_common_args(ns, result_filename=result_filename)
        + seed_tts_bench_argv(locale=ns.seed_tts_locale)
        + [
            "--seed-tts-wer-eval",
            "--extra-body",
            json.dumps(extra, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    if ns.seed_tts_wer_save_items:
        argv.append("--seed-tts-wer-save-items")
    if ns.seed_tts_file_ref_audio:
        argv.append("--seed-tts-file-ref-audio")
    extra_env: dict[str, str] = {"SEED_TTS_WER_EVAL": "1"}
    if ns.seed_tts_eval_device:
        extra_env["SEED_TTS_EVAL_DEVICE"] = ns.seed_tts_eval_device
    print("\n$", vllm, *argv, "\n", flush=True)
    run_vllm_bench_subprocess(vllm, argv, extra_env=extra_env)
    out = Path(ns.result_dir) / result_filename
    if not out.is_file():
        raise FileNotFoundError(f"Expected result JSON at {out}")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default=os.environ.get("ACC_BENCH_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ACC_BENCH_PORT", "8000")))
    p.add_argument(
        "--model",
        default=os.environ.get(
            "ACC_BENCH_MODEL",
            "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        ),
        help="Model id passed to ``vllm bench serve`` (must match the running server).",
    )
    p.add_argument("--num-prompts", type=int, default=int(os.environ.get("ACC_BENCH_NUM_PROMPTS", "2000")))
    p.add_argument("--max-concurrency", type=int, default=int(os.environ.get("ACC_BENCH_MAX_CONCURRENCY", "10")))
    p.add_argument("--num-warmups", type=int, default=int(os.environ.get("ACC_BENCH_NUM_WARMUPS", "0")))
    p.add_argument(
        "--percentile-metrics",
        default=os.environ.get("ACC_BENCH_PERCENTILE_METRICS", "ttft,tpot,itl,e2el,audio_ttfp,audio_rtf"),
    )
    p.add_argument(
        "--ready-check-timeout-sec",
        type=int,
        default=None,
        help="If set, forwarded to ``vllm bench serve`` (probe first request until success). "
        "Omit to use upstream default (typically skip).",
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        default=Path(os.environ.get("ACC_BENCH_RESULT_DIR", str(_default_result_dir()))),
    )

    p.add_argument("--skip-daily-omni", action="store_true")
    p.add_argument("--skip-seed-tts", action="store_true")

    p.add_argument(
        "--daily-omni-repo",
        type=str,
        default=None,
        help="Hugging Face dataset id for Daily-Omni Hub mode (sets VLLM_DAILY_OMNI_REPO). "
        "Ignored when local qa.json + video dir are used.",
    )
    p.add_argument(
        "--daily-omni-qa-json",
        type=Path,
        default=None,
        help="Optional local qa.json; if omitted with no env, uses Hub liarliar/Daily-Omni.",
    )
    p.add_argument(
        "--daily-omni-video-dir",
        type=Path,
        default=None,
        help="Optional local Videos root; if omitted, media is fetched lazily from Hub Videos.tar.",
    )
    p.add_argument("--daily-omni-input-mode", choices=("all", "visual", "audio"), default="all")
    p.add_argument(
        "--daily-extra-body-json",
        default='{"modalities":["text"]}',
        help="JSON merged into each chat request for Daily-Omni (default matches common L4 / text-output runs).",
    )
    p.add_argument(
        "--daily-omni-save-eval-items",
        action="store_true",
        help="Sets env via CLI flag so per-item rows are stored in the result JSON.",
    )
    p.add_argument(
        "--min-daily-omni-accuracy",
        type=float,
        default=float((os.environ.get("ACC_BENCH_MIN_DAILY_OMNI_ACCURACY") or "0.69").strip() or "0.69"),
        help="Fail when daily_omni_accuracy is strictly below this threshold (0–1). "
        "Default baseline 0.69; override with env ACC_BENCH_MIN_DAILY_OMNI_ACCURACY or pass 0 to disable the floor.",
    )

    p.add_argument(
        "--seed-tts-dataset-path",
        type=str,
        default=None,
        help="Optional local root or Hub id; if omitted, uses zhaochenyang20/seed-tts-eval.",
    )
    p.add_argument("--seed-tts-root", type=Path, default=None, help="Optional override for Seed-TTS filesystem root.")
    p.add_argument("--seed-tts-locale", choices=("en", "zh"), default="en")
    p.add_argument(
        "--seed-extra-body-json",
        default='{"modalities":["text","audio"]}',
        help="JSON for Seed-TTS chat requests (must include audio for synthesis + PCM capture).",
    )
    p.add_argument("--seed-tts-wer-save-items", action="store_true")
    p.add_argument(
        "--seed-tts-file-ref-audio",
        action="store_true",
        help="Use file:// ref_audio; server must allow local media paths.",
    )
    p.add_argument(
        "--seed-tts-eval-device",
        default=os.environ.get("SEED_TTS_EVAL_DEVICE"),
        help="Sets SEED_TTS_EVAL_DEVICE for Whisper / WavLM / UTMOS (e.g. cuda:0).",
    )
    p.add_argument(
        "--max-seed-tts-mean-wer",
        type=float,
        default=0.02,
        help="If set, fail when seed_tts_content_error_mean is strictly above this value.",
    )
    p.add_argument(
        "--min-seed-tts-mean-sim",
        type=float,
        default=None,
        help="If set, fail when seed_tts_sim_mean is strictly below this value.",
    )
    p.add_argument(
        "--min-seed-tts-mean-utmos",
        type=float,
        default=None,
        help="If set, fail when seed_tts_utmos_mean is strictly below this value.",
    )
    return p


def parse_acc_benchmark_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args; when ``argv`` is ``None``, use ``sys.argv[1:]`` (standalone script)."""
    if argv is None:
        argv = sys.argv[1:]
    return build_arg_parser().parse_args(argv)


def run_acc_benchmark(ns: argparse.Namespace) -> int:
    """Run Daily-Omni and/or Seed-TTS client benches against a running server; return 0 on success."""
    failed: list[str] = []

    with _preserve_benchmark_dataset_env():
        sync_dataset_env_from_ns(ns)

        vllm = find_vllm_cli()
        print(f"Using vLLM CLI: {vllm}", flush=True)
        print(f"Repo root (for cwd reference): {_repo_root()}", flush=True)

        if not ns.skip_daily_omni:
            path = run_daily_omni(ns, vllm)
            print(f"\n[Daily-Omni] result JSON: {path}", flush=True)
            data = load_benchmark_result(path)
            errs = _validate_daily_omni(data, min_accuracy=ns.min_daily_omni_accuracy)
            if errs:
                failed.extend([f"[Daily-Omni] {e}" for e in errs])
            else:
                print(
                    f"[Daily-Omni] daily_omni_accuracy={data.get('daily_omni_accuracy')} "
                    f"evaluated_ok={data.get('daily_omni_evaluated_ok')}",
                    flush=True,
                )

        if not ns.skip_seed_tts:
            path = run_seed_tts(ns, vllm)
            print(f"\n[Seed-TTS] result JSON: {path}", flush=True)
            data = load_benchmark_result(path)
            errs = _validate_seed_tts(
                data,
                max_mean_wer=ns.max_seed_tts_mean_wer,
                min_mean_sim=ns.min_seed_tts_mean_sim,
                min_mean_utmos=ns.min_seed_tts_mean_utmos,
            )
            if errs:
                failed.extend([f"[Seed-TTS] {e}" for e in errs])
            else:
                print(
                    f"[Seed-TTS] mean_wer={data.get('seed_tts_content_error_mean')} "
                    f"mean_sim={data.get('seed_tts_sim_mean')} mean_utmos={data.get('seed_tts_utmos_mean')} "
                    f"evaluated={data.get('seed_tts_content_evaluated')}",
                    flush=True,
                )

    if failed:
        print("\nACCURACY CHECK FAILED:", file=sys.stderr)
        for line in failed:
            print(f"  - {line}", file=sys.stderr)
        return 1

    print("\nAll configured accuracy checks passed.", flush=True)
    return 0


def main() -> int:
    return run_acc_benchmark(parse_acc_benchmark_args())


if __name__ == "__main__":
    raise SystemExit(main())
