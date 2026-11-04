# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from base64 import b64decode, b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from hashlib import sha1
from io import BytesIO
from mimetypes import guess_type
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest
import requests
import torch
from PIL import Image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

FTFY_SITECUSTOMIZE_MOCK_DIR = Path(__file__).with_name("ftfy_mock")
MEDIA_TOOL_TIMEOUT_SECONDS = 5 * 60


class IdentityFtfy:
    @staticmethod
    def fix_text(text: str) -> str:
        return text


def apply_ftfy_mock(*, wan_i2v_module: Any | None = None) -> None:
    """Mock ftfy library for test cases in the main process"""
    module: Any = wan_i2v_module
    if module is None:
        from diffusers.pipelines.wan import pipeline_wan_i2v as wan_pipeline

        module = wan_pipeline

    if not hasattr(module, "ftfy"):
        module.ftfy = IdentityFtfy()
        print("ftfy (text encoding sanitizer) is not installed. Using mock ftfy implementation (identity function)")
    else:
        print("ftfy (text encoding sanitizer) is installed. Using actual ftfy implementation.")


def env_to_apply_ftfy_mock_in_subproc(env: dict[str, str] | None = None) -> dict[str, str]:
    """Mock ftfy library for test cases in subprocesses, by prepending the sitecustomize.py directory to PYTHONPATH"""
    env_dict = dict(env or {})
    pythonpath = env_dict.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    path_entries = [str(FTFY_SITECUSTOMIZE_MOCK_DIR)]
    if pythonpath:
        path_entries.append(pythonpath)
    env_dict["PYTHONPATH"] = os.pathsep.join(path_entries)
    return env_dict


def reset_artifact_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def infer_model_label(model: str) -> str:
    label = Path(model.rstrip("/\\")).name or "model"
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in label)


def model_output_dir(parent_dir: Path, model: str) -> Path:
    safe_model_name = model.split("/")[-1].replace(".", "_")
    path = parent_dir / safe_model_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json_artifact(path: Path, value: Any) -> None:
    """Write a UTF-8 accuracy artifact, rejecting non-finite JSON numbers."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _omni_acc_bench_core():
    """Shared Daily-Omni / Video-MME / Seed-TTS argv helpers.

    Imported lazily so diffusion accuracy tests that use this module do not load
    the omni benchmark core at import time.
    """
    from tests.e2e.accuracy.qwen3_omni import qwen3_omni_acc_bench_core as core

    return core


def daily_omni_bench_argv() -> list[str]:
    """Daily-Omni dataset argv. Accuracy tests patch this to inline local media."""
    return _omni_acc_bench_core().daily_omni_bench_argv()


def build_acc_benchmark_cli_argv(*args, **kwargs) -> list[str]:
    """Assemble argv for a live Omni server accuracy bench."""
    return _omni_acc_bench_core().build_acc_benchmark_cli_argv(*args, **kwargs)


def find_vllm_cli() -> str:
    """Locate the vLLM CLI used by accuracy benches."""
    return _omni_acc_bench_core().find_vllm_cli()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_result_dir() -> Path:
    return Path(__file__).resolve().parent / "qwen3_omni" / "results" / "qwen_omni_acc"


def _accuracy_hardware_label() -> str:
    """Local device token for the accuracy result, or ``""`` when detection fails.

    The accuracy client can target a remote server, so ``device_count`` /
    ``get_device_name`` may be unusable on the host that writes the JSON.
    Dashboard metadata must not abort the run before the accuracy gate.
    """
    try:
        from tests.helpers.hardware import get_runtime_resource_label, hardware_json_value

        return hardware_json_value(get_runtime_resource_label())
    except Exception:
        logging.getLogger(__name__).warning(
            "Could not detect Hardware for the accuracy result; writing an empty label", exc_info=True
        )
        return ""


def _write_accuracy_result_metadata(
    path: Path,
    data: dict[str, Any],
    baseline: dict[str, Any],
    *,
    dataset: str,
) -> None:
    """Write dashboard fields into the bench result JSON.

    ``vllm bench serve`` saves measured scores only. The thresholds from the
    accuracy test (``--min-daily-omni-accuracy``, ``--max-seed-tts-mean-wer``, …)
    are applied afterwards, so persist them under ``baseline`` before the check.
    Unset optional gates (``None``) are omitted.

    ``dataset`` is the filename token (``daily_omni``, ``videomme``, ``seed_tts``).
    Model and dataset names both contain underscores, so the dashboard reads
    ``dataset`` from this field. ``model_id`` stays as ``vllm bench serve`` wrote it.

    ``Hardware`` uses the same runtime label as ``tests/dfx/perf/scripts/run_benchmark.py``
    (``H100``, ``A3``, …; empty when the device is unknown or detection fails).

    ``branch`` is the Buildkite branch of this run (``BUILDKITE_BRANCH``), matching
    performance result records and Omni-DuplexEval accuracy artifacts. It is
    empty outside CI.
    """
    data["dataset"] = dataset
    data["baseline"] = {key: value for key, value in baseline.items() if value is not None}
    data["Hardware"] = _accuracy_hardware_label()
    data["branch"] = os.environ.get("BUILDKITE_BRANCH", "")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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


def _validate_videomme(result: dict[str, Any], *, min_accuracy: float | None) -> list[str]:
    errs: list[str] = []
    acc = result.get("videomme_accuracy")
    if acc is None:
        errs.append("Missing videomme_accuracy (wrong dataset or no gold-evaluated rows).")
        return errs
    ev = int(result.get("videomme_evaluated_ok", 0) or 0)
    if ev <= 0:
        errs.append("videomme_evaluated_ok is 0; no successful MCQ rows to score.")
    failed = int(result.get("videomme_request_failed", 0) or 0)
    if failed > 0:
        errs.append(
            f"videomme_request_failed={failed}; the accuracy gate requires every "
            "gold-labeled request to complete HTTP successfully "
            "(videomme_accuracy excludes HTTP failures)."
        )
    submitted = int(result.get("videomme_submitted", 0) or 0)
    unique = int(result.get("videomme_unique_question_ids", 0) or 0)
    if submitted <= 0 or unique != submitted:
        errs.append(
            f"Video-MME coverage is incomplete: submitted={submitted}, unique={unique}; "
            "the accuracy gate requires distinct questions, not oversampled replacements."
        )
    skipped = int(result.get("videomme_skipped_rows", 0) or 0)
    if skipped:
        errs.append(
            f"videomme_skipped_rows={skipped}; fix missing or unreadable source rows before accuracy evaluation."
        )
    no_gold = int(result.get("videomme_no_gold", 0) or 0)
    if no_gold:
        errs.append(f"videomme_no_gold={no_gold}; every submitted question must have a gold answer.")
    if min_accuracy is not None and float(acc) + 1e-12 < float(min_accuracy):
        errs.append(f"videomme_accuracy={acc:.6f} < --min-videomme-accuracy={min_accuracy}")
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
    timed_out = int(result.get("timed_out_requests", 0) or 0)
    cancelled = int(result.get("cancelled_after_timeout_requests", 0) or 0)
    if timed_out or cancelled:
        errs.append(
            "Seed-TTS benchmark coverage is incomplete: "
            f"timed_out_requests={timed_out}, cancelled_after_timeout_requests={cancelled}; "
            "accuracy metrics exclude unsuccessful outputs."
        )
    n = int(result.get("seed_tts_content_evaluated", 0) or 0)
    if n <= 0:
        errs.append("seed_tts_content_evaluated is 0 (enable --seed-tts-wer-eval and check PCM capture).")
    mean_wer = result.get("seed_tts_content_error_mean")
    if n > 0 and max_mean_wer is not None and mean_wer is None:
        errs.append(
            f"seed_tts_content_error_mean (WER) is missing; cannot apply --max-seed-tts-mean-wer={max_mean_wer}"
        )
    elif mean_wer is not None and max_mean_wer is not None and float(mean_wer) > float(max_mean_wer) + 1e-12:
        errs.append(f"seed_tts_content_error_mean (WER)={mean_wer:.6f} > --max-seed-tts-mean-wer={max_mean_wer}")
    errs.extend(
        _seed_tts_floor_errors(
            result,
            mean_key="seed_tts_sim_mean",
            count_key="seed_tts_sim_evaluated",
            expected=_seed_tts_sim_expected_count(result),
            floor=min_mean_sim,
            flag="--min-seed-tts-mean-sim",
            enable_env="SEED_TTS_SIM_EVAL=1",
        )
    )
    errs.extend(
        _seed_tts_floor_errors(
            result,
            mean_key="seed_tts_utmos_mean",
            count_key="seed_tts_utmos_evaluated",
            expected=_seed_tts_utmos_expected_count(result),
            floor=min_mean_utmos,
            flag="--min-seed-tts-mean-utmos",
            enable_env="SEED_TTS_UTMOS_EVAL=1",
        )
    )
    return errs


def _seed_tts_floor_errors(
    result: dict[str, Any],
    *,
    mean_key: str,
    count_key: str,
    expected: int | None,
    floor: float | None,
    flag: str,
    enable_env: str,
) -> list[str]:
    """Fail closed when a configured floor has no score or only a partial one.

    A missing SIM or UTMOS mean used to skip the comparison, so a CI baseline
    passed while the scorer was still off. A later forward error is counted in
    ``*_failed`` and left out of the mean, so a high mean over the clips that
    survived must not pass either. ``expected`` is the number of rows that
    reached this scorer; rows the evaluator never hands to it do not count.
    """
    if floor is None:
        return []
    mean = result.get(mean_key)
    count = int(result.get(count_key, 0) or 0)
    failed_key = count_key.replace("_evaluated", "_failed")
    failed = int(result.get(failed_key, 0) or 0)
    if mean is None or count <= 0:
        return [f"{mean_key} is missing (evaluated={count}); set {enable_env} before applying {flag}={floor}"]
    errs: list[str] = []
    if failed > 0 or (expected is not None and count < expected):
        expected_text = "unknown" if expected is None else str(expected)
        errs.append(
            f"{count_key}={count} failed={failed} expected={expected_text}; coverage is incomplete for {flag}={floor}"
        )
    if float(mean) + 1e-12 < float(floor):
        errs.append(f"{mean_key}={float(mean):.6f} < {flag}={floor}")
    return errs


def _seed_tts_sim_expected_count(result: dict[str, Any]) -> int | None:
    """SIM runs only on WER-scored rows that have a reference wav."""
    content = result.get("seed_tts_content_evaluated")
    if content is None:
        return None
    expected = int(content or 0) - int(result.get("seed_tts_sim_skipped_no_ref", 0) or 0)
    return max(expected, 0)


def _seed_tts_utmos_expected_count(result: dict[str, Any]) -> int | None:
    """UTMOS runs on every captured row with non-empty audio, independent of ASR."""
    turn_count = result.get("seed_tts_turn_count")
    if turn_count is None:
        content = result.get("seed_tts_content_evaluated")
        return None if content is None else int(content or 0)
    expected = int(turn_count or 0)
    expected -= int(result.get("seed_tts_request_failed", 0) or 0)
    expected -= int(result.get("seed_tts_no_pcm", 0) or 0)
    expected -= int(result.get("seed_tts_empty_audio", 0) or 0)
    return max(expected, 0)


def seed_tts_eval_env(ns: argparse.Namespace) -> dict[str, str]:
    """Env for the Seed-TTS accuracy subprocess.

    WER is always on for this path. SIM and UTMOS turn on when their floor is
    set and the parent process has not already chosen an explicit value.
    """
    env: dict[str, str] = {"SEED_TTS_WER_EVAL": "1"}
    if ns.seed_tts_eval_device:
        env["SEED_TTS_EVAL_DEVICE"] = ns.seed_tts_eval_device
    if ns.min_seed_tts_mean_sim is not None and not os.environ.get("SEED_TTS_SIM_EVAL", "").strip():
        env["SEED_TTS_SIM_EVAL"] = "1"
    if ns.min_seed_tts_mean_utmos is not None and not os.environ.get("SEED_TTS_UTMOS_EVAL", "").strip():
        env["SEED_TTS_UTMOS_EVAL"] = "1"
    return env


def sync_dataset_env_from_ns(ns: argparse.Namespace) -> None:
    """Mirror CLI path flags into env vars read by dataset ``*_bench_argv`` helpers."""
    repo = getattr(ns, "daily_omni_repo", None)
    if repo is not None and str(repo).strip():
        os.environ["VLLM_DAILY_OMNI_REPO"] = str(repo).strip()
    if ns.daily_omni_qa_json is not None:
        os.environ["VLLM_DAILY_OMNI_QA_JSON"] = str(Path(ns.daily_omni_qa_json).expanduser().resolve())
    if ns.daily_omni_video_dir is not None:
        os.environ["VLLM_DAILY_OMNI_VIDEO_DIR"] = str(Path(ns.daily_omni_video_dir).expanduser().resolve())
    vm_repo = getattr(ns, "videomme_repo", None)
    if vm_repo is not None and str(vm_repo).strip():
        os.environ["VLLM_VIDEOMME_REPO"] = str(vm_repo).strip()
    vm_path = getattr(ns, "videomme_dataset_path", None)
    if vm_path is not None and str(vm_path).strip():
        raw = str(vm_path).strip()
        p = Path(raw).expanduser()
        os.environ["VLLM_VIDEOMME_DATASET_PATH"] = str(p.resolve()) if p.exists() and p.is_dir() else raw
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
        "VLLM_VIDEOMME_REPO",
        "VLLM_VIDEOMME_DATASET_PATH",
        "VIDEOMME_ROOT",
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


def _acc_model_name(model: str) -> str:
    """Checkpoint name safe for a result filename.

    ``Qwen/Qwen3-Omni-30B-A3B-Instruct`` becomes ``Qwen3-Omni-30B-A3B-Instruct``.
    """
    name = str(model).strip().replace("\\", "/").rstrip("/")
    name = name.rsplit("/", 1)[-1]
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name).strip("._-")
    return safe or "model"


def _acc_result_filename(model: str, suite: str, when: datetime | None = None) -> str:
    """``result_acc_{model}_{suite}_{date}_{timestamp}.json``.

    Daily-Omni on Qwen3-Omni and MiniCPM-o 4.5 both use this shape, for example
    ``result_acc_Qwen3-Omni-30B-A3B-Instruct_daily_omni_20261014_164800.json``.
    """
    when = datetime.now() if when is None else when
    return f"result_acc_{_acc_model_name(model)}_{suite}_{when.strftime('%Y%m%d')}_{when.strftime('%H%M%S')}.json"


def _build_common_args(
    ns: argparse.Namespace,
    *,
    result_filename: str,
    backend: str = "openai-chat-omni",
    endpoint: str = "/v1/chat/completions",
) -> list[str]:
    return _omni_acc_bench_core().build_serve_common_argv(
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
        trust_remote_code=ns.trust_remote_code,
        backend=backend,
        endpoint=endpoint,
        temperature=getattr(ns, "temperature", None),
        output_len=getattr(ns, "output_len", None),
    )


def run_daily_omni(ns: argparse.Namespace, vllm: str) -> Path:
    ns.result_dir.mkdir(parents=True, exist_ok=True)
    result_filename = _acc_result_filename(ns.model, "daily_omni")
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
    pack_mode = getattr(ns, "daily_omni_pack_mode", None)
    if pack_mode:
        argv.extend(["--daily-omni-pack-mode", pack_mode])
    if ns.daily_omni_save_eval_items:
        argv.append("--daily-omni-save-eval-items")
    print("\n$", vllm, *argv, "\n", flush=True)
    _omni_acc_bench_core().run_vllm_bench_subprocess(vllm, argv)
    out = Path(ns.result_dir) / result_filename
    if not out.is_file():
        raise FileNotFoundError(f"Expected result JSON at {out}")
    return out


def run_videomme(ns: argparse.Namespace, vllm: str) -> Path:
    ns.result_dir.mkdir(parents=True, exist_ok=True)
    result_filename = _acc_result_filename(ns.model, "videomme")
    extra = json.loads(ns.videomme_extra_body_json)
    argv = (
        _build_common_args(ns, result_filename=result_filename)
        + _omni_acc_bench_core().videomme_bench_argv()
        + [
            "--disable-shuffle",
            "--videomme-pack-mode",
            ns.videomme_pack_mode,
            "--videomme-duration",
            ns.videomme_duration,
            "--extra-body",
            json.dumps(extra, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    if ns.videomme_max_frames is not None:
        argv.extend(["--videomme-max-frames", str(int(ns.videomme_max_frames))])
    if ns.videomme_use_subtitle:
        argv.append("--videomme-use-subtitle")
    if ns.videomme_inline_local_video:
        argv.append("--videomme-inline-local-video")
    if ns.videomme_save_eval_items:
        argv.append("--videomme-save-eval-items")
    if ns.videomme_video_dir is not None:
        argv.extend(["--videomme-video-dir", str(Path(ns.videomme_video_dir).expanduser().resolve())])
    if ns.videomme_parquet is not None:
        argv.extend(["--videomme-parquet", str(Path(ns.videomme_parquet).expanduser().resolve())])
    if ns.videomme_subtitle_dir is not None:
        argv.extend(["--videomme-subtitle-dir", str(Path(ns.videomme_subtitle_dir).expanduser().resolve())])
    print("\n$", vllm, *argv, "\n", flush=True)
    _omni_acc_bench_core().run_vllm_bench_subprocess(vllm, argv)
    out = Path(ns.result_dir) / result_filename
    if not out.is_file():
        raise FileNotFoundError(f"Expected result JSON at {out}")
    return out


def run_seed_tts(ns: argparse.Namespace, vllm: str) -> Path:
    ns.result_dir.mkdir(parents=True, exist_ok=True)
    result_filename = _acc_result_filename(ns.model, "seed_tts")
    extra = json.loads(ns.seed_extra_body_json)
    argv = (
        _build_common_args(
            ns,
            result_filename=result_filename,
            backend=ns.seed_backend,
            endpoint=ns.seed_endpoint,
        )
        + _omni_acc_bench_core().seed_tts_bench_argv(locale=ns.seed_tts_locale)
        + [
            "--seed-tts-wer-eval",
            "--extra-body",
            json.dumps(extra, ensure_ascii=False, separators=(",", ":")),
        ]
    )
    if ns.seed_tts_turns_per_session > 1:
        argv.extend(["--seed-tts-turns-per-session", str(ns.seed_tts_turns_per_session)])
    if ns.seed_tts_wer_save_items:
        argv.append("--seed-tts-wer-save-items")
    if ns.seed_tts_file_ref_audio:
        argv.append("--seed-tts-file-ref-audio")
    if getattr(ns, "omni_fail_fast_on_timeout", False):
        argv.append("--omni-fail-fast-on-timeout")
    extra_env = seed_tts_eval_env(ns)
    print("\n$", vllm, *argv, "\n", flush=True)
    _omni_acc_bench_core().run_vllm_bench_subprocess(vllm, argv, extra_env=extra_env)
    out = Path(ns.result_dir) / result_filename
    if not out.is_file():
        raise FileNotFoundError(f"Expected result JSON at {out}")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
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
        "--omni-fail-fast-on-timeout",
        action="store_true",
        help="Forward the benchmark timeout circuit breaker to Seed-TTS runs.",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code to ``vllm bench serve`` for models with custom code.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Optional sampling temperature forwarded to ``vllm bench serve`` (e.g. 0 for MiniCPM-o Daily-Omni MCQ).",
    )
    p.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Optional max output tokens forwarded to ``vllm bench serve`` (e.g. 512 for MiniCPM-o Daily-Omni).",
    )
    p.add_argument(
        "--result-dir",
        type=Path,
        default=Path(os.environ.get("ACC_BENCH_RESULT_DIR", str(_default_result_dir()))),
    )

    p.add_argument("--skip-daily-omni", action="store_true")
    p.add_argument("--skip-seed-tts", action="store_true")
    # Video-MME is opt-in: default skip=True so existing Daily-Omni / Seed-TTS callers stay unchanged.
    p.set_defaults(skip_videomme=True)
    p.add_argument(
        "--skip-videomme",
        action="store_true",
        dest="skip_videomme",
        help="Skip Video-MME (default).",
    )
    p.add_argument(
        "--run-videomme",
        action="store_false",
        dest="skip_videomme",
        help="Enable the Video-MME accuracy suite.",
    )

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
        "--daily-omni-pack-mode",
        choices=("qwen", "minicpm-interleave"),
        default=None,
        help="Daily-Omni multimodal pack recipe. Use ``minicpm-interleave`` for MiniCPM-o "
        "(1fps interleaved image/audio; see PR #5606).",
    )
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
        "--videomme-repo",
        type=str,
        default=None,
        help="Hugging Face dataset id for Video-MME Hub mode (sets VLLM_VIDEOMME_REPO). "
        "Default when no local path is set: lmms-eval/Video-MME. "
        "Ignored when a local --videomme-dataset-path / VLLM_VIDEOMME_DATASET_PATH is used.",
    )
    p.add_argument(
        "--videomme-dataset-path",
        type=str,
        default=None,
        help="Optional local Video-MME root (parquet + videos). Existing directories skip the Hub "
        "download; Hub ids are also accepted and passed through as --dataset-path. "
        "Sets VLLM_VIDEOMME_DATASET_PATH.",
    )
    p.add_argument(
        "--videomme-parquet",
        type=Path,
        default=None,
        help="Optional explicit path to videomme/test-*.parquet.",
    )
    p.add_argument(
        "--videomme-video-dir",
        type=Path,
        default=None,
        help="Optional explicit directory of extracted Video-MME videos.",
    )
    p.add_argument(
        "--videomme-subtitle-dir",
        type=Path,
        default=None,
        help="Optional subtitle directory (used with --videomme-use-subtitle).",
    )
    p.add_argument(
        "--videomme-pack-mode",
        choices=("minicpm-frames", "minicpm-interleave", "video_url"),
        default="minicpm-frames",
        help="Video-MME multimodal pack recipe (OmniEvalKit MiniCPM default: minicpm-frames).",
    )
    p.add_argument(
        "--videomme-max-frames",
        type=int,
        default=None,
        help="Override max sampled frames (OmniEvalKit: 96 for minicpm-frames, 64 for interleave).",
    )
    p.add_argument(
        "--videomme-duration",
        choices=("all", "short", "medium", "long"),
        default="all",
        help="Filter Video-MME by duration bucket (default: all).",
    )
    p.add_argument(
        "--videomme-use-subtitle",
        action="store_true",
        help="Prepend subtitle text to the user prompt (Video-MME w/ subs).",
    )
    p.add_argument(
        "--videomme-inline-local-video",
        action="store_true",
        help="Embed frames/audio as base64 data URLs (no --allowed-local-media-path on the server).",
    )
    p.add_argument(
        "--videomme-extra-body-json",
        default='{"modalities":["text"],"chat_template_kwargs":{"enable_thinking":false}}',
        help="JSON merged into each chat request for Video-MME.",
    )
    p.add_argument(
        "--videomme-save-eval-items",
        action="store_true",
        help="Include per-request Video-MME accuracy rows in the saved JSON.",
    )
    p.add_argument(
        "--min-videomme-accuracy",
        type=float,
        default=float((os.environ.get("ACC_BENCH_MIN_VIDEOMME_ACCURACY") or "0.68").strip() or "0.68"),
        help="Fail when videomme_accuracy is strictly below this threshold (0–1). "
        "Default 0.68 (~2pp under MiniCPM-o 4.5 reported 70.4 w/o subs); "
        "override with ACC_BENCH_MIN_VIDEOMME_ACCURACY or pass 0 to disable the floor.",
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
    p.add_argument(
        "--seed-backend",
        default="openai-chat-omni",
        help="Seed-TTS benchmark backend (for example openai-realtime-tts).",
    )
    p.add_argument(
        "--seed-endpoint",
        default="/v1/chat/completions",
        help="Seed-TTS serving endpoint corresponding to --seed-backend.",
    )
    p.add_argument(
        "--seed-tts-turns-per-session",
        type=int,
        default=1,
        help="Group this many Seed-TTS target texts into each Realtime session.",
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
        default=0.35,
        help="If set, fail when seed_tts_content_error_mean is strictly above this value.",
    )
    p.add_argument(
        "--min-seed-tts-mean-sim",
        type=float,
        default=None,
        help="If set, fail when seed_tts_sim_mean is missing or strictly below this value.",
    )
    p.add_argument(
        "--min-seed-tts-mean-utmos",
        type=float,
        default=None,
        help="If set, fail when seed_tts_utmos_mean is missing or strictly below this value.",
    )
    return p


def parse_acc_benchmark_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse accuracy-bench args. Pytest callers pass ``argv``; ``None`` uses ``sys.argv[1:]``."""
    if argv is None:
        argv = sys.argv[1:]
    return build_arg_parser().parse_args(argv)


def run_acc_benchmark(ns: argparse.Namespace) -> int:
    """Run Daily-Omni / Video-MME / Seed-TTS client benches against a running server; return 0 on success."""
    failed: list[str] = []

    with _preserve_benchmark_dataset_env():
        sync_dataset_env_from_ns(ns)

        vllm = _omni_acc_bench_core().find_vllm_cli()
        print(f"Using vLLM CLI: {vllm}", flush=True)
        print(f"Repo root (for cwd reference): {_repo_root()}", flush=True)

        if not ns.skip_daily_omni:
            path = run_daily_omni(ns, vllm)
            print(f"\n[Daily-Omni] result JSON: {path}", flush=True)
            data = _omni_acc_bench_core().load_benchmark_result(path)
            _write_accuracy_result_metadata(
                path,
                data,
                {"min_daily_omni_accuracy": ns.min_daily_omni_accuracy},
                dataset="daily_omni",
            )
            errs = _validate_daily_omni(data, min_accuracy=ns.min_daily_omni_accuracy)
            if errs:
                failed.extend([f"[Daily-Omni] {e}" for e in errs])
            else:
                print(
                    f"[Daily-Omni] daily_omni_accuracy={data.get('daily_omni_accuracy')} "
                    f"evaluated_ok={data.get('daily_omni_evaluated_ok')}",
                    flush=True,
                )

        if not ns.skip_videomme:
            path = run_videomme(ns, vllm)
            print(f"\n[Video-MME] result JSON: {path}", flush=True)
            data = _omni_acc_bench_core().load_benchmark_result(path)
            _write_accuracy_result_metadata(
                path,
                data,
                {"min_videomme_accuracy": ns.min_videomme_accuracy},
                dataset="videomme",
            )
            errs = _validate_videomme(data, min_accuracy=ns.min_videomme_accuracy)
            if errs:
                failed.extend([f"[Video-MME] {e}" for e in errs])
            else:
                print(
                    f"[Video-MME] videomme_accuracy={data.get('videomme_accuracy')} "
                    f"evaluated_ok={data.get('videomme_evaluated_ok')} "
                    f"per_duration={data.get('videomme_per_duration_accuracy')}",
                    flush=True,
                )

        if not ns.skip_seed_tts:
            path = run_seed_tts(ns, vllm)
            print(f"\n[Seed-TTS] result JSON: {path}", flush=True)
            data = _omni_acc_bench_core().load_benchmark_result(path)
            _write_accuracy_result_metadata(
                path,
                data,
                {
                    "max_seed_tts_mean_wer": ns.max_seed_tts_mean_wer,
                    "min_seed_tts_mean_sim": ns.min_seed_tts_mean_sim,
                    "min_seed_tts_mean_utmos": ns.min_seed_tts_mean_utmos,
                },
                dataset="seed_tts",
            )
            errs = _validate_seed_tts(
                data,
                max_mean_wer=ns.max_seed_tts_mean_wer,
                min_mean_sim=ns.min_seed_tts_mean_sim,
                min_mean_utmos=ns.min_seed_tts_mean_utmos,
            )
            if ns.seed_tts_turns_per_session > 1:
                expected_turns = ns.num_prompts * ns.seed_tts_turns_per_session
                evaluated_turns = int(data.get("seed_tts_content_evaluated") or 0)
                if evaluated_turns != expected_turns:
                    errs.append(f"seed_tts_content_evaluated={evaluated_turns} != expected {expected_turns}")
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


DEFAULT_DEVICE_PROFILE = "default"


@dataclass(frozen=True, slots=True)
class SimilarityThresholds:
    """SSIM/PSNR floors for accuracy comparisons."""

    ssim: float
    psnr: float


def resolve_device_profile(
    device_name: str | None = None,
    *,
    profiles: Mapping[str, Any],
) -> str:
    """Map a device marketing name to a key in *profiles*.

    When *device_name* is omitted, uses ``current_omni_platform.get_device_name()``.
    Non-``default`` keys are matched as substrings; unmatched devices use ``default``.
    """
    if device_name is None:
        try:
            from vllm_omni.platforms import current_omni_platform

            device_name = current_omni_platform.get_device_name()
        except Exception:
            return DEFAULT_DEVICE_PROFILE

    if not device_name:
        return DEFAULT_DEVICE_PROFILE

    for profile in profiles:
        if profile == DEFAULT_DEVICE_PROFILE:
            continue
        if profile in device_name:
            return profile
    return DEFAULT_DEVICE_PROFILE


def resolve_similarity_thresholds(
    table: Mapping[str, SimilarityThresholds],
    device_profile: str | None = None,
    *,
    device_name: str | None = None,
) -> SimilarityThresholds:
    """Pick SSIM/PSNR thresholds from *table* for the current (or given) device profile."""
    if DEFAULT_DEVICE_PROFILE not in table:
        raise ValueError("threshold table must include a 'default' entry")
    profile = device_profile or resolve_device_profile(device_name, profiles=table)
    return table.get(profile, table[DEFAULT_DEVICE_PROFILE])


def resolve_device_threshold(
    thresholds: float | int | Mapping[str, float],
    *,
    device_name: str | None = None,
    label: str = "threshold",
) -> tuple[str, float]:
    """Resolve a scalar or per-GPU threshold for the current CUDA device.

    Returns ``(profile_key, threshold)``. A bare number uses profile ``default``.
    A mapping matches keys as substrings of the device name via
    ``resolve_device_profile`` (for example ``{"H100": 0.3, "B200": 0.36}``).
    Prefer an explicit ``default`` entry when unmatched GPUs should fall back;
    otherwise unmatched devices raise.
    """
    if isinstance(thresholds, float | int):
        return DEFAULT_DEVICE_PROFILE, float(thresholds)

    profile = resolve_device_profile(device_name, profiles=thresholds)
    if profile in thresholds:
        return profile, float(thresholds[profile])
    if DEFAULT_DEVICE_PROFILE in thresholds:
        return DEFAULT_DEVICE_PROFILE, float(thresholds[DEFAULT_DEVICE_PROFILE])
    known = ", ".join(str(key) for key in thresholds)
    raise RuntimeError(f"no {label} entry for device profile {profile!r}; known: {known}")


_SSIM_RE = re.compile(r"All:(?P<score>[0-9.]+)")
_PSNR_RE = re.compile(r"average:(?P<score>[0-9.]+)")


def parse_video_metadata(payload: dict[str, Any]) -> dict[str, int | float]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError(f"ffprobe payload did not include video streams: {payload}")

    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    fps = float(Fraction(stream["avg_frame_rate"]))
    frame_count_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_count_value is None:
        raise ValueError(f"ffprobe payload did not include frame count: {payload}")
    frame_count = int(frame_count_value)
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def parse_ssim_score(output: str) -> float:
    match = _SSIM_RE.search(output)
    if match is None:
        raise ValueError(f"Could not parse SSIM score from ffmpeg output:\n{output}")
    return float(match.group("score"))


def parse_psnr_score(output: str) -> float:
    match = _PSNR_RE.search(output)
    if match is None:
        raise ValueError(f"Could not parse PSNR score from ffmpeg output:\n{output}")
    return float(match.group("score"))


def probe_binary(binary: str, *, test_name: str = "video similarity e2e test") -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        pytest.skip(f"{binary} is required for {test_name}.")
    assert resolved is not None
    return resolved


def probe_video(path: Path) -> dict[str, int | float]:
    ffprobe = probe_binary("ffprobe")
    print(f"Probing video metadata: {path}", flush=True)
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
        timeout=MEDIA_TOOL_TIMEOUT_SECONDS,
    )
    return parse_video_metadata(json.loads(result.stdout))


def run_ffmpeg_similarity(filter_name: str, first: Path, second: Path) -> str:
    ffmpeg = probe_binary("ffmpeg")
    print(f"Running FFmpeg {filter_name} comparison: {first} vs {second}", flush=True)
    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-i",
            str(first),
            "-i",
            str(second),
            "-lavfi",
            f"[0:v][1:v]{filter_name}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
        stdin=subprocess.DEVNULL,
        timeout=MEDIA_TOOL_TIMEOUT_SECONDS,
    )
    return result.stderr


def online_timeout_seconds(configured: int | None, *, default: int = 1200) -> int:
    return int(configured) if configured is not None else default


def is_remote_image_source(source: str) -> bool:
    return source.startswith("http://") or source.startswith("https://")


def validate_image_source(source: str) -> None:
    if is_remote_image_source(source):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        return

    image_path = Path(source)
    if not image_path.exists():
        raise FileNotFoundError(f"Local image source does not exist: {image_path}")


def build_online_image_reference(source: str) -> str:
    if source.startswith("data:image"):
        return source

    if is_remote_image_source(source):
        response = requests.get(source, timeout=60)
        response.raise_for_status()
        mime_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if not mime_type.startswith("image/"):
            mime_type = guess_type(urlparse(source).path)[0] or "image/png"
        encoded = b64encode(response.content).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    image_path = Path(source)
    mime_type = guess_type(image_path.name)[0] or "image/png"
    encoded = b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def materialize_image_source(source: str, output_dir: Path) -> str:
    if not is_remote_image_source(source):
        return source

    from diffusers.utils import load_image

    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = Path(urlparse(source).path).name or "input.png"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
    if "." not in safe_name:
        safe_name = f"{safe_name}.png"
    image_path = output_dir / safe_name
    if not image_path.exists():
        image = load_image(source).convert("RGB")
        image.save(image_path)
    return str(image_path)


def video_artifact_dir(result_root: Path, source: str) -> Path:
    if is_remote_image_source(source):
        source_name = Path(urlparse(source).path).stem or "remote"
    else:
        source_name = Path(source).stem or "local"

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_name)
    digest = sha1(source.encode("utf-8")).hexdigest()[:8]
    return result_root / f"{safe_name}-{digest}"


def send_video_request_with_timeout(
    online_client,
    request_config: dict[str, Any],
    *,
    timeout_seconds: int,
) -> bytes:
    form_data = request_config.get("form_data")
    if not isinstance(form_data, dict):
        raise ValueError("Video request_config must contain 'form_data'")

    if not form_data.get("prompt"):
        raise ValueError("Video request_config['form_data'] must contain 'prompt'")

    normalized_form_data = {key: str(value) for key, value in form_data.items() if value is not None}

    files: dict[str, tuple[str, BytesIO, str]] = {}
    image_reference = request_config.get("image_reference")
    if image_reference:
        if image_reference.startswith("data:image"):
            header, encoded = image_reference.split(",", 1)
            content_type = header.split(";")[0].removeprefix("data:")
            extension = content_type.split("/")[-1]
            file_data = b64decode(encoded)
            files["input_reference"] = (
                f"reference.{extension}",
                BytesIO(file_data),
                content_type,
            )
        else:
            normalized_form_data["image_reference"] = json.dumps({"image_url": image_reference})

    start_time = time.perf_counter()
    create_url = online_client._build_url("/v1/videos")
    response = requests.post(
        create_url,
        data=normalized_form_data,
        files=files,
        headers={"Accept": "application/json"},
        timeout=60,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise requests.HTTPError(f"{exc}; response body: {response.text}", response=response) from exc
    job_data = response.json()
    video_id = job_data["id"]
    online_client._wait_until_video_completed(video_id, timeout_seconds=timeout_seconds)
    video_content = online_client._download_video_content(video_id)
    print(f"online_video_e2e_latency_s={time.perf_counter() - start_time:.3f}")
    return video_content


def assert_video_metadata(
    metadata: dict[str, int | float],
    *,
    width: int,
    height: int,
    fps: int | float,
    frame_count: int,
) -> None:
    assert metadata["width"] == width
    assert metadata["height"] == height
    assert metadata["fps"] == float(fps)
    assert metadata["frame_count"] == frame_count


def assert_video_similarity_metrics(
    *,
    label: str,
    online_path: Path,
    offline_path: Path,
    ssim_threshold: float,
    psnr_threshold: float,
) -> tuple[float, float]:
    ssim_output = run_ffmpeg_similarity("ssim", online_path, offline_path)
    psnr_output = run_ffmpeg_similarity("psnr", online_path, offline_path)
    ssim_score = parse_ssim_score(ssim_output)
    psnr_score = parse_psnr_score(psnr_output)

    print(f"{label} similarity metrics:")
    print(
        "  SSIM:"
        f" value={ssim_score:.6f},"
        f" threshold>={ssim_threshold:.6f},"
        " range=[0, 1],"
        " higher_is_better=True,"
        " interpretation=structural_similarity"
    )
    print(
        "  PSNR:"
        f" value={psnr_score:.6f} dB,"
        f" threshold>={psnr_threshold:.6f} dB,"
        " range=[0, +inf),"
        " higher_is_better=True,"
        " interpretation=pixel_error_in_decibels"
    )
    print(f"online_video={online_path}")
    print(f"offline_video={offline_path}")

    assert ssim_score >= ssim_threshold, (
        f"SSIM below threshold: got {ssim_score:.6f}, expected >= {ssim_threshold:.6f}. "
        f"online={online_path} offline={offline_path}"
    )
    assert psnr_score >= psnr_threshold, (
        f"PSNR below threshold: got {psnr_score:.6f}, expected >= {psnr_threshold:.6f}. "
        f"online={online_path} offline={offline_path}"
    )
    return ssim_score, psnr_score


def assert_similarity(
    *,
    model_name: str,
    vllm_image: Image.Image,
    diffusers_image: Image.Image,
    ssim_threshold: float,
    psnr_threshold: float,
    width: int | None = None,
    height: int | None = None,
    compare_mode: str = "RGB",
) -> None:
    requested_size = (width, height) if width is not None and height is not None else None
    if requested_size is not None and diffusers_image.size != requested_size:
        pytest.skip(
            "Skipping as diffusers baseline output is corrupt and not comparable: "
            f"dimensions do not match requested size; requested={requested_size}, got={diffusers_image.size}."
        )

    assert vllm_image.size == diffusers_image.size, (
        f"Online and diffusers output sizes mismatch: online={vllm_image.size}, diffusers={diffusers_image.size}"
    )

    ssim_score, psnr_score = compute_image_ssim_psnr(
        prediction=vllm_image,
        reference=diffusers_image,
        compare_mode=compare_mode,
    )
    assert_similarity_metrics(
        model_name=model_name,
        ssim_score=ssim_score,
        psnr_score=psnr_score,
        ssim_threshold=ssim_threshold,
        psnr_threshold=psnr_threshold,
    )


def assert_similarity_metrics(
    *,
    model_name: str,
    ssim_score: float,
    psnr_score: float,
    ssim_threshold: float,
    psnr_threshold: float,
) -> None:
    """Check computed image metrics, allowing callers to save all evidence first."""
    print(f"{model_name} similarity metrics:")
    print(f"  SSIM: value={ssim_score:.6f}, threshold>={ssim_threshold:.6f}, range=[-1, 1], higher_is_better=True")
    print(
        f"  PSNR: value={psnr_score:.6f} dB, threshold>={psnr_threshold:.6f} dB, range=[0, +inf), higher_is_better=True"
    )

    assert ssim_score >= ssim_threshold, (
        f"SSIM below threshold for {model_name}: got {ssim_score:.6f}, expected >= {ssim_threshold:.6f}."
    )
    assert psnr_score >= psnr_threshold, (
        f"PSNR below threshold for {model_name}: got {psnr_score:.6f}, expected >= {psnr_threshold:.6f}."
    )


def assert_image_sequence_similarity(
    *,
    model_name: str,
    vllm_images: list[Image.Image],
    diffusers_images: list[Image.Image],
    ssim_threshold: float,
    psnr_threshold: float,
    compare_mode: str = "RGB",
) -> None:
    assert len(vllm_images) == len(diffusers_images), (
        f"Output image count mismatch for {model_name}: online={len(vllm_images)}, diffusers={len(diffusers_images)}"
    )
    for index, (vllm_image, diffusers_image) in enumerate(zip(vllm_images, diffusers_images, strict=True), start=1):
        assert_similarity(
            model_name=f"{model_name}[layer={index}]",
            vllm_image=vllm_image,
            diffusers_image=diffusers_image,
            ssim_threshold=ssim_threshold,
            psnr_threshold=psnr_threshold,
            compare_mode=compare_mode,
        )


def assert_images_pixel_close(
    *,
    model_name: str,
    vllm_image: Image.Image,
    diffusers_image: Image.Image,
    mean_threshold: float,
    p99_threshold: float,
    compare_mode: str = "RGB",
) -> None:
    """Assert full-image pixel closeness between online serving and diffusers output.

    Match the threshold style used by diffusion parallelism tests: convert both
    images to float32 RGB tensors in the [0, 1] range, then compare mean and p99
    absolute channel differences.

    Accuracy improves as both values move lower:
    - ``mean_abs_diff`` tracks global average drift across the whole image.
    - ``p99_abs_diff`` tracks tail drift while ignoring the noisiest 1% of
      channel samples.

    The printed mismatch ratios are diagnostics only. ``pixel_ratio`` counts
    pixels where any channel exceeds a tolerance, while ``channel_ratio`` counts
    individual channel samples above that tolerance.
    """
    assert vllm_image.size == diffusers_image.size, (
        f"Online and diffusers output sizes mismatch: online={vllm_image.size}, diffusers={diffusers_image.size}"
    )

    vllm_array = np.asarray(vllm_image.convert(compare_mode), dtype=np.float32) / 255.0
    diffusers_array = np.asarray(diffusers_image.convert(compare_mode), dtype=np.float32) / 255.0
    channel_abs_diff = np.abs(vllm_array - diffusers_array)
    mean_abs_diff = float(channel_abs_diff.mean())
    p99_abs_diff = float(np.quantile(channel_abs_diff, 0.99))
    percentiles = {
        percentile: float(np.quantile(channel_abs_diff, percentile / 100.0)) for percentile in (50, 90, 95, 99, 99.9)
    }

    print(f"{model_name} pixel metrics:")
    print(
        f"  mean_abs_diff: value={mean_abs_diff:.6e}, threshold<={mean_threshold:.6e}, "
        "range=[0, 1], lower_is_better=True"
    )
    print(
        f"  p99_abs_diff: value={p99_abs_diff:.6e}, threshold<={p99_threshold:.6e}, range=[0, 1], lower_is_better=True"
    )
    print("  abs_diff_percentiles:")
    for percentile, value in percentiles.items():
        print(f"    p{percentile}: value={value:.6e}, range=[0, 1], lower_is_better=True")
    print("  mismatch_ratios_by_channel_threshold:")
    for tolerance in (0, 1, 2, 4, 8, 16, 32, 64, 128):
        normalized_tolerance = tolerance / 255.0
        pixel_mismatch = np.any(channel_abs_diff > normalized_tolerance, axis=-1)
        pixel_mismatch_ratio = float(np.count_nonzero(pixel_mismatch) / pixel_mismatch.size)
        channel_mismatch_ratio = float(
            np.count_nonzero(channel_abs_diff > normalized_tolerance) / channel_abs_diff.size
        )
        print(
            f"    threshold>{normalized_tolerance:.6e} ({tolerance}/255): pixel_ratio={pixel_mismatch_ratio:.8f}, "
            f"channel_ratio={channel_mismatch_ratio:.8f}"
        )

    assert mean_abs_diff <= mean_threshold and p99_abs_diff <= p99_threshold, (
        f"Image diff exceeded threshold for {model_name}: mean_abs_diff={mean_abs_diff:.6e}, "
        f"p99_abs_diff={p99_abs_diff:.6e} (thresholds: mean<={mean_threshold:.6e}, "
        f"p99<={p99_threshold:.6e})"
    )


def compute_image_ssim_psnr(
    *,
    prediction: Image.Image,
    reference: Image.Image,
    compare_mode: str = "RGB",
) -> tuple[float, float]:
    pred_tensor = _pil_to_batched_tensor(prediction, compare_mode=compare_mode)
    ref_tensor = _pil_to_batched_tensor(reference, compare_mode=compare_mode)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)

    ssim_value = float(ssim_metric(pred_tensor, ref_tensor).item())
    psnr_value = float(psnr_metric(pred_tensor, ref_tensor).item())
    return ssim_value, psnr_value


class CLIPScorer:
    def __init__(self, model_name_or_path: str = "openai/clip-vit-base-patch16"):
        from transformers import CLIPModel, CLIPProcessor

        self._model = CLIPModel.from_pretrained(model_name_or_path)
        self._processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self._model.eval()

    def score(self, image: Image.Image, text: str) -> float:
        inputs = self._processor(text=[text], images=[image], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
        img_emb = outputs.image_embeds
        txt_emb = outputs.text_embeds
        img_emb = img_emb / img_emb.norm(p=2, dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(p=2, dim=-1, keepdim=True)
        similarity = (img_emb * txt_emb).sum(dim=-1)
        return float(similarity.item() * 100)

    def image_image_score(self, image1: Image.Image, image2: Image.Image) -> float:
        inputs = self._processor(text=[""], images=[image1, image2], return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model(**inputs)
        img_emb1, img_emb2 = outputs.image_embeds.chunk(2)
        img_emb1 = img_emb1 / img_emb1.norm(p=2, dim=-1, keepdim=True)
        img_emb2 = img_emb2 / img_emb2.norm(p=2, dim=-1, keepdim=True)
        similarity = (img_emb1 * img_emb2).sum(dim=-1)
        return float(similarity.item() * 100)

    def assert_score(self, *, model_name: str, image: Image.Image, text: str, threshold: float) -> None:
        value = self.score(image, text)
        print(f"{model_name} CLIP score:")
        print(f"  CLIPScore: value={value:.4f}, threshold>={threshold:.4f}, higher_is_better=True")
        assert value >= threshold, (
            f"CLIP score below threshold for {model_name}: got {value:.4f}, expected >= {threshold:.4f}."
        )


def _pil_to_clip_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


def _pil_to_batched_tensor(image: Image.Image, *, compare_mode: str) -> torch.Tensor:
    array = np.asarray(image.convert(compare_mode), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor


class SemanticSimilarityScorer:
    """Semantic similarity scorer for text-text and image-text comparisons"""

    def __init__(
        self,
        text_model_name: str = "BAAI/bge-m3",
    ):
        """
        Initialize semantic similarity scorer

        Args:
            text_model_name: Model for text-text similarity (BGE-M3)
        """
        self._text_model = None
        self._text_model_name = text_model_name

    def _load_text_model(self):
        """Lazy load text similarity model (BGE-M3)"""
        if self._text_model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel

                print(f"Loading text similarity model: {self._text_model_name}")
                self._text_model = BGEM3FlagModel(self._text_model_name, use_fp16=True)
            except ImportError:
                raise ImportError("FlagEmbedding not installed. Install with: pip install FlagEmbedding")
        return self._text_model

    def text_similarity(self, text1: str, text2: str) -> dict[str, float | int]:
        """
        Compute semantic similarity between two texts using BGE-M3
        """
        results: dict[str, float | int] = {}
        # 1. Text prefix matching (maximum matching character length)
        match_count = 0
        for char_vllm, char_ref in zip(text1, text2):
            if char_vllm == char_ref:
                match_count += 1
            else:
                break
        results["text_prefix_match_count"] = match_count

        # 2. Full CoT semantic similarity comparison
        model = self._load_text_model()
        emb1 = model.encode([text1], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0]
        emb2 = model.encode([text2], return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"][0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        results["cot_semantic_sim"] = float(similarity)
        return results

    def release_models(self):
        """Release loaded models to free memory"""
        if self._text_model is not None:
            del self._text_model
            self._text_model = None


def download_images(urls: list[str]) -> list[Image.Image]:
    """Download and cache multiple images"""
    images = []
    for i, url in enumerate(urls):
        print(f"Downloading image {i} from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        images.append(image)

    return images
