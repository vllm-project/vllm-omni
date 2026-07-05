# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Qwen3-Omni Daily-Omni / Seed-TTS ``vllm bench serve --omni`` accuracy runs.

Local dataset paths are **optional**. When ``VLLM_DAILY_OMNI_QA_JSON`` + ``VLLM_DAILY_OMNI_VIDEO_DIR``
point to existing files, those are used with inline video. Otherwise the benchmark falls back to
the HuggingFace dataset id (``liarliar/Daily-Omni``); QA loads via ``datasets``, and the first
bench request that needs media downloads ``Videos.tar`` from the Hub when no video dir is set.

Similarly for Seed-TTS: a local directory wins; otherwise ``--dataset-path`` uses the Hub id
and ``huggingface_hub.snapshot_download`` inside ``resolve_seed_tts_root`` pulls files on demand.

Use :func:`build_acc_benchmark_cli_argv` to assemble ``argv`` for a live Omni server (host/port/model
and small bench defaults) before ``parse_args`` / ``run_acc_benchmark`` in the accuracy driver.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Protocol

DEFAULT_DAILY_OMNI_HF_REPO = "liarliar/Daily-Omni"
DEFAULT_SEED_TTS_HF_REPO = "zhaochenyang20/seed-tts-eval"


class OmniBenchServerEndpoint(Protocol):
    """Anything with ``host`` / ``port`` / ``model`` (e.g. :class:`tests.conftest.OmniServer`)."""

    host: str
    port: int
    model: str


def build_acc_benchmark_cli_argv(
    server: OmniBenchServerEndpoint,
    *,
    skip_seed: bool,
    skip_daily: bool,
    num_prompts: int | None = None,
    max_concurrency: int | None = None,
) -> list[str]:
    """Prefix argv for :func:`run_qwen_omni_acc_benchmark.parse_acc_benchmark_args` + :func:`run_acc_benchmark`.

    Wires ``--host`` / ``--port`` / ``--model`` to a running Omni OpenAI server, sets small
    ``--num-prompts`` / ``--max-concurrency`` defaults (overridable via ``ACC_BENCH_NUM_PROMPTS`` /
    ``ACC_BENCH_MAX_CONCURRENCY``), and when Daily-Omni runs adds ``--daily-omni-repo`` so Hub QA
    matches :func:`daily_omni_bench_argv` once ``run_acc_benchmark`` mirrors ``--daily-omni-repo`` into env.
    """
    n_prompts = int(os.environ.get("ACC_BENCH_NUM_PROMPTS", "2000")) if num_prompts is None else int(num_prompts)
    n_conc = int(os.environ.get("ACC_BENCH_MAX_CONCURRENCY", "10")) if max_concurrency is None else int(max_concurrency)
    argv = [
        "--host",
        server.host,
        "--port",
        str(server.port),
        "--model",
        server.model,
        "--num-prompts",
        str(n_prompts),
        "--max-concurrency",
        str(n_conc),
    ]
    if not skip_daily:
        repo = os.environ.get("VLLM_DAILY_OMNI_REPO", DEFAULT_DAILY_OMNI_HF_REPO).strip() or DEFAULT_DAILY_OMNI_HF_REPO
        argv.extend(["--daily-omni-repo", repo])
    if skip_seed:
        argv.append("--skip-seed-tts")
    if skip_daily:
        argv.append("--skip-daily-omni")
    return argv


def daily_omni_bench_argv() -> list[str]:
    """CLI args for Daily-Omni (after ``vllm bench serve --omni``)."""
    qa = os.environ.get("VLLM_DAILY_OMNI_QA_JSON", "").strip()
    vd = os.environ.get("VLLM_DAILY_OMNI_VIDEO_DIR", "").strip()
    if qa and vd:
        qap = Path(qa).expanduser()
        vdp = Path(vd).expanduser()
        if qap.is_file() and vdp.is_dir():
            return [
                "--dataset-name",
                "daily-omni",
                "--daily-omni-qa-json",
                str(qap.resolve()),
                "--daily-omni-video-dir",
                str(vdp.resolve()),
                "--daily-omni-inline-local-video",
            ]
    repo = os.environ.get("VLLM_DAILY_OMNI_REPO", DEFAULT_DAILY_OMNI_HF_REPO).strip() or DEFAULT_DAILY_OMNI_HF_REPO
    return [
        "--dataset-name",
        "daily-omni",
        "--dataset-path",
        repo,
    ]


def seed_tts_bench_argv(*, locale: str = "en") -> list[str]:
    """CLI args for Seed-TTS (after ``vllm bench serve --omni``)."""
    dp = os.environ.get("VLLM_SEED_TTS_DATASET_PATH", "").strip()
    if dp:
        p = Path(dp).expanduser()
        # Preserve Hugging Face repo ids verbatim. Only canonicalize to an
        # absolute path when the value actually exists as a local directory.
        dataset_path = str(p.resolve()) if p.exists() and p.is_dir() else dp
    else:
        dataset_path = (
            os.environ.get("VLLM_SEED_TTS_REPO", DEFAULT_SEED_TTS_HF_REPO).strip() or DEFAULT_SEED_TTS_HF_REPO
        )
    out = ["--dataset-name", "seed-tts", "--dataset-path", dataset_path]
    root = os.environ.get("SEED_TTS_ROOT", "").strip()
    if root:
        out.extend(["--seed-tts-root", str(Path(root).expanduser().resolve())])
    out.extend(["--seed-tts-locale", locale])
    return out


def find_vllm_cli() -> str:
    exe = shutil.which("vllm")
    if not exe:
        raise FileNotFoundError("Could not find `vllm` on PATH (install vLLM-Omni with CLI entrypoints).")
    return exe


def run_vllm_bench_subprocess(vllm: str, argv: list[str], *, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    subprocess.run([vllm, *argv], env=env, check=True)


def load_benchmark_result(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def build_serve_common_argv(
    *,
    host: str,
    port: int,
    model: str,
    num_prompts: int,
    max_concurrency: int,
    num_warmups: int,
    percentile_metrics: str,
    result_dir: Path,
    result_filename: str,
    ready_check_timeout_sec: int | None = None,
) -> list[str]:
    out = [
        "bench",
        "serve",
        "--omni",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat-omni",
        "--request-rate",
        "inf",
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(max_concurrency),
        "--no-oversample",
        "--num-warmups",
        str(num_warmups),
        "--percentile-metrics",
        percentile_metrics,
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]
    if ready_check_timeout_sec is not None:
        out.extend(["--ready-check-timeout-sec", str(int(ready_check_timeout_sec))])
    return out


def assert_daily_omni_scored(result: dict[str, Any]) -> None:
    acc = result.get("daily_omni_accuracy")
    assert acc is not None, "daily_omni_accuracy missing — wrong dataset or benchmark wiring"
    assert int(result.get("daily_omni_evaluated_ok", 0) or 0) > 0, "no successful MCQ rows (daily_omni_evaluated_ok==0)"


def assert_seed_tts_scored(result: dict[str, Any]) -> None:
    err = result.get("seed_tts_eval_setup_error")
    assert not err, f"Seed-TTS eval deps/setup failed: {err}"
    assert int(result.get("seed_tts_content_evaluated", 0) or 0) > 0, (
        "seed_tts_content_evaluated==0 — enable WER eval and check PCM capture / modalities"
    )
