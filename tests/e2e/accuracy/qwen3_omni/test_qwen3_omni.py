# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni accuracy benchmarks (Daily-Omni MCQ + Seed-TTS WER) via ``vllm bench serve --omni``.

Starts a **module-scoped** Omni OpenAI-compatible server (same pattern as ``tests/dfx/perf`` and
``tests/e2e/online_serving/test_qwen3_omni.py``), then runs the client benches against
``omni_server.host`` / ``omni_server.port`` / ``omni_server.model``.

**Daily-Omni from Hugging Face:** unless ``VLLM_DAILY_OMNI_QA_JSON`` and ``VLLM_DAILY_OMNI_VIDEO_DIR``
point at a full local tree, the bench uses ``--dataset-path`` (default ``liarliar/Daily-Omni`` via
``VLLM_DAILY_OMNI_REPO`` / ``--daily-omni-repo``). QA loads through ``datasets``; ``Videos.tar`` is
downloaded and extracted under ``HF_HOME`` on demand. The tests patch in
``--daily-omni-inline-local-video`` so multimodal payloads use data URLs (no
``--allowed-local-media-path`` on the server). Use small ``--num-prompts`` defaults suitable for CI
(override with ``ACC_BENCH_NUM_PROMPTS`` / ``ACC_BENCH_MAX_CONCURRENCY``; see
:func:`tests.e2e.accuracy.qwen3_omni.qwen3_omni_acc_bench_core.build_acc_benchmark_cli_argv`).

This package lives under ``tests/e2e/accuracy/qwen3_omni/``, so pytest still loads
``tests/e2e/accuracy/conftest.py``, which imports ``tests.conftest`` (heavy deps: ``vllm``, ``torch``, …).
A broken or partial install can therefore **fail during collection** before these tests run.

If ``vllm`` is not on ``PATH``, the tests **skip** instead of erroring. Without
``VLLM_SKIP_ACC_BENCH=1``, a failed bench still yields a **failed** run (non-zero subprocess exit).

Run::

    pytest -sv tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py

Only the subprocess accuracy marker::

    pytest -sv tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py -m qwen3_omni_acc

Skip when you do not have GPUs, a server, or datasets (CI opt-out)::

    VLLM_SKIP_ACC_BENCH=1 pytest -sv tests/e2e/accuracy/qwen3_omni/test_qwen3_omni.py

Standalone CLI (expects a server already up; uses ``ACC_BENCH_*`` env defaults)::

    python tests/e2e/accuracy/qwen3_omni/run_qwen_omni_acc_benchmark.py --help
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.e2e.accuracy.qwen3_omni import run_qwen_omni_acc_benchmark as _acc_bench
from tests.e2e.accuracy.qwen3_omni.qwen3_omni_acc_bench_core import (
    build_acc_benchmark_cli_argv,
    find_vllm_cli,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.platforms import current_omni_platform

_E2E_ROOT = Path(__file__).resolve().parent.parent.parent

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

_CI_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe.yaml")


def get_chunk_config(config_path: str | None = None):
    """Load the qwen3_omni CI deploy yaml with async_chunk modifications for streaming mode."""
    if config_path is None:
        config_path = _CI_DEPLOY
    # TODO: remove this workaround once legacy `stage_args` path is deleted.
    # The pipeline (qwen3_omni/pipeline.py) already wires
    # thinker2talker_async_chunk / talker2code2wav_async_chunk on stage 0/1,
    # so only async_chunk needs flipping. Writing nested `engine_args:` into
    # the new-schema overlay trips _parse_stage_deploy's legacy branch and
    # drops flat fields (load_format, max_num_seqs, ...).
    return modify_stage_config(config_path, updates={"async_chunk": True})


if current_omni_platform.is_xpu():
    stage_configs = [_CI_DEPLOY]
else:  # CUDA + ROCm MI325 share the same deploy config
    stage_configs = [get_chunk_config()]

test_params = [
    OmniServerParams(model=model, stage_config_path=stage_config) for model in models for stage_config in stage_configs
]


def _require_vllm_cli() -> None:
    try:
        find_vllm_cli()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(autouse=True)
def _daily_omni_hub_inline_media(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hub / lazy-cache mode uses local files → default ``file://`` needs server allowlist.

    ``run_qwen_omni_acc_benchmark`` binds ``daily_omni_bench_argv`` at import time; patch that copy
    so we append ``--daily-omni-inline-local-video`` whenever the core helper did not already set it
    (local qa.json + video-dir mode already passes the flag).
    """
    orig = _acc_bench.daily_omni_bench_argv

    def _wrapped() -> list[str]:
        out = list(orig())
        if "--daily-omni-inline-local-video" not in out:
            out.append("--daily-omni-inline-local-video")
        return out

    monkeypatch.setattr(_acc_bench, "daily_omni_bench_argv", _wrapped)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_TEST_CLEAN_GPU_MEMORY", "0")


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_qwen3_omni_daily_omni_accuracy_bench(omni_server) -> None:
    _require_vllm_cli()
    pytest.importorskip("datasets")
    pytest.importorskip("huggingface_hub")
    ns = _acc_bench.parse_acc_benchmark_args(
        build_acc_benchmark_cli_argv(omni_server, skip_seed=True, skip_daily=False)
    )
    assert _acc_bench.run_acc_benchmark(ns) == 0


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_qwen3_omni_seed_tts_wer_bench(omni_server) -> None:
    _require_vllm_cli()
    pytest.importorskip("huggingface_hub")
    ns = _acc_bench.parse_acc_benchmark_args(
        build_acc_benchmark_cli_argv(omni_server, skip_seed=False, skip_daily=True)
    )
    assert _acc_bench.run_acc_benchmark(ns) == 0
