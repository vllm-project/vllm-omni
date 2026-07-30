# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o 4.5 Daily-Omni + Seed-TTS accuracy regression coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.e2e.accuracy.qwen3_omni import run_qwen_omni_acc_benchmark as _acc_bench
from tests.e2e.accuracy.qwen3_omni.qwen3_omni_acc_bench_core import (
    build_acc_benchmark_cli_argv,
    find_vllm_cli,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

_MODEL = "/data/models/MiniCPM-o-4_5"
_DEPLOY_CONFIG = get_deploy_config_path("minicpmo_4_5.yaml")
_RESULT_DIR = Path(__file__).resolve().parent / "results"
_MIN_DAILY_OMNI_ACCURACY = 0.64
_MAX_SEED_TTS_MEAN_WER = 0.05
_CHAT_EXTRA_BODY = {
    "modalities": ["text", "audio"],
    "chat_template_kwargs": {
        "enable_thinking": False,
        "use_tts_template": True,
    },
}

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

test_params = [
    OmniServerParams(
        model=_MODEL,
        stage_config_path=_DEPLOY_CONFIG,
        server_args=["--trust-remote-code"],
    )
]


def _require_vllm_cli() -> None:
    try:
        find_vllm_cli()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(autouse=True)
def _inline_daily_omni_media(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inline cached Hub videos because the test server has no local-media allowlist."""
    original = _acc_bench.daily_omni_bench_argv

    def _wrapped() -> list[str]:
        argv = list(original())
        if "--daily-omni-inline-local-video" not in argv:
            argv.append("--daily-omni-inline-local-video")
        return argv

    monkeypatch.setattr(_acc_bench, "daily_omni_bench_argv", _wrapped)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_minicpmo_4_5_daily_omni_accuracy_bench(omni_server) -> None:
    _require_vllm_cli()
    pytest.importorskip("datasets")
    pytest.importorskip("huggingface_hub")

    argv = build_acc_benchmark_cli_argv(
        omni_server,
        skip_seed=True,
        skip_daily=False,
    )
    argv.extend(
        [
            "--result-dir",
            str(_RESULT_DIR),
            "--min-daily-omni-accuracy",
            str(_MIN_DAILY_OMNI_ACCURACY),
            "--daily-extra-body-json",
            json.dumps(_CHAT_EXTRA_BODY, separators=(",", ":")),
            "--trust-remote-code",
        ]
    )

    assert _acc_bench.run_acc_benchmark(_acc_bench.parse_acc_benchmark_args(argv)) == 0


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_minicpmo_4_5_seed_tts_wer_bench(omni_server) -> None:
    _require_vllm_cli()
    pytest.importorskip("huggingface_hub")

    argv = build_acc_benchmark_cli_argv(
        omni_server,
        skip_seed=False,
        skip_daily=True,
    )
    argv.extend(
        [
            "--result-dir",
            str(_RESULT_DIR),
            "--max-seed-tts-mean-wer",
            str(_MAX_SEED_TTS_MEAN_WER),
            "--seed-extra-body-json",
            json.dumps(_CHAT_EXTRA_BODY, separators=(",", ":")),
            "--trust-remote-code",
        ]
    )

    assert _acc_bench.run_acc_benchmark(_acc_bench.parse_acc_benchmark_args(argv)) == 0
