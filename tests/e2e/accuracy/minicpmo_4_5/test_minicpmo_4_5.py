# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o 4.5 Daily-Omni + Seed-TTS accuracy regression coverage.

Daily-Omni settings follow the MiniCPM interleaved AV recipe that reaches
~78% overall accuracy on Daily-Omni (``minicpm-interleave``, ``temperature=0``,
text modalities with ``use_tts_template``, and server ``--interleave-mm-strings``
+ 1fps / 128-frame media-io kwargs, also pinned in ``minicpmo_4_5.yaml``).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from tests.e2e.accuracy.qwen3_omni import run_qwen_omni_acc_benchmark as _acc_bench
from tests.e2e.accuracy.qwen3_omni.qwen3_omni_acc_bench_core import (
    build_acc_benchmark_cli_argv,
    find_vllm_cli,
)
from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import SERVER_PARAMS as DUPLEX_TEST_PARAMS
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

_MODEL = os.environ.get("VLLM_TEST_MINICPMO_4_5_MODEL", "openbmb/MiniCPM-o-4_5")
_DEPLOY_CONFIG = get_deploy_config_path("minicpmo_4_5.yaml")
_RESULT_DIR = Path(
    os.environ.get(
        "ACC_BENCH_RESULT_DIR",
        str(Path(__file__).resolve().parent / "results"),
    )
)

_MIN_DAILY_OMNI_ACCURACY = 0.78
_MAX_SEED_TTS_MEAN_WER = 0.05
# Match the validated Daily-Omni client body from daily_omni_bench.sh.
_DAILY_EXTRA_BODY = {
    "modalities": ["text"],
    "chat_template_kwargs": {
        "enable_thinking": False,
        "use_tts_template": True,
    },
}
_SEED_EXTRA_BODY = {
    "modalities": ["text", "audio"],
    "chat_template_kwargs": {
        "enable_thinking": False,
        "use_tts_template": True,
    },
}
# Server flags required for MiniCPM interleaved image/audio packs (Daily-Omni only).
# Do not reuse these for Seed-TTS: ``--interleave-mm-strings`` + TTS ref_audio can trip
# msgspec ValidationError and kill the orchestrator mid-suite.
# Also pinned in ``minicpmo_4_5.yaml`` stage 0 so they survive config-path routing.
_DAILY_OMNI_SERVER_ARGS = [
    "--trust-remote-code",
    "--interleave-mm-strings",
    "--media-io-kwargs",
    '{"video":{"fps":1,"num_frames":128}}',
]
_SEED_TTS_SERVER_ARGS = [
    "--trust-remote-code",
]

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

# Evaluated at setup time, before the ``omni_server`` fixture boots a server. An
# ``importorskip`` inside the test body would skip only after paying several minutes
# of startup, and the resulting skip is easy to miss in a nightly log.
_missing_daily_omni_deps = [name for name in ("datasets", "huggingface_hub") if importlib.util.find_spec(name) is None]
requires_daily_omni_deps = pytest.mark.skipif(
    bool(_missing_daily_omni_deps),
    reason=f"Daily-Omni accuracy bench needs {', '.join(_missing_daily_omni_deps)}",
)

daily_test_params = [
    OmniServerParams(
        model=_MODEL,
        stage_config_path=_DEPLOY_CONFIG,
        server_args=list(_DAILY_OMNI_SERVER_ARGS),
    )
]

seed_test_params = [
    OmniServerParams(
        model=_MODEL,
        stage_config_path=_DEPLOY_CONFIG,
        server_args=list(_SEED_TTS_SERVER_ARGS),
    )
]


def _require_vllm_cli() -> None:
    try:
        find_vllm_cli()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


@pytest.fixture(autouse=True)
def _inline_daily_omni_media(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inline cached Hub/local videos because the test server has no local-media allowlist."""
    original = _acc_bench.daily_omni_bench_argv

    def _wrapped() -> list[str]:
        argv = list(original())
        if "--daily-omni-inline-local-video" not in argv:
            argv.append("--daily-omni-inline-local-video")
        return argv

    monkeypatch.setattr(_acc_bench, "daily_omni_bench_argv", _wrapped)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@requires_daily_omni_deps
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", daily_test_params, indirect=True)
def test_minicpmo_4_5_daily_omni_accuracy_bench(omni_server) -> None:
    _require_vllm_cli()

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
            "--temperature",
            "0",
            "--output-len",
            "512",
            "--daily-omni-input-mode",
            "all",
            "--daily-omni-pack-mode",
            "minicpm-interleave",
            "--daily-extra-body-json",
            json.dumps(_DAILY_EXTRA_BODY, separators=(",", ":")),
            "--trust-remote-code",
        ]
    )

    assert _acc_bench.run_acc_benchmark(_acc_bench.parse_acc_benchmark_args(argv)) == 0


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", seed_test_params, indirect=True)
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
            json.dumps(_SEED_EXTRA_BODY, separators=(",", ":")),
            "--trust-remote-code",
        ]
    )

    assert _acc_bench.run_acc_benchmark(_acc_bench.parse_acc_benchmark_args(argv)) == 0


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", DUPLEX_TEST_PARAMS, indirect=True)
def test_minicpmo_4_5_duplex_seed_tts_wer_bench(omni_server) -> None:
    """Gate Seed-TTS WER through the explicit Realtime TTS contract."""
    _require_vllm_cli()
    pytest.importorskip("huggingface_hub")

    argv = build_acc_benchmark_cli_argv(
        omni_server,
        skip_seed=False,
        skip_daily=True,
        num_prompts=50,
        max_concurrency=1,
    )
    argv.extend(
        [
            "--result-dir",
            str(_RESULT_DIR),
            "--max-seed-tts-mean-wer",
            str(_MAX_SEED_TTS_MEAN_WER),
            "--seed-extra-body-json",
            json.dumps({"save_duplex_request_metrics": True}, separators=(",", ":")),
            "--seed-backend",
            "openai-realtime-tts",
            "--seed-endpoint",
            "/v1/realtime",
            "--seed-tts-turns-per-session",
            "4",
            "--trust-remote-code",
        ]
    )

    assert _acc_bench.run_acc_benchmark(_acc_bench.parse_acc_benchmark_args(argv)) == 0
