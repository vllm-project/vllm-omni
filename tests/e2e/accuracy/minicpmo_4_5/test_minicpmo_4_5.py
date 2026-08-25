# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o 4.5 Daily-Omni + Seed-TTS + Video-MME accuracy regression coverage.

Daily-Omni settings follow the MiniCPM interleaved AV recipe that reaches
~78% overall accuracy on Daily-Omni (``minicpm-interleave``, ``temperature=0``,
text modalities with ``use_tts_template``, and server ``--interleave-mm-strings``
+ 1fps / 128-frame media-io kwargs, also pinned in ``minicpmo_4_5.yaml``).

Video-MME settings follow OpenBMB OmniEvalKit MiniCPM ``videomme`` (w/o subs):
``minicpm-frames``, ``max_frames=96``, ``temperature=0``, ``output_len=128``.
Official MiniCPM-o 4.5 reports **70.4**; the gate is **0.68** (~2pp margin).
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
from tests.helpers.mark import hardware_test
from tests.helpers.minicpmo_4_5_duplex import SERVER_PARAMS as DUPLEX_TEST_PARAMS
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
# MiniCPM-o 4.5 reports 70.4 on Video-MME (w/o subs); ~2pp margin like Daily-Omni.
_MIN_VIDEOMME_ACCURACY = float(os.environ.get("ACC_BENCH_MIN_VIDEOMME_ACCURACY", "0.68"))
_MAX_SEED_TTS_MEAN_WER = 0.0156
_MIN_SEED_TTS_MEAN_SIM = 0.689
# Full Seed-TTS zh split (seed-tts-eval/zh/meta.lst) is 2020 rows.
_SEED_TTS_LOCALE = "zh"
_SEED_TTS_NUM_PROMPTS = 2020
# Match the validated Daily-Omni / Video-MME client body from the MiniCPM run scripts.
_DAILY_EXTRA_BODY = {
    "modalities": ["text"],
    "chat_template_kwargs": {"enable_thinking": False},
}
_VIDEOMME_EXTRA_BODY = {
    "modalities": ["text"],
    "chat_template_kwargs": {"enable_thinking": False},
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
# Video-MME ``minicpm-frames`` sends sampled frames as image_url (no AV interleave).
# Allow local ``file://`` frame-cache URLs so CI does not need megabyte-scale base64
# payloads; the autouse fixture still appends ``--videomme-inline-local-video`` when
# the allowlist is unavailable.
_VIDEOMME_SERVER_ARGS = [
    "--trust-remote-code",
    "--allowed-local-media-path",
    os.environ.get("VLLM_TEST_ALLOWED_LOCAL_MEDIA_PATH", "/"),
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

videomme_test_params = [
    OmniServerParams(
        model=_MODEL,
        stage_config_path=_DEPLOY_CONFIG,
        server_args=list(_VIDEOMME_SERVER_ARGS),
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


def _optional_local_videomme_dataset_path() -> str | None:
    """Return an explicit local Video-MME root, or ``None`` to use the Hub default.

    Same rule as Daily-Omni / Seed-TTS: only an env-provided existing directory counts as
    local. Hard-coded workspace paths are intentionally not probed so CI defaults to the
    Hub id in ``VIDEOMME_DEFAULT_HF_REPO`` (``lmms-eval/Video-MME``).
    """
    for key in ("VLLM_VIDEOMME_DATASET_PATH", "VIDEOMME_ROOT"):
        raw = os.environ.get(key, "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_absolute() and path.is_dir():
            return str(path.resolve())
    return None


@pytest.fixture(autouse=True)
def _inline_local_media_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inline Hub/local media when the server has no usable local-media allowlist.

    Daily-Omni always inlines (matches the historical Qwen/MiniCPM accuracy fixture).
    Video-MME prefers ``file://`` via ``--allowed-local-media-path`` (see
    ``_VIDEOMME_SERVER_ARGS``); force inline only when ``VLLM_VIDEOMME_FORCE_INLINE=1``.
    """
    original_daily = _acc_bench.daily_omni_bench_argv
    original_videomme = _acc_bench.videomme_bench_argv

    def _wrap_daily() -> list[str]:
        argv = list(original_daily())
        if "--daily-omni-inline-local-video" not in argv:
            argv.append("--daily-omni-inline-local-video")
        return argv

    def _wrap_videomme() -> list[str]:
        argv = list(original_videomme())
        force_inline = os.environ.get("VLLM_VIDEOMME_FORCE_INLINE", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if force_inline and "--videomme-inline-local-video" not in argv:
            argv.append("--videomme-inline-local-video")
        return argv

    monkeypatch.setattr(_acc_bench, "daily_omni_bench_argv", _wrap_daily)
    monkeypatch.setattr(_acc_bench, "videomme_bench_argv", _wrap_videomme)
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
        skip_videomme=True,
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


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", videomme_test_params, indirect=True)
def test_minicpmo_4_5_videomme_accuracy_bench(omni_server) -> None:
    """Gate MiniCPM-o 4.5 Video-MME (w/o subs) overall accuracy at the OmniEvalKit recipe."""
    _require_vllm_cli()
    pytest.importorskip("datasets")
    pytest.importorskip("huggingface_hub")
    pytest.importorskip("av")

    dataset_path = _optional_local_videomme_dataset_path()
    num_prompts = int(os.environ.get("ACC_BENCH_VIDEOMME_NUM_PROMPTS", "2700"))
    max_concurrency = int(os.environ.get("ACC_BENCH_VIDEOMME_MAX_CONCURRENCY", "4"))

    argv = build_acc_benchmark_cli_argv(
        omni_server,
        skip_seed=True,
        skip_daily=True,
        skip_videomme=False,
        num_prompts=num_prompts,
        max_concurrency=max_concurrency,
    )
    argv.extend(
        [
            "--result-dir",
            str(_RESULT_DIR),
            "--min-videomme-accuracy",
            str(_MIN_VIDEOMME_ACCURACY),
            "--temperature",
            "0",
            "--output-len",
            "128",
            "--videomme-pack-mode",
            "minicpm-frames",
            "--videomme-max-frames",
            "96",
            "--videomme-duration",
            os.environ.get("ACC_BENCH_VIDEOMME_DURATION", "all"),
            "--videomme-extra-body-json",
            json.dumps(_VIDEOMME_EXTRA_BODY, separators=(",", ":")),
            "--trust-remote-code",
        ]
    )
    # Absolute local root only; otherwise --videomme-repo (Hub) from build_acc_benchmark_cli_argv.
    if dataset_path is not None:
        argv.extend(["--videomme-dataset-path", dataset_path])

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
        skip_videomme=True,
        num_prompts=_SEED_TTS_NUM_PROMPTS,
        max_concurrency=4,
    )
    argv.extend(
        [
            "--result-dir",
            str(_RESULT_DIR),
            "--max-seed-tts-mean-wer",
            str(_MAX_SEED_TTS_MEAN_WER),
            "--min-seed-tts-mean-sim",
            str(_MIN_SEED_TTS_MEAN_SIM),
            "--seed-tts-locale",
            _SEED_TTS_LOCALE,
            "--seed-extra-body-json",
            json.dumps(_SEED_EXTRA_BODY, separators=(",", ":")),
            "--trust-remote-code",
        ]
    )

    assert _acc_bench.run_acc_benchmark(_acc_bench.parse_acc_benchmark_args(argv)) == 0


@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", DUPLEX_TEST_PARAMS, indirect=True)
def test_minicpmo_4_5_duplex_seed_tts_wer_bench(omni_server) -> None:
    """Gate Seed-TTS WER through the explicit Realtime TTS contract."""
    _require_vllm_cli()
    pytest.importorskip("huggingface_hub")

    argv = build_acc_benchmark_cli_argv(
        omni_server,
        skip_seed=False,
        skip_daily=True,
        skip_videomme=True,
        num_prompts=50,
        max_concurrency=1,
    )
    argv.extend(
        [
            "--result-dir",
            str(_RESULT_DIR),
            "--max-seed-tts-mean-wer",
            str(_MAX_SEED_TTS_MEAN_WER),
            "--min-seed-tts-mean-sim",
            str(_MIN_SEED_TTS_MEAN_SIM),
            "--seed-tts-locale",
            _SEED_TTS_LOCALE,
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
