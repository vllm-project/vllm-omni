"""
L5(b) reliability integration tests (RFC: test_reliability.py).

Loads ``tests/dfx/reliability/tests/scenarios.json``, runs callable fault-injection
helpers from ``tests.dfx.reliability.conftest``, then validates request behavior.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Protocol

import pytest

from tests.conftest import (
    OmniServerParams,
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.dfx.conftest import create_unique_server_params, load_configs
from tests.dfx.reliability.conftest import (
    inject_gpu_oom,
    stop_gpu_oom_hogs,
)
from tests.utils import hardware_test
from vllm_omni.platforms import current_omni_platform

REL_DIR = Path(__file__).resolve().parent.parent
SCENARIOS_PATH = REL_DIR / "tests" / "scenarios.json"
# Reuse e2e stage configs for reliability tests.
E2E_STAGE_CONFIGS_DIR = REL_DIR.parent.parent / "e2e" / "stage_configs"
OOM_INJECTION_CONFIG = {
    # Optional: set to "0,1,2" for explicit multi-GPU injection.
    # If omitted, devices are auto-derived from stage yaml runtime.devices.
    # "device": "0,1,2",
    "target_mem_ratio": 0.95,
    "hold_seconds": 45,
    "startup_timeout_sec": 20,
    "strict": True,
}


def _configs_with_platform_stage_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve stage config path for XPU vs CUDA/ROCm (async_chunk yaml is CUDA-only)."""
    out: list[dict[str, Any]] = []
    for c in configs:
        c2 = copy.deepcopy(c)
        if current_omni_platform.is_xpu():
            c2["server_params"]["stage_config_name"] = "xpu/qwen3_omni_ci.yaml"
            c2["server_params"].pop("update", None)
            c2["server_params"].pop("delete", None)
        out.append(c2)
    return out


def _load_reliability_configs() -> list[dict[str, Any]]:
    try:
        raw = load_configs(str(SCENARIOS_PATH))
    except (ValueError, FileNotFoundError):
        return []
    return _configs_with_platform_stage_configs(raw)


def _get_system_prompt() -> dict:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def _get_mix_prompt() -> str:
    return "What is recited in the audio? What is in this image? Describe the video briefly."


class _HasServeArgs(Protocol):
    serve_args: list[str]


def _stage_config_path_from_omni_server(omni_server: _HasServeArgs) -> str | None:
    """Read effective stage yaml path from server CLI (OmniServer does not expose it as an attribute)."""
    args: list[str] = omni_server.serve_args
    for i, arg in enumerate(args):
        if arg == "--stage-configs-path" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--stage-configs-path="):
            return arg.split("=", 1)[1]
    return None


def _parse_stage_devices(stage_config_path: str) -> str:
    text = Path(stage_config_path).read_text(encoding="utf-8")
    raw_devices: list[str] = re.findall(r"^\s*devices:\s*\"?([0-9,\s]+)\"?\s*$", text, flags=re.MULTILINE)
    devices: set[int] = set()
    for item in raw_devices:
        for token in item.split(","):
            token = token.strip()
            if token:
                devices.add(int(token))
    if not devices:
        raise ValueError(f"No runtime.devices found in stage config: {stage_config_path}")
    return ",".join(str(x) for x in sorted(devices))


def _resolve_oom_device_spec(config: dict[str, Any], stage_config_path: str | None) -> str:
    explicit = config.get("device")
    if explicit is not None:
        return str(explicit)
    if not stage_config_path:
        raise ValueError("OOM device is not configured and stage_config_path is empty.")
    return _parse_stage_devices(stage_config_path)


CONFIGS = _load_reliability_configs()
_UNIQUE_PARAMS = create_unique_server_params(CONFIGS, E2E_STAGE_CONFIGS_DIR) if CONFIGS else []
TEST_PARAMS = [OmniServerParams(model=m, stage_config_path=p) for _, m, p in _UNIQUE_PARAMS]


def test_scenarios_json_loads() -> None:
    """Guarantee RFC ``scenarios.json`` exists and parses (collect-only / CI smoke)."""
    assert SCENARIOS_PATH.is_file(), f"Missing {SCENARIOS_PATH}"
    cfg = load_configs(str(SCENARIOS_PATH))
    assert isinstance(cfg, list)
    assert len(cfg) > 0, "scenarios.json should list at least one scenario"


@pytest.mark.slow
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.skipif(not TEST_PARAMS, reason="no server params available")
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_reliability_gpu_oom_injection(omni_server, openai_client) -> None:
    """Inject GPU OOM from test code and verify request fails during injection."""
    device_spec = _resolve_oom_device_spec(
        OOM_INJECTION_CONFIG, _stage_config_path_from_omni_server(omni_server)
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
    )
    try:
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
        messages = dummy_messages_from_mix_data(
            system_prompt=_get_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text=_get_mix_prompt(),
        )
        request_config = {
            "model": omni_server.model,
            "messages": messages,
            "stream": True,
            "key_words": {
                "audio": ["test"],
            },
        }
        raised = False
        try:
            openai_client.send_omni_request(request_config, request_num=1)
        except Exception as exc:
            raised = True
            text = str(exc).lower()
            assert any(
                key in text for key in ("oom", "out of memory", "cuda", "internal", "500", "timeout", "connection")
            ), f"unexpected error under OOM injection: {exc}"
        assert raised, "expected request failure during GPU OOM injection"
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(
    current_omni_platform.is_rocm() or current_omni_platform.is_xpu(),
    reason="CUDA sidecar OOM injection is CUDA-only for phase-1",
)
@pytest.mark.skipif(not TEST_PARAMS, reason="no server params available")
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_reliability_gpu_oom_text_to_text(omni_server, openai_client) -> None:
    """OOM reliability case aligned with qwen3_omni text_to_text request style."""
    device_spec = _resolve_oom_device_spec(
        OOM_INJECTION_CONFIG, _stage_config_path_from_omni_server(omni_server)
    )
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=OOM_INJECTION_CONFIG["target_mem_ratio"],
        hold_seconds=OOM_INJECTION_CONFIG["hold_seconds"],
        startup_timeout_sec=OOM_INJECTION_CONFIG["startup_timeout_sec"],
        strict=OOM_INJECTION_CONFIG["strict"],
    )
    try:
        messages = dummy_messages_from_mix_data(
            system_prompt=_get_system_prompt(),
            content_text="What is the capital of China? Answer in 20 words.",
        )
        request_config = {
            "model": omni_server.model,
            "messages": messages,
            "stream": False,
            "modalities": ["text"],
            "key_words": {"text": ["beijing"]},
        }
        raised = False
        try:
            openai_client.send_omni_request(request_config, request_num=1)
        except Exception as exc:
            raised = True
            text = str(exc).lower()
            assert any(
                key in text for key in ("oom", "out of memory", "cuda", "internal", "500", "timeout", "connection")
            ), f"unexpected error under OOM injection: {exc}"
        assert raised, "expected request failure during GPU OOM injection"
    finally:
        stop_gpu_oom_hogs(handle)
