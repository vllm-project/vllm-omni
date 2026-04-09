"""
L5(b) reliability integration tests (RFC: test_reliability.py).

Loads ``tests/dfx/reliability/tests/scenarios.json``, runs callable fault-injection
helpers from ``tests.dfx.reliability.conftest``, then validates request behavior.
"""

from __future__ import annotations

import copy
import os
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
    make_process_kill_fault_injector,
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
FAULT_ERROR_KEYWORDS = (
    "oom",
    "out of memory",
    "cuda",
    "internal",
    "500",
    "503",
    "timeout",
    "connection",
    "engine",
    "orchestrator",
    "dead",
)


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


def _assert_fault_exception(exc: Exception) -> None:
    text = str(exc).lower()
    assert any(key in text for key in FAULT_ERROR_KEYWORDS), f"unexpected error under fault injection: {exc}"


# Qwen3-Omni staged stack uses StageEngineCoreProc_* / Worker in argv (see ps -f on serve).
# Older or non-staged builds may still show EngineCore / orchestrator in cmdline.
_RUNTIME_PROCESS_KILL_PATTERNS = (
    "VLLM::StageEngineCoreProc",
    "VLLM::Worker",
    "VLLM::EngineCore",
    "vllm_omni.engine.orchestrator",
    "EngineCore",
)


def _supports_video_generation(model_name: str) -> bool:
    lower = model_name.lower()
    return any(key in lower for key in ("wan", "video", "i2v", "t2v"))


def _supports_chat_generation(model_name: str) -> bool:
    return not _supports_video_generation(model_name)


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
        # Scenarios may omit stage yaml path; reliability OOM cases here are single-card tests.
        return "0"
    return _parse_stage_devices(stage_config_path)


def _extract_server_args_by_test_name(configs: list[dict[str, Any]]) -> dict[str, list[str] | None]:
    mapping: dict[str, list[str] | None] = {}
    for cfg in configs:
        test_name = str(cfg.get("test_name"))
        server_params = cfg.get("server_params") or {}
        raw_args = server_params.get("server_args")
        mapping[test_name] = [str(item) for item in raw_args] if isinstance(raw_args, list) else None
    return mapping


CONFIGS = _load_reliability_configs()
_SERVER_ARGS_BY_TEST_NAME = _extract_server_args_by_test_name(CONFIGS)
_UNIQUE_PARAMS = create_unique_server_params(CONFIGS, E2E_STAGE_CONFIGS_DIR) if CONFIGS else []
TEST_PARAMS = [
    OmniServerParams(
        model=model, stage_config_path=stage_config_path, server_args=_SERVER_ARGS_BY_TEST_NAME.get(test_name)
    )
    for test_name, model, stage_config_path in _UNIQUE_PARAMS
]

OMNI_CHAT_PARAMS = [param for param in TEST_PARAMS if _supports_chat_generation(param.model)]
DIFFUSION_VIDEO_PARAMS = [param for param in TEST_PARAMS if _supports_video_generation(param.model)]


def validate_scenarios_json_loads() -> None:
    """Validate that reliability ``scenarios.json`` exists and parses."""
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
@pytest.mark.skipif(not OMNI_CHAT_PARAMS, reason="no omni-chat server params available")
@pytest.mark.parametrize(
    "oom_cfg",
    [
        {
            "target_mem_ratio": OOM_INJECTION_CONFIG["target_mem_ratio"],
            "hold_seconds": OOM_INJECTION_CONFIG["hold_seconds"],
            "startup_timeout_sec": OOM_INJECTION_CONFIG["startup_timeout_sec"],
            "strict": OOM_INJECTION_CONFIG["strict"],
        }
    ],
    ids=["oom_default"],
)
@pytest.mark.parametrize("omni_server", OMNI_CHAT_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_chat_large_payload_failure(omni_server, openai_client, oom_cfg) -> None:
    """Large chat payload under OOM injection should fail."""
    device_spec = _resolve_oom_device_spec(OOM_INJECTION_CONFIG, _stage_config_path_from_omni_server(omni_server))
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=oom_cfg["target_mem_ratio"],
        hold_seconds=oom_cfg["hold_seconds"],
        startup_timeout_sec=oom_cfg["startup_timeout_sec"],
        strict=oom_cfg["strict"],
    )
    try:
        # Keep chat-form request but scale payload size to emulate heavy input pressure.
        video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 161)['base64']}"
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
        audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(20, 1)['base64']}"
        messages = dummy_messages_from_mix_data(
            system_prompt=_get_system_prompt(),
            video_data_url=video_data_url,
            image_data_url=image_data_url,
            audio_data_url=audio_data_url,
            content_text=f"{_get_mix_prompt()} " * 200,
        )
        request_config = {
            "model": omni_server.model,
            "messages": messages,
            "stream": True,
            "key_words": {"audio": ["test"]},
        }
        try:
            openai_client.send_omni_request(request_config, request_num=1)
        except Exception as exc:
            _assert_fault_exception(exc)
        else:
            pytest.fail("expected large chat payload request failure during GPU OOM injection")
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
@pytest.mark.skipif(not DIFFUSION_VIDEO_PARAMS, reason="no diffusion-video server params available")
@pytest.mark.parametrize(
    "oom_cfg",
    [
        {
            "target_mem_ratio": OOM_INJECTION_CONFIG["target_mem_ratio"],
            "hold_seconds": OOM_INJECTION_CONFIG["hold_seconds"],
            "startup_timeout_sec": OOM_INJECTION_CONFIG["startup_timeout_sec"],
            "strict": OOM_INJECTION_CONFIG["strict"],
        }
    ],
    ids=["oom_default"],
)
@pytest.mark.parametrize("omni_server", DIFFUSION_VIDEO_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_video_large_request_failure(omni_server, openai_client, oom_cfg) -> None:
    """Video-style large request under OOM injection (closer to RFC #2327 trigger path)."""
    device_spec = _resolve_oom_device_spec(OOM_INJECTION_CONFIG, _stage_config_path_from_omni_server(omni_server))
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=oom_cfg["target_mem_ratio"],
        hold_seconds=oom_cfg["hold_seconds"],
        startup_timeout_sec=oom_cfg["startup_timeout_sec"],
        strict=oom_cfg["strict"],
    )
    try:
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"
        request_config = {
            "form_data": {
                "prompt": "Generate a realistic road-driving video with camera motion.",
                "width": 512,
                "height": 512,
                "fps": 8,
                "num_frames": 8,
                "guidance_scale": 1.0,
                "flow_shift": 5.0,
                "num_inference_steps": 4,
                "seed": 42,
            },
            "image_reference": image_data_url,
            "stream": False,
        }
        try:
            openai_client.send_video_diffusion_request(request_config, request_num=1)
        except Exception as exc:
            _assert_fault_exception(exc)
        else:
            pytest.fail("expected large /v1/videos request failure during GPU OOM injection")
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
@pytest.mark.skipif(not OMNI_CHAT_PARAMS, reason="no omni-chat server params available")
@pytest.mark.parametrize(
    "oom_cfg",
    [
        {
            "target_mem_ratio": OOM_INJECTION_CONFIG["target_mem_ratio"],
            "hold_seconds": OOM_INJECTION_CONFIG["hold_seconds"],
            "startup_timeout_sec": OOM_INJECTION_CONFIG["startup_timeout_sec"],
            "strict": OOM_INJECTION_CONFIG["strict"],
        }
    ],
    ids=["oom_default"],
)
@pytest.mark.parametrize("omni_server", OMNI_CHAT_PARAMS, indirect=True)
def test_reliability_fault_gpu_oom_concurrent_pressure_failure(omni_server, openai_client, oom_cfg) -> None:
    """Concurrent request pressure under OOM should produce request failures."""
    device_spec = _resolve_oom_device_spec(OOM_INJECTION_CONFIG, _stage_config_path_from_omni_server(omni_server))
    handle = inject_gpu_oom(
        device=device_spec,
        target_mem_ratio=oom_cfg["target_mem_ratio"],
        hold_seconds=oom_cfg["hold_seconds"],
        startup_timeout_sec=oom_cfg["startup_timeout_sec"],
        strict=oom_cfg["strict"],
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
        try:
            openai_client.send_omni_request(request_config, request_num=4)
        except Exception as exc:
            _assert_fault_exception(exc)
        else:
            pytest.fail("expected concurrent request failure under OOM injection")
    finally:
        stop_gpu_oom_hogs(handle)


@pytest.mark.slow
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@pytest.mark.skipif(not OMNI_CHAT_PARAMS, reason="no omni-chat server params available")
@pytest.mark.parametrize(
    "fault_injector",
    [
        pytest.param(
            make_process_kill_fault_injector(
                grep_patterns="VLLM::Worker",
                signal_name="SIGKILL",
                limit=1,
            ),
            id="runtime_process_chain",
        ),
    ],
    indirect=True,
)
@pytest.mark.parametrize("omni_server", OMNI_CHAT_PARAMS, indirect=True)
def test_reliability_fault_process_kill_request_failure(omni_server_after_fault, openai_client) -> None:
    """Engine-fatal style injection: kill one runtime process and verify request failure."""
    messages = dummy_messages_from_mix_data(
        system_prompt=_get_system_prompt(),
        content_text="What is the capital of China? Answer in 20 words.",
    )
    request_config = {
        "model": omni_server_after_fault.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }
    try:
        openai_client.send_omni_request(request_config, request_num=1)
    except Exception as exc:
        _assert_fault_exception(exc)
    else:
        pytest.fail("expected request failure after process-kill injection")
