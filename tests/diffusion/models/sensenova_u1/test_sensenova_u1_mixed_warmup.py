# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova mixed-traffic warmup configuration and request coverage."""

from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.lora.manager import LoRABackend
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    SenseNovaU1Pipeline,
    _parse_mixed_warmup_config,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_mixed_warmup_covers_selected_task_and_resolution_shapes(monkeypatch):
    host = object.__new__(SenseNovaU1Pipeline)
    host._mixed_warmup = _parse_mixed_warmup_config(
        {"resolutions": [[1024, 1024], [1536, 1536]], "text_to_text": True, "image_to_text": True},
        grid_factor=16,
    )
    host._mixed_warmup_done = False
    host.patch_size = 2
    host.merge_size = 8
    host.od_config = SimpleNamespace(lora_backend=LoRABackend.PEFT, lora_path=None)
    calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        host,
        "_forward_text",
        lambda params, images: calls.append(("text", bool(images), params.extra_args["max_tokens"])),
    )
    monkeypatch.setattr(
        host,
        "_forward_t2i",
        lambda params: calls.append(
            ("image", params.image_size, params.num_steps, params.think_mode, params.cfg_scale)
        ),
    )

    SenseNovaU1Pipeline._warm_mixed_shapes(host)
    SenseNovaU1Pipeline._warm_mixed_shapes(host)

    assert calls == [
        ("text", False, 1),
        ("text", True, 1),
        ("image", (1024, 1024), 1, False, 4.0),
        ("image", (1536, 1536), 1, False, 4.0),
    ]


@pytest.mark.parametrize(
    "profile",
    [
        {"resolutions": [[1025, 1024]]},
        {"resolutions": [[1024, 1024], [1024, 1024]]},
        {"resolutions": [[1024, 1024]] * 4},
        {"resolutions": [[True, 1024]]},
        {"resolutions": [[4096, 4112]]},
        {"resolutions": [], "text_to_text": 1},
        {"resolutions": [[1024, 1024]], "cfg_scale": -1},
        {"resolutions": [[1024, 1024]], "cfg_scale": True},
        {"unknown": True},
        {},
    ],
)
def test_mixed_warmup_rejects_invalid_or_unbounded_profiles(profile):
    with pytest.raises(ValueError):
        _parse_mixed_warmup_config(profile, grid_factor=16)


def test_mixed_warmup_uses_distilled_lora_cfg(monkeypatch):
    host = object.__new__(SenseNovaU1Pipeline)
    host._mixed_warmup = _parse_mixed_warmup_config({"resolutions": [[1024, 1024]]}, grid_factor=16)
    host._mixed_warmup_done = False
    host.patch_size = 2
    host.merge_size = 8
    host.od_config = SimpleNamespace(lora_backend=LoRABackend.DISTILL, lora_path="adapter")
    seen = []
    monkeypatch.setattr(host, "_forward_t2i", lambda params: seen.append(params.cfg_scale))

    SenseNovaU1Pipeline._warm_mixed_shapes(host)

    assert seen == [1.0]


def test_mixed_warmup_uses_explicit_cfg_scale(monkeypatch):
    host = object.__new__(SenseNovaU1Pipeline)
    host._mixed_warmup = _parse_mixed_warmup_config({"resolutions": [[1024, 1024]], "cfg_scale": 2.5}, grid_factor=16)
    host._mixed_warmup_done = False
    host.patch_size = 2
    host.merge_size = 8
    host.od_config = SimpleNamespace(lora_backend=LoRABackend.PEFT, lora_path=None)
    seen = []
    monkeypatch.setattr(host, "_forward_t2i", lambda params: seen.append(params.cfg_scale))

    SenseNovaU1Pipeline._warm_mixed_shapes(host)

    assert seen == [2.5]


def test_mixed_warmup_uses_only_plain_engine_dummy():
    plain = SimpleNamespace(request_id=DUMMY_DIFFUSION_REQUEST_ID)
    kv_profile = SimpleNamespace(request_id=f"{DUMMY_DIFFUSION_REQUEST_ID}/kv-profile-0")
    real = SimpleNamespace(request_id="real-request")

    assert SenseNovaU1Pipeline._is_engine_dummy_request(plain)
    assert SenseNovaU1Pipeline._is_engine_dummy_request(DiffusionRequestBatch([plain]))
    assert not SenseNovaU1Pipeline._is_engine_dummy_request(kv_profile)
    assert not SenseNovaU1Pipeline._is_engine_dummy_request(DiffusionRequestBatch([kv_profile]))
    assert not SenseNovaU1Pipeline._is_engine_dummy_request(DiffusionRequestBatch([plain, real]))
    assert not SenseNovaU1Pipeline._is_engine_dummy_request(real)


def _forward_test_host(monkeypatch, calls, *, fail_on=None):
    host = object.__new__(SenseNovaU1Pipeline)
    host._mixed_warmup = _parse_mixed_warmup_config(
        {"resolutions": [[1024, 1024]], "text_to_text": True, "image_to_text": True},
        grid_factor=16,
    )
    host._mixed_warmup_done = False
    host.patch_size = 2
    host.merge_size = 8
    host.od_config = SimpleNamespace(lora_backend=LoRABackend.PEFT, lora_path=None)

    def parse(request):
        assert isinstance(request.sampling_params, OmniDiffusionSamplingParams)
        return SenseNovaU1Pipeline._parse_request(host, request)

    monkeypatch.setattr(host, "_parse_request", parse)

    def record(label):
        calls.append(label)
        if label == fail_on:
            raise RuntimeError(f"selected {label} failed")
        return DiffusionOutput()

    monkeypatch.setattr(host, "_warm_ar_decode", lambda: calls.append("ar_decode"))
    monkeypatch.setattr(host, "_forward_it2i", lambda params, images: record("generic_dummy"))
    monkeypatch.setattr(
        host,
        "_forward_text",
        lambda params, images: record("image_to_text" if images else "text_to_text"),
    )
    monkeypatch.setattr(
        host, "_forward_t2i", lambda params: record(f"text_to_image_{params.image_size[0]}x{params.image_size[1]}")
    )
    return host


def _forward_dummy(request_id=DUMMY_DIFFUSION_REQUEST_ID):
    return OmniDiffusionRequest(
        prompt={"prompt": "dummy run", "multi_modal_data": {"image": Image.new("RGB", (16, 16))}},
        sampling_params=OmniDiffusionSamplingParams(width=512, height=512, num_inference_steps=2, seed=42),
        request_id=request_id,
    )


def test_forward_runs_mixed_profile_after_engine_dummy_once(monkeypatch):
    calls = []
    host = _forward_test_host(monkeypatch, calls)

    host.forward(DiffusionRequestBatch([_forward_dummy(f"{DUMMY_DIFFUSION_REQUEST_ID}/kv-profile-0")]))
    assert calls == ["ar_decode", "generic_dummy"]
    assert not host._mixed_warmup_done

    host.forward(DiffusionRequestBatch([_forward_dummy()]))
    assert calls == [
        "ar_decode",
        "generic_dummy",
        "ar_decode",
        "generic_dummy",
        "text_to_text",
        "image_to_text",
        "text_to_image_1024x1024",
    ]
    assert host._mixed_warmup_done

    host.forward(DiffusionRequestBatch([_forward_dummy()]))
    assert calls[-2:] == ["ar_decode", "generic_dummy"]
    assert len(calls) == 9


@pytest.mark.parametrize("failed_operation", ["text_to_text", "image_to_text", "text_to_image_1024x1024"])
def test_forward_profile_error_fails_engine_startup(monkeypatch, failed_operation):
    calls = []
    host = _forward_test_host(monkeypatch, calls, fail_on=failed_operation)
    engine = object.__new__(DiffusionEngine)
    engine._make_dummy_request = lambda **kwargs: _forward_dummy()
    engine.close = lambda: calls.append("engine_closed")

    def run_request(request):
        try:
            return host.forward(DiffusionRequestBatch([request]))
        except RuntimeError as exc:
            return DiffusionOutput(error=str(exc))

    engine.add_req_and_wait_for_response = run_request

    with pytest.raises(RuntimeError, match=f"Dummy run failed: selected {failed_operation} failed"):
        engine.run_startup_warmup()

    assert calls[0:2] == ["ar_decode", "generic_dummy"]
    assert calls[-2:] == [failed_operation, "engine_closed"]
    assert not host._mixed_warmup_done
