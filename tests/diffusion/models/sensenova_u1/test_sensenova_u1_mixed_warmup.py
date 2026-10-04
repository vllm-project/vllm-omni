# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova mixed-traffic warmup configuration and request coverage."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.lora.manager import LoRABackend
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    SenseNovaU1Pipeline,
    _parse_mixed_warmup_config,
)
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

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
