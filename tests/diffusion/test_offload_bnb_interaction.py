# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn as nn

from vllm_omni.diffusion.offloader import sequential_backend
from vllm_omni.diffusion.offloader.base import OffloadConfig, OffloadStrategy
from vllm_omni.diffusion.quantization.bitsandbytes import set_bnb_offload_skip_components


def test_model_level_offload_respects_bnb_skip(monkeypatch):
    captured: dict[str, list[nn.Module]] = {}

    def _fake_apply_sequential_offload(*, offload_dit_modules, offload_encoder_modules, **kwargs):
        captured["offload_dit_modules"] = list(offload_dit_modules)
        captured["offload_encoder_modules"] = list(offload_encoder_modules)

    monkeypatch.setattr(sequential_backend, "apply_sequential_offload", _fake_apply_sequential_offload)

    class DummyPipeline(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = nn.Linear(4, 4)
            self.text_encoder = nn.Linear(4, 4)

    pipeline = DummyPipeline()
    set_bnb_offload_skip_components(pipeline, {"transformer"})

    backend = sequential_backend.ModelLevelOffloadBackend(
        OffloadConfig(strategy=OffloadStrategy.MODEL_LEVEL),
        device=torch.device("cpu"),
    )
    backend.enable(pipeline)

    assert pipeline.transformer not in captured["offload_dit_modules"]
    assert pipeline.text_encoder in captured["offload_encoder_modules"]
