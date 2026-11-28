# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Protect native H3 Video VAE selection and model-owned INT8 placement."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config.load import LoadConfig

from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.minimax_h3 import vae, vae_int8
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.quantization.component_config import ComponentQuantizationConfig
from vllm_omni.quantization.int8_config import DiffusionInt8Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_litevae_and_online_int8_are_rejected_before_checkpoint_access(monkeypatch):
    monkeypatch.setattr(vae_int8, "_require_npu_device", Mock())
    read_config = Mock(side_effect=AssertionError("VAE configuration was read before validation"))
    monkeypatch.setattr(vae, "_load_component_config", read_config)

    with pytest.raises(ValueError, match="LiteVAE"):
        vae.MiniMaxH3VideoVAE(
            "unused",
            device=torch.device("cpu"),
            lightvae_decoder_path="unused.safetensors",
            quant_config=DiffusionInt8Config(),
        )
    read_config.assert_not_called()


@pytest.mark.parametrize(
    ("registered_name", "explicit_opt_in", "expected_device"),
    [("VAEInt8Alias", True, "cpu"), ("MiniMaxH3Pipeline", False, "meta")],
)
def test_default_loader_respects_only_the_resolved_class_opt_in(
    monkeypatch, registered_name, explicit_opt_in, expected_device
):
    owns = Mock(return_value=True)

    class SupportedPipeline:
        owns_eager_quantization = staticmethod(owns)

    class UnverifiedDerivedPipeline(SupportedPipeline):
        pass

    resolved_class = SupportedPipeline if explicit_opt_in else UnverifiedDerivedPipeline
    monkeypatch.setattr(DiffusionModelRegistry, "_try_load_model_cls", Mock(return_value=resolved_class))
    int8 = DiffusionInt8Config()
    loader = DiffusersPipelineLoader(
        LoadConfig(),
        SimpleNamespace(
            model_class_name=registered_name,
            dtype=torch.float32,
            parallel_config=SimpleNamespace(use_hsdp=False),
            quantization_config=ComponentQuantizationConfig({"video_vae": int8, "transformer": None}),
        ),
    )

    class ConstructionReachedError(Exception):
        pass

    init = Mock(side_effect=ConstructionReachedError)
    monkeypatch.setattr(loader, "_init_from_load_format", init)

    with pytest.raises(ConstructionReachedError):
        loader.load_model(load_device="cpu", device=torch.device("meta"))

    assert init.call_args.args[1] == torch.device(expected_device)
    if explicit_opt_in:
        owns.assert_called_once_with("video_vae", int8)
    else:
        owns.assert_not_called()
