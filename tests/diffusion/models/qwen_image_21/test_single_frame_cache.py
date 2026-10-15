# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-frame cache allocation and public pipeline configuration regressions."""

import pytest
import torch

from tests.diffusion.models.qwen_image_21.test_vae import _make_vae
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image_21 import pipeline_qwen_image_21 as pipeline_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("size", [8, 16])
def test_single_frame_decode_without_temporal_cache_is_exact_and_request_independent(batch, size):
    torch.manual_seed(13)
    vae = _make_vae().eval()
    torch.set_num_threads(1)
    z = torch.randn(batch, 4, 1, size, size)
    other = torch.randn_like(z)
    with torch.no_grad():
        expected = vae._decode(z, return_dict=False)[0]
        vae.skip_single_frame_cache = True
        actual = vae._decode(z, return_dict=False)[0]
        vae._decode(other, return_dict=False)
        repeated = vae._decode(z, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
    assert all(value is None for value in vae._feat_map)


def test_tiled_single_frame_decode_is_unchanged():
    torch.manual_seed(7)
    vae = _make_vae().eval()
    vae.enable_tiling(
        tile_sample_min_height=32,
        tile_sample_min_width=32,
        tile_sample_stride_height=24,
        tile_sample_stride_width=24,
    )
    z = torch.randn(1, 4, 1, 16, 16)
    with torch.no_grad():
        expected = vae.decode(z, return_dict=False)[0]
        vae.skip_single_frame_cache = True
        actual = vae.decode(z, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("skip_cache", [False, True])
def test_multiframe_input_keeps_existing_rejection(skip_cache):
    # This image VAE squeezes the singleton frame axis before its 2D convolutions.
    # The optimization does not introduce video/multiframe support.
    vae = _make_vae().eval()
    vae.skip_single_frame_cache = skip_cache
    with pytest.raises(RuntimeError, match="Expected 3D .* or 4D"):
        vae.decode(torch.randn(1, 4, 2, 8, 8), return_dict=False)


class _Processor:
    tokenizer = None

    def __init__(self):
        self.tokenizer = self

    def apply_chat_template(self, messages, tokenize, return_dict):
        return [[1, 2]]

    def encode(self, text):
        return [3]


@pytest.fixture
def configured_pipeline(monkeypatch, tmp_path):
    # Isolate checkpoint loading, retaining the real config and pipeline __init__.
    vae = _make_vae().eval()
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_module, "prefetch_subfolders", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pipeline_module.FlowMatchEulerDiscreteScheduler, "from_pretrained", lambda *args, **kwargs: object()
    )

    def load_component(loader, model, *, subfolder, **kwargs):
        return {"text_encoder": torch.nn.Identity(), "vae": vae, "processor": _Processor()}[subfolder]

    monkeypatch.setattr(pipeline_module, "from_pretrained_with_prefetch", load_component)
    monkeypatch.setattr(pipeline_module, "get_transformer_config_kwargs", lambda *args: {})
    monkeypatch.setattr(pipeline_module, "QwenImage21Transformer2DModel", lambda **kwargs: torch.nn.Identity())

    def build(additional_config):
        config = OmniDiffusionConfig(model=str(tmp_path), additional_config=additional_config, enforce_eager=True)
        return pipeline_module.QwenImage21Pipeline(od_config=config)

    return build


@pytest.mark.parametrize(
    "additional_config,enabled",
    [
        ({}, False),
        ({"qwen21_skip_single_frame_vae_cache": True}, True),
        ({"qwen21_skip_single_frame_vae_cache": False}, False),
    ],
)
@pytest.mark.parametrize("tiled", [False, True])
def test_public_configuration_controls_decoder_cache_during_decode(
    configured_pipeline, additional_config, enabled, tiled
):
    pipeline = configured_pipeline(additional_config)
    vae = pipeline.vae
    assert vae.skip_single_frame_cache is enabled
    if tiled:
        vae.enable_tiling(
            tile_sample_min_height=32,
            tile_sample_min_width=32,
            tile_sample_stride_height=24,
            tile_sample_stride_width=24,
        )
    seen = []

    def observe_decoder(module, args, kwargs):
        cache = kwargs["feat_cache"]
        seen.append(cache is None)
        if cache is not None:
            assert cache is vae._feat_map

    handle = vae.decoder.register_forward_pre_hook(observe_decoder, with_kwargs=True)
    try:
        with torch.no_grad():
            vae.decode(torch.randn(1, 4, 1, 16, 16), return_dict=False)
    finally:
        handle.remove()
    assert seen, "Production decoder was not called"
    assert all(value == (enabled and not tiled) for value in seen)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_public_configuration_rejects_non_bool(configured_pipeline, value):
    with pytest.raises(ValueError, match="qwen21_skip_single_frame_vae_cache must be a bool"):
        configured_pipeline({"qwen21_skip_single_frame_vae_cache": value})
