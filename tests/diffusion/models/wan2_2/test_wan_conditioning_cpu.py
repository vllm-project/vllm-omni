# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent CPU checks of the complete pure helper module and Diffusers VAE.

Load the module normally by file to avoid importing Wan's package initializer
(which eagerly imports vLLM transformers). No source/AST extraction or mocks.
Can run with pytest --noconftest when vLLM is unavailable. Pipeline integration
coverage lives separately in test_wan22_stage_runtime.py.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import PIL.Image
import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan

_PATH = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/wan2_2/conditioning.py"
_SPEC = importlib.util.spec_from_file_location("wan_conditioning_cpu", _PATH)
conditioning = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(conditioning)


@pytest.fixture(scope="module")
def vae():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield AutoencoderKLWan(
        base_dim=4,
        decoder_base_dim=4,
        z_dim=4,
        dim_mult=[1, 2, 2, 2],
        num_res_blocks=1,
        latents_mean=[0.1] * 4,
        latents_std=[2.0] * 4,
    ).eval()
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("outputs", [1, 3])
@pytest.mark.parametrize("count", [1, 2])
def test_real_vae_full_vs_encode_then_repeat(vae, outputs, count):
    images = [PIL.Image.new("RGB", (32, 16), (20 + i * 50, 80, 120)) for i in range(count)]
    tensor = conditioning.prepare_wan_image_tensor(images, 16, 32, 8)
    state = torch.random.get_rng_state().clone()
    with torch.no_grad():
        full = conditioning.encode_wan_image_condition(vae, tensor.repeat_interleave(outputs, 0), torch.device("cpu"))
        encoded = torch.cat(
            [conditioning.encode_wan_image_condition(vae, row, torch.device("cpu")) for row in tensor.split(1)]
        )
    torch.testing.assert_close(full, encoded.repeat_interleave(outputs, 0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.random.get_rng_state(), state)
    metadata = conditioning.conditioning_metadata(16, 32, 5, 8, 4, True)
    for row in encoded.split(1):
        assert conditioning.validate_wan_conditioning(row, metadata, expected=metadata, channels=4)
    mask = conditioning.wan_first_frame_mask(torch.empty(count * outputs, 4, 2, 2, 4))
    assert torch.count_nonzero(mask[:, :, 0]) == 0
    assert torch.all(mask[:, :, 1] == 1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 2),
        ("version", True),
        ("normalization", "raw"),
        ("layout", "CTHW"),
        ("height", 32),
        ("width", 16),
        ("num_frames", 9),
        ("spatial_scale", 16),
        ("temporal_scale", 8),
        ("has_image", 1),
    ],
)
def test_metadata_rejects_incompatible_contract(key, value):
    expected = conditioning.conditioning_metadata(16, 32, 5, 8, 4, True)
    bad = dict(expected, **{key: value})
    with pytest.raises(ValueError, match=key):
        conditioning.validate_wan_conditioning(torch.zeros(1, 4, 1, 2, 4), bad, expected=expected, channels=4)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        torch.zeros(4, 1, 2, 4),
        torch.zeros(2, 4, 1, 2, 4),
        torch.zeros(1, 4, 2, 2, 4),
        torch.zeros(1, 4, 1, 2, 4).half(),
        torch.full((1, 4, 1, 2, 4), float("nan")),
    ],
)
def test_tensor_contract_rejects_missing_shape_dtype_and_nan(bad):
    expected = conditioning.conditioning_metadata(16, 32, 5, 8, 4, True)
    with pytest.raises(ValueError, match="wan_image_condition"):
        conditioning.validate_wan_conditioning(bad, expected, expected=expected, channels=4)


def test_no_image_contract_requires_metadata_and_no_latents():
    expected = conditioning.conditioning_metadata(16, 32, 5, 8, 4, False)
    assert not conditioning.validate_wan_conditioning(None, expected, expected=expected, channels=4)
    with pytest.raises(ValueError, match="wan_conditioning_metadata"):
        conditioning.validate_wan_conditioning(None, None, expected=expected, channels=4)
    with pytest.raises(ValueError, match="has_image=False"):
        conditioning.validate_wan_conditioning(torch.zeros(1), expected, expected=expected, channels=4)


def test_effective_dimensions_preserve_wan_rounding():
    sampling = SimpleNamespace(height=385, width=417, num_frames=18)
    assert conditioning.effective_wan_dimensions(sampling, 16, 4, (1, 2, 2)) == (384, 416, 17)


@pytest.mark.parametrize("stage", ["encode", "generation"])
@pytest.mark.parametrize(
    "field,first,second",
    [
        ("CFG", True, False),
        ("max_sequence_length", 512, 256),
        ("height, width and num_frames", (16, 32, 5), (32, 32, 5)),
        ("guidance scales", (4.0, 4.0), (4.0, 5.0)),
        ("num_outputs_per_prompt", 1, 2),
        ("boundary_ratio", 0.875, 0.5),
        ("output_type", "np", "latent"),
        ("num_inference_steps", 40, 20),
        ("sample_solver", "unipc", "euler"),
        ("flow_shift", 5.0, 6.0),
    ],
)
def test_batch_settings_reject_only_effective_mismatches(stage, field, first, second):
    settings = [{field: first}, {field: first}, {field: second}]
    with pytest.raises(ValueError, match=f"{stage}.*{field}.*request 2"):
        conditioning.validate_wan_batch_settings(settings, stage=stage)
    assert settings == [{field: first}, {field: first}, {field: second}]


@pytest.mark.parametrize("settings", [[], [{}], [{"CFG": True}, {"CFG": True}]])
def test_batch_settings_accept_empty_single_and_compatible(settings):
    conditioning.validate_wan_batch_settings(settings, stage="encode")


def test_batch_settings_accept_rounded_equivalent_dimensions():
    settings = [
        {"dimensions": conditioning.effective_wan_dimensions(sampling, 8, 4, (1, 2, 2))}
        for sampling in [
            SimpleNamespace(height=16, width=32, num_frames=5),
            SimpleNamespace(height=31, width=47, num_frames=4),
        ]
    ]
    conditioning.validate_wan_batch_settings(settings, stage="generation")
