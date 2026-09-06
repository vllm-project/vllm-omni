# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tiny native runtime contracts, without checkpoint or distributed claims.

Only CPU platform discovery/dispatch is adapted by conftest. The constructor,
request parsing, Q-Former, DiT, CFG, scheduler and VAE arithmetic are real.
"""

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionOutput, DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
    _build_mammoth_config,
    _validate_sequence_parallel_runtime,
)
from vllm_omni.diffusion.output_formatter import format_diffusion_outputs, normalize_diffusion_postprocess_output
from vllm_omni.diffusion.registry import _apply_sequence_parallel_if_enabled, get_diffusion_post_process_func
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.parallel]


def _config(degree=1):
    raw = {
        "llm_config": {
            "model_type": "mammothmoda2_qwen2_5_vl",
            "gen_vocab_start_index": 100,
            "text_config": {"hidden_size": 16, "gen_vocab_start_index": 100},
        },
        "gen_dit_config": {
            "hidden_size": 126,
            "num_attention_heads": 21,
            "num_kv_heads": 7,
            "axes_dim_rope": (2, 2, 2),
            "axes_lens": (32, 32, 32),
            "num_layers": 2,
            "num_refiner_layers": 1,
            "in_channels": 4,
            "text_feat_dim": 16,
            "multiple_of": 8,
        },
        "gen_vae_config": {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 4,
            "block_out_channels": (8, 8, 8, 8),
            "down_block_types": ("DownEncoderBlock2D",) * 4,
            "up_block_types": ("UpDecoderBlock2D",) * 4,
            "layers_per_block": 1,
            "norm_num_groups": 4,
        },
        "gen_image_condition_refiner_config": {"num_queries": 2, "num_layers": 1},
        "gen_axes_dim_rope": (2, 2, 2),
        "gen_axes_lens": (32, 32, 32),
    }
    return OmniDiffusionConfig(
        # An existing directory avoids Hub resolution in offline GPU workers.
        # The explicit tiny config supplies the architecture; no weights load.
        model=str(Path(__file__).parent),
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(raw),
        dtype=torch.float32,
        enforce_eager=True,
        parallel_config=DiffusionParallelConfig(ulysses_degree=degree, ulysses_mode="advanced_uaa"),
        diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
    )


def _pipeline(config, monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit.get_local_device",
        lambda: torch.device("cpu"),
    )
    torch.manual_seed(19)
    with set_current_diffusion_config(config), set_forward_context(omni_diffusion_config=config):
        return MammothModa2DiTPipeline(od_config=config).eval()


def _request(text_len, guidance, seed=42):
    generator = torch.Generator().manual_seed(31 + text_len)
    return DiffusionRequestBatch(
        [
            OmniDiffusionRequest(
                request_id=f"tiny-{text_len}",
                prompt={
                    "prompt": "",
                    "additional_information": {
                        "full_hidden_states": torch.randn(text_len + 2, 16, generator=generator),
                        "full_token_ids": list(range(10, 10 + text_len)) + [100, 101],
                        "answer_start_index": text_len,
                    },
                },
                sampling_params=OmniDiffusionSamplingParams(
                    height=32,
                    width=48,
                    num_inference_steps=2,
                    seed=seed,
                    guidance_scale=guidance,
                ),
            )
        ]
    )


def test_shared_registry_discovers_main_dit_and_installs_both_boundaries(monkeypatch):
    pipeline = _pipeline(_config(), monkeypatch)
    keys = tuple(pipeline.state_dict())
    with set_forward_context(omni_diffusion_config=_config(2)):
        _apply_sequence_parallel_if_enabled(pipeline, _config(2))
        assert get_forward_context().sp_plan_hooks_applied
    transformer = pipeline.gen_transformer
    for boundary, name in (
        (transformer.sp_input_boundary, "sp_input---sp_input_boundary"),
        (transformer.sp_output_boundary, "sp_output---sp_output_boundary"),
    ):
        assert boundary._hook_registry.get_hook(name) is not None
    for component in (pipeline.gen_vae, pipeline.gen_image_condition_refiner):
        assert not any(hasattr(module, "_hook_registry") for module in component.modules())
    assert tuple(pipeline.state_dict()) == keys


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("step_execution", True, "request mode"),
        ("max_num_seqs", 2, "max_num_seqs=1"),
        ("enforce_eager", False, "eager execution"),
        ("cache_backend", "cache_dit", "cache acceleration"),
        ("enable_cpu_offload", True, "offload"),
    ],
)
def test_sp_rejects_unqualified_runtime_modes(field, value, message):
    config = replace(_config(2), **{field: value})
    with pytest.raises(ValueError, match=message):
        _validate_sequence_parallel_runtime(config, _build_mammoth_config(config))


def test_sp_rejects_dev_without_changing_single_rank_runtime():
    config = _config(2)
    model_config = _build_mammoth_config(config)
    model_config.llm_config.model_type = "mammothmoda2_qwen3_vl"
    with pytest.raises(ValueError, match="Preview text-to-image, not Dev"):
        _validate_sequence_parallel_runtime(config, model_config)
    _validate_sequence_parallel_runtime(_config(), model_config)


@pytest.mark.parametrize("guidance", [1.0, 4.0], ids=["no_cfg", "sequential_cfg"])
def test_single_rank_runtime_matches_pre_boundary_pipeline_and_replays_requests(monkeypatch, guidance):
    config = _config()
    pipeline = _pipeline(config, monkeypatch)
    baseline = _pipeline(config, monkeypatch)
    baseline.load_state_dict(pipeline.state_dict(), strict=True)

    def legacy_main(hidden_states, attention_mask, rotary_emb, temb):
        for layer in baseline.gen_transformer.layers:
            hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)
        return hidden_states

    monkeypatch.setattr(baseline.gen_transformer, "_apply_transformer_layers", legacy_main)
    observed_lengths = []

    def record_branch(module, args, kwargs):
        observed_lengths.append(kwargs["text_hidden_states"].shape[1])

    handle = pipeline.gen_transformer.register_forward_pre_hook(record_branch, with_kwargs=True)
    outputs = []
    try:
        with torch.inference_mode():
            # One context per request, with two differently shaped CFG branches.
            for text_len in (3, 2, 3):
                with set_forward_context(omni_diffusion_config=config):
                    actual = pipeline(_request(text_len, guidance)).output
                    ctx = get_forward_context()
                    assert (ctx.sp_original_seq_len, ctx.sp_padding_size, ctx._sp_shard_depth) == (None, 0, 0)
                with set_forward_context(omni_diffusion_config=config):
                    expected = baseline(_request(text_len, guidance)).output
                assert actual.shape == (1, 3, 32, 48)
                assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                outputs.append(actual)
    finally:
        handle.remove()
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)
    assert not torch.equal(outputs[0], outputs[1])
    expected_lengths = []
    for positive in (5, 4, 5):
        expected_lengths.extend(([positive, 0] if guidance > 1 else [positive]) * 2)
    assert observed_lengths == expected_lengths

    # Follow the real engine postprocess/format path through a savable image,
    # not merely a finite raw VAE tensor.
    postprocess = get_diffusion_post_process_func(config)
    assert postprocess is not None
    result = DiffusionOutput(output=outputs[0])
    formatted = format_diffusion_outputs(
        request=_request(3, guidance).requests[0],
        od_config=config,
        diffusion_output=result,
        output_data=result.output,
        postprocess_output=normalize_diffusion_postprocess_output(postprocess(result.output)),
    )
    image = formatted[0].images[0]
    assert isinstance(image, Image.Image) and image.size == (48, 32)
    with BytesIO() as buffer:
        image.save(buffer, format="PNG")
        assert buffer.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("output_type", ["pil", "np", "pt"])
def test_registered_postprocess_denormalizes_decoded_images(output_type):
    config = replace(_config(), output_type=output_type)
    postprocess = get_diffusion_post_process_func(config)
    assert postprocess is not None
    # Positive-only values still need VAE denormalization; range guessing is
    # not equivalent to the released [-1, 1] image contract.
    decoded = torch.full((1, 3, 2, 4), 0.5)
    result = postprocess(decoded)
    if output_type == "pil":
        assert isinstance(result[0], Image.Image) and result[0].size == (4, 2)
        assert result[0].getpixel((0, 0)) == (191, 191, 191)
    elif output_type == "np":
        assert isinstance(result, np.ndarray) and result.shape == (1, 2, 4, 3)
        np.testing.assert_array_equal(result, np.full((1, 2, 4, 3), 0.75))
    else:
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, torch.full_like(decoded, 0.75))


def test_decoded_output_does_not_claim_to_be_latents():
    with pytest.raises(ValueError, match="decoded images"):
        get_diffusion_post_process_func(replace(_config(), output_type="latent"))
