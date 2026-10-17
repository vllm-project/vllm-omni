# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU unit tests for the π0.5 VLA model.

Deliberately lightweight: config plumbing and the checkpoint-boundary rule, the
state discretization path, relative actions, and the weight-load remap. Runs on
CPU with no weights and no lerobot. The full LeRobot parity check lives in
``tests/diffusion/models/pi05/test_pi05_parity.py``.

Most assertions here guard *silent* failure modes — cases where the wrong
behaviour still produces a well-shaped, finite action chunk:

  * normalization must run before state discretization;
  * the normalization mode comes from the checkpoint, since π0.5 defaults to
    quantile where π0 defaults to ``mean_std``;
  * relative-action checkpoints must be honoured, and an unresolvable
    ``relative_exclude_joints`` must raise rather than silently make every
    dimension relative;
  * MEM / RTC checkpoints must be rejected rather than served incorrectly.

    pytest tests/diffusion/models/pi05/test_pi05_units.py -v
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.pi05 import exact_qk_pi05 as exact_qk
from vllm_omni.diffusion.models.pi05 import modeling_pi05
from vllm_omni.diffusion.models.pi05 import precision_kernels_pi05 as kernels
from vllm_omni.diffusion.models.pi05.config import (
    Pi05Config,
    UnsupportedCheckpointCapabilityError,
    load_lerobot_norm_stats,
    resolve_excluded_action_indices,
)
from vllm_omni.diffusion.models.pi05.cuda_graph_pi05 import Pi05CUDAGraphs
from vllm_omni.diffusion.models.pi05.modeling_pi05 import (
    GemmaVariantConfig,
    Pi05AdaRMSNorm,
    Pi05ForActionPrediction,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
)
from vllm_omni.diffusion.models.pi05.processor_pi05 import (
    Pi05ImageProcessor,
    Pi05Processor,
    Pi05RelativeActions,
    _assemble_model_inputs,
    apply_norm,
    as_state_vector,
    build_norm_stats,
    build_pi05_prompt,
    discretize_state,
    resize_with_pad,
)
from vllm_omni.diffusion.models.pi05.runtime_pi05 import Pi05RuntimeOptions
from vllm_omni.diffusion.models.pi05.vision_pi05 import Pi05VisionOptions

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("args", [{"pi05_vision_precision": "bf16"}, {"pi05_batch_cameras": "false"}])
def test_p0_rejects_invalid_vision_options(args):
    with pytest.raises(ValueError):
        Pi05VisionOptions.from_pipeline_args(args)


def _p0_vision_model():
    """Small real SigLIP; no checkpoint or multi-billion-parameter backbone."""
    from transformers import SiglipVisionConfig, SiglipVisionModel

    model = object.__new__(Pi05ForActionPrediction)
    torch.nn.Module.__init__(model)
    backbone = object.__new__(modeling_pi05.PaliGemmaWithActionExpertPi05)
    torch.nn.Module.__init__(backbone)
    backbone.paligemma = torch.nn.Module()
    backbone.paligemma.model = torch.nn.Module()
    pg = backbone.paligemma.model
    pg.vision_tower = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
            vision_use_head=False,
        )
    )
    pg.multi_modal_projector = torch.nn.Linear(16, 8)
    pg.language_model = torch.nn.Module()
    pg.language_model.embed_tokens = torch.nn.Embedding(32, 8).to(torch.bfloat16)
    backbone.gemma_expert = torch.nn.Linear(8, 8).to(torch.bfloat16)
    backbone.vision_options = Pi05VisionOptions()
    model.paligemma_with_expert = backbone
    model.config = Pi05Config()
    model.cuda_graphs = None
    model._vision_configured = False
    return model.eval()


@pytest.mark.parametrize("precision", ["fp32", "bf16_autocast", "selective_bf16"])
@pytest.mark.parametrize("batch_cameras", [False, True])
def test_p0_vision_policies_preserve_prefix_interface(precision, batch_cameras):
    model = _p0_vision_model()
    backbone = model.paligemma_with_expert
    expert_before = backbone.gemma_expert.weight.detach().clone()
    model.configure_vision(Pi05VisionOptions(precision, batch_cameras))
    tower = backbone.paligemma.model.vision_tower
    vision = getattr(tower, "vision_model", tower)
    assert vision.embeddings.patch_embedding.weight.dtype == torch.float32
    assert vision.embeddings.position_embedding.weight.dtype == torch.float32
    expected = torch.bfloat16 if precision == "selective_bf16" else torch.float32
    assert vision.encoder.layers[0].self_attn.q_proj.weight.dtype == expected
    assert backbone.paligemma.model.multi_modal_projector.weight.dtype == expected
    torch.testing.assert_close(backbone.gemma_expert.weight, expert_before, rtol=0, atol=0)
    images = [torch.randn(2, 3, 8, 8) for _ in range(3)]
    masks = [torch.tensor([True, False]), torch.tensor([False, True]), torch.tensor([True, True])]
    with torch.no_grad():
        embs, pad, att = model.embed_prefix(
            images, masks, torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.bool)
        )
    assert embs.dtype == torch.float32 and torch.isfinite(embs).all()
    assert embs.shape == (2, 14, 8)
    for index, mask in enumerate(masks):
        torch.testing.assert_close(pad[:, index * 4 : (index + 1) * 4], mask[:, None].expand(2, 4))
    assert not att.any()
    with pytest.raises(RuntimeError, match="initialization-only"):
        model.configure_vision(Pi05VisionOptions())


@pytest.mark.parametrize("num_real_views", [1, 2, 3])
def test_p0_camera_batching_preserves_slot_and_sample_order(num_real_views):
    from types import SimpleNamespace

    calls = []

    def embed_image(image):
        calls.append(image.shape[0])
        return image[:, :1, :1, :1].reshape(-1, 1, 1).expand(-1, 2, 2)

    backbone = SimpleNamespace(
        vision_options=Pi05VisionOptions(),
        embed_image=embed_image,
        embed_language_tokens=lambda tokens: torch.zeros(tokens.shape[0], tokens.shape[1], 2),
    )
    model = SimpleNamespace(config=Pi05Config(), paligemma_with_expert=backbone)
    images = [
        torch.tensor([10 * camera, 10 * camera + 1], dtype=torch.float32).reshape(2, 1, 1, 1) for camera in range(3)
    ]
    masks = [torch.tensor([camera < num_real_views] * 2) for camera in range(3)]
    tokens, lang_mask = torch.zeros(2, 1, dtype=torch.long), torch.ones(2, 1, dtype=torch.bool)
    sequential = Pi05ForActionPrediction.embed_prefix(model, images, masks, tokens, lang_mask)
    assert calls == [2, 2, 2]
    calls.clear()
    backbone.vision_options = Pi05VisionOptions(batch_cameras=True)
    batched = Pi05ForActionPrediction.embed_prefix(model, images, masks, tokens, lang_mask)
    assert calls == [6]
    for actual, expected in zip(batched, sequential):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_p0_cannot_change_vision_options_after_graph_install():
    model = _p0_vision_model()
    model.cuda_graphs = object()
    with pytest.raises(RuntimeError, match="initialization-only"):
        model.configure_vision(Pi05VisionOptions(precision="selective_bf16"))


def _p0_runtime_model(dtype, lm_precision="legacy"):
    """Real small Gemma/SigLIP with nonzero conditioning, preserving model APIs."""
    from types import SimpleNamespace

    from transformers import GemmaConfig, SiglipVisionConfig, SiglipVisionModel
    from transformers.models.gemma.modeling_gemma import GemmaForCausalLM

    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import _set_inference_dtype

    variant = GemmaVariantConfig(16, 2, 32, 2, 1, 8)
    config = GemmaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        hidden_activation="gelu_pytorch_tanh",
    )
    backbone = object.__new__(modeling_pi05.PaliGemmaWithActionExpertPi05)
    torch.nn.Module.__init__(backbone)
    backbone.paligemma = torch.nn.Module()
    backbone.paligemma.model = torch.nn.Module()
    pg = backbone.paligemma.model
    pg.language_model = GemmaForCausalLM(config).model
    pg.vision_tower = SiglipVisionModel(
        SiglipVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=8,
            patch_size=4,
            vision_use_head=False,
        )
    )
    pg.multi_modal_projector = torch.nn.Linear(16, 16)
    backbone.paligemma.config = SimpleNamespace(text_config=config, vision_config=pg.vision_tower.config)
    backbone.gemma_expert = GemmaForCausalLM(config)
    backbone.gemma_expert.model.embed_tokens = None
    backbone.adarms_cond_dim = 16
    backbone._install_adarms_norms(variant)
    backbone.fused_kernels = False
    backbone.vision_options = Pi05VisionOptions()
    with torch.no_grad():
        for layer in backbone.gemma_expert.modules():
            if isinstance(layer, Pi05AdaRMSNorm):
                layer.dense.weight.normal_(std=0.04)
                layer.dense.bias.normal_(std=0.03)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(modeling_pi05, "get_gemma_config", lambda _variant: variant)
        patch.setattr(modeling_pi05, "PaliGemmaWithActionExpertPi05", lambda *_args: backbone)
        model = Pi05ForActionPrediction(
            Pi05Config(
                max_action_dim=4,
                max_state_dim=4,
                chunk_size=4,
                n_action_steps=4,
                tokenizer_max_length=4,
                image_resolution=(8, 8),
                num_inference_steps=10,
                lm_precision=lm_precision,
            )
        )
    _set_inference_dtype(model, dtype)
    model.eval()
    model.kv_cache = model.new_kv_cache(batch_size=1)
    return model


@pytest.mark.parametrize("value", ["bf16", None, False, ["lerobot_bf16"]])
def test_lm_precision_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="lm_precision"):
        Pi05Config(lm_precision=value)


@pytest.mark.parametrize("mode", ["legacy", "lerobot_bf16"])
def test_lm_precision_pipeline_argument_reaches_config(mode):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    pipeline = object.__new__(Pi05Pipeline)
    pipeline.model_dir = None
    args = SimpleNamespace(model_config={}, custom_pipeline_args={"pi05_lm_precision": mode})
    assert pipeline._build_config(args).lm_precision == mode
    assert args.model_config == {}  # Resolving an option must not mutate the caller.
    assert Pi05Config().lm_precision == "legacy"


@pytest.mark.parametrize("mode", ["legacy", "lerobot_bf16"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lm_precision_controls_prefix_and_expert_norm_residual_dtype(mode, dtype):
    model = _p0_runtime_model(dtype, mode)
    backbone = model.paligemma_with_expert
    expected_dtype = torch.bfloat16 if mode == "lerobot_bf16" and dtype == torch.bfloat16 else torch.float32
    observed = []
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def record(_module, args, result):
        values = result if isinstance(result, tuple) else (result,)
        observed.append((args[0].dtype, tuple(value.dtype for value in values if value is not None)))

    for lm in (backbone.paligemma.model.language_model, backbone.gemma_expert.model):
        for layer in lm.layers:
            handles.extend(
                norm.register_forward_hook(record) for norm in (layer.input_layernorm, layer.post_attention_layernorm)
            )
        handles.append(lm.norm.register_forward_hook(record))

    prefix = torch.randn(1, 16, 16)
    suffix = torch.randn(1, 4, 16)
    cond = torch.randn(1, 16)  # The factory uses nonzero AdaRMS weights and biases.
    cache = model.new_kv_cache(batch_size=1)
    try:
        prefix_output, _ = backbone.forward(
            inputs_embeds=[prefix, None],
            attention_mask=torch.zeros(1, 1, 16, 16),
            position_ids=torch.arange(16)[None],
            past_key_values=cache,
            use_cache=True,
        )
        suffix_output, _ = backbone.forward(
            inputs_embeds=[None, suffix],
            attention_mask=torch.zeros(1, 1, 4, 20),
            position_ids=torch.arange(16, 20)[None],
            past_key_values=cache,
            adarms_cond=cond,
        )
    finally:
        for handle in handles:
            handle.remove()
    assert prefix_output[0].dtype == suffix_output[1].dtype == expected_dtype
    assert len(observed) == 10
    assert all(input_dtype == expected_dtype for input_dtype, _ in observed)
    assert all(all(output_dtype == expected_dtype for output_dtype in outputs) for _, outputs in observed)
    assert prefix.dtype == suffix.dtype == cond.dtype == torch.float32
    assert model.action_in_proj.weight.dtype == model.action_out_proj.weight.dtype == torch.float32
    prefix_rope = model.prepare_prefix_rotary(prefix, torch.arange(16)[None])
    context = model.prepare_denoise_context(torch.ones(1, 16, dtype=torch.bool))
    assert prefix_rope[0].dtype == prefix_rope[1].dtype == dtype
    assert context.cos.dtype == context.sin.dtype == dtype


def test_legacy_lm_precision_keeps_original_prefix_hidden_and_kv_exact():
    model = _p0_runtime_model(torch.bfloat16)
    backbone = model.paligemma_with_expert
    prefix = torch.randn(1, 16, 16)
    mask = torch.zeros(1, 1, 16, 16)
    positions = torch.arange(16)[None]
    hidden = prefix
    expected_kv = []
    for index in range(2):
        hidden, kv = modeling_pi05._compute_layer_prefix_only(index, hidden, mask, positions, backbone.paligemma)
        expected_kv.append(kv)
    hidden = backbone.paligemma.model.language_model.norm(hidden)
    cache = model.new_kv_cache(batch_size=1)
    output, _ = backbone.forward(
        inputs_embeds=[prefix, None],
        attention_mask=mask,
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
    )
    torch.testing.assert_close(output[0], hidden, rtol=0, atol=0)
    for index, (key, value) in enumerate(expected_kv):
        torch.testing.assert_close(cache.key[index, :, :, :16], key, rtol=0, atol=0)
        torch.testing.assert_close(cache.value[index, :, :, :16], value, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_lm_precision_leaves_vision_timestep_and_fp32_execution_unchanged(dtype):
    import copy

    torch.manual_seed(421)
    legacy = _p0_runtime_model(dtype)
    aligned = copy.deepcopy(legacy)
    aligned.paligemma_with_expert.lm_precision = "lerobot_bf16"
    inputs = dict(
        images=[torch.randn(1, 3, 8, 8) for _ in range(3)],
        image_masks=[torch.tensor([True]), torch.tensor([False]), torch.tensor([True])],
        lang_tokens=torch.tensor([[1, 2, 3, 4]]),
        lang_masks=torch.tensor([[True, True, False, False]]),
    )
    for actual, expected in zip(aligned.embed_prefix(**inputs), legacy.embed_prefix(**inputs)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for model in (legacy, aligned):
        model.configure_runtime(Pi05RuntimeOptions(precompute_adarms=True))
        model.prepare_runtime_caches()
    for actual, expected in zip(aligned.runtime_step_table(10, 1), legacy.runtime_step_table(10, 1)):
        for name in ("timestep", "time_cond", "modulations"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
    noise = torch.randn(1, 4, 4)
    original = legacy.sample_actions(**inputs, noise=noise)
    result = aligned.sample_actions(**inputs, noise=noise)
    if dtype == torch.float32:
        torch.testing.assert_close(result, original, rtol=0, atol=0)
    else:
        assert not torch.equal(result, original), "The BF16 decoder-entry policy must actually affect execution."
    assert result.dtype == torch.float32  # FP32 Euler integration and action output are preserved.


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_p0_precomputed_adarms_matches_nonzero_live_projections(dtype):
    model = _p0_runtime_model(dtype)
    model.configure_runtime(Pi05RuntimeOptions(precompute_adarms=True))
    model.prepare_runtime_caches()
    table = model.runtime_step_table(num_steps=10, batch_size=1)
    assert table is not None and len(table) == 10
    expert = model.paligemma_with_expert.gemma_expert.model
    norms = [norm for layer in expert.layers for norm in (layer.input_layernorm, layer.post_attention_layernorm)]
    norms.append(expert.norm)
    for index, data in enumerate(table):
        torch.testing.assert_close(data.timestep, torch.full((1,), 1.0 + index * (-1.0 / 10)), rtol=0, atol=0)
        expected_cond = model.embed_timestep(data.timestep)
        torch.testing.assert_close(data.time_cond, expected_cond, rtol=0, atol=0)
        assert torch.count_nonzero(data.modulations)
        for row, norm in zip(data.modulations, norms):
            expected_modulation = norm.dense(expected_cond.to(norm.dense.weight.dtype))
            torch.testing.assert_close(row, expected_modulation, rtol=0, atol=0)
            x = torch.randn(1, 4, 16)
            expected = norm(x, expected_cond)
            actual = norm(x, modulation=row)
            for lhs, rhs in zip(actual, expected):
                torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    assert model.runtime_step_table(num_steps=1, batch_size=1) is None
    assert model.runtime_step_table(num_steps=4, batch_size=1) is None
    assert model.runtime_step_table(num_steps=10, batch_size=2) is None


@pytest.mark.parametrize("mutation", ["inplace", "replace", "dtype"])
def test_p0_precomputed_tables_are_invalidated_after_parameter_changes(mutation):
    model = _p0_runtime_model(torch.bfloat16)
    model.configure_runtime(Pi05RuntimeOptions(precompute_adarms=True))
    model.prepare_runtime_caches()
    assert model.runtime_step_table(num_steps=10, batch_size=1) is not None
    dense = model.paligemma_with_expert.gemma_expert.model.layers[0].input_layernorm.dense
    if mutation == "inplace":
        with torch.no_grad():
            dense.weight.add_(0.01)
    elif mutation == "replace":
        dense.weight = torch.nn.Parameter(dense.weight.detach().clone())
    else:
        model.to(dtype=torch.float32)
    assert model.runtime_step_table(num_steps=10, batch_size=1) is None


def test_p0_pipeline_precompute_survives_outer_inference_mode(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.pi05 import pipeline_pi05

    def load_checkpoint(_pipeline, model):
        # Use a different weight from construction to prove preparation follows
        # loading. These are real tiny-model parameters, not inference tensors.
        assert not torch.is_inference(model.time_mlp_in.weight)
        model.time_mlp_in.weight.fill_(0.07)

    (tmp_path / "model.safetensors").touch()
    pipeline = object.__new__(pipeline_pi05.Pi05Pipeline)
    pipeline.model_dir = str(tmp_path)
    pipeline.config = object()
    pipeline.vision_options = Pi05VisionOptions(precision="selective_bf16", batch_cameras=True)
    pipeline.runtime_options = Pi05RuntimeOptions(precompute_adarms=True)
    pipeline._device = torch.device("cpu")
    pipeline._torch_dtype = torch.bfloat16
    monkeypatch.setattr(pipeline_pi05, "Pi05ForActionPrediction", lambda _config: _p0_runtime_model(torch.bfloat16))
    monkeypatch.setattr(pipeline_pi05.Pi05Pipeline, "_load_checkpoint", load_checkpoint)
    with torch.inference_mode():
        model = pipeline._initialize_model()
        assert all(not torch.is_inference(param) for param in model.parameters())
        table = model.runtime_step_table(num_steps=10, batch_size=1)
        assert table is not None
        for data in table:
            assert not torch.is_inference(data.modulations)
            torch.testing.assert_close(data.time_cond, model.embed_timestep(data.timestep), rtol=0, atol=0)


def _p0_pipeline_with_runtime_model(*, captured):
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    pipeline = object.__new__(Pi05Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.model = _p0_runtime_model(torch.bfloat16)
    pipeline.model.configure_runtime(Pi05RuntimeOptions(precompute_adarms=True))
    pipeline.model.prepare_runtime_caches()
    if captured:
        # Exercise the real adapter/model guards without pretending CPU can
        # capture a CUDA graph. Actual loader/capture coverage runs on GPU.
        pipeline.model.cuda_graphs = Pi05CUDAGraphs(pipeline.model)
    return pipeline


def test_p0_pipeline_accepts_loader_noop_placement_without_touching_graph_inputs():
    pipeline = _p0_pipeline_with_runtime_model(captured=True)
    model = pipeline.model
    graphs, rows = model.cuda_graphs, model.runtime_step_table(10, 1)
    tensors = list(pipeline.parameters()) + list(pipeline.buffers()) + [model.kv_cache.key, model.kv_cache.value]
    before = [(tensor.data_ptr(), tensor.dtype, tensor._version) for tensor in tensors]
    assert pipeline.to(torch.device("cpu")) is pipeline
    assert pipeline.to(device="cpu", non_blocking=True) is pipeline
    assert pipeline.to() is pipeline
    assert [(tensor.data_ptr(), tensor.dtype, tensor._version) for tensor in tensors] == before
    assert model.cuda_graphs is graphs and model.runtime_step_table(10, 1) is rows


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "meta"},
        {"dtype": torch.float32},
        {"dtype": torch.bfloat16},
        {"memory_format": torch.channels_last},
    ],
)
def test_p0_pipeline_rejects_conversion_after_capture(kwargs):
    pipeline = _p0_pipeline_with_runtime_model(captured=True)
    with pytest.raises(RuntimeError, match="require a new pipeline"):
        pipeline.to(**kwargs)
    assert pipeline.model.runtime_step_table(10, 1) is not None
    assert pipeline.model.action_in_proj.weight.device.type == "cpu"
    assert pipeline.model.action_in_proj.weight.dtype == torch.float32


def test_p0_pipeline_noop_placement_validates_buffers_and_unknown_arguments():
    pipeline = _p0_pipeline_with_runtime_model(captured=True)
    with pytest.raises(TypeError):
        pipeline.to(device="cpu", unexpected_option=True)
    with pytest.raises(RuntimeError, match="copy argument"):
        pipeline.to(device="cpu", copy=True)
    pipeline.register_buffer("other_device_buffer", torch.empty(0, device="meta"))
    with pytest.raises(RuntimeError, match="require a new pipeline"):
        pipeline.to("cpu")


def test_p0_pipeline_noop_cuda_device_resolves_current_index(monkeypatch):
    from types import SimpleNamespace

    pipeline = _p0_pipeline_with_runtime_model(captured=True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    # Only the placement metadata is simulated here; the preceding tests use
    # real tensors and verify pointer/version/cache preservation.
    monkeypatch.setattr(pipeline, "parameters", lambda: iter([SimpleNamespace(device=torch.device("cuda:3"))]))
    monkeypatch.setattr(pipeline, "buffers", lambda: iter([]))
    assert pipeline.to("cuda") is pipeline
    with pytest.raises(RuntimeError, match="require a new pipeline"):
        pipeline.to("cuda:4")


def test_p0_pipeline_ordinary_conversion_before_capture_is_unchanged():
    pipeline = _p0_pipeline_with_runtime_model(captured=False)
    assert pipeline.to(dtype=torch.float32) is pipeline
    assert all(parameter.dtype == torch.float32 for parameter in pipeline.parameters())
    assert pipeline.model.runtime_step_table(10, 1) is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_steps", [1, 4, 10])
@pytest.mark.parametrize("lm_precision", ["legacy", "lerobot_bf16"])
@pytest.mark.parametrize(
    "flags",
    [
        {"precompute_adarms": True},
        {"hoist_denoise_invariants": True},
        {"prefix_kv_only": True},
        {"precompute_adarms": True, "hoist_denoise_invariants": True, "prefix_kv_only": True},
    ],
)
def test_p0_runtime_flags_preserve_actions_across_requests(dtype, num_steps, lm_precision, flags):
    import copy

    model = _p0_runtime_model(dtype, lm_precision)
    optimized = copy.deepcopy(model)
    optimized.configure_runtime(Pi05RuntimeOptions(**flags))
    optimized.prepare_runtime_caches()
    images = [torch.randn(1, 3, 8, 8) for _ in range(3)]
    tokens = torch.tensor([[1, 2, 3, 4]])
    noise = torch.randn(1, 4, 4)
    requests = [
        ([True, False, True], [True, True, False, False]),
        ([False, True, False], [True, True, True, True]),
        ([True, False, True], [True, True, False, False]),
    ]
    outputs = []
    for camera_masks, text_mask in requests:
        inputs = dict(
            images=images,
            image_masks=[torch.tensor([mask]) for mask in camera_masks],
            lang_tokens=tokens,
            lang_masks=torch.tensor([text_mask]),
            noise=noise,
            num_steps=num_steps,
        )
        expected = model.sample_actions(**inputs)
        actual = optimized.sample_actions(**inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        outputs.append(actual.clone())
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("lm_precision", ["legacy", "lerobot_bf16"])
def test_p0_prefix_kv_only_keeps_general_forward_and_skips_unused_modules(dtype, lm_precision, monkeypatch):
    model = _p0_runtime_model(dtype, lm_precision)
    backbone = model.paligemma_with_expert
    embs = torch.randn(1, 16, 16)
    position_ids = torch.arange(16)[None, :]
    attention_mask = torch.zeros(1, 1, 16, 16)
    cache = model.new_kv_cache(batch_size=1)
    inputs = dict(
        inputs_embeds=[embs, None],
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
    )
    full_hidden, _ = backbone.forward(**inputs)
    assert full_hidden[0] is not None and full_hidden[0].shape == embs.shape
    expected_k, expected_v = cache.key.clone(), cache.value.clone()

    def unexpected(*args, **kwargs):
        raise AssertionError("KV-only executed unused final prefix hidden-state work")

    final_layer = backbone.paligemma.model.language_model.layers[-1]
    for module in (
        final_layer.self_attn.q_proj,
        final_layer.self_attn.o_proj,
        final_layer.mlp,
        backbone.paligemma.model.language_model.norm,
    ):
        monkeypatch.setattr(module, "forward", unexpected)
    result, _ = backbone.forward(**inputs, prefix_kv_only=True)
    assert result[0] is None
    torch.testing.assert_close(cache.key[..., :16, :], expected_k[..., :16, :], rtol=0, atol=0)
    torch.testing.assert_close(cache.value[..., :16, :], expected_v[..., :16, :], rtol=0, atol=0)


class _P0GraphAdapterModel:
    """Records adapter fallbacks; no mocked model math is treated as parity."""

    def __init__(self, options):
        from types import SimpleNamespace

        self.runtime_options = options
        self.calls = []
        self.paligemma_with_expert = SimpleNamespace(forward=self.forward)

    def forward(self, **kwargs):
        self.calls.append(("prefix", kwargs))
        return [torch.tensor(123), None], kwargs["past_key_values"]

    def denoise_step(self, **kwargs):
        self.calls.append(("denoise", kwargs))
        return torch.tensor(-999)


def _p0_graph_context(value):
    return modeling_pi05.Pi05DenoiseContext(
        torch.full((1, 1, 2, 5), value, dtype=torch.float32),
        torch.full((1, 2), value, dtype=torch.long),
        torch.full((1, 2, 4), value, dtype=torch.float32),
        torch.full((1, 2, 4), -value, dtype=torch.float32),
    )


def _p0_graph_step(value):
    return modeling_pi05.Pi05DenoiseStepData(
        torch.tensor([1.0 - value / 10]),
        torch.full((1, 2), value, dtype=torch.float32),
        torch.full((3, 1, 6), value, dtype=torch.float32),
    )


def test_p0_graph_adapter_refreshes_steps_and_request_invariants(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.pi05 import cuda_graph_pi05 as adapter

    model = _P0GraphAdapterModel(Pi05RuntimeOptions(True, True, True))
    graphs = Pi05CUDAGraphs(model)
    cache = object()
    context, row = _p0_graph_context(0), _p0_graph_step(0)
    mask, actions, output = torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, 2, 2), torch.zeros(1)
    loads = []
    original_load = adapter._load

    def load(static, source):
        if static is not source:
            loads.append(static.data_ptr())
        original_load(static, source)

    def replay():
        output.copy_(row.modulations.sum() + row.time_cond.sum() + context.attention_mask.sum())

    monkeypatch.setattr(adapter, "_load", load)
    graphs._denoise_step = adapter._DenoiseStepGraph(
        SimpleNamespace(replay=replay), mask, cache, actions, row.timestep, output, context, row
    )
    results = []
    for request in (_p0_graph_context(2), _p0_graph_context(7), _p0_graph_context(2)):
        prepared = graphs.prepare_denoise_context(request)
        assert prepared is context
        before = len(loads)
        values = []
        for value in (1, 2, 3):
            step = _p0_graph_step(value)
            result = graphs.denoise_step(
                mask.clone(), cache, actions.clone(), step.timestep, context=prepared, step_data=step
            )
            expected = step.modulations.sum() + step.time_cond.sum() + request.attention_mask.sum()
            torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
            values.append(result.clone())
        # x_t, timestep, time_cond, one complete modulation tensor per step;
        # the request's mask/position/RoPE are not recopied each step.
        assert len(loads) - before == 12
        results.append(values)
    assert loads.count(row.modulations.data_ptr()) == 9
    assert graphs.num_replays["denoise_step"] == 9
    for index in range(3):
        torch.testing.assert_close(results[0][index], results[2][index], rtol=0, atol=0)
        assert not torch.equal(results[0][index], results[1][index])
    assert not torch.equal(results[0][0], results[0][1])
    graphs.denoise_step(mask, cache, actions, row.timestep, context=context)
    assert model.calls[-1][0] == "denoise" and graphs.num_replays["denoise_step"] == 9


def test_p0_graph_kv_only_capture_preserves_general_forward_fallback():
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.pi05 import cuda_graph_pi05 as adapter

    model = _P0GraphAdapterModel(Pi05RuntimeOptions(prefix_kv_only=True))
    graphs = Pi05CUDAGraphs(model)
    cache = object()
    mask, positions, embeddings = torch.zeros(1, 1, 3, 3), torch.zeros(1, 3, dtype=torch.long), torch.zeros(1, 3, 4)
    graphs._prefix_forward = adapter._PrefixForwardGraph(
        SimpleNamespace(replay=lambda: None), mask, positions, embeddings, cache, ([None, None], cache), True
    )
    inputs = dict(
        attention_mask=mask,
        position_ids=positions,
        past_key_values=cache,
        inputs_embeds=[embeddings, None],
        use_cache=True,
    )
    full, _ = graphs.prefix_forward(**inputs)
    assert full[0].item() == 123 and graphs.num_replays["prefix_forward"] == 0
    kv, _ = graphs.prefix_forward(**inputs, prefix_kv_only=True)
    assert kv[0] is None and graphs.num_replays["prefix_forward"] == 1


# ----------------------------------------------------------------------------
# Pi05Config dataclass + checkpoint resolver
# ----------------------------------------------------------------------------
_LEROBOT_CFG: dict[str, Any] = {
    "type": "pi05",
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "chunk_size": 50,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "num_inference_steps": 10,
    "image_resolution": [224, 224],
    "tokenizer_max_length": 200,
    "dtype": "float32",
    "input_features": {
        "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.left_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.right_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [32]},
    },
    "output_features": {"action": {"type": "ACTION", "shape": [32]}},
}

_EXPECTED_CAMERA_ORDER = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]

_ACTION_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]


def test_config_parses_lerobot_pi05_json():
    """``_LEROBOT_CFG`` mirrors the runtime keys of a real checkpoint."""
    c = Pi05Config.from_model_config(_LEROBOT_CFG)
    assert c.tokenizer_max_length == 200
    assert c.state_num_bins == 256
    assert c.image_feature_keys == _EXPECTED_CAMERA_ORDER
    assert c.image_resolution == (224, 224)


def test_config_rejects_wrong_model_type():
    with pytest.raises(ValueError, match="type='pi05'"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, type="pi0"))


def test_config_rejects_unsupported_partial_action_chunk():
    with pytest.raises(UnsupportedCheckpointCapabilityError, match="n_action_steps"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, n_action_steps=10))


@pytest.mark.parametrize(
    "key,value",
    [
        ("output_features", {"action": {"type": "ACTION", "shape": [33]}}),
        ("input_features", {**_LEROBOT_CFG["input_features"], "observation.state": {"type": "STATE", "shape": [33]}}),
    ],
    ids=["action", "state"],
)
def test_config_rejects_a_schema_wider_than_the_model(key, value):
    """``max_action_dim`` / ``max_state_dim`` are the widths the weights were
    built for, so a wider checkpoint schema cannot be served."""
    with pytest.raises(ValueError, match="over max_state_dim"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, **{key: value}))


@pytest.mark.parametrize(
    "key,value,attribute",
    [
        ("output_features", {"action": {"type": "ACTION", "shape": [7]}}, "action_dim"),
        (
            "input_features",
            {**_LEROBOT_CFG["input_features"], "observation.state": {"type": "STATE", "shape": [7]}},
            "state_dim",
        ),
    ],
    ids=["action", "state"],
)
def test_config_derives_the_real_width_from_the_schema(key, value, attribute):
    """The declared width decides how many values reach the prompt and how many
    columns come back, so a 7-joint checkpoint must not be padded out to 32."""
    config = Pi05Config.from_model_config(dict(_LEROBOT_CFG, **{key: value}))
    assert getattr(config, attribute) == 7


def test_config_rejects_inconsistent_openpi_metadata():
    with pytest.raises(ValueError, match="policy_server_config.action_dim"):
        Pi05Config.from_model_config(
            dict(
                _LEROBOT_CFG,
                output_features={"action": {"type": "ACTION", "shape": [7]}},
                policy_server_config={"action_dim": 32},
            )
        )


def test_config_non_square_resolution_raises():
    with pytest.raises(ValueError):
        Pi05Config(image_resolution=(224, 256))


@pytest.mark.parametrize("key", ["max_cameras", "num_inference_steps"])
def test_config_rejects_counts_below_one(key):
    """``build_model_inputs`` emits one image slot per camera and the Euler loop
    steps ``-1 / num_inference_steps``; neither is defined at zero."""
    with pytest.raises(ValueError, match=f"{key} must be at least 1"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, **{key: 0}))


@pytest.mark.parametrize("key", ["state", "action"])
def test_config_rejects_norm_stats_that_do_not_match_the_declared_width(key):
    """A checkpoint's statistics and its feature schema come from different
    files, so they can disagree."""
    stats = {key: {"mode": "quantile", "q01": [0.0] * 2, "q99": [1.0] * 2}}
    with pytest.raises(ValueError, match=rf"norm_stats\[{key}\]\[q01\] has 2 entries, expected 32"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, norm_stats=stats))


# ----------------------------------------------------------------------------
# LeRobot normalization-stats sidecar
# ----------------------------------------------------------------------------
# LeRobot keeps stats out of config.json: policy_preprocessor.json holds the
# structure and one safetensors file per stateful step holds the numbers.
def _normalized(raw, stats):
    """The serving path's state normalization, without the pipeline."""
    return apply_norm(torch.as_tensor(raw, dtype=torch.float32), build_norm_stats({"state": stats}, "state")).numpy()


def _write_lerobot_checkpoint(
    tmp_path,
    *,
    norm_map: dict[str, str] | None = None,
    stats: dict[str, dict[str, list[float]]] | None = None,
    write_state_file: bool = True,
) -> str:
    """Write a minimal LeRobot-shaped checkpoint dir; return its path."""
    import json as _json

    import safetensors.torch as _st

    tmp_path.mkdir(parents=True, exist_ok=True)
    state_file = "policy_preprocessor_step_3_normalizer_processor.safetensors"
    step: dict = {
        "registry_name": "normalizer_processor",
        "config": {
            "eps": 1e-08,
            "features": {},
            "norm_map": norm_map if norm_map is not None else {"STATE": "QUANTILES", "ACTION": "QUANTILES"},
        },
    }
    if write_state_file:
        step["state_file"] = state_file
        flat = {
            f"{feature}.{stat}": torch.tensor(values, dtype=torch.float32)
            for feature, entry in (stats or {}).items()
            for stat, values in entry.items()
        }
        _st.save_file(flat, str(tmp_path / state_file))

    (tmp_path / "policy_preprocessor.json").write_text(
        _json.dumps({"name": "policy_preprocessor", "steps": [step]}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text(_json.dumps(_config_for(stats)), encoding="utf-8")
    return str(tmp_path)


def _config_for(stats: dict | None) -> dict:
    """``_LEROBOT_CFG`` with the state/action widths the stats actually cover."""
    if not stats:
        return _LEROBOT_CFG
    config = dict(_LEROBOT_CFG)
    state = next(iter(stats.get("observation.state", {}).values()), None)
    action = next(iter(stats.get("action", {}).values()), None)
    if state is not None:
        features = dict(config["input_features"])
        features["observation.state"] = {"type": "STATE", "shape": [len(state)]}
        config["input_features"] = features
    if action is not None:
        config["output_features"] = {"action": {"type": "ACTION", "shape": [len(action)]}}
    return config


# A LeRobot state_dict carries every stat compute_stats emits, not just the two
# the norm_map selects.
_FULL_DATASET_STATS = {
    "observation.state": {
        "mean": [10.0, 20.0],
        "std": [1.0, 2.0],
        "min": [-5.0, -5.0],
        "max": [5.0, 5.0],
        "q01": [-1.0, -2.0],
        "q99": [1.0, 2.0],
    },
    "action": {
        "mean": [0.5, 0.5],
        "std": [0.1, 0.1],
        "min": [-1.0, -1.0],
        "max": [1.0, 1.0],
        "q01": [-0.25, -0.5],
        "q99": [0.25, 0.5],
    },
}


@pytest.mark.parametrize(
    "declared,expected",
    [
        ("QUANTILES", {"mode": "quantile", "q01": [-1.0, -2.0], "q99": [1.0, 2.0]}),
        ("MEAN_STD", {"mode": "mean_std", "mean": [10.0, 20.0], "std": [1.0, 2.0]}),
        ("IDENTITY", None),
    ],
)
def test_load_lerobot_norm_stats_takes_the_mode_from_norm_map(tmp_path, declared, expected):
    """The regression guard. ``_FULL_DATASET_STATS`` carries all six statistics,
    as a real state_dict does, so an implementation that sniffed the present keys
    would answer mean_std for every case here. Only norm_map distinguishes them.
    """
    path = _write_lerobot_checkpoint(
        tmp_path, norm_map={"STATE": declared, "ACTION": "QUANTILES"}, stats=_FULL_DATASET_STATS
    )
    stats = load_lerobot_norm_stats(path)

    if expected is None:
        assert "state" not in stats
    else:
        assert stats["state"] == expected
    assert stats["action"]["mode"] == "quantile"


@pytest.mark.parametrize("shape", ["no_sidecar", "no_state_file"])
def test_load_lerobot_norm_stats_absent_stats_are_none(tmp_path, shape):
    """``lerobot/pi05_base`` is the second shape: a normalizer step with no
    stats attached. Neither is an error — the client is then expected to send an
    already-normalized state."""
    if shape == "no_sidecar":
        path = str(tmp_path)
    else:
        path = _write_lerobot_checkpoint(tmp_path, write_state_file=False)

    assert load_lerobot_norm_stats(path) is None


def test_load_lerobot_norm_stats_missing_declared_state_file_raises(tmp_path):
    path = _write_lerobot_checkpoint(tmp_path, stats=_FULL_DATASET_STATS)
    (tmp_path / "policy_preprocessor_step_3_normalizer_processor.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match="normalizer state"):
        load_lerobot_norm_stats(path)


def test_pipeline_rejects_missing_model_weights_before_initialization(tmp_path):
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    pipeline = object.__new__(Pi05Pipeline)
    pipeline.model_dir = str(tmp_path)

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        pipeline._initialize_model()


def test_build_config_merges_the_deploy_yaml_over_the_checkpoint(tmp_path):
    """The deploy yaml carries the runtime knobs and the checkpoint carries the
    feature schema and the stats. Both have to land on one config."""
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    pipeline = object.__new__(Pi05Pipeline)
    pipeline.model_dir = _write_lerobot_checkpoint(tmp_path, stats=_FULL_DATASET_STATS)

    config = pipeline._build_config(SimpleNamespace(model_config={"type": "pi05", "num_inference_steps": 4}))

    assert config.num_inference_steps == 4
    assert (config.state_dim, config.action_dim) == (2, 2)
    assert config.image_feature_keys == _EXPECTED_CAMERA_ORDER
    assert config.norm_stats["state"]["mode"] == "quantile"


class _WideActionModel:
    """Stands in for the 3.6B model, which is the one collaborator here that is
    genuinely expensive to build. It returns the padded ``max_action_dim`` width
    the real model emits, so the assertion is about the pipeline's cropping."""

    def __init__(self, chunk_size: int, max_action_dim: int):
        self._shape = (1, chunk_size, max_action_dim)

    def sample_actions(self, **kwargs):
        del kwargs
        return torch.ones(self._shape)


def test_pipeline_crops_actions_to_checkpoint_output_schema(monkeypatch):
    from vllm_omni.diffusion.models.pi05 import pipeline_pi05
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    config = Pi05Config.from_model_config(
        dict(_LEROBOT_CFG, output_features={"action": {"type": "ACTION", "shape": [7]}})
    )
    pipeline = object.__new__(pipeline_pi05.Pi05Pipeline)
    pipeline.config = config
    pipeline.tokenizer = object()
    pipeline._device = torch.device("cpu")
    pipeline.processor = Pi05Processor(config, object(), torch.device("cpu"))
    pipeline.model = _WideActionModel(config.chunk_size, config.max_action_dim)
    monkeypatch.setattr(
        pipeline.processor,
        "build_model_inputs",
        lambda robot_obs: ([torch.empty(0)], [torch.empty(0)], torch.empty(0), torch.empty(0)),
    )
    request = OmniDiffusionRequest(
        prompt="",
        sampling_params=OmniDiffusionSamplingParams(
            extra_args={"robot_obs": {"state": np.zeros(config.max_state_dim)}},
        ),
        request_id="crop-test",
    )

    result = pipeline.forward(request)

    assert result.output["actions"].shape == (config.chunk_size, 7)


class _GraphTarget:
    """Stands in for the model: all ``_install_cuda_graphs`` touches are the
    graph slot and the fused-kernel switch, whose calls ``events`` records."""

    cuda_graphs = None

    def __init__(self, events: list, fused_kernels_error: Exception | None = None):
        self.events = events
        self.fused_kernels_error = fused_kernels_error

    def enable_fused_kernels(self):
        if self.fused_kernels_error is not None:
            raise self.fused_kernels_error
        self.events.append("enable_fused_kernels")


def _graph_install_pipeline(device: str, events: list, monkeypatch, fused_kernels_error=None):
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    # Capture needs a GPU; test_pi05_cuda_graph_parity.py runs the real one.
    monkeypatch.setattr(Pi05CUDAGraphs, "capture", lambda graphs: events.append("capture"))
    pipeline = object.__new__(Pi05Pipeline)
    pipeline._device = torch.device(device)
    pipeline.model = _GraphTarget(events, fused_kernels_error)
    return pipeline


@pytest.mark.parametrize(
    "enforce_eager,device,installed",
    [
        (True, "cuda", False),
        (False, "cuda", True),
        # No CUDA device: the knob asks for graphs, the pipeline stays eager.
        (False, "cpu", False),
        (True, "cpu", False),
    ],
)
def test_pipeline_installs_cuda_graphs_unless_enforce_eager(enforce_eager, device, installed, monkeypatch):
    """The optimized path is the fused kernels with CUDA graphs captured over
    them: the kernels are enabled first, then the graphs are captured at init
    and installed. The eager baseline gets neither."""
    from types import SimpleNamespace

    events: list[str] = []
    pipeline = _graph_install_pipeline(device, events, monkeypatch)

    pipeline._install_cuda_graphs(SimpleNamespace(enforce_eager=enforce_eager))

    assert isinstance(pipeline.model.cuda_graphs, Pi05CUDAGraphs) is installed
    assert events == (["enable_fused_kernels", "capture"] if installed else [])


def test_pipeline_fails_when_the_fused_kernels_cannot_run(monkeypatch):
    """``enforce_eager=False`` asks for the optimized path: kernels that cannot
    run must stop startup rather than silently serve eagerly."""
    from types import SimpleNamespace

    events: list[str] = []
    pipeline = _graph_install_pipeline("cuda", events, monkeypatch, fused_kernels_error=RuntimeError("no Triton"))

    with pytest.raises(RuntimeError, match="enforce_eager") as excinfo:
        pipeline._install_cuda_graphs(SimpleNamespace(enforce_eager=False))
    assert "no Triton" in str(excinfo.value.__cause__)
    assert events == []
    assert pipeline.model.cuda_graphs is None


def test_load_lerobot_norm_stats_unknown_mode_raises(tmp_path):
    """Fail loud: an unreproducible mode served anyway is silently wrong."""
    path = _write_lerobot_checkpoint(
        tmp_path, norm_map={"STATE": "SOMETHING_NEW", "ACTION": "QUANTILES"}, stats=_FULL_DATASET_STATS
    )
    with pytest.raises(ValueError, match="SOMETHING_NEW"):
        load_lerobot_norm_stats(path)


def test_from_pretrained_backfills_sidecar_stats(tmp_path):
    """End to end: the stats reach norm_stats without any deploy yaml."""
    path = _write_lerobot_checkpoint(tmp_path, stats=_FULL_DATASET_STATS)
    config = Pi05Config.from_pretrained(path)

    assert config.norm_stats["state"]["mode"] == "quantile"
    assert config.norm_stats["action"]["mode"] == "quantile"


def test_from_pretrained_does_not_override_explicit_norm_stats(tmp_path):
    """config.json wins over the sidecar, matching LeRobot's override rule."""
    import json as _json

    path = _write_lerobot_checkpoint(tmp_path, stats=_FULL_DATASET_STATS)
    explicit = {"state": {"mode": "min_max", "min": [0.0] * 32, "max": [1.0] * 32}}
    (tmp_path / "config.json").write_text(_json.dumps({**_LEROBOT_CFG, "norm_stats": explicit}), encoding="utf-8")

    assert Pi05Config.from_pretrained(path).norm_stats == explicit


# ----------------------------------------------------------------------------
# Checkpoint boundary: declared-but-unconsumed capabilities must raise
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "key,value",
    [
        ("use_visual_memory", True),
        ("use_proprioceptive_memory", True),
        ("rtc_config", {"mode": "trained", "inference_delay": 2}),
        ("n_obs_steps", 6),
    ],
)
def test_config_rejects_unsupported_capability(key, value):
    """MEM and RTC change what a correct action chunk looks like and are not
    visible in the weights. Serving such a checkpoint must fail loudly."""
    with pytest.raises(UnsupportedCheckpointCapabilityError, match=key):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, **{key: value}))


# ----------------------------------------------------------------------------
# Relative actions
# ----------------------------------------------------------------------------
def test_relative_actions_without_action_names_raises():
    """``action_feature_names`` is populated from dataset metadata at training
    time. There is no dataset at serving time, so a relative-action checkpoint
    that omits it cannot resolve ``relative_exclude_joints`` and must fail."""
    with pytest.raises(UnsupportedCheckpointCapabilityError, match="action_feature_names"):
        Pi05Config.from_model_config(dict(_LEROBOT_CFG, use_relative_actions=True, action_feature_names=None))


def test_relative_actions_empty_exclude_list_needs_no_action_names():
    c = Pi05Config.from_model_config(dict(_LEROBOT_CFG, use_relative_actions=True, relative_exclude_joints=[]))
    assert c.use_relative_actions


def test_resolve_excluded_action_indices_unresolvable_raises():
    with pytest.raises(UnsupportedCheckpointCapabilityError):
        resolve_excluded_action_indices(["no_such_joint"], _ACTION_NAMES)


def _relative_step(enabled=True):
    return Pi05RelativeActions(
        enabled=enabled,
        exclude_joints=["gripper"],
        action_names=_ACTION_NAMES,
        max_action_dim=32,
    )


def test_relative_mask_excludes_gripper_and_padding():
    rel = _relative_step()
    assert rel.relative_mask[0]
    assert not rel.relative_mask[6], "gripper must stay absolute"
    assert not rel.relative_mask[len(_ACTION_NAMES) :].any(), "padded dims carry no signal"
    assert rel.num_relative_dims == 6


def test_relative_absolute_round_trip_is_exact():
    """The two directions are one object precisely so they cannot disagree."""
    rel = _relative_step()
    g = torch.Generator().manual_seed(0)
    actions = torch.randn(1, 50, 32, generator=g)
    state = torch.randn(32, generator=g)
    assert torch.allclose(rel.to_absolute(rel.to_relative(actions, state), state), actions, atol=1e-6)


def test_relative_rejects_mismatched_action_dim():
    rel = _relative_step()
    with pytest.raises(ValueError, match="max_action_dim"):
        rel.to_absolute(torch.zeros(1, 4, 8), torch.zeros(32))


# ----------------------------------------------------------------------------
# State discretization — π0.5's defining input path
# ----------------------------------------------------------------------------
def test_discretize_underflows_to_minus_one_and_saturates_at_the_top():
    """Out-of-range input is asymmetric, and that asymmetry is LeRobot's.

    ``np.digitize`` returns 0 for anything below the first bin edge, so the
    ``- 1`` makes a below-range state land in bin ``-1``; above-range input
    saturates at ``num_bins - 1`` on its own. The checkpoint was trained with
    ``" -1"`` in the state prompt for under-range dims, so clipping that to 0
    would change the tokens the model sees — LeRobot parity catches it.
    """
    out = discretize_state(np.array([-9.0, 9.0], dtype=np.float32), num_bins=256)
    assert out.tolist() == [-1, 255]


def test_normalize_state_does_not_clip_out_of_range():
    """The parity-critical contract: normalization is affine and nothing else.

    Clamping to [-1, 1] here would hide an under-range dimension from
    ``discretize_state``, turning LeRobot's ``-1`` bin into ``0`` and changing
    the state tokens in the prompt. LeRobot applies
    ``2.0 * (x - q01) / (q99 - q01) - 1.0`` with no clip.
    """
    stats = {"q01": [0.0] * 32, "q99": [10.0] * 32}
    out = _normalized([-10.0, 20.0] + [5.0] * 30, stats)
    assert out[0] == pytest.approx(-3.0)  # would be -1.0 if clipped
    assert out[1] == pytest.approx(3.0)  # would be 1.0 if clipped
    # and the under-range dim must reach the -1 bin end to end
    assert discretize_state(out, num_bins=256)[0] == -1


def test_prompt_serializes_the_declared_state_width():
    """LeRobot's tokenizer step discretizes the state at its real width and
    never pads to ``max_state_dim``."""
    prompt = build_pi05_prompt(task="x", normalized_state=np.zeros(7, dtype=np.float32))
    bins = prompt.split("State: ")[1].split(";")[0].split()
    assert len(bins) == 7


def test_prompt_template_matches_lerobot():
    prompt = build_pi05_prompt(
        task="  pick_up the red\nblock ",
        normalized_state=np.zeros(4, dtype=np.float32),
    )
    assert prompt.startswith("Task: pick up the red block, State: ")
    assert prompt.endswith(";\nAction: ")


# ----------------------------------------------------------------------------
# Input contract: fixed camera slots x 256 tokens + constant 200 text tokens
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("views,expected", [(1, 456), (2, 712), (3, 968)])
def test_prefix_length_is_fixed_by_max_cameras(views, expected):
    """``max_cameras × 256`` image tokens plus a constant ``tokenizer_max_length``.

    The text segment is padded to its maximum and the camera slots are padded to
    ``max_cameras``, so the prefix shape does not vary with the request — only
    the attention mask does.
    """
    keys = _EXPECTED_CAMERA_ORDER[:views]
    config = Pi05Config(max_cameras=views, tokenizer_max_length=200, image_feature_keys=keys)
    observation = {
        "state": np.zeros(32, dtype=np.float32),
        "images": {key: np.zeros((4, 4, 3), dtype=np.uint8) for key in keys},
    }

    images, masks, lang_tokens, _ = _assemble_model_inputs(
        observation, config, _FakeTokenizer(), torch.device("cpu"), None
    )

    assert len(images) == views
    assert all(bool(mask) for mask in masks)
    assert lang_tokens.shape[-1] == 200
    assert 256 * len(images) + lang_tokens.shape[-1] == expected


# ----------------------------------------------------------------------------
# Shared math helpers (same contracts as π0)
# ----------------------------------------------------------------------------
def test_gemma_variant_dims():
    vlm = get_gemma_config("gemma_2b")
    assert (vlm.width, vlm.depth, vlm.mlp_dim, vlm.num_kv_heads, vlm.head_dim) == (2048, 18, 16384, 1, 256)
    expert = get_gemma_config("gemma_300m")
    assert (expert.width, expert.depth, expert.mlp_dim) == (1024, 18, 4096)


def test_sinusoidal_embedding_shape_and_oddity():
    out = create_sinusoidal_pos_embedding(torch.tensor([0.0, 1.0]), 64)
    assert out.shape == (2, 64)
    with pytest.raises(ValueError):
        create_sinusoidal_pos_embedding(torch.tensor([0.0]), 63)


def test_make_att_2d_masks_respects_padding():
    pad = torch.tensor([[True, True, False]])
    att = torch.tensor([[0, 0, 0]])
    out = make_att_2d_masks(pad, att)
    assert not out[0, :, 2].any()


def test_build_norm_stats_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported normalization mode"):
        build_norm_stats({"action": {"mode": "qunatile"}}, "action")


@pytest.mark.parametrize(
    "entry",
    [
        {"mode": "mean_std", "mean": [0.0]},
        {"mode": "min_max", "min": [0.0]},
        {"mode": "quantile", "q01": [0.0]},
    ],
)
def test_build_norm_stats_rejects_incomplete_stats(entry):
    with pytest.raises(ValueError, match="requires"):
        build_norm_stats({"action": entry}, "action")


def test_normalize_quantile_round_trip_and_padded_tail():
    stats = build_norm_stats({"action": {"q01": [0.0, 0.0], "q99": [2.0, 4.0]}}, "action")
    x = torch.tensor([[0.5, -0.5, 7.0]])  # third entry is padding beyond the stats
    back = apply_norm(apply_norm(x, stats), stats, inverse=True)
    assert torch.allclose(back, x, atol=1e-5)
    assert back[0, 2] == 7.0, "padded tail must pass through untouched"


# ----------------------------------------------------------------------------
# Image processor
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("value,expected", [(0.0, -1.0), (0.25, -0.5), (1.0, 1.0)])
@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_float_image_uses_lerobot_zero_one_domain(value, expected, container):
    if container == "numpy":
        image = np.full((4, 4, 3), value, dtype=np.float32)
    else:
        image = torch.full((3, 4, 4), value, dtype=torch.float32)

    out = Pi05ImageProcessor(image_size=4).preprocess_single(image)

    assert out.shape == (1, 3, 4, 4)
    assert torch.allclose(out, torch.full_like(out, expected))


@pytest.mark.parametrize("bad", [-0.01, 1.01, np.nan, np.inf])
@pytest.mark.parametrize("container", ["numpy", "torch"])
def test_float_image_rejects_values_outside_lerobot_domain(bad, container):
    if container == "numpy":
        image = np.full((4, 4, 3), bad, dtype=np.float32)
    else:
        image = torch.full((3, 4, 4), bad, dtype=torch.float32)

    with pytest.raises(ValueError):
        Pi05ImageProcessor(image_size=4).preprocess_single(image)


def test_resize_with_pad_pads_with_minus_one():
    out = resize_with_pad(torch.zeros(1, 3, 100, 200), 224, 224)
    assert out.shape == (1, 3, 224, 224)
    assert out[0, 0, 0, 0] == -1.0


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        del text
        length = kwargs["max_length"]
        return {"input_ids": [0] * length, "attention_mask": [1] * length}


@pytest.mark.parametrize("explicit_none", [False, True])
def test_build_model_inputs_preserves_missing_middle_camera_slot(explicit_none):
    config = Pi05Config(image_feature_keys=_EXPECTED_CAMERA_ORDER, max_cameras=3)
    images = {
        _EXPECTED_CAMERA_ORDER[0]: np.zeros((4, 4, 3), dtype=np.uint8),
        _EXPECTED_CAMERA_ORDER[2]: np.full((4, 4, 3), 255, dtype=np.uint8),
    }
    if explicit_none:
        images[_EXPECTED_CAMERA_ORDER[1]] = None
    observation = {
        "images": images,
        "state": np.zeros(32, dtype=np.float32),
    }

    slots, masks, _, _ = _assemble_model_inputs(observation, config, _FakeTokenizer(), torch.device("cpu"), None)

    assert [bool(mask.item()) for mask in masks] == [True, False, True]
    torch.testing.assert_close(slots[0], torch.full_like(slots[0], -1.0))
    torch.testing.assert_close(slots[1], torch.full_like(slots[1], -1.0))
    torch.testing.assert_close(slots[2], torch.full_like(slots[2], 1.0))


def test_build_model_inputs_requires_configured_camera():
    config = Pi05Config(image_feature_keys=_EXPECTED_CAMERA_ORDER)
    observation = {
        "unknown_camera": np.zeros((4, 4, 3), dtype=np.uint8),
        "state": np.zeros(32, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="configured camera"):
        _assemble_model_inputs(observation, config, _FakeTokenizer(), torch.device("cpu"), None)


# ----------------------------------------------------------------------------
# Model behaviour + weight-load remap (tiny action dims; full Gemma backbone)
# ----------------------------------------------------------------------------
# Module *structure* (no ``state_proj``, ``time_mlp_{in,out}`` widths, AdaRMS on
# the expert norms but not the prefix) is not asserted here: ``load_weights``
# audits every model parameter against the checkpoint and raises on a missing,
# unmatched or π0-shaped key, so a structural drift fails loudly at load time
# rather than silently. ``test_pi0_shaped_keys_are_not_silently_loaded`` guards
# that audit, and ``test_pi05_parity.py`` is the numerical oracle.
# Both Gemma towers keep their real widths but drop from 18 layers to 2 for
# every test in this section. Nothing here depends on depth: the weight remap
# matches by name, and the AR mask, prefix length and ODE determinism are
# shape-driven. Full-scale numerical correctness is ``test_pi05_parity.py``'s
# job, against the real checkpoint.
#
# Two dimensions must NOT be shrunk, both because something outside these
# configs is sized against them:
#
# * ``width`` — the vision tower's ``multi_modal_projector`` projects image
#   embeddings to the PaliGemma text width so the two can be concatenated.
#   Narrowing the text tower alone makes that concat fail.
# * the vision tower itself — So400m/14 on 224x224 is what yields exactly 256
#   tokens per view, which ``test_prefix_length_and_valid_len_for_each_view_count``
#   exists to verify.
#
# Depths are equal across the two towers on purpose: the action expert runs
# layer-by-layer alongside the VLM, so a mismatch would not be a valid model.
_SHRUNK_GEMMA = {
    "gemma_2b": GemmaVariantConfig(2048, 2, 1024, 8, 1, 256),
    "gemma_300m": GemmaVariantConfig(1024, 2, 512, 8, 1, 256),
}


@pytest.fixture(scope="module", autouse=True)
def _shrink_gemma_backbones():
    """Patch the module-level lookup ``Pi05ForActionPrediction.__init__`` calls."""
    patch = pytest.MonkeyPatch()
    patch.setattr(modeling_pi05, "get_gemma_config", lambda variant: _SHRUNK_GEMMA[variant])
    yield
    patch.undo()


def _tiny_pi05_model():
    """A fresh model. Use for tests that load weights or otherwise mutate it."""
    return Pi05ForActionPrediction(Pi05Config(max_action_dim=8, max_state_dim=8, chunk_size=4, n_action_steps=4))


@pytest.fixture(scope="module")
def tiny_model():
    """Shared read-only model, so the tests that only inspect it pay for one
    construction between them rather than one each."""
    return _tiny_pi05_model()


@pytest.fixture(autouse=True)
def _silence_pi05_loader():
    logging.getLogger("vllm_omni.diffusion.models.pi05.modeling_pi05").setLevel(logging.CRITICAL)
    yield


@pytest.mark.slow
def test_suffix_has_no_state_token(tiny_model):
    """π0's suffix is ``[state, actions...]`` with AR mask ``[1, 1, 0...]``;
    π0.5's is actions only with ``[1, 0...]``."""
    embs, pad, att, cond = tiny_model.embed_suffix(torch.randn(1, 4, 8), torch.tensor([1.0]))
    assert embs.shape[1] == 4
    assert att[0].tolist() == [1, 0, 0, 0]
    assert cond.shape == (1, tiny_model.expert_width)


# Not marked slow: constructs one small norm layer, no backbone.
def test_adarms_norm_returns_gate_only_when_conditioned():
    norm = Pi05AdaRMSNorm(8, cond_dim=4)
    out, gate = norm(torch.randn(1, 3, 8), torch.randn(1, 4))
    assert gate is not None and gate.shape == (1, 1, 8)

    plain = Pi05AdaRMSNorm(8, cond_dim=None)
    out, gate = plain(torch.randn(1, 3, 8))
    assert gate is None


@pytest.mark.slow
def test_lm_head_remap_to_embed_tokens():
    model = _tiny_pi05_model()
    target = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    payload = torch.full(dict(model.named_parameters())[target].shape, 0.5)
    model.load_weights([("model.paligemma_with_expert.paligemma.lm_head.weight", payload)], strict=False)
    assert torch.allclose(dict(model.named_parameters())[target], payload)


@pytest.mark.slow
def test_pi0_shaped_keys_are_not_silently_loaded():
    """A π0 checkpoint pointed at the π0.5 class would leave the AdaRMS expert
    randomly initialized. ``state_proj`` is the tell."""
    model = _tiny_pi05_model()
    with pytest.raises(RuntimeError, match="π0-shaped"):
        model.load_weights([("model.state_proj.weight", torch.zeros(1024, 8))])


@pytest.mark.slow
@pytest.mark.parametrize("num_real_views", [1, 2, 3])
def test_prefix_length_is_fixed_and_valid_len_tracks_masks(num_real_views, tiny_model):
    """The deployed model has three slots; only their validity varies."""
    live_text = 120
    images = [torch.zeros(1, 3, 224, 224) for _ in range(3)]
    masks = [torch.tensor([index < num_real_views]) for index in range(3)]
    lang = torch.zeros(1, 200, dtype=torch.long)
    lang_mask = torch.zeros(1, 200, dtype=torch.bool)
    lang_mask[:, :live_text] = True

    embs, pad_masks, _ = tiny_model.embed_prefix(images, masks, lang, lang_mask)
    assert embs.shape[1] == 256 * 3 + 200
    assert int(pad_masks.sum()) == 256 * num_real_views + live_text


@pytest.mark.slow
def test_sample_actions_shape_and_determinism(tiny_model):
    """Flow matching is an ODE: fixed noise must give a bit-identical chunk."""
    model = tiny_model.eval()
    images = [torch.zeros(1, 3, 224, 224) for _ in range(3)]
    masks = [torch.tensor([True]), torch.tensor([False]), torch.tensor([False])]
    lang = torch.zeros(1, 200, dtype=torch.long)
    lang_mask = torch.ones(1, 200, dtype=torch.bool)
    noise = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(42))

    with torch.no_grad():
        a1 = model.sample_actions(
            images=images, image_masks=masks, lang_tokens=lang, lang_masks=lang_mask, noise=noise, num_steps=2
        )
        a2 = model.sample_actions(
            images=images, image_masks=masks, lang_tokens=lang, lang_masks=lang_mask, noise=noise, num_steps=2
        )
    assert a1.shape == (1, 4, 8)
    assert torch.isfinite(a1).all()
    assert torch.equal(a1, a2)


@pytest.mark.slow
def test_sample_actions_uses_request_generator(tiny_model):
    model = tiny_model.eval()
    images = [torch.zeros(1, 3, 224, 224) for _ in range(3)]
    masks = [torch.tensor([True]), torch.tensor([False]), torch.tensor([False])]
    lang = torch.zeros(1, 200, dtype=torch.long)
    lang_mask = torch.ones(1, 200, dtype=torch.bool)

    def run(seed):
        return model.sample_actions(
            images=images,
            image_masks=masks,
            lang_tokens=lang,
            lang_masks=lang_mask,
            num_steps=2,
            generator=torch.Generator().manual_seed(seed),
        )

    with torch.no_grad():
        first = run(42)
        repeated = run(42)
        different = run(43)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


class _RecordingCUDAGraphs(Pi05CUDAGraphs):
    """Records which ``sample_actions`` region runs through the graph path."""

    def __init__(self, model):
        super().__init__(model)
        self.calls: list[str] = []

    def embed_prefix(self, *args, **kwargs):
        self.calls.append("embed_prefix")
        return super().embed_prefix(*args, **kwargs)

    def prefix_forward(self, *args, **kwargs):
        self.calls.append("prefix_forward")
        return super().prefix_forward(*args, **kwargs)

    def denoise_step(self, *args, **kwargs):
        self.calls.append("denoise_step")
        return super().denoise_step(*args, **kwargs)


@pytest.mark.slow
def test_sample_actions_routes_each_region_through_cuda_graphs(tiny_model):
    """Once installed, every region goes through ``cuda_graphs`` (the denoise
    step once per step), and falling back to eager reproduces the baseline bit
    for bit. The real-checkpoint check is ``test_pi05_cuda_graph_parity.py``."""
    model = tiny_model.eval()
    images = [torch.zeros(1, 3, 224, 224) for _ in range(3)]
    masks = [torch.tensor([True]), torch.tensor([False]), torch.tensor([False])]
    lang = torch.zeros(1, 200, dtype=torch.long)
    lang_mask = torch.ones(1, 200, dtype=torch.bool)
    noise = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(42))

    def run():
        return model.sample_actions(
            images=images, image_masks=masks, lang_tokens=lang, lang_masks=lang_mask, noise=noise, num_steps=3
        )

    graphs = _RecordingCUDAGraphs(model)
    with torch.no_grad():
        eager = run()
        model.cuda_graphs = graphs
        try:
            optimized = run()
        finally:
            model.cuda_graphs = None

    assert graphs.calls == ["embed_prefix", "prefix_forward"] + ["denoise_step"] * 3
    assert torch.equal(optimized, eager)


@pytest.mark.slow
def test_cuda_graph_capture_failure_raises(tiny_model):
    """``enforce_eager=False`` asks for the graph path: a failed capture must
    stop startup rather than silently serve eagerly."""
    assert tiny_model.kv_cache is None
    with pytest.raises(RuntimeError, match="enforce_eager"):
        Pi05CUDAGraphs(tiny_model).capture()


@pytest.mark.slow
def test_preallocated_kv_cache_is_rewritten_every_call():
    """``sample_actions`` writes every K/V slot it reads before reading it, so
    the preallocated cache reproduces a one-off cache bit for bit whatever an
    earlier call left in it."""
    model = _tiny_pi05_model().eval()
    # AdaRMS ``dense`` is zero-initialized, which closes every expert gate and
    # leaves the chunk independent of the K/V. Open the gates.
    generator = torch.Generator().manual_seed(0)
    for module in model.modules():
        if isinstance(module, Pi05AdaRMSNorm) and module.dense is not None:
            module.dense.weight.data.normal_(std=0.02, generator=generator)
    images = [torch.zeros(1, 3, 224, 224) for _ in range(3)]
    masks = [torch.tensor([True]), torch.tensor([False]), torch.tensor([False])]
    lang_mask = torch.ones(1, 200, dtype=torch.bool)
    noise = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(42))

    def run(lang_tokens):
        return model.sample_actions(
            images=images, image_masks=masks, lang_tokens=lang_tokens, lang_masks=lang_mask, noise=noise, num_steps=2
        )

    prompt = torch.zeros(1, 200, dtype=torch.long)
    other_prompt = torch.ones(1, 200, dtype=torch.long)
    with torch.no_grad():
        one_off = run(prompt)
        cache = model.new_kv_cache(batch_size=1)
        assert cache.fits(1, 3 * 256 + 200)
        cache.key.fill_(float("nan"))
        cache.value.fill_(float("nan"))
        model.kv_cache = cache
        preallocated = run(prompt)
        other = run(other_prompt)
        after_other = run(prompt)

    assert torch.isfinite(cache.key).all() and torch.isfinite(cache.value).all()
    assert torch.equal(preallocated, one_off)
    assert not torch.equal(other, one_off)
    assert torch.equal(after_other, one_off)


@pytest.mark.slow
def test_forward_requires_a_kv_cache(tiny_model):
    backbone = tiny_model.paligemma_with_expert
    with pytest.raises(TypeError, match="Pi05KVCache"):
        backbone.forward(inputs_embeds=[torch.zeros(1, 4, tiny_model.vlm_width), None], use_cache=True)
    with pytest.raises(TypeError, match="Pi05KVCache"):
        backbone.forward(inputs_embeds=[None, torch.zeros(1, 4, tiny_model.expert_width)], past_key_values=[])


@pytest.mark.slow
def test_fused_kernels_are_off_until_enabled_and_need_cuda():
    """The eager baseline never runs the fused kernels; enabling them on a
    model they cannot serve raises and leaves it on the baseline. The kernels
    themselves are ``test_pi05_fused_kernels.py``'s (CUDA)."""
    model = _tiny_pi05_model()
    assert model.paligemma_with_expert.fused_kernels is False
    with pytest.raises(RuntimeError, match="not a CUDA device"):
        model.enable_fused_kernels()
    assert model.paligemma_with_expert.fused_kernels is False


@pytest.mark.slow
def test_fused_kernels_serve_only_batch_one_on_a_single_head_kv_cache(tiny_model):
    """Calls the kernels do not cover run eagerly even once they are enabled."""
    backbone = tiny_model.paligemma_with_expert
    serve = modeling_pi05._fused_kernels_serve
    cache = tiny_model.new_kv_cache(batch_size=1)
    backbone.fused_kernels = True
    try:
        assert serve(backbone, cache, batch_size=1)
        assert not serve(backbone, cache, batch_size=2)
        assert not serve(backbone, tiny_model.new_kv_cache(batch_size=2), batch_size=2)
        assert not serve(backbone, [], batch_size=1)
        backbone.fused_kernels = False
        assert not serve(backbone, cache, batch_size=1)
    finally:
        backbone.fused_kernels = False


# ----------------------------------------------------------------------------
# Serving dtype
# ----------------------------------------------------------------------------
# The check that decides what runs lives on the pipeline, not on Pi05Config:
# Pi05Config.dtype records what the checkpoint declares and never reaches a
# cast, while _initialize_model casts using the top-level OmniDiffusionConfig.
@pytest.mark.parametrize(
    "declared,expected",
    [
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_resolve_dtype_accepts_supported_serving_dtypes(declared, expected):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    assert Pi05Pipeline._resolve_dtype(OmniDiffusionConfig(dtype=declared)) is expected


@pytest.mark.parametrize("declared", ["float16", "fp16", torch.float16])
def test_resolve_dtype_rejects_unvalidated_dtypes(declared):
    """float16 resolves to a real torch dtype, so a permissive lookup would run
    the model in a precision nothing here has been checked in.

    Only dtypes that survive ``OmniDiffusionConfig`` normalization are worth
    asserting on: it maps a string it does not recognize onto bfloat16 with a
    warning, so e.g. ``"float64"`` never reaches this guard as float64.
    """
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    with pytest.raises(ValueError, match="Unsupported π0.5 dtype"):
        Pi05Pipeline._resolve_dtype(OmniDiffusionConfig(dtype=declared))


def test_bfloat16_layout_is_applied_before_checkpoint_load(monkeypatch, tmp_path):
    """Checkpoint tensors must be copied directly into their final dtypes.

    LeRobot casts only the inner PaliGemma/expert module; the outer action and
    time projections remain float32. The layout is established before loading
    so float32-retained weights never pass through bfloat16.
    """
    from vllm_omni.diffusion.models.pi05 import pipeline_pi05

    class _TinyPi05(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.paligemma_with_expert = torch.nn.Module()
            self.paligemma_with_expert.backbone = torch.nn.Linear(2, 2)
            self.paligemma_with_expert.vision_tower = torch.nn.Linear(2, 2)
            self.paligemma_with_expert.input_layernorm = torch.nn.LayerNorm(2)
            self.paligemma_with_expert.register_buffer("floating_buffer", torch.ones(1))
            self.action_in_proj = torch.nn.Linear(2, 2)
            self.action_out_proj = torch.nn.Linear(2, 2)
            self.time_mlp_in = torch.nn.Linear(2, 2)
            self.time_mlp_out = torch.nn.Linear(2, 2)

        def configure_vision(self, options):
            assert options.precision == "fp32" and not options.batch_cameras

        def configure_runtime(self, options):
            assert not any((options.precompute_adarms, options.hoist_denoise_invariants, options.prefix_kv_only))

        def prepare_runtime_caches(self):
            assert not self.training

    # DiffusersLoader constructs custom pipelines under the requested default
    # dtype. Start fully in bfloat16 to verify that the helper restores every
    # float32-retained tensor rather than relying on the construction default.
    model = _TinyPi05().to(dtype=torch.bfloat16)
    observed_dtypes = {}

    def fake_load_checkpoint(_self, model_to_load):
        observed_dtypes.update({name: param.dtype for name, param in model_to_load.named_parameters()})

    (tmp_path / "model.safetensors").touch()
    pipeline = object.__new__(pipeline_pi05.Pi05Pipeline)
    pipeline.model_dir = str(tmp_path)
    pipeline.config = object()
    pipeline.vision_options = pipeline_pi05.Pi05VisionOptions()
    pipeline.runtime_options = pipeline_pi05.Pi05RuntimeOptions()
    pipeline._device = torch.device("cpu")
    pipeline._torch_dtype = torch.bfloat16
    monkeypatch.setattr(pipeline_pi05, "Pi05ForActionPrediction", lambda _config: model)
    monkeypatch.setattr(pipeline_pi05.Pi05Pipeline, "_load_checkpoint", fake_load_checkpoint)

    initialized = pipeline._initialize_model()

    assert observed_dtypes["paligemma_with_expert.backbone.weight"] is torch.bfloat16
    assert model.paligemma_with_expert.floating_buffer.dtype is torch.bfloat16
    for name in (
        "paligemma_with_expert.vision_tower.weight",
        "paligemma_with_expert.input_layernorm.weight",
        "action_in_proj.weight",
        "action_out_proj.weight",
        "time_mlp_in.weight",
        "time_mlp_out.weight",
    ):
        assert observed_dtypes[name] is torch.float32
    assert initialized is model
    assert not initialized.training


@pytest.mark.slow
def test_bfloat16_runs_and_tracks_float32(tiny_model):
    """bfloat16 is a supported serving dtype, so it has to produce a usable
    chunk — and one that still resembles float32.

    The bound is deliberately loose. This is not a parity check (bfloat16
    deviates from float32 by ~1-2% of the action range on the real checkpoint,
    which is why recipes/lerobot/Pi05.md tells operators to validate it on their
    own task). What it catches is gross breakage: a cast that silently did not
    happen, activations overflowing to inf, or a dtype mismatch that would make
    the two outputs unrelated.
    """
    import copy

    images = [torch.zeros(1, 3, 224, 224) for _ in range(3)]
    masks = [torch.tensor([True]), torch.tensor([False]), torch.tensor([False])]
    lang = torch.zeros(1, 200, dtype=torch.long)
    lang_mask = torch.ones(1, 200, dtype=torch.bool)
    noise = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(7))

    def run(model, dtype):
        with torch.no_grad():
            return model.sample_actions(
                images=[img.to(dtype) for img in images],
                image_masks=masks,
                lang_tokens=lang,
                lang_masks=lang_mask,
                noise=noise.to(dtype),
                num_steps=2,
            ).float()

    reference = run(tiny_model.eval(), torch.float32)
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import _set_inference_dtype

    bf16_model = copy.deepcopy(tiny_model).eval()
    _set_inference_dtype(bf16_model, torch.bfloat16)

    # The cast actually happened — otherwise everything below passes trivially.
    assert any(param.dtype is torch.bfloat16 for param in bf16_model.parameters())
    assert bf16_model.action_in_proj.weight.dtype is torch.float32
    assert bf16_model.action_out_proj.weight.dtype is torch.float32
    assert bf16_model.time_mlp_in.weight.dtype is torch.float32
    assert bf16_model.time_mlp_out.weight.dtype is torch.float32

    actual = run(bf16_model, torch.bfloat16)
    assert actual.shape == reference.shape
    assert torch.isfinite(actual).all()

    scale = float(reference.abs().max())
    assert float((actual - reference).abs().max()) < max(0.5 * scale, 1e-2)


# ----------------------------------------------------------------------------
# State normalization runs π0's stats path
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "stats,state,expected",
    [
        # A denominator that is small but not zero. ``normalize`` divides by
        # ``std + eps``; replacing a small std with 1.0 would be a different map.
        ({"mode": "mean_std", "mean": [0.0], "std": [1e-7]}, [1e-7], 1e-7 / (1e-7 + 1e-8)),
        ({"mode": "quantile", "q01": [0.0], "q99": [1e-7]}, [1e-7], 1.0),
        ({"mode": "quantile", "q01": [0.0], "q99": [1e-9]}, [1e-9], 1.0),
        # An exactly zero range maps the bound to -1 instead of dividing by zero.
        ({"mode": "quantile", "q01": [2.0], "q99": [2.0]}, [2.0], -1.0),
    ],
)
def test_normalize_state_uses_the_shared_denominator_rule(stats, state, expected):
    """State normalization and action unnormalization are the same affine map,
    so a narrow range keeps producing LeRobot's answer."""
    assert _normalized(state, stats)[0] == pytest.approx(expected, rel=1e-5)


# ----------------------------------------------------------------------------
# State wire contract
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_state_rejects_non_finite_values(bad):
    """``np.digitize`` maps nan and inf to the top bin and -inf to -1, all legal
    state tokens, so a dead sensor would reach the model as an ordinary prompt."""
    with pytest.raises(ValueError, match="NaN or Inf"):
        as_state_vector([0.0, bad], 2)


def test_state_rejects_a_multi_row_matrix():
    """Flattening would read a (2, 16) request as a 32-value state."""
    with pytest.raises(ValueError, match=r"\(D,\) or \(1, D\)"):
        as_state_vector(np.zeros((2, 16), dtype=np.float32), 32)


# ----------------------------------------------------------------------------
# Per-request denoising steps reach the model
# ----------------------------------------------------------------------------
class _SpyModel:
    """Records the ``num_steps`` ``forward`` hands to ``sample_actions``."""

    def __init__(self):
        self.seen: list = []
        self.generators: list = []

    def sample_actions(self, *, images, image_masks, lang_tokens, lang_masks, num_steps, generator):
        del images, image_masks, lang_tokens, lang_masks
        self.seen.append(num_steps)
        self.generators.append(generator)
        return torch.zeros(1, 50, 32)


def _spy_pipeline(spy: _SpyModel):
    from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

    pipeline = object.__new__(Pi05Pipeline)
    pipeline.config = Pi05Config.from_model_config(_LEROBOT_CFG)
    pipeline.tokenizer = _FakeTokenizer()
    pipeline._device = torch.device("cpu")
    pipeline.model = spy
    pipeline.processor = Pi05Processor(pipeline.config, _FakeTokenizer(), torch.device("cpu"))
    return pipeline


def _spy_request(num_inference_steps, generator=None):
    params = type("_Params", (), {})()
    params.extra_args = {
        "robot_obs": {
            "state": np.zeros(32, dtype=np.float32),
            "images": {key: np.zeros((4, 4, 3), dtype=np.uint8) for key in _EXPECTED_CAMERA_ORDER},
        }
    }
    params.num_inference_steps = num_inference_steps
    params.generator = generator
    request = type("_Request", (), {})()
    request.sampling_params = params
    return request


@pytest.mark.parametrize("requested", [None, 2, 25])
def test_request_denoising_steps_reach_sample_actions(requested):
    """The OpenPI E2E cannot see this: a pipeline that ignored the override and
    ran the static default would still return a finite chunk of the right shape."""
    spy = _SpyModel()
    output = _spy_pipeline(spy).forward(_spy_request(requested))

    assert spy.seen == [requested]
    assert output.output["actions"].shape == (50, 32)


def test_request_generator_reaches_sample_actions():
    generator = torch.Generator().manual_seed(42)
    spy = _SpyModel()

    _spy_pipeline(spy).forward(_spy_request(2, generator))

    assert spy.generators == [generator]


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4"])
def test_request_denoising_steps_must_be_a_positive_integer(bad):
    spy = _SpyModel()
    with pytest.raises(ValueError, match="num_inference_steps must be a positive integer"):
        _spy_pipeline(spy).forward(_spy_request(bad))
    assert spy.seen == []


def _exact_norm_reference(x, weight, add=None, shift=None, eps=1e-6):
    residual = x if add is None else x + add
    value = residual.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    value = value * (1.0 + weight.float())
    if shift is not None:
        value = value + shift.float()
    return residual, value.to(residual.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 17), (15, 1024), (2, 16, 2048)])
@pytest.mark.parametrize("adaptive", [False, True])
def test_exact_norm_cpu_fallback(dtype, shape, adaptive):
    generator = torch.Generator().manual_seed(419)
    x = torch.randn(shape, generator=generator).to(dtype)
    add = torch.randn(shape, generator=generator).to(dtype)
    weight = torch.randn(shape[-1], generator=generator) * 0.2
    shift = torch.randn(shape[-1], generator=generator) * 0.2 if adaptive else None
    assert not kernels.can_use_exact_rms_norm(x, weight, add=add, shift=shift)
    actual = kernels.residual_rms_norm(x, weight, add=add, shift=shift)
    expected = _exact_norm_reference(x, weight, add, shift)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_exact_norm_fallback_preserves_autograd_and_input_ownership():
    x = torch.randn(16, 17, requires_grad=True)
    weight = torch.randn(17, requires_grad=True)
    residual, actual = kernels.residual_rms_norm(x, weight)
    _, expected = _exact_norm_reference(x, weight)
    assert residual is x
    assert torch.equal(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (x, weight), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (x, weight))
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))


@pytest.mark.parametrize("version", ["2.11.0+cu128", "2.13.1", "2.14.0.dev20261012", "3.0.0"])
def test_exact_norm_unknown_torch_release_uses_fallback(monkeypatch, version):
    monkeypatch.setattr(torch, "__version__", version)
    assert not kernels._supported_torch_reduction()


def test_exact_norm_amd_uses_fallback(monkeypatch):
    monkeypatch.setattr(torch, "__version__", "2.13.0+rocm7.0")
    monkeypatch.setattr(torch.version, "hip", "7.0")
    assert not kernels._supported_torch_reduction()


def test_exact_norm_batched_modulation_fallback():
    x = torch.randn(2, 16, 1024).to(torch.bfloat16)
    scale = torch.randn(2, 1, 1024)
    shift = torch.randn_like(scale)
    actual = kernels.residual_rms_norm(x, scale, shift=shift)
    expected = _exact_norm_reference(x, scale, shift=shift)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["none", "broadcast", "strided"])
@pytest.mark.parametrize("output_dtype", [None, torch.float32])
def test_exact_softmax_cpu_fallback(dtype, mask_kind, output_dtype):
    generator = torch.Generator().manual_seed(193)
    scores = torch.randn(2, 3, 5, 17, generator=generator).to(dtype)
    mask = None if mask_kind == "none" else torch.randn(2, 1, 5, 17, generator=generator)
    if mask_kind == "strided":
        mask = torch.stack((mask, mask), dim=-1)[..., 0]
        assert not mask.is_contiguous()
    assert not kernels.can_use_exact_scaled_masked_softmax(scores, mask, 0.3, output_dtype=output_dtype)
    expected = scores * 0.3
    if mask is not None:
        expected = expected + mask
    expected = torch.softmax(expected, dim=-1, dtype=torch.float32).to(output_dtype or dtype)
    actual = kernels.scaled_masked_softmax(scores, mask, 0.3, output_dtype=output_dtype)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_softmax_fallback_preserves_autograd(dtype):
    scores = torch.randn(1, 2, 3, 11).to(dtype).requires_grad_()
    mask = torch.randn(1, 1, 3, 11, requires_grad=True)
    actual = kernels.scaled_masked_softmax(scores, mask, 1.0 / 16)
    expected = torch.softmax(scores * (1.0 / 16) + mask, dim=-1, dtype=torch.float32).to(dtype)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (scores, mask), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (scores, mask), grad)
    assert torch.equal(actual, expected)
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_softmax_attend_opt_in_keeps_default_path(monkeypatch, dtype):
    query = torch.randn(1, 2, 3, 8).to(dtype)
    key = torch.randn(1, 1, 7, 8).to(dtype)
    value = torch.randn_like(key)
    # An oversized mask must still be sliced at the KV boundary.
    mask = torch.randn(1, 1, 3, 9)
    calls = []
    original = modeling_pi05.scaled_masked_softmax

    def record(scores, sliced_mask, scale, **kwargs):
        calls.append((scores.shape, sliced_mask.shape, kwargs))
        return original(scores, sliced_mask, scale, **kwargs)

    monkeypatch.setattr(modeling_pi05, "scaled_masked_softmax", record)
    expected = modeling_pi05._attend(query, key, value, mask, 2, 1.0 / 16)
    assert not calls
    actual = modeling_pi05._attend(query, key, value, mask, 2, 1.0 / 16, exact_bf16_softmax=True)
    assert calls == [((1, 2, 3, 7), (1, 1, 3, 7), {"output_dtype": dtype})]
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("candidate_available", [False, True])
def test_exact_qk_attention_dispatch_preserves_pv_and_default(monkeypatch, candidate_available):
    """Observe routing with real CPU attention math; this does not test a CUDA kernel."""
    query = torch.randn(1, 2, 3, 8).to(torch.bfloat16)
    key = torch.randn(1, 1, 7, 8).to(torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.randn(1, 1, 3, 9)
    original_matmul = torch.matmul
    qk_calls, matmul_shapes = [], []

    def candidate(q, k, *, num_kv_groups, h100_score_kernel):
        assert q is query and k is key and num_kv_groups == 2
        assert h100_score_kernel is vars(modeling_pi05).get("_attn_scores_kernel")
        qk_calls.append(True)
        if candidate_available:
            return original_matmul(q, modeling_pi05._repeat_kv(k, num_kv_groups).transpose(2, 3))
        return None

    def matmul(left, right):
        matmul_shapes.append((left.shape, right.shape))
        return original_matmul(left, right)

    monkeypatch.setattr(modeling_pi05, "try_exact_bf16_qk", candidate)
    monkeypatch.setattr(torch, "matmul", matmul)
    expected = modeling_pi05._attend(query, key, value, mask, 2, 1.0 / 16)
    assert not qk_calls and len(matmul_shapes) == 2
    matmul_shapes.clear()
    actual = modeling_pi05._attend(query, key, value, mask, 2, 1.0 / 16, exact_bf16_softmax=True, exact_bf16_qk=True)
    assert qk_calls == [True]
    assert len(matmul_shapes) == (1 if candidate_available else 2)
    assert matmul_shapes[-1] == ((1, 2, 3, 7), (1, 2, 7, 8))
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("capability", [(9, 0), (12, 0), (8, 0)])
@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "no_triton",
        "torch",
        "cuda",
        "hip",
        "no_gluon",
        "old_triton",
        "cpu",
        "device",
        "dtype",
        "query_shape",
        "key_shape",
        "groups",
        "query_stride",
        "key_stride",
        "noncontiguous",
        "query_alignment",
        "key_alignment",
        "gradient",
    ],
)
def test_exact_qk_guard(monkeypatch, capability, unsupported):
    """CUDA metadata only; no JIT or GPU memory is used by these guard checks."""

    def tensor(shape, strides):
        return SimpleNamespace(
            shape=shape,
            dtype=torch.bfloat16,
            device=torch.device("cuda:1"),
            is_contiguous=lambda: True,
            stride=lambda: strides,
            data_ptr=lambda: 4096,
            requires_grad=False,
        )

    query = tensor((1, 8, 50, 256), (102400, 12800, 256, 1))
    key = tensor((1, 1, 1018, 256), (260608, 260608, 256, 1))
    groups = 8
    monkeypatch.setattr(exact_qk, "HAS_TRITON", True)
    monkeypatch.setattr(exact_qk, "HAS_GLUON", True)
    monkeypatch.setattr(exact_qk, "triton", SimpleNamespace(__version__="3.7.1"))
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu132")
    monkeypatch.setattr(torch.version, "cuda", "13.2")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    if unsupported == "no_triton":
        monkeypatch.setattr(exact_qk, "HAS_TRITON", False)
    elif unsupported == "torch":
        monkeypatch.setattr(torch, "__version__", "2.13.1")
    elif unsupported == "cuda":
        monkeypatch.setattr(torch.version, "cuda", "13.1")
    elif unsupported == "hip":
        monkeypatch.setattr(torch.version, "hip", "7.0")
    elif unsupported == "no_gluon":
        monkeypatch.setattr(exact_qk, "HAS_GLUON", False)
    elif unsupported == "old_triton":
        monkeypatch.setattr(exact_qk, "triton", SimpleNamespace(__version__="3.6.0"))
    elif unsupported == "cpu":
        query.device = key.device = torch.device("cpu")
    elif unsupported == "device":
        key.device = torch.device("cuda:0")
    elif unsupported == "dtype":
        key.dtype = torch.float32
    elif unsupported == "query_shape":
        query.shape = (1, 8, 49, 256)
    elif unsupported == "key_shape":
        key.shape = (1, 1, 1017, 256)
    elif unsupported == "groups":
        groups = 4
    elif unsupported == "query_stride":
        query.stride = lambda: (102400, 256, 2048, 1)
    elif unsupported == "key_stride":
        key.stride = lambda: (260608, 260608, 1, 1018)
    elif unsupported == "noncontiguous":
        key.is_contiguous = lambda: False
    elif unsupported == "query_alignment":
        query.data_ptr = lambda: 4098
    elif unsupported == "key_alignment":
        key.data_ptr = lambda: 4098
    elif unsupported == "gradient":
        key.requires_grad = True
    expected = capability in ((9, 0), (12, 0)) and (
        unsupported is None or (capability == (9, 0) and unsupported in ("no_gluon", "old_triton"))
    )
    with torch.enable_grad():
        assert exact_qk.can_use_exact_bf16_qk(query, key, groups) == expected
        if not expected:
            assert exact_qk.exact_bf16_qk(query, key, num_kv_groups=groups) is None


def test_exact_qk_missing_gluon_import_preserves_torch_fallback(monkeypatch):
    import builtins
    import importlib.util

    import vllm.triton_utils as triton_utils

    original_import = builtins.__import__

    def without_gluon(name, *args, **kwargs):
        if name.startswith("triton.experimental"):
            raise ImportError("Gluon intentionally unavailable in this import test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(triton_utils, "HAS_TRITON", True)
    monkeypatch.setattr(builtins, "__import__", without_gluon)
    spec = importlib.util.spec_from_file_location("pi05_qk_without_gluon", exact_qk.__file__)
    assert spec is not None and spec.loader is not None
    isolated = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(isolated)
    assert not isolated.HAS_GLUON
    assert not hasattr(isolated, "_native_k8_kernel")
    query = torch.randn(1, 2, 3, 8, requires_grad=True)
    key = torch.randn(1, 1, 7, 8, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    mask = torch.zeros(1, 1, 3, 7)
    monkeypatch.setattr(modeling_pi05, "try_exact_bf16_qk", isolated.exact_bf16_qk)
    expected = modeling_pi05._attend(query, key, value, mask, 2, 1.0 / 16)
    actual = modeling_pi05._attend(query, key, value, mask, 2, 1.0 / 16, exact_bf16_qk=True)
    assert torch.equal(actual, expected)
    gradient = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (query, key, value), gradient, retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (query, key, value), gradient)
    assert all(torch.equal(a, b) for a, b in zip(actual_grads, expected_grads))


@pytest.mark.parametrize("shape", [(50, 1018), (968, 968)])
@pytest.mark.parametrize("unsupported", [None, "cuda_build", "other_geometry", "broadcast_mask", "gradient"])
def test_exact_softmax_geometry_guard(monkeypatch, shape, unsupported):
    """Metadata-only CUDA dispatch checks; numerical fallback tests use real tensors."""
    device = torch.device("cuda:0")

    def tensor(dimensions):
        return SimpleNamespace(
            shape=dimensions,
            dtype=torch.bfloat16,
            device=device,
            is_contiguous=lambda: True,
            requires_grad=False,
        )

    scores, mask = tensor((1, 8, *shape)), tensor((1, 1, *shape))
    mask.dtype = torch.float32
    monkeypatch.setattr(kernels, "HAS_TRITON", True)
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu132")
    monkeypatch.setattr(torch.version, "cuda", "13.2")
    monkeypatch.setattr(torch.version, "hip", None)
    if unsupported == "cuda_build":
        monkeypatch.setattr(torch.version, "cuda", "13.1")
    elif unsupported == "other_geometry":
        other_shape = (968, 968) if shape == (50, 1018) else (50, 1018)
        mask.shape = (1, 1, *other_shape)
    elif unsupported == "broadcast_mask":
        mask.shape = (1, 1, 1, shape[-1])
    elif unsupported == "gradient":
        scores.requires_grad = True
    expected = unsupported is None or (shape == (50, 1018) and unsupported == "cuda_build")
    with torch.enable_grad():
        assert kernels.can_use_exact_scaled_masked_softmax(scores, mask, 1.0 / 16) == expected


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "torch",
        "cuda",
        "hip",
        "sm",
        "cpu",
        "dtype",
        "rows",
        "width",
        "stride",
        "bias",
        "weight",
        "device",
        "heads",
        "rope",
    ],
)
def test_accurate_expert_qkv_guard(monkeypatch, unsupported):
    """Metadata-only checks; these stand-ins never execute CUDA or GEMM math."""
    device = torch.device("cuda:1")

    def tensor(shape):
        return SimpleNamespace(shape=shape, dtype=torch.bfloat16, device=device, is_contiguous=lambda: True)

    x, cos, sin = tensor((50, 1024)), tensor((50, 256)), tensor((50, 256))
    q, k, v = tensor((8, 50, 256)), tensor((50, 256)), tensor((50, 256))
    attn = SimpleNamespace(
        **{
            name: SimpleNamespace(weight=tensor(shape), bias=None)
            for name, shape in (("q_proj", (2048, 1024)), ("k_proj", (256, 1024)), ("v_proj", (256, 1024)))
        }
    )
    monkeypatch.setattr(modeling_pi05, "HAS_TRITON", True)
    monkeypatch.setattr(torch, "__version__", "2.13.0+cu132")
    monkeypatch.setattr(torch.version, "cuda", "13.2")
    monkeypatch.setattr(torch.version, "hip", None)
    queried_devices = []

    def capability(requested_device):
        queried_devices.append(requested_device)
        return (12, 0) if unsupported == "sm" else (9, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    if unsupported == "torch":
        monkeypatch.setattr(torch, "__version__", "2.13.1+cu132")
    elif unsupported == "cuda":
        monkeypatch.setattr(torch.version, "cuda", "13.3")
    elif unsupported == "hip":
        monkeypatch.setattr(torch.version, "hip", "7.0")
    elif unsupported == "cpu":
        x.device = torch.device("cpu")
    elif unsupported == "dtype":
        x.dtype = torch.float32
    elif unsupported == "rows":
        x.shape = (49, 1024)
    elif unsupported == "width":
        x.shape = (50, 2048)
    elif unsupported == "stride":
        k.is_contiguous = lambda: False
    elif unsupported == "bias":
        attn.k_proj.bias = tensor((256,))
    elif unsupported == "weight":
        attn.v_proj.weight.shape = (512, 1024)
    elif unsupported == "device":
        attn.q_proj.weight.device = torch.device("cuda:0")
    elif unsupported == "heads":
        q.shape = (4, 50, 256)
    elif unsupported == "rope":
        cos.dtype = torch.float32
    assert modeling_pi05._can_use_accurate_expert_triton_qkv(x, attn, cos, sin, q, k, v) == (unsupported is None)
    assert queried_devices == ([device] if unsupported in (None, "sm") else [])
