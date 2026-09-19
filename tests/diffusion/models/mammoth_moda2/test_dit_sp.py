# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the MammothModa2 SP boundary, not distributed validation.

The hook tests simulate rank ownership/gather on CPU. Arithmetic checks execute
real shared SDPA at SP=1; real two-rank collectives need a separate CUDA test.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed import parallel_state, sp_sharding
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig, validate_sp_plan
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.hooks import sequence_parallel as sp_hooks
from vllm_omni.diffusion.models.mammoth_moda2.mammothmoda2_dit_model import Transformer2DModel, TransformerBlock

from .test_dit_attention import _reference_attention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.parallel]


def _config(**parallel_kwargs):
    return OmniDiffusionConfig(
        parallel_config=DiffusionParallelConfig(**parallel_kwargs),
        diffusion_attention_config={"default": {"backend": "TORCH_SDPA"}},
    )


def _model():
    # Preserve the released 21/7 GQA ratio, with small dimensions for L1.
    torch.manual_seed(7)
    with set_current_diffusion_config(_config()):
        return Transformer2DModel(
            hidden_size=126,
            num_attention_heads=21,
            num_kv_heads=7,
            axes_dim_rope=(2, 2, 2),
            axes_lens=(32, 32, 32),
            num_layers=2,
            num_refiner_layers=1,
            in_channels=4,
            text_feat_dim=16,
            multiple_of=8,
        ).eval()


def test_plan_shards_only_the_main_joint_stream():
    model = _model()
    validate_sp_plan(model._sp_plan)
    assert set(model._sp_plan) == {"sp_input_boundary", "sp_output_boundary"}
    assert not model.sp_input_boundary.state_dict()
    assert not model.sp_output_boundary.state_dict()
    for refiners in (model.context_refiner, model.noise_refiner, model.ref_image_refiner):
        for block in refiners:
            assert block.attn.omni_attn.skip_sequence_parallel
    assert all(not block.attn.omni_attn.skip_sequence_parallel for block in model.layers)


@pytest.mark.parametrize("hooks_applied", [False, True])
def test_refiners_stay_local_in_a_fresh_sp_forward_context(hooks_applied):
    model = _model()
    with set_forward_context(omni_diffusion_config=_config(ulysses_degree=2, ulysses_mode="advanced_uaa")):
        ctx = get_forward_context()
        ctx.sp_plan_hooks_applied = hooks_applied
        for refiners in (model.context_refiner, model.noise_refiner, model.ref_image_refiner):
            for block in refiners:
                assert block.attn.omni_attn._get_active_parallel_strategy().name == "none"


@pytest.mark.parametrize("rank", [0, 1])
def test_real_hooks_preserve_global_rope_masks_and_reset_cfg_padding(monkeypatch, rank):
    model = _model()
    sp_hooks.apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=2), model._sp_plan)
    # Simulate rank ownership only. No process group or collective is claimed.
    for module in (parallel_state, sp_sharding):
        monkeypatch.setattr(module, "get_sequence_parallel_world_size", lambda: 2)
        monkeypatch.setattr(module, "get_sequence_parallel_rank", lambda: rank)
    monkeypatch.setattr(parallel_state, "get_ring_parallel_world_size", lambda: 1)

    with set_forward_context(omni_diffusion_config=_config(ulysses_degree=2, ulysses_mode="advanced_uaa")):
        ctx = get_forward_context()
        ctx.sp_plan_hooks_applied = True
        # Odd conditional, even unconditional, then the original branch again.
        for seq in (7, 4, 7):
            hidden = torch.arange(2 * 126 * seq, dtype=torch.float32).reshape(2, 126, seq).transpose(1, 2)
            angles = torch.arange(6 * seq, dtype=torch.float32).reshape(1, 6, seq).transpose(1, 2) + 13
            rotary = (angles.cos(), angles.sin())
            mask = torch.ones(2, seq, dtype=torch.bool)
            mask[1, -2:] = False
            pad = seq % 2
            full_hidden = F.pad(hidden, (0, 0, 0, pad))
            full_mask = F.pad(mask, (0, pad), value=False)
            local_hidden = full_hidden.chunk(2, dim=1)[rank]
            local_mask = full_mask.chunk(2, dim=1)[rank]
            local_rope = tuple(F.pad(x, (0, 0, 0, pad)).chunk(2, dim=1)[rank] for x in rotary)
            seen = []

            def check_layer(hidden_states, attention_mask, image_rotary_emb, temb, query_attention_mask):
                torch.testing.assert_close(hidden_states, local_hidden, rtol=0, atol=0)
                torch.testing.assert_close(attention_mask, full_mask)
                torch.testing.assert_close(query_attention_mask, local_mask)
                for actual, expected in zip(image_rotary_emb, local_rope):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert ctx.sp_active and ctx._sp_shard_depth == 1
                assert ctx._sp_equal_pad_stack == [True]
                assert ctx.sp_rank_local_seq_lens_equal
                assert ctx.sp_padding_size == pad
                assert ctx.sp_original_seq_len == (seq if pad else None)
                seen.append(True)
                return hidden_states

            def simulated_gather(tensor, dim, validate=True):
                assert dim == 1
                torch.testing.assert_close(tensor, local_hidden, rtol=0, atol=0)
                return full_hidden

            for layer in model.layers:
                monkeypatch.setattr(layer, "forward", check_layer)
            monkeypatch.setattr(sp_hooks, "sp_gather", simulated_gather)
            actual = model._apply_transformer_layers(hidden, mask, rotary, None)
            torch.testing.assert_close(actual, hidden, rtol=0, atol=0)
            assert len(seen) == len(model.layers)
            assert ctx.sp_padding_size == 0
            assert ctx.sp_original_seq_len is None
            assert ctx._sp_shard_depth == 0 and not ctx.sp_active
            assert ctx._sp_equal_pad_stack == []


def test_processor_uses_global_key_mask_and_local_output_mask(monkeypatch):
    block = _model().layers[0]
    hidden = torch.randn(1, 3, 126)
    global_mask = torch.tensor([[True, True, True, True, True, False]])
    local_mask = global_mask[:, 3:]
    seen = []

    def already_restored_local_attention(query, key, value, attn_metadata):
        # This stub isolates the post-all-to-all output-row contract, not math.
        assert query.shape == (1, 3, 21, 6)
        assert key.shape == value.shape == (1, 3, 7, 6)
        torch.testing.assert_close(attn_metadata.attn_mask, global_mask)
        seen.append(True)
        return torch.ones_like(query)

    monkeypatch.setattr(block.attn.omni_attn, "forward", already_restored_local_attention)
    out = block.attn(
        hidden_states=hidden,
        encoder_hidden_states=hidden,
        attention_mask=global_mask,
        query_attention_mask=local_mask,
    )
    assert seen and out.shape == hidden.shape
    assert torch.count_nonzero(out[~local_mask]) == 0
    assert torch.count_nonzero(out[local_mask]) > 0


@pytest.mark.parametrize("failure_at", ["split", "block", "gather"])
@pytest.mark.parametrize("outer_boundaries", [(), (False,)], ids=["fresh", "nested"])
def test_sp_context_is_restored_when_a_branch_raises(monkeypatch, failure_at, outer_boundaries):
    model = _model()
    sp_hooks.apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=2), model._sp_plan)
    for module in (parallel_state, sp_sharding):
        monkeypatch.setattr(module, "get_sequence_parallel_world_size", lambda: 2)
        monkeypatch.setattr(module, "get_sequence_parallel_rank", lambda: 0)
    monkeypatch.setattr(parallel_state, "get_ring_parallel_world_size", lambda: 1)
    hidden = torch.randn(1, 7, 126)
    mask = torch.ones(1, 7, dtype=torch.bool)
    rotary = (torch.ones(1, 7, 6), torch.zeros(1, 7, 6))

    def fail(*args, **kwargs):
        raise RuntimeError("injected branch failure")

    def local_identity(hidden_states, *args):
        return hidden_states

    for layer in model.layers:
        monkeypatch.setattr(layer, "forward", local_identity)
    if failure_at == "split":
        monkeypatch.setattr(sp_hooks.SequenceParallelSplitHook, "_shard_with_auto_pad", fail)
    elif failure_at == "block":
        monkeypatch.setattr(model.layers[0], "forward", fail)
    else:
        monkeypatch.setattr(sp_hooks, "sp_gather", fail)

    with set_forward_context(omni_diffusion_config=_config(ulysses_degree=2, ulysses_mode="advanced_uaa")):
        ctx = get_forward_context()
        ctx.sp_plan_hooks_applied = True
        # Restore the caller's metadata exactly, not just a fresh-context default.
        ctx.sp_original_seq_len, ctx.sp_padding_size = 11, 1
        ctx._sp_shard_depth = len(outer_boundaries)
        ctx._sp_equal_pad_stack.extend(outer_boundaries)
        original_stack = ctx._sp_equal_pad_stack
        with pytest.raises(RuntimeError, match="injected branch failure"):
            model._apply_transformer_layers(hidden, mask, rotary, None)
        assert (ctx.sp_original_seq_len, ctx.sp_padding_size, ctx._sp_shard_depth) == (11, 1, len(outer_boundaries))
        assert ctx.sp_active == bool(outer_boundaries)
        assert ctx._sp_equal_pad_stack is original_stack
        assert ctx._sp_equal_pad_stack == list(outer_boundaries)


@pytest.mark.parametrize("text_len", [0, 3, 4])
def test_single_rank_full_transformer_matches_pre_boundary_forward(monkeypatch, text_len):
    model = _model()
    hidden = torch.randn(1, 4, 4, 4)
    text = torch.randn(1, text_len, 16)
    mask = torch.ones(1, text_len, dtype=torch.bool)
    rotary = model.rope_embedder.get_freqs_real((2, 2, 2), (32, 32, 32), 10000)
    args = (hidden, torch.tensor([0.5]), text, rotary, mask)
    with torch.no_grad():
        actual = model(*args)

        def legacy_main(hidden_states, attention_mask, rotary_emb, temb):
            for layer in model.layers:
                hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)
            return hidden_states

        monkeypatch.setattr(model, "_apply_transformer_layers", legacy_main)
        expected = model(*args)
    assert torch.isfinite(actual).all() and actual.shape == hidden.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_released_head_geometry_matches_single_rank_legacy_attention():
    torch.manual_seed(1)
    with set_current_diffusion_config(_config()):
        block = TransformerBlock(2520, 21, 7, multiple_of=256, ffn_dim_multiplier=1.0, norm_eps=1e-5).eval()
    hidden = torch.randn(1, 5, 2520)
    mask = torch.tensor([[True, True, True, False, False]])
    angles = torch.randn(1, 5, 120) + 13
    rotary = (angles.cos(), angles.sin())
    with torch.no_grad():
        actual = block.attn(
            hidden_states=hidden, encoder_hidden_states=hidden, attention_mask=mask, image_rotary_emb=rotary
        )
        expected = _reference_attention(block.attn, hidden, mask, rotary)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "parallel_kwargs",
    [
        {"ulysses_degree": 4},
        {"ring_degree": 2},
        {"allgather_degree": 2},
        {"ulysses_degree": 2, "cfg_parallel_size": 2},
        {"ulysses_degree": 2, "tensor_parallel_size": 2},
        {"ulysses_degree": 2, "pipeline_parallel_size": 2},
        {"ulysses_degree": 2, "data_parallel_size": 2},
        {"ulysses_degree": 2, "vae_patch_parallel_size": 2},
        {"ulysses_degree": 2, "use_hsdp": True, "hsdp_shard_size": 2},
        {"ulysses_degree": 2, "enable_expert_parallel": True},
    ],
)
def test_unsupported_parallel_combinations_fail_before_computation(parallel_kwargs):
    model = _model()
    with set_forward_context(omni_diffusion_config=_config(**parallel_kwargs)):
        with pytest.raises(ValueError, match="only two-rank Ulysses"):
            model._validate_sequence_parallel()


def test_released_uneven_heads_reject_strict_mode():
    model = _model()
    with set_forward_context(omni_diffusion_config=_config(ulysses_degree=2)):
        with pytest.raises(ValueError, match="advanced_uaa"):
            model._validate_sequence_parallel()


@pytest.mark.parametrize("missing_boundary", ["sp_input_boundary", "sp_output_boundary"])
def test_missing_runtime_hooks_fail_before_computation(missing_boundary):
    model = _model()
    partial_plan = {key: value for key, value in model._sp_plan.items() if key != missing_boundary}
    sp_hooks.apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=2), partial_plan)
    with set_forward_context(omni_diffusion_config=_config(ulysses_degree=2, ulysses_mode="advanced_uaa")):
        with pytest.raises(RuntimeError, match="complete Transformer2DModel._sp_plan"):
            model._validate_sequence_parallel()


def test_hooks_without_initialized_attention_do_not_claim_sp():
    model = _model()
    sp_hooks.apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=2), model._sp_plan)
    with set_forward_context(omni_diffusion_config=_config(ulysses_degree=2, ulysses_mode="advanced_uaa")):
        with pytest.raises(RuntimeError, match="initialized Ulysses ForwardContext"):
            model._validate_sequence_parallel()


def test_hooks_without_sp_config_fail_before_computation():
    model = _model()
    sp_hooks.apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=2), model._sp_plan)
    with pytest.raises(RuntimeError, match="on every forward"):
        model._validate_sequence_parallel()


def test_runtime_hook_degree_mismatch_fails_before_computation():
    model = _model()
    sp_hooks.apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=4), model._sp_plan)
    with set_forward_context(omni_diffusion_config=_config(ulysses_degree=2, ulysses_mode="advanced_uaa")):
        with pytest.raises(RuntimeError, match="same two-rank Ulysses configuration"):
            model._validate_sequence_parallel()


def test_initialized_sp_attention_requires_context_on_each_forward(monkeypatch):
    class EnabledAttention(NoParallelAttention):
        @property
        def enabled(self) -> bool:
            return True

    model = _model()
    monkeypatch.setattr(model.layers[0].attn.omni_attn, "parallel_strategy", EnabledAttention())
    with pytest.raises(RuntimeError, match="on every forward"):
        model._validate_sequence_parallel()
