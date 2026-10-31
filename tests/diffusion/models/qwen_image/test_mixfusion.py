# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side tests for the Qwen-Image MixFusion adapter and driver guards (CPU).

MixFusion splits the packed mixed-resolution path into three layers: the
generic ``MixFusionDriver`` (dispatch, admission, fallback), the model-agnostic
hooks + ``PackedExecutionPlan``, and the model-side ``MixFusionAdapter``. This
module exercises the Qwen-Image adapter and the driver admission from the
model's point of view:

* ``prepare_inputs`` owns packed-layout preparation (row expansion, chunk
  split, RoPE tables, block-agnostic plan) and its step-invariant cache.
* The block-hook guard refuses to pack while blocks carry hooks (layerwise
  offload / HSDP ``fully_shard``); the driver falls back to the serial path,
  whose per-request loops route every block through the hooked module call.
* ``attention_kwargs`` must be threaded by the serial path or rejected loudly
  at prepare, never silently dropped by the packed layout.
* The packed cross-attention flattens mixed requests into one
  ``forward_packed`` call, preserving the eager path's RoPE dtype contract
  (#7494) and per-document semantics on both the varlen route and the dense
  bucketed fallback.

Framework-level behavior (generic dispatch, protocol validation, layout
primitives) lives in ``tests/diffusion/mixfusion``; the hook mechanism itself
is covered by ``tests/diffusion/hooks``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    MixFusionStep,
    get_forward_context,
    override_forward_context,
)
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook, module_has_active_hooks
from vllm_omni.diffusion.hooks.mixfusion import plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout, StreamSpec
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.mixfusion.execution_plan import build_execution_plan
from vllm_omni.diffusion.mixfusion.packed_attention import (
    MIXFUSION_PLAN_KEY,
    run_mixfusion_packed_attention,
)
from vllm_omni.diffusion.models.qwen_image.mixfusion_adapter import QwenImageMixFusionAdapter
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageCrossAttention,
    QwenImageTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

# A non-empty sentinel: any truthy dict stands in for diffusers-style
# per-attention arguments (e.g. IP-Adapter or LoRA scale).
SENTINEL_KWARGS = {"scale": 0.5}

ROPE_DIM = 32


class _PlainBlock(nn.Module):
    def forward(self, x):
        return x


def _transformer(blocks: list[nn.Module] | None = None) -> QwenImageTransformer2DModel:
    """A transformer with only the guards' dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed;
    only non-Module attributes are set, so ``object.__setattr__`` bypasses the
    nn.Module machinery.
    """
    transformer = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    object.__setattr__(transformer, "zero_cond_t", False)
    object.__setattr__(transformer, "parallel_config", SimpleNamespace(sequence_parallel_size=1))
    # ``plan_foreign_hooks`` expands ``*`` only over ``nn.ModuleList``.
    object.__setattr__(transformer, "transformer_blocks", nn.ModuleList([_PlainBlock()] if blocks is None else blocks))
    return transformer


def _prepare_transformer() -> tuple[QwenImageTransformer2DModel, list[int]]:
    """A transformer with only the prepare path's dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed;
    ``pos_embed`` / ``img_in`` are stubbed with plain attributes so the RoPE
    tables are zeros of the right shape and ``img_in`` is an identity (the
    prepared ``hidden_states`` then equals the chunk-stacked latents).
    ``pos_embed_calls`` records how often the (step-invariant) tables were
    rebuilt.
    """
    transformer = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    object.__setattr__(transformer, "zero_cond_t", False)
    object.__setattr__(transformer, "parallel_config", SimpleNamespace(sequence_parallel_size=1))
    object.__setattr__(transformer, "transformer_blocks", nn.ModuleList([_PlainBlock()]))
    object.__setattr__(transformer, "img_in", nn.Identity())
    pos_embed_calls: list[int] = [0]

    def pos_embed(img_shapes, txt_seq_lens, device):
        # img_shapes entry: (img_token_count, ...); freq rows must cover the
        # image row length and at least the text length.
        pos_embed_calls[0] += 1
        img_tokens = int(img_shapes[0])
        txt_len = int(txt_seq_lens[0])
        return torch.zeros(img_tokens, ROPE_DIM), torch.zeros(txt_len + 4, ROPE_DIM)

    object.__setattr__(transformer, "pos_embed", pos_embed)
    return transformer, pos_embed_calls


def _pipeline(transformer):
    """The driver's pipeline surface, without constructing the real pipeline."""
    return SimpleNamespace(
        transformer=transformer,
        # ``interrupt`` is a read-only property on the real pipeline.
        interrupt=False,
        attention_kwargs=None,
        parallel_config=SimpleNamespace(sequence_parallel_size=1),
    )


class _StubTransformer:
    """Transformer stub recording packed-path calls for the driver tests."""

    def __init__(self, *, plan=None, blocks=None):
        self._mixfusion_plan = plan
        self.transformer_blocks = nn.ModuleList(blocks or [])
        self.packed_calls = 0
        self.calls = []
        self.armed_at_call = None
        # Stamped by ``apply_mixfusion``; the driver refuses the packed
        # call when the registry never installed the hooks.
        self._mixfusion_hooks_applied = True

    def __call__(self, **kwargs):
        self.packed_calls += 1
        self.calls.append(kwargs)
        # The driver must arm the step for exactly the transformer call.
        self.armed_at_call = get_forward_context().mixfusion_step is not None
        return "packed"


class _StubAdapter:
    """Minimal model-side adapter; the transformer stub consumes any payload."""

    def build_packed_call(self, input_batch):
        return {"hidden_states": input_batch.latents}


def _driver(transformer, adapter):
    driver = MixFusionDriver(_pipeline(transformer), transformer, adapter)
    serial_inputs = []
    driver.serial_ragged = lambda batch, states, dense_step: serial_inputs.append(batch) or "serial"
    return driver, serial_inputs


def _packed_input_batch():
    return SimpleNamespace(
        latents=[torch.zeros(1, 8, 16), torch.zeros(1, 8, 16)],
        do_true_cfg=False,
        image_latents=None,
        # The driver aggregates per-request thresholds; lowering the minimum
        # lets this tiny fixture stay packable.
        request_extras=[{"mixfusion_min_chunk_tokens": 1}],
        timesteps=torch.tensor([500.0]),
        guidance=None,
        prompt_embeds=torch.zeros(2, 4, 8),
        prompt_embeds_mask=None,
        img_shapes=[(16, 2, 2), (16, 2, 2)],
        txt_seq_lens=[4, 4],
    )


def _dense_step(batch, **kwargs):
    return "dense"


# ---------------------------------------------------------------------------
# Block-hook guard: packed runs must refuse hooked blocks and fall back.
# ---------------------------------------------------------------------------


def _hooked_block() -> nn.Module:
    """A block with a registered hook, as installed by layerwise offload."""
    block = _PlainBlock()
    HookRegistry.get_or_create(block).register_hook("layerwise_offload", ModelHook())
    return block


def _hsdp_like_block() -> nn.Module:
    """A block with a PyTorch-native forward pre-hook, as registered by
    HSDP's ``fully_shard`` (the unshard/all-gather pre-hook)."""
    block = _PlainBlock()
    block.register_forward_pre_hook(lambda module, args: args)
    return block


def test_module_has_active_hooks():
    plain = _PlainBlock()
    registry_only = _PlainBlock()
    HookRegistry.get_or_create(registry_only)  # empty registry must not count

    assert not module_has_active_hooks(plain)
    assert not module_has_active_hooks(registry_only)
    assert module_has_active_hooks(_hooked_block())


def test_plan_foreign_hooks_reports_block_hooks():
    plan = QwenImageTransformer2DModel._mixfusion_plan
    assert plan_foreign_hooks(_transformer([_PlainBlock()]), plan) == []
    assert plan_foreign_hooks(_transformer([_PlainBlock(), _hooked_block()]), plan) == [
        "transformer_blocks.*.img_mod:layerwise_offload"
    ]


def test_plan_foreign_hooks_reports_native_block_hooks():
    # HSDP's fully_shard registers PyTorch-native hooks (invisible to the
    # Omni HookRegistry); the guard must fall back for those too.
    plan = QwenImageTransformer2DModel._mixfusion_plan
    assert plan_foreign_hooks(_transformer([_PlainBlock()]), plan) == []
    assert plan_foreign_hooks(_transformer([_PlainBlock(), _hsdp_like_block()]), plan) == [
        "transformer_blocks.*.img_mod:torch-native"
    ]


def _prepare_packed(transformer, **overrides):
    kwargs = dict(
        hidden_states=[torch.zeros(1, 8, 16)],
        encoder_hidden_states=torch.zeros(1, 4, 8),
        img_shapes=[(16, 2, 2)],
        txt_seq_lens=[4],
    )
    kwargs.update(overrides)
    QwenImageMixFusionAdapter().prepare_inputs(transformer, **kwargs)


def test_packed_prepare_rejects_hooked_blocks():
    with pytest.raises(ValueError, match="unvalidated semantics"):
        _prepare_packed(_transformer([_hooked_block()]))


def test_packed_prepare_rejects_native_hooked_blocks():
    with pytest.raises(ValueError, match="unvalidated semantics"):
        _prepare_packed(_transformer([_hsdp_like_block()]))


def test_packed_prepare_passes_the_hook_guard_without_hooks():
    # Without hooks the guard must not fire; execution continues to the
    # regular input validation (empty hidden states).
    with pytest.raises(ValueError, match="at least one hidden-state tensor"):
        _prepare_packed(_transformer([_PlainBlock()]), hidden_states=[])


def test_driver_falls_back_to_serial_when_blocks_are_hooked():
    transformer = _StubTransformer(
        plan={"transformer_blocks.*": "img_mod"},
        blocks=[_hooked_block()],
    )
    assert plan_foreign_hooks(transformer, transformer._mixfusion_plan) == ["transformer_blocks.*:layerwise_offload"]
    driver, serial_inputs = _driver(transformer, _StubAdapter())
    input_batch = _packed_input_batch()

    assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_falls_back_to_serial_when_blocks_carry_native_hooks():
    transformer = _StubTransformer(
        plan={"transformer_blocks.*": "img_mod"},
        blocks=[_hsdp_like_block()],
    )
    driver, serial_inputs = _driver(transformer, _StubAdapter())
    input_batch = _packed_input_batch()

    assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_packs_when_blocks_are_not_hooked():
    transformer = _StubTransformer(plan={"transformer_blocks.*": "img_mod"}, blocks=[_PlainBlock()])
    driver, serial_inputs = _driver(transformer, _StubAdapter())
    input_batch = _packed_input_batch()

    with override_forward_context(ForwardContext()):
        assert driver.run(input_batch, states=[], dense_step=_dense_step) == "packed"
    assert transformer.packed_calls == 1
    assert serial_inputs == []


# ---------------------------------------------------------------------------
# attention_kwargs must survive the dispatch (serial) or fail loudly (packed).
# ---------------------------------------------------------------------------


def test_prepare_rejects_attention_kwargs_on_packed_input():
    with pytest.raises(ValueError, match="does not consume attention_kwargs"):
        QwenImageMixFusionAdapter().prepare_inputs(
            _transformer(),
            hidden_states=[torch.zeros(1, 8, 16)],
            encoder_hidden_states=torch.zeros(1, 4, 8),
            img_shapes=[(16, 2, 2)],
            txt_seq_lens=[4],
            attention_kwargs=dict(SENTINEL_KWARGS),
        )


def test_prepare_accepts_empty_attention_kwargs_on_packed_input():
    # An empty dict carries no per-attention arguments, so the packed dispatch
    # must proceed (here: into the regular input validation).
    with pytest.raises(ValueError, match="at least one hidden-state tensor"):
        QwenImageMixFusionAdapter().prepare_inputs(
            _transformer(),
            hidden_states=[],
            encoder_hidden_states=torch.zeros(1, 4, 8),
            img_shapes=[(16, 2, 2)],
            txt_seq_lens=[4],
            attention_kwargs={},
        )


def test_driver_falls_back_to_serial_with_attention_kwargs():
    transformer = _StubTransformer()
    driver, serial_inputs = _driver(transformer, QwenImageMixFusionAdapter())
    driver.pipeline.attention_kwargs = dict(SENTINEL_KWARGS)
    input_batch = SimpleNamespace(
        latents=[torch.zeros(1, 8, 16)],
        do_true_cfg=False,
        image_latents=None,
        request_extras=None,
        timesteps=torch.tensor([500.0]),
    )

    assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_falls_back_when_hooks_missing():
    transformer = _StubTransformer()
    transformer._mixfusion_hooks_applied = False
    driver, serial_inputs = _driver(transformer, QwenImageMixFusionAdapter())
    input_batch = _packed_input_batch()

    with override_forward_context(ForwardContext()):
        assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_falls_back_with_foreign_root_hooks():
    transformer = _StubTransformer()
    # A non-MixFusion hook on the root sees the swapped-in packed batch;
    # cache-style hooks would build keys on the wrong layout.
    transformer._hook_registry = SimpleNamespace(_hooks={"cache": object()})
    driver, serial_inputs = _driver(transformer, QwenImageMixFusionAdapter())
    input_batch = _packed_input_batch()

    assert driver.run(input_batch, states=[], dense_step=_dense_step) == "serial"
    assert serial_inputs == [input_batch]
    assert transformer.packed_calls == 0


def test_driver_arms_step_and_forwards_layout_kwargs():
    transformer = _StubTransformer()
    driver = MixFusionDriver(_pipeline(transformer), transformer, QwenImageMixFusionAdapter())
    input_batch = _packed_input_batch()

    with override_forward_context(ForwardContext()):
        assert driver.run(input_batch, states=[], dense_step=_dense_step) == "packed"
        # Restored by the try/finally so later calls see no armed step.
        assert get_forward_context().mixfusion_step is None

    assert transformer.packed_calls == 1
    assert transformer.armed_at_call is True
    call = transformer.calls[0]
    # The root hook packs the list itself; the adapter forwards the layout
    # arguments it needs instead of a pre-built bundle.
    assert call["hidden_states"] is input_batch.latents
    assert call["img_shapes"] == input_batch.img_shapes
    assert call["txt_seq_lens"] == input_batch.txt_seq_lens
    assert call["return_dict"] is False


# ---------------------------------------------------------------------------
# prepare_inputs: packed-layout preparation and its step-invariant cache.
# ---------------------------------------------------------------------------


def _prepare(transformer, latents, *, prompt_rows, seq_len_txt=4, cached=None, img_shapes=None):
    if img_shapes is None:
        img_shapes = [(int(latent.shape[1]), 2, 2) for latent in latents]
    return QwenImageMixFusionAdapter().prepare_inputs(
        transformer,
        hidden_states=latents,
        encoder_hidden_states=torch.zeros(prompt_rows, seq_len_txt, 8),
        encoder_hidden_states_mask=None,
        img_shapes=img_shapes,
        txt_seq_lens=[seq_len_txt] * len(latents),
        cached=cached,
    )


def test_prepare_builds_packed_inputs():
    transformer, _ = _prepare_transformer()
    latents = [torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)]
    prepared = _prepare(transformer, latents, prompt_rows=1)

    # img_in is an identity: hidden_states is the chunk-stacked latents.
    assert torch.equal(prepared.hidden_states, torch.cat([latents[0]], dim=0))
    assert prepared.chunk_size == 8
    assert prepared.image_row_counts == [1]
    assert prepared.plan is not None
    assert prepared.plan.chunk_to_request.tolist() == [0]
    # No mask given: the plan sees the full text length per row.
    assert prepared.encoder_hidden_states_mask is None
    # RoPE tables are stacked per chunk row / per image row.
    assert prepared.image_rotary_emb[0].shape == (1, 8, ROPE_DIM)
    assert prepared.image_rotary_emb[1].shape == (1, 4, ROPE_DIM)


def test_prepare_normalizes_all_valid_mask():
    transformer, _ = _prepare_transformer()
    latents = [torch.zeros(1, 8, 2)]
    prepared = QwenImageMixFusionAdapter().prepare_inputs(
        transformer,
        hidden_states=latents,
        encoder_hidden_states=torch.zeros(1, 4, 8),
        encoder_hidden_states_mask=torch.ones(1, 4),
        img_shapes=[(8, 2, 2)],
        txt_seq_lens=[4],
    )
    assert prepared.encoder_hidden_states_mask is None


def test_prepare_reuses_step_invariant_layout_from_cached():
    transformer, pos_embed_calls = _prepare_transformer()
    latents_step_1 = [torch.ones(1, 8, 2), 2 * torch.ones(1, 8, 2)]
    latents_step_2 = [3 * torch.ones(1, 8, 2), 4 * torch.ones(1, 8, 2)]

    first = _prepare(transformer, latents_step_1, prompt_rows=2)
    assert pos_embed_calls[0] == 2  # one RoPE build per image row

    second = _prepare(transformer, latents_step_2, prompt_rows=2, cached=first)
    # Step-invariant layout reused verbatim: no RoPE or plan rebuild.
    assert pos_embed_calls[0] == 2
    assert second.plan is first.plan
    assert second.image_rotary_emb is first.image_rotary_emb
    assert second.row_vid_freqs is first.row_vid_freqs
    assert second.text_freqs is first.text_freqs
    assert second.chunk_size == first.chunk_size
    assert second.key == first.key
    # Per-step latents are re-split and re-projected.
    assert second.hidden_states is not first.hidden_states
    assert torch.equal(second.hidden_states, torch.cat(latents_step_2, dim=0))


def test_prepare_rebuilds_layout_when_shapes_change():
    transformer, pos_embed_calls = _prepare_transformer()
    first = _prepare(transformer, [torch.ones(1, 8, 2)], prompt_rows=1)
    assert pos_embed_calls[0] == 1

    # A different request mix (two rows, different length) must not reuse
    # the cached single-row layout.
    latents = [torch.ones(1, 8, 2), torch.ones(1, 8, 2)]
    second = _prepare(transformer, latents, prompt_rows=2, cached=first)

    assert pos_embed_calls[0] == 3  # rebuild: +2 rows
    assert second.plan is not first.plan
    assert second.key != first.key
    assert second.image_row_counts == [1, 1]
    assert second.hidden_states.shape == (2, 8, 2)


def test_prepare_splits_mixed_lengths_into_equal_chunks():
    transformer, _ = _prepare_transformer()
    latents = [torch.ones(1, 8, 2), torch.ones(1, 12, 2)]
    prepared = _prepare(transformer, latents, prompt_rows=2)

    # gcd(8, 12) = 4: 2 + 3 chunks, stacked on the batch dimension.
    assert prepared.chunk_size == 4
    assert prepared.hidden_states.shape == (5, 4, 2)
    assert prepared.plan.chunk_to_request.tolist() == [0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# Packed cross-attention: one forward_packed call over mixed requests.
# ---------------------------------------------------------------------------

HEADS, KV_HEADS, HEAD_DIM = 2, 1, 4
Q_SIZE = HEADS * HEAD_DIM
QKV_DIM = Q_SIZE + 2 * KV_HEADS * HEAD_DIM
ROT_DIM = HEAD_DIM // 2

# Two requests with different text lengths and chunk counts:
#   req0: 3 text tokens + 2 image chunks (8 tokens)  -> seq 11
#   req1: 5 text tokens + 3 image chunks (12 tokens) -> seq 17
SEQ_LEN_TXT = 7  # padded text length shared by both requests
REAL_TXT_LENS = [3, 5]
REQUEST_CHUNK_RANGES = [(0, 2), (2, 5)]
CHUNK_SIZE = 4
NUM_CHUNKS = 5
REQ_LENS = [11, 17]
CU_SEQLENS = [0, 11, 28]


class _CapturingRope:
    """Stand-in for RotaryEmbedding that records the cos/sin dtype."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, tensor, cos, sin):
        self.calls.append(
            {
                "tensor_dtype": tensor.dtype,
                "cos_dtype": cos.dtype,
                "sin_dtype": sin.dtype,
            }
        )
        return tensor


def _segment_mean_flat(query, key, value, cu_seqlens):
    """Per-document semantics: each document's rows gain its own K/V means.

    A cross-document leak changes the means, so routing this through both the
    varlen route and the bucketed dense fallback makes the two observable
    parity checks of the packed layout, not just of shape plumbing.
    """
    out = torch.empty_like(query)
    lengths = cu_seqlens.diff().tolist()
    offset = 0
    for length in lengths:
        if length:
            out[offset : offset + length] = (
                query[offset : offset + length]
                + key[offset : offset + length].mean(dim=0, keepdim=True)
                + value[offset : offset + length].mean(dim=0, keepdim=True)
            )
        offset += length
    return out


class _PackedAttentionStub:
    """Attention stand-in exposing the ``forward_packed`` entry point.

    ``varlen`` applies the per-document semantics directly (the single-kernel
    route); ``bucketed`` delegates to the production dense fallback so the
    model's packing works unchanged on backends without packed varlen support.
    """

    def __init__(self, mode: str):
        assert mode in ("varlen", "bucketed")
        self.mode = mode
        self.calls: list[dict] = []
        self.dense_shapes: list[tuple] = []

    def __call__(self, query, key, value, attn_metadata):
        # Mirror the metadata-keyed dispatch at the top of ``Attention.forward``
        # (the real layer routes the same way; this stub replaces the layer so
        # the test stays on CPU).
        if attn_metadata.extra.get(MIXFUSION_PLAN_KEY) is not None:
            return run_mixfusion_packed_attention(self, query, key, value, attn_metadata)
        raise AssertionError("stub reached without a mixfusion plan")

    def forward_packed(self, query, key, value, *, cu_seqlens, max_seqlen):
        self.calls.append(
            {
                "query": query,
                "key": key,
                "value": value,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
            }
        )
        if self.mode == "varlen":
            return _segment_mean_flat(query, key, value, cu_seqlens)

        def dense_forward(dense_query, dense_key, dense_value, attn_metadata):
            self.dense_shapes.append(tuple(dense_query.shape))
            return dense_query + dense_key.mean(dim=1, keepdim=True) + dense_value.mean(dim=1, keepdim=True)

        fake_layer = SimpleNamespace(forward=dense_forward)
        return Attention._forward_packed_dense_buckets(fake_layer, query, key, value, cu_seqlens)


def _mixfusion_module(*, attn: _PackedAttentionStub, rope: _CapturingRope):
    """A QwenImageCrossAttention with only the packed-path dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed:
    identity projections keep q/k/v rows equal to the input rows, which makes
    the packed token order directly assertable. Only non-Module attributes
    are set, so ``object.__setattr__`` bypasses the nn.Module machinery.
    """

    module = QwenImageCrossAttention.__new__(QwenImageCrossAttention)
    attrs = {
        "parallel_config": None,
        "query_num_heads": HEADS,
        "kv_num_heads": KV_HEADS,
        "add_query_num_heads": HEADS,
        "add_kv_num_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "to_qkv": lambda x: (x, None),
        "add_kv_proj": lambda x: (x, None),
        "to_out": lambda x: x,
        "to_add_out": lambda x: x,
        "norm_q": lambda t: t,
        "norm_k": lambda t: t,
        "norm_added_q": lambda t: t,
        "norm_added_k": lambda t: t,
        "qk_norm": True,
        "eps": 1e-6,
        "rope": rope,
        "attn": attn,
    }
    for name, value in attrs.items():
        object.__setattr__(module, name, value)
    return module


def _mixfusion_inputs():
    gen = torch.Generator().manual_seed(2026)
    image_chunks = torch.randn(NUM_CHUNKS, CHUNK_SIZE, QKV_DIM, generator=gen).to(torch.bfloat16)
    encoder_hidden_states = torch.randn(2, SEQ_LEN_TXT, QKV_DIM, generator=gen).to(torch.bfloat16)
    image_freq_chunks = torch.randn(NUM_CHUNKS, CHUNK_SIZE, ROT_DIM, generator=gen, dtype=torch.complex64)
    text_freqs = torch.randn(2, SEQ_LEN_TXT, ROT_DIM, generator=gen, dtype=torch.complex64)
    chunk_to_request = torch.tensor([0, 0, 1, 1, 1])
    return image_chunks, encoder_hidden_states, image_freq_chunks, text_freqs, chunk_to_request


def _attention_layout():
    return AttentionLayout((StreamSpec("text", "request"), StreamSpec("image", "chunked")))


def _forward(module):
    image_chunks, encoder_hidden_states, image_freq_chunks, text_freqs, chunk_to_request = _mixfusion_inputs()
    layout = _attention_layout()
    plan = build_execution_plan(
        layout=layout,
        chunk_size=CHUNK_SIZE,
        request_chunk_ranges=REQUEST_CHUNK_RANGES,
        image_row_counts=[1, 1],
        request_stream_lens={
            "text": REAL_TXT_LENS,
            "image": [(end - start) * CHUNK_SIZE for start, end in REQUEST_CHUNK_RANGES],
        },
        stream_padded_lens={"text": SEQ_LEN_TXT},
        chunk_to_request=chunk_to_request,
        device=image_chunks.device,
    )
    # The attention reads the plan and layout from the armed step, mirroring
    # the pipeline.
    with override_forward_context(ForwardContext(mixfusion_step=MixFusionStep(plan=plan, layout=layout))):
        return module.forward(
            hidden_states=image_chunks,
            encoder_hidden_states=encoder_hidden_states,
            vid_freqs=image_freq_chunks,
            txt_freqs=text_freqs,
        )


def _assert_rope_matches_eager_dtype_contract(rope: _CapturingRope):
    # The eager path (_qwen_image_qk_norm_rope) casts cos/sin to the activation
    # dtype; FP32 rotary numerics drop Omni-vs-Diffusers PSNR (#7494), so the
    # mixfusion path must not diverge from the default forward.
    assert rope.calls, "RoPE was never applied"
    for call in rope.calls:
        assert call["tensor_dtype"] == torch.bfloat16
        assert call["cos_dtype"] == torch.bfloat16
        assert call["sin_dtype"] == torch.bfloat16


def _assert_output_shapes(img_out, txt_out):
    assert img_out.shape == (NUM_CHUNKS, CHUNK_SIZE, Q_SIZE)
    assert txt_out.shape == (len(REQ_LENS), SEQ_LEN_TXT, Q_SIZE)
    # Text rows beyond a request's real length are zero-padded, matching the
    # mask-unpad path used by the non-mixfusion forward.
    assert torch.all(txt_out[0, REAL_TXT_LENS[0] :] == 0)
    assert torch.all(txt_out[1, REAL_TXT_LENS[1] :] == 0)


def test_packed_attention_packs_mixed_requests_into_one_packed_call():
    attn = _PackedAttentionStub(mode="varlen")
    rope = _CapturingRope()
    module = _mixfusion_module(attn=attn, rope=rope)

    img_out, txt_out = _forward(module)

    # One forward_packed call with per-request boundaries.
    assert len(attn.calls) == 1
    call = attn.calls[0]
    query, key, value = call["query"], call["key"], call["value"]
    total_tokens = sum(REQ_LENS)
    assert query.shape == (total_tokens, HEADS, HEAD_DIM)
    assert key.shape == (total_tokens, KV_HEADS, HEAD_DIM)
    assert value.shape == (total_tokens, KV_HEADS, HEAD_DIM)
    assert torch.equal(call["cu_seqlens"], torch.tensor(CU_SEQLENS, dtype=torch.int32))
    assert call["max_seqlen"] == max(REQ_LENS)

    # Each request's tokens are contiguous in "text, then image chunks" order.
    # Identity projections make the packed rows equal to the input rows.
    image_chunks, encoder_hidden_states, *_ = _mixfusion_inputs()

    def _txt_rows(req_idx):
        return encoder_hidden_states[req_idx, : REAL_TXT_LENS[req_idx], :Q_SIZE].unflatten(-1, (HEADS, HEAD_DIM))

    def _img_rows(chunk_start, chunk_end):
        return image_chunks[chunk_start:chunk_end].reshape(-1, QKV_DIM)[:, :Q_SIZE].unflatten(-1, (HEADS, HEAD_DIM))

    offset = 0
    for req_idx, (chunk_start, chunk_end) in enumerate(REQUEST_CHUNK_RANGES):
        txt_len = REAL_TXT_LENS[req_idx]
        img_len = (chunk_end - chunk_start) * CHUNK_SIZE
        assert torch.equal(query[offset : offset + txt_len], _txt_rows(req_idx))
        assert torch.equal(query[offset + txt_len : offset + txt_len + img_len], _img_rows(chunk_start, chunk_end))
        offset += txt_len + img_len
    assert offset == total_tokens

    _assert_rope_matches_eager_dtype_contract(rope)
    _assert_output_shapes(img_out, txt_out)
    # The recovered rows come from the packed result: the varlen stub writes
    # q + per-document K/V means, so check against the same reference.
    flat = _segment_mean_flat(query, call["key"], call["value"], call["cu_seqlens"])
    # Text rows are recovered in request order: req0's text opens the packed
    # layout, req1's text opens at its own request offset.
    req0_txt = flat[: REAL_TXT_LENS[0]]
    req1_txt = flat[REQ_LENS[0] : REQ_LENS[0] + REAL_TXT_LENS[1]]
    assert torch.equal(txt_out[0, : REAL_TXT_LENS[0]], req0_txt.flatten(1, 2))
    assert torch.equal(txt_out[1, : REAL_TXT_LENS[1]], req1_txt.flatten(1, 2))
    # req0's image chunks sit right after its text rows in the packed layout.
    assert torch.equal(img_out[0], flat[REAL_TXT_LENS[0] : REAL_TXT_LENS[0] + CHUNK_SIZE].flatten(1, 2))


def test_forward_mixfusion_bucketed_fallback_matches_the_varlen_result():
    # The model no longer picks a route itself: the same forward_packed call
    # must serve backends without packed varlen support through the layer's
    # bucketed dense fallback, and the recovered outputs must match what a
    # single varlen kernel over the same packing would produce.
    varlen_attn = _PackedAttentionStub(mode="varlen")
    varlen_module = _mixfusion_module(attn=varlen_attn, rope=_CapturingRope())
    varlen_img_out, varlen_txt_out = _forward(varlen_module)

    bucketed_attn = _PackedAttentionStub(mode="bucketed")
    rope = _CapturingRope()
    bucketed_module = _mixfusion_module(attn=bucketed_attn, rope=rope)
    img_out, txt_out = _forward(bucketed_module)

    # Same single packed call; equal-length documents would share a bucket,
    # but 11 and 17 differ so each dense forward holds one request.
    assert len(bucketed_attn.calls) == 1
    assert sorted(bucketed_attn.dense_shapes) == [
        (1, REQ_LENS[0], HEADS, HEAD_DIM),
        (1, REQ_LENS[1], HEADS, HEAD_DIM),
    ]
    assert torch.equal(img_out, varlen_img_out)
    assert torch.equal(txt_out, varlen_txt_out)

    _assert_rope_matches_eager_dtype_contract(rope)
    _assert_output_shapes(img_out, txt_out)
