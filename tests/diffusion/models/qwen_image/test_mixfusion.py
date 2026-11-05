# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side tests for the Qwen-Image MixFusion adapter (CPU).

The generic driver, hooks and layout primitives are covered under
``tests/diffusion/mixfusion`` and ``tests/diffusion/hooks``. This module pins
what is Qwen-specific:

* the adapter's own admission limits and that the real ``_mixfusion_plan``
  exposes the block boundaries to the foreign-hook guard;
* the pipeline's dense keyword builder feeding a packed list through an armed
  step, with and without sequential true-CFG;
* the serial-ragged fallback keeping each request's spatial RoPE when
  equal-area images differ in aspect ratio;
* ``prepare_inputs``: row expansion, chunk split, RoPE tables, plan and the
  step-invariant cache;
* the packed cross-attention: one ``forward_packed`` call over mixed requests
  that keeps the eager RoPE dtype contract (#7494), and the mask-capable dense
  route matching it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    MixFusionStep,
    get_forward_context,
    override_forward_context,
)
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.diffusion.hooks.mixfusion import plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.mixfusion.execution_plan import build_execution_plan
from vllm_omni.diffusion.mixfusion.packed_attention import (
    MIXFUSION_PLAN_KEY,
    run_mixfusion_packed_attention,
)
from vllm_omni.diffusion.models.qwen_image.mixfusion_adapter import (
    QWEN_ATTENTION_LAYOUT,
    QwenImageMixFusionAdapter,
)
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    ImageRopePrepare,
    QwenEmbedRope,
    QwenImageCrossAttention,
    QwenImageTransformer2DModel,
)
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

ROPE_DIM = 32


class _PlainBlock(nn.Module):
    def forward(self, x):
        return x


def _transformer(blocks: list[nn.Module] | None = None) -> tuple[QwenImageTransformer2DModel, list[int]]:
    """A transformer with only the adapter's dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed.
    ``img_in`` is an identity (prepared ``hidden_states`` equal the
    chunk-stacked latents) and ``pos_embed`` returns zero tables of the right
    shape; the returned counter records how often they were rebuilt.
    """
    transformer = QwenImageTransformer2DModel.__new__(QwenImageTransformer2DModel)
    object.__setattr__(transformer, "zero_cond_t", False)
    # ``plan_foreign_hooks`` expands ``*`` only over ``nn.ModuleList``.
    object.__setattr__(transformer, "transformer_blocks", nn.ModuleList([_PlainBlock()] if blocks is None else blocks))
    object.__setattr__(transformer, "img_in", nn.Identity())
    pos_embed_calls = [0]

    def pos_embed(img_shapes, txt_seq_lens, device):
        pos_embed_calls[0] += 1
        return torch.zeros(int(img_shapes[0]), ROPE_DIM), torch.zeros(int(txt_seq_lens[0]) + 4, ROPE_DIM)

    object.__setattr__(transformer, "pos_embed", pos_embed)
    return transformer, pos_embed_calls


# ---------------------------------------------------------------------------
# Adapter admission and the plan's foreign-hook coverage.
# ---------------------------------------------------------------------------


def test_can_pack_rejects_zero_cond_t_and_missing_shapes():
    adapter = QwenImageMixFusionAdapter()
    transformer, _ = _transformer()
    batch = SimpleNamespace(img_shapes=[(16, 2, 2)], txt_seq_lens=[4], image_latents=None)
    assert adapter.can_pack(transformer, batch) == (True, "")
    assert adapter.can_pack(transformer, SimpleNamespace(img_shapes=None, txt_seq_lens=[4], image_latents=None)) == (
        False,
        "img_shapes and txt_seq_lens are required",
    )
    edit_batch = SimpleNamespace(img_shapes=[(16, 2, 2)], txt_seq_lens=[4], image_latents=torch.zeros(1, 4, 8))
    assert adapter.can_pack(transformer, edit_batch) == (
        False,
        "image_latents (reference images) are not supported",
    )
    object.__setattr__(transformer, "zero_cond_t", True)
    assert adapter.can_pack(transformer, batch) == (False, "zero_cond_t/editing path is not supported")


def _offloaded_block() -> nn.Module:
    """A block with a registered hook, as installed by layerwise offload."""
    block = _PlainBlock()
    HookRegistry.get_or_create(block).register_hook("layerwise_offload", ModelHook())
    return block


def _hsdp_block() -> nn.Module:
    """A block with HSDP ``fully_shard``'s PyTorch-native pre-hook."""
    block = _PlainBlock()
    block.register_forward_pre_hook(lambda module, args: args)
    return block


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        (_PlainBlock, []),
        (_offloaded_block, ["transformer_blocks.*.img_mod:layerwise_offload"]),
        (_hsdp_block, ["transformer_blocks.*.img_mod:torch-native"]),
    ],
)
def test_real_plan_exposes_block_hooks_to_the_guard(block, expected):
    # The ``transformer_blocks.*.img_mod`` boundary must make hooks on the
    # enclosing block visible, so admission keeps those models serial.
    transformer, _ = _transformer([_PlainBlock(), block()])
    assert plan_foreign_hooks(transformer, QwenImageTransformer2DModel._mixfusion_plan) == expected


# ---------------------------------------------------------------------------
# The pipeline's dense keyword builder inside an armed packed step.
# ---------------------------------------------------------------------------


class _StubTransformer:
    """Records each call; returns one prediction per request like the root hook."""

    def __init__(self):
        self._mixfusion_plan = {}
        self.transformer_blocks = nn.ModuleList()
        self._mixfusion_hooks_applied = True
        self.calls: list[dict] = []

    def __call__(self, hidden_states, **kwargs):
        self.calls.append(
            {
                "hidden_states": hidden_states,
                "cfg_branch": getattr(self, "cfg_branch", None),
                "armed": get_forward_context().mixfusion_step is not None,
                **kwargs,
            }
        )
        return [torch.full_like(latent, float(len(self.calls))) for latent in hidden_states]

    forward = __call__


class _StubPipeline(CFGParallelMixin):
    """The pipeline surface ``denoise_step`` needs, without weights."""

    denoise_step = QwenImagePipeline.denoise_step
    _build_denoise_kwargs = QwenImagePipeline._build_denoise_kwargs

    def __init__(self, transformer):
        self.transformer = transformer
        self.interrupt = False
        self.attention_kwargs = None
        self.parallel_config = SimpleNamespace(sequence_parallel_size=1, cfg_parallel_size=1)


def _packed_input_batch(*, do_true_cfg: bool = False):
    return SimpleNamespace(
        latents=[torch.zeros(1, 8, 16), torch.zeros(1, 8, 16)],
        image_latents=None,
        # Lowering the minimum keeps this tiny fixture packable.
        request_extras=[{"mixfusion_min_chunk_tokens": 1}],
        timesteps=torch.tensor([500.0, 500.0]),
        guidance=None,
        prompt_embeds=torch.zeros(2, 4, 8),
        prompt_embeds_mask=None,
        img_shapes=[(16, 2, 2), (16, 2, 2)],
        txt_seq_lens=[4, 4],
        do_true_cfg=do_true_cfg,
        true_cfg_scale=4.0,
        cfg_normalize=False,
        # The negative prompt has its own text length, kept per request.
        negative_prompt_embeds=torch.ones(2, 3, 8) if do_true_cfg else None,
        negative_prompt_embeds_mask=None,
        negative_txt_seq_lens=[3, 3] if do_true_cfg else None,
    )


@pytest.mark.parametrize("do_true_cfg", [False, True])
def test_driver_runs_the_pipeline_denoise_step_packed(do_true_cfg):
    transformer = _StubTransformer()
    pipeline = _StubPipeline(transformer)
    driver = MixFusionDriver(pipeline, transformer, QwenImageMixFusionAdapter())
    batch = _packed_input_batch(do_true_cfg=do_true_cfg)

    with override_forward_context(ForwardContext()):
        result = driver.run(batch, states=[], dense_step=pipeline.denoise_step)
        assert get_forward_context().mixfusion_step is None

    branches = ["positive", "negative"] if do_true_cfg else ["positive"]
    assert [call["cfg_branch"] for call in transformer.calls] == branches
    for call in transformer.calls:
        assert call["armed"]
        # The keyword builder passes the per-request list through for the
        # root hook to pack, with one timestep per latent row.
        assert call["hidden_states"] is batch.latents
        assert call["img_shapes"] == batch.img_shapes
        assert call["timestep"].shape == (2,)
    # The CFG stamp never leaks past the helper.
    assert getattr(transformer, "cfg_branch", None) is None

    # The result stays one prediction per request; CFG combines per request.
    assert isinstance(result, list) and len(result) == len(batch.latents)
    expected = 2.0 + batch.true_cfg_scale * (1.0 - 2.0) if do_true_cfg else 1.0
    for out, latent in zip(result, batch.latents):
        assert torch.equal(out, torch.full_like(latent, expected))


# ---------------------------------------------------------------------------
# Serial-ragged fallback keeps each request's spatial RoPE.
# ---------------------------------------------------------------------------


def test_serial_fallback_keeps_each_requests_rope_for_equal_area_aspect_ratios():
    """A rejected batch's dense groups must not share one image geometry.

    2x4 and 4x2 token grids have the same sequence length, but the real
    ``ImageRopePrepare`` / ``QwenEmbedRope`` builds one table from
    ``img_shapes[0]`` per dense call, so grouping them would hand the second
    request the first request's spatial RoPE.
    """
    prepare = ImageRopePrepare(nn.Identity(), QwenEmbedRope(theta=10000, axes_dim=[4, 6, 6], scale_rope=True))
    grids = [(1, 2, 4), (1, 4, 2), (1, 3, 4)]
    states = [
        StepRequestState(
            request_id=f"req-{idx}",
            sampling=SimpleNamespace(true_cfg_scale=None, cfg_normalize=False),
            latents=torch.zeros(1, frame * height * width, 4),
            timesteps=torch.tensor([500.0]),
            img_shapes=[[(frame, height, width)]],
            txt_seq_lens=[3],
            extra={"use_ragged_latents": True},
        )
        for idx, (frame, height, width) in enumerate(grids)
    ]

    def dense_step(batch, **kwargs):
        # Same call the transformer makes: one RoPE table per dense batch.
        _, vid_freqs, _ = prepare(batch.latents, batch.img_shapes, batch.txt_seq_lens)
        return [vid_freqs for _ in range(batch.num_reqs)]

    # A chunk policy no batch can meet forces the serial fallback.
    batch = InputBatch.make_batch(states)
    for extra in batch.request_extras:
        extra["mixfusion_min_chunk_tokens"] = 1024
    driver = MixFusionDriver(_StubPipeline(_StubTransformer()), _StubTransformer(), QwenImageMixFusionAdapter())
    with override_forward_context(ForwardContext()):
        outputs = driver.run(batch, states, dense_step)

    for output, state in zip(outputs, states):
        expected, _ = prepare.pos_embed(state.img_shapes[0], state.txt_seq_lens, device=torch.device("cpu"))
        assert torch.equal(output, expected), state.request_id


# ---------------------------------------------------------------------------
# prepare_inputs: packed-layout preparation and its step-invariant cache.
# ---------------------------------------------------------------------------


def _prepare(transformer, latents, *, mask=None, cached=None):
    return QwenImageMixFusionAdapter().prepare_inputs(
        transformer,
        hidden_states=latents,
        encoder_hidden_states=torch.zeros(sum(latent.shape[0] for latent in latents), 4, 8),
        encoder_hidden_states_mask=mask,
        img_shapes=[(int(latent.shape[1]), 2, 2) for latent in latents],
        txt_seq_lens=[4] * len(latents),
        cached=cached,
    )


def test_prepare_splits_mixed_lengths_into_equal_chunks():
    transformer, pos_embed_calls = _transformer()
    latents = [torch.ones(1, 8, 2), torch.ones(1, 12, 2)]
    # An all-valid mask is normalized away: the plan sees full text lengths.
    prepared = _prepare(transformer, latents, mask=torch.ones(2, 4))

    # gcd(8, 12) = 4: 2 + 3 chunks stacked on the batch dimension; img_in is
    # an identity, so hidden_states are the chunk-stacked latents.
    assert prepared.plan.chunk_size == 4
    assert prepared.hidden_states.shape == (5, 4, 2)
    assert prepared.plan.chunk_to_request.tolist() == [0, 0, 1, 1, 1]
    assert prepared.plan.image_row_counts == (1, 1)
    assert prepared.plan.request_stream_lens == {"text": (4, 4), "image": (8, 12)}
    assert prepared.encoder_hidden_states_mask is None
    # One RoPE build per image row; tables stacked per chunk / per image row.
    assert pos_embed_calls[0] == 2
    assert prepared.image_rotary_emb[0].shape == (5, 4, ROPE_DIM)
    assert prepared.image_rotary_emb[1].shape == (2, 4, ROPE_DIM)


def test_prepare_reuses_the_step_invariant_layout():
    transformer, pos_embed_calls = _transformer()
    first = _prepare(transformer, [torch.ones(1, 8, 2), 2 * torch.ones(1, 8, 2)])
    step_2 = [3 * torch.ones(1, 8, 2), 4 * torch.ones(1, 8, 2)]
    second = _prepare(transformer, step_2, cached=first)

    # Same request shapes: no RoPE or plan rebuild, only the latents change.
    assert pos_embed_calls[0] == 2
    assert second.plan is first.plan
    assert second.image_rotary_emb is first.image_rotary_emb
    assert torch.equal(second.hidden_states, torch.cat(step_2, dim=0))

    # A different request mix must not reuse the cached layout.
    third = _prepare(transformer, [torch.ones(1, 8, 2)], cached=second)
    assert pos_embed_calls[0] == 3
    assert third.plan is not first.plan
    assert third.hidden_states.shape == (1, 8, 2)


# ---------------------------------------------------------------------------
# Packed cross-attention: one forward_packed call over mixed requests.
# ---------------------------------------------------------------------------

HEADS, KV_HEADS, HEAD_DIM = 2, 1, 4
Q_SIZE = HEADS * HEAD_DIM
QKV_DIM = Q_SIZE + 2 * KV_HEADS * HEAD_DIM
ROT_DIM = HEAD_DIM // 2
CHUNK_SIZE = 4


class _CapturingRope:
    """Stand-in for RotaryEmbedding that records the cos/sin dtype."""

    def __init__(self):
        self.dtypes: list[tuple[torch.dtype, ...]] = []

    def __call__(self, tensor, cos, sin):
        self.dtypes.append((tensor.dtype, cos.dtype, sin.dtype))
        return tensor


def _doc_semantics(query, key, value, valid=None):
    """Reference attention: each row gains its document's mean K/V.

    A cross-document or padding leak changes the means, so this makes routing
    differences observable without a real kernel. ``valid`` is the
    ``[B, L]`` key mask of a dense batch.
    """
    if valid is None:
        return query + key.mean(dim=-3, keepdim=True) + value.mean(dim=-3, keepdim=True)
    weights = valid.to(query.dtype)[:, :, None, None]
    total = weights.sum(dim=1, keepdim=True)
    return query + (key * weights).sum(dim=1, keepdim=True) / total + (value * weights).sum(dim=1, keepdim=True) / total


class _MaskCapableBackend:
    @classmethod
    def supports_attention_mask(cls, attention_spec=None):
        return True


class _PackedAttentionStub:
    """Attention stand-in: varlen (one ``forward_packed``) or mask-capable dense."""

    def __init__(self, *, varlen: bool):
        self.varlen = varlen
        self.packed_calls: list[dict] = []
        self.dense_calls: list[dict] = []
        self.attn_backend = None if varlen else _MaskCapableBackend
        self.attn_spec = None

    def __call__(self, query, key, value, attn_metadata):
        # Mirror the metadata-keyed dispatch at the top of ``Attention.forward``.
        assert attn_metadata.extra.get(MIXFUSION_PLAN_KEY) is not None
        return run_mixfusion_packed_attention(self, query, key, value, attn_metadata)

    def supports_multi_doc_packed_varlen(self, kv_cache_dtype=None):
        return self.varlen

    def forward(self, query, key, value, attn_metadata):
        self.dense_calls.append({"shape": tuple(query.shape), "mask": attn_metadata.attn_mask})
        return _doc_semantics(query, key, value, attn_metadata.attn_mask)

    def forward_packed(self, query, key, value, *, cu_seqlens, max_seqlen):
        self.packed_calls.append({"query": query, "cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen})
        bounds = cu_seqlens.tolist()
        return torch.cat(
            [_doc_semantics(query[a:b], key[a:b], value[a:b]) for a, b in zip(bounds, bounds[1:])],
            dim=0,
        )


def _mixfusion_module(attn: _PackedAttentionStub, rope: _CapturingRope):
    """A QwenImageCrossAttention with identity projections (rows are assertable)."""
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


def _forward(module, *, txt_lens, chunks_per_request, txt_padded):
    """Run the packed cross-attention over requests with the given shapes."""
    gen = torch.Generator().manual_seed(2026)
    num_chunks = sum(chunks_per_request)
    ranges, start = [], 0
    for count in chunks_per_request:
        ranges.append((start, start + count))
        start += count
    plan = build_execution_plan(
        layout=QWEN_ATTENTION_LAYOUT,
        chunk_size=CHUNK_SIZE,
        request_chunk_ranges=ranges,
        image_row_counts=[1] * len(ranges),
        request_stream_lens={"text": txt_lens, "image": [count * CHUNK_SIZE for count in chunks_per_request]},
        stream_padded_lens={"text": txt_padded},
        chunk_to_request=torch.tensor([idx for idx, count in enumerate(chunks_per_request) for _ in range(count)]),
        device=torch.device("cpu"),
    )
    image_chunks = torch.randn(num_chunks, CHUNK_SIZE, QKV_DIM, generator=gen).to(torch.bfloat16)
    text = torch.randn(len(ranges), txt_padded, QKV_DIM, generator=gen).to(torch.bfloat16)
    with override_forward_context(ForwardContext(mixfusion_step=MixFusionStep(plan=plan))):
        img_out, txt_out = module.forward(
            hidden_states=image_chunks,
            encoder_hidden_states=text,
            vid_freqs=torch.randn(num_chunks, CHUNK_SIZE, ROT_DIM, generator=gen, dtype=torch.complex64),
            txt_freqs=torch.randn(len(ranges), txt_padded, ROT_DIM, generator=gen, dtype=torch.complex64),
        )
    return image_chunks, text, img_out, txt_out


def _assert_common(rope, img_out, txt_out, *, txt_lens, num_chunks, txt_padded):
    # The eager path casts cos/sin to the activation dtype; FP32 rotary
    # numerics drop Omni-vs-Diffusers PSNR (#7494).
    assert rope.dtypes and set(rope.dtypes) == {(torch.bfloat16,) * 3}
    assert img_out.shape == (num_chunks, CHUNK_SIZE, Q_SIZE)
    assert txt_out.shape == (len(txt_lens), txt_padded, Q_SIZE)
    # Text rows beyond a request's real length stay zero-padded.
    for row, length in enumerate(txt_lens):
        assert torch.all(txt_out[row, length:] == 0)


def test_packed_attention_packs_mixed_requests_into_one_packed_call():
    # req0: 3 text + 2 chunks (8 tokens) -> 11; req1: 5 text + 3 chunks -> 17.
    txt_lens, chunks, padded = [3, 5], [2, 3], 7
    attn, rope = _PackedAttentionStub(varlen=True), _CapturingRope()
    image_chunks, text, img_out, txt_out = _forward(
        _mixfusion_module(attn, rope), txt_lens=txt_lens, chunks_per_request=chunks, txt_padded=padded
    )

    assert len(attn.packed_calls) == 1
    call = attn.packed_calls[0]
    assert torch.equal(call["cu_seqlens"], torch.tensor([0, 11, 28], dtype=torch.int32))
    assert call["max_seqlen"] == 17
    # Each request's rows are contiguous in "text, then image chunks" order.
    query = call["query"]
    assert query.shape == (28, HEADS, HEAD_DIM)

    def heads(rows):
        return rows[..., :Q_SIZE].unflatten(-1, (HEADS, HEAD_DIM))

    offset, chunk_start = 0, 0
    for req_idx, (txt_len, count) in enumerate(zip(txt_lens, chunks)):
        img_len = count * CHUNK_SIZE
        image_rows = image_chunks[chunk_start : chunk_start + count].reshape(-1, QKV_DIM)
        assert torch.equal(query[offset : offset + txt_len], heads(text[req_idx, :txt_len]))
        assert torch.equal(query[offset + txt_len : offset + txt_len + img_len], heads(image_rows))
        offset += txt_len + img_len
        chunk_start += count

    _assert_common(rope, img_out, txt_out, txt_lens=txt_lens, num_chunks=sum(chunks), txt_padded=padded)


def test_mask_capable_dense_backend_matches_the_varlen_result():
    # Three requests share one image shape but differ in text length: a
    # mask-capable dense backend fuses them into one masked forward.
    shapes = {"txt_lens": [3, 5, 2], "chunks_per_request": [2, 2, 2], "txt_padded": 5}
    varlen_module = _mixfusion_module(_PackedAttentionStub(varlen=True), _CapturingRope())
    _, _, varlen_img, varlen_txt = _forward(varlen_module, **shapes)
    attn, rope = _PackedAttentionStub(varlen=False), _CapturingRope()
    _, _, img_out, txt_out = _forward(_mixfusion_module(attn, rope), **shapes)

    assert attn.packed_calls == []
    assert len(attn.dense_calls) == 1
    image_tokens = 2 * CHUNK_SIZE
    assert attn.dense_calls[0]["shape"] == (3, 5 + image_tokens, HEADS, HEAD_DIM)
    # True on each request's real text and its whole image stream.
    mask = attn.dense_calls[0]["mask"]
    for row, text_len in enumerate(shapes["txt_lens"]):
        expected = torch.zeros(5 + image_tokens, dtype=torch.bool)
        expected[:text_len] = True
        expected[5:] = True
        assert torch.equal(mask[row], expected)

    # bf16 reduction order differs between the routes; a padding leak would
    # move the K/V mean far outside these tolerances.
    torch.testing.assert_close(txt_out, varlen_txt, atol=1e-2, rtol=2e-2)
    torch.testing.assert_close(img_out, varlen_img, atol=1e-2, rtol=2e-2)
    _assert_common(rope, img_out, txt_out, txt_lens=shapes["txt_lens"], num_chunks=6, txt_padded=5)
