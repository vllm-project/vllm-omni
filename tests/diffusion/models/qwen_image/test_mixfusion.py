# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side tests for Qwen-Image's declarative MixFusion plan (CPU).

The generic driver, hooks, chunked adapter and layout primitives are covered
under ``tests/diffusion/mixfusion`` and ``tests/diffusion/hooks``. This module
pins what is Qwen-specific:

* declared input requirements and that the real ``_mixfusion_plan`` exposes the
  block boundaries to the foreign-hook guard;
* the pipeline's dense keyword builder feeding a packed list through an armed
  step, with and without sequential true-CFG;
* the real ``ImageRopePrepare`` boundary under the plan: per-request RoPE
  identical to individual execution, chunk-stacked;
* the packed cross-attention: the model checks for an armed MixFusion step,
  constructs metadata with ``MIXFUSION_PLAN_KEY``, and the attention layer
  dispatches to ``run_mixfusion_packed_attention``.
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
from vllm_omni.diffusion.hooks.mixfusion import apply_mixfusion, plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.chunked import JOINT_TEXT_IMAGE_LAYOUT, ChunkedJointAdapter
from vllm_omni.diffusion.mixfusion.driver import MixFusionDriver
from vllm_omni.diffusion.mixfusion.execution_plan import build_execution_plan
from vllm_omni.diffusion.mixfusion.packed_attention import (
    MIXFUSION_PLAN_KEY,
    run_mixfusion_packed_attention,
)
from vllm_omni.diffusion.mixfusion.plan import MixFusionRoot
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionRequestInfo
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    ImageRopePrepare,
    QwenEmbedRope,
    QwenImageCrossAttention,
    QwenImageTransformer2DModel,
    _reference_qk_norm_rope,
)
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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


def test_validate_inputs_rejects_reference_images_and_missing_shapes():
    transformer, _ = _transformer()
    adapter = ChunkedJointAdapter(QwenImageTransformer2DModel._mixfusion_plan[""])
    batch = SimpleNamespace(img_shapes=[(16, 2, 2)], txt_seq_lens=[4], image_latents=None)
    adapter.validate_inputs(transformer, batch)
    batch.img_shapes = None
    with pytest.raises(ValueError, match="missing img_shapes"):
        adapter.validate_inputs(transformer, batch)
    batch.img_shapes = [(16, 2, 2)]
    batch.image_latents = torch.zeros(1, 4, 8)
    with pytest.raises(ValueError, match="unsupported image_latents"):
        adapter.validate_inputs(transformer, batch)


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
        request_ids=["a", "b"],
        use_mixfusion=True,
        states=[
            StepRequestState(
                request_id=rid, sampling=OmniDiffusionSamplingParams(), mixfusion_info=MixFusionRequestInfo(1, 8, 1, 8)
            )
            for rid in ("a", "b")
        ],
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
    driver = MixFusionDriver(pipeline, transformer, ChunkedJointAdapter(MixFusionRoot()))
    pipeline.mixfusion_driver = driver
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
# The declared ImageRopePrepare boundary under an armed step.
# ---------------------------------------------------------------------------


class _QwenRopeOnly(nn.Module):
    """The real Qwen plan's root and RoPE boundaries around a recording body."""

    _mixfusion_plan = {
        "": QwenImageTransformer2DModel._mixfusion_plan[""],
        "image_rope_prepare": QwenImageTransformer2DModel._mixfusion_plan["image_rope_prepare"],
    }

    def __init__(self):
        super().__init__()
        self.image_rope_prepare = ImageRopePrepare(
            nn.Linear(4, 6), QwenEmbedRope(theta=10000, axes_dim=[4, 6, 6], scale_rope=True)
        )
        self.seen: list[tuple] = []

    def forward(
        self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask=None, img_shapes=None, txt_seq_lens=None
    ):
        hidden, vid_freqs, txt_freqs = self.image_rope_prepare(hidden_states, img_shapes, txt_seq_lens)
        self.seen.append((hidden, vid_freqs, txt_freqs))
        return SimpleNamespace(sample=hidden)


def test_rope_boundary_matches_individual_execution_per_request():
    torch.manual_seed(0)
    model = _QwenRopeOnly()
    apply_mixfusion(model, model._mixfusion_plan)
    # Two images for request 0 (one prompt row each), one for request 1, and
    # a 2x4 vs 4x2 aspect pair so a shared table would be wrong.
    grids = [(1, 2, 4), (1, 4, 2)]
    latents = [torch.randn(2, 8, 4), torch.randn(1, 8, 4)]
    text = torch.zeros(3, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    kwargs = dict(
        encoder_hidden_states=text,
        encoder_hidden_states_mask=mask,
        img_shapes=[[grid] for grid in grids],
        txt_seq_lens=[5, 5, 5],
    )
    with override_forward_context(ForwardContext(mixfusion_step=MixFusionStep())):
        out = model(hidden_states=latents, **kwargs)
        plan = get_forward_context().mixfusion_step.plan

    packed_hidden, packed_vid, packed_txt = model.seen[0]
    assert plan.chunk_size == 8 and plan.num_chunks == 3
    assert plan.request_stream_lens == {"text": (3, 3, 5), "image": (8, 8, 8)}
    rows = [(0, 0), (0, 1), (1, 0)]
    for chunk, (req, image) in enumerate(rows):
        hidden_ref, vid_ref, txt_ref = model.image_rope_prepare(latents[req][image : image + 1], [[grids[req]]], [5])
        assert torch.equal(packed_hidden[chunk], hidden_ref[0])
        assert torch.equal(packed_vid[chunk], vid_ref)
        assert torch.equal(packed_txt[chunk], txt_ref[:5])
    # The root reassembles one [num_images, S, C] prediction per request.
    assert [tuple(o.shape) for o in out] == [(2, 8, 6), (1, 8, 6)]


def test_root_reuses_the_plan_while_the_layout_is_unchanged():
    model = _QwenRopeOnly()
    apply_mixfusion(model, model._mixfusion_plan)
    kwargs = dict(
        encoder_hidden_states=torch.zeros(2, 4, 8), img_shapes=[[(1, 2, 4)], [(1, 4, 4)]], txt_seq_lens=[4, 4]
    )
    plans = []
    for latents in ([torch.ones(1, 8, 4), torch.ones(1, 16, 4)],) * 2 + ([torch.ones(1, 8, 4)],):
        shapes = (
            kwargs
            if len(latents) == 2
            else {
                **kwargs,
                "encoder_hidden_states": torch.zeros(1, 4, 8),
                "img_shapes": [[(1, 2, 4)]],
                "txt_seq_lens": [4],
            }
        )
        with override_forward_context(ForwardContext(mixfusion_step=MixFusionStep())):
            model(hidden_states=latents, **shapes)
            plans.append(get_forward_context().mixfusion_step.plan)
    assert plans[1] is plans[0]
    assert plans[2] is not plans[0] and plans[2].num_chunks == 1


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

    def effective_attention(self):
        return self, self.attn_backend, self.attn_spec

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
        "reference_precision": False,
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
        layout=JOINT_TEXT_IMAGE_LAYOUT,
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


def _assert_common(rope, img_out, txt_out, *, txt_lens, num_chunks, txt_padded, reference_precision=False):
    # The eager path casts cos/sin to the activation dtype; FP32 rotary
    # numerics drop Omni-vs-Diffusers PSNR (#7494).
    if reference_precision:
        assert not rope.dtypes  # The reference uses FP32 complex rotation.
    else:
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


@pytest.mark.parametrize("reference_precision", [False, True])
def test_mask_capable_dense_backend_matches_the_varlen_result(reference_precision):
    # Three requests share one image shape but differ in text length: a
    # mask-capable dense backend fuses them into one masked forward.
    txt_lens = [3, 5, 2]
    shapes = {"txt_lens": txt_lens, "chunks_per_request": [2, 2, 2], "txt_padded": 5}
    varlen_module = _mixfusion_module(_PackedAttentionStub(varlen=True), _CapturingRope())
    object.__setattr__(varlen_module, "reference_precision", reference_precision)
    _, _, varlen_img, varlen_txt = _forward(varlen_module, **shapes)
    attn, rope = _PackedAttentionStub(varlen=False), _CapturingRope()
    dense_module = _mixfusion_module(attn, rope)
    object.__setattr__(dense_module, "reference_precision", reference_precision)
    _, _, img_out, txt_out = _forward(dense_module, **shapes)

    assert attn.packed_calls == []
    assert len(attn.dense_calls) == 1
    image_tokens = 2 * CHUNK_SIZE
    assert attn.dense_calls[0]["shape"] == (3, 5 + image_tokens, HEADS, HEAD_DIM)
    # True on each request's real text and its whole image stream.
    mask = attn.dense_calls[0]["mask"]
    for row, text_len in enumerate(txt_lens):
        expected = torch.zeros(5 + image_tokens, dtype=torch.bool)
        expected[:text_len] = True
        expected[5:] = True
        assert torch.equal(mask[row], expected)

    # bf16 reduction order differs between the routes; a padding leak would
    # move the K/V mean far outside these tolerances.
    torch.testing.assert_close(txt_out, varlen_txt, atol=1e-2, rtol=2e-2)
    torch.testing.assert_close(img_out, varlen_img, atol=1e-2, rtol=2e-2)
    _assert_common(
        rope,
        img_out,
        txt_out,
        txt_lens=shapes["txt_lens"],
        num_chunks=6,
        txt_padded=5,
        reference_precision=reference_precision,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_reference_rope_per_row_matches_independent_requests(dtype):
    """Chunk rows retain the reference arithmetic and their own RoPE positions."""
    gen = torch.Generator().manual_seed(22)
    q = torch.randn(3, 5, HEADS, HEAD_DIM, generator=gen).to(dtype)
    k = torch.randn(3, 5, KV_HEADS, HEAD_DIM, generator=gen).to(dtype)
    freqs = torch.randn(3, 5, ROT_DIM, generator=gen, dtype=torch.complex64)
    norm = nn.RMSNorm(HEAD_DIM).to(dtype)
    actual_q, actual_k = _reference_qk_norm_rope(q, k, norm, norm, freqs)
    expected = [_reference_qk_norm_rope(q[i : i + 1], k[i : i + 1], norm, norm, freqs[i]) for i in range(3)]
    # CPU complex multiplication can round one FP32 ulp differently for separate rows.
    tolerance = {"rtol": 2e-6, "atol": 2e-7} if dtype == torch.float32 else {"rtol": 0, "atol": 0}
    torch.testing.assert_close(actual_q, torch.cat([row[0] for row in expected]), **tolerance)
    torch.testing.assert_close(actual_k, torch.cat([row[1] for row in expected]), **tolerance)
