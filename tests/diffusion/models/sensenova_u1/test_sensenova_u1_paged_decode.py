# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the paged decode cache and its CUDA graph.

The cache exists so a decode step has static shapes and can be captured. That
puts the burden on two things a normal test would not look at: the prefix must
survive the hand-off from the ordinary cache, and every quantity that varies
between steps must live in a device tensor. A Python-level index is correct
eagerly and silently wrong once captured, because capture bakes it in -- which
is exactly what the last test here pins.
"""

import pytest
import torch

import vllm_omni.diffusion.models.sensenova_u1.paged_decode as paged_decode
from vllm_omni.diffusion.models.sensenova_u1.paged_decode import (
    BLOCK_SIZE,
    PagedDecodeCache,
    _bucket_for,
)
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1Attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

LAYERS, KV_HEADS, HEAD_DIM = 2, 4, 8
N_HEADS = KV_HEADS * 2


class _Layer:
    def __init__(self, keys=None, values=None):
        self.keys, self.values = keys, values


class _Cache:
    """Stands in for the transformers cache: layers with [B, H, S, D] tensors."""

    def __init__(self, layers):
        self.layers = layers


def _dyn_cache(prefix):
    torch.manual_seed(0)
    return _Cache(
        [
            _Layer(torch.randn(1, KV_HEADS, prefix, HEAD_DIM), torch.randn(1, KV_HEADS, prefix, HEAD_DIM))
            for _ in range(LAYERS)
        ]
    )


def test_bucket_rounds_up_and_is_block_aligned():
    assert _bucket_for(1) == 512
    assert _bucket_for(512) == 512
    assert _bucket_for(513) == 1024
    assert _bucket_for(100_000) % BLOCK_SIZE == 0


def test_prefix_survives_the_hand_off():
    dyn = _dyn_cache(37)
    paged = PagedDecodeCache.from_dynamic_cache(dyn, LAYERS, torch.device("cpu"), torch.float32)
    assert paged.length == 37
    for i in range(LAYERS):
        stored = paged.k[i].view(-1, KV_HEADS, HEAD_DIM)[:37]
        torch.testing.assert_close(stored, dyn.layers[i].keys[0].transpose(0, 1))


def test_write_back_restores_what_was_handed_over():
    """Decode is paged but the stage after it reads the ordinary cache."""
    dyn = _dyn_cache(21)
    original = [layer.keys.clone() for layer in dyn.layers]
    paged = PagedDecodeCache.from_dynamic_cache(dyn, LAYERS, torch.device("cpu"), torch.float32)
    paged.to_dynamic_cache(dyn)
    for before, layer in zip(original, dyn.layers):
        torch.testing.assert_close(layer.keys, before)


def test_growing_keeps_the_tokens_already_stored():
    dyn = _dyn_cache(500)
    paged = PagedDecodeCache.from_dynamic_cache(dyn, LAYERS, torch.device("cpu"), torch.float32)
    assert paged.bucket == 512
    before = [k.view(-1, KV_HEADS, HEAD_DIM)[:500].clone() for k in paged.k]
    gen = paged.generation
    assert paged.grow(513) is True
    assert paged.bucket == 1024
    assert paged.generation > gen, "a reallocation must invalidate captured graphs"
    for old, new in zip(before, paged.k):
        torch.testing.assert_close(new.view(-1, KV_HEADS, HEAD_DIM)[:500], old)


def test_length_lives_in_device_tensors():
    """Both the attended length and the write slot must be tensors.

    A captured graph reads them at replay; a Python int would be frozen at
    capture time and every replay would attend the wrong span and overwrite the
    same slot.
    """
    dyn = _dyn_cache(10)
    paged = PagedDecodeCache.from_dynamic_cache(dyn, LAYERS, torch.device("cpu"), torch.float32)
    paged.set_length(15)
    assert torch.is_tensor(paged.seqused) and int(paged.seqused) == 15
    assert torch.is_tensor(paged.pos) and int(paged.pos) == 14


# ---------------------------------------------------------------------------
# The capture contract. These need CUDA and the bundled flash-attention kernel.
# ---------------------------------------------------------------------------

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda_only
@pytest.mark.cuda
@pytest.mark.L4
def test_paged_attention_matches_sdpa_over_the_same_buffer():
    import torch.nn.functional as F

    from vllm_omni.diffusion.models.sensenova_u1.paged_decode import PagedDecodeCache

    heads, kv_heads, dim, n = 8, 2, 64, 100
    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)
    dyn = _Cache(
        [
            _Layer(
                torch.randn(1, kv_heads, n, dim, device=dev, dtype=dt),
                torch.randn(1, kv_heads, n, dim, device=dev, dtype=dt),
            )
        ]
    )
    paged = PagedDecodeCache.from_dynamic_cache(dyn, 1, dev, dt)
    paged.set_length(n)
    q = torch.randn(1, heads, 1, dim, device=dev, dtype=dt)
    k = torch.zeros(1, kv_heads, 1, dim, device=dev, dtype=dt)
    scale = dim**-0.5
    # write the last slot back as itself so attend() is a pure read
    k[0, :, 0] = dyn.layers[0].keys[0, :, n - 1]
    v = torch.zeros_like(k)
    v[0, :, 0] = dyn.layers[0].values[0, :, n - 1]
    got = paged.attend(0, q, k, v, scale)
    flat_k = paged.k[0].view(-1, kv_heads, dim)[:n]
    flat_v = paged.v[0].view(-1, kv_heads, dim)[:n]
    want = F.scaled_dot_product_attention(
        q,
        flat_k.unsqueeze(0).transpose(1, 2).contiguous(),
        flat_v.unsqueeze(0).transpose(1, 2).contiguous(),
        enable_gqa=True,
        scale=scale,
    ).transpose(1, 2)
    torch.testing.assert_close(got.float(), want.float(), atol=2e-3, rtol=2e-3)


@cuda_only
@pytest.mark.cuda
@pytest.mark.L4
def test_a_captured_graph_follows_a_later_length():
    """One capture has to serve every length in the bucket.

    If the attended span were a Python value the graph would keep attending the
    capture-time length, and the run would still look fast.
    """
    import torch.nn.functional as F

    from vllm_omni.diffusion.models.sensenova_u1.paged_decode import PagedDecodeCache

    heads, kv_heads, dim, n = 8, 2, 64, 200
    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)
    dyn = _Cache(
        [
            _Layer(
                torch.randn(1, kv_heads, n, dim, device=dev, dtype=dt),
                torch.randn(1, kv_heads, n, dim, device=dev, dtype=dt),
            )
        ]
    )
    paged = PagedDecodeCache.from_dynamic_cache(dyn, 1, dev, dt)
    q = torch.randn(1, heads, 1, dim, device=dev, dtype=dt)
    k = torch.zeros(1, kv_heads, 1, dim, device=dev, dtype=dt)
    v = torch.zeros_like(k)
    scale = dim**-0.5

    paged.set_length(n)
    k[0, :, 0] = dyn.layers[0].keys[0, :, n - 1]
    v[0, :, 0] = dyn.layers[0].values[0, :, n - 1]

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            paged.attend(0, q, k, v, scale)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged.attend(0, q, k, v, scale)

    shorter = n - 43
    paged.set_length(shorter)
    graph.replay()
    torch.accelerator.synchronize()

    flat_k = paged.k[0].view(-1, kv_heads, dim)[:shorter]
    flat_v = paged.v[0].view(-1, kv_heads, dim)[:shorter]
    want = F.scaled_dot_product_attention(
        q,
        flat_k.unsqueeze(0).transpose(1, 2).contiguous(),
        flat_v.unsqueeze(0).transpose(1, 2).contiguous(),
        enable_gqa=True,
        scale=scale,
    ).transpose(1, 2)
    torch.testing.assert_close(out.float(), want.float(), atol=2e-3, rtol=2e-3)


# ---------------------------------------------------------------------------
# The cache is only worth anything if production actually reaches it, and the
# kwargs it needs are not in every wheel that exports the kernel. The tests
# above would all still pass with the `forward_und` branch deleted, so these
# drive the real dispatch instead of the cache object.
# ---------------------------------------------------------------------------


class _PagedProbe:
    def __init__(self, out):
        self.out = out
        self.calls: list[tuple[int, torch.Size, float]] = []

    def attend(self, layer_idx, query, key, value, scaling):
        self.calls.append((layer_idx, query.shape, scaling))
        return self.out


class _AttnHost:
    """Carries only what ``forward_und`` touches on the way to the branch."""

    forward_und = SenseNovaU1Attention.forward_und
    qkv_proj = q_norm = k_norm = q_norm_hw = k_norm_hw = None
    layer_idx = 3
    scaling = HEAD_DIM**-0.5

    def __init__(self, seq, paged_out):
        self._qkv = (
            torch.randn(1, N_HEADS, seq, HEAD_DIM),
            torch.randn(1, KV_HEADS, seq, HEAD_DIM),
            torch.randn(1, KV_HEADS, seq, HEAD_DIM),
        )
        self.paged = _PagedProbe(paged_out)
        self.sdpa_calls: list[torch.Tensor | None] = []

    def _project_and_rope(self, *args, **kwargs):
        return self._qkv

    def _run_attn(self, query, key, value, mask):
        self.sdpa_calls.append(mask)
        return torch.zeros(1, N_HEADS, query.shape[2], HEAD_DIM)

    def o_proj(self, hidden_states):
        return hidden_states * 2, None


def test_forward_und_sends_a_single_token_decode_to_the_paged_cache():
    """Goes through the real dispatch, so deleting the branch turns this red."""
    out = torch.randn(1, N_HEADS, 1, HEAD_DIM)
    host = _AttnHost(seq=1, paged_out=out)
    got = host.forward_und(torch.zeros(1, 1, 16), None, None, paged_cache=host.paged)
    assert len(host.paged.calls) == 1, "the paged hook was never reached"
    assert host.sdpa_calls == [], "the same step also ran the SDPA path"
    assert host.paged.calls[0][0] == host.layer_idx
    torch.testing.assert_close(got, out.reshape(1, 1, -1).contiguous() * 2)


@pytest.mark.parametrize(
    "seq,mask",
    [(2, None), (1, torch.zeros(1, 1, 1, 1))],
    ids=["multi_token", "masked"],
)
def test_only_an_unmasked_single_token_step_takes_the_paged_path(seq, mask):
    """Both other shapes are uncapturable, and the cache holds one row per step."""
    host = _AttnHost(seq=seq, paged_out=torch.zeros(1, N_HEADS, seq, HEAD_DIM))
    host.forward_und(torch.zeros(1, seq, 16), None, mask, paged_cache=host.paged)
    assert host.paged.calls == [], "a step that cannot be captured reached the paged cache"
    assert len(host.sdpa_calls) == 1


def test_a_kernel_without_the_paged_kwargs_is_not_supported(monkeypatch):
    """An older wheel exports the same name without ``seqused_k``/``block_table``.
    Probing the signature is what keeps that install on SDPA instead of raising
    TypeError at the first decode step."""

    def old_kernel(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, causal=False):
        raise AssertionError("must not be called")  # pragma: no cover

    monkeypatch.setattr(paged_decode, "_flash_varlen", lambda: old_kernel)
    assert paged_decode.paged_decode_supported(torch.device("cuda"), HEAD_DIM) is False

    def current_kernel(*args, seqused_k=None, block_table=None, **kwargs):
        raise AssertionError("must not be called")  # pragma: no cover

    monkeypatch.setattr(paged_decode, "_flash_varlen", lambda: current_kernel)
    assert paged_decode.paged_decode_supported(torch.device("cuda"), HEAD_DIM) is True


def test_the_kill_switch_returns_no_cache(monkeypatch):
    """The recipe documents `VLLM_OMNI_SENSENOVA_PAGED_DECODE=0` as the way back
    to the ordinary cache, so the switch needs a test rather than only prose."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        SenseNovaU1Pipeline,
    )

    monkeypatch.setenv("VLLM_OMNI_SENSENOVA_PAGED_DECODE", "0")
    # The env check returns before anything else on the pipeline is touched.
    host = object.__new__(SenseNovaU1Pipeline)
    assert SenseNovaU1Pipeline._decode_context(host, object()) is None


class _Tok:
    """The two tokenizer calls ``_generate_text`` makes."""

    EOS = 7

    def convert_tokens_to_ids(self, token):
        return self.EOS

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


class _Out:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class _TextHost:
    """Carries only what ``_generate_text`` touches."""

    def __init__(self, context):
        self._context = context
        self.seen: list[object] = []
        self.tokenizer = _Tok()

    def _decode_context(self, past_key_values):
        return self._context

    def _ar_step(self, next_token, t_idx, past_key_values, decode=None):
        self.seen.append(decode)
        # Stop the loop: argmax of this row is the EOS id.
        logits = torch.full((1, 1, 8), -1.0)
        logits[0, 0, _Tok.EOS] = 1.0
        return _Out(logits, past_key_values)


def test_text_decoding_uses_the_paged_context_too():
    """Image-to-text and text-to-text run their own decode loop. It used to call
    ``_ar_step`` without the context, so the paged path was documented but never
    reached outside think."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        SenseNovaU1Pipeline,
    )

    context = object()
    host = _TextHost(context)
    prefix = torch.full((1, 1, 8), -1.0)
    prefix[0, 0, 5] = 1.0  # not EOS, so one step runs
    SenseNovaU1Pipeline._generate_text(host, prefix, past_key_values=None, t_idx=0)
    assert host.seen == [context], f"text decoding ran with decode={host.seen}"


class _WarmupReq:
    """The engine's dummy request, which carries the reserved id."""

    def __init__(self, request_id):
        self.request_id = request_id
        self.prompts = [{"prompt": "", "modalities": ["image"]}]


def test_the_dummy_warmup_request_drives_one_decode_step(monkeypatch):
    """The dummy request is a text-to-image run with think off, so it exercises
    the prefill shape and never the decode one. Without this hook whatever the
    deploy config compiles for decode lands on the first real request."""
    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
        SenseNovaU1Pipeline,
    )
    from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID

    calls: list[str] = []
    host = object.__new__(SenseNovaU1Pipeline)
    monkeypatch.setattr(host, "_warm_ar_decode", lambda: calls.append("warm"), raising=False)
    monkeypatch.setattr(
        SenseNovaU1Pipeline,
        "_parse_request",
        lambda self, req: (_ for _ in ()).throw(_StopForwardError()),
        raising=False,
    )

    with pytest.raises(_StopForwardError):
        SenseNovaU1Pipeline.forward(host, _WarmupReq(DUMMY_DIFFUSION_REQUEST_ID))
    assert calls == ["warm"], "the dummy warmup request did not drive a decode step"

    calls.clear()
    with pytest.raises(_StopForwardError):
        SenseNovaU1Pipeline.forward(host, _WarmupReq("a-real-request"))
    assert calls == [], "a real request paid for the warmup"


class _StopForwardError(Exception):
    """Stops ``forward`` right after the warmup hook, so the test needs no model."""
