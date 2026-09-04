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

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from vllm.lora.layers import BaseLayerWithLoRA

import vllm_omni.diffusion.models.sensenova_u1.paged_decode as paged_decode
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.sensenova_u1.paged_decode import (
    BLOCK_SIZE,
    BUCKETS,
    TAIL_STEP,
    PagedDecodeCache,
    _bucket_for,
)
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1Attention,
)
from vllm_omni.platforms import current_omni_platform

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


def _dyn_cache(prefix, device=None):
    torch.manual_seed(0)
    return _Cache(
        [
            _Layer(
                torch.randn(1, KV_HEADS, prefix, HEAD_DIM, device=device),
                torch.randn(1, KV_HEADS, prefix, HEAD_DIM, device=device),
            )
            for _ in range(LAYERS)
        ]
    )


def test_bucket_rounds_up_and_is_block_aligned():
    assert _bucket_for(1) == 512
    assert _bucket_for(512) == 512
    assert _bucket_for(513) == 1024
    assert _bucket_for(100_000) % BLOCK_SIZE == 0


def test_past_the_last_bucket_the_schedule_grows_in_steps():
    """Every bucket change reallocates the cache and re-captures the graph, so
    the tail of the schedule must not step by BLOCK_SIZE. A think edit with two
    2048x2048 inputs runs to about 9,100 tokens, so this regime is reachable,
    and a request decodes at most 1024 steps.
    """
    last = BUCKETS[-1]
    assert _bucket_for(last + 1) == last + TAIL_STEP
    assert _bucket_for(100_000) % BLOCK_SIZE == 0
    for start in (last + 1, last + TAIL_STEP - 1, last + TAIL_STEP + 7):
        bucket, grows = _bucket_for(start), 0
        for n in range(start, start + 1025):
            if _bucket_for(n) != bucket:
                bucket = _bucket_for(n)
                grows += 1
        assert grows <= 1, f"start {start}: {grows} reallocations over 1024 decode steps"


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


def test_a_failed_grow_leaves_the_cache_on_its_old_buffers(monkeypatch):
    """A half-applied resize would strand a capture on freed storage.

    ``generation`` is half of the key a captured graph is stored under, so
    publishing new K without publishing V, the block table, the bucket and the
    generation leaves the old key selecting a graph whose recorded addresses
    have just been freed. A later request replays it. Injecting a failure into
    the last allocation is the cheapest way to reach that window.
    """
    dyn = _dyn_cache(500)
    paged = PagedDecodeCache.from_dynamic_cache(dyn, LAYERS, torch.device("cpu"), torch.float32)
    before_k, before_v = list(paged.k), list(paged.v)
    before_table, before_bucket = paged.block_table, paged.bucket
    before_generation, before_length = paged.generation, paged.length

    real_zeros = torch.zeros
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2 * LAYERS:  # the last of the two allocations per layer
            raise RuntimeError("injected allocation failure")
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", flaky)
    with pytest.raises(RuntimeError, match="injected"):
        paged.grow(513)
    monkeypatch.undo()

    assert [id(t) for t in paged.k] == [id(t) for t in before_k], "K was published on its own"
    assert [id(t) for t in paged.v] == [id(t) for t in before_v], "V was published on its own"
    assert paged.block_table is before_table
    assert paged.bucket == before_bucket
    assert paged.generation == before_generation, "the graph key moved without new buffers"
    assert paged.length == before_length


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

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available() or not paged_decode.paged_decode_supported(torch.device("cuda"), 64),
    reason="CUDA with bundled paged flash-attention required",
)


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
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
@hardware_test(res={"cuda": "L4"}, num_cards=1)
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


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_a_captured_graph_writes_each_step_to_its_own_slot():
    """The write slot has to be a device tensor, not a Python index.

    ``test_a_captured_graph_follows_a_later_length`` pins the attended span.
    The slot written is the other half: a Python index is frozen at capture, so
    every replay would overwrite the capture-time slot and the cache would stop
    growing -- correct eagerly, silently wrong once captured.
    """
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
    q = torch.randn(1, heads, 1, dim, device=dev, dtype=dt)
    k = torch.zeros(1, kv_heads, 1, dim, device=dev, dtype=dt)
    v = torch.zeros_like(k)
    scale = dim**-0.5

    # Capture on the step that writes slot n-1, restoring what is already there
    # so the capture itself changes nothing.
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
        paged.attend(0, q, k, v, scale)

    kept = paged.k[0].view(-1, kv_heads, dim)[n - 1].clone()

    # The next decode step: one token further, different K/V.
    paged.set_length(n + 1)
    k[0, :, 0] = 1.5
    v[0, :, 0] = -2.5
    graph.replay()
    current_omni_platform.synchronize()

    flat_k = paged.k[0].view(-1, kv_heads, dim)
    flat_v = paged.v[0].view(-1, kv_heads, dim)
    torch.testing.assert_close(flat_k[n].float(), torch.full_like(flat_k[n], 1.5).float())
    torch.testing.assert_close(flat_v[n].float(), torch.full_like(flat_v[n], -2.5).float())
    torch.testing.assert_close(flat_k[n - 1].float(), kept.float())


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_a_whole_decode_loop_matches_the_unpaged_path():
    """Per-step kernel equivalence is not loop equivalence.

    With the switch off a single-token step attends over a cache grown by
    ``torch.cat`` and no mask; with it on the same step reads a bucket-sized
    buffer through ``seqused_k``. Run the same sequence of steps through both,
    crossing a bucket boundary on the way, and require the same argmax at every
    step off a fixed projection -- which is the token id the pipeline would emit.
    """
    import torch.nn.functional as F

    from vllm_omni.diffusion.models.sensenova_u1.paged_decode import PagedDecodeCache

    heads, kv_heads, dim = 8, 2, 64
    prefill, steps = 500, 40  # crosses the 512 bucket at step 13
    dev, dt = torch.device("cuda"), torch.bfloat16
    scale = dim**-0.5
    torch.manual_seed(7)

    keys = torch.randn(1, kv_heads, prefill, dim, device=dev, dtype=dt)
    values = torch.randn(1, kv_heads, prefill, dim, device=dev, dtype=dt)
    paged = PagedDecodeCache.from_dynamic_cache(_Cache([_Layer(keys, values)]), 1, dev, dt)
    ref_k, ref_v = keys.clone(), values.clone()
    proj = torch.randn(dim * heads, 128, device=dev, dtype=torch.float32)

    paged_ids, ref_ids, worst = [], [], 0.0
    for step in range(steps):
        q = torch.randn(1, heads, 1, dim, device=dev, dtype=dt)
        k = torch.randn(1, kv_heads, 1, dim, device=dev, dtype=dt)
        v = torch.randn(1, kv_heads, 1, dim, device=dev, dtype=dt)

        if paged.length + 1 > paged.bucket:
            paged.grow(paged.length + 1)
        paged.set_length(paged.length + 1)
        got = paged.attend(0, q, k, v, scale)

        ref_k = torch.cat([ref_k, k], dim=2)
        ref_v = torch.cat([ref_v, v], dim=2)
        want = F.scaled_dot_product_attention(q, ref_k, ref_v, enable_gqa=True, scale=scale).transpose(1, 2)

        worst = max(worst, (got.float() - want.float()).abs().max().item())
        paged_ids.append(int((got.float().reshape(1, -1) @ proj).argmax()))
        ref_ids.append(int((want.float().reshape(1, -1) @ proj).argmax()))

    assert paged.bucket == 1024, "the loop never crossed a bucket boundary"
    assert paged_ids == ref_ids, (
        f"argmax diverges at step {next(i for i, (a, b) in enumerate(zip(paged_ids, ref_ids)) if a != b)}"
    )
    assert worst < 5e-2, f"largest per-step deviation {worst:.3e}"


class _StubLM(torch.nn.Module):
    """Just enough CUDA work inside the capture for a pool to be allocated."""

    def __init__(self, device):
        super().__init__()
        self.w = torch.randn(16, 64, device=device)

    def forward(self, input_ids, indexes, past_key_values=None, use_cache=False, paged_cache=None):
        x = input_ids.float().expand(1, 16) @ self.w
        return SimpleNamespace(logits=x.unsqueeze(0))


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_the_decode_graph_captures_into_the_shared_platform_pool(monkeypatch):
    """``_decode_context`` builds a runner per request, so a private capture pool
    hands every request its own arena and never gives it back. Measured on a
    real model that was about 40 MiB of device memory per request, rising
    linearly over twelve sequential think requests while the same twelve with
    the switch off held flat.
    """
    from vllm.platforms import current_platform

    from vllm_omni.diffusion.models.sensenova_u1.paged_decode import DecodeGraphRunner

    pools = []
    real_graph = torch.cuda.graph

    def spy(graph, pool=None, **kwargs):
        pools.append(pool)
        return real_graph(graph, pool=pool, **kwargs)

    monkeypatch.setattr(torch.cuda, "graph", spy)

    dev = torch.device("cuda")
    cache = PagedDecodeCache.from_dynamic_cache(_dyn_cache(8, dev), LAYERS, dev, torch.float32)
    runner = DecodeGraphRunner(_StubLM(dev), cache, dev)
    runner.step(0, 0)

    assert pools, "no graph was captured"
    assert pools[0] is not None, "captured into a fresh private pool"
    assert pools[0] == current_platform.get_global_graph_pool()
    assert not hasattr(runner, "_pool"), "the runner still keeps a pool of its own"


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
    got = host.forward_und(torch.zeros(1, 1, 16), None, None, position_embeddings=None, paged_cache=host.paged)
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
    host.forward_und(torch.zeros(1, seq, 16), None, mask, position_embeddings=None, paged_cache=host.paged)
    assert host.paged.calls == [], "a step that cannot be captured reached the paged cache"
    assert len(host.sdpa_calls) == 1


def test_a_kernel_missing_any_kwarg_we_pass_is_not_supported(monkeypatch):
    """The call names every argument, so the probe has to cover every name.

    An older wheel exports ``flash_attn_varlen_func`` without the paged kwargs,
    and a future one may rename the standard ones. Either way the probe is what
    keeps that install on SDPA instead of raising TypeError at the first decode
    step, with the feature on by default.
    """

    def no_paged_kwargs(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, causal=False):
        raise AssertionError("must not be called")  # pragma: no cover

    def renamed_standard_args(q, k, v, max_q, cu_q, max_k, seqused_k=None, block_table=None, causal=False):
        raise AssertionError("must not be called")  # pragma: no cover

    for kernel in (no_paged_kwargs, renamed_standard_args):
        monkeypatch.setattr(paged_decode, "_flash_varlen", lambda kernel=kernel: kernel)
        assert paged_decode.paged_decode_supported(torch.device("cuda"), HEAD_DIM) is False

    def current_kernel(
        q,
        k,
        v,
        max_seqlen_q,
        cu_seqlens_q,
        max_seqlen_k,
        seqused_k=None,
        block_table=None,
        softmax_scale=None,
        causal=False,
    ):
        raise AssertionError("must not be called")  # pragma: no cover

    monkeypatch.setattr(paged_decode, "_flash_varlen", lambda: current_kernel)
    assert paged_decode.paged_decode_supported(torch.device("cuda"), HEAD_DIM) is True


def test_only_q_k_and_v_reach_the_kernel_positionally(monkeypatch):
    """A positional standard argument binds to whatever a future wheel puts in
    that slot, and the probe above only checks that the names exist."""
    seen: dict[str, Any] = {}

    def recorder(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return torch.zeros(1, N_HEADS, HEAD_DIM)

    monkeypatch.setattr(paged_decode, "_flash_varlen", lambda: recorder)
    cache = PagedDecodeCache(1, KV_HEADS, HEAD_DIM, 8, torch.device("cpu"), torch.float32)
    cache.set_length(4)
    cache.attend(
        0,
        torch.zeros(1, N_HEADS, 1, HEAD_DIM),
        torch.zeros(1, KV_HEADS, 1, HEAD_DIM),
        torch.zeros(1, KV_HEADS, 1, HEAD_DIM),
        1.0,
    )
    assert len(seen["args"]) == 3, "only q, k and v may be positional"
    assert set(seen["kwargs"]) == {
        "max_seqlen_q",
        "cu_seqlens_q",
        "max_seqlen_k",
        "seqused_k",
        "block_table",
        "softmax_scale",
        "causal",
    }


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


def _pipeline_host(monkeypatch, device="cpu", language_model=None):
    """A bare pipeline carrying only what ``_decode_context`` touches."""
    from vllm_omni.diffusion.models.sensenova_u1 import pipeline_sensenova_u1 as pipe_mod

    monkeypatch.setenv("VLLM_OMNI_SENSENOVA_PAGED_DECODE", "1")
    monkeypatch.setattr(pipe_mod, "paged_decode_supported", lambda dev, head_dim: True)
    layer = SimpleNamespace(self_attn=SimpleNamespace(head_dim=HEAD_DIM))
    host = object.__new__(pipe_mod.SenseNovaU1Pipeline)
    # The pipeline is an nn.Module, and the language model assigned below is one
    # too, so the bookkeeping has to exist before the assignment.
    torch.nn.Module.__init__(host)
    host.device = device
    lm = language_model if language_model is not None else torch.nn.Module()
    lm.model = SimpleNamespace(layers=[layer] * LAYERS)
    host.language_model = lm
    return pipe_mod, host


def test_the_decode_context_is_reused_across_requests(monkeypatch):
    """A capture binds the addresses it recorded, so a cache built per request
    forces a capture per request. Two requests whose prefill fits the same
    bucket have to come back with the same cache and the same runner, and with
    the generation unchanged so the captures stay valid.
    """
    pipe_mod, host = _pipeline_host(monkeypatch)
    monkeypatch.setattr(pipe_mod, "DecodeGraphRunner", lambda lm, cache, dev: SimpleNamespace(cache=cache))
    ctx = pipe_mod.SenseNovaU1Pipeline._decode_context

    first = ctx(host, _dyn_cache(10))
    assert first is not None
    second_input = _dyn_cache(20)
    second = ctx(host, second_input)
    assert second[0] is first[0], "the cache was rebuilt for a request that fits it"
    assert second[1] is first[1], "the runner was rebuilt, so its captures are gone"
    assert second[0].generation == first[0].generation, "reuse must not invalidate a capture"
    assert second[0].length == 20

    # The reused buffers must carry the new request's prefill, not the old one.
    flat = second[0].k[0].view(-1, KV_HEADS, HEAD_DIM)[:20]
    torch.testing.assert_close(flat, second_input.layers[0].keys[0].transpose(0, 1))

    # A prefill past the bucket cannot share the buffers.
    bigger = ctx(host, _dyn_cache(BUCKETS[0] + 1))
    assert bigger[0] is not first[0], "a prefill past the bucket must get new buffers"


def test_releasing_the_captures_forces_a_rebuild(monkeypatch):
    """Sleep level 2 discards the memory a capture recorded, so what the
    pipeline reuses across requests has to be droppable."""
    pipe_mod, host = _pipeline_host(monkeypatch)
    monkeypatch.setattr(pipe_mod, "DecodeGraphRunner", lambda lm, cache, dev: SimpleNamespace(cache=cache))
    ctx = pipe_mod.SenseNovaU1Pipeline._decode_context

    first = ctx(host, _dyn_cache(10))
    assert ctx(host, _dyn_cache(10))[0] is first[0], "the fixture is not reusing to begin with"
    pipe_mod.SenseNovaU1Pipeline.release_captured_graphs(host)
    after = ctx(host, _dyn_cache(10))
    assert after[0] is not first[0], "the cache survived the release"
    assert after[1] is not first[1], "the runner survived the release"


class _FakeLoRAWrapper(BaseLayerWithLoRA):
    """Stands in for what ``replace_submodule`` leaves in the module tree."""

    def __init__(self):
        super().__init__()


def test_no_capture_crosses_a_request_once_lora_wrappers_are_installed(monkeypatch):
    """Base -> LoRA -> base. The reuse test cannot see an adapter change.

    ``DiffusionLoRAManager`` replaces the decode path's linear layers with
    ``BaseLayerWithLoRA`` and binds, rescales or resets slot 0 per request, and
    the wrappers stay in the tree afterwards. A graph captured before an adapter
    was bound would replay without its matmuls; one captured with it bound would
    survive its removal. Neither moves the shape, dtype or bucket that decide
    reuse, so the wrappers themselves have to end it.
    """
    pipe_mod, host = _pipeline_host(monkeypatch)
    monkeypatch.setattr(pipe_mod, "DecodeGraphRunner", lambda lm, cache, dev: SimpleNamespace(cache=cache))
    ctx = pipe_mod.SenseNovaU1Pipeline._decode_context

    base = ctx(host, _dyn_cache(10))
    assert ctx(host, _dyn_cache(10))[0] is base[0], "the fixture is not reusing to begin with"

    host.language_model.add_module("q_proj", _FakeLoRAWrapper())
    adapted = ctx(host, _dyn_cache(10))
    assert adapted[0] is not base[0], "an adapted request reused the base capture"
    assert adapted[1] is not base[1], "an adapted request reused the base runner"

    back = ctx(host, _dyn_cache(10))
    assert back[0] is not adapted[0], "a later request reused the adapted capture"
    assert back[1] is not adapted[1], "a later request reused the adapted runner"
    assert getattr(host, "_paged_decode", None) is None, "an adapted capture was left on the pipeline"


@cuda_only
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_a_second_request_replays_the_first_capture(monkeypatch):
    """Reuse is only worth anything if the capture survives it."""
    dev = torch.device("cuda")
    pipe_mod, host = _pipeline_host(monkeypatch, device="cuda", language_model=_StubLM(dev))
    ctx = pipe_mod.SenseNovaU1Pipeline._decode_context

    cache, runner = ctx(host, _dyn_cache(10, dev))
    cache.set_length(cache.length + 1)
    runner.step(0, 0)
    assert runner.captures == 1

    cache2, runner2 = ctx(host, _dyn_cache(12, dev))
    assert cache2 is cache and runner2 is runner
    cache2.set_length(cache2.length + 1)
    runner2.step(0, 0)
    assert runner.captures == 1, "the second request captured the graph again"


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
