# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the AR-Diffusion KV cache helpers (Phase 1, PR-2).

Covers the request adapter, the chunk-window spec/manager (registration + the
eviction policy), and the pool builder — exercised against the installed vLLM
KV stack on CPU (block bookkeeping only, no GPU tensors).
"""

import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheSpecKind, get_kv_cache_spec_kind
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.request import RequestStatus

from tests.helpers.mark import hardware_test
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import (
    ARDiffusionKVCache,
    ARDiffusionKVConfig,
    ARDiffusionRequestAdapter,
    ChunkWindowManager,
    ChunkWindowSpec,
    allocate_kv_pool_with_views,
    build_kv_manager,
    compute_num_blocks,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import chunk_window_skipped_tokens

BLOCK = 16

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def make_spec(*, chunk_size=BLOCK, window_chunks=2, sink_chunks=0, reset_at_boundary=False):
    return ChunkWindowSpec(
        block_size=BLOCK,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
        sliding_window=window_chunks * chunk_size,
        chunk_size=chunk_size,
        window_chunks=window_chunks,
        sink_chunks=sink_chunks,
        reset_at_boundary=reset_at_boundary,
    )


# --- ChunkWindowSpec registration -------------------------------------------


def test_spec_registration_resolves_to_chunk_window_manager():
    # Without explicit registration the MRO walk would fall back to the parent
    # SlidingWindowManager; assert the subclass manager wins.
    spec = make_spec()
    assert KVCacheSpecRegistry.get_manager_class(spec) is ChunkWindowManager


def test_spec_kind_is_sliding_window():
    assert get_kv_cache_spec_kind(make_spec()) == KVCacheSpecKind.SLIDING_WINDOW


def test_spec_rejects_inconsistent_window():
    with pytest.raises(ValueError):
        ChunkWindowSpec(
            block_size=BLOCK,
            num_kv_heads=4,
            head_size=64,
            dtype=torch.float16,
            sliding_window=99,  # != window_chunks * chunk_size
            chunk_size=BLOCK,
            window_chunks=2,
        )


# --- eviction policy (pure) -------------------------------------------------


def test_sliding_replace_keeps_window():
    # window = 2 chunks * 16 = 32. Base sliding formula keeps `window` tokens;
    # the snap is to chunk boundaries.
    def skip(n):
        return chunk_window_skipped_tokens(n, chunk_size=16, sliding_window=32, sink_chunks=0, reset_at_boundary=False)

    assert skip(32) == 0  # nothing past the window yet
    assert skip(48) == 16  # one chunk fell out of the window
    assert skip(64) == 32


def test_sliding_replace_snaps_to_chunk_boundary():
    # A non-chunk-aligned overflow must snap down so a chunk is never half-evicted.
    skip = chunk_window_skipped_tokens(50, chunk_size=16, sliding_window=32, sink_chunks=0, reset_at_boundary=False)
    assert skip % 16 == 0 and skip == 16


def test_sink_chunks_protected():
    # sink = 1 chunk (16 tokens) is never skipped.
    skip = chunk_window_skipped_tokens(80, chunk_size=16, sliding_window=32, sink_chunks=1, reset_at_boundary=False)
    assert skip == 32


def test_reset_at_boundary_drops_completed_past_sink():
    skip = chunk_window_skipped_tokens(48, chunk_size=16, sliding_window=32, sink_chunks=1, reset_at_boundary=True)
    # completed = 48; sink = 16 -> drop 32
    assert skip == 32


# --- ARDiffusionKVConfig ------------------------------------------------------------


def test_kv_config_sliding_window_property():
    assert ARDiffusionKVConfig(chunk_size=16, window_chunks=3).sliding_window == 48
    assert ARDiffusionKVConfig(chunk_size=16, window_chunks=None).sliding_window is None


# --- ARDiffusionRequestAdapter ------------------------------------------------------


def test_adapter_advances_per_chunk_not_per_step():
    a = ARDiffusionRequestAdapter("r0", chunk_size=16)
    assert a.num_computed_tokens == 0
    assert a.num_tokens == 16  # in-flight chunk
    a.on_chunk_committed()
    assert a.num_computed_tokens == 16
    assert a.num_tokens == 32


def test_adapter_accounts_for_prefill_prefix():
    a = ARDiffusionRequestAdapter("r0", chunk_size=16, prefill_prefix_tokens=4)
    assert a.num_computed_tokens == 4
    assert a.num_prompt_tokens == 4
    assert a.num_tokens == 20


def test_adapter_status_is_vllm_enum():
    assert isinstance(ARDiffusionRequestAdapter("r0", chunk_size=16).status, RequestStatus)


# --- pool / manager ---------------------------------------------------------


def test_compute_num_blocks():
    # 1 MiB budget at fraction 0.5 with 16 KiB pages -> 32 blocks.
    assert compute_num_blocks(1 << 20, 0.5, 16 << 10) == 32


def test_paged_pool_layout_exposes_flat_slot_views():
    kv_pools, k_pools, v_pools = allocate_kv_pool_with_views(
        num_blocks=4,
        block_size=BLOCK,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    key_cache, value_cache = kv_pools[0]
    assert key_cache.shape == value_cache.shape == (4, BLOCK, 4, 64)
    assert k_pools[0].shape == (4 * BLOCK, 4, 64)
    assert v_pools[0].shape == (4 * BLOCK, 4, 64)

    # The flat views must alias their block-shaped cache, so a slot write is
    # visible through the layout the attention kernel reads.
    k_pools[0][BLOCK + 3].fill_(7)
    v_pools[0][2 * BLOCK + 5].fill_(11)
    assert torch.equal(key_cache[1, 3], k_pools[0][BLOCK + 3])
    assert torch.equal(value_cache[2, 5], v_pools[0][2 * BLOCK + 5])


def test_key_and_value_caches_do_not_share_storage():
    """K and V must be separate allocations, not halves of one tensor.

    The paged-write custom op declares both as mutated. When they alias one
    storage, inductor's reinplace pass can re-inplace only the first of the
    two, and auto_functionalized_v2's clone of the entire second pool survives
    into the compiled graph -- one full pool copied per compiled region per
    denoising step.
    """
    kv_pools, k_pools, v_pools = allocate_kv_pool_with_views(
        num_blocks=4,
        block_size=BLOCK,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    seen = set()
    for layer, (key_cache, value_cache) in enumerate(kv_pools):
        k_storage = key_cache.untyped_storage().data_ptr()
        v_storage = value_cache.untyped_storage().data_ptr()
        assert k_storage != v_storage, f"layer {layer}: K and V share one allocation"
        assert k_pools[layer].untyped_storage().data_ptr() == k_storage
        assert v_pools[layer].untyped_storage().data_ptr() == v_storage
        seen.update((k_storage, v_storage))
    # Every layer's K and V are distinct allocations as well.
    assert len(seen) == 2 * len(kv_pools)

    # Negative control: writing V must not disturb K.
    k_pools[0].fill_(0)
    v_pools[0].fill_(5)
    assert torch.equal(k_pools[0], torch.zeros_like(k_pools[0]))


def _pool_sized_allocations(code: str, pool_numel: int) -> list[int]:
    """Sizes of generated allocations at least as large as one pool."""
    import re

    sizes = []
    for shape in re.findall(r"empty_strided_\w+\(\(([\d, ]*?)\)", code):
        dims = [int(part) for part in shape.replace(" ", "").strip(",").split(",") if part]
        numel = 1
        for dim_size in dims:
            numel *= dim_size
        if dims and numel >= pool_numel:
            sizes.append(numel)
    return sizes


def _generated_code_for_pools(key_cache, value_cache):
    """Compile a paged-write-shaped op over these pools, return the generated code."""
    from torch._inductor import config as inductor_config
    from torch._inductor.utils import run_and_get_code

    class _Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("key_cache", key_cache)
            self.register_buffer("value_cache", value_cache)

        def forward(self, value):
            return torch.ops.test_kv_pool_clone.write(self.key_cache, self.value_cache, value)

    torch._dynamo.reset()
    # Inductor caches compiled code across ``torch._dynamo.reset()``, so the
    # second call in a process would otherwise return the first one's code and
    # both branches of this test would describe the same graph.
    with inductor_config.patch(force_disable_caches=True):
        _, codes = run_and_get_code(torch.compile(_Holder(), backend="inductor", fullgraph=True), torch.ones(1))
    return "\n".join(codes)


def test_compiled_paged_write_does_not_clone_the_pool():
    """The compiled graph must write the pools in place, not through a clone.

    ``test_key_and_value_caches_do_not_share_storage`` pins the mechanism this
    change introduces -- two allocations rather than one. This pins the effect
    that mechanism buys, which is the reason the change exists: with K and V as
    halves of one tensor, inductor's reinplace pass re-inplaces only the first,
    and ``auto_functionalized_v2``'s clone of the pool survives into the
    generated code as a pool-sized allocation plus a copy, once per compiled
    region per denoising step.

    The shared layout is compiled here as a positive control. Without it, a
    codegen change that stops emitting ``empty_strided_*`` would leave the
    detector matching nothing and this test passing for the wrong reason.
    """
    if "write" not in dir(getattr(torch.ops, "test_kv_pool_clone", object())):

        @torch.library.custom_op("test_kv_pool_clone::write", mutates_args=("key_pool", "value_pool"))
        def _write(key_pool: torch.Tensor, value_pool: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            key_pool[0, 0, 0, 0] = value[0]
            value_pool[0, 0, 0, 0] = value[0]
            return value * 2

        @_write.register_fake
        def _(key_pool, value_pool, value):
            return torch.empty_like(value)

    # The stand-in models the real op's contract; that declaration is what
    # drives the reinplace pass, so a drift here would silently test nothing.
    real_schema = torch.ops.vllm_omni.ar_diffusion_paged_write_attn.default._schema
    real_mutated = [arg.name for arg in real_schema.arguments if arg.alias_info and arg.alias_info.is_write]
    assert real_mutated == ["key_pool", "value_pool"], (
        f"the real paged-write op now mutates {real_mutated}; this test still models two pools"
    )

    num_blocks, heads, dim = 8, 4, 16
    cache_shape = (num_blocks, BLOCK, heads, dim)
    pool_numel = num_blocks * BLOCK * heads * dim

    # Positive control: one allocation, K and V aliasing it.
    shared = torch.empty(2, *cache_shape)
    shared_allocations = _pool_sized_allocations(_generated_code_for_pools(shared[0], shared[1]), pool_numel)
    assert shared_allocations, (
        "the shared layout produced no pool-sized allocation, so this test can no "
        "longer tell the two layouts apart -- inductor's codegen or the detector changed"
    )

    # The allocator under test.
    kv_pools, _, _ = allocate_kv_pool_with_views(
        num_blocks=num_blocks,
        block_size=BLOCK,
        num_layers=1,
        num_kv_heads=heads,
        head_dim=dim,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    key_cache, value_cache = kv_pools[0]
    separate_allocations = _pool_sized_allocations(_generated_code_for_pools(key_cache, value_cache), pool_numel)
    assert not separate_allocations, (
        "the compiled graph allocates a pool-sized buffer, so the pool is being cloned "
        f"rather than written in place: {separate_allocations} (control saw {shared_allocations})"
    )


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_real_paged_write_op_does_not_clone_the_pool_on_cuda():
    """Same assertion as the CPU test, against the real op and CUDA codegen.

    The CPU test compiles a stand-in with the same ``mutates_args`` contract,
    because the real op needs a device and an attention kernel. That leaves two
    things unverified: whether the real op's full signature behaves the same,
    and whether Triton codegen does, which is the path the reported regression
    was observed on. This closes both.

    Note the pools passed here are the flat views, since that is what the op
    takes -- it indexes them by slot and unflattens internally.
    """
    import re

    from torch._inductor import config as inductor_config
    from torch._inductor.utils import run_and_get_code

    device = torch.device("cuda")
    num_blocks, heads, dim, num_tokens = 8, 4, 64, 8
    pool_numel = num_blocks * BLOCK * heads * dim

    class _Holder(torch.nn.Module):
        def __init__(self, key_flat, value_flat):
            super().__init__()
            self.register_buffer("key_flat", key_flat)
            self.register_buffer("value_flat", value_flat)

        def forward(self, query, k_curr, v_curr, video_slots, action_slots, block_table, query_start_loc, seq_lens):
            return torch.ops.vllm_omni.ar_diffusion_paged_write_attn(
                query,
                k_curr,
                v_curr,
                None,
                None,
                self.key_flat,
                self.value_flat,
                BLOCK,
                video_slots,
                action_slots,
                block_table,
                query_start_loc,
                seq_lens,
                num_tokens,
                num_tokens,
                dim**-0.5,
            )

    def compiled_pool_allocations(key_flat, value_flat) -> list[int]:
        args = (
            torch.randn(num_tokens, heads, dim, dtype=torch.bfloat16, device=device),
            torch.randn(num_tokens, heads, dim, dtype=torch.bfloat16, device=device),
            torch.randn(num_tokens, heads, dim, dtype=torch.bfloat16, device=device),
            torch.arange(num_tokens, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0),
            torch.tensor([0, num_tokens], device=device, dtype=torch.int32),
            torch.tensor([num_tokens], device=device, dtype=torch.int32),
        )
        torch._dynamo.reset()
        # See the CPU test: inductor's cache outlives torch._dynamo.reset().
        with inductor_config.patch(force_disable_caches=True):
            _, codes = run_and_get_code(
                torch.compile(_Holder(key_flat, value_flat), backend="inductor", fullgraph=True), *args
            )
        sizes = []
        for shape in re.findall(r"empty_strided_cuda\(\(([\d, ]*?)\)", "\n".join(codes)):
            dims = [int(part) for part in shape.replace(" ", "").strip(",").split(",") if part]
            numel = 1
            for dim_size in dims:
                numel *= dim_size
            if dims and numel >= pool_numel:
                sizes.append(numel)
        return sizes

    # Positive control: one allocation, K and V as its two halves.
    shared = torch.empty(2, num_blocks * BLOCK, heads, dim, dtype=torch.bfloat16, device=device)
    control = compiled_pool_allocations(shared[0], shared[1])
    assert control, (
        "the shared layout produced no pool-sized allocation, so this test can no longer tell the two layouts apart"
    )

    _, k_pools, v_pools = allocate_kv_pool_with_views(
        num_blocks=num_blocks,
        block_size=BLOCK,
        num_layers=1,
        num_kv_heads=heads,
        head_dim=dim,
        dtype=torch.bfloat16,
        device=device,
    )
    separate = compiled_pool_allocations(k_pools[0], v_pools[0])
    assert not separate, (
        "the compiled graph allocates a pool-sized buffer, so the pool is being cloned "
        f"rather than written in place: {separate} (control saw {control})"
    )


def test_build_manager_allocate_free_roundtrip():
    """End-to-end: a ARDiffusionRequestAdapter drives a real KVCacheManager.

    This is the adapter conformance check — if the adapter were missing an
    attribute the manager reads, allocate_slots/free would raise here.
    """
    spec = make_spec()
    mgr = build_kv_manager(spec, ["layer0"], num_blocks=16, max_model_len=1024)
    free_before = mgr.block_pool.get_num_free_blocks()

    adapter = ARDiffusionRequestAdapter("req-0", chunk_size=BLOCK)
    blocks = mgr.allocate_slots(adapter, num_new_tokens=BLOCK, full_sequence_must_fit=True)
    assert blocks is not None
    assert mgr.block_pool.get_num_free_blocks() < free_before

    mgr.free(adapter)
    assert mgr.block_pool.get_num_free_blocks() == free_before


def test_cross_attn_pool_deducted_from_self_attn_budget():
    """The cross-attn pool is allocated directly; its bytes are subtracted from
    the self-attn paged-pool budget so the two together stay within the free
    memory budget (review: zwhzzz0821)."""
    L = 512
    avail = 1 << 30  # 1 GiB
    kv = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=2, gpu_memory_fraction=0.5),
        num_layers=2,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float16,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=avail,
        kv_branches=(ARDiffusionKVBranchSpec("positive", 0), ARDiffusionKVBranchSpec("negative", 0)),
        session_capacity=1,
        cross_attention_lengths={"text": L},
    )
    cross_bytes = 2 * 2 * L * 4 * 64 * torch.float16.itemsize * 2  # K+V, pos+neg, layers
    page_bytes = kv.spec.page_size_bytes * 2
    scratch_bytes = kv.scratch_num_blocks * page_bytes
    expected = (int(avail * 0.5) - cross_bytes - scratch_bytes) // page_bytes
    assert kv.num_blocks == expected
    assert kv.cross_attention_reserved_bytes == cross_bytes
    assert (kv.num_blocks_total * page_bytes) + cross_bytes <= int(avail * 0.5)
    assert expected > 1 * (2 + 1) + 2  # above the local-slot minimum, so the cross deduction is what's tested


def _make_tiny_capacity_kv(
    *,
    requested_capacity: int,
    available_bytes: int,
    gpu_memory_fraction: float = 1.0,
    model_owned_state_bytes_per_session: int = 0,
) -> ARDiffusionKVCache:
    return ARDiffusionKVCache(
        ARDiffusionKVConfig(
            enable=True,
            chunk_size=1,
            window_chunks=3,
            sink_chunks=3,
            gpu_memory_fraction=gpu_memory_fraction,
        ),
        num_layers=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        block_size=1,
        max_model_len=128,
        available_bytes=available_bytes,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=requested_capacity,
        cross_attention_lengths={"text": 2},
        frames_per_block=3,
        model_owned_state_bytes_per_session=model_owned_state_bytes_per_session,
        device=torch.device("cpu"),
    )


def test_capacity_two_retains_both_windows_and_allocates_next_block():
    # LingBot-like single-branch geometry: page=8 bytes; capacity=2 requires
    # 17 managed + 3 scratch blocks, plus two 16-byte cross-attention
    # reservations: 192 bytes exactly.
    kv = _make_tiny_capacity_kv(requested_capacity=2, available_bytes=192)
    assert kv.requested_session_capacity == 2
    assert kv.session_capacity == 2
    assert kv.managed_num_blocks == 17

    adapters = [kv.begin_request(session_id) for session_id in ("first", "second")]
    for adapter in adapters:
        for _ in range(6):  # three sink + three window blocks
            kv.allocate_token_slots(adapter, 1)
            kv.commit_chunk(adapter)
        assert len(kv.window_block_ids(adapter)) == 6

    # Only one request is in flight, so its three-frame block is counted once.
    kv.allocate_token_slots(adapters[0], 3)


def test_requested_capacity_is_capped_and_cross_reservation_uses_effective_capacity():
    kv = _make_tiny_capacity_kv(requested_capacity=64, available_bytes=192)

    assert kv.requested_session_capacity == 64
    assert kv.session_capacity == 2
    assert kv.cross_attention_bytes_per_session == 16
    assert kv.cross_attention_reserved_bytes == 32
    assert kv.num_blocks_total * 8 + kv.cross_attention_reserved_bytes == kv.memory_budget_bytes


def test_capacity_rejects_budget_that_cannot_fit_one_session():
    # One session needs 11 managed + 3 scratch blocks and 16 cross bytes:
    # 14 * 8 + 16 = 128 bytes.
    with pytest.raises(ValueError, match="cannot fit one session.*available=127 bytes.*required=128 bytes"):
        _make_tiny_capacity_kv(requested_capacity=1, available_bytes=127)


def test_fraction_is_soft_floor_when_one_session_fits_actual_memory():
    kv = _make_tiny_capacity_kv(
        requested_capacity=2,
        available_bytes=1280,
        gpu_memory_fraction=0.05,
    )

    assert kv.configured_memory_budget_bytes == 64
    assert kv.memory_budget_bytes == 128
    assert kv.session_capacity == 1


def test_model_owned_state_reduces_effective_session_capacity():
    kv = _make_tiny_capacity_kv(
        requested_capacity=2,
        available_bytes=200,
        model_owned_state_bytes_per_session=16,
    )

    assert kv.session_capacity == 1
    assert kv.model_owned_state_reserved_bytes == 16
    assert (
        kv.num_blocks_total * 8 + kv.cross_attention_reserved_bytes + kv.model_owned_state_reserved_bytes
        <= kv.memory_budget_bytes
    )


def _make_shipped_lingbot_geometry(*, gpu_memory_fraction: float) -> ARDiffusionKVCache:
    return ARDiffusionKVCache(
        ARDiffusionKVConfig(
            enable=True,
            chunk_size=1560,
            window_chunks=9,
            sink_chunks=9,
            gpu_memory_fraction=gpu_memory_fraction,
        ),
        num_layers=40,
        num_kv_heads=40,
        head_size=128,
        dtype=torch.bfloat16,
        block_size=1560,
        max_model_len=1 << 20,
        available_bytes=100 * (1 << 30),
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=2,
        cross_attention_lengths={"text": 512},
        frames_per_block=3,
        model_owned_state_bytes_per_session=7_488_000,
    )


def test_shipped_lingbot_geometry_default_fraction_admits_one_session():
    kv = _make_shipped_lingbot_geometry(gpu_memory_fraction=0.1)

    assert kv.configured_memory_budget_bytes == 10 * (1 << 30)
    assert kv.memory_budget_bytes == 33_653_670_400
    assert kv.session_capacity == 1


def test_shipped_lingbot_geometry_tuned_fraction_admits_two_sessions():
    kv = _make_shipped_lingbot_geometry(gpu_memory_fraction=0.6)

    assert kv.memory_budget_bytes == 60 * (1 << 30)
    assert kv.session_capacity == 2


def _make_kv(
    *,
    local_branches,
    num_frame_per_block=2,
    window_chunks=9,
    max_scratch_tokens_per_branch=0,
):
    kv_branches = (
        (ARDiffusionKVBranchSpec("positive", 0), ARDiffusionKVBranchSpec("negative", 0))
        if local_branches == 1
        else (ARDiffusionKVBranchSpec("positive", 0), ARDiffusionKVBranchSpec("negative", 1))
    )
    page_bytes = 2 * BLOCK * 4 * 64 * torch.float32.itemsize
    declared_scratch_blocks = (max_scratch_tokens_per_branch + BLOCK - 1) // BLOCK
    scratch_blocks = local_branches * (num_frame_per_block + declared_scratch_blocks)
    managed_blocks = local_branches * (window_chunks + num_frame_per_block) + 2
    return ARDiffusionKVCache(
        ARDiffusionKVConfig(
            enable=True,
            chunk_size=BLOCK,
            window_chunks=window_chunks,
            gpu_memory_fraction=1.0,
        ),
        num_layers=1,
        num_kv_heads=4,
        head_size=64,
        dtype=torch.float32,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=(managed_blocks + scratch_blocks) * page_bytes,
        device=torch.device("cpu"),
        kv_branches=kv_branches,
        session_capacity=1,
        frames_per_block=num_frame_per_block,
        max_scratch_tokens_per_branch=max_scratch_tokens_per_branch,
    )


def test_pool_floor_is_branch_aware():
    """CFG-parallel rank (one local kv_branch) sizes for one window + in-flight chunk;
    a single-process run (both kv_branches) sizes for two. Scratch scales the same way."""
    one = _make_kv(local_branches=1)
    two = _make_kv(local_branches=2)

    assert one.managed_num_blocks == 1 * (9 + 2) + 2  # 13
    assert two.managed_num_blocks == 2 * (9 + 2) + 2  # 24
    assert one.scratch_num_blocks == one.scratch_blocks_per_kv_branch
    assert two.scratch_num_blocks == 2 * two.scratch_blocks_per_kv_branch
    assert one.scratch_blocks_per_kv_branch == 2
    assert one.num_blocks_total == 13 + one.scratch_blocks_per_kv_branch


def test_scratch_capacity_is_derived_from_declared_geometry(monkeypatch):
    monkeypatch.delenv("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH", raising=False)
    kv = _make_kv(
        local_branches=1,
        num_frame_per_block=6,
        max_scratch_tokens_per_branch=BLOCK + 1,
    )
    assert kv.scratch_blocks_per_kv_branch == 8  # six video blocks + two auxiliary blocks


def test_scratch_env_override_cannot_reduce_declared_minimum(monkeypatch):
    monkeypatch.setenv("AR_DIFFUSION_KV_SCRATCH_BLOCKS_PER_BRANCH", "1")
    kv = _make_kv(
        local_branches=1,
        num_frame_per_block=6,
        max_scratch_tokens_per_branch=BLOCK + 1,
    )
    assert kv.scratch_blocks_per_kv_branch == 8


def test_scratch_maps_to_slot_zero_with_one_local_branch():
    """A CFG-parallel rank runs exactly one kv_branch: whichever CFG side it is,
    its scratch lands in the rank's single slot (no dead second slot)."""
    one = _make_kv(local_branches=1)
    assert one.scratch_block_ids("positive", 0, 2) == one.scratch_block_ids("negative", 0, 2)

    two = _make_kv(local_branches=2)
    assert two.scratch_block_ids("positive", 0, 2) != two.scratch_block_ids("negative", 0, 2)


def test_scratch_exhaustion_still_raises():
    one = _make_kv(local_branches=1)
    cap = one.scratch_blocks_per_kv_branch
    with pytest.raises(RuntimeError, match="scratch blocks exhausted"):
        one.scratch_block_ids("positive", 0, cap + 1)


def test_non_contiguous_branch_indices_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        ARDiffusionKVCache(
            ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=2),
            num_layers=1,
            num_kv_heads=4,
            head_size=64,
            dtype=torch.float32,
            block_size=BLOCK,
            max_model_len=4096,
            available_bytes=1 << 16,
            kv_branches=(ARDiffusionKVBranchSpec("main", 1),),
            session_capacity=1,
        )
