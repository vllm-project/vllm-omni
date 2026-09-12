"""Hidden-state prefix cache correctness tests for OmniTensorPrefixCache merge paths."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache

pytestmark = [pytest.mark.core_model]

NUM_BLOCKS = 4
BLOCK_SIZE = 4
HIDDEN_SIZE = 8


class _BlockTableForPrefixCacheTest:
    """Mimic vLLM InputBatch.block_table: .block_tables length check + [0].block_table.cpu."""

    def __init__(self, block_cpu: torch.Tensor):
        self.block_tables = [SimpleNamespace(block_table=SimpleNamespace(cpu=block_cpu))]

    def __getitem__(self, idx: int) -> SimpleNamespace:
        return self.block_tables[idx]


def _input_batch(
    req_ids: list[str],
    num_computed_per_req: list[int],
    block_cpu: torch.Tensor,
) -> SimpleNamespace:
    return SimpleNamespace(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        num_computed_tokens_cpu=torch.tensor(num_computed_per_req, dtype=torch.long),
        block_table=_BlockTableForPrefixCacheTest(block_cpu),
    )


@pytest.mark.cuda
@pytest.mark.L4
def test_cuda_hidden_states_update_and_merge():
    """GPU tensors are coerced to CPU for cache read/write; merged prefix + new tail match.

    ``L4`` matches .buildkite/test-ready.yml "CUDA Unit Test with single card"
    (``-m 'core_model and cuda and L4 and not distributed_cuda'``).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    device = torch.device("cuda:0")

    cache = OmniTensorPrefixCache(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        hs_dtype=torch.float32,
    )

    prefix_len = 8
    slot_mapping = torch.arange(8, 8 + prefix_len, dtype=torch.long)
    prefix_hs = torch.stack(
        [torch.full((HIDDEN_SIZE,), float(i), device=device, dtype=torch.float32) for i in range(prefix_len)]
    )

    cache.update_omni_tensor_prefix_cache(
        hidden_states=prefix_hs,
        multimodal_outputs=None,
        num_tokens_unpadded=prefix_len,
        slot_mapping=slot_mapping,
    )

    req_id = "r0"
    max_blocks = 4
    block_cpu = torch.zeros(1, max_blocks, dtype=torch.long)
    block_cpu[0, 0] = 2
    block_cpu[0, 1] = 3
    input_batch = _input_batch([req_id], [prefix_len], block_cpu)
    cache.add_prefix_cached_new_req_id(req_id)

    new_len = 2
    new_hs = torch.stack(
        [torch.full((HIDDEN_SIZE,), 100.0 + float(j), device=device, dtype=torch.float32) for j in range(new_len)]
    )
    query_start_loc = torch.tensor([0], dtype=torch.int32, device="cpu")

    merged = cache.get_merged_hidden_states(
        query_start_loc=query_start_loc,
        input_batch=input_batch,
        hidden_states=new_hs,
        num_scheduled_tokens={req_id: new_len},
    )[req_id]

    assert merged.device.type == "cpu"
    assert merged.shape == (prefix_len + new_len, HIDDEN_SIZE)
    assert torch.allclose(merged[:prefix_len], prefix_hs.detach().cpu())
    assert torch.allclose(merged[prefix_len:], new_hs.detach().cpu())


@pytest.mark.cpu
def test_dual_prefix_hits_shared_vs_distinct_block_tables():
    """Two prefix-hit reqs: same block rows share cached prefix; different rows stay isolated; shapes align."""
    torch.manual_seed(0)

    cache = OmniTensorPrefixCache(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        hs_dtype=torch.float32,
    )

    # --- Shared physical prefix (blocks 2 and 3, slots 8..15) ---
    prefix_len = 8
    slot_shared = torch.arange(8, 16, dtype=torch.long)
    shared_pattern = torch.arange(prefix_len * HIDDEN_SIZE, dtype=torch.float32).reshape(prefix_len, HIDDEN_SIZE)
    cache.update_omni_tensor_prefix_cache(
        hidden_states=shared_pattern,
        multimodal_outputs=None,
        num_tokens_unpadded=prefix_len,
        slot_mapping=slot_shared,
    )

    max_blocks = 4
    block_same = torch.tensor([[2, 3, 0, 0]], dtype=torch.long).repeat(2, 1)
    input_batch = _input_batch(["ra", "rb"], [prefix_len, prefix_len], block_same)
    cache.add_prefix_cached_new_req_id("ra")
    cache.add_prefix_cached_new_req_id("rb")

    sched_a, sched_b = 2, 2
    tail_a = torch.ones((sched_a, HIDDEN_SIZE)) * 11.0
    tail_b = torch.ones((sched_b, HIDDEN_SIZE)) * 22.0
    batched = torch.cat([tail_a, tail_b], dim=0)
    query_start_loc = torch.tensor([0, sched_a], dtype=torch.int32)

    merged = cache.get_merged_hidden_states(
        query_start_loc=query_start_loc,
        input_batch=input_batch,
        hidden_states=batched,
        num_scheduled_tokens={"ra": sched_a, "rb": sched_b},
    )

    assert torch.equal(merged["ra"][:prefix_len], merged["rb"][:prefix_len])
    assert torch.equal(merged["ra"][:prefix_len], shared_pattern)
    assert torch.equal(merged["ra"][prefix_len:], tail_a)
    assert torch.equal(merged["rb"][prefix_len:], tail_b)
    assert merged["ra"].shape == (prefix_len + sched_a, HIDDEN_SIZE)
    assert merged["rb"].shape == (prefix_len + sched_b, HIDDEN_SIZE)

    # --- Distinct prefixes: ra reads blocks 2-3, rb reads blocks 0-1 ---
    cache2 = OmniTensorPrefixCache(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        hs_dtype=torch.float32,
    )
    pat_lo = torch.ones((prefix_len, HIDDEN_SIZE)) * 3.0
    pat_hi = torch.ones((prefix_len, HIDDEN_SIZE)) * 7.0
    slot_lo = torch.arange(0, 8, dtype=torch.long)
    slot_hi = torch.arange(8, 16, dtype=torch.long)
    cache2.update_omni_tensor_prefix_cache(pat_lo, None, prefix_len, slot_lo)
    cache2.update_omni_tensor_prefix_cache(pat_hi, None, prefix_len, slot_hi)

    block_mixed = torch.zeros(2, max_blocks, dtype=torch.long)
    block_mixed[0, 0] = 2
    block_mixed[0, 1] = 3
    block_mixed[1, 0] = 0
    block_mixed[1, 1] = 1
    ib2 = _input_batch(["ra", "rb"], [prefix_len, prefix_len], block_mixed)
    cache2.add_prefix_cached_new_req_id("ra")
    cache2.add_prefix_cached_new_req_id("rb")

    tail_a2 = torch.full((1, HIDDEN_SIZE), 0.5)
    tail_b2 = torch.full((1, HIDDEN_SIZE), 0.25)
    batched2 = torch.cat([tail_a2, tail_b2], dim=0)
    qsl2 = torch.tensor([0, 1], dtype=torch.int32)

    m2 = cache2.get_merged_hidden_states(
        query_start_loc=qsl2,
        input_batch=ib2,
        hidden_states=batched2,
        num_scheduled_tokens={"ra": 1, "rb": 1},
    )

    assert torch.equal(m2["ra"][:prefix_len], pat_hi)
    assert torch.equal(m2["rb"][:prefix_len], pat_lo)
    assert not torch.equal(m2["ra"][:prefix_len], m2["rb"][:prefix_len])
    assert m2["ra"].shape == (prefix_len + 1, HIDDEN_SIZE)
    assert m2["rb"].shape == (prefix_len + 1, HIDDEN_SIZE)
