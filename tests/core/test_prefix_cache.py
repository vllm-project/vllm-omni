import math
from typing import NamedTuple

import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


DEFAULT_SEQ_LEN = 15
NUM_BLOCKS = 10
BLOCK_SIZE = 4
HIDDEN_SIZE = 2
DTYPE = torch.float32
OTHER_DTYPE = torch.float16
DEFAULT_SHAPE = torch.Size([NUM_BLOCKS, BLOCK_SIZE, HIDDEN_SIZE])


class MockInputBatch:
    def __init__(self, num_computed_tokens_cpu, block_table=None):
        self.req_ids = ["req1", "req2"]
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(self.req_ids)}
        self.num_computed_tokens_cpu = num_computed_tokens_cpu

        # Block table is only mocked for validation of length;
        # we don't actually need to add valid values here since
        # we patch the table when testing.
        if block_table is None:

            class _DummyBlockTable:
                pass

            self.block_table = _DummyBlockTable()
            self.block_table.block_tables = [None]
        else:

            class _TensorWrapper:
                def __init__(self, tensor):
                    self.cpu = tensor

            class _BlockGroup:
                def __init__(self, tensor):
                    self.block_table = _TensorWrapper(tensor)

            class _BlockTable:
                def __init__(self, tensor):
                    self._group = _BlockGroup(tensor)
                    self.block_tables = [self._group.block_table]

                def __getitem__(self, idx):
                    assert idx == 0
                    return self._group

            self.block_table = _BlockTable(block_table)


def get_omni_pcache_with_mm_tensors(feat_dims, seq_len) -> OmniTensorPrefixCache:
    """Build an OmniTensorPrefixCache and init mm tensors."""
    cache = get_omni_pcache()
    mm_outputs = get_multimodal_outputs(feat_dims, seq_len)
    cache.maybe_init_missing_mm_cache_keys(mm_outputs, seq_len)
    return cache


def get_omni_pcache() -> OmniTensorPrefixCache:
    """Build an OmniTensorPrefixCache, but don't init mm tensors."""
    cache = OmniTensorPrefixCache(
        num_blocks=NUM_BLOCKS,
        block_size=BLOCK_SIZE,
        hidden_size=HIDDEN_SIZE,
        hs_dtype=DTYPE,
    )
    return cache


def get_multimodal_outputs(feat_dims: dict[str, int], seq_len: int) -> dict[str, torch.Tensor]:
    fake_mm_inputs = {}
    for mm_key, feat_dim in feat_dims.items():
        fake_mm_inputs[mm_key] = torch.rand((seq_len, feat_dim), dtype=DTYPE)
    return fake_mm_inputs


### Tests for initialization
def test_initialization_simple():
    """Check default initialization only creates the hidden states."""
    cache = get_omni_pcache()
    assert isinstance(cache.hidden_states_cache, torch.Tensor)
    assert cache.hidden_states_cache.shape == DEFAULT_SHAPE
    assert len(cache.mm_outputs_cache) == 0
    assert len(cache.mm_cache_keys) == 0


def test_initialization_with_multimodal():
    """Check initialization + registration of multimodal outputs."""
    cache = get_omni_pcache()
    feat_dims = {"foo": 100, "bar": 50, "baz": 10}
    mm_outputs = get_multimodal_outputs(
        feat_dims,
        seq_len=DEFAULT_SEQ_LEN,
    )
    # Cast one of the keys to a different dtype; the dtype of the tensor
    # that is used to initialize the cache dictates the cache dtype.
    mm_outputs["foo"] = mm_outputs["foo"].to(OTHER_DTYPE)

    cache.maybe_init_missing_mm_cache_keys(mm_outputs, DEFAULT_SEQ_LEN)
    assert len(cache.mm_cache_keys) == 3
    assert set(cache.mm_cache_keys) == set(feat_dims.keys())
    for mm_key in cache.mm_cache_keys:
        cache_tensor = cache.mm_outputs_cache[mm_key]
        assert isinstance(cache_tensor, torch.Tensor)
        assert cache_tensor.shape[-1] == feat_dims[mm_key]
        assert mm_outputs[mm_key].dtype == cache_tensor.dtype


def test_init_missing_mm_cache_keys_is_idempotent():
    """Ensure that the cache doesn't reinitialize old keys."""
    cache = get_omni_pcache()
    mm_key = "foo"
    feat_dims = {mm_key: 100}
    mm_outputs = get_multimodal_outputs(
        feat_dims,
        seq_len=DEFAULT_SEQ_LEN,
    )
    cache.maybe_init_missing_mm_cache_keys(mm_outputs, DEFAULT_SEQ_LEN)
    assert len(cache.mm_cache_keys) == 1
    assert mm_key in cache.mm_cache_keys

    # Cache is initialized to 0 - fill it with 1s
    cache.mm_outputs_cache[mm_key].fill_(1)

    # Ensure that running another initialization
    # doesn't zero out our cache values
    cache.maybe_init_missing_mm_cache_keys(mm_outputs, DEFAULT_SEQ_LEN)
    assert len(cache.mm_cache_keys) == 1
    assert mm_key in cache.mm_cache_keys
    assert torch.all(cache.mm_outputs_cache[mm_key] == 1)


### Tests for Update
def test_update_no_multimodal():
    """Test that slot mappings act as row indices hidden states."""
    cache = get_omni_pcache()

    num_tokens_unpadded = 8
    slot_offset = 8
    slot_mapping = torch.arange(slot_offset, slot_offset + num_tokens_unpadded)
    new_hidden_states = torch.rand((num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)

    cache.update_omni_tensor_prefix_cache(
        hidden_states=new_hidden_states,
        multimodal_outputs=None,
        num_tokens_unpadded=num_tokens_unpadded,
        slot_mapping=slot_mapping,
    )

    # Ensure that if we reshape our 3D cache back to 2D, we can use the
    # indices in our slot mappings to access the hidden states as expected
    hs_rows = cache.hidden_states_cache.view(NUM_BLOCKS * BLOCK_SIZE, HIDDEN_SIZE)
    for slot_idx, new_states in zip(slot_mapping, new_hidden_states):
        slot_states = hs_rows[slot_idx]
        assert torch.all(slot_states == new_states)


def test_update_uses_precomputed_hidden_states_cpu():
    """Precomputed CPU staging should be the only hidden-state cache source."""
    cache = get_omni_pcache()

    num_tokens_unpadded = 8
    slot_offset = 8
    slot_mapping = torch.arange(slot_offset, slot_offset + num_tokens_unpadded)
    hidden_states = torch.zeros((num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)
    hidden_states_cpu = torch.rand((num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)

    cache.update_omni_tensor_prefix_cache(
        hidden_states=hidden_states,
        hidden_states_cpu=hidden_states_cpu,
        multimodal_outputs=None,
        num_tokens_unpadded=num_tokens_unpadded,
        slot_mapping=slot_mapping,
    )

    hs_rows = cache.hidden_states_cache.view(NUM_BLOCKS * BLOCK_SIZE, HIDDEN_SIZE)
    for slot_idx, new_states in zip(slot_mapping, hidden_states_cpu):
        slot_states = hs_rows[slot_idx]
        assert torch.all(slot_states == new_states)


@pytest.mark.parametrize(
    "hidden_states_cpu",
    [
        torch.rand((4, HIDDEN_SIZE), dtype=DTYPE),
        torch.rand((8, HIDDEN_SIZE), dtype=OTHER_DTYPE),
        torch.rand((8, HIDDEN_SIZE + 1), dtype=DTYPE),
        torch.rand((HIDDEN_SIZE, 8), dtype=DTYPE).t(),
    ],
)
def test_update_rejects_invalid_hidden_states_cpu(hidden_states_cpu):
    cache = get_omni_pcache()

    num_tokens_unpadded = 8
    slot_mapping = torch.arange(num_tokens_unpadded)
    hidden_states = torch.rand((num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)

    with pytest.raises(RuntimeError):
        cache.update_omni_tensor_prefix_cache(
            hidden_states=hidden_states,
            hidden_states_cpu=hidden_states_cpu,
            multimodal_outputs=None,
            num_tokens_unpadded=num_tokens_unpadded,
            slot_mapping=slot_mapping,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_update_rejects_gpu_hidden_states_cpu():
    cache = get_omni_pcache()

    num_tokens_unpadded = 8
    slot_mapping = torch.arange(num_tokens_unpadded)
    hidden_states = torch.rand((num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)
    hidden_states_cpu = torch.rand((num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE, device="cuda")

    with pytest.raises(RuntimeError):
        cache.update_omni_tensor_prefix_cache(
            hidden_states=hidden_states,
            hidden_states_cpu=hidden_states_cpu,
            multimodal_outputs=None,
            num_tokens_unpadded=num_tokens_unpadded,
            slot_mapping=slot_mapping,
        )


@pytest.mark.parametrize(
    "feat_dims",
    [
        {"foo": 100, "bar": 100},
        {"foo": 100, "bar": 50, "baz": 10},
    ],
)
def test_update_with_multimodal_outputs(feat_dims):
    """Test that slot mappings are correct for multimodal tensors."""
    cache = get_omni_pcache_with_mm_tensors(feat_dims, seq_len=DEFAULT_SEQ_LEN)

    num_tokens_unpadded = 8
    slot_offset = 8
    slot_mapping = torch.arange(slot_offset, slot_offset + num_tokens_unpadded)
    feature_dims = {key: val.shape[-1] for key, val in cache.mm_outputs_cache.items()}
    mm_outputs = {key: torch.rand((num_tokens_unpadded, feature_dims[key]), dtype=DTYPE) for key in cache.mm_cache_keys}
    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs=mm_outputs,
        num_tokens_unpadded=num_tokens_unpadded,
        slot_mapping=slot_mapping,
    )

    for mm_key in feat_dims.keys():
        assert mm_key in cache.mm_outputs_cache
        key_feat_dim = feature_dims[mm_key]
        mm_state_rows = cache.mm_outputs_cache[mm_key].view(NUM_BLOCKS * BLOCK_SIZE, key_feat_dim)

        # Similar to hidden states, but for each key in the dict;
        # Different tensors may have different feature dims
        new_mm_outputs = mm_outputs[mm_key]
        for slot_idx, new_output in zip(slot_mapping, new_mm_outputs):
            slot_states = mm_state_rows[slot_idx]
            assert torch.all(slot_states == new_output)


def test_deferred_multimodal_cache_write_commits_on_completion():
    cache = get_omni_pcache()
    mm_key = "codes.audio"
    block_table = torch.tensor(
        [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    input_batch = MockInputBatch(
        num_computed_tokens_cpu=torch.tensor([4, 0], dtype=torch.long),
        block_table=block_table,
    )

    first_chunk = torch.arange(8, dtype=torch.long).reshape(4, 2)
    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs={mm_key: first_chunk},
        num_tokens_unpadded=4,
        slot_mapping=torch.arange(8, 12),
        skip_mm_cache_keys={mm_key},
    )

    rows = cache.mm_outputs_cache[mm_key].view(-1, 2)
    assert torch.all(rows[8:12] == 0)

    cache.stage_deferred_mm_outputs(
        query_start_loc=torch.tensor([0, 4], dtype=torch.long),
        input_batch=input_batch,
        multimodal_outputs={mm_key: first_chunk},
        num_scheduled_tokens={"req1": 4, "req2": 0},
        deferred_mm_cache_keys={mm_key},
    )

    second_chunk = torch.arange(8, 16, dtype=torch.long).reshape(4, 2)
    input_batch.num_computed_tokens_cpu = torch.tensor([8, 0], dtype=torch.long)
    cache.stage_deferred_mm_outputs(
        query_start_loc=torch.tensor([0, 4], dtype=torch.long),
        input_batch=input_batch,
        multimodal_outputs={mm_key: second_chunk},
        num_scheduled_tokens={"req1": 4, "req2": 0},
        deferred_mm_cache_keys={mm_key},
    )

    cache.commit_deferred_mm_outputs({"req1"}, input_batch)

    expected = torch.cat([first_chunk, second_chunk], dim=0)
    rows = cache.mm_outputs_cache[mm_key].view(-1, 2)
    assert torch.equal(rows[8:16], expected)


def test_discard_deferred_mm_outputs_drops_pending_chunks():
    """discard_deferred_mm_outputs is the abort path: a request that
    leaves without a cache commit must not leak its staged chunks into
    the cache if commit_deferred_mm_outputs is later called for it."""
    cache = get_omni_pcache()
    mm_key = "codes.audio"
    block_table = torch.tensor(
        [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    input_batch = MockInputBatch(
        num_computed_tokens_cpu=torch.tensor([4, 0], dtype=torch.long),
        block_table=block_table,
    )

    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs={mm_key: torch.arange(8, dtype=torch.long).reshape(4, 2)},
        num_tokens_unpadded=4,
        slot_mapping=torch.arange(8, 12),
        skip_mm_cache_keys={mm_key},
    )

    chunk = torch.arange(8, dtype=torch.long).reshape(4, 2)
    cache.stage_deferred_mm_outputs(
        query_start_loc=torch.tensor([0, 4], dtype=torch.long),
        input_batch=input_batch,
        multimodal_outputs={mm_key: chunk},
        num_scheduled_tokens={"req1": 4, "req2": 0},
        deferred_mm_cache_keys={mm_key},
    )
    assert "req1" in cache._deferred_mm_outputs

    cache.discard_deferred_mm_outputs("req1")
    assert "req1" not in cache._deferred_mm_outputs

    # A later commit call for the now-discarded req_id must be a no-op,
    # not a resurrection of the dropped chunk.
    cache.commit_deferred_mm_outputs({"req1"}, input_batch)
    rows = cache.mm_outputs_cache[mm_key].view(-1, 2)
    assert torch.all(rows[8:12] == 0)


def test_deferred_mm_outputs_request_id_reused_after_discard():
    """A request ID reused after discard_deferred_mm_outputs must start
    clean: the new occupant's committed data must not be mixed with the
    previous (discarded) occupant's staged chunks."""
    cache = get_omni_pcache()
    mm_key = "codes.audio"
    block_table = torch.tensor(
        [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    input_batch = MockInputBatch(
        num_computed_tokens_cpu=torch.tensor([4, 0], dtype=torch.long),
        block_table=block_table,
    )

    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs={mm_key: torch.arange(8, dtype=torch.long).reshape(4, 2)},
        num_tokens_unpadded=4,
        slot_mapping=torch.arange(8, 12),
        skip_mm_cache_keys={mm_key},
    )

    stale_chunk = torch.full((4, 2), -1, dtype=torch.long)
    cache.stage_deferred_mm_outputs(
        query_start_loc=torch.tensor([0, 4], dtype=torch.long),
        input_batch=input_batch,
        multimodal_outputs={mm_key: stale_chunk},
        num_scheduled_tokens={"req1": 4, "req2": 0},
        deferred_mm_cache_keys={mm_key},
    )
    cache.discard_deferred_mm_outputs("req1")

    # "req1" is reused by a brand-new request with different data.
    fresh_chunk = torch.arange(8, dtype=torch.long).reshape(4, 2)
    cache.stage_deferred_mm_outputs(
        query_start_loc=torch.tensor([0, 4], dtype=torch.long),
        input_batch=input_batch,
        multimodal_outputs={mm_key: fresh_chunk},
        num_scheduled_tokens={"req1": 4, "req2": 0},
        deferred_mm_cache_keys={mm_key},
    )
    cache.commit_deferred_mm_outputs({"req1"}, input_batch)

    rows = cache.mm_outputs_cache[mm_key].view(-1, 2)
    assert torch.equal(rows[8:12], fresh_chunk)


def test_deferred_multimodal_cache_can_be_merged_on_full_block_hit():
    cache = get_omni_pcache()
    mm_key = "codes.audio"
    block_table = torch.tensor(
        [
            [2, 3, 4, 5],
            [6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    input_batch = MockInputBatch(
        num_computed_tokens_cpu=torch.tensor([8, 0], dtype=torch.long),
        block_table=block_table,
    )
    cached_codes = torch.arange(16, dtype=torch.long).reshape(8, 2)
    cache.stage_deferred_mm_outputs(
        query_start_loc=torch.tensor([0, 8], dtype=torch.long),
        input_batch=input_batch,
        multimodal_outputs={mm_key: cached_codes},
        num_scheduled_tokens={"req1": 8, "req2": 0},
        deferred_mm_cache_keys={mm_key},
    )
    cache.commit_deferred_mm_outputs({"req1"}, input_batch)

    new_codes = torch.tensor([[99, 100]], dtype=torch.long)
    cache.add_prefix_cached_new_req_id("req1")
    merged = cache.get_merged_multimodal_states(
        query_start_loc=torch.tensor([0, 1], dtype=torch.long),
        input_batch=input_batch,
        multimodal_outputs={mm_key: new_codes},
        num_scheduled_tokens={"req1": 1, "req2": 0},
    )

    assert torch.equal(merged[mm_key]["req1"], torch.cat([cached_codes, new_codes], dim=0))


### Tests for Merging
def fake_get_cached_block_ids(self, req_idx, *args, **kwargs):
    """Fake block table lookup.

    Assumption:
        req_idx 0 is a cache hit with slots 8, 9, ..., 15
        req_idx 1 is a cache miss
    """
    assert req_idx < 2
    if req_idx == 0:
        # With the slot offset we provided (8), the corresponding
        # blocks IDs are 2 & 3 because the block size is 4.
        return torch.tensor([2, 3], dtype=torch.long)
    return torch.tensor([], dtype=torch.long)


@pytest.mark.parametrize("num_tokens_padded", [None, 16])
def test_get_merged_hidden_states(num_tokens_padded, mocker):
    """Ensure that hidden states are merged correctly."""
    cache = get_omni_pcache()

    orig_num_tokens_unpadded = 8
    slot_offset = 8  # We'll put our states in slots 8, 9, 10, ..., 15
    orig_slot_mapping = torch.arange(slot_offset, slot_offset + orig_num_tokens_unpadded)
    orig_hidden_states = torch.rand((orig_num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)

    cache.update_omni_tensor_prefix_cache(
        hidden_states=orig_hidden_states,
        multimodal_outputs=None,
        num_tokens_unpadded=orig_num_tokens_unpadded,
        slot_mapping=orig_slot_mapping,
        num_tokens_padded=num_tokens_padded,
    )

    # Say that we have two requests, but only one of them is a cache hit
    num_new_toks_req1 = 3
    num_new_toks_req2 = 2
    cache.add_prefix_cached_new_req_id("req1")

    num_scheduled_tokens = {
        "req1": num_new_toks_req1,
        "req2": num_new_toks_req2,
    }
    new_hidden_states = torch.rand(
        (num_new_toks_req1 + num_new_toks_req2, HIDDEN_SIZE),
        dtype=DTYPE,
    )
    req1_new_states = new_hidden_states[:num_new_toks_req1]
    req2_new_states = new_hidden_states[-num_new_toks_req2:]

    input_batch = MockInputBatch(num_computed_tokens_cpu=torch.Tensor([orig_num_tokens_unpadded, 0]))

    mocker.patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    )
    merged_states = cache.get_merged_hidden_states(
        query_start_loc=[0, num_new_toks_req1],
        input_batch=input_batch,
        hidden_states=new_hidden_states,
        num_scheduled_tokens=num_scheduled_tokens,
    )

    assert "req1" in merged_states and "req2" in merged_states
    req1_merged_states = merged_states["req1"]
    req2_merged_states = merged_states["req2"]

    # First, check the cache hit case
    assert req1_merged_states.shape == torch.Size([orig_num_tokens_unpadded + num_new_toks_req1, HIDDEN_SIZE])
    # Ensure that the req1 merged states are the cached states + the new req1 states
    assert torch.all(req1_merged_states[:orig_num_tokens_unpadded] == orig_hidden_states)
    assert torch.all(req1_merged_states[-num_new_toks_req1:] == req1_new_states)

    # Next, ensure that the cache miss case only has the new states
    assert req2_merged_states.shape == torch.Size([num_new_toks_req2, HIDDEN_SIZE])
    assert torch.all(req2_merged_states == req2_new_states)


def test_get_merged_hidden_states_uses_precomputed_hidden_states_cpu(mocker):
    cache = get_omni_pcache()

    orig_num_tokens_unpadded = 8
    slot_offset = 8
    orig_slot_mapping = torch.arange(slot_offset, slot_offset + orig_num_tokens_unpadded)
    orig_hidden_states = torch.rand((orig_num_tokens_unpadded, HIDDEN_SIZE), dtype=DTYPE)

    cache.update_omni_tensor_prefix_cache(
        hidden_states=orig_hidden_states,
        multimodal_outputs=None,
        num_tokens_unpadded=orig_num_tokens_unpadded,
        slot_mapping=orig_slot_mapping,
    )

    num_new_toks_req1 = 3
    num_new_toks_req2 = 2
    cache.add_prefix_cached_new_req_id("req1")
    num_scheduled_tokens = {
        "req1": num_new_toks_req1,
        "req2": num_new_toks_req2,
    }
    new_hidden_states = torch.zeros(
        (num_new_toks_req1 + num_new_toks_req2, HIDDEN_SIZE),
        dtype=DTYPE,
    )
    hidden_states_cpu = torch.rand_like(new_hidden_states)
    req1_new_states = hidden_states_cpu[:num_new_toks_req1]
    req2_new_states = hidden_states_cpu[-num_new_toks_req2:]
    input_batch = MockInputBatch(num_computed_tokens_cpu=torch.Tensor([orig_num_tokens_unpadded, 0]))

    mocker.patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    )
    merged_states = cache.get_merged_hidden_states(
        query_start_loc=[0, num_new_toks_req1],
        input_batch=input_batch,
        hidden_states=new_hidden_states,
        hidden_states_cpu=hidden_states_cpu,
        num_scheduled_tokens=num_scheduled_tokens,
    )

    assert torch.all(merged_states["req1"][:orig_num_tokens_unpadded] == orig_hidden_states)
    assert torch.all(merged_states["req1"][-num_new_toks_req1:] == req1_new_states)
    assert torch.all(merged_states["req2"] == req2_new_states)


def test_get_merged_hidden_states_rejects_short_hidden_states_cpu(mocker):
    cache = get_omni_pcache()
    num_scheduled_tokens = {
        "req1": 3,
        "req2": 2,
    }
    input_batch = MockInputBatch(num_computed_tokens_cpu=torch.Tensor([0, 0]))

    mocker.patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    )
    with pytest.raises(RuntimeError):
        cache.get_merged_hidden_states(
            query_start_loc=[0, 3],
            input_batch=input_batch,
            hidden_states=torch.rand((5, HIDDEN_SIZE), dtype=DTYPE),
            hidden_states_cpu=torch.rand((4, HIDDEN_SIZE), dtype=DTYPE),
            num_scheduled_tokens=num_scheduled_tokens,
        )


@pytest.mark.parametrize("num_tokens_padded", [None, 16])
@pytest.mark.parametrize(
    "feat_dims",
    [
        {"foo": 100, "bar": 100},
        {"foo": 100, "bar": 50, "baz": 10},
    ],
)
def test_get_merged_multimodal_outputs(feat_dims, num_tokens_padded, mocker):
    cache = get_omni_pcache_with_mm_tensors(feat_dims, seq_len=DEFAULT_SEQ_LEN)

    orig_num_tokens_unpadded = 8
    slot_offset = 8  # We'll put our states in slots 8, 9, 10, ..., 15
    orig_slot_mapping = torch.arange(slot_offset, slot_offset + orig_num_tokens_unpadded)
    feature_dims = {key: val.shape[-1] for key, val in cache.mm_outputs_cache.items()}
    orig_mm_outputs = {
        key: torch.rand((orig_num_tokens_unpadded, feature_dims[key]), dtype=DTYPE) for key in cache.mm_cache_keys
    }

    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs=orig_mm_outputs,
        num_tokens_unpadded=orig_num_tokens_unpadded,
        slot_mapping=orig_slot_mapping,
        num_tokens_padded=num_tokens_padded,
    )

    # Similar to hs test- say that we have two requests, but only one of them is a cache hit
    num_new_toks_req1 = 3
    num_new_toks_req2 = 2
    cache.add_prefix_cached_new_req_id("req1")

    num_scheduled_tokens = {
        "req1": num_new_toks_req1,
        "req2": num_new_toks_req2,
    }

    new_mm_outputs = {}
    for mm_key in cache.mm_cache_keys:
        new_mm_outputs[mm_key] = torch.rand(
            (num_new_toks_req1 + num_new_toks_req2, feature_dims[mm_key]),
            dtype=DTYPE,
        )
    # We also want to make sure passthrough data (outside of our keys) isn't dropped
    new_mm_outputs["passthrough_data"] = "Something else"
    # Lists are a special case because we can't split them yet if we want to match
    # the nonprefix cache behavior, because this runs before post process.
    new_mm_outputs["passthrough_list"] = ["should", "not", "split"]

    input_batch = MockInputBatch(num_computed_tokens_cpu=torch.Tensor([orig_num_tokens_unpadded, 0]))

    mocker.patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    )
    merged_mm_outputs = cache.get_merged_multimodal_states(
        query_start_loc=[0, num_new_toks_req1],
        input_batch=input_batch,
        multimodal_outputs=new_mm_outputs,
        num_scheduled_tokens=num_scheduled_tokens,
    )

    # Ensure the passthrough data wasn't dropped
    assert "passthrough_data" in merged_mm_outputs
    assert "passthrough_list" in merged_mm_outputs

    for mm_key, mm_output in merged_mm_outputs.items():
        # Ensure passthrough data is just forwarded normally and not duplicated
        assert isinstance(mm_output, dict)
        assert "req1" in mm_output and "req2" in mm_output
        if mm_key == "passthrough_data":
            assert mm_key not in cache.mm_cache_keys
            assert new_mm_outputs[mm_key] == mm_output["req1"]
            assert new_mm_outputs[mm_key] == mm_output["req2"]
        elif mm_key == "passthrough_list":
            assert mm_key not in cache.mm_cache_keys
            assert new_mm_outputs[mm_key] == mm_output["req1"]
            assert new_mm_outputs[mm_key] == mm_output["req2"]
        else:
            assert mm_key in cache.mm_cache_keys
            curr_feat_dim = feature_dims[mm_key]
            # Ensure that req1 (cache hit) merged the mm data
            req1_merged_mm_outputs = mm_output["req1"]
            req1_new_mm_outputs = new_mm_outputs[mm_key][:num_new_toks_req1]

            assert req1_merged_mm_outputs.shape == torch.Size(
                [orig_num_tokens_unpadded + num_new_toks_req1, curr_feat_dim]
            )
            # Ensure that the req1 merged mm data are the cached data + the new data
            assert torch.all(req1_merged_mm_outputs[:orig_num_tokens_unpadded] == orig_mm_outputs[mm_key])
            assert torch.all(req1_merged_mm_outputs[-num_new_toks_req1:] == req1_new_mm_outputs)

            # Ensure that req2 (cache miss) only has the new mm data
            req2_merged_mm_outputs = mm_output["req2"]
            req2_new_mm_outputs = new_mm_outputs[mm_key][-num_new_toks_req2:]

            assert req2_merged_mm_outputs.shape == torch.Size([num_new_toks_req2, curr_feat_dim])
            assert torch.all(req2_merged_mm_outputs == req2_new_mm_outputs)


@pytest.mark.parametrize(
    "passthrough_shape",
    [(5,), (5, 3)],
    ids=["1d_token_metadata", "2d_feature_tensor"],
)
def test_get_merged_multimodal_outputs_slices_passthrough_tensor_per_request(passthrough_shape, mocker):
    """A passthrough key that is itself a real tensor (not the string/list
    passthrough already covered above) must be sliced per request through
    to_payload_element's tensor branch, under the same mixed cache
    hit/miss + uneven scheduled-length batch as test_get_merged_multimodal_outputs.

    Covers both a 1D token-aligned metadata tensor (e.g. ref_code_len,
    codec_streaming -- shape (total_tokens,)) and a 2D per-token feature
    tensor (shape (total_tokens, feat_dim)), since W2's batching could
    easily special-case one and silently mishandle the other.
    """
    feat_dims = {"foo": 100, "bar": 100}
    cache = get_omni_pcache_with_mm_tensors(feat_dims, seq_len=DEFAULT_SEQ_LEN)

    orig_num_tokens_unpadded = 8
    slot_offset = 8
    orig_slot_mapping = torch.arange(slot_offset, slot_offset + orig_num_tokens_unpadded)
    feature_dims = {key: val.shape[-1] for key, val in cache.mm_outputs_cache.items()}
    orig_mm_outputs = {
        key: torch.rand((orig_num_tokens_unpadded, feature_dims[key]), dtype=DTYPE) for key in cache.mm_cache_keys
    }

    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs=orig_mm_outputs,
        num_tokens_unpadded=orig_num_tokens_unpadded,
        slot_mapping=orig_slot_mapping,
        num_tokens_padded=None,
    )

    num_new_toks_req1 = 3
    num_new_toks_req2 = 2
    cache.add_prefix_cached_new_req_id("req1")

    num_scheduled_tokens = {
        "req1": num_new_toks_req1,
        "req2": num_new_toks_req2,
    }

    new_mm_outputs = {}
    for mm_key in cache.mm_cache_keys:
        new_mm_outputs[mm_key] = torch.rand(
            (num_new_toks_req1 + num_new_toks_req2, feature_dims[mm_key]),
            dtype=DTYPE,
        )
    total_scheduled_tokens = num_new_toks_req1 + num_new_toks_req2
    assert passthrough_shape[0] == total_scheduled_tokens
    passthrough_tensor = torch.arange(int(torch.prod(torch.tensor(passthrough_shape))), dtype=DTYPE).reshape(
        passthrough_shape
    )
    new_mm_outputs["passthrough_tensor"] = passthrough_tensor

    input_batch = MockInputBatch(num_computed_tokens_cpu=torch.Tensor([orig_num_tokens_unpadded, 0]))

    mocker.patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    )
    merged_mm_outputs = cache.get_merged_multimodal_states(
        query_start_loc=[0, num_new_toks_req1],
        input_batch=input_batch,
        multimodal_outputs=new_mm_outputs,
        num_scheduled_tokens=num_scheduled_tokens,
    )

    assert "passthrough_tensor" not in cache.mm_cache_keys

    passthrough_out = merged_mm_outputs["passthrough_tensor"]
    # req1 (cache hit) gets tokens [0:3), req2 (cache miss) gets tokens [3:5) --
    # a passthrough tensor is sliced by scheduled position, not merged with
    # any cached history (unlike the mm_cache_keys tensors above).
    assert torch.equal(passthrough_out["req1"], passthrough_tensor[0:num_new_toks_req1])
    assert torch.equal(passthrough_out["req2"], passthrough_tensor[num_new_toks_req1:total_scheduled_tokens])


class MergeHarness(NamedTuple):
    """A primed cache plus the arguments get_merged_multimodal_states needs."""

    cache: OmniTensorPrefixCache
    input_batch: MockInputBatch
    num_scheduled_tokens: dict[str, int]
    query_start_loc: list[int]
    total_scheduled_tokens: int
    new_mm_outputs: dict[str, object]


def build_mixed_hit_miss_harness(mocker, num_new_toks_req1: int = 3, num_new_toks_req2: int = 2) -> MergeHarness:
    """Two requests over one cache key: req1 hits 8 cached tokens, req2 misses,
    and their scheduled lengths differ -- the same batch shape as
    test_get_merged_multimodal_outputs, which is what makes a per-request
    boundary error observable. Callers add their passthrough keys to
    ``new_mm_outputs`` before calling get_merged_multimodal_states.
    """
    cache = get_omni_pcache_with_mm_tensors({"foo": 100}, seq_len=DEFAULT_SEQ_LEN)

    orig_num_tokens_unpadded = 8
    feature_dims = {key: val.shape[-1] for key, val in cache.mm_outputs_cache.items()}
    cache.update_omni_tensor_prefix_cache(
        hidden_states=None,
        multimodal_outputs={
            key: torch.rand((orig_num_tokens_unpadded, feature_dims[key]), dtype=DTYPE) for key in cache.mm_cache_keys
        },
        num_tokens_unpadded=orig_num_tokens_unpadded,
        slot_mapping=torch.arange(8, 8 + orig_num_tokens_unpadded),
        num_tokens_padded=None,
    )
    cache.add_prefix_cached_new_req_id("req1")

    total_scheduled_tokens = num_new_toks_req1 + num_new_toks_req2
    mocker.patch(
        "vllm_omni.core.prefix_cache.OmniTensorPrefixCache._get_cached_block_ids",
        new=fake_get_cached_block_ids,
    )
    return MergeHarness(
        cache=cache,
        input_batch=MockInputBatch(num_computed_tokens_cpu=torch.Tensor([orig_num_tokens_unpadded, 0])),
        num_scheduled_tokens={"req1": num_new_toks_req1, "req2": num_new_toks_req2},
        query_start_loc=[0, num_new_toks_req1],
        total_scheduled_tokens=total_scheduled_tokens,
        new_mm_outputs={
            key: torch.rand((total_scheduled_tokens, feature_dims[key]), dtype=DTYPE) for key in cache.mm_cache_keys
        },
    )


def test_get_merged_multimodal_outputs_passes_per_request_tensor_list_through(mocker):
    """The passthrough shape qwen3-tts actually emits: a nested ``codes``/``meta``
    dict whose ``codes.ref`` holds one tensor per request
    (qwen3_tts_talker.make_omni_output), carried through to_payload_element's
    pass_lists_through branch beside 1D int metadata that IS sliced per request.

    Each request must receive the list *whole*. Per-request indexing happens
    later, in the runner's _unwrap_lists; a W2 that split the list here would
    make that second index select a different request's reference codes. The
    existing list coverage passes a list of strings, which cannot catch a
    tensor-specific regression, and no other test covers an int dtype.
    """
    harness = build_mixed_hit_miss_harness(mocker)

    ref_codes = [torch.tensor([11, 12], dtype=torch.long), torch.tensor([21], dtype=torch.long)]
    ref_code_len = torch.arange(harness.total_scheduled_tokens, dtype=torch.int32)
    harness.new_mm_outputs["codes"] = {"ref": ref_codes}
    harness.new_mm_outputs["meta"] = {"ref_code_len": ref_code_len}

    merged = harness.cache.get_merged_multimodal_states(
        query_start_loc=harness.query_start_loc,
        input_batch=harness.input_batch,
        multimodal_outputs=harness.new_mm_outputs,
        num_scheduled_tokens=harness.num_scheduled_tokens,
    )

    assert "codes" not in harness.cache.mm_cache_keys
    for req_id in ("req1", "req2"):
        req_refs = merged["codes"][req_id]["ref"]
        assert len(req_refs) == len(ref_codes)
        for got, want in zip(req_refs, ref_codes):
            assert torch.equal(got, want)
            assert got.dtype == torch.long

    # Per-request clones: the branch exists so one request's downstream mutation
    # cannot reach another request's payload or the source list.
    merged["codes"]["req1"]["ref"][0] += 100
    assert torch.equal(merged["codes"]["req2"]["ref"][0], torch.tensor([11, 12], dtype=torch.long))
    assert torch.equal(ref_codes[0], torch.tensor([11, 12], dtype=torch.long))

    # Token-aligned 1D metadata is still split per request, unlike the list.
    req1_len = harness.num_scheduled_tokens["req1"]
    assert torch.equal(merged["meta"]["req1"]["ref_code_len"], ref_code_len[:req1_len])
    assert torch.equal(merged["meta"]["req2"]["ref_code_len"], ref_code_len[req1_len:])
    assert merged["meta"]["req1"]["ref_code_len"].dtype == torch.int32


@pytest.mark.parametrize(
    "passthrough_shape",
    [(3, 4), (1, 4), (7,)],
    ids=["shorter_2d", "single_row_2d", "longer_1d"],
)
def test_get_merged_multimodal_outputs_broadcasts_non_token_aligned_passthrough(passthrough_shape, mocker):
    """A passthrough tensor whose first dimension is not the scheduled-token
    count is request-invariant: to_payload_element hands every request the whole
    tensor, cloned, rather than slicing it.

    W2 batching most plausibly breaks by assuming every passthrough tensor is
    token-aligned; that assumption would slice these and give each request a
    truncated view, or an empty one once start exceeds the first dimension.
    """
    harness = build_mixed_hit_miss_harness(mocker)
    assert passthrough_shape[0] != harness.total_scheduled_tokens

    passthrough = torch.arange(math.prod(passthrough_shape), dtype=DTYPE).reshape(passthrough_shape)
    expected = passthrough.clone()
    harness.new_mm_outputs["speaker_embedding"] = passthrough

    merged = harness.cache.get_merged_multimodal_states(
        query_start_loc=harness.query_start_loc,
        input_batch=harness.input_batch,
        multimodal_outputs=harness.new_mm_outputs,
        num_scheduled_tokens=harness.num_scheduled_tokens,
    )

    assert "speaker_embedding" not in harness.cache.mm_cache_keys
    out = merged["speaker_embedding"]
    for req_id in ("req1", "req2"):
        assert out[req_id].shape == passthrough_shape
        assert torch.equal(out[req_id], expected)

    out["req1"] += 100
    assert torch.equal(out["req2"], expected)
    assert torch.equal(passthrough, expected)
