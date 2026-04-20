import pytest
import torch

from vllm_omni.core.prefix_cache import OmniTensorPrefixCache

DEFAULT_SEQ_LEN = 15
NUM_BLOCKS = 10
BLOCK_SIZE = 4
HIDDEN_SIZE = 2
DTYPE = torch.float32
OTHER_DTYPE = torch.float16
DEFAULT_SHAPE = torch.Size([NUM_BLOCKS, BLOCK_SIZE, HIDDEN_SIZE])


class MockInputBatch:
    def __init__(self, num_computed_tokens_cpu):
        self.req_ids = ["req1", "req2"]
        self.req_id_to_index = {req_id: i for i, req_id in enumerate(self.req_ids)}
        self.num_computed_tokens_cpu = num_computed_tokens_cpu

        # Block table is only mocked for validation of length;
        # we don't actually need to add valid values here since
        # we patch the table when testing.
        class _DummyBlockTable:
            pass

        self.block_table = _DummyBlockTable()
        self.block_table.block_tables = [None]


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
