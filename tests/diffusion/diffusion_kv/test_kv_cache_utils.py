# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native multimodal block hashing and diffusion-specific identity helpers."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import BlockHash, generate_block_hash_extra_keys, hash_block_tokens

import vllm_omni.diffusion.diffusion_engine as engine_module
import vllm_omni.diffusion.diffusion_kv.kv_cache_utils as kv_utils
import vllm_omni.diffusion.diffusion_kv.request as request_module
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.kv_cache_utils import get_cache_namespace, hash_prefix_cache_value
from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _request(*, num_rows=1, seed=7, adapter_id=None, scale=1.0):
    request = OmniDiffusionRequest(
        request_id="fake", prompt="fake model", sampling_params=OmniDiffusionSamplingParams(seed=seed)
    )
    request.sampling_params.lora_request = None if adapter_id is None else SimpleNamespace(lora_int_id=adapter_id)
    request.sampling_params.lora_scale = scale
    request.diffusion_kv_requests = tuple(
        DiffusionKVRequest(f"fake/{row}", sequence_id=row, prefix_len=8, target_len=4, seq_len=12)
        for row in range(num_rows)
    )
    return request


def _feature(*, value="image", start=4, end=8):
    return MultiModalFeatureSpec(
        data=None,
        modality="image",
        identifier=hash_prefix_cache_value((value, start, end)).hex(),
        mm_position=PlaceholderRange(offset=start, length=end - start),
    )


def _prepare(request, feature=None):
    namespace = get_cache_namespace("fake-model-v1", request.sampling_params)
    for row in request.diffusion_kv_requests:
        row.cache_token_ids = tuple(range(row.prefix_len))
        row.mm_features = [feature or _feature()]
        row.cache_namespace = namespace


def _hashes(request, feature=None):
    _prepare(request, feature)
    row = request.diffusion_kv_requests[0]
    row.build_block_hashes(4, get_hash_fn_by_name("sha256"))
    return row.block_hashes


def test_partial_hits_and_randomness_are_model_opt_in():
    base = _hashes(_request())
    assert base == _hashes(_request(seed=999))
    changed = _hashes(_request(), _feature(value="other image"))
    assert base[0] == changed[0]
    assert base[1] != changed[1]
    changed_geometry = _hashes(_request(), _feature(end=9))
    assert base[0] == changed_geometry[0]
    assert base[1] != changed_geometry[1]


@pytest.mark.parametrize(("adapter_id", "scale"), [(None, 1.0), (2, 0.5), (1, 1.0)])
def test_lora_is_common_request_wide_identity(adapter_id, scale):
    base = _hashes(_request(adapter_id=1, scale=0.5))
    changed = _hashes(_request(adapter_id=adapter_id, scale=scale))
    assert all(a != b for a, b in zip(base, changed, strict=True))
    assert _hashes(_request()) == _hashes(_request(scale=0.5))


def test_block_hashing_uses_upstream_extra_keys_directly(monkeypatch):
    row = DiffusionKVRequest(
        "native",
        sequence_id=0,
        prefix_len=14,
        target_len=2,
        seq_len=16,
        cache_token_ids=range(16),
        mm_features=[
            _feature(start=2, end=7),
            _feature(start=7, end=10),
        ],
    )
    upstream = Mock(wraps=generate_block_hash_extra_keys)
    monkeypatch.setattr(request_module, "generate_block_hash_extra_keys", upstream)
    hash_fn = get_hash_fn_by_name("sha256")
    row.build_block_hashes(4, hash_fn)
    assert [(call.args[1], call.args[2]) for call in upstream.call_args_list] == [(0, 4), (4, 8), (8, 12)]
    # Independent expected native extras: a crossing item has a negative
    # relative offset; a block can include the end of one item and the next.
    first, second = (feature.identifier for feature in row.mm_features)
    extras = [((first, 2),), ((first, -2), (second, 3)), ((second, -1),)]
    parent = BlockHash(hash_fn(("diffusion-prefix-root-v1", row.cache_namespace)))
    expected = []
    for start, keys in zip(range(0, 12, 4), extras, strict=True):
        parent = hash_block_tokens(hash_fn, parent, list(range(start, start + 4)), keys)
        expected.append(parent)
    assert row.block_hashes == expected
    # Never hash the partial prefix tail or mutable target [14, 16), even
    # though cache_token_ids deliberately contains the whole sequence.
    assert len(row.block_hashes) == 3


def test_same_identifier_at_different_offsets_changes_block_hash():
    first = _feature(start=4, end=6)
    shifted = _feature(start=5, end=7)
    shifted.identifier = first.identifier
    a = _hashes(_request(), first)
    b = _hashes(_request(), shifted)
    assert a[0] == b[0]
    assert a[1] != b[1]


def test_multimodal_identity_beyond_prefix_does_not_affect_hashes():
    assert _hashes(_request(), _feature(value="a", start=8, end=12)) == _hashes(
        _request(), _feature(value="b", start=8, end=12)
    )


@pytest.mark.parametrize(
    "positions",
    [
        [(-1, 2)],
        [(4, 4)],
        [(4, 13)],
        [(5, 7), (4, 5)],
        [(3, 7), (4, 6)],
    ],
)
def test_invalid_or_overlapping_positions_fail_before_hashing(positions):
    row = _request().diffusion_kv_requests[0]
    row.cache_token_ids = tuple(range(8))
    row.mm_features = [_feature(start=start, end=end) for start, end in positions]
    fail = Mock(side_effect=AssertionError("invalid positions must fail before hashing"))
    with pytest.raises(ValueError, match="sorted, non-overlapping"):
        row.build_block_hashes(4, fail)
    fail.assert_not_called()
    assert row.block_hashes == []


def test_empty_multimodal_identifier_is_rejected():
    row = _request().diffusion_kv_requests[0]
    row.cache_token_ids = tuple(range(8))
    feature = _feature()
    feature.identifier = ""
    row.mm_features = [feature]
    with pytest.raises(ValueError, match="content identifiers"):
        row.build_block_hashes(4, get_hash_fn_by_name("sha256"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool])
def test_tensor_identity_includes_dtype_shape_and_values(dtype):
    tensor = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    assert hash_prefix_cache_value(tensor) == hash_prefix_cache_value(tensor.clone())
    assert hash_prefix_cache_value(tensor) == hash_prefix_cache_value(tensor.t().contiguous().t())
    assert hash_prefix_cache_value(tensor) != hash_prefix_cache_value(tensor.reshape(4))
    changed = tensor.clone()
    changed[0, 0] = 1
    assert hash_prefix_cache_value(tensor) != hash_prefix_cache_value(changed)
    assert hash_prefix_cache_value(tensor) != hash_prefix_cache_value(tensor.to(torch.float64))


def test_numpy_scalar_shape_and_nested_dict_order():
    assert hash_prefix_cache_value({"b": {"y": 2, "x": 1}, "a": None}) == hash_prefix_cache_value(
        {"a": None, "b": {"x": 1, "y": 2}}
    )
    assert hash_prefix_cache_value(np.array(1)) != hash_prefix_cache_value(np.array([1]))
    assert hash_prefix_cache_value(np.int64(1)) == hash_prefix_cache_value(1)


@pytest.mark.parametrize(
    ("first", "second"), [(1, True), (1, 1.0), ("1", b"1"), ([], {}), ([None], []), (["ab", "c"], ["a", "bc"])]
)
def test_identity_has_unambiguous_type_and_container_boundaries(first, second):
    assert hash_prefix_cache_value(first) != hash_prefix_cache_value(second)


@pytest.mark.parametrize("value", [object(), {1: "value"}, np.array([object()]), np.longdouble(1)])
def test_identity_rejects_unsupported_values_without_pickle(value):
    with pytest.raises(TypeError):
        hash_prefix_cache_value(value)


@pytest.mark.parametrize(
    ("mode", "enabled", "expect_hook"),
    [
        (DiffusionKVCacheMode.PAGED_SCHEDULER, False, False),
        (DiffusionKVCacheMode.PAGED_SCHEDULER, True, True),
        (DiffusionKVCacheMode.DENSE_LEGACY, True, False),
    ],
)
def test_engine_gates_hook_loading_and_admission(monkeypatch, mode, enabled, expect_hook):
    events = []
    request = _request()

    def preprocess(req):
        events.append("layout")
        req.prepared_layout = "fake layout"
        return req

    def prepare(req):
        assert req.prepared_layout == "fake layout"
        assert req.diffusion_kv_requests[0].cache_token_ids == ()
        events.append("cache_inputs")
        _prepare(req)

    hook = Mock(side_effect=prepare if expect_hook else AssertionError("disabled hook"))
    load_hook = Mock(return_value=hook)
    if not expect_hook:
        load_hook.side_effect = AssertionError("disabled hook factory")
        monkeypatch.setattr(kv_utils, "hash_prefix_cache_value", Mock(side_effect=AssertionError("disabled hash")))
    monkeypatch.setattr(engine_module, "get_diffusion_pre_process_func", lambda _config: preprocess)
    monkeypatch.setattr(engine_module, "get_diffusion_post_process_func", lambda _config: None)
    monkeypatch.setattr(engine_module, "get_diffusion_prefix_cache_func", load_hook)
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(diffusion_kv_mode=mode, enable_prefix_caching=enabled)
    engine._init_process_hooks(engine.od_config)
    engine.prefix_cache_func = hook
    assert engine._prepare_request_for_admission(request) is request
    row = request.diffusion_kv_requests[0]
    if expect_hook:
        assert events == ["layout", "cache_inputs"]
        assert row.cache_token_ids == tuple(range(8))
        load_hook.assert_called_once_with(engine.od_config)
        hook.assert_called_once_with(request)
    else:
        assert events == ["layout"]
        assert row.cache_token_ids == ()
        assert row.mm_features == []
        load_hook.assert_not_called()
        hook.assert_not_called()


def test_engine_rejects_out_of_profile_request_before_hashing():
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER, enable_prefix_caching=True
    )
    engine._diffusion_kv_profile_limits = (1, 8, 4)
    engine.prefix_cache_func = Mock(side_effect=AssertionError("invalid request must fail before identity work"))
    with pytest.raises(ValueError, match="exceeds the startup memory-profile envelope"):
        engine._prepare_request_for_admission(_request())
    engine.prefix_cache_func.assert_not_called()
