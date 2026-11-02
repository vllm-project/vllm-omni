# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model Runner V2 binding of the omni prefix cache."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput

import vllm_omni.worker_v2.prefix_cache as mrv2_prefix_cache
from tests.worker_v2.test_omni_ar_model_runner import _async_output
from vllm_omni.core.prefix_cache import MRv2FullAttentionGroupView
from vllm_omni.core.prefix_cache.interface import ModelCachePolicy, OmniPrefixCacheUnmatchError, PrefixCacheConfig
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner
from vllm_omni.worker_v2.prefix_cache import PrefixCacheRewrite

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakePrefixCache:
    """Merged rows: each rewritten request gets ``prefix[req_id]``."""

    def __init__(self, rewrite_ids=(), *, prefix=None, fail=False):
        self.rewrite_ids = frozenset(rewrite_ids)
        self.prefix = prefix or {}
        self.fail = fail
        self.consumed: list[tuple[str, int]] = []

    def step_rewrite_req_ids(self, step_id):
        return self.rewrite_ids

    def materialize(self, step_id, req_ids, *, include_uncached=True):
        assert include_uncached is False
        self.consumed.append(("materialize", step_id))
        if self.fail:
            raise RuntimeError("materialize failed")
        hidden = {r: self.prefix[r]["hidden"] for r in req_ids}
        mm = {"codes.audio": {r: self.prefix[r]["codes.audio"] for r in req_ids}}
        return SimpleNamespace(hidden_states=hidden, mm_outputs=mm)

    def discard_step(self, step_id):
        self.consumed.append(("discard", step_id))


@pytest.mark.parametrize("opted_in", [False, True])
def test_policy_is_native_kv_only_unless_model_opts_in(opted_in):
    runner = object.__new__(OmniGPUModelRunner)
    runner._snapshot_prefix_cache_model_policy(SimpleNamespace(supports_mrv2_prefix_cache_outputs=opted_in))
    assert runner._omni_cache_policy.native_kv_cache_only is (not opted_in)
    assert runner._omni_cache_policy.needs_full_hidden_states is opted_in


def test_tracking_adapter_and_ledger_view():
    runner = object.__new__(OmniGPUModelRunner)
    adapter = runner._new_prefix_cache_adapter()
    assert adapter._track
    view = runner._build_prefix_cache_group_view(SimpleNamespace(block_size=16), adapter)
    assert isinstance(view, MRv2FullAttentionGroupView) and view.block_size == 16


def test_shutdown_releases_cache_after_runner_resources(monkeypatch):
    calls = []
    runner = object.__new__(OmniGPUModelRunner)
    runner._omni_data_plane = SimpleNamespace(close=lambda: calls.append("plane"))
    runner._shutdown_omni_prefix_cache = lambda: calls.append("prefix_cache")
    monkeypatch.setattr(GPUModelRunner, "shutdown", lambda self: calls.append("upstream"))
    runner.shutdown()
    assert calls == ["plane", "prefix_cache", "upstream"]


def _init_runner(monkeypatch, cfg, *, block_sizes=(16,), dcp_size=1, concurrent=2):
    runner = object.__new__(OmniGPUModelRunner)
    for name in ("kv_cache_config", "cache_config", "scheduler_config", "model_config"):
        setattr(runner, name, SimpleNamespace())
    runner.is_pooling_model = False
    runner.vllm_config = SimpleNamespace(max_concurrent_batches=concurrent, speculative_config=None)
    runner.block_tables = SimpleNamespace(block_sizes=list(block_sizes))
    runner.dcp_size = dcp_size
    runner._shutdown_omni_prefix_cache = MagicMock()
    runner._ensure_omni_prefix_cache = MagicMock()
    monkeypatch.setattr(mrv2_prefix_cache, "stage_prefix_cache_config", MagicMock(return_value=cfg))
    return runner


def test_init_stages_config_with_async_depth(monkeypatch):
    runner = _init_runner(monkeypatch, PrefixCacheConfig(num_blocks=8, block_size=16), concurrent=3)
    runner._init_omni_prefix_cache(is_profiling=False)
    runner._shutdown_omni_prefix_cache.assert_called_once()
    runner._ensure_omni_prefix_cache.assert_called_once()
    assert runner._omni_prefix_cache_cfg.staging_depth == 7

    runner = _init_runner(monkeypatch, PrefixCacheConfig(num_blocks=8, block_size=16))
    runner._init_omni_prefix_cache(is_profiling=True)
    runner._ensure_omni_prefix_cache.assert_not_called()

    runner = _init_runner(monkeypatch, None)
    runner._init_omni_prefix_cache(is_profiling=False)
    runner._ensure_omni_prefix_cache.assert_not_called()


@pytest.mark.parametrize(
    ("block_sizes", "dcp_size", "match"), [((32,), 1, "does not match"), ((16,), 2, "decode context")]
)
def test_init_refuses_unmirrored_layouts(monkeypatch, block_sizes, dcp_size, match):
    cfg = PrefixCacheConfig(num_blocks=8, block_size=16)
    runner = _init_runner(monkeypatch, cfg, block_sizes=block_sizes, dcp_size=dcp_size)
    with pytest.raises(OmniPrefixCacheUnmatchError, match=match):
        runner._init_omni_prefix_cache(is_profiling=False)
    runner._ensure_omni_prefix_cache.assert_not_called()


@pytest.mark.parametrize(
    ("rewrite_ids", "output_type", "written_in_sample", "kept"),
    [
        ({"b"}, "latent", False, True),
        (set(), "latent", False, False),
        ({"b"}, "text", False, False),
        ({"b"}, "latent", True, False),
    ],
)
def test_save_keeps_step_only_for_rewrites(rewrite_ids, output_type, written_in_sample, kept):
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner.omni_prefix_cache = cache = _FakePrefixCache(rewrite_ids)
    runner._prefix_cache_group_view = view = MagicMock()
    runner._omni_cache_policy = ModelCachePolicy(
        needs_full_hidden_states=not written_in_sample, mm_outputs_written_in_sample=written_in_sample
    )
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(engine_output_type=output_type))
    runner._prefix_cache_save_step = MagicMock(return_value=9)
    batch = SimpleNamespace(req_ids=["a", "b", "pad"], num_reqs=2, num_tokens=3, num_tokens_after_padding=4)

    rewrite = runner._prefix_cache_save_mrv2(batch, torch.zeros(4, 2), {})

    assert [c.args[0] for c in view.bind.call_args_list] == [batch, None]
    assert runner._prefix_cache_save_step.call_args.kwargs == {"num_tokens_unpadded": 3, "num_tokens_padded": 4}
    if kept:
        assert rewrite.rows == ((1, "b"),) and cache.consumed == []
    else:
        assert rewrite is None and cache.consumed == [("discard", 9)]


def test_save_is_a_noop_without_cache():
    runner = OmniARModelRunner.__new__(OmniARModelRunner)
    runner._prefix_cache_save_step = MagicMock()
    assert runner._prefix_cache_save_mrv2(SimpleNamespace(), torch.zeros(1), None) is None
    runner._prefix_cache_save_step.assert_not_called()


def test_finalizer_without_rewrite_is_the_base_finalizer():
    base = MagicMock()
    assert OmniARModelRunner._prefix_cache_output_finalizer(None, base, mergeable=False) is base


def test_finalizer_refuses_model_owned_outputs():
    cache = _FakePrefixCache({"hit"})
    with pytest.raises(OmniPrefixCacheUnmatchError, match="model-owned"):
        OmniARModelRunner._prefix_cache_output_finalizer(PrefixCacheRewrite(cache, 2, ()), None, mergeable=False)
    assert cache.consumed == [("discard", 2)]


def _two_request_output(monkeypatch, finalize_output, *, async_chunk):
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    batch = SimpleNamespace(
        num_reqs=2,
        query_start_loc_np=np.array([0, 1, 3]),
        num_scheduled_tokens=np.array([1, 2]),
        num_tokens_after_padding=3,
    )
    return _async_output(
        req_ids=["hit", "miss"],
        sampler_output=SamplerOutput(torch.tensor([[1], [2]]), None, None, torch.tensor([1, 1]), torch.tensor([0, 0])),
        num_sampled_tokens=torch.tensor([1, 1]),
        text_hidden=torch.arange(6, dtype=torch.float32).reshape(3, 2),
        multimodal_outputs={"codes": {"audio": torch.arange(3).reshape(3, 1)}},
        input_batch=batch,
        async_chunk=async_chunk,
        finalize_output=finalize_output,
    )


@pytest.mark.parametrize("async_chunk", [False, True])
def test_rewrite_replaces_only_rewritten_payloads_before_publication(monkeypatch, async_chunk):
    merged = {"hidden": torch.full((4, 2), 7.0), "codes.audio": torch.full((4, 1), 9)}
    cache = _FakePrefixCache({"hit"}, prefix={"hit": merged})
    published = []

    def publish(output):
        published.append([dict(p) for p in output.inter_stage_outputs])
        return output

    finalize = OmniARModelRunner._prefix_cache_output_finalizer(
        PrefixCacheRewrite(cache, 5, ((0, "hit"),)), publish, mergeable=True
    )
    result = _two_request_output(monkeypatch, finalize, async_chunk=async_chunk).get_output()

    [(hit, miss)] = published
    assert torch.equal(hit["codes.audio"], merged["codes.audio"])
    assert torch.equal(miss["codes.audio"], torch.tensor([[1], [2]]))
    if async_chunk:
        assert "hidden" not in hit
    else:
        assert torch.equal(hit["hidden"], merged["hidden"])
        assert torch.equal(miss["hidden"], torch.tensor([[2.0, 3.0], [4.0, 5.0]]))
        assert torch.equal(result.multimodal_outputs[0]["codes.audio"], merged["codes.audio"])
    assert cache.consumed == [("materialize", 5)]


def test_rewrite_consumes_step_once_even_when_materialize_fails(monkeypatch):
    cache = _FakePrefixCache({"hit"}, fail=True)
    rewrite = PrefixCacheRewrite(cache, 3, ((0, "hit"),))
    finalize = OmniARModelRunner._prefix_cache_output_finalizer(rewrite, None, mergeable=True)
    with pytest.raises(RuntimeError, match="materialize failed"):
        _two_request_output(monkeypatch, finalize, async_chunk=False).get_output()
    with pytest.raises(RuntimeError, match="already consumed"):
        rewrite.discard()
    assert cache.consumed == [("materialize", 3)]
