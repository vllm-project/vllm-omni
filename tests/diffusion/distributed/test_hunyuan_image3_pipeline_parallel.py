# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HunyuanImage3 DiT-only Pipeline-Parallelism wiring.

The generic ``PipelineParallelMixin`` plumbing (async send/recv, the diffuse /
predict_noise / scheduler_step wrappers) is covered by
``test_pipeline_parallel.py``. This file covers the *HunyuanImage3-specific*
logic added on top of it, which that generic test does not exercise:

* the two ``HunyuanImage3Model.forward`` PP guards (PP+SP unsupported; non-first
  stage requires ``intermediate_tensors``),
* ``HunyuanImage3Pipeline._owned_layers`` and the per-layer KV-state helpers,
  which key state by *global* layer index so it stays consistent when pipeline
  parallelism leaves only a layer subset on each rank,
* the ``_pad_tensor_rows`` step-batch row merge.

The per-layer numeric forward across stages needs the real ~158 GB checkpoint
and is covered by the end-to-end / pixel-accuracy suites, not here.
"""

import os
import socket
import types

import pytest
import torch
from vllm.model_executor.models.utils import PPMissingLayer, make_layers

import vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer as hi3_transformer
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    initialize_model_parallel,
)
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Model,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _STEP_AR_KV,
    _STEP_PROMPT_KV,
    HunyuanImage3Pipeline,
)
from vllm_omni.platforms import current_omni_platform


def _find_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return str(s.getsockname()[1])


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeKVManager:
    """Stand-in for ``layer.self_attn.image_attn`` holding only the KV state
    the pipeline's PP helpers read and write."""

    def __init__(self):
        self._injected_ar_kv = None
        self.image_kv_cache_map = None
        self.image_kv_cache_lens = None


class _FakeAttn:
    def __init__(self):
        self.image_attn = _FakeKVManager()


class _FakeDecoderLayer(torch.nn.Module):
    """Minimal decoder layer exposing ``self_attn.image_attn`` so that
    ``_owned_layers`` treats it as a real (non-placeholder) layer.

    An ``nn.Module`` so it can be partitioned by ``make_layers`` (which returns
    an ``nn.ModuleList``); ``PPMissingLayer`` stands in for layers owned by
    other stages and is skipped because it has no ``self_attn``.
    """

    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttn()


class _KVHarness:
    """Borrows the real HunyuanImage3Pipeline PP helper methods onto a light
    object whose only state is ``self.model.layers``.

    Subclassing the real pipeline would drag in ``nn.Module``/PreTrainedModel
    initialisation; the helpers only touch ``self.model.layers`` (via
    ``_owned_layers``) and ``self._pad_tensor_rows``, so binding the unbound
    methods here exercises the real code without that machinery.
    """

    _owned_layers = HunyuanImage3Pipeline._owned_layers
    _snapshot_injected_ar_kv = HunyuanImage3Pipeline._snapshot_injected_ar_kv
    _restore_injected_ar_kv = HunyuanImage3Pipeline._restore_injected_ar_kv
    _accumulate_prompt_kv_cache = HunyuanImage3Pipeline._accumulate_prompt_kv_cache
    _finalize_prompt_kv_cache = HunyuanImage3Pipeline._finalize_prompt_kv_cache
    _capture_prompt_kv_cache = HunyuanImage3Pipeline._capture_prompt_kv_cache
    _restore_prompt_kv_cache = HunyuanImage3Pipeline._restore_prompt_kv_cache
    _prompt_kv_prefix_lens = HunyuanImage3Pipeline._prompt_kv_prefix_lens
    # staticmethods: re-wrap so `self._pad_tensor_rows(rows)` does not bind self.
    _pad_tensor_rows = staticmethod(HunyuanImage3Pipeline._pad_tensor_rows)

    def __init__(self, layers):
        self.model = types.SimpleNamespace(layers=layers)


def _make_state(step_index: int):
    """Minimal DiffusionRequestState stand-in: the KV helpers only read
    ``.step_index``, ``.request_id`` and the ``.extra`` dict."""
    return types.SimpleNamespace(step_index=step_index, request_id="req-0", extra={})


# ===========================================================================
# 1. HunyuanImage3Model.forward PP guards
# ===========================================================================


@pytest.mark.cpu
class TestForwardPPGuards:
    @staticmethod
    def _stub_model():
        config = types.SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            use_return_dict=True,
        )
        return types.SimpleNamespace(config=config)

    def test_pp_with_sp_raises_not_implemented(self, monkeypatch):
        monkeypatch.setattr(hi3_transformer, "get_pipeline_parallel_world_size", lambda: 2)
        monkeypatch.setattr(hi3_transformer, "get_sequence_parallel_world_size", lambda: 2)

        with pytest.raises(NotImplementedError, match="pipeline parallelism with sequence parallelism"):
            HunyuanImage3Model.forward(self._stub_model())

    def test_non_first_stage_requires_intermediate_tensors(self, monkeypatch):
        monkeypatch.setattr(hi3_transformer, "get_pipeline_parallel_world_size", lambda: 2)
        monkeypatch.setattr(hi3_transformer, "get_sequence_parallel_world_size", lambda: 1)
        monkeypatch.setattr(hi3_transformer, "is_pipeline_first_stage", lambda: False)

        with pytest.raises(RuntimeError, match="intermediate_tensors must be provided"):
            HunyuanImage3Model.forward(self._stub_model(), intermediate_tensors=None)


# ===========================================================================
# 2. _owned_layers: skip placeholders, key by global layer index
# ===========================================================================


@pytest.mark.cpu
class TestOwnedLayers:
    def test_skips_placeholders_and_yields_global_indices(self):
        # Global layers 0..3; this "rank" owns 1 and 2, others are placeholders.
        layers = [PPMissingLayer(), _FakeDecoderLayer(), _FakeDecoderLayer(), PPMissingLayer()]
        harness = _KVHarness(layers)

        owned = list(harness._owned_layers())
        assert owned == [layers[1], layers[2]]

        indexed = list(harness._owned_layers(with_index=True))
        assert [idx for idx, _ in indexed] == [1, 2], "indices must be global, not 0-based within the owned subset"
        assert [layer for _, layer in indexed] == [layers[1], layers[2]]


# ===========================================================================
# 3. AR-KV snapshot/restore — global-index keying round trip (CPU)
# ===========================================================================


@pytest.mark.cpu
class TestArKvRoundTrip:
    def test_snapshot_is_global_indexed_and_clears_source(self):
        layers = [PPMissingLayer(), _FakeDecoderLayer(), _FakeDecoderLayer(), PPMissingLayer()]
        # Distinct per-layer KV so a local-vs-global indexing bug is observable.
        for gidx in (1, 2):
            k = torch.full((1, 3), float(gidx))
            v = torch.full((1, 3), float(gidx) + 0.5)
            layers[gidx].self_attn.image_attn._injected_ar_kv = [(k, v)]
        harness = _KVHarness(layers)

        snapshot = harness._snapshot_injected_ar_kv()

        assert snapshot is not None
        assert len(snapshot) == 4
        assert snapshot[0] is None and snapshot[3] is None, "placeholder stages contribute None"
        assert snapshot[1] is not None and snapshot[2] is not None
        torch.testing.assert_close(snapshot[1][0][0], torch.full((1, 3), 1.0))
        torch.testing.assert_close(snapshot[2][0][0], torch.full((1, 3), 2.0))
        # snapshotting detaches the manager's copy
        for gidx in (1, 2):
            assert layers[gidx].self_attn.image_attn._injected_ar_kv is None

    def test_snapshot_returns_none_when_nothing_injected(self):
        layers = [PPMissingLayer(), _FakeDecoderLayer(), PPMissingLayer()]
        harness = _KVHarness(layers)
        assert harness._snapshot_injected_ar_kv() is None

    def test_restore_reads_state_by_global_index(self):
        layers = [PPMissingLayer(), _FakeDecoderLayer(), _FakeDecoderLayer(), PPMissingLayer()]
        for gidx in (1, 2):
            k = torch.full((1, 3), float(gidx))
            v = torch.full((1, 3), float(gidx) + 0.5)
            layers[gidx].self_attn.image_attn._injected_ar_kv = [(k, v)]
        harness = _KVHarness(layers)

        state = _make_state(step_index=1)
        state.extra[_STEP_AR_KV] = harness._snapshot_injected_ar_kv()

        harness._restore_injected_ar_kv([state], row_state_indexes=[0], row_branches=[0])

        # Each owned layer gets back exactly its own (global-indexed) KV.
        torch.testing.assert_close(layers[1].self_attn.image_attn._injected_ar_kv[0][0], torch.full((1, 3), 1.0))
        torch.testing.assert_close(layers[2].self_attn.image_attn._injected_ar_kv[0][0], torch.full((1, 3), 2.0))


# ===========================================================================
# 4. _pad_tensor_rows: step-batch row merge
# ===========================================================================


@pytest.mark.cpu
class TestPadTensorRows:
    def test_equal_shapes_concatenate(self):
        rows = [torch.ones(1, 2, 3), torch.full((1, 2, 3), 2.0)]
        out = HunyuanImage3Pipeline._pad_tensor_rows(rows)
        assert out.shape == (2, 2, 3)
        torch.testing.assert_close(out[0], torch.ones(2, 3))
        torch.testing.assert_close(out[1], torch.full((2, 3), 2.0))

    def test_ragged_rows_are_zero_padded_to_max(self):
        rows = [torch.ones(1, 2, 4), torch.ones(1, 5, 4)]  # differ on dim 1
        out = HunyuanImage3Pipeline._pad_tensor_rows(rows)
        assert out.shape == (2, 5, 4)
        # original content preserved
        torch.testing.assert_close(out[0, :2], torch.ones(2, 4))
        torch.testing.assert_close(out[1, :5], torch.ones(5, 4))
        # padding region is zero
        torch.testing.assert_close(out[0, 2:], torch.zeros(3, 4))

    def test_empty_rows_raise(self):
        with pytest.raises(ValueError, match="empty"):
            HunyuanImage3Pipeline._pad_tensor_rows([])


# ===========================================================================
# 5. GPU / distributed: global-index keying under a real make_layers partition
# ===========================================================================


def _partition_kv_worker(local_rank, world_size, master_port, pp_size, num_layers, result_queue):
    """On each PP rank, build the layer list via the real ``make_layers``
    partition and check that both KV families key state by global layer index."""
    os.environ.update(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": master_port,
        }
    )
    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment

    init_distributed_environment()
    initialize_model_parallel(pipeline_parallel_size=pp_size)

    start_layer, end_layer, layers = make_layers(
        num_layers,
        lambda prefix: _FakeDecoderLayer(),
        prefix="layers",
    )
    harness = _KVHarness(layers)

    owned = [idx for idx, _ in harness._owned_layers(with_index=True)]

    # AR-KV: snapshot must place non-None entries exactly at this rank's owned
    # global indices.
    for gidx, layer in harness._owned_layers(with_index=True):
        k = torch.full((1, 2), float(gidx), device=device)
        layer.self_attn.image_attn._injected_ar_kv = [(k, k.clone())]
    ar_snapshot = harness._snapshot_injected_ar_kv()
    ar_nonnull = [i for i, s in enumerate(ar_snapshot) if s is not None]

    # Prompt-KV: capture must key the per-state cache dict by global index too.
    for _, layer in harness._owned_layers(with_index=True):
        mgr = layer.self_attn.image_attn
        mgr.image_kv_cache_map = (torch.ones(1, 2, device=device), torch.ones(1, 2, device=device))
        mgr.image_kv_cache_lens = torch.tensor([2], device=device)
    state = _make_state(step_index=1)
    harness._capture_prompt_kv_cache([state], row_state_indexes=[0], row_branches=[0])
    prompt_keys = sorted(state.extra[_STEP_PROMPT_KV].keys())

    result_queue.put(
        {
            "rank": local_rank,
            "start": start_layer,
            "end": end_layer,
            "owned": owned,
            "ar_nonnull": ar_nonnull,
            "prompt_keys": prompt_keys,
            "ar_len": len(ar_snapshot),
        }
    )

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    destroy_distributed_env()


@pytest.mark.gpu
@pytest.mark.parallel
@pytest.mark.parametrize(
    "pp_size, num_layers",
    [
        pytest.param(
            2,
            4,
            marks=pytest.mark.skipif(current_omni_platform.get_device_count() < 2, reason="Need at least 2 GPUs"),
            id="pp2-4layers",
        ),
        pytest.param(
            3,
            6,
            marks=pytest.mark.skipif(current_omni_platform.get_device_count() < 3, reason="Need at least 3 GPUs"),
            id="pp3-6layers",
        ),
    ],
)
def test_kv_helpers_key_by_global_index_under_real_partition(pp_size, num_layers):
    """Under a real ``make_layers`` PP partition, every rank's owned layers are
    its contiguous global slice ``[start, end)``, and both the AR-KV snapshot and
    the prompt-KV capture key their per-layer state by that global index — so
    the partition is reconstructed identically across stages."""
    mp_context = torch.multiprocessing.get_context("spawn")
    manager = mp_context.Manager()
    q = manager.Queue()

    port = _find_free_port()
    torch.multiprocessing.spawn(
        _partition_kv_worker,
        args=(pp_size, port, pp_size, num_layers, q),
        nprocs=pp_size,
    )

    results = [q.get() for _ in range(pp_size)]
    seen_indices: list[int] = []
    for r in results:
        expected = list(range(r["start"], r["end"]))
        assert r["owned"] == expected, f"rank {r['rank']} owned {r['owned']} != global slice {expected}"
        assert r["ar_nonnull"] == expected, f"rank {r['rank']} AR-KV snapshot not keyed by global index"
        assert r["prompt_keys"] == expected, f"rank {r['rank']} prompt-KV cache not keyed by global index"
        assert r["ar_len"] == num_layers, "AR-KV snapshot must span all global layers"
        seen_indices.extend(r["owned"])

    # The ranks together cover every global layer exactly once.
    assert sorted(seen_indices) == list(range(num_layers))
