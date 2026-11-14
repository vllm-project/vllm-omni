# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import gc
import weakref

import pytest
import torch
from vllm.platforms import current_platform

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.diffusion.models.qwen_image_21 import decode_graph
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Transformer2DModel
from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _stub_tp_group(monkeypatch):
    """CPU unit tests construct the transformer without a distributed init;
    stub the TP group so decode-graph eligibility queries don't require it."""
    import types

    import vllm.distributed as vd
    import vllm.distributed.parallel_state as ps

    stub = types.SimpleNamespace(world_size=1, rank_in_group=0)
    monkeypatch.setattr(ps, "get_tp_group", lambda: stub, raising=False)
    monkeypatch.setattr(vd, "get_tp_group", lambda: stub, raising=False)
    monkeypatch.setattr(vd, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(vd, "get_tensor_model_parallel_rank", lambda: 0, raising=False)


class DecodeModel(torch.nn.Module):
    in_channels = 4

    def _decode_graph_forward(self, entry):
        return (
            entry.hidden.sin()
            + entry.k[0].mean(dim=(1, 2, 3))[:, None, None]
            + entry.v[0].mean(dim=(1, 2, 3))[:, None, None]
            + entry.freqs[None, :, :]
        )


@pytest.fixture
def graph_case(request, monkeypatch):
    device = "cuda" if request.node.get_closest_marker("cuda") else "cpu"
    manager = decode_graph.QwenImage21DecodeGraphManager(DecodeModel(), max_entries=2)
    monkeypatch.setattr(manager, "eligible", lambda: True)
    monkeypatch.setattr(manager, "_backend_name", lambda: "test")
    cleanup = []
    monkeypatch.setattr(decode_graph.current_omni_platform, "empty_cache", lambda: cleanup.append(True))
    return manager, device, cleanup


def make_cache(device, value):
    return [
        {
            "cond": {
                "key": torch.full((1, 2, 1, 4), value, device=device),
                "value": torch.full((1, 2, 1, 4), value + 1, device=device),
            }
        }
    ]


def register(manager, cache, layout=(1, 2, 2)):
    device = cache[0]["cond"]["key"].device
    manager.register_prefill(
        kv_cache=cache,
        cache_branch="cond",
        prefix_len=2,
        img_shapes=[layout],
        target_freqs=torch.full((4, 4), float(layout[1]), device=device),
        joint_key_valid=None,
        dtype=torch.float32,
    )


def entry_key(manager, layout):
    want = (tuple(layout),)
    matches = [key for key in manager.entries if key[3] == want]
    assert len(matches) == 1, matches
    return matches[0]


def decode(manager, cache, layout=(1, 2, 2)):
    device = cache[0]["cond"]["key"].device
    return manager.try_decode(
        hidden_states=torch.ones(1, 4, 4, device=device),
        timestep=torch.ones(1, device=device),
        kv_cache=cache,
        cache_branch="cond",
        img_shapes=[layout],
        img_mask=torch.tensor([[False, False, True]], device=device),
        encoder_hidden_states_mask=None,
    )


@pytest.mark.cpu
def test_release_idle_entries_drops_dead_owners_and_keeps_live_ones(graph_case):
    manager, device, cleanup = graph_case
    manager.max_entries = 8
    # Legacy (pre-A2) behavior: idle entries are dropped each request.
    manager.persistent_entries = False
    other = (1, 2, 2)
    live = (1, 4, 2)
    same_key_later = (1, 8, 2)

    finished = make_cache(device, 1.0)
    register(manager, finished, other)
    other_key = entry_key(manager, other)
    static_prefix = weakref.ref(manager.entries[other_key].k[0])
    del finished

    still_running = make_cache(device, 2.0)
    register(manager, still_running, live)
    live_key = entry_key(manager, live)

    finished_same_shape = make_cache(device, 3.0)
    register(manager, finished_same_shape, same_key_later)
    same_key = entry_key(manager, same_key_later)
    same_static = weakref.ref(manager.entries[same_key].k[0])
    del finished_same_shape
    gc.collect()

    manager.release_idle_entries()

    assert other_key not in manager.entries
    assert same_key not in manager.entries
    gc.collect()
    assert static_prefix() is None
    assert same_static() is None
    assert live_key in manager.entries
    assert manager._owners[live_key]() is still_running[0]["cond"]["key"]
    assert cleanup


@pytest.mark.cpu
def test_persistent_entries_retain_idle_graphs_for_reuse(graph_case):
    manager, device, cleanup = graph_case
    manager.persistent_entries = True
    manager.max_entries = 8

    first = make_cache(device, 1.0)
    register(manager, first)
    key = entry_key(manager, (1, 2, 2))
    entry = manager.entries[key]
    entry.refresh_prefix(first)
    buffer_ref = weakref.ref(entry.k[0])
    del first
    gc.collect()

    # The owner is dead, but persistent mode keeps the entry and its buffers
    # so the next matching request can replay without recapturing (#8586 A2).
    manager.release_idle_entries()
    assert key in manager.entries
    assert manager.entries[key] is entry
    assert buffer_ref() is not None
    assert not cleanup  # nothing dropped, so no empty_cache reclaim

    # A new request with the same key re-adopts the retained entry rather than
    # allocating a replacement, and its prefix is refreshed on the next decode.
    second = make_cache(device, 5.0)
    register(manager, second)
    assert len(manager.entries) == 1
    assert manager.entries[key] is entry
    assert manager._owners[key]() is second[0]["cond"]["key"]
    assert not entry.prefix_sources


@pytest.mark.cpu
def test_persistent_entries_still_evict_at_lru_cap(graph_case):
    manager, device, _ = graph_case
    manager.persistent_entries = True
    manager.max_entries = 2
    # Idle entries are retained, but the LRU cap still bounds memory: a third
    # distinct key evicts the oldest even though no owner is live.
    for value, layout in zip((1, 2, 3), [(1, 2, 2), (1, 3, 2), (1, 4, 2)]):
        cache = make_cache(device, float(value))
        register(manager, cache, layout)
        del cache
        manager.release_idle_entries()
    assert len(manager.entries) == manager.max_entries
    assert all(key[3] != ((1, 2, 2),) for key in manager.entries)


@pytest.mark.cpu
def test_persistent_entries_drop_failure_markers_for_retry(graph_case, monkeypatch):
    manager, device, cleanup = graph_case
    manager.persistent_entries = True
    cache = make_cache(device, 1.0)
    constructor = decode_graph.QwenImage21DecodeGraphEntry

    def fail_allocation(**kwargs):
        raise torch.OutOfMemoryError("injected allocation OOM")

    monkeypatch.setattr(decode_graph, "QwenImage21DecodeGraphEntry", fail_allocation)
    register(manager, cache)
    assert list(manager.entries.values()) == [None]
    # Persistent mode keeps captured graphs but still clears failure markers so
    # a later request can retry instead of staying eager for that key forever.
    manager.release_idle_entries()
    assert len(manager.entries) == 0
    assert cleanup
    monkeypatch.setattr(decode_graph, "QwenImage21DecodeGraphEntry", constructor)
    register(manager, cache)
    assert len(manager.entries) == 1
    assert next(iter(manager.entries.values())) is not None


@pytest.mark.cpu
def test_persistence_is_opt_in_default(graph_case):
    manager, device, cleanup = graph_case
    # Retention trades VRAM for recapture latency, so it is opt-in: the
    # default keeps the #8831 drop-idle-entries-each-request reclaim.
    assert manager.persistent_entries is False
    cache = make_cache(device, 1.0)
    register(manager, cache)
    key = entry_key(manager, (1, 2, 2))
    del cache
    gc.collect()
    manager.release_idle_entries()
    assert key not in manager.entries
    assert cleanup


@pytest.mark.cpu
def test_persistent_readoption_refuses_second_live_owner(graph_case):
    manager, device, _ = graph_case
    manager.persistent_entries = True
    manager.max_entries = 8
    first = make_cache(device, 1.0)
    register(manager, first)
    key = entry_key(manager, (1, 2, 2))
    entry = manager.entries[key]
    del first
    gc.collect()
    manager.release_idle_entries()
    assert manager.entries[key] is entry

    # Re-adoption by a new request makes it the single live owner again.
    second = make_cache(device, 5.0)
    register(manager, second)
    assert manager._owners[key]() is second[0]["cond"]["key"]

    # A third in-flight request aliasing the re-adopted key is refused: the
    # re-adopting owner keeps the static buffers until its cache is released.
    third = make_cache(device, 9.0)
    register(manager, third)
    assert manager.entries[key] is entry
    assert manager._owners[key]() is second[0]["cond"]["key"]
    assert bool((third[0]["cond"]["key"] == 9).all())


@pytest.mark.cpu
def test_persistent_lru_cap_with_mixed_live_and_dead_owners(graph_case):
    manager, device, _ = graph_case
    manager.persistent_entries = True
    manager.max_entries = 2
    dead = make_cache(device, 1.0)
    register(manager, dead, (1, 2, 2))
    dead_key = entry_key(manager, (1, 2, 2))
    dead_buffer = weakref.ref(manager.entries[dead_key].k[0])
    del dead
    gc.collect()
    manager.release_idle_entries()
    assert dead_key in manager.entries  # idle entry retained

    live = make_cache(device, 2.0)
    register(manager, live, (1, 3, 2))
    live_key = entry_key(manager, (1, 3, 2))
    live_entry = manager.entries[live_key]

    # At the cap, the LRU (dead-owner) entry is evicted first; the live
    # owner's entry and ownership survive untouched.
    newest = make_cache(device, 3.0)
    register(manager, newest, (1, 4, 2))
    assert len(manager.entries) == manager.max_entries
    assert dead_key not in manager.entries
    gc.collect()
    assert dead_buffer() is None
    assert manager.entries[live_key] is live_entry
    assert manager._owners[live_key]() is live[0]["cond"]["key"]

    # A later idle sweep keeps the live owner and retains the now-dead newest.
    del newest
    gc.collect()
    manager.release_idle_entries()
    assert len(manager.entries) == 2
    assert manager._owners[live_key]() is live[0]["cond"]["key"]


@pytest.mark.cpu
def test_model_level_offload_without_staging_disables_decode_graphs():
    manager = decode_graph.QwenImage21DecodeGraphManager(DecodeModel(), model_level_offload=True)
    assert manager._offload_reason() is not None
    assert manager.eligible() is False


@pytest.mark.cpu
def test_top_level_offload_hook_without_staging_disables_decode_graphs():
    model = DecodeModel()
    manager = decode_graph.QwenImage21DecodeGraphManager(model)
    registry = HookRegistry.get_or_create(model)
    # A present-but-empty registry (e.g. after hooks were removed) must not disqualify.
    assert manager._offload_reason() is None
    registry.register_hook("sequential_offload", ModelHook())
    assert manager._offload_reason() is not None
    assert manager.eligible() is False


@pytest.mark.cpu
def test_persistent_staging_makes_model_level_offload_graph_eligible():
    model = DecodeModel()
    manager = decode_graph.QwenImage21DecodeGraphManager(model, model_level_offload=True)
    hook = SequentialOffloadHook(offload_targets=[], device=torch.device("cpu"))
    HookRegistry.get_or_create(model).register_hook(SequentialOffloadHook._HOOK_NAME, hook)
    # No staging storage yet: a swap would allocate fresh storage per generation.
    assert manager._offload_reason() is not None
    # Staging established: fixed device storage keeps captured pointers valid.
    hook._stager = object()
    assert manager._offload_reason() is None
    # A second, non-offload hook on the top-level module still disqualifies.
    model._hook_registry.register_hook("teacache", ModelHook())
    assert manager._offload_reason() is not None


@pytest.mark.cpu
def test_allocation_oom_releases_partial_buffers_and_remembers_failure(graph_case, monkeypatch):
    manager, device, cleanup = graph_case
    cache = make_cache(device, 1.0)
    constructor = decode_graph.QwenImage21DecodeGraphEntry
    allocated = []

    def fail_allocation(**kwargs):
        entry = constructor(**kwargs)
        allocated.append(weakref.ref(entry.k[0]))
        raise torch.OutOfMemoryError("injected allocation OOM")

    monkeypatch.setattr(decode_graph, "QwenImage21DecodeGraphEntry", fail_allocation)
    register(manager, cache)
    register(manager, cache)
    assert len(allocated) == 1
    assert allocated[0]() is None
    assert list(manager.entries.values()) == [None]
    assert len(cleanup) == 1
    assert torch.equal(cache[0]["cond"]["key"], torch.ones(1, 2, 1, 4))


@pytest.mark.cpu
def test_evict_before_allocating_replacement(graph_case, monkeypatch):
    manager, device, _ = graph_case
    manager.max_entries = 1
    cache = make_cache(device, 1.0)
    register(manager, cache)
    old_buffer = weakref.ref(next(iter(manager.entries.values())).k[0])
    constructor = decode_graph.QwenImage21DecodeGraphEntry

    def allocate_after_release(**kwargs):
        assert old_buffer() is None
        return constructor(**kwargs)

    monkeypatch.setattr(decode_graph, "QwenImage21DecodeGraphEntry", allocate_after_release)
    register(manager, cache, (1, 1, 4))
    assert len(manager.entries) == 1


@pytest.mark.cpu
def test_registration_preserves_request_cache_and_separates_layouts(graph_case):
    manager, device, _ = graph_case
    first, second = make_cache(device, 1.0), make_cache(device, 5.0)
    original = first[0]["cond"]["key"]
    register(manager, first)
    register(manager, second)
    assert first[0]["cond"]["key"] is original
    assert torch.equal(original, torch.ones_like(original))
    register(manager, second, (1, 1, 4))
    assert len(manager.entries) == 2
    entries = list(manager.entries.values())
    assert not torch.equal(entries[0].freqs, entries[1].freqs)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("failure_phase", ["warmup", "capture"])
def test_capture_failure_releases_buffers_and_keeps_request_cache(graph_case, monkeypatch, failure_phase):
    manager, device, cleanup = graph_case
    cache = make_cache(device, 1.0)
    register(manager, cache)
    entry = next(iter(manager.entries.values()))
    entry_ref, buffer_ref = weakref.ref(entry), weakref.ref(entry.k[0])
    del entry

    original = manager.model._decode_graph_forward

    def fail_capture(entry):
        output = original(entry)
        if failure_phase == "warmup" or torch.cuda.is_current_stream_capturing():
            raise torch.OutOfMemoryError("injected capture OOM")
        return output

    monkeypatch.setattr(manager.model, "_decode_graph_forward", fail_capture)
    assert decode(manager, cache) is None
    assert entry_ref() is None and buffer_ref() is None
    assert list(manager.entries.values()) == [None]
    register(manager, cache)
    assert decode(manager, cache) is None
    assert len(cleanup) == 1
    assert bool((cache[0]["cond"]["key"] == 1).all())


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_recapture_after_releasing_last_graph_with_live_pool_output(graph_case):
    manager, device, _ = graph_case
    retained_outputs = []
    original_stream = torch.cuda.current_stream()
    with torch.cuda.stream(original_stream), torch.inference_mode():
        for value in (1.0, 2.0, 3.0):
            cache = make_cache(device, value)
            register(manager, cache)
            result = decode(manager, cache)
            assert result is not None
            expected = torch.ones_like(result).sin() + value + (value + 1) + 2
            torch.testing.assert_close(result, expected)
            # Keep a pool allocation alive after its last graph is destroyed.
            # Reusing a global pool handle then hits the allocator's use_count
            # assertion even though the request's static prefix was released.
            entry = next(iter(manager.entries.values()))
            retained_outputs.append(entry.output)
            del entry, cache
            manager.release_idle_entries()
            assert not manager.entries
        assert torch.cuda.current_stream() == original_stream
    assert len(retained_outputs) == 3


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("failure_phase", ["begin", "end"])
def test_capture_context_failure_restores_caller_stream(graph_case, monkeypatch, failure_phase):
    manager, device, cleanup = graph_case
    cache = make_cache(device, 1.0)
    register(manager, cache)
    original_stream = torch.cuda.current_stream()

    class FailingCaptureGraph(torch.cuda.CUDAGraph):
        def capture_begin(self, *args, **kwargs):
            if failure_phase == "begin":
                raise RuntimeError("injected capture_begin failure")
            return super().capture_begin(*args, **kwargs)

        def capture_end(self):
            super().capture_end()
            if failure_phase == "end":
                raise RuntimeError("injected capture_end failure")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", FailingCaptureGraph)
    # Isolate the injected begin/end error from any retired global pool left
    # alive by the preceding lifecycle regression on an unfixed implementation.
    monkeypatch.setattr(current_platform, "get_global_graph_pool", torch.cuda.graph_pool_handle)
    # Restore test state even on an unfixed implementation whose graph
    # context switches streams before raising in __enter__ or __exit__.
    with torch.cuda.stream(original_stream):
        assert decode(manager, cache) is None
        assert torch.cuda.current_stream() == original_stream
        assert list(manager.entries.values()) == [None]
        assert bool((cache[0]["cond"]["key"] == 1).all())
        assert cleanup


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_replay_uses_active_request_and_layout(graph_case):
    manager, device, _ = graph_case
    first, second = make_cache(device, 1.0), make_cache(device, 5.0)
    first_key = weakref.ref(first[0]["cond"]["key"])
    with torch.inference_mode():
        register(manager, first)
        # A second live request aliasing the same key is refused: it keeps its
        # own cache and decodes eagerly while the owner is in flight.
        register(manager, second)
        assert decode(manager, second) is None
        output = decode(manager, first)
        expected = torch.ones_like(output).sin() + first[0]["cond"]["key"].mean() + first[0]["cond"]["value"].mean() + 2
        torch.testing.assert_close(output, expected)
        # Once the owner's cache is released, the key is free to adopt.
        del first
        assert first_key() is None
        register(manager, second)
        output = decode(manager, second)
        expected = (
            torch.ones_like(output).sin() + second[0]["cond"]["key"].mean() + second[0]["cond"]["value"].mean() + 2
        )
        torch.testing.assert_close(output, expected)
        original = decode(manager, second).clone()
        register(manager, second, (1, 1, 4))
        other = decode(manager, second, (1, 1, 4))
        torch.testing.assert_close(other, torch.ones_like(other).sin() + 5 + 6 + 1)
        torch.testing.assert_close(decode(manager, second), original)
        assert all(entry.captures == 1 for entry in manager.entries.values())


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_persistent_entry_replays_across_requests_without_recapture(graph_case):
    manager, device, _ = graph_case
    manager.persistent_entries = True
    first, second = make_cache(device, 1.0), make_cache(device, 5.0)
    with torch.inference_mode():
        register(manager, first)
        out_first = decode(manager, first)
        expected_first = (
            torch.ones_like(out_first).sin() + first[0]["cond"]["key"].mean() + first[0]["cond"]["value"].mean() + 2
        )
        torch.testing.assert_close(out_first, expected_first)
        key = entry_key(manager, (1, 2, 2))
        entry = manager.entries[key]
        assert entry.captures == 1

        # The request finishes. Persistent mode keeps the captured graph and
        # static buffers instead of dropping them (#8586 A2).
        del first
        manager.release_idle_entries()
        assert manager.entries[key] is entry

        # A new request with the same key re-adopts the entry, refreshes the
        # prefix into the static buffers, and replays without recapturing.
        register(manager, second)
        out_second = decode(manager, second)
        expected_second = (
            torch.ones_like(out_second).sin() + second[0]["cond"]["key"].mean() + second[0]["cond"]["value"].mean() + 2
        )
        torch.testing.assert_close(out_second, expected_second)
        assert manager.entries[key] is entry
        assert entry.captures == 1

        # A third live request aliasing the re-adopted key is refused decode
        # until the re-adopting owner's cache is released (single live owner).
        third = make_cache(device, 9.0)
        register(manager, third)
        assert decode(manager, third) is None
        assert manager._owners[key]() is second[0]["cond"]["key"]
        torch.testing.assert_close(decode(manager, second), expected_second)
        assert entry.captures == 1


@pytest.mark.cpu
@pytest.mark.parametrize("inference", [False, True])
def test_prefix_reuse_refreshes_every_layer_without_retaining_requests(graph_case, inference):
    manager, device, _ = graph_case
    with torch.inference_mode(inference):
        cache = make_cache(device, 1.0) + make_cache(device, 2.0)
        register(manager, cache)
        entry = next(iter(manager.entries.values()))
        entry.refresh_prefix(cache)
        versions = [tensor._version for tensor in entry.k + entry.v]
        entry.refresh_prefix(cache)
        assert versions == [tensor._version for tensor in entry.k + entry.v]
        cache[1]["cond"]["value"] = torch.full_like(cache[1]["cond"]["value"], 7.0)
        entry.refresh_prefix(cache)
        assert bool((entry.v[1] == 7).all())
        source = weakref.ref(cache[0]["cond"]["key"])
        del cache
        assert source() is None


@pytest.mark.cpu
def test_prefix_reuse_detects_inplace_changes_and_prefill_registration(graph_case):
    manager, device, _ = graph_case
    cache = make_cache(device, 1.0)
    register(manager, cache)
    entry = next(iter(manager.entries.values()))
    entry.refresh_prefix(cache)
    cache[0]["cond"]["value"].add_(3)
    entry.refresh_prefix(cache)
    assert bool((entry.v[0] == 5).all())
    register(manager, cache)
    assert not entry.prefix_sources


@pytest.mark.cpu
def test_all_valid_mask_shares_graph_entry_and_padding_falls_back(graph_case):
    manager, device, _ = graph_case
    cache = make_cache(device, 1.0)
    register(manager, cache)
    entry = next(iter(manager.entries.values()))
    for valid in (True, False):
        manager.register_prefill(
            kv_cache=cache,
            cache_branch="cond",
            prefix_len=2,
            img_shapes=[(1, 2, 2)],
            target_freqs=torch.ones(4, 4),
            joint_key_valid=torch.full((1, 6), valid),
            dtype=torch.float32,
        )
        assert list(manager.entries.values()) == [entry]
        assert entry.attn_metadata is None


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_batch_changes_and_padding_do_not_reuse_stale_prefix(graph_case):
    manager, device, _ = graph_case
    first = make_cache(device, 1.0)
    second = make_cache(device, 4.0)
    merged = [
        {"cond": {part: torch.cat([first[0]["cond"][part], second[0]["cond"][part]]) for part in ("key", "value")}}
    ]
    with torch.inference_mode():
        register(manager, first)
        single = decode(manager, first).clone()
        register(manager, merged)
        kwargs = dict(
            hidden_states=torch.ones(2, 4, 4, device=device),
            timestep=torch.ones(2, device=device),
            kv_cache=merged,
            cache_branch="cond",
            img_shapes=[(1, 2, 2)],
            img_mask=torch.tensor([[False, False, True]] * 2, device=device),
            encoder_hidden_states_mask=torch.ones(2, 3, dtype=torch.bool, device=device),
        )
        result = manager.try_decode(**kwargs)
        assert result is not None and result.shape == (2, 4, 4)
        expected = torch.ones_like(result).sin() + torch.tensor([5.0, 11.0], device=device)[:, None, None]
        torch.testing.assert_close(result, expected)
        # A rebuilt cache (e.g. reordered rows) is a different live cache
        # aliasing the same key: decode refuses the shared entry until the
        # previous owner's tensors are released.
        flipped = [{"cond": {part: tensor.flip(0) for part, tensor in merged[0]["cond"].items()}}]
        kwargs["kv_cache"] = flipped
        assert manager.try_decode(**kwargs) is None
        del merged
        register(manager, flipped)
        torch.testing.assert_close(manager.try_decode(**kwargs), expected.flip(0))
        kwargs["encoder_hidden_states_mask"][1, 0] = False
        assert manager.try_decode(**kwargs) is None
        torch.testing.assert_close(decode(manager, first), single, rtol=0, atol=0)


class WorkspaceGemmModel(DecodeModel):
    """Decode body whose split-K GEMM needs a cuBLAS workspace."""

    def __init__(self, device):
        super().__init__()
        generator = torch.Generator(device=device).manual_seed(0)
        self.lhs = torch.randn(32768, 640, device=device, dtype=torch.bfloat16, generator=generator).T
        self.rhs = torch.randn(32768, 640, device=device, dtype=torch.bfloat16, generator=generator)

    def _decode_graph_forward(self, entry):
        return super()._decode_graph_forward(entry) + torch.mm(self.lhs, self.rhs).float().mean()


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
def test_evicting_a_graph_keeps_survivor_workspace_when_warmup_aliases_capture(graph_case, monkeypatch):
    manager, device, _ = graph_case
    manager.model = WorkspaceGemmModel(device)
    capture_stream = torch.cuda.graph.default_capture_stream or torch.cuda.Stream()
    monkeypatch.setattr(torch.cuda.graph, "default_capture_stream", capture_stream)
    # Pooled streams recycle: every 32nd warm-up stream is the capture stream.
    monkeypatch.setattr(torch.cuda, "Stream", lambda *args, **kwargs: capture_stream)
    gemm_mean = torch.mm(manager.model.lhs, manager.model.rhs).float().mean()
    layouts = [(1, rows, 2) for rows in (2, 3, 4)]
    with torch.inference_mode():
        caches = [make_cache(device, float(value)) for value in (1, 2, 3)]
        for cache, layout in zip(caches[:2], layouts):
            register(manager, cache, layout)
            assert decode(manager, cache, layout) is not None
        # Registering a third layout evicts the oldest captured graph.
        register(manager, caches[2], layouts[2])
        assert len(manager.entries) == manager.max_entries
        torch.accelerator.empty_cache()
        for _ in range(3):
            result = decode(manager, caches[1], layouts[1])
        torch.accelerator.synchronize()
    # Survivor terms: key 2, value 3, RoPE frequencies equal to the layout rows.
    expected = torch.ones_like(result).sin() + 2.0 + 3.0 + layouts[1][1] + gemm_mean
    torch.testing.assert_close(result, expected, rtol=0, atol=5e-2)


def _tiny_transformer(od_config=None, **kwargs):
    """One-block transformer small enough to construct on CPU in unit tests."""
    return QwenImage21Transformer2DModel(
        od_config=od_config if od_config is not None else OmniDiffusionConfig(),
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        context_in_dim=8,
        axes_dims_rope=(2, 2, 4),
        **kwargs,
    )


@pytest.mark.cpu
def test_transformer_ctor_persistent_decode_false_drops_idle_entries(monkeypatch):
    # Constructor-level opt-out: the manager keeps the legacy drop-idle path
    # (#8831 reclaim) instead of retaining entries across requests.
    import vllm.distributed as vd

    monkeypatch.setattr(vd, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    transformer = _tiny_transformer(cuda_graph_persistent_decode=False)
    manager = transformer._decode_graph_manager
    assert manager is not None
    assert manager.persistent_entries is False
    monkeypatch.setattr(manager, "eligible", lambda: True)
    monkeypatch.setattr(manager, "_backend_name", lambda: "test")
    cache = make_cache("cpu", 1.0)
    register(manager, cache)
    key = entry_key(manager, (1, 2, 2))
    del cache
    gc.collect()
    manager.release_idle_entries()
    assert key not in manager.entries


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("ctor_kwarg", "od_value", "expected"),
    [
        ({}, None, False),  # default: retention opt-in off
        ({"cuda_graph_persistent_decode": False}, True, True),  # operator opts in
        ({"cuda_graph_persistent_decode": True}, False, False),  # operator opts out
        ({"cuda_graph_persistent_decode": True}, None, True),  # checkpoint opts in
    ],
)
def test_transformer_ctor_persistent_decode_resolution(ctor_kwarg, od_value, expected, monkeypatch):
    # The operator-facing knob (CLI --cuda-graph-persistent-decode / stage
    # YAML, carried by OmniDiffusionConfig) overrides the checkpoint-level
    # constructor default when set; None defers to the constructor.
    import vllm.distributed as vd

    monkeypatch.setattr(vd, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    od_config = OmniDiffusionConfig()
    od_config.cuda_graph_persistent_decode = od_value
    transformer = _tiny_transformer(od_config=od_config, **ctor_kwarg)
    manager = transformer._decode_graph_manager
    assert manager is not None
    assert manager.persistent_entries is expected
