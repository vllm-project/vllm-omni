# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref

import pytest
import torch

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.diffusion.models.qwen_image_21 import decode_graph
from vllm_omni.diffusion.offloader.sequential_backend import SequentialOffloadHook

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


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


@pytest.mark.cuda
@pytest.mark.gpu
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


@pytest.mark.cuda
@pytest.mark.gpu
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


@pytest.mark.cuda
@pytest.mark.gpu
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
