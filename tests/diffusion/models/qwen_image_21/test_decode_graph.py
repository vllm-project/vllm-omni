# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import weakref

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21 import decode_graph

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
    with torch.inference_mode():
        register(manager, first)
        register(manager, second)
        for cache in (first, second, first):
            output = decode(manager, cache)
            expected = (
                torch.ones_like(output).sin() + cache[0]["cond"]["key"].mean() + cache[0]["cond"]["value"].mean() + 2
            )
            torch.testing.assert_close(output, expected)
        original = decode(manager, first).clone()
        register(manager, second, (1, 1, 4))
        other = decode(manager, second, (1, 1, 4))
        torch.testing.assert_close(other, torch.ones_like(other).sin() + 5 + 6 + 1)
        torch.testing.assert_close(decode(manager, first), original)
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
        kwargs["kv_cache"] = [{"cond": {part: tensor.flip(0) for part, tensor in merged[0]["cond"].items()}}]
        torch.testing.assert_close(manager.try_decode(**kwargs), expected.flip(0))
        kwargs["encoder_hidden_states_mask"][1, 0] = False
        assert manager.try_decode(**kwargs) is None
        torch.testing.assert_close(decode(manager, first), single, rtol=0, atol=0)
