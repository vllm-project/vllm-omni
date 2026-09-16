# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-request eager and CUDA-graph AuK DiT sampling equivalence."""

import gc
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.models.auk.auk_transformer import AuKTransformer, sample_latents
from vllm_omni.diffusion.models.auk.cudagraph_wrapper import AuKCUDAGraphWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_dit(device: str) -> AuKTransformer:
    torch.manual_seed(12)
    return (
        AuKTransformer(
            dim=32,
            heads=2,
            dim_head=16,
            ff_mult=2,
            latent_dim=4,
            text_hidden_dim=8,
            num_layers=2,
            num_single_layers=2,
        )
        .eval()
        .to(device)
    )


def _sample_inputs(device: str, offset: float = 0.0) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(42)
    return {
        "text": torch.randn(1, 7, 8, generator=generator, device=device) + offset,
        "c_mask": torch.ones(1, 7, dtype=torch.bool, device=device),
        "ref": torch.randn(1, 4, 4, generator=generator, device=device) - offset,
        "ref_mask": torch.ones(1, 4, dtype=torch.bool, device=device),
    }


def _find_address(address: int) -> dict[str, Any] | None:
    for segment in torch.cuda.memory._snapshot()["segments"]:
        segment_address = int(segment["address"])
        segment_size = int(segment["total_size"])
        if not segment_address <= address < segment_address + segment_size:
            continue
        for block in segment["blocks"]:
            block_address = int(block["address"])
            block_size = int(block["size"])
            if block_address <= address < block_address + block_size:
                return {
                    "segment_pool_id": segment.get("segment_pool_id"),
                    "segment_address": segment_address,
                    "segment_size": segment_size,
                    "block_address": block_address,
                    "block_size": block_size,
                    "requested_size": int(block.get("requested_size", 0)),
                    "state": block["state"],
                }
    return None


def _active_workspace_addresses(pool_handle, workspace_size: int) -> set[int]:
    addresses = set()
    snapshot = torch.cuda.memory._snapshot()

    for segment in snapshot["segments"]:
        if segment.get("segment_pool_id") != pool_handle:
            continue
        for block in segment["blocks"]:
            if block["state"] == "active_allocated" and int(block.get("requested_size", 0)) == workspace_size:
                addresses.add(int(block["address"]))
    return addresses


def _capture_pool_reuse_graph(
    pool_handle,
    num_bytes: int,
    *,
    prefix_bytes: int = 0,
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor | None, torch.Tensor]:
    graph = torch.cuda.CUDAGraph()
    prefix = None
    with torch.cuda.graph(graph, pool=pool_handle):
        if prefix_bytes:
            prefix = torch.empty(prefix_bytes, dtype=torch.uint8, device="cuda")
        allocation = torch.empty(num_bytes, dtype=torch.uint8, device="cuda")
        allocation.fill_(1)
    return graph, prefix, allocation


def _assert_graph_matches_eager(
    dit: AuKTransformer,
    wrapper: AuKCUDAGraphWrapper,
    *,
    gen_frames: int,
) -> dict[str, torch.Tensor]:
    inputs = _sample_inputs("cuda")
    common = dict(
        **inputs,
        gen_frames=gen_frames,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=0.0,
    )
    eager = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7))
    graph = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper)
    torch.testing.assert_close(graph, eager, atol=3e-6, rtol=3e-5)
    bucketed = wrapper._bucket_inputs(
        torch.empty(1, gen_frames, dit.latent_dim, device="cuda"),
        inputs["text"],
        inputs["c_mask"],
        inputs["ref"],
        inputs["ref_mask"],
    )
    assert wrapper._key(bucketed[0], bucketed[2], bucketed[4], False) in wrapper._cache
    return inputs


@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_single_request_graph_wrapper_cpu_falls_back_to_eager(cfg_strength: float, mocker) -> None:
    dit = _make_dit("cpu")
    wrapper = AuKCUDAGraphWrapper(dit)
    run_spy = mocker.spy(wrapper, "_run")
    run_cfg_spy = mocker.spy(wrapper, "_run_cfg")
    capture_spy = mocker.spy(wrapper, "_capture")
    inputs = _sample_inputs("cpu")
    common = dict(
        **inputs,
        gen_frames=9,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=cfg_strength,
    )
    eager = sample_latents(dit, **common, generator=torch.Generator().manual_seed(7))
    graph = sample_latents(dit, **common, generator=torch.Generator().manual_seed(7), sampler=wrapper)
    torch.testing.assert_close(graph, eager)

    if cfg_strength >= 1e-5:
        run_spy.assert_not_called()
        assert run_cfg_spy.call_count == 2
    else:
        assert run_spy.call_count == 2
        run_cfg_spy.assert_not_called()

    capture_spy.assert_not_called()
    assert not wrapper._cache


def test_graph_inputs_use_bounded_length_buckets() -> None:
    wrapper = AuKCUDAGraphWrapper(_make_dit("cpu"))
    x = torch.ones(1, 65, 4)
    text = torch.ones(1, 65, 8)
    c_mask = torch.ones(1, 65, dtype=torch.bool)
    ref = torch.ones(1, 51, 4)
    ref_mask = torch.ones(1, 51, dtype=torch.bool)

    padded = wrapper._bucket_inputs(x, text, c_mask, ref, ref_mask)

    assert padded[0].shape == (1, 128, 4)
    assert padded[1].shape == (1, 128)
    assert padded[2].shape == (1, 128, 8)
    assert padded[3].shape == (1, 128)
    assert padded[4].shape == (1, 100, 4)
    assert padded[5].shape == (1, 100)
    assert [mask.sum().item() for mask in (padded[1], padded[3], padded[5])] == [65, 65, 51]
    assert wrapper._key(padded[0], padded[2], padded[4], False) == (128, 128, 100, False)
    assert wrapper.max_graphs == 32


@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_bucket_padding_preserves_real_frame_outputs(cfg_strength: float) -> None:
    dit = _make_dit("cpu")
    wrapper = AuKCUDAGraphWrapper(dit)
    inputs = _sample_inputs("cpu")
    x = torch.randn(1, 9, 4)
    timestep = torch.tensor(0.4)
    cfg = torch.tensor(cfg_strength)
    if cfg_strength >= 1e-5:
        eager = wrapper._run_cfg(
            x,
            None,
            inputs["text"],
            inputs["c_mask"],
            inputs["ref"],
            inputs["ref_mask"],
            timestep,
            cfg_strength=cfg,
        )
    else:
        eager = wrapper._run(
            x,
            None,
            inputs["text"],
            inputs["c_mask"],
            inputs["ref"],
            inputs["ref_mask"],
            timestep,
        )
    dit.clear_cache()

    bucketed = wrapper._bucket_inputs(x, inputs["text"], inputs["c_mask"], inputs["ref"], inputs["ref_mask"])
    if cfg_strength >= 1e-5:
        padded = wrapper._run_cfg(*bucketed, timestep, cfg_strength=cfg)
    else:
        padded = wrapper._run(*bucketed, timestep)
    dit.clear_cache()

    torch.testing.assert_close(padded[:, : x.shape[1]], eager)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_single_request_graph_replay_matches_eager_and_updates_inputs(cfg_strength: float) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit)

    for offset in (0.0, 0.25):
        inputs = _sample_inputs("cuda", offset)
        common = dict(
            **inputs,
            gen_frames=9,
            t_grid=[0.0, 0.4, 1.0],
            cfg_strength=cfg_strength,
        )
        eager = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7))
        graph = sample_latents(dit, **common, generator=torch.Generator(device="cuda").manual_seed(7), sampler=wrapper)
        torch.testing.assert_close(graph, eager, atol=3e-6, rtol=3e-5)

        bucketed = wrapper._bucket_inputs(
            torch.empty(1, 9, 4, device="cuda"),
            inputs["text"],
            inputs["c_mask"],
            inputs["ref"],
            inputs["ref_mask"],
        )
        key = wrapper._key(bucketed[0], bucketed[2], bucketed[4], cfg_strength >= 1e-5)
        assert key in wrapper._cache

        entry = wrapper._cache[key]
        torch.testing.assert_close(entry.static_x_mask, bucketed[1])
        torch.testing.assert_close(entry.static_text, bucketed[2])
        torch.testing.assert_close(entry.static_c_mask, bucketed[3])
        torch.testing.assert_close(entry.static_ref, bucketed[4])
        torch.testing.assert_close(entry.static_ref_mask, bucketed[5])
        torch.testing.assert_close(entry.static_timestep, torch.tensor(0.4, device="cuda"))

    assert len(wrapper._cache) == 1


# Regression test for #6457


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_lru_eviction_keeps_retained_entries_replayable(monkeypatch: pytest.MonkeyPatch) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit, max_graphs=3)

    inputs = _sample_inputs("cuda")
    workspace_size = int(torch._C._cuda_getCublasWorkspaceSize())
    torch._C._cuda_clearCublasWorkspaces()
    before_a_addresses = {
        int(block["address"])
        for segment in torch.cuda.memory._snapshot()["segments"]
        for block in segment["blocks"]
        if block["state"] == "active_allocated" and int(block.get("requested_size", 0)) == workspace_size
    }

    real_cuda_graph = torch.cuda.graph
    clear_before_a_capture = True

    def graph_with_fresh_cublas_workspace(*args, **kwargs):
        nonlocal clear_before_a_capture
        if clear_before_a_capture:
            torch._C._cuda_clearCublasWorkspaces()
            clear_before_a_capture = False
        return real_cuda_graph(*args, **kwargs)

    with monkeypatch.context() as capture_patch:
        capture_patch.setattr(torch.cuda, "graph", graph_with_fresh_cublas_workspace)
        _assert_graph_matches_eager(dit, wrapper, gen_frames=64)

    assert wrapper._pool_handle is not None
    workspace_addresses_after_a = _active_workspace_addresses(wrapper._pool_handle, workspace_size)
    print(f"after A workspace addresses: {workspace_addresses_after_a}")
    workspace_candidates = workspace_addresses_after_a - before_a_addresses
    assert len(workspace_candidates) == 1, (
        f"expected one new capture-time cuBLAS workspace of {workspace_size} bytes, got {workspace_candidates}"
    )
    assert len(workspace_addresses_after_a) == 1
    workspace_address = next(iter(workspace_addresses_after_a))
    after_a = _find_address(workspace_address)
    assert after_a is not None
    assert after_a["requested_size"] == workspace_size
    assert after_a["state"] == "active_allocated"

    _assert_graph_matches_eager(dit, wrapper, gen_frames=65)
    workspace_addresses_after_b = _active_workspace_addresses(wrapper._pool_handle, workspace_size)
    print(f"after B workspace addresses: {workspace_addresses_after_b}")
    assert workspace_addresses_after_b == {workspace_address}

    _assert_graph_matches_eager(dit, wrapper, gen_frames=129)
    workspace_addresses_after_c = _active_workspace_addresses(wrapper._pool_handle, workspace_size)
    print(f"after C workspace addresses: {workspace_addresses_after_c}")
    assert workspace_addresses_after_c == {workspace_address}

    keys = []
    for gen_frames in (64, 65, 129):
        bucketed = wrapper._bucket_inputs(
            torch.empty(1, gen_frames, 4, device="cuda"),
            inputs["text"],
            inputs["c_mask"],
            inputs["ref"],
            inputs["ref_mask"],
        )
        keys.append(wrapper._key(bucketed[0], bucketed[2], bucketed[4], False))
    key_a, key_b, key_c = keys

    before_eviction = _find_address(workspace_address)
    assert before_eviction is not None
    assert before_eviction["state"] == "active_allocated"

    _assert_graph_matches_eager(dit, wrapper, gen_frames=193)
    bucketed_d = wrapper._bucket_inputs(
        torch.empty(1, 193, 4, device="cuda"),
        inputs["text"],
        inputs["c_mask"],
        inputs["ref"],
        inputs["ref_mask"],
    )
    key_d = wrapper._key(bucketed_d[0], bucketed_d[2], bucketed_d[4], False)

    assert key_a not in wrapper._cache
    assert set(wrapper._cache) == {key_b, key_c, key_d}

    torch._C._cuda_clearCublasWorkspaces()
    torch.accelerator.synchronize()
    gc.collect()
    after_eviction = _find_address(workspace_address)
    assert after_eviction is not None
    assert after_eviction["state"] == "inactive"

    prefix_bytes = workspace_address - after_eviction["block_address"]
    reuse_graph, reuse_prefix, reuse_allocation = _capture_pool_reuse_graph(
        wrapper._pool_handle,
        workspace_size,
        prefix_bytes=prefix_bytes,
    )
    assert reuse_allocation.data_ptr() == workspace_address
    after_reuse = _find_address(workspace_address)
    assert after_reuse is not None
    assert after_reuse["state"] == "active_allocated"
    assert after_reuse["requested_size"] == workspace_size
    assert after_reuse["segment_pool_id"] == after_eviction["segment_pool_id"]
    assert after_reuse["segment_pool_id"] == wrapper._pool_handle

    reuse_graph.replay()
    torch.accelerator.synchronize()
    _assert_graph_matches_eager(dit, wrapper, gen_frames=65)
    _assert_graph_matches_eager(dit, wrapper, gen_frames=129)
    del reuse_graph, reuse_prefix, reuse_allocation


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_capture_failure_is_propagated(mocker) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit)
    mocker.patch.object(wrapper, "_capture", side_effect=RuntimeError("capture failed"))
    inputs = _sample_inputs("cuda")

    with pytest.raises(RuntimeError, match="capture failed"):
        sample_latents(
            dit,
            **inputs,
            gen_frames=9,
            t_grid=[0.0, 0.4, 1.0],
            cfg_strength=0.0,
            generator=torch.Generator(device="cuda").manual_seed(7),
            sampler=wrapper,
        )
