# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-request eager and CUDA-graph AuK DiT sampling equivalence."""

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_lru_eviction_keeps_retained_entries_replayable() -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit, max_graphs=2)

    inputs = _sample_inputs("cuda")
    keys = []
    for gen_frames in (64, 65, 129):
        _assert_graph_matches_eager(dit, wrapper, gen_frames=gen_frames)
        bucketed = wrapper._bucket_inputs(
            torch.empty(1, gen_frames, 4, device="cuda"),
            inputs["text"],
            inputs["c_mask"],
            inputs["ref"],
            inputs["ref_mask"],
        )
        keys.append(wrapper._key(bucketed[0], bucketed[2], bucketed[4], False))
    key_a, key_b, key_c = keys

    assert key_a not in wrapper._cache
    assert set(wrapper._cache) == {key_b, key_c}
    assert wrapper._pool_handle is not None

    _assert_graph_matches_eager(dit, wrapper, gen_frames=65)
    _assert_graph_matches_eager(dit, wrapper, gen_frames=129)


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
