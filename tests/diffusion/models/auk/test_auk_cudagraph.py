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


def _sample_inputs(device: str, offset: float = 0.0, ref_frames: int = 4) -> dict[str, torch.Tensor]:
    """``ref_frames=0`` is the text-only (instruct TTS) request shape."""
    generator = torch.Generator(device=device).manual_seed(42)
    return {
        "text": torch.randn(1, 7, 8, generator=generator, device=device) + offset,
        "c_mask": torch.ones(1, 7, dtype=torch.bool, device=device),
        "ref": (torch.randn(1, 4, 4, generator=generator, device=device) - offset)[:, :ref_frames],
        "ref_mask": torch.ones(1, ref_frames, dtype=torch.bool, device=device),
    }


@torch.inference_mode()
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_single_request_graph_wrapper_cpu_falls_back_to_eager(cfg_strength: float, mocker) -> None:
    dit = _make_dit("cpu")
    wrapper = AuKCUDAGraphWrapper(dit)
    prepare_spy = mocker.spy(wrapper, "_prepare")
    step_spy = mocker.spy(wrapper, "_step")
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

    # The conditioning is prepared once per request and reused by every step.
    assert prepare_spy.call_count == 1
    assert prepare_spy.call_args.args[-1] is (cfg_strength >= 1e-5)
    assert step_spy.call_count == 2
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

    assert padded[0].shape == (1, 96, 4)
    assert padded[1].shape == (1, 96)
    assert padded[2].shape == (1, 96, 8)
    assert padded[3].shape == (1, 96)
    assert padded[4].shape == (1, 100, 4)
    assert padded[5].shape == (1, 100)
    assert [mask.sum().item() for mask in (padded[1], padded[3], padded[5])] == [65, 65, 51]
    assert wrapper._key(padded[0], padded[2], padded[4], False) == (96, 96, 100, False)
    assert wrapper.max_graphs == 32


def test_full_graph_cache_retires_as_one_generation() -> None:
    wrapper = AuKCUDAGraphWrapper(_make_dit("cpu"), max_graphs=2)
    first = (64, 64, 50, False)
    second = (128, 64, 50, False)
    wrapper._cache[first] = object()

    wrapper._retire_graph_generation_if_full()
    assert list(wrapper._cache) == [first]

    wrapper._cache[second] = object()
    wrapper._retire_graph_generation_if_full()
    assert not wrapper._cache


@torch.inference_mode()
@pytest.mark.parametrize("ref_frames", [0, 4])
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_bucket_padding_preserves_real_frame_outputs(cfg_strength: float, ref_frames: int) -> None:
    dit = _make_dit("cpu")
    wrapper = AuKCUDAGraphWrapper(dit)
    inputs = _sample_inputs("cpu", ref_frames=ref_frames)
    x = torch.randn(1, 9, 4)
    timestep = torch.tensor(0.4)
    cfg = torch.tensor(cfg_strength)
    uses_cfg = cfg_strength >= 1e-5
    text, c_mask, ref, ref_mask = inputs["text"], inputs["c_mask"], inputs["ref"], inputs["ref_mask"]

    ctx = wrapper._prepare(x, None, text, c_mask, ref, ref_mask, uses_cfg)
    eager = wrapper._step(x, timestep, ctx, cfg)

    bucketed = wrapper._bucket_inputs(x, text, c_mask, ref, ref_mask)
    padded_ctx = wrapper._prepare(*bucketed, uses_cfg)
    padded = wrapper._step(bucketed[0], timestep, padded_ctx, cfg)

    torch.testing.assert_close(padded[:, : x.shape[1]], eager)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
@pytest.mark.parametrize("ref_frames", [0, 4])
@pytest.mark.parametrize("cfg_strength", [0.0, 2.0])
def test_single_request_graph_replay_matches_eager_and_updates_inputs(cfg_strength: float, ref_frames: int) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit)

    for offset in (0.0, 0.25):
        inputs = _sample_inputs("cuda", offset, ref_frames=ref_frames)
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

        # The static context was refreshed for this request's conditioning.
        entry = wrapper._cache[key]
        fresh = wrapper._prepare(*bucketed, cfg_strength >= 1e-5)
        for static, want in zip(entry.static_ctx.tensors(), fresh.tensors(), strict=True):
            if want is None:
                assert static is None
            else:
                torch.testing.assert_close(static, want)
        torch.testing.assert_close(entry.static_timestep, torch.tensor(0.4, device="cuda"))

    assert len(wrapper._cache) == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
@torch.inference_mode()
def test_graph_capture_failure_is_propagated(mocker) -> None:
    dit = _make_dit("cuda")
    wrapper = AuKCUDAGraphWrapper(dit)
    capture = mocker.patch.object(wrapper, "_capture", side_effect=RuntimeError("capture failed"))
    eager = mocker.spy(wrapper, "_step")
    inputs = _sample_inputs("cuda")
    common = dict(
        **inputs,
        gen_frames=9,
        t_grid=[0.0, 0.4, 1.0],
        cfg_strength=0.0,
    )

    with pytest.raises(RuntimeError, match="capture failed"):
        sample_latents(
            dit,
            **common,
            generator=torch.Generator(device="cuda").manual_seed(7),
            sampler=wrapper,
        )

    result = sample_latents(
        dit,
        **common,
        generator=torch.Generator(device="cuda").manual_seed(7),
        sampler=wrapper,
    )

    assert result.shape == (1, 9, 4)
    assert capture.call_count == 1
    assert eager.call_count == 2
    assert not wrapper.enabled
    assert not wrapper._cache
