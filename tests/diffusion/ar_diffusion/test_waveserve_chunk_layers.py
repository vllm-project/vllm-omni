# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""WaveServe model unit checks (layer split + FlowEuler + CPU tiny helper).

Omni / diffusion registry wiring lives in
``tests/entrypoints/test_resolve_waveserve_wan_config.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.ar_diffusion.waveserve_tiny import TinyChunkAdapter, TinyStageWanTransformer
from vllm_omni.diffusion.models.waveserve_wan.pipeline_waveserve_wan import (
    FlowEuler,
    WaveServeWanPipeline,
    _LatentChunkAdapter,
)
from vllm_omni.diffusion.models.waveserve_wan.transformer import stage_layer_range
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.experimental.ar_diffusion.chunk_executor import (
    ARDiffusionChunkContext,
    ChunkRunSpec,
    ChunkTopology,
    run_chunk_pipeline,
)
from vllm_omni.experimental.ar_diffusion.chunk_schedule import ChunkSchedule, Ordering, build_chunk_plan
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec, NoisyKVCache, NoisyKVState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stage_layer_range_covers_all_layers():
    covered = []
    for g in range(2):
        start, end = stage_layer_range(30, g, 2)
        covered.extend(range(start, end))
    assert covered == list(range(30))
    assert stage_layer_range(30, 0, 2) == (0, 15)
    assert stage_layer_range(30, 1, 2) == (15, 30)


def test_flow_euler_matches_diffusers_shifted_endpoint():
    """Match FlowMatchEulerDiscreteScheduler: linspace ends at warp(1/1000), then warp again."""
    shift = 5.0
    steps = 4

    def warp(sigma: torch.Tensor) -> torch.Tensor:
        return shift * sigma / (1 + (shift - 1) * sigma)

    sampler = FlowEuler(steps, shift=shift)
    expected = warp(torch.linspace(1.0, float(warp(torch.tensor(1.0 / 1000))), steps, dtype=torch.float64))
    assert sampler.sigmas == [*expected.float().tolist(), 0.0]
    # Diffusers' own endpoint double-warp yields a larger last step than warp(1/1000).
    last_delta = abs(sampler.sigmas[-1] - sampler.sigmas[-2])
    assert last_delta > 0.01


def test_waveserve_chunk_noise_matches_reference_seed():
    shape = (1, 16, 1, 2, 2)
    seed, chunk = 7, 2
    adapter = _LatentChunkAdapter(
        SimpleNamespace(),  # type: ignore[arg-type]
        sampler=FlowEuler(3, shift=5.0),
        prompt_embeds=torch.empty(1, 1, 32),
        latent_shape=shape,
        seed=seed,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    reference = torch.randn(shape, generator=torch.Generator().manual_seed(seed * 1_000_003 + chunk * 4096))
    torch.testing.assert_close(adapter._init_noise(chunk), reference)
    torch.testing.assert_close(adapter._init_noise(chunk), reference)


def test_chunk_schedule_rejects_unknown_values():
    pipeline = WaveServeWanPipeline.__new__(WaveServeWanPipeline)
    pipeline.stage_parallel_size = 1
    pipeline.layer_groups = 1
    pipeline.max_history_chunks = 1
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="sched-0",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"chunk_schedule": "latset"}),
    )
    with pytest.raises(ValueError, match="chunk_schedule"):
        pipeline._plan_for(req)


def test_waveserve_tiny_helper_forward_cpu():
    transformer = TinyStageWanTransformer(num_layers=2, dim=32, num_heads=2, ffn_dim=64)
    block_size = 16
    cache = NoisyKVCache(
        ARDiffusionNoisyKVSpec(
            num_layers=transformer.local_num_layers,
            num_kv_heads=transformer.num_heads,
            head_size=transformer.head_dim,
            block_size=block_size,
            max_chunk_tokens=block_size,
            max_history_chunks=1,
        ),
        dtype=torch.float32,
        device=torch.device("cpu"),
        layer_groups=1,
        max_batch_size=1,
    )
    ctx = ARDiffusionChunkContext(
        spec=ChunkRunSpec(topology=ChunkTopology(stages=1, layer_groups=1), rank=0),
        kv=NoisyKVState(cache),
    )
    plan = build_chunk_plan(
        ChunkSchedule(
            chunks=1,
            num_denoise_steps=1,
            stages=1,
            layer_groups=1,
            ordering=Ordering.SERIAL,
            kv_history_chunks=1,
        )
    )
    ctx.enqueue("ws-0", plan, chunk_tokens=block_size)
    seed_dim = transformer.in_features if transformer.is_stage_first else transformer.dim
    hidden = torch.zeros(1, block_size, seed_dim, dtype=torch.float32)
    run_chunk_pipeline(ctx=ctx, adapter=TinyChunkAdapter(transformer, seed_hidden=hidden))
    assert not ctx.inflight
    assert len(cache.pool.keys) == 0


def test_latent_adapter_g_gt1_packs_hidden_and_advances_on_last_only():
    """Non-last groups forward tokens; only stage-last runs FlowEuler."""
    from vllm.sequence import IntermediateTensors

    shape = (1, 4, 1, 2, 2)
    device = torch.device("cpu")
    dtype = torch.float32
    prompt = torch.zeros(1, 8, 16, device=device, dtype=dtype)

    class _FakeTransformer:
        def __init__(self, *, is_stage_last: bool) -> None:
            self.is_stage_last = is_stage_last

        def forward_latent_step(
            self, latent, *, timestep, encoder_hidden_states, kv_contexts=None, intermediate_tensors=None
        ):
            del timestep, encoder_hidden_states, kv_contexts
            if not self.is_stage_last:
                assert intermediate_tensors is None or "hidden_states" in intermediate_tensors.tensors
                tokens = torch.ones(latent.shape[0], 3, 8, device=latent.device, dtype=latent.dtype)
                if intermediate_tensors is not None:
                    tokens = tokens + intermediate_tensors["hidden_states"]
                return IntermediateTensors({"hidden_states": tokens})
            # Stage-last: return a 5D pred matching latent shape.
            return torch.zeros_like(latent)

    sampler = FlowEuler(2, shift=1.0)
    mid = _LatentChunkAdapter(
        _FakeTransformer(is_stage_last=False),  # type: ignore[arg-type]
        sampler=sampler,
        prompt_embeds=prompt,
        latent_shape=shape,
        seed=0,
        device=device,
        dtype=dtype,
    )
    last = _LatentChunkAdapter(
        _FakeTransformer(is_stage_last=True),  # type: ignore[arg-type]
        sampler=sampler,
        prompt_embeds=prompt,
        latent_shape=shape,
        seed=0,
        device=device,
        dtype=dtype,
    )

    tasks = [("r0", (0, 0))]
    mid_out = mid.forward(tasks, [None], hidden=None)
    assert set(mid_out) == {"latent", "hidden_states"}
    packed = mid.pack_activation(mid_out)
    assert "latent" in packed and "hidden_states" in packed

    last_in = last.unpack_activation(packed)
    last_out = last.forward(tasks, [None], hidden=last_in)
    assert set(last_out) == {"latent"}
    assert 0 not in last.finished.get("r0", {})
    # Second denoise step finishes the chunk on stage-last.
    last_out2 = last.forward([("r0", (0, 1))], [None], hidden=last_out)
    assert 0 in last.finished["r0"]
    assert last_out2["latent"].shape == shape
