# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the real native H3 projection/refiner/DiT stack on CUDA."""

from __future__ import annotations

from pathlib import Path
from time import monotonic

import pytest
import torch
import torch.multiprocessing as mp

from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


def _native_worker(rank: int, rendezvous: str, world_size: int = 1, output_root: str | None = None) -> None:
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
    from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
    from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
    from vllm_omni.diffusion.models.solarwm_h3.attention import SolarAttentionContext
    from vllm_omni.diffusion.models.solarwm_h3.cache import SolarRawKVCache
    from vllm_omni.diffusion.models.solarwm_h3.layout import build_layout
    from vllm_omni.diffusion.models.solarwm_h3.transformer import SolarWMH3Transformer
    from vllm_omni.platforms import current_omni_platform

    torch.set_num_threads(1)
    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    od_config = OmniDiffusionConfig(
        model="solarwm-h3-tiny",
        model_class_name="SolarWMH3Pipeline",
        dtype=torch.bfloat16,
        tf_model_config=TransformerConfig.from_dict(
            {
                "num_layers": 2,
                "token_refiner_num_layers": 1,
                "hidden_size": 128,
                "num_attention_heads": 2,
                "attention_head_dim": 128,
                "ffn_hidden_size": 256,
                "text_dim": 8,
                "timestep_input_dim": 32,
                "time_embed_hidden_size": 128,
                "time_embed_dim": 64,
                "adaln_out_features": 18 * 128,
                "final_adaln_out_features": 2 * 128,
            }
        ),
        parallel_config=DiffusionParallelConfig(sequence_parallel_size=world_size, ulysses_degree=world_size),
        cache_config={"minimax_h3_adaln_offload": False},
    )
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=rank,
        distributed_init_method=f"file://{rendezvous}",
    )
    try:
        initialize_model_parallel(sequence_parallel_size=world_size, ulysses_degree=world_size)
        with (
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))),
            set_current_diffusion_config(od_config),
            set_forward_context(omni_diffusion_config=od_config),
        ):
            model = SolarWMH3Transformer(od_config, diffusers_weights=False).to(device).eval()
            if world_size > 1:
                apply_sequence_parallel(model, SequenceParallelConfig(ulysses_degree=world_size), model._sp_plan)
                get_forward_context().sp_plan_hooks_applied = True
            seen_inputs = {}

            def observe_input(name):
                def observe(module, args):
                    del module
                    seen_inputs[name] = args[0].clone()

                return observe

            model.video_patch_proj.register_forward_pre_hook(observe_input("video"))
            model.audio_patch_proj.register_forward_pre_hook(observe_input("audio"))
            generator = torch.Generator(device=device).manual_seed(41)
            with torch.no_grad():
                model.rope.inv_freq.copy_(10000 ** -(torch.arange(16, device=device, dtype=torch.float32) / 16))
                for name, parameter in model.named_parameters():
                    if name.endswith("norm.weight"):
                        parameter.fill_(1.0)
                    else:
                        parameter.copy_(
                            torch.randn(parameter.shape, device=device, dtype=parameter.dtype, generator=generator)
                            * 0.02
                        )
            tags = torch.tensor([1, 0, 1], dtype=torch.long, device=device)
            layout = build_layout(tags, latent_height=2, latent_width=4, num_audio_latents=2, device=device)
            prompt = torch.randn(1, 3, 8, dtype=torch.bfloat16, device=device, generator=generator)
            anchor = torch.randn(1, 2, 96, dtype=torch.float32, device=device, generator=generator)
            audio = torch.randn(1, 4, 32, dtype=torch.float32, device=device, generator=generator)
            cache = SolarRawKVCache(num_layers=2)
            # Whole trajectories stay on CPU; only the current window is copied.
            viewmats = torch.eye(4).repeat(1, 41, 1, 1)
            viewmats[0, :, 0, 3] = torch.arange(41) * 0.03
            intrinsics = torch.eye(3).repeat(1, 41, 1, 1)
            predictions: list[torch.Tensor] = []
            with torch.inference_mode():
                for index in range(8):
                    current = torch.randn(1, 24, 5, 2, 4, dtype=torch.float32, device=device, generator=generator)
                    context = SolarAttentionContext(layout, viewmats, intrinsics, index, cache)
                    probe = model.predict_video(prompt, anchor, current, audio, layout, 0.4, 0.5, context)
                    assert probe.shape == current.shape and torch.isfinite(probe).all()
                    assert seen_inputs["video"].dtype is torch.float32
                    assert seen_inputs["audio"].dtype is torch.float32
                    if world_size == 1:
                        torch.testing.assert_close(seen_inputs["video"][:2], anchor[0], atol=0, rtol=0)
                        torch.testing.assert_close(seen_inputs["audio"], audio[0], atol=0, rtol=0)
                    assert cache.pending == {} and cache.chunk_index == index
                    clean = SolarAttentionContext(layout, viewmats, intrinsics, index, cache, commit_cache=True)
                    result = model.predict_video(prompt, anchor, current, audio, layout, 1.0, 0.5, clean)
                    predictions.extend((probe.cpu(), result.cpu()))
                    assert result.shape == current.shape and torch.isfinite(result).all()
                    assert set(cache.pending) == {0, 1} and cache.chunk_index == index
                    cache.commit_chunk(index)
                    for layer in range(2):
                        history = cache.layers[layer]
                        assert [entry[0] for entry in history] == list(range(max(0, index - 4), index + 1))
                        assert history[-1][1].shape == (1, 10, 2 // world_size, 128)
                        assert history[-1][1].dtype is torch.bfloat16
            assert model.adaln_cache.hits > 0
            if output_root is not None:
                torch.save(
                    {
                        "predictions": torch.stack(predictions),
                        "keys": torch.stack([entry[1].cpu() for layer in cache.layers.values() for entry in layer]),
                        "values": torch.stack([entry[2].cpu() for layer in cache.layers.values() for entry in layer]),
                    },
                    Path(output_root) / f"world{world_size}_rank{rank}.pt",
                )
    finally:
        destroy_distributed_env()


def test_real_native_transformer_predicts_and_commits_bounded_history(tmp_path, monkeypatch):
    if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() < 1:
        pytest.skip("requires CUDA")
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    process = mp.spawn(_native_worker, args=(str(tmp_path / "rendezvous"),), nprocs=1, join=False)
    try:
        if not process.join(timeout=120):
            pytest.fail("native SolarWM-H3 worker timed out")
    finally:
        for worker in process.processes:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)


@pytest.mark.parallel
def test_pure_ulysses_matches_native_predictions_and_raw_history(tmp_path, monkeypatch):
    if not current_omni_platform.is_cuda() or current_omni_platform.get_device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    for world_size in (1, 2):
        process = mp.spawn(
            _native_worker,
            args=(str(tmp_path / f"rendezvous{world_size}"), world_size, str(tmp_path)),
            nprocs=world_size,
            join=False,
        )
        try:
            deadline = monotonic() + 180
            while not process.join(timeout=30):
                if monotonic() > deadline:
                    pytest.fail("native SolarWM-H3 Ulysses worker timed out")
        finally:
            for worker in process.processes:
                if worker.is_alive():
                    worker.terminate()
                worker.join(timeout=5)
    baseline = torch.load(tmp_path / "world1_rank0.pt", weights_only=True)
    ranks = [torch.load(tmp_path / f"world2_rank{rank}.pt", weights_only=True) for rank in (0, 1)]
    for result in ranks:
        torch.testing.assert_close(result["predictions"], baseline["predictions"], atol=2e-4, rtol=2e-3)
    for key in ("keys", "values"):
        # Ulysses owns one of the two original heads on each rank. It must
        # retain those raw heads, before any window or camera rotation.
        joined = torch.cat([result[key] for result in ranks], dim=3)
        torch.testing.assert_close(joined, baseline[key], atol=2e-3, rtol=2e-3)
