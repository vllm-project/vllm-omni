# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Two-rank Mammoth stage-construction and real VAE-weight decode smoke test."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.registry import initialize_model
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
assert world_size == 2
torch.accelerator.set_device_index(local_rank)
torch.manual_seed(2026)
init_distributed_environment(world_size=world_size, rank=rank, local_rank=local_rank, backend="nccl")
initialize_model_parallel(sequence_parallel_size=2, ulysses_degree=2, backend="nccl")

try:
    config_path = Path("/root/autodl-tmp/mammoth-vae-weights/config.json")
    shard_path = Path("/root/autodl-tmp/mammoth-vae-weights/model-00006-of-00008.safetensors")
    raw_config = json.loads(config_path.read_text())
    od_config = OmniDiffusionConfig(
        model=str(config_path.parent),
        model_class_name="MammothModa2DiTPipeline",
        tf_model_config=TransformerConfig.from_dict(raw_config),
        parallel_config=DiffusionParallelConfig(
            data_parallel_size=1,
            sequence_parallel_size=2,
            ulysses_degree=2,
            vae_patch_parallel_size=2,
        ),
        num_gpus=2,
        dtype=torch.float16,
    )
    start = time.perf_counter()
    with set_forward_context(omni_diffusion_config=od_config):
        pipeline = initialize_model(od_config)
    constructor_sec = time.perf_counter() - start
    assert isinstance(pipeline.gen_vae, DistributedAutoencoderKL)
    assert pipeline.gen_vae.use_tiling and od_config.vae_use_tiling
    assert pipeline.gen_vae.distributed_executor.parallel_size == 2
    assert not hasattr(pipeline, "vae")
    assert all(
        block.attn.omni_attn.skip_sequence_parallel
        for block in [
            *pipeline.gen_transformer.context_refiner,
            *pipeline.gen_transformer.noise_refiner,
            *pipeline.gen_transformer.layers,
        ]
    )

    pipeline.gen_transformer.to(device=f"cuda:{local_rank}", dtype=torch.float16).eval()
    dit_latents = torch.randn((1, 16, 8, 8), generator=torch.Generator().manual_seed(17)).to(
        device=f"cuda:{local_rank}", dtype=torch.float16
    )
    llm_hidden_size = pipeline.gen_transformer.time_caption_embed.caption_embedder[0].weight.numel()
    text = torch.randn((1, 2, llm_hidden_size), generator=torch.Generator().manual_seed(19)).to(
        device=f"cuda:{local_rank}", dtype=torch.float16
    )
    with torch.inference_mode(), set_forward_context(omni_diffusion_config=od_config):
        dit_result = pipeline.gen_transformer(
            hidden_states=dit_latents,
            timestep=torch.tensor([0.5], device=f"cuda:{local_rank}"),
            text_hidden_states=text,
            text_attention_mask=torch.ones((1, 2), device=f"cuda:{local_rank}", dtype=torch.bool),
            freqs_cis=pipeline.gen_freqs_cis,
        )
    assert dit_result.shape == dit_latents.shape
    assert torch.isfinite(dit_result).all()
    dit_results = [torch.empty_like(dit_result) for _ in range(world_size)]
    dist.all_gather(dit_results, dit_result.contiguous())
    dit_max_abs_rank_delta = (dit_results[0].float() - dit_results[1].float()).abs().max().item()
    assert dit_max_abs_rank_delta < 1e-2, dit_max_abs_rank_delta

    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        names = [name for name in shard.keys() if name.startswith("gen_vae.")]
        state = {name.removeprefix("gen_vae."): shard.get_tensor(name) for name in names}
    pipeline.gen_vae.load_state_dict(state, strict=True)
    pipeline.gen_vae.to(device=f"cuda:{local_rank}", dtype=torch.float16).eval()
    if pipeline.gen_image_condition_refiner is not None:
        pipeline.gen_image_condition_refiner.to(device=f"cuda:{local_rank}", dtype=torch.float16).eval()
    latents = torch.randn(
        (1, raw_config["gen_vae_config"]["latent_channels"], 128, 128),
        generator=torch.Generator().manual_seed(7),
    )
    vae_config = raw_config["gen_vae_config"]
    latents = (latents / vae_config["scaling_factor"] + vae_config["shift_factor"]).to(
        device=f"cuda:{local_rank}", dtype=torch.float16
    )
    if rank == 1:
        latents.add_(0.5)
    before_sync = [torch.empty_like(latents) for _ in range(world_size)]
    dist.all_gather(before_sync, latents)
    latent_max_abs_before_sync = (before_sync[0].float() - before_sync[1].float()).abs().max().item()
    latents = pipeline._sync_latents_for_vae_decode(latents)
    after_sync = [torch.empty_like(latents) for _ in range(world_size)]
    dist.all_gather(after_sync, latents)
    latent_max_abs_after_sync = (after_sync[0].float() - after_sync[1].float()).abs().max().item()
    assert latent_max_abs_before_sync > 0
    assert latent_max_abs_after_sync == 0
    dist.barrier()
    with torch.inference_mode():
        result = pipeline.gen_vae.decode(latents, return_dict=False)[0]
    torch.accelerator.synchronize()

    full_hidden_states = torch.randn(
        (2, llm_hidden_size), generator=torch.Generator().manual_seed(31), dtype=torch.float16
    )
    request = OmniDiffusionRequest(
        request_id="synthetic-ar-one-step",
        prompt={
            "prompt": "",
            "height": 64,
            "width": 64,
            "additional_information": {
                "full_hidden_states": full_hidden_states,
                "full_token_ids": [10, int(pipeline.config.llm_config.gen_vocab_start_index)],
                "answer_start_index": 1,
            },
        },
        sampling_params=OmniDiffusionSamplingParams(
            height=64,
            width=64,
            seed=42,
            guidance_scale=1.0,
            num_inference_steps=1,
        ),
    )
    with torch.inference_mode(), set_forward_context(omni_diffusion_config=od_config):
        stage_result = pipeline.forward(DiffusionRequestBatch([request])).output
    stage_output_finite = bool(torch.isfinite(stage_result).all()) if stage_result.numel() else True
    assert stage_output_finite
    if rank == 0:
        assert list(stage_result.shape) == [1, 3, 64, 64]
    else:
        assert stage_result.numel() == 0

    local_record = {
        "rank": rank,
        "output_shape": list(result.shape),
        "output_numel": result.numel(),
        "output_finite": bool(torch.isfinite(result).all()) if result.numel() else True,
        "joint_dit_vae_peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
        "dit_output_shape": list(dit_result.shape),
        "dit_max_abs_rank_delta": dit_max_abs_rank_delta,
        "stage_output_shape": list(stage_result.shape),
        "stage_output_finite": stage_output_finite,
    }
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_record)
    if rank == 0:
        assert gathered[0]["output_shape"] == [1, 3, 1024, 1024]
        assert gathered[1]["output_numel"] == 0
        record = {
            "kind": "registry-stage-vae-integration-smoke",
            "world_size": world_size,
            "model_class": type(pipeline).__name__,
            "vae_class": type(pipeline.gen_vae).__name__,
            "vae_patch_parallel_size": pipeline.gen_vae.distributed_executor.parallel_size,
            "vae_use_tiling": pipeline.gen_vae.use_tiling,
            "sequence_parallel_size": od_config.parallel_config.sequence_parallel_size,
            "loaded_vae_tensors": len(names),
            "constructor_sec": constructor_sec,
            "dit_max_abs_rank_delta": dit_max_abs_rank_delta,
            "latent_max_abs_before_sync": latent_max_abs_before_sync,
            "latent_max_abs_after_sync": latent_max_abs_after_sync,
            "ranks": gathered,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "limitations": (
                "Full stage forward uses synthetic AR conditions and random/unloaded DiT weights; "
                "this is not end-to-end generation."
            ),
        }
        print(json.dumps(record, indent=2), flush=True)
        Path("/root/autodl-tmp/mammoth-vae-weights/registry-vae-smoke.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
finally:
    destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()
