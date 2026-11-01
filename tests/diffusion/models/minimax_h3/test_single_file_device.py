# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real HSDP loading and pre-sharded rejection for a compressed H3 checkpoint."""

import glob
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda, pytest.mark.parallel]


def _load_worker(rank, checkpoint_dir, rendezvous):
    from safetensors.torch import load_file, save_file
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.load import LoadConfig
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    from vllm_omni.diffusion.config import set_current_diffusion_config
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
    from vllm_omni.diffusion.distributed import hsdp
    from vllm_omni.diffusion.forward_context import set_forward_context
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    torch.set_num_threads(2)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    checkpoint = Path(checkpoint_dir) / "hybrid[beta5].safetensors"
    arch = dict(
        num_layers=1,
        token_refiner_num_layers=1,
        hidden_size=64,
        num_attention_heads=2,
        attention_head_dim=32,
        ffn_hidden_size=128,
        latents_dim=2,
        audio_latents_dim=2,
        patch_size=[1, 2, 2],
        text_dim=6,
        time_embed_dim=8,
        adaln_curve_grid=5,
        adaln_out_features=1152,
        final_adaln_out_features=128,
        rope_inv_freq_len=2,
    )
    config = OmniDiffusionConfig(
        model=checkpoint_dir,
        model_class_name="MiniMaxH3Pipeline",
        tf_model_config=TransformerConfig.from_dict(arch),
        diffusion_attention_config={"default": "TORCH_SDPA"},
        parallel_config=DiffusionParallelConfig(use_hsdp=True, hsdp_shard_size=2),
    )
    vconfig = VllmConfig(device_config=DeviceConfig(device="cuda"))
    with set_current_vllm_config(vconfig):
        init_distributed_environment(
            world_size=2,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"file://{rendezvous}",
            backend="nccl",
            timeout=timedelta(seconds=120),
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        try:

            class Pipeline(nn.Module):
                _dit_modules = ["transformer"]
                _encoder_modules = []
                remap_checkpoint_key = MiniMaxH3Pipeline.remap_checkpoint_key

                def __init__(self):
                    super().__init__()
                    with set_current_diffusion_config(config):
                        self.transformer = MiniMaxH3DiTModel(config, diffusers_weights=False)
                    self.weights_sources = [
                        DiffusersPipelineLoader.ComponentSource(
                            checkpoint_dir,
                            None,
                            None,
                            "transformer.",
                            False,
                            allow_patterns_overrides=[glob.escape(checkpoint.name)],
                        )
                    ]

                def load_weights(self, weights):
                    loaded = self.transformer.load_weights(
                        (name.removeprefix("transformer."), weight) for name, weight in weights
                    )
                    self.transformer.post_load_weights()
                    return {"transformer." + name for name in loaded}

            if rank == 0:
                torch.manual_seed(42)
                template = Pipeline().transformer
                with torch.no_grad():
                    for name, parameter in template.named_parameters():
                        parameter.copy_(torch.randn_like(parameter) * 0.03)
                        if "norm" in name:
                            parameter.fill_(1)
                    template.rope.inv_freq.fill_(0.01)
                weights = {
                    "adaln_t_table" if name == "time_embedder.table" else name: value.to(torch.bfloat16).contiguous()
                    for name, value in template.state_dict().items()
                }
                save_file(weights, checkpoint)
                # The selected file must override both sibling files and indices.
                save_file(
                    {name: torch.zeros_like(value) for name, value in weights.items()},
                    checkpoint.with_name("other.safetensors"),
                )
                checkpoint.with_name("model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {name: "other.safetensors" for name in weights}})
                )
            torch.distributed.barrier()
            group = SimpleNamespace(world_size=2, rank_in_group=rank)
            with patch.object(hsdp, "get_world_group", return_value=group):

                def load():
                    loader = DiffusersPipelineLoader(LoadConfig(), config)
                    with patch.object(loader, "_init_from_load_format", return_value=Pipeline()):
                        return loader._load_model_with_hsdp(device)

                full = load()
                config.hsdp_weight_load_strategy = "pre_sharded"
                with pytest.raises(ValueError, match="tensor transforms are unsupported") as error:
                    load()
                assert "transformer.time_embedder.table" in str(error.value)
                assert "transformer.blocks.0.attn.qkv_proj.weight" in str(error.value)
            assert full.transformer.time_embedder.table.dtype == torch.float32
            stored = load_file(checkpoint)
            torch.testing.assert_close(
                full.transformer.time_embedder.table.cpu(), stored["adaln_t_table"].float(), rtol=0, atol=0
            )
            assert full.transformer.blocks[0].adaln_proj.linear.weight.dtype == torch.float32
            assert full.transformer.final_layer.adaln_proj.linear.weight.dtype == torch.float32

            inputs = dict(
                x=torch.randn(1, 8, 8, device=device),
                audio_x=torch.randn(1, 8, 2, device=device),
                prompt_embeds=torch.randn(2, 6, device=device),
                img_position_ids=torch.zeros(1, 8, 3, device=device, dtype=torch.long),
                unique_timesteps=torch.tensor([0.67, 0.3, 0.0], device=device),
                inverse_indices=torch.tensor([2, 2, 0, 0, 0, 0, 1, 1], device=device),
                token_tags=torch.tensor([0, 0, 1, 1, 1, 1, 2, 2], device=device),
                update_mask=torch.ones(4, device=device),
                img_pos_info={"position_ids": torch.arange(2, 6, device=device)},
                audio_pos_info={"position_ids": torch.arange(6, 8, device=device)},
                text_pos_info={"position_ids": torch.arange(2, device=device)},
                img_pos_for_infer_output_info={"position_ids": torch.arange(2, 6, device=device)},
                packed_seq_params={
                    "cu_seqlens_q": torch.tensor([0, 8, 8], device=device, dtype=torch.int32),
                    "max_seqlen_q": 8,
                },
                refiner_packed_seq_params={
                    "cu_seqlens_q": torch.tensor([0, 2, 2], device=device, dtype=torch.int32),
                    "max_seqlen_q": 2,
                },
            )
            with torch.no_grad(), set_forward_context(vllm_config=vconfig, omni_diffusion_config=config):
                reference_output = full.transformer(**inputs)
                for _ in range(2):
                    output = full.transformer(**inputs)
                    for observed, reference in zip(output, reference_output, strict=True):
                        assert torch.isfinite(observed).all()
                        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        finally:
            cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "world_size", [pytest.param(2, marks=hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=2))]
)
def test_compressed_checkpoint_hsdp_load_and_pre_sharded_rejection(tmp_path, world_size):
    if not torch.cuda.is_available() or torch.accelerator.device_count() < world_size:
        pytest.skip("Requires two CUDA GPUs")
    context = mp.spawn(_load_worker, args=(str(tmp_path), str(tmp_path / "rendezvous")), nprocs=world_size, join=False)
    try:
        for _ in range(2):
            if context.join(timeout=90):
                return
        pytest.fail("H3 HSDP loading workers timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
