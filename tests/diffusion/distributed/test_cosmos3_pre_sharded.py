# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare full and pre-sharded Cosmos loading with real two-rank FSDP."""

from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model, pytest.mark.gpu]


class _SDPAAttention(nn.Module):
    """Use deterministic native attention while testing loading and FSDP."""

    def __init__(self, *, causal, softmax_scale, **kwargs):
        super().__init__()
        self.causal = causal
        self.scale = softmax_scale

    def forward(self, query, key, value):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=self.causal,
            scale=self.scale,
        ).transpose(1, 2)


def _load_cosmos_worker(rank, rendezvous, checkpoint_dir, edge):
    from tests.model_executor.helpers import bootstrap_vllm_layer_custom_op_modules

    bootstrap_vllm_layer_custom_op_modules()

    from safetensors.torch import save_file
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.load import LoadConfig
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.distributed import hsdp
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3, transformer_cosmos3_edge
    from vllm_omni.platforms import current_omni_platform

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        with ExitStack() as stack:
            stack.enter_context(set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))))
            stack.enter_context(set_default_torch_dtype(torch.bfloat16))
            # HSDP spans both workers, while every vLLM linear uses TP=1.
            for module in (linear, parameter):
                stack.enter_context(patch.object(module, "get_tensor_model_parallel_rank", return_value=0))
                stack.enter_context(patch.object(module, "get_tensor_model_parallel_world_size", return_value=1))
            for module in (transformer_cosmos3, transformer_cosmos3_edge):
                stack.enter_context(patch.object(module, "get_tensor_model_parallel_world_size", return_value=1))
                stack.enter_context(patch.object(module, "FrameworkAttention", _SDPAAttention))
            stack.enter_context(patch.object(transformer_cosmos3, "_get_ulysses_state", return_value=(1, 0, None)))
            stack.enter_context(
                patch.object(hsdp, "get_world_group", return_value=SimpleNamespace(world_size=2, rank_in_group=rank))
            )

            model_cls = (
                transformer_cosmos3_edge.Cosmos3EdgeVFMTransformer
                if edge
                else transformer_cosmos3.Cosmos3VFMTransformer
            )
            tf_config = dict(
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=8,
                intermediate_size=32,
                vocab_size=32,
                latent_patch_size=2,
                latent_channel=48,
                temporal_compression_factor=4,
                rope_scaling={"mrope_section": [2, 1, 1]},
            )
            if edge:
                tf_config.update(
                    backbone_type=transformer_cosmos3_edge.COSMOS3_EDGE_BACKBONE_TYPE,
                    qk_norm_for_text=False,
                    use_und_k_norm_for_gen=True,
                )
            config = SimpleNamespace(
                tf_model_config=tf_config,
                dtype=torch.bfloat16,
                quantization_config=None,
                lora_path=None,
                num_weight_load_threads=1,
                hsdp_weight_load_strategy="full",
                parallel_config=SimpleNamespace(
                    use_hsdp=True, hsdp_replicate_size=1, hsdp_shard_size=2, tensor_parallel_size=1
                ),
            )

            class Pipeline(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.transformer = model_cls(config)
                    self.transformer.register_buffer("scratch", torch.tensor([7.0]), persistent=False)
                    self.vae = nn.Linear(2, 2)
                    self.weights_sources = [
                        DiffusersPipelineLoader.ComponentSource(checkpoint_dir, None, None, "transformer.", False)
                    ]

                def load_weights(self, weights):
                    state = dict(weights)
                    self.transformer.load_state_dict(
                        {name.removeprefix("transformer."): t for name, t in state.items()}
                    )
                    self.transformer.post_load_weights()
                    self.transformer.eval()
                    self.transformer.validate_loaded_weights(set(state))
                    return set(state)

            full = Pipeline()
            if rank == 0:
                torch.manual_seed(42)
                with torch.no_grad():
                    for name, tensor in full.transformer.named_parameters():
                        tensor.copy_(torch.randn_like(tensor) * 0.02)
                        if "norm" in name:
                            tensor.fill_(1)
                save_file(full.transformer.state_dict(), str(Path(checkpoint_dir) / "model.safetensors"))
            dist.barrier()

            def load(model):
                loader = DiffusersPipelineLoader(LoadConfig(), config)
                with patch.object(loader, "_init_from_load_format", return_value=model):
                    return loader._load_model_with_hsdp(device)

            full = load(full)
            config.hsdp_weight_load_strategy = "pre_sharded"
            pre_sharded = load(Pipeline())
            full_parameters = dict(full.transformer.named_parameters())
            from torch.distributed.tensor import DTensor

            for name, actual in pre_sharded.transformer.named_parameters():
                expected = full_parameters[name]
                if name.startswith("time_embedder."):
                    assert not isinstance(actual, DTensor)
                    assert actual.dtype == torch.float32
                else:
                    assert isinstance(actual, DTensor)
                    actual = actual.full_tensor()
                    expected = expected.full_tensor()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert pre_sharded.vae.weight.device == device
            torch.testing.assert_close(pre_sharded.transformer.scratch, full.transformer.scratch)
            assert all(t.device.type != "meta" for t in pre_sharded.transformer.buffers())

            inputs = dict(
                hidden_states=torch.linspace(-1, 1, 48 * 4, device=device).reshape(1, 48, 1, 2, 2),
                timestep=torch.tensor([1.0], device=device),
                text_ids=torch.tensor([[1, 2]], device=device),
                text_mask=torch.ones(1, 2, dtype=torch.long, device=device),
                video_shape=(1, 2, 2),
                fps=24.0,
            )
            with torch.no_grad():
                # Also exercise the cached UND path on the second denoising step.
                for step in (1.0, 0.5):
                    inputs["timestep"].fill_(step)
                    expected = full.transformer(**inputs)
                    actual = pre_sharded.transformer(**inputs)
                    assert torch.isfinite(actual).all()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("edge", [False, True], ids=["cosmos3", "cosmos3_edge"])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 2 or not dist.is_nccl_available(),
    reason="requires two CUDA GPUs and NCCL",
)
def test_cosmos3_pre_sharded_matches_full_hsdp(tmp_path, edge):
    torch.multiprocessing.spawn(
        _load_cosmos_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path), edge),
        nprocs=2,
        join=True,
    )
