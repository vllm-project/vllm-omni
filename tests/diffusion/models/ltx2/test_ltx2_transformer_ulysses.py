# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-transformer numeric parity for LTX Ulysses sequence parallelism.

This FP32/SDPA check verifies that Ulysses preserves the complete transformer
mathematical path. It does not guarantee pixel-level E2E parity: production
BF16 and compiled attention kernels can change rounding order, and diffusion
can amplify that drift across denoising steps.
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel]

_FP32_REL_L2 = 1e-5
_TINY_CONFIG = {
    "in_channels": 4,
    "out_channels": 4,
    "num_attention_heads": 2,
    "attention_head_dim": 4,
    "cross_attention_dim": 8,
    "vae_scale_factors": (1, 1, 1),
    "pos_embed_max_pos": 4,
    "base_height": 2,
    "base_width": 2,
    "audio_in_channels": 4,
    "audio_out_channels": 4,
    "audio_num_attention_heads": 2,
    "audio_attention_head_dim": 4,
    "audio_cross_attention_dim": 8,
    "audio_scale_factor": 1,
    "audio_pos_embed_max_pos": 4,
    "audio_sampling_rate": 16,
    "audio_hop_length": 4,
    "num_layers": 1,
    "caption_channels": 8,
    "rope_type": "split",
    "use_prompt_embeddings": False,
    "use_keyframes_abs_pos_embedding": True,
}


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).double().norm() / expected.double().norm()).item()


def _omni_config(sp_size: int) -> OmniDiffusionConfig:
    return OmniDiffusionConfig.from_kwargs(
        model="test",
        dtype=torch.float32,
        parallel_config=DiffusionParallelConfig(
            pipeline_parallel_size=1,
            data_parallel_size=1,
            tensor_parallel_size=1,
            sequence_parallel_size=sp_size,
            ulysses_degree=sp_size,
            ring_degree=1,
            cfg_parallel_size=1,
        ),
        diffusion_attention_backend="TORCH_SDPA",
    )


def _inputs(device: torch.device) -> dict[str, torch.Tensor | int | bool]:
    generator = torch.Generator().manual_seed(123)
    return {
        "hidden_states": torch.randn(1, 8, 4, generator=generator, device="cpu").to(device),
        "audio_hidden_states": torch.randn(1, 4, 4, generator=generator, device="cpu").to(device),
        "encoder_hidden_states": torch.randn(1, 3, 8, generator=generator, device="cpu").to(device),
        "audio_encoder_hidden_states": torch.randn(1, 3, 8, generator=generator, device="cpu").to(device),
        "timestep": torch.linspace(0.1, 0.8, 8, device=device).view(1, 8),
        "audio_timestep": torch.linspace(0.2, 0.5, 4, device=device).view(1, 4),
        "keyframes_mask": torch.tensor([[[1.0], [0.0], [1.0], [0.0], [0.0], [1.0], [0.0], [0.0]]], device=device),
        "sigma": torch.tensor([0.1], device=device),
        "audio_sigma": torch.tensor([0.2], device=device),
        "num_frames": 2,
        "height": 2,
        "width": 2,
        "audio_num_frames": 4,
        "return_dict": False,
    }


def _run_transformer_parity(rank: int, world_size: int, master_port: int) -> None:
    device = torch.device(f"{current_omni_platform.device_type}:{rank}")
    current_omni_platform.set_device(device)
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )

    try:
        init_distributed_environment()
        initialize_model_parallel(sequence_parallel_size=world_size, ulysses_degree=world_size)

        sp_config = _omni_config(world_size)
        torch.manual_seed(42)
        with set_forward_context(omni_diffusion_config=sp_config), set_current_diffusion_config(sp_config):
            model = LTX2VideoTransformer3DModel(**_TINY_CONFIG).to(device).eval()
        with torch.no_grad():
            for _, parameter in sorted(model.named_parameters()):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)

        inputs = _inputs(device)
        reference_config = _omni_config(1)
        with torch.no_grad(), set_forward_context(omni_diffusion_config=reference_config):
            expected_video, expected_audio = model(**inputs)

        apply_sequence_parallel(
            model,
            SequenceParallelConfig(ulysses_degree=world_size, ring_degree=1),
            model._sp_plan,
        )
        with torch.no_grad(), set_forward_context(omni_diffusion_config=sp_config):
            get_forward_context().sp_plan_hooks_applied = True
            actual_video, actual_audio = model(**inputs)

        v2a_attention = model.transformer_blocks[0].video_to_audio_attn.attn
        assert v2a_attention._video_to_audio_strategy is not None
        assert torch.isfinite(expected_video).all()
        assert torch.isfinite(expected_audio).all()
        for modality, actual, expected in (
            ("video", actual_video, expected_video),
            ("audio", actual_audio, expected_audio),
        ):
            drift = _relative_l2(actual, expected)
            assert drift <= _FP32_REL_L2, (
                f"LTX SP{world_size} {modality} output drifted rel_l2={drift:.3e} from SP1 (bound {_FP32_REL_L2:.1e})"
            )
    finally:
        destroy_distributed_env()


@pytest.mark.full_model
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=2)
def test_ltx_ulysses_transformer_matches_sp1(unused_tcp_port) -> None:
    """Exercise real plan hooks, RoPE/timestep sharding, and attention routing."""
    world_size = 2
    device_count = current_omni_platform.device_count()
    assert device_count >= world_size, (
        f"LTX Ulysses parity requires {world_size} accelerator devices; only {device_count} are visible"
    )
    torch.multiprocessing.spawn(
        _run_transformer_parity,
        args=(world_size, unused_tcp_port),
        nprocs=world_size,
    )
