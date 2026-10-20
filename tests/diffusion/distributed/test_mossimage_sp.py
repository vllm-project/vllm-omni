# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare real MOSS-Image Ulysses execution against single-GPU inference."""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.diffusion.test_mossimage_rope import make_inputs, make_transformer
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelConfig
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.hooks.sequence_parallel import apply_sequence_parallel
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.parallel]


def _sp_worker(rank: int, world_size: int, init_method: str, output_dir: str):
    device = torch.device(f"cuda:{rank}")
    current_omni_platform.set_device(device)
    torch.set_num_threads(1)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, distributed_init_method=init_method)
    try:
        initialize_model_parallel(ulysses_degree=world_size)
        outputs = {}
        with torch.inference_mode():
            for dtype in (torch.float32, torch.bfloat16):
                config = OmniDiffusionConfig(
                    dtype=dtype,
                    parallel_config=DiffusionParallelConfig(ulysses_degree=world_size),
                    diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
                )
                transformer = make_transformer(config, device, dtype)
                if world_size > 1:
                    apply_sequence_parallel(
                        transformer, SequenceParallelConfig(ulysses_degree=world_size), transformer._sp_plan
                    )
                for lengths in ((3,), (3, 5)):
                    latents, text, mask = make_inputs(lengths)
                    # Four image tokens + max(lengths) is odd before SP padding.
                    assert (4 + max(lengths)) % 2 == 1
                    with set_forward_context(omni_diffusion_config=config):
                        get_forward_context().sp_plan_hooks_applied = world_size > 1
                        result = transformer(
                            latents.to(device=device, dtype=dtype),
                            500.0,
                            text.to(device=device, dtype=dtype),
                            mask.to(device),
                        ).sample
                    key = f"{dtype}_{len(lengths)}"
                    outputs[key] = result.cpu().contiguous()
                del transformer
        save_file(outputs, str(Path(output_dir) / f"sp{world_size}_rank{rank}.safetensors"))
    finally:
        destroy_distributed_env()


@hardware_test(res={"cuda": ["L4", "B200"]}, num_cards=2)
def test_mossimage_sp_matches_single_gpu_with_odd_joint_length(tmp_path):
    for world_size in (1, 2):
        torch.multiprocessing.spawn(
            _sp_worker,
            args=(world_size, (tmp_path / f"init_sp{world_size}").as_uri(), str(tmp_path)),
            nprocs=world_size,
        )
    expected = load_file(str(tmp_path / "sp1_rank0.safetensors"))
    rank_outputs = [load_file(str(tmp_path / f"sp2_rank{rank}.safetensors")) for rank in range(2)]
    for name, reference in expected.items():
        tolerance = 2e-5 if reference.dtype == torch.float32 else 2e-2
        for actual in rank_outputs:
            torch.testing.assert_close(actual[name], reference, atol=tolerance, rtol=tolerance, msg=name)
        torch.testing.assert_close(rank_outputs[0][name], rank_outputs[1][name], atol=0, rtol=0)
