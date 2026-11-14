# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online-FP8 MiniMax-H3: full and pre-sharded HSDP loading must agree on two GPUs."""

from __future__ import annotations

from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.core_model]


def _worker(rank: int, rendezvous: str, root: str) -> None:
    from safetensors.torch import save_file
    from torch.distributed.tensor import DTensor
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.load import LoadConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    from tests.diffusion.distributed import test_minimax_h3_pre_sharded as reduced
    from vllm_omni.diffusion.distributed import hsdp
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.platforms import current_omni_platform
    from vllm_omni.quantization.fp8_config import DiffusionFp8Config

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    root_dir = Path(root)
    try:
        with ExitStack() as stack:
            reduced._patched_h3(stack)
            vllm_config = VllmConfig(device_config=DeviceConfig(device="cuda"))
            vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
            stack.enter_context(set_current_vllm_config(vllm_config))
            stack.enter_context(set_default_torch_dtype(torch.bfloat16))
            stack.enter_context(
                patch.object(hsdp, "get_world_group", return_value=SimpleNamespace(world_size=2, rank_in_group=rank))
            )
            stack.enter_context(patch.object(hsdp, "current_omni_platform", current_omni_platform))

            checkpoint, encoder_dir = root_dir / "transformer", root_dir / "text_encoder"
            if rank == 0:
                reference, _ = reduced._pipeline(False, root_dir, root_dir, "full")
                torch.manual_seed(0)
                with torch.no_grad():
                    for tensor in reference.transformer.parameters():
                        tensor.copy_(torch.randn_like(tensor))
                checkpoint.mkdir()
                encoder_dir.mkdir()
                reduced._write_checkpoint(reference.transformer, checkpoint, diffusers=False)
                save_file(
                    {"proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(encoder_dir / "model.safetensors")
                )
            dist.barrier()

            results = {}
            for strategy in ("full", "pre_sharded"):
                quant = DiffusionFp8Config()
                pipeline, config = reduced._pipeline(
                    False, checkpoint, encoder_dir, strategy, quant_config=quant, device=device
                )
                config.parallel_config.hsdp_shard_size = 2
                loader = DiffusersPipelineLoader(LoadConfig(), config)
                with patch.object(loader, "_init_from_load_format", return_value=pipeline):
                    results[strategy] = loader._load_model_with_hsdp(device).transformer

            full = dict(results["full"].named_parameters())
            fp8_params = 0
            for name, actual in results["pre_sharded"].named_parameters():
                expected = full[name]
                actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
                expected = expected.full_tensor() if isinstance(expected, DTensor) else expected
                assert actual.dtype == expected.dtype, name
                fp8_params += actual.dtype == torch.float8_e4m3fn
                torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0, msg=name)
            assert fp8_params > 0, "online FP8 did not quantize any DiT linear"
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.accelerator.device_count() < 2 or not dist.is_nccl_available(),
    reason="requires two CUDA GPUs and NCCL",
)
def test_minimax_h3_online_fp8_pre_sharded_matches_full_hsdp(tmp_path):
    torch.multiprocessing.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=True,
    )
