# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastH3 QAD layout: a packed NVFP4 sidecar next to a BF16 index, full vs pre-sharded HSDP."""

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


def _flashinfer_pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    import flashinfer

    global_scale = 448 * 6 / weight.float().abs().amax().item()
    qweight, scales = flashinfer.nvfp4_quantize(
        weight.to("cuda", torch.bfloat16),
        torch.tensor([global_scale], device="cuda"),
        sfLayout=flashinfer.SfLayout.layout_128x4,
        enable_pdl=False,
    )
    return qweight.cpu(), scales.view(torch.uint8).reshape(weight.shape[0], -1).cpu(), global_scale


def _worker(rank: int, rendezvous: str, root: str) -> None:
    from safetensors.torch import load_file, save_file
    from torch.distributed.tensor import DTensor
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.load import LoadConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    from tests.diffusion.distributed import test_minimax_h3_pre_sharded as reduced
    from vllm_omni.diffusion.distributed import hsdp
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
    from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config, _Projection
    from vllm_omni.platforms import current_omni_platform

    device = torch.device("cuda", rank)
    current_omni_platform.set_device(device)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    root_dir = Path(root)
    checkpoint, encoder_dir = root_dir / "transformer", root_dir / "text_encoder"
    try:
        with ExitStack() as stack:
            reduced._patched_h3(stack)
            reduced.nvfp4_arch(stack)
            vllm_config = VllmConfig(device_config=DeviceConfig(device="cuda"))
            vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
            stack.enter_context(set_current_vllm_config(vllm_config))
            stack.enter_context(set_default_torch_dtype(torch.bfloat16))
            stack.enter_context(
                patch.object(hsdp, "get_world_group", return_value=SimpleNamespace(world_size=2, rank_in_group=rank))
            )
            stack.enter_context(patch.object(hsdp, "current_omni_platform", current_omni_platform))

            if rank == 0:
                reference, _ = reduced._pipeline(True, root_dir, root_dir, "full")
                torch.manual_seed(0)
                with torch.no_grad():
                    for tensor in reference.transformer.parameters():
                        tensor.copy_(torch.randn_like(tensor) * 0.05)
                checkpoint.mkdir()
                encoder_dir.mkdir()
                reduced.write_qad_checkpoint(checkpoint, reference.transformer, _flashinfer_pack)
                save_file(
                    {"proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(encoder_dir / "model.safetensors")
                )
            dist.barrier()
            sidecar = load_file(str(checkpoint / "nvfp4_weights.safetensors"))

            results = {}
            for strategy in ("full", "pre_sharded"):
                quant = FastH3NVFP4Config.from_dit(checkpoint / "nvfp4_weights.safetensors", rank=0)
                pipeline, config = reduced._pipeline(
                    True, checkpoint, encoder_dir, strategy, quant_config=quant, device=device
                )
                reduced.add_qad_sidecar_source(pipeline, checkpoint)
                config.parallel_config.hsdp_shard_size = 2
                loader = DiffusersPipelineLoader(LoadConfig(), config)
                with patch.object(loader, "_init_from_load_format", return_value=pipeline):
                    results[strategy] = loader._load_model_with_hsdp(device).transformer

            full = dict(results["full"].named_parameters())
            packed_params = 0
            for name, actual in results["pre_sharded"].named_parameters():
                expected = full[name]
                actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
                expected = expected.full_tensor() if isinstance(expected, DTensor) else expected
                assert actual.dtype == expected.dtype, name
                packed_params += ".packed." in name
                torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0, msg=name)
            assert packed_params == len(sidecar), (packed_params, len(sidecar))

            # Run every packed projection through FlashInfer inside the HSDP
            # gather, against the same projection loaded without HSDP.
            generator = torch.Generator(device=device).manual_seed(1)
            for strategy, transformer in results.items():
                checked = 0
                for block in transformer.blocks:
                    block.unshard()
                    try:
                        for projection in block.modules():
                            if not isinstance(projection, _Projection):
                                continue
                            spec = projection.spec
                            with torch.device(device):
                                standalone = _Projection(spec, projection.rows, projection.columns, rank=0)
                            for field, key in spec.tensors.items():
                                standalone.weight_loader(getattr(standalone, field), sidecar[key].to(device))
                            # Small inputs keep the 428x input scale inside the E4M3 block-scale range.
                            x = 0.25 * torch.randn(
                                128, projection.columns, generator=generator, device=device, dtype=torch.bfloat16
                            )
                            actual = projection(x)
                            assert torch.isfinite(actual).all(), (strategy, spec.source)
                            torch.testing.assert_close(
                                actual, standalone(x), rtol=0, atol=0, msg=f"{strategy} {spec.source}"
                            )
                            checked += 1
                    finally:
                        block.reshard()
                assert checked == reduced._NUM_LAYERS * len(reduced.NVFP4_PROJECTIONS), (strategy, checked)
    finally:
        dist.destroy_process_group()


@hardware_test(res={"cuda": "RTX5090"}, num_cards=2)
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.accelerator.device_count() < 2
    or torch.cuda.get_device_capability()[0] not in (10, 12)
    or not dist.is_nccl_available(),
    reason="requires two Blackwell CUDA GPUs and NCCL",
)
def test_minimax_h3_qad_sidecar_pre_sharded_matches_full_hsdp(tmp_path):
    pytest.importorskip("flashinfer")
    torch.multiprocessing.spawn(
        _worker,
        args=(f"file://{tmp_path / 'rendezvous'}", str(tmp_path)),
        nprocs=2,
        join=True,
    )
