# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare full and pre-sharded HSDP loading of a reduced MiniMax-H3 DiT end to end."""

from __future__ import annotations

import gc
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.tensor import DTensor
from vllm.utils.network_utils import get_file_store_init_method

from tests.diffusion.distributed.minimax_h3_pre_sharded_helpers import (
    NUM_LAYERS,
    NVFP4_INPUT_SCALED,
    add_qad_sidecar_source,
    build_pipeline,
    nvfp4_arch,
    patched_h3,
    write_checkpoint,
    write_qad_checkpoint,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu, pytest.mark.core_model]


@pytest.fixture(scope="module")
def cpu_process_group():
    if dist.is_initialized():
        yield
        return
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=get_file_store_init_method())
    try:
        yield
    finally:
        dist.destroy_process_group()
        gc.collect()


def _random_pack(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Arbitrary packed bytes; loading never interprets them. The global scale is not BF16-exact."""
    rows, columns = weight.shape
    qweight = torch.randint(0, 256, (rows, columns // 2), dtype=torch.uint8)
    scales = torch.randint(0, 256, (rows, columns // 16), dtype=torch.uint8)
    return qweight, scales, 448 * 6 / 1.2345


@pytest.mark.parametrize("strategy", ["full", "pre_sharded"])
def test_minimax_h3_nvfp4_scales_stay_fp32_through_hsdp_gather(cpu_process_group, tmp_path, strategy):
    """FlashInfer's mm_fp4 needs an FP32 alpha; a BF16 HSDP gather would also round the scales."""
    from vllm.config.load import LoadConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.models.minimax_h3.checkpoint_quantization import FastH3NVFP4Config, _Projection

    checkpoint, encoder_dir = tmp_path / "transformer", tmp_path / "text_encoder"
    with ExitStack() as stack:
        patched_h3(stack)
        nvfp4_arch(stack)
        stack.enter_context(set_default_torch_dtype(torch.bfloat16))
        reference, _ = build_pipeline(True, tmp_path, tmp_path, "full")
        checkpoint.mkdir()
        encoder_dir.mkdir()
        torch.manual_seed(0)
        write_qad_checkpoint(checkpoint, reference.transformer, _random_pack)
        save_file({"proj.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, str(encoder_dir / "model.safetensors"))
        sidecar = load_file(str(checkpoint / "nvfp4_weights.safetensors"))

        quant = FastH3NVFP4Config.from_dit(checkpoint / "nvfp4_weights.safetensors", rank=0)
        pipeline, config = build_pipeline(True, checkpoint, encoder_dir, strategy, quant_config=quant)
        add_qad_sidecar_source(pipeline, checkpoint)
        loader = DiffusersPipelineLoader(LoadConfig(), config)
        with patch.object(loader, "_init_from_load_format", return_value=pipeline):
            transformer = loader._load_model_with_hsdp(torch.device("cpu")).transformer

    checked = 0
    for block in transformer.blocks:
        block.unshard()
        try:
            for projection in block.modules():
                if not isinstance(projection, _Projection):
                    continue
                source = projection.spec.source
                expected = {"alpha": sidecar[f"{source}::_nvfp4_alpha"]}
                if f"{source}::_nvfp4_input_global_sf" in sidecar:
                    expected["input_scale"] = sidecar[f"{source}::_nvfp4_input_global_sf"]
                    checked += 1
                for field, value in expected.items():
                    gathered = getattr(projection, field)
                    assert gathered.dtype == torch.float32, (source, field, gathered.dtype)
                    torch.testing.assert_close(gathered, value.reshape(1), rtol=0, atol=0, msg=f"{source} {field}")
                assert (projection.alpha / projection.input_scale).dtype == torch.float32, source
        finally:
            block.reshard()
    assert checked == NUM_LAYERS * len(NVFP4_INPUT_SCALED), checked


@pytest.mark.parametrize("diffusers", [False, True], ids=["native", "diffusers"])
def test_minimax_h3_pre_sharded_matches_full_hsdp(cpu_process_group, tmp_path, diffusers):
    from vllm.config.load import LoadConfig
    from vllm.utils.torch_utils import set_default_torch_dtype

    with ExitStack() as stack:
        patched_h3(stack)
        stack.enter_context(set_default_torch_dtype(torch.bfloat16))
        reference, _ = build_pipeline(diffusers, tmp_path, tmp_path, "full")
        torch.manual_seed(0)
        with torch.no_grad():
            for name, tensor in reference.transformer.named_parameters():
                tensor.copy_(torch.randn_like(tensor))
        checkpoint = tmp_path / "transformer"
        encoder_dir = tmp_path / "text_encoder"
        checkpoint.mkdir()
        encoder_dir.mkdir()
        write_checkpoint(reference.transformer, checkpoint, diffusers=diffusers)
        encoder_weight = torch.arange(4, dtype=torch.bfloat16).reshape(2, 2)
        save_file({"proj.weight": encoder_weight}, str(encoder_dir / "model.safetensors"))

        results = {}
        for strategy in ("full", "pre_sharded"):
            pipeline, config = build_pipeline(diffusers, checkpoint, encoder_dir, strategy)
            loader = DiffusersPipelineLoader(LoadConfig(), config)
            with patch.object(loader, "_init_from_load_format", return_value=pipeline):
                results[strategy] = loader._load_model_with_hsdp(torch.device("cpu"))

    full, pre_sharded = results["full"].transformer, results["pre_sharded"].transformer
    expected = dict(full.named_parameters())
    for name, actual in pre_sharded.named_parameters():
        want = expected[name]
        actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
        want = want.full_tensor() if isinstance(want, DTensor) else want
        assert actual.dtype == want.dtype, name
        torch.testing.assert_close(actual, want, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(pre_sharded.rope.inv_freq, full.rope.inv_freq, rtol=0, atol=0)
    # The loaded weights must equal the reference model that wrote the checkpoint.
    reference_state = dict(reference.transformer.named_parameters())
    for name, actual in pre_sharded.named_parameters():
        actual = actual.full_tensor() if isinstance(actual, DTensor) else actual
        torch.testing.assert_close(actual, reference_state[name].to(actual.dtype), rtol=0, atol=0, msg=name)
    # The non-DiT source is loaded through the ordinary path in both strategies.
    for strategy in ("full", "pre_sharded"):
        weight = results[strategy].text_encoder.proj.weight
        torch.testing.assert_close(weight, encoder_weight.to(weight.dtype), rtol=0, atol=0)
