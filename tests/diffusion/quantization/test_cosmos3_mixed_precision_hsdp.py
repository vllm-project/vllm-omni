# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise live MAPS weights across real production FSDP gather/reshard cycles."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


def _run_hsdp(rank, world_size, init_method, fmt, backend, parallel):
    from torch.distributed.tensor import DTensor
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ReplicatedLinear
    from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant_dequant

    from vllm_omni.diffusion.distributed.hsdp import HSDPInferenceConfig, apply_hsdp_to_model
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.diffusion.models.cosmos3.mixed_precision import (
        Cosmos3MixedPrecisionConfig,
        Cosmos3MixedPrecisionRuntime,
    )
    from vllm_omni.diffusion.quantization.hsdp_fp8 import prepare_fp8_layers_for_fsdp
    from vllm_omni.platforms import current_omni_platform

    current_omni_platform.set_device(torch.device(f"cuda:{rank}"))
    init_distributed_environment(world_size, rank, init_method, rank)
    replicas = 2 if parallel == "replicated" else 1
    shard_size = world_size // replicas
    config = VllmConfig()
    config.kernel_config.linear_backend = backend
    try:
        initialize_model_parallel(
            data_parallel_size=1,
            fully_shard_degree=shard_size,
            use_hsdp=True,
            ulysses_degree=world_size if parallel == "sp" else 1,
            cfg_parallel_size=2 if parallel == "cfg" else 1,
        )
        with set_current_vllm_config(config), torch.device(f"cuda:{rank}"), torch.no_grad():
            config.model_config = SimpleNamespace(dtype=torch.bfloat16)
            names = ["gen_layers.0.linear", "language_model.layers.0.linear"]
            quant = ModelOptMixedPrecisionConfig.from_config(
                {
                    "quant_method": "modelopt",
                    "quant_algo": "MIXED_PRECISION",
                    "group_size": 16,
                    "exclude_modules": [],
                    "quantized_layers": {name: {"quant_algo": fmt} for name in names},
                }
            )
            k, n = 128, 256

            class Block(torch.nn.Module):
                def __init__(self, prefix):
                    super().__init__()
                    self.linear = ReplicatedLinear(
                        k, n, bias=True, params_dtype=torch.bfloat16, quant_config=quant, prefix=prefix
                    )
                    self.check_gather = False

                def forward(self, x):
                    if self.check_gather:
                        for name, expected in self.expected_params.items():
                            actual = getattr(self.linear, name)
                            assert not isinstance(actual, DTensor), name
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(
                            self.linear.quant_method.strategy.materialize(self.linear),
                            self.dense,
                            rtol=0,
                            atol=0,
                        )
                    return self.linear(x)[0]

            class Model(torch.nn.Module):
                _hsdp_shard_conditions = [lambda name, module: isinstance(module, Block)]

                def __init__(self):
                    super().__init__()
                    self.gen_layers = torch.nn.ModuleList([Block(names[0])])
                    self.language_model = torch.nn.Module()
                    self.language_model.layers = torch.nn.ModuleList([Block(names[1])])

                def forward(self, x):
                    return tuple(block(x) for block in (*self.gen_layers, *self.language_model.layers))

            model = Model()
            runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig())
            runtime.install(model)
            blocks = list(model.gen_layers) + list(model.language_model.layers)
            references = []
            native_calls = [0] * len(blocks)
            torch.manual_seed(42)
            x = (torch.randint(-8, 9, (2, 3, k)).float() / 16).bfloat16()
            # Different tokens per SP rank and unequal branch lengths for CFG.
            # Weight generation remains identical on every rank.
            if parallel in ("sp", "replicated"):
                x = x + rank / 16
            elif parallel == "cfg":
                x = x[:, : 3 - rank].contiguous()
            bias = (torch.arange(n).float() / n / 4).bfloat16()
            for index, block in enumerate(blocks):
                layer = block.linear
                if fmt == "FP8":
                    weight = torch.randint(-8, 9, (n, k)).to(torch.float8_e4m3fn)
                    scale = torch.tensor([1 / 64], dtype=torch.float32)
                    values = dict(weight=weight, weight_scale=scale, input_scale=torch.tensor([1 / 16]))
                    dense = (weight.float() * scale).bfloat16()
                    native_x = x
                else:
                    codes = (torch.arange(n * k).reshape(n, k) // 3 % 16).to(torch.uint8)
                    scales = (torch.arange(n * k // 16).reshape(n, k // 16) % 7 + 1).to(torch.float8_e4m3fn)
                    global_scale = torch.tensor([1 / 32], dtype=torch.float32)
                    input_scale = torch.tensor([1 / 512], dtype=torch.float32)
                    values = dict(
                        weight=codes[:, 0::2] | (codes[:, 1::2] << 4),
                        weight_scale=scales,
                        weight_scale_2=global_scale,
                        input_scale=input_scale,
                    )
                    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
                    dense = (lut[codes.long()] * scales.float().repeat_interleave(16, dim=1) * global_scale).bfloat16()
                    native_x = ref_nvfp4_quant_dequant(x.reshape(-1, k), input_scale.reciprocal(), 16).reshape_as(x)
                for name, value in dict(values, bias=bias).items():
                    param = getattr(layer, name)
                    param.weight_loader(param, value)
                layer.quant_method.process_weights_after_loading(layer)
                block.dense = dense
                high_ref = F.linear(x.reshape(-1, k), dense, bias).reshape(*x.shape[:-1], n)
                native_ref = F.linear(native_x, dense) + bias if fmt == "NVFP4" else high_ref
                references.append((high_ref, native_ref))
                original = layer.quant_method.base_method.apply

                def counted_apply(layer, x, bias=None, *, original=original, index=index):
                    native_calls[index] += 1
                    return original(layer, x, bias)

                layer.quant_method.base_method.apply = counted_apply

            # Same loaded parameters, inputs, grad context and kernels before/after FSDP.
            baselines = []
            for step in range(10):
                runtime.set_step(step, 10)
                baselines.append(tuple(out.clone() for out in model(x)))
            runtime.reset()
            assert prepare_fp8_layers_for_fsdp(model) == (2 if fmt == "FP8" else 0)
            for block in blocks:
                block.expected_params = {name: p.detach().clone() for name, p in block.linear.named_parameters()}
                block.check_gather = True
            apply_hsdp_to_model(
                model,
                HSDPInferenceConfig(enabled=True, hsdp_shard_size=shard_size, hsdp_replicate_size=replicas),
                target_device=torch.device(f"cuda:{rank}"),
            )
            for block in blocks:
                assert isinstance(block.linear.weight, DTensor) == (fmt == "FP8")
                sharded = block.linear.weight if fmt == "FP8" else block.linear.weight_scale
                assert isinstance(sharded, DTensor)
                assert tuple(sharded.device_mesh.shape) == (replicas, shard_size)
                assert sharded.placements[0].is_replicate()
                assert sharded.placements[1].is_shard(0)
                assert sharded.to_local().shape[0] == sharded.shape[0] // shard_size
            for first, last in ((3, 3), (0, 0), (6, 6)):
                policy = Cosmos3MixedPrecisionConfig(first_steps=first, last_steps=last)
                runtime.config = policy
                for request in range(2):
                    before = native_calls.copy()
                    for step in range(10):
                        runtime.set_step(step, 10)
                        outputs = model(x)
                        for index, actual in enumerate(outputs):
                            high = index == 1 or policy.use_high_precision(step, 10)
                            expected = references[index][0 if high else 1]
                            rtol, atol = (0.02, 0.003) if fmt == "FP8" else (0.025, 0.025)
                            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                            baseline = baselines[0 if high else 3][index]
                            torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
                    expected_native = sum(not policy.use_high_precision(step, 10) for step in range(10))
                    assert native_calls == [before[0] + expected_native, before[1]], (rank, request, first, last)
                    runtime.reset()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    "fmt,backend,parallel,world_size",
    [
        pytest.param(
            fmt,
            backend,
            parallel,
            world_size,
            id=f"{fmt}-{backend}-{parallel}-{world_size}",
            marks=hardware_marks(
                res={"cuda": "B200" if backend == "flashinfer_cutedsl" else ["H100", "B200"]},
                num_cards=world_size,
            ),
        )
        for fmt, backend in (("FP8", "auto"), ("NVFP4", "emulation"), ("NVFP4", "flashinfer_cutedsl"))
        for parallel, world_size in (
            ("hsdp", 1),
            ("hsdp", 2),
            ("hsdp", 4),
            ("sp", 2),
            ("sp", 4),
            ("cfg", 2),
            ("replicated", 4),
        )
    ],
)
def test_modelopt_maps_hsdp(tmp_path, parallel, world_size, fmt, backend):
    if not torch.cuda.is_available() or torch.accelerator.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    capabilities = [torch.cuda.get_device_capability(i) for i in range(world_size)]
    if fmt == "FP8" and any(capability < (8, 9) for capability in capabilities):
        pytest.skip("Requires native FP8 support")
    if fmt == "NVFP4" and backend != "emulation" and any(capability[0] != 10 for capability in capabilities):
        pytest.skip("Native NVFP4 requires SM10x")
    torch.multiprocessing.spawn(
        _run_hsdp,
        args=(world_size, f"file://{tmp_path / 'rendezvous'}", fmt, backend, parallel),
        nprocs=world_size,
        join=True,
        daemon=True,
    )
