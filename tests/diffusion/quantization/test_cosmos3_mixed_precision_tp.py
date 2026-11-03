# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""FP8/NVFP4 MAPS loading, dispatch, and real diffusion TP/SP/CFG collectives.

Independent references expose wrong TP membership, SP token order, and CFG
branch communication. TP replicas share a checkpoint; SP/CFG inputs differ.
"""

from dataclasses import dataclass
from math import prod
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


TOPOLOGIES = [
    pytest.param((1, 1, 1), id="baseline"),
    pytest.param((2, 1, 1), id="tp2"),
    pytest.param((4, 1, 1), id="tp4"),
    pytest.param((1, 2, 1), id="sp2"),
    pytest.param((1, 4, 1), id="sp4"),
    pytest.param((1, 1, 2), id="cfg2"),
    pytest.param((2, 2, 1), id="tp2-sp2"),
    pytest.param((2, 1, 2), id="tp2-cfg2"),
    pytest.param((1, 2, 2), id="sp2-cfg2"),
]


def initialize_parallel(rank, init_method, topology):
    from vllm.distributed import get_tp_group

    from vllm_omni.diffusion.distributed.parallel_state import (
        get_cfg_group,
        get_sp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm_omni.platforms import current_omni_platform

    tp, sp, cfg = topology
    current_omni_platform.set_device(torch.device(f"cuda:{rank}"))
    init_distributed_environment(tp * sp * cfg, rank, init_method, rank)
    initialize_model_parallel(
        data_parallel_size=1,
        tensor_parallel_size=tp,
        ulysses_degree=sp,
        cfg_parallel_size=cfg,
        use_hsdp=False,
    )
    tp_rank, sp_rank, cfg_rank = rank % tp, rank // tp % sp, rank // (tp * sp)
    # Hand-derived tp-sp-cfg layout, independent of production RankGenerator.
    assert get_tp_group().ranks == [cfg_rank * tp * sp + sp_rank * tp + i for i in range(tp)]
    assert get_sp_group().ranks == [cfg_rank * tp * sp + i * tp + tp_rank for i in range(sp)]
    assert get_cfg_group().ranks == [i * tp * sp + sp_rank * tp + tp_rank for i in range(cfg)]
    print(
        f"MAPS tensor rank={rank} tp={get_tp_group().ranks} sp={get_sp_group().ranks} cfg={get_cfg_group().ranks}",
        flush=True,
    )
    return tp_rank, sp_rank, cfg_rank


def parallel_inputs(k, sp, cfg, sp_rank, cfg_rank, fmt, *, divisor=1):
    generator = torch.Generator(device="cuda").manual_seed(123)
    if fmt == "FP8":
        base = (torch.randint(-8, 9, (2, 3 * sp, k), generator=generator).float() / 16).bfloat16()
    else:
        base = torch.linspace(-3.1, 2.7, 6 * sp * k).reshape(2, 3 * sp, k).bfloat16()
    base = base / divisor
    # Unequal CFG branches share shape; SP ranks see different sequence tokens.
    branches = [base if i == 0 else -base.roll(7, dims=-1) for i in range(cfg)]
    return branches, branches[cfg_rank].chunk(sp, dim=1)[sp_rank].contiguous()


def assemble_output(output, sp, cfg, cfg_rank):
    from vllm_omni.diffusion.distributed.parallel_state import get_cfg_group, get_sp_group

    if sp > 1:
        output = get_sp_group().all_gather(output, dim=1)
    if cfg > 1:
        # Guidance-six branch combination over the actual CFG group.
        output = get_cfg_group().all_reduce(output * (6 if cfg_rank == 0 else -5))
    return output


def assert_parallel_output(actual, references, cfg, *, rtol, atol):
    expected = references[0] if cfg == 1 else references[0] * 6 - references[1] * 5
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@dataclass
class _LinearCase:
    checkpoint: dict[str, torch.Tensor]
    dense_weight: torch.Tensor
    inputs: torch.Tensor
    high_reference: torch.Tensor
    native_reference: torch.Tensor
    high_references: list[torch.Tensor]
    native_references: list[torch.Tensor]
    rtol: float
    atol: float


def _fp8_case(kind, k, n, topology, sp_rank, cfg_rank):
    tp, sp, cfg = topology
    # Exactly representable values isolate sharding/reduction errors
    # from activation quantization noise. Each rank receives the same
    # full checkpoint and must load a different weight partition.
    torch.manual_seed(42)
    weight = torch.randint(-8, 9, (n, k)).to(torch.float8_e4m3fn)
    weight_scale = torch.tensor(1 / 64, dtype=torch.float32)
    input_scale = torch.tensor(1 / 16, dtype=torch.float32)
    bias = (torch.arange(n, dtype=torch.float32) / (n * 4)).to(torch.bfloat16)
    branches, x = parallel_inputs(k, sp, cfg, sp_rank, cfg_rank, "FP8")
    dense_weight = (weight.float() * weight_scale).to(torch.bfloat16)

    def dense_reference(activations):
        if kind == "column":
            return F.linear(activations, dense_weight, bias)
        # BF16 partial GEMMs round before TP reduction. Derive them
        # independently; a full GEMM differs after CFG amplification.
        inputs = activations.chunk(tp, dim=-1)
        weights = dense_weight.chunk(tp, dim=1)
        partials = [
            F.linear(part_x.contiguous(), part_weight.contiguous(), bias if i == 0 else None)
            for i, (part_x, part_weight) in enumerate(zip(inputs, weights))
        ]
        return sum(partials[1:], partials[0])

    reference = dense_reference(x)
    references = [dense_reference(branch) for branch in branches]
    return _LinearCase(
        checkpoint={"weight": weight, "weight_scale": weight_scale, "input_scale": input_scale, "bias": bias},
        dense_weight=dense_weight,
        inputs=x,
        high_reference=reference,
        native_reference=reference,
        high_references=references,
        native_references=references,
        rtol=0.02,
        atol=0.003,
    )


def _nvfp4_case(kind, k, n, topology, sp_rank, cfg_rank):
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import ref_nvfp4_quant_dequant

    tp, sp, cfg = topology
    # Different rank partitions and nonuniform block scales expose
    # packed-input offsets, scale sharding and layout mistakes.
    codes = (torch.arange(n * k).reshape(n, k) // 3 % 16).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = (torch.arange(n * k // 16).reshape(n, k // 16) % 7 + 1).to(torch.float8_e4m3fn)
    weight_global = torch.tensor([1 / 32], dtype=torch.float32)
    input_global = torch.tensor([1 / 512], dtype=torch.float32)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    dense = (lut[codes.long()] * scales.float().repeat_interleave(16, dim=1) * weight_global).bfloat16()
    # TP4 uses bounded activations: every one of 120 BF16 summation
    # orders fits the same tolerance, independent of NCCL topology.
    branches, x = parallel_inputs(k, sp, cfg, sp_rank, cfg_rank, "NVFP4", divisor=16 if tp == 4 else 1)
    quantized_x = ref_nvfp4_quant_dequant(x.reshape(-1, k), input_global.reciprocal(), 16).reshape_as(x)
    bias = (torch.arange(n).float() / n / 4).bfloat16()

    def dense_reference(activations, *, fused_bias):
        if kind == "column":
            flat = activations.reshape(-1, k)
            out = F.linear(flat, dense, bias if fused_bias else None)
            if not fused_bias:
                out = out + bias
        else:
            # Independent dense partials include BF16 rounding before
            # TP reduction and bias contributed only by TP rank zero.
            inputs = activations.chunk(tp, dim=-1)
            weights = dense.chunk(tp, dim=1)
            partials = []
            for i, (part_x, part_weight) in enumerate(zip(inputs, weights)):
                out = F.linear(
                    part_x.reshape(-1, k // tp).contiguous(),
                    part_weight.contiguous(),
                    bias if fused_bias and i == 0 else None,
                )
                if not fused_bias and i == 0:
                    out = out + bias
                partials.append(out)
            # Mathematical sum of BF16 GEMM partials. The fixture
            # keeps every legal BF16 reduction order within tolerance.
            out = torch.stack(partials).float().sum(0).bfloat16()
        return out.reshape(*activations.shape[:-1], n)

    high_reference = dense_reference(x, fused_bias=True)
    native_reference = dense_reference(quantized_x, fused_bias=False)
    high_references = [dense_reference(branch, fused_bias=True) for branch in branches]
    native_references = [
        dense_reference(
            ref_nvfp4_quant_dequant(branch.reshape(-1, k), input_global.reciprocal(), 16).reshape_as(branch),
            fused_bias=False,
        )
        for branch in branches
    ]
    assert not torch.allclose(high_reference, native_reference, rtol=0.01, atol=0.01)
    if tp == 4:
        wrong_partition = F.linear(x.reshape(-1, k), dense.roll(k // tp, dims=1), bias)
        assert not torch.allclose(high_reference.reshape(-1, n), wrong_partition, rtol=0.025, atol=0.025)
    if cfg > 1:
        assert not torch.allclose(high_references[0], high_references[1], rtol=0.025, atol=0.025)
    if sp > 1:
        assert not torch.allclose(
            high_references[cfg_rank],
            high_references[cfg_rank].roll(3, dims=1),
            rtol=0.025,
            atol=0.025,
        )
    return _LinearCase(
        checkpoint={
            "weight": packed,
            "weight_scale": scales,
            "weight_scale_2": weight_global,
            "input_scale": input_global,
            "bias": bias,
        },
        dense_weight=dense,
        inputs=x,
        high_reference=high_reference,
        native_reference=native_reference,
        high_references=high_references,
        native_references=native_references,
        rtol=0.025,
        atol=0.025,
    )


def _quant_config(precision):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptMixedPrecisionConfig

    if precision == "fp8":
        return ModelOptFp8Config("FP8", True, None, [])
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "group_size": 16,
            "exclude_modules": [],
            "quantized_layers": {
                "gen_layers.0.linear": {"quant_algo": "NVFP4"},
                "language_model.layers.0.linear": {"quant_algo": "NVFP4"},
            },
        }
    )


def _run_parallel(
    rank: int, init_method: str, precision: str, backend: str | None, topology: tuple[int, int, int]
) -> None:
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear

    from vllm_omni.diffusion.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
    from vllm_omni.diffusion.models.cosmos3.mixed_precision import (
        Cosmos3MixedPrecisionConfig,
        Cosmos3MixedPrecisionRuntime,
    )

    tp, sp, cfg = topology
    tp_rank, sp_rank, cfg_rank = initialize_parallel(rank, init_method, topology)
    config = VllmConfig()
    if backend is not None:
        config.kernel_config.linear_backend = backend
    quant = _quant_config(precision)
    make_case = _fp8_case if precision == "fp8" else _nvfp4_case
    try:
        with set_current_vllm_config(config), torch.device(f"cuda:{rank}"), torch.inference_mode():
            config.model_config = SimpleNamespace(dtype=torch.bfloat16)
            for kind in ("column", "row", "row_partitioned"):
                k, n = (128, 256) if kind == "column" else (256, 128)
                kwargs = dict(bias=True, params_dtype=torch.bfloat16, quant_config=quant)

                def make_linear(prefix):
                    if kind == "column":
                        return ColumnParallelLinear(k, n, gather_output=True, prefix=prefix, **kwargs)
                    return RowParallelLinear(k, n, input_is_parallel=kind == "row_partitioned", prefix=prefix, **kwargs)

                generation = make_linear("gen_layers.0.linear")
                reasoner = make_linear("language_model.layers.0.linear")
                transformer = torch.nn.Module()
                transformer.gen_layers = torch.nn.ModuleList([generation])
                transformer.language_model = torch.nn.Module()
                transformer.language_model.layers = torch.nn.ModuleList([reasoner])
                loaders = [layer.weight.weight_loader for layer in (generation, reasoner)]
                runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig())
                runtime.install(transformer)

                case = make_case(kind, k, n, topology, sp_rank, cfg_rank)
                local_dense = case.dense_weight.chunk(tp, dim=0 if kind == "column" else 1)[tp_rank]
                local_x = (
                    case.inputs.chunk(tp, dim=-1)[tp_rank].contiguous() if kind == "row_partitioned" else case.inputs
                )

                native_calls = [0, 0]
                for index, layer in enumerate((generation, reasoner)):
                    assert layer.weight.weight_loader == loaders[index]
                    for name, value in case.checkpoint.items():
                        param = getattr(layer, name)
                        param.weight_loader(param, value)
                    layer.quant_method.process_weights_after_loading(layer)
                    torch.testing.assert_close(
                        layer.quant_method.strategy.materialize(layer), local_dense, rtol=0, atol=0
                    )
                    original = layer.quant_method.base_method.apply

                    def counted_apply(layer, x, bias=None, *, original=original, index=index):
                        native_calls[index] += 1
                        return original(layer, x, bias)

                    layer.quant_method.base_method.apply = counted_apply

                for first, last, high_steps, expected_native in (
                    (3, 3, {0, 1, 2, 7, 8, 9}, 4),
                    (0, 0, set(), 10),
                    (6, 6, set(range(10)), 0),
                ):
                    runtime.config = Cosmos3MixedPrecisionConfig(first_steps=first, last_steps=last)
                    previous: dict[tuple[int, int], torch.Tensor] = {}
                    for request in range(2):
                        before = native_calls.copy()
                        for step in range(10):
                            runtime.set_step(step, 10)
                            assert runtime.use_high_precision("generation") == (step in high_steps)
                            for index, layer in enumerate((generation, reasoner)):
                                high = index == 1 or step in high_steps
                                expected = case.high_reference if high else case.native_reference
                                actual, _ = layer(local_x)
                                torch.testing.assert_close(
                                    actual,
                                    expected,
                                    rtol=case.rtol,
                                    atol=case.atol,
                                    msg=lambda msg: (
                                        f"{precision}/{backend} {kind} topology={topology} rank={rank} "
                                        f"step={step} high={high}: {msg}"
                                    ),
                                )
                                assembled = assemble_output(actual, sp, cfg, cfg_rank)
                                references = case.high_references if high else case.native_references
                                assert_parallel_output(assembled, references, cfg, rtol=case.rtol, atol=case.atol)
                                if request:
                                    torch.testing.assert_close(assembled, previous[step, index], rtol=0, atol=0)
                                else:
                                    previous[step, index] = assembled.clone()
                        assert native_calls == [before[0] + expected_native, before[1]]
                        runtime.reset()
                        assert not runtime.use_high_precision("generation")
                        assert runtime.use_high_precision("reasoner")
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    "precision,backend,topology",
    [
        pytest.param(
            precision,
            backend,
            *case.values,
            id=f"{precision}-{backend or 'native'}-{case.id}",
            marks=hardware_marks(
                res={"cuda": "B200" if backend == "flashinfer_cutedsl" else ["H100", "B200"]},
                num_cards=prod(case.values[0]),
            ),
        )
        for precision, backend in (("fp8", None), ("nvfp4", "emulation"), ("nvfp4", "flashinfer_cutedsl"))
        for case in TOPOLOGIES
    ],
)
def test_modelopt_maps_parallel(tmp_path, precision, backend, topology) -> None:
    world_size = prod(topology)
    if not torch.cuda.is_available() or torch.accelerator.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    if precision == "fp8" and any(torch.cuda.get_device_capability(i) < (8, 9) for i in range(world_size)):
        pytest.skip("Requires native FP8 CUDA support")
    if backend == "flashinfer_cutedsl" and any(torch.cuda.get_device_capability(i)[0] != 10 for i in range(world_size)):
        pytest.skip("Native FlashInfer CuteDSL NVFP4 requires SM10x GPUs")
    # Match diffusion's in-process Quack compilation for daemon workers.
    torch.multiprocessing.spawn(
        _run_parallel,
        args=(f"file://{tmp_path / 'rendezvous'}", precision, backend, topology),
        nprocs=world_size,
        join=True,
        daemon=True,
    )
