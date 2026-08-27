# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for the final-layout FP8 Host Weight Runtime path."""

from __future__ import annotations

import dataclasses
import gc
import mmap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from vllm.model_executor.kernels.linear.scaled_mm.pytorch import (
    PerTensorTorchFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerTensorOnlineLinearMethod,
)

from vllm_omni.diffusion.model_loader.host_weight_plan import HostWeightPlan, TensorBinding
from vllm_omni.diffusion.model_loader.host_weights import (
    FINAL_LAYOUT_FP8_POLICY,
    FinalLayoutFP8ModelPreparation,
    FinalLayoutFP8Producer,
    FinalLayoutIdentityContext,
    FinalLayoutLoaderIdentity,
    FinalLayoutParallelIdentity,
    FinalLayoutRequest,
    FinalLayoutTensorRestorer,
    ImplementationIdentity,
    PreparedWeightSource,
    build_final_layout_identity,
)
from vllm_omni.diffusion.model_loader.host_weights.contracts import (
    FinalLayoutContractCode,
    FinalLayoutContractError,
)
from vllm_omni.diffusion.model_loader.host_weights.producers import final_layout_fp8 as producer_module
from vllm_omni.diffusion.model_loader.host_weights.producers.final_layout_fp8 import _tensor_ranges
from vllm_omni.diffusion.model_loader.host_weights.runtime_fp8 import (
    RuntimeFP8UnavailableError,
    runtime_fp8_requested,
    validate_online_fp8,
)
from vllm_omni.diffusion.model_loader.host_weights.tensor_layout import (
    RuntimeTensorTarget,
    split_tensor_targets_by_bytes,
)
from vllm_omni.diffusion.models.host_weight_contract import FinalLayoutModelContract
from vllm_omni.host_weight_runtime import (
    AdaptationIdentity,
    HostWeightLease,
    HostWeightRuntime,
    HostWeightRuntimeConfig,
    ProductionPolicy,
    ResolutionOutcome,
    RuntimeMode,
    StorageDomainPolicy,
    TensorFileWriter,
    TensorKind,
)
from vllm_omni.host_weight_runtime import lease as lease_module
from vllm_omni.host_weight_runtime.filesystem import detect_storage_class

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _online_fp8_method() -> Fp8PerTensorOnlineLinearMethod:
    method = object.__new__(Fp8PerTensorOnlineLinearMethod)
    method.uses_meta_device = True
    method.use_marlin = False
    method.marlin_input_dtype = None
    method.fp8_linear = object.__new__(PerTensorTorchFP8ScaledMMLinearKernel)
    return method


class _TinyFP8DiT(nn.Module):
    host_weight_restore_contract = FinalLayoutModelContract(
        implementation_id="test-tiny-fp8-dit",
        version="1",
    )

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
        self.proj.quant_method = _online_fp8_method()

    def validate_restored_host_weights(self) -> None:
        assert self.proj.weight.dtype is torch.float8_e4m3fn
        assert tuple(self.proj.weight.shape) == (3, 2)
        assert self.proj.weight.stride() == (1, 3)
        assert self.proj.weight_scale.dtype is torch.float32
        assert tuple(self.proj.weight_scale.shape) == (1,)


class _TinyFP8Pipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _TinyFP8DiT()


class _CPUFinalLayoutFP8Producer(FinalLayoutFP8Producer):
    """Exercise artifact production while replacing only the CUDA quantizer."""

    def _write_fp8_weight(self, output: TensorFileWriter, record: RuntimeTensorTarget) -> torch.Tensor:
        values = torch.arange(record.tensor.numel(), dtype=torch.float32).reshape(record.tensor.shape)
        output.write_tensor(record.name, values.to(torch.float8_e4m3fn))
        return torch.tensor([0.25], dtype=torch.float32)


def _request(*, model_config_fingerprint: str = "test-model-config-v1") -> FinalLayoutRequest:
    return FinalLayoutRequest(
        model_id="test-org/tiny-fp8",
        loader=FinalLayoutLoaderIdentity(
            implementation=ImplementationIdentity(
                implementation_id="test-fp8-loader",
                version="1",
                fingerprint="test-fp8-loader-v1",
            ),
            model_config_fingerprint=model_config_fingerprint,
            weight_transform_fingerprint="test-fp8-transform-v1",
        ),
    )


def _prepared_fp8_model(
    tmp_path: Path,
    *,
    model_config_fingerprint: str = "test-model-config-v1",
) -> tuple[
    _TinyFP8Pipeline,
    FinalLayoutFP8ModelPreparation,
    FinalLayoutIdentityContext,
    HostWeightPlan,
]:
    checkpoint = tmp_path / "canonical" / "model.safetensors"
    checkpoint.parent.mkdir(exist_ok=True)
    if not checkpoint.exists():
        save_file(
            {"proj.weight": torch.arange(6, dtype=torch.float32).to(torch.bfloat16).reshape(2, 3)},
            str(checkpoint),
        )
    source = PreparedWeightSource(
        model_or_path=str(checkpoint.parent),
        subfolder=None,
        requested_revision=None,
        prefix="transformer.",
        resolved_root=checkpoint.parent,
        weight_files=(checkpoint,),
        use_safetensors=True,
    )
    model = _TinyFP8Pipeline()
    dit_modules = (("transformer", model.transformer),)
    preparation = FinalLayoutFP8ModelPreparation(dit_modules)
    preparation.prepare()
    context = build_final_layout_identity(
        model,
        dit_modules=dit_modules,
        prepared_sources=(source,),
        request=_request(model_config_fingerprint=model_config_fingerprint),
        policy=FINAL_LAYOUT_FP8_POLICY,
    )
    checkpoint_plan = HostWeightPlan(
        backing_kind="checkpoint_mmap",
        bindings={
            "transformer.proj.weight": TensorBinding(
                checkpoint_key="proj.weight",
                file_path=str(checkpoint),
            )
        },
        planned_source_prefixes=frozenset({"transformer."}),
    )
    return model, preparation, context, checkpoint_plan


def _runtime(root: Path) -> HostWeightRuntime:
    return HostWeightRuntime.from_config(
        HostWeightRuntimeConfig(
            mode=RuntimeMode.PREFERRED,
            domain=StorageDomainPolicy(root=root, storage_class=detect_storage_class(root.parent)),
            production=ProductionPolicy(allow_local_build=True),
        )
    )


def _scope_config(**changes: object) -> SimpleNamespace:
    values = {
        "host_weight_runtime_mode": RuntimeMode.PREFERRED.value,
        "host_weight_runtime_root": "/tmp/hwr",
        "enable_distributed_layerwise_offload": True,
        "dlo_use_allgather": True,
        "lora_path": None,
        "data_parallel_size": 2,
        "sequence_parallel_size": 1,
        "tensor_parallel_size": 1,
        "use_hsdp": False,
    }
    values.update(changes)
    parallel = SimpleNamespace(
        data_parallel_size=values.pop("data_parallel_size"),
        sequence_parallel_size=values.pop("sequence_parallel_size"),
        tensor_parallel_size=values.pop("tensor_parallel_size"),
        use_hsdp=values.pop("use_hsdp"),
    )
    return SimpleNamespace(parallel_config=parallel, **values)


@pytest.mark.parametrize(
    ("changes", "load_format", "device", "message"),
    [
        ({"host_weight_runtime_root": None}, "default", torch.device("cuda"), "root is not configured"),
        ({"enable_distributed_layerwise_offload": False}, "default", torch.device("cuda"), "offload is disabled"),
        ({"dlo_use_allgather": False}, "default", torch.device("cuda"), "requires DLO AllGather"),
        (
            {"data_parallel_size": 1, "sequence_parallel_size": 1},
            "default",
            torch.device("cuda"),
            "more than one DLO rank",
        ),
        ({}, "default", torch.device("cpu"), "requires CUDA"),
        ({}, "custom", torch.device("cuda"), "load_format='default'"),
        ({"tensor_parallel_size": 2}, "default", torch.device("cuda"), "TP=1 without HSDP"),
        ({"use_hsdp": True}, "default", torch.device("cuda"), "TP=1 without HSDP"),
        ({"lora_path": "adapter"}, "default", torch.device("cuda"), "base weights only"),
    ],
)
def test_runtime_fp8_scope_preferred_falls_back_and_required_raises(
    changes: dict[str, object],
    load_format: str,
    device: torch.device,
    message: str,
) -> None:
    config = _scope_config(**changes)
    assert not runtime_fp8_requested(config, load_format, device)

    config.host_weight_runtime_mode = RuntimeMode.REQUIRED.value
    with pytest.raises(RuntimeFP8UnavailableError, match=message):
        runtime_fp8_requested(config, load_format, device)


@pytest.mark.parametrize(
    ("dp", "sp"),
    [(2, 1), (1, 2)],
)
def test_runtime_fp8_scope_accepts_multi_rank_allgather(dp: int, sp: int) -> None:
    config = _scope_config(data_parallel_size=dp, sequence_parallel_size=sp)
    assert runtime_fp8_requested(config, "default", torch.device("cuda"))

    config.host_weight_runtime_mode = RuntimeMode.DISABLED.value
    assert not runtime_fp8_requested(config, "default", torch.device("cuda"))


def test_online_fp8_validation_accepts_only_exclusive_per_tensor_methods() -> None:
    valid = nn.Module()
    valid.layer = nn.Linear(2, 2)
    valid.layer.quant_method = _online_fp8_method()
    validate_online_fp8((("transformer", valid),))

    with pytest.raises(RuntimeFP8UnavailableError, match="exclusively per-tensor FP8"):
        validate_online_fp8((("transformer", nn.Linear(2, 2)),))

    invalid = nn.Linear(2, 2)
    invalid.quant_method = SimpleNamespace(uses_meta_device=True)
    with pytest.raises(RuntimeFP8UnavailableError, match="exclusively per-tensor FP8"):
        validate_online_fp8((("transformer", invalid),))


def test_fp8_policy_validates_request_weight_and_scale_contract() -> None:
    FINAL_LAYOUT_FP8_POLICY.validate_request(_request())
    weight = RuntimeTensorTarget(
        "transformer.proj.weight",
        torch.empty((2, 3), dtype=torch.float8_e4m3fn),
        TensorKind.PARAMETER,
        "fp8_weight",
    )
    scale = RuntimeTensorTarget(
        "transformer.proj.weight_scale",
        torch.empty((1,), dtype=torch.float32),
        TensorKind.PARAMETER,
        "fp8_scale",
    )
    FINAL_LAYOUT_FP8_POLICY.validate_target(weight)
    FINAL_LAYOUT_FP8_POLICY.validate_collection((weight, scale))

    with pytest.raises(ValueError, match="load_format='default'"):
        FINAL_LAYOUT_FP8_POLICY.validate_request(dataclasses.replace(_request(), load_format="custom"))
    with pytest.raises(ValueError, match="unmodified base weights"):
        FINAL_LAYOUT_FP8_POLICY.validate_request(
            dataclasses.replace(
                _request(),
                adaptation=AdaptationIdentity(kind="merged-lora", fingerprint="adapter-sha256"),
            )
        )
    with pytest.raises(ValueError, match="TP=1 without HSDP"):
        FINAL_LAYOUT_FP8_POLICY.validate_request(
            dataclasses.replace(
                _request(),
                parallel=FinalLayoutParallelIdentity(tensor_parallel_size=2),
            )
        )
    with pytest.raises(FinalLayoutContractError, match="two-dimensional") as shape_error:
        FINAL_LAYOUT_FP8_POLICY.validate_target(
            dataclasses.replace(weight, tensor=torch.empty(3, dtype=weight.tensor.dtype))
        )
    assert shape_error.value.code is FinalLayoutContractCode.TENSOR_UNSUPPORTED
    with pytest.raises(FinalLayoutContractError, match="unsupported dtype") as dtype_error:
        FINAL_LAYOUT_FP8_POLICY.validate_target(
            RuntimeTensorTarget(
                "transformer.complex",
                torch.empty(1, dtype=torch.complex64),
                TensorKind.PARAMETER,
                "preserved_parameter",
            )
        )
    assert dtype_error.value.code is FinalLayoutContractCode.DTYPE_UNSUPPORTED
    with pytest.raises(FinalLayoutContractError, match="matching FP32 scalar"):
        FINAL_LAYOUT_FP8_POLICY.validate_collection((weight, dataclasses.replace(scale, tensor=torch.empty(2))))


def test_fp8_model_preparation_restores_canonical_storage_then_activates_kernel_view() -> None:
    model = _TinyFP8Pipeline()
    preparation = FinalLayoutFP8ModelPreparation((("transformer", model.transformer),))

    preparation.prepare()

    layer = model.transformer.proj
    assert layer.weight.is_meta
    assert layer.weight.dtype is torch.float8_e4m3fn
    assert tuple(layer.weight.shape) == (2, 3)
    assert layer.weight_scale.is_meta
    physical = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.float8_e4m3fn)
    layer._parameters["weight"] = nn.Parameter(physical, requires_grad=False)
    layer._parameters["weight_scale"] = nn.Parameter(torch.tensor([0.25]), requires_grad=False)
    storage_pointer = layer.weight.untyped_storage().data_ptr()

    preparation.activate_kernel_views()

    assert tuple(layer.weight.shape) == (3, 2)
    assert layer.weight.stride() == (1, 3)
    assert layer.weight.untyped_storage().data_ptr() == storage_pointer


def test_split_targets_keeps_tensors_whole_when_weight_and_scale_cross_a_shard_boundary() -> None:
    weight = RuntimeTensorTarget(
        "layer.weight",
        torch.empty((2, 3), dtype=torch.float8_e4m3fn),
        TensorKind.PARAMETER,
        "fp8_weight",
    )
    scale = RuntimeTensorTarget(
        "layer.weight_scale",
        torch.empty((1,), dtype=torch.float32),
        TensorKind.PARAMETER,
        "fp8_scale",
    )

    shards = split_tensor_targets_by_bytes((weight, scale), max_shard_bytes=weight.nbytes)

    assert tuple(tuple(target.name for target in shard) for shard in shards) == (
        ("layer.weight",),
        ("layer.weight_scale",),
    )


def test_tensor_ranges_resolve_exact_safetensors_payload_bytes(tmp_path: Path) -> None:
    tensors = {
        "layer.scale": torch.tensor([0.25], dtype=torch.float32),
        "layer.weight": torch.arange(6, dtype=torch.float32).to(torch.bfloat16).reshape(2, 3),
    }
    path = tmp_path / "weights.safetensors"
    save_file(tensors, str(path))

    ranges = _tensor_ranges(str(path))

    assert set(ranges) == set(tensors)
    with path.open("rb") as handle:
        for name, tensor in tensors.items():
            offset, nbytes = ranges[name]
            handle.seek(offset)
            expected = tensor.contiguous().view(torch.uint8).numpy().tobytes()
            assert nbytes == len(expected)
            assert handle.read(nbytes) == expected


def test_fp8_producer_cold_warm_sharding_determinism_and_restore_rejection(tmp_path: Path) -> None:
    model, preparation, context, checkpoint_plan = _prepared_fp8_model(tmp_path)
    producer = _CPUFinalLayoutFP8Producer(
        context,
        model,
        (("transformer", model.transformer),),
        checkpoint_plan,
        device=torch.device("cpu"),
        max_shard_bytes=6,
    )
    runtime = _runtime(tmp_path / "store")

    cold = runtime.resolve(context.identity, producer=producer)
    assert cold.report.outcome is ResolutionOutcome.LOCAL_PRODUCTION
    assert cold.lease is not None
    entries = {entry.name: entry for entry in cold.lease.manifest.tensors}
    assert entries["transformer.proj.weight"].file_name != entries["transformer.proj.weight_scale"].file_name

    warm_producer = _CPUFinalLayoutFP8Producer(
        context,
        model,
        (("transformer", model.transformer),),
        checkpoint_plan,
        device=torch.device("cpu"),
    )

    def unexpected_production(_writer: object) -> object:
        raise AssertionError("warm resolution must not invoke the producer")

    warm_producer.produce = unexpected_production  # type: ignore[method-assign]
    warm = runtime.resolve(context.identity, producer=warm_producer)
    assert warm.report.outcome is ResolutionOutcome.LOCAL_HIT
    assert warm.lease is not None

    second_model, _, second_context, second_checkpoint_plan = _prepared_fp8_model(tmp_path)
    second = _runtime(tmp_path / "second-store").resolve(
        second_context.identity,
        producer=_CPUFinalLayoutFP8Producer(
            second_context,
            second_model,
            (("transformer", second_model.transformer),),
            second_checkpoint_plan,
            device=torch.device("cpu"),
            max_shard_bytes=1024,
        ),
    )
    assert second.report.outcome is ResolutionOutcome.LOCAL_PRODUCTION
    assert second.lease is not None
    assert cold.lease.manifest.artifact_content_sha256 == second.lease.manifest.artifact_content_sha256
    assert {entry.name: entry.sha256 for entry in cold.lease.manifest.tensors} == {
        entry.name: entry.sha256 for entry in second.lease.manifest.tensors
    }

    wrong_model, _, wrong_context, _ = _prepared_fp8_model(
        tmp_path,
        model_config_fingerprint="different-model-config",
    )
    with pytest.raises(ValueError, match="semantic identity differs"):
        FinalLayoutTensorRestorer(wrong_context).plan_restore(wrong_model, warm.lease)

    restored_pointer = warm.lease.tensors["transformer.proj.weight"].untyped_storage().data_ptr()
    plan = FinalLayoutTensorRestorer(
        context,
        post_commit=preparation.activate_kernel_views,
    ).plan_restore(model, warm.lease)
    plan.commit()
    assert model.transformer.proj.weight.untyped_storage().data_ptr() == restored_pointer

    cold.lease.close()
    second.lease.close()
    del model, preparation, plan
    gc.collect()
    warm.lease.close()


def test_scale_production_orders_both_quantization_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    current_stream = object()

    class Event:
        recorded_on: object | None = None

        def record(self, stream: object) -> None:
            self.recorded_on = stream

    class Stream:
        def __init__(self) -> None:
            self.waited_for: list[Event] = []

        def wait_event(self, event: Event) -> None:
            self.waited_for.append(event)

    class Ready:
        def synchronize(self) -> None:
            pass

    event = Event()
    streams = (Stream(), Stream())
    slots = (
        SimpleNamespace(
            stream=streams[0],
            pending_rows=1,
            ready=Ready(),
            host_amax=torch.tensor([2.0]),
        ),
        SimpleNamespace(stream=streams[1], pending_rows=0),
    )
    producer = object.__new__(FinalLayoutFP8Producer)
    producer._device = torch.device("cpu")
    monkeypatch.setattr(producer_module.torch.cuda, "Event", lambda: event)
    monkeypatch.setattr(producer_module.torch.cuda, "current_stream", lambda _device: current_stream)
    monkeypatch.setattr(
        producer_module.ops,
        "scaled_fp8_quant",
        lambda probe: (probe, torch.tensor([0.25])),
    )

    scale = producer._find_scale(torch.empty((0, 1)), slots, rows_per_chunk=1)  # type: ignore[arg-type]

    assert torch.equal(scale, torch.tensor([0.25]))
    assert event.recorded_on is current_stream
    assert all(stream.waited_for == [event] for stream in streams)


def test_release_tensor_pages_uses_page_aligned_madvise(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int, int]] = []
    tensor = torch.arange(17, dtype=torch.float32)
    storage = tensor.untyped_storage()
    address = storage.data_ptr()
    expected_start = address - address % mmap.PAGESIZE
    expected_end = (address + storage.nbytes() + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE

    def madvise(start: int, length: int, advice: int) -> int:
        calls.append((start, length, advice))
        return 0

    monkeypatch.setattr(lease_module, "_MADVISE", madvise)
    lease = object.__new__(HostWeightLease)

    lease.release_tensor_pages(tensor)
    lease.release_tensor_pages(torch.empty(0))

    assert calls == [(expected_start, expected_end - expected_start, mmap.MADV_DONTNEED)]
