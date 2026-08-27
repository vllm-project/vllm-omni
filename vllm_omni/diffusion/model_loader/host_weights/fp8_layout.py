# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Online per-tensor FP8 policy for shared final-layout artifacts."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from vllm_omni.diffusion.models.host_weight_contract import (
    FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA,
)
from vllm_omni.host_weight_runtime import (
    CanonicalJson,
    ProducerIdentity,
    TensorKind,
    WeightRepresentation,
)

from .contracts import (
    FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA,
    FinalLayoutArtifactSpec,
    FinalLayoutContractCode,
    FinalLayoutContractError,
    FinalLayoutRequest,
    implementation_abi_fingerprint,
)
from .tensor_layout import RuntimeTensorTarget

FINAL_LAYOUT_FP8_PRODUCER_ID = "vllm-omni.diffusion.final-layout-fp8"
FINAL_LAYOUT_FP8_VERSION = "1"
FINAL_LAYOUT_FP8_REPRESENTATION = "diffusion-online-fp8-per-tensor"
FINAL_LAYOUT_FP8_MANIFEST_SCHEMA = "diffusion-final-layout-fp8-manifest-v1"

_FP8_DTYPE = torch.float8_e4m3fn
_FP8_IMPLEMENTATION_ABI = CanonicalJson.from_value(
    {
        "artifact_identity": "diffusion-final-layout-identity-v1",
        "model_contract": FINAL_LAYOUT_TENSOR_MODEL_CONTRACT_SCHEMA,
        "producer": "bounded-checkpoint-stream-fp8-v1",
        "representation_policy": "online-per-tensor-e4m3-canonical-storage-v1",
        "restorer": "exact-rebind-plus-zero-copy-kernel-view-v1",
        "source_identity": "prepared-diffusion-weight-source-v2",
        "tensor_contract": "complete-strided-tensor-ownership-v1",
    }
)

FINAL_LAYOUT_FP8_SPEC = FinalLayoutArtifactSpec(
    representation=WeightRepresentation(
        name=FINAL_LAYOUT_FP8_REPRESENTATION,
        dtype=str(_FP8_DTYPE),
        metadata=CanonicalJson.from_value(
            {
                "activation": "dynamic-per-token",
                "scale": "one-fp32-scale-per-weight-tensor",
                "weight_storage": "contiguous-nk-zero-copy-kernel-view",
            }
        ),
    ),
    producer=ProducerIdentity(
        producer_id=FINAL_LAYOUT_FP8_PRODUCER_ID,
        version=FINAL_LAYOUT_FP8_VERSION,
        implementation_fingerprint=implementation_abi_fingerprint(_FP8_IMPLEMENTATION_ABI),
        manifest_schema=FINAL_LAYOUT_FP8_MANIFEST_SCHEMA,
        restorer_schema=FINAL_LAYOUT_TENSOR_RESTORER_SCHEMA,
    ),
    implementation_abi=_FP8_IMPLEMENTATION_ABI,
    layout_name="diffusion-online-fp8-canonical-storage-v1",
)


class FinalLayoutFP8Policy:
    @property
    def spec(self) -> FinalLayoutArtifactSpec:
        return FINAL_LAYOUT_FP8_SPEC

    def validate_request(self, request: FinalLayoutRequest) -> None:
        if request.load_format != "default":
            raise ValueError("runtime FP8 artifacts require load_format='default'")
        if request.adaptation.kind != "base" or request.adaptation.fingerprint is not None:
            raise ValueError("runtime FP8 artifacts require unmodified base weights")
        if request.parallel.tensor_parallel_size != 1 or request.parallel.use_hsdp:
            raise ValueError("runtime FP8 artifacts require TP=1 without HSDP")

    def tensor_role(self, name: str, tensor: torch.Tensor, kind: TensorKind) -> str:
        if kind is TensorKind.BUFFER:
            return "persistent_buffer"
        if tensor.dtype is _FP8_DTYPE:
            return "fp8_weight"
        if name.endswith(".weight_scale"):
            return "fp8_scale"
        return "preserved_parameter"

    def validate_target(self, target: RuntimeTensorTarget) -> None:
        if target.role == "fp8_weight" and target.tensor.ndim != 2:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                f"{target.name!r} must be a two-dimensional FP8 weight",
            )
        if target.tensor.dtype not in {
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            _FP8_DTYPE,
        }:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.DTYPE_UNSUPPORTED,
                f"{target.name!r} uses unsupported dtype {target.tensor.dtype}",
            )

    def validate_collection(self, targets: Sequence[RuntimeTensorTarget]) -> None:
        by_name = {target.name: target for target in targets}
        weights = [target for target in targets if target.role == "fp8_weight"]
        if not weights:
            raise FinalLayoutContractError(
                FinalLayoutContractCode.DTYPE_UNSUPPORTED,
                "runtime FP8 artifacts require online per-tensor FP8 weights",
            )
        for weight in weights:
            scale = by_name.get(f"{weight.name.removesuffix('.weight')}.weight_scale")
            if scale is None or scale.tensor.dtype is not torch.float32 or tuple(scale.tensor.shape) != (1,):
                raise FinalLayoutContractError(
                    FinalLayoutContractCode.TENSOR_UNSUPPORTED,
                    f"{weight.name!r} has no matching FP32 scalar weight_scale",
                )

    def build_format_metadata(
        self,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> CanonicalJson:
        return CanonicalJson.from_value(
            {
                "component_names": list(component_names),
                "format": FINAL_LAYOUT_FP8_REPRESENTATION,
                "tensor_contract_sha256": tensor_contract_digest,
                "tensor_count": tensor_count,
                "weight_storage": "contiguous-nk-zero-copy-kernel-view",
            }
        )

    def validate_format_metadata(
        self,
        metadata: CanonicalJson,
        *,
        component_names: tuple[str, ...],
        tensor_contract_digest: str,
        tensor_count: int,
    ) -> None:
        expected = self.build_format_metadata(
            component_names=component_names,
            tensor_contract_digest=tensor_contract_digest,
            tensor_count=tensor_count,
        )
        if metadata != expected:
            raise ValueError("lease format metadata differs from the runtime FP8 artifact policy")


FINAL_LAYOUT_FP8_POLICY = FinalLayoutFP8Policy()


class FinalLayoutFP8ModelPreparation:
    """Declare canonical FP8 storage, then activate runtime kernel views."""

    def __init__(self, dit_modules: Sequence[tuple[str, nn.Module]]) -> None:
        from vllm.model_executor.layers.quantization.online.fp8 import (
            Fp8PerTensorOnlineLinearMethod,
        )

        layers = {
            id(module): module
            for _, dit in dit_modules
            for module in dit.modules()
            if isinstance(getattr(module, "quant_method", None), Fp8PerTensorOnlineLinearMethod)
        }
        self._layers = list(layers.values())

    def prepare(self) -> None:
        if not self._layers:
            raise ValueError("no online per-tensor FP8 layers were discovered")
        for layer in self._layers:
            weight = layer.weight
            layer._parameters["weight"] = nn.Parameter(
                torch.empty(weight.shape, device="meta", dtype=_FP8_DTYPE),
                requires_grad=False,
            )
            layer._parameters["weight_scale"] = nn.Parameter(
                torch.empty((1,), device="meta", dtype=torch.float32),
                requires_grad=False,
            )
            layer.input_scale = None
            layer._already_called_process_weights_after_loading = True

    def activate_kernel_views(self) -> None:
        from vllm.model_executor.utils import replace_parameter

        for layer in self._layers:
            method = layer.quant_method
            replace_parameter(layer, "weight", layer.weight.t().data)
            if method.use_marlin and hasattr(method.fp8_linear, "marlin_input_dtype"):
                method.fp8_linear.marlin_input_dtype = method.marlin_input_dtype
            method.fp8_linear.process_weights_after_loading(layer)


__all__ = [
    "FINAL_LAYOUT_FP8_MANIFEST_SCHEMA",
    "FINAL_LAYOUT_FP8_POLICY",
    "FINAL_LAYOUT_FP8_PRODUCER_ID",
    "FINAL_LAYOUT_FP8_REPRESENTATION",
    "FINAL_LAYOUT_FP8_SPEC",
    "FINAL_LAYOUT_FP8_VERSION",
    "FinalLayoutFP8ModelPreparation",
    "FinalLayoutFP8Policy",
]
