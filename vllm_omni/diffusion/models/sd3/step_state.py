# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-owned SD3 state, without live scheduler or generator objects."""

import os
from dataclasses import fields
from typing import TYPE_CHECKING, Any

import diffusers
import msgspec
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from vllm import envs
from vllm.lora.request import LoRARequest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.distributed.omni_connectors.utils.serialization import OmniMsgpackDecoder, OmniMsgpackEncoder
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig


class _Snapshot(msgspec.Struct, frozen=True):
    policy_ref: str
    torch_version: str
    diffusers_version: str
    scheduler_type: str
    state: bytes
    execution_profile: bytes
    format: str = "sd3-state-v2"


class _RequestFields(msgspec.Struct):
    request_id: str
    external_req_id: str | None
    prompt: msgspec.Raw
    step_index: int


class _RequestProjection(msgspec.Struct):
    request: _RequestFields
    sampling: msgspec.Raw


class _StateEncoder(OmniMsgpackEncoder):
    def __init__(self, max_bytes: int):
        super().__init__()
        self.remaining = max_bytes

    def _encode_tensor(self, tensor: torch.Tensor) -> dict[str, Any]:
        self.remaining -= tensor.numel() * tensor.element_size()
        if self.remaining < 0:
            raise ValueError("SD3 snapshot exceeds its byte budget")
        return {**super()._encode_tensor(tensor), "state_device": tensor.device.type}


class _StateDecoder(OmniMsgpackDecoder):
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

    def _decode_tensor(self, obj: dict[str, Any]) -> torch.Tensor:
        source = obj["state_device"]
        if source not in ("cpu", self.device.type):
            raise ValueError("SD3 snapshot device backend is incompatible")
        tensor = super()._decode_tensor(obj)
        return tensor if source == "cpu" else tensor.to(self.device)


def step_execution_profile(config: "OmniDiffusionConfig", device: torch.device) -> bytes:
    """Record the worker's numeric execution settings at a snapshot boundary."""
    accelerator = None
    if device.type == "cuda":
        matmul = torch.backends.cuda.matmul
        accelerator = {
            "capability": torch.cuda.get_device_capability(device),
            "tf32": matmul.allow_tf32,
            "bf16_reduced_precision_reduction": matmul.allow_bf16_reduced_precision_reduction,
            "fp16_reduced_precision_reduction": matmul.allow_fp16_reduced_precision_reduction,
            "blas": str(torch.backends.cuda.preferred_blas_library()),
            "sdpa": (
                torch.backends.cuda.flash_sdp_enabled(),
                torch.backends.cuda.mem_efficient_sdp_enabled(),
                torch.backends.cuda.math_sdp_enabled(),
                torch.backends.cuda.cudnn_sdp_enabled(),
            ),
            "cudnn": (
                torch.backends.cudnn.enabled,
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
                torch.backends.cudnn.allow_tf32,
            ),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        }
    return msgspec.json.encode(
        {
            "backend": device.type,
            "accelerator": accelerator,
            "dtype": str(config.dtype),
            "attention": config.diffusion_attention_config,
            "parallel": config.parallel_config,
            "cfg_batching": config.enable_cfg_batching,
            "flash_deterministic": config.fa_deterministic,
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_precision": torch.get_float32_matmul_precision(),
            "batch_invariant": envs.VLLM_BATCH_INVARIANT,
            "compile": (
                None
                if config.enforce_eager
                else (config.diffusion_compile_granularity, config.diffusion_compile_dynamic)
            ),
        },
        order="deterministic",
    )


def export_step_state(
    state: StepRequestState,
    *,
    policy_ref: str,
    max_bytes: int,
    execution_profile: bytes,
    scheduler_cls: type[FlowMatchEulerDiscreteScheduler] = FlowMatchEulerDiscreteScheduler,
) -> bytes:
    """Copy a completed step boundary; leave the running state unchanged."""
    if not policy_ref or max_bytes <= 0:
        raise ValueError("SD3 snapshot requires a policy reference and positive byte budget")
    if type(state.scheduler) is not scheduler_cls or state.scheduler.config.stochastic_sampling:
        raise ValueError("SD3 snapshot scheduler does not match the pipeline's scheduler contract")
    if state.latents is None or not 0 <= state.step_index < state.total_steps:
        raise ValueError("SD3 snapshot requires an initialized, unfinished request")
    if state.sampling.lora_request is not None and type(state.sampling.lora_request) is not LoRARequest:
        raise ValueError("SD3 snapshot currently supports one runtime LoRA request")
    if (
        state.interaction_sessions
        or state.chunk_num_steps is not None
        or state.mixfusion_info is not None
        or state.kv_sender_info is not None
        or state.prepared_layout is not None
        or state.attention_schedule
        or state.attention_sigma_schedule
    ):
        raise ValueError("SD3 snapshot does not support interactive, KV, mixed-resolution or scheduled-attention state")
    sampling = {
        field.name: getattr(state.sampling, field.name) for field in fields(state.sampling) if field.name != "generator"
    }
    generator = state.sampling.generator
    generators = generator if isinstance(generator, list) else ([] if generator is None else [generator])
    unique_generators = {id(item): item for item in generators}
    generator_indices = {identity: index for index, identity in enumerate(unique_generators)}
    payload = {
        "request": {
            field.name: getattr(state, field.name)
            for field in fields(state)
            if field.name not in ("sampling", "scheduler")
        },
        "sampling": sampling,
        "generators": [{"device": item.device.type, "state": item.get_state()} for item in unique_generators.values()],
        "generator_indices": [generator_indices[id(item)] for item in generators],
        "generator_is_list": isinstance(generator, list),
        "scheduler_config": dict(state.scheduler.config),
        "scheduler_state": {name: value for name, value in vars(state.scheduler).items() if name != "_internal_dict"},
    }
    data = _StateEncoder(max_bytes).encode(payload)
    scheduler_type = f"{scheduler_cls.__module__}.{scheduler_cls.__qualname__}"
    encoded = msgspec.msgpack.encode(
        _Snapshot(policy_ref, str(torch.__version__), diffusers.__version__, scheduler_type, data, execution_profile)
    )
    if len(encoded) > max_bytes:
        raise ValueError("SD3 snapshot exceeds its byte budget")
    return encoded


def _decode_snapshot(payload: bytes | memoryview, policy_ref: str, max_bytes: int) -> _Snapshot:
    if max_bytes <= 0 or not policy_ref or not 0 < len(payload) <= max_bytes:
        raise ValueError("Invalid SD3 snapshot identity or byte budget")
    snapshot = msgspec.msgpack.decode(payload, type=_Snapshot)
    if snapshot.format != "sd3-state-v2" or snapshot.policy_ref != policy_ref:
        raise ValueError("SD3 snapshot format or policy does not match")
    if snapshot.torch_version != str(torch.__version__) or snapshot.diffusers_version != diffusers.__version__:
        raise ValueError("SD3 snapshot runtime versions do not match")
    return snapshot


def step_request_from_snapshot(payload: bytes | memoryview, *, policy_ref: str, max_bytes: int) -> OmniDiffusionRequest:
    """Read scheduler inputs without constructing the worker's tensors or RNG."""
    snapshot = _decode_snapshot(payload, policy_ref, max_bytes)
    projected = msgspec.msgpack.decode(snapshot.state, type=_RequestProjection)
    decoder = OmniMsgpackDecoder()
    sampling = OmniDiffusionSamplingParams.from_dict(decoder.decode(bytes(projected.sampling)))
    sampling.step_index = projected.request.step_index
    return OmniDiffusionRequest(
        prompt=decoder.decode(bytes(projected.request.prompt)),
        sampling_params=sampling,
        request_id=projected.request.request_id,
        external_req_id=projected.request.external_req_id,
        sampling_resolved=True,
    )


def restore_step_state(
    payload: bytes,
    *,
    policy_ref: str,
    device: torch.device,
    max_bytes: int,
    execution_profile: bytes,
    scheduler_cls: type[FlowMatchEulerDiscreteScheduler] = FlowMatchEulerDiscreteScheduler,
) -> StepRequestState:
    """Reconstruct under an already installed policy; never replay preparation."""
    snapshot = _decode_snapshot(payload, policy_ref, max_bytes)
    if snapshot.scheduler_type != f"{scheduler_cls.__module__}.{scheduler_cls.__qualname__}":
        raise ValueError("SD3 snapshot scheduler does not match the pipeline's scheduler contract")
    if snapshot.execution_profile != execution_profile:
        raise ValueError("SD3 snapshot execution profile does not match")
    values = _StateDecoder(device).decode(snapshot.state)
    sampling = OmniDiffusionSamplingParams.from_dict(values["sampling"])
    generators = []
    for item in values["generators"]:
        source = item["device"]
        if source not in ("cpu", device.type):
            raise ValueError("SD3 generator device backend is incompatible")
        generator = torch.Generator(device="cpu" if source == "cpu" else device)
        generator.set_state(item["state"])
        generators.append(generator)
    ordered_generators = [generators[index] for index in values["generator_indices"]]
    sampling.generator = (
        ordered_generators if values["generator_is_list"] else (ordered_generators[0] if ordered_generators else None)
    )
    scheduler = scheduler_cls.from_config(values["scheduler_config"])
    vars(scheduler).update(values["scheduler_state"])
    request = values["request"]
    request["attention_schedule"] = tuple(request["attention_schedule"])
    request["attention_sigma_schedule"] = tuple(request["attention_sigma_schedule"])
    state = StepRequestState(**request, sampling=sampling, scheduler=scheduler)
    if state.latents is None or not 0 <= state.step_index < state.total_steps:
        raise ValueError("SD3 snapshot has invalid latent state or progress")
    if state.scheduler.step_index is not None and state.scheduler.step_index != state.step_index:
        raise ValueError("SD3 scheduler and request progress do not match")
    return state
