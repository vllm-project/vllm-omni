from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

_VOXCPM_LATENT_MAGIC = 131071


def _serialize_latent_to_codes(latent: Any) -> list[int]:
    latent_tensor = latent if isinstance(latent, torch.Tensor) else torch.as_tensor(latent)
    latent_tensor = latent_tensor.detach().cpu().contiguous()
    if latent_tensor.ndim == 3:
        if latent_tensor.shape[0] != 1:
            raise ValueError(f"Expected batch=1 latent tensor, got shape={tuple(latent_tensor.shape)}")
        latent_tensor = latent_tensor.squeeze(0)
    if latent_tensor.ndim != 2:
        raise ValueError(f"Unsupported latent_audio_feat shape for async chunk: {tuple(latent_tensor.shape)}")
    latent_dim, time_dim = int(latent_tensor.shape[0]), int(latent_tensor.shape[1])
    packed = latent_tensor.to(torch.bfloat16).contiguous().view(torch.uint16).reshape(-1).to(torch.int32)
    return [_VOXCPM_LATENT_MAGIC, latent_dim, time_dim, *packed.tolist()]


def _coerce_finished_flag(value: Any) -> bool:
    """Normalize VoxCPM async-chunk finished markers to a Python bool."""
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"finished tensor must be scalar, got shape={tuple(value.shape)}")
        return bool(value.detach().cpu().item())
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        if len(value) != 1:
            raise ValueError(f"finished container must have one element, got len={len(value)}")
        return _coerce_finished_flag(value[0])
    return bool(value)


def latent2vae(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data

    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")

    source_outputs = stage_list[source_stage_id].engine_outputs
    if source_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    vae_inputs: list[OmniTokensPrompt] = []
    for source_output in source_outputs:
        output = source_output.outputs[0]
        multimodal_output = getattr(output, "multimodal_output", None)
        if not isinstance(multimodal_output, dict) or "latent_audio_feat" not in multimodal_output:
            raise ValueError(
                "VoxCPM latent stage output missing 'latent_audio_feat'. "
                f"request_id={getattr(source_output, 'request_id', None)}"
            )

        additional_information = {
            "latent_audio_feat": multimodal_output["latent_audio_feat"],
        }
        if "sr" in multimodal_output:
            additional_information["sample_rate"] = [int(multimodal_output["sr"])]

        vae_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return vae_inputs


def latent2vae_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> dict[str, Any] | None:
    """Stage-0 latent → stage-1 VAE under ``async_chunk`` (connector payload)."""
    # Kept for callback signature compatibility with OmniChunkTransferAdapter.
    _ = transfer_manager
    finished_request = _coerce_finished_flag(is_finished)
    if callable(getattr(request, "is_finished", None)):
        finished_request = finished_request or _coerce_finished_flag(request.is_finished())
    if not isinstance(pooling_output, dict):
        if finished_request:
            return {
                "code_predictor_codes": [],
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        return None

    latent = pooling_output.get("latent_audio_feat")
    if isinstance(latent, torch.Tensor) and latent.numel() == 0:
        latent = None

    if latent is None:
        if finished_request:
            return {
                "code_predictor_codes": [],
                "finished": torch.tensor(True, dtype=torch.bool),
            }
        return None

    serialized_codes = _serialize_latent_to_codes(latent)
    out: dict[str, Any] = {
        "code_predictor_codes": serialized_codes,
        "finished": torch.tensor(finished_request, dtype=torch.bool),
    }
    return out
