# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt


def _compute_talker_prompt_ids_length(info, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    thinker_sequences = torch.tensor(info["ids.all"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(info["ids.prompt"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    im_start_indexes = torch.cat(
        [
            torch.nonzero(input_ids[0] == im_start_token_id).squeeze(1),
            torch.tensor([thinker_sequences.shape[-1]], device=input_ids.device, dtype=input_ids.dtype),
        ],
        dim=0,
    )

    sum_user_len = 0
    assistant_len = 0
    for i in range(len(im_start_indexes) - 1):
        s = int(im_start_indexes[i].item())
        e = int(im_start_indexes[i + 1].item())
        role = int(input_ids[0, s + 1].item())
        if role == system_token_id:
            continue
        elif role == user_token_id:
            sum_user_len += e - s
        elif role == assistant_token_id and i == len(im_start_indexes) - 2:
            assistant_len += 9  # 3 + 4 + 1 + 1
        else:
            pass

    return sum_user_len + assistant_len


# =========================
# Common helpers
# =========================


def _ensure_list(x):
    """Convert ConstantList / tensor-like to Python list."""
    if hasattr(x, "_x"):
        return list(x._x)
    elif not isinstance(x, list):
        return x
    return list(x)


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


# =========================
# Thinker -> Talker
# =========================


def thinker2talker_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> list[dict[str, Any]]:
    """
    Process thinker outputs to create talker inputs.
    1. thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information
    """

    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk[request_id]
    if chunk_id == 0:
        all_token_ids = request.all_token_ids  # prefill + decode
        prompt_token_ids = request.prompt_token_ids
        # Convert ConstantList to regular list for OmniSerializer serialization
        all_token_ids = _ensure_list(all_token_ids)
        prompt_token_ids = _ensure_list(prompt_token_ids)
        talker_additional_info = {
            "embed.prefill": pooling_output.get("hidden_states.layer_0").detach().cpu(),
            "hidden_states.output": pooling_output.get("hidden_states.layer_24").detach().cpu(),
            "ids.all": all_token_ids,
            "ids.prompt": prompt_token_ids,
            # Provide thinker-side TTS token embeddings for talker projection
            "embed.tts_bos": pooling_output.get("embed.tts_bos").detach().cpu(),
            "embed.tts_eos": pooling_output.get("embed.tts_eos").detach().cpu(),
            "embed.tts_pad": pooling_output.get("embed.tts_pad").detach().cpu(),
            "meta.finished": torch.tensor(is_finished, dtype=torch.bool),
        }
        if transfer_manager.request_payload.get(request_id) is None:
            if not is_finished:
                transfer_manager.request_payload[request_id] = talker_additional_info
                return None
        else:
            save_payload = transfer_manager.request_payload.pop(request_id)
            talker_additional_info["embed.prefill"] = torch.cat(
                (
                    save_payload.get("embed.prefill"),
                    talker_additional_info.get("embed.prefill"),
                ),
                dim=0,
            )
            talker_additional_info["hidden_states.output"] = torch.cat(
                (save_payload.get("hidden_states.output"), talker_additional_info.get("hidden_states.output")),
                dim=0,
            )
    else:
        output_token_ids = request.output_token_ids
        # Convert ConstantList to regular list for OmniSerializer serialization
        output_token_ids = _ensure_list(output_token_ids)

        talker_additional_info = {
            "meta.finished": torch.tensor(is_finished, dtype=torch.bool),
        }
        if output_token_ids:
            talker_additional_info["meta.override_keys"] = ["embed.decode", "ids.output"]
            talker_additional_info["embed.decode"] = pooling_output.get("hidden_states.layer_0").detach().cpu()
            talker_additional_info["ids.output"] = output_token_ids
        else:
            # When prefilling a chunked thinker, thinker_hidden_states needs to be updated.
            talker_additional_info["embed.prefill"] = pooling_output.get("hidden_states.layer_0").detach().cpu()
            talker_additional_info["hidden_states.output"] = pooling_output.get("hidden_states.layer_24").detach().cpu()
    return talker_additional_info


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process thinker outputs to create talker inputs.

    Workflow:
    1. Extract thinker's text generation outputs (token IDs + hidden states)
    2. Split hidden states into: prompt embeddings + generated embeddings
    3. Package for talker with additional information

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [0] for thinker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for talker stage
    """
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    # Process each thinker output
    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]

        info = {
            "embed.prefill": output.multimodal_output["hidden_states.layer_0"]
            .detach()
            .to(device=device, dtype=torch.float),
            "hidden_states.output": output.multimodal_output["hidden_states.layer_24"]
            .detach()
            .to(device=device, dtype=torch.float),
            "ids.all": (thinker_output.prompt_token_ids + output.token_ids),  # the thinker_sequences is the whole ids
            "ids.prompt": thinker_output.prompt_token_ids,
            # Provide thinker-side TTS token embeddings for talker projection
            "embed.tts_bos": output.multimodal_output["embed.tts_bos"].detach().to(device=device, dtype=torch.float),
            "embed.tts_eos": output.multimodal_output["embed.tts_eos"].detach().to(device=device, dtype=torch.float),
            "embed.tts_pad": output.multimodal_output["embed.tts_pad"].detach().to(device=device, dtype=torch.float),
        }

        prompt_len = _compute_talker_prompt_ids_length(info, device=device)

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


# =========================
# Talker -> Code2Wav
# =========================


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
):
    """
    Pooling version.
    """
    if "codes.audio" not in pooling_output:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    code_predictor_codes = pooling_output["codes.audio"]

    if code_predictor_codes is None:
        return None
    if isinstance(code_predictor_codes, torch.Tensor):
        if code_predictor_codes.numel() == 0:
            return None
    elif hasattr(code_predictor_codes, "__len__"):
        if len(code_predictor_codes) == 0:
            return None

    if isinstance(code_predictor_codes, torch.Tensor):
        if not code_predictor_codes.any():
            return None
    else:
        code_tensor = torch.tensor(code_predictor_codes, dtype=torch.long)
        if not code_tensor.any():
            return None

    codec_codes = code_predictor_codes.to(torch.long).transpose(0, 1).cpu().to(torch.long).reshape(-1).tolist()
    if sum(codec_codes) == 0:
        return None

    request_id = request.external_req_id
    transfer_manager.code_prompt_token_ids[request_id].append(codec_codes)
    length = len(transfer_manager.code_prompt_token_ids[request_id])
    chunk_length = length % chunk_size_config
    if chunk_length != 0 and not is_finished:
        return None

    context_length = chunk_length if chunk_length != 0 else chunk_size_config
    # ensure left context does not exceed available length
    left_context_size = max(0, min(length - context_length, left_context_size_config))
    end_index = min(length, left_context_size + context_length)

    codes = (
        torch.tensor(transfer_manager.code_prompt_token_ids[request_id][-end_index:])
        .transpose(0, 1)
        .reshape(-1)
        .tolist()
    )

    info = {
        "codes.audio": codes,
        "meta.left_context_size": left_context_size,
        "meta.finished": torch.tensor(is_finished, dtype=torch.bool),
    }
    return info


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    Process talker outputs to create code2wav inputs.

    Workflow:
    1. Extract talker's codec code outputs (8-layer RVQ codes)
    2. Flatten codes for code2wav input
    3. Package for code2wav stage

    Args:
        stage_list: List of stage objects
        engine_input_source: Source stage IDs (typically [1] for talker)
        prompt: Original prompt data
        requires_multimodal_data: Whether multimodal data is required

    Returns:
        List of OmniTokensPrompt for code2wav stage
    """
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    # Process each talker output
    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        seq_len = len(output.token_ids) - 1
        # Extract codec codes from talker output
        # Expected shape: [8, seq_len] (8-layer RVQ codes)
        codec_codes = (
            output.multimodal_output["codes.audio"][-seq_len:]
            .to(torch.long)
            .transpose(0, 1)
            .cpu()
            .to(torch.long)
            .reshape(-1)
            .tolist()
        )  # 16, seq_len
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
