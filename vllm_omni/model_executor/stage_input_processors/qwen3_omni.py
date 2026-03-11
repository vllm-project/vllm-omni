# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
"""Stage input processor for Qwen3 Omni MoE: Thinker → Talker transition."""

import logging
from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.platforms import current_platform

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)

# Pooling output layer keys: "0" = word embedding, "24" = accept_hidden_layer
_EMBED_LAYER_KEY = "0"
_HIDDEN_LAYER_KEY = "24"


def _compute_talker_prompt_ids_length(info, device: torch.device | str = "cuda") -> int:
    im_start_token_id = 151644
    system_token_id = 8948
    user_token_id = 872
    assistant_token_id = 77091

    thinker_sequences = torch.tensor(info["thinker_sequences"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

    input_ids = torch.tensor(info["thinker_input_ids"], dtype=torch.long, device=device).unsqueeze(0)  # [1, T]

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
    """Process thinker outputs to create talker inputs (async chunk mode)."""

    request_id = request.external_req_id
    chunk_id = transfer_manager.put_req_chunk[request_id]
    if chunk_id == 0:
        all_token_ids = _ensure_list(request.all_token_ids)
        prompt_token_ids = _ensure_list(request.prompt_token_ids)
        talker_additional_info = {
            "thinker_prefill_embeddings": pooling_output.get(_EMBED_LAYER_KEY).detach().cpu(),
            "thinker_hidden_states": pooling_output.get(_HIDDEN_LAYER_KEY).detach().cpu(),
            "thinker_sequences": all_token_ids,
            "thinker_input_ids": prompt_token_ids,
            "tts_bos_embed": pooling_output.get("tts_bos_embed").detach().cpu(),
            "tts_eos_embed": pooling_output.get("tts_eos_embed").detach().cpu(),
            "tts_pad_embed": pooling_output.get("tts_pad_embed").detach().cpu(),
            "finished": torch.tensor(is_finished, dtype=torch.bool),
        }
        if transfer_manager.request_payload.get(request_id) is None:
            if not is_finished:
                transfer_manager.request_payload[request_id] = talker_additional_info
                return None
        else:
            save_payload = transfer_manager.request_payload.pop(request_id)
            talker_additional_info["thinker_prefill_embeddings"] = torch.cat(
                (
                    save_payload.get("thinker_prefill_embeddings"),
                    talker_additional_info.get("thinker_prefill_embeddings"),
                ),
                dim=0,
            )
            talker_additional_info["thinker_hidden_states"] = torch.cat(
                (save_payload.get("thinker_hidden_states"), talker_additional_info.get("thinker_hidden_states")),
                dim=0,
            )
    else:
        output_token_ids = _ensure_list(request.output_token_ids)

        talker_additional_info = {
            "thinker_prefill_embeddings": pooling_output.get(_EMBED_LAYER_KEY).detach().cpu(),
            "thinker_hidden_states": pooling_output.get(_HIDDEN_LAYER_KEY).detach().cpu(),
            "thinker_sequences": output_token_ids,
            "finished": torch.tensor(is_finished, dtype=torch.bool),
        }
        if output_token_ids:
            talker_additional_info["override_keys"] = ["thinker_decode_embeddings", "thinker_output_token_ids"]
            talker_additional_info["thinker_decode_embeddings"] = pooling_output.get("0").detach().cpu()
            talker_additional_info["thinker_output_token_ids"] = output_token_ids
        else:
            talker_additional_info["thinker_prefill_embeddings"] = pooling_output.get(_EMBED_LAYER_KEY).detach().cpu()
            talker_additional_info["thinker_hidden_states"] = pooling_output.get(_HIDDEN_LAYER_KEY).detach().cpu()
    return talker_additional_info


def _get_prefill_stage(stage_list: list[Any], source_stage_id: int) -> Any | None:
    """Return the preceding prefill stage if PD disaggregation is active."""
    if source_stage_id <= 0:
        return None
    source_stage = stage_list[source_stage_id]
    if not getattr(source_stage, "is_decode_only", False):
        return None
    prev_stage = stage_list[source_stage_id - 1]
    if getattr(prev_stage, "is_prefill_only", False) and prev_stage.engine_outputs is not None:
        return prev_stage
    return None


def _merge_pd_embeddings(
    decode_emb: torch.Tensor,
    decode_hid: torch.Tensor,
    prefill_mm: dict[str, Any],
    device: torch.device,
    expected_total: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge prefill prompt embeddings with decode generated embeddings.

    Computes overlap = prefill_len + decode_len - expected_total, then
    concatenates: prefill + decode[overlap:].
    """
    try:
        p_emb = prefill_mm[_EMBED_LAYER_KEY].detach().to(device=device, dtype=torch.float)
        p_hid = prefill_mm[_HIDDEN_LAYER_KEY].detach().to(device=device, dtype=torch.float)
    except (KeyError, AttributeError, TypeError):
        return decode_emb, decode_hid

    if p_emb.shape[0] == 0 or decode_emb.shape[0] == 0:
        return decode_emb, decode_hid

    raw_total = p_emb.shape[0] + decode_emb.shape[0]
    overlap = max(0, raw_total - expected_total) if expected_total is not None else 0

    merged_emb = torch.cat([p_emb, decode_emb[overlap:]], dim=0)
    merged_hid = torch.cat([p_hid, decode_hid[overlap:]], dim=0)
    return merged_emb, merged_hid


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Process thinker outputs to create talker inputs.

    In PD mode, merges prefill prompt embeddings with decode embeddings
    so the talker receives the complete sequence.
    """
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    device = torch.device(current_platform.device_type)

    source_stage_id = engine_input_source[0]
    prefill_stage = _get_prefill_stage(stage_list, source_stage_id)

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]

        decode_emb = output.multimodal_output[_EMBED_LAYER_KEY].detach().to(device=device, dtype=torch.float)
        decode_hid = output.multimodal_output[_HIDDEN_LAYER_KEY].detach().to(device=device, dtype=torch.float)

        expected_total = len(thinker_output.prompt_token_ids) + len(output.token_ids)

        # Merge prefill embeddings in PD mode
        if prefill_stage is not None:
            try:
                prefill_eos = prefill_stage.engine_outputs
                prefill_eo = prefill_eos[min(i, len(prefill_eos) - 1)]
                prefill_mm = prefill_eo.outputs[0].multimodal_output
                decode_emb, decode_hid = _merge_pd_embeddings(
                    decode_emb, decode_hid, prefill_mm, device, expected_total=expected_total,
                )
            except Exception as exc:
                logger.warning("[PD] Could not merge prefill embeddings: %s", exc)

        def _tts(key: str) -> torch.Tensor:
            val = output.multimodal_output.get(key)
            if val is None and prefill_stage is not None:
                try:
                    val = prefill_stage.engine_outputs[0].outputs[0].multimodal_output.get(key)
                except Exception:
                    pass
            return val.detach().to(device=device, dtype=torch.float) if val is not None else None

        info = {
            "thinker_prefill_embeddings": decode_emb,
            "thinker_hidden_states": decode_hid,
            "thinker_sequences": thinker_output.prompt_token_ids + output.token_ids,
            "thinker_input_ids": thinker_output.prompt_token_ids,
            "tts_bos_embed": _tts("tts_bos_embed"),
            "tts_eos_embed": _tts("tts_eos_embed"),
            "tts_pad_embed": _tts("tts_pad_embed"),
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
    """Pooling version."""
    if "code_predictor_codes" not in pooling_output:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size_config = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 25))

    code_predictor_codes = pooling_output["code_predictor_codes"]

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
    left_context_size = max(0, min(length - context_length, left_context_size_config))
    end_index = min(length, left_context_size + context_length)

    codes = (
        torch.tensor(transfer_manager.code_prompt_token_ids[request_id][-end_index:])
        .transpose(0, 1)
        .reshape(-1)
        .tolist()
    )

    info = {
        "code_predictor_codes": codes,
        "left_context_size": left_context_size,
        "finished": torch.tensor(is_finished, dtype=torch.bool),
    }
    return info


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Process talker outputs to create code2wav inputs."""
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        seq_len = len(output.token_ids) - 1
        # [num_quantizers, seq_len] -> [seq_len, num_quantizers] -> flat
        codec_codes = (
            output.multimodal_output["code_predictor_codes"][:, -seq_len:]
            .to(torch.long)
            .transpose(0, 1)
            .cpu()
            .to(torch.long)
            .reshape(-1)
            .tolist()
        )
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
