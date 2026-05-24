from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.inputs.data import OmniTokensPrompt

AUDIO_PAD_TOKEN_ID = 2049
logger = init_logger(__name__)


def _is_engine_input_source(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(v, int) for v in value)


def _normalise_source_outputs(
    source_outputs: Any,
    prompt: Any,
    requires_multimodal_data: Any,
    streaming_context: Any,
) -> tuple[list[Any], Any, bool, Any]:
    if _is_engine_input_source(prompt):
        source_outputs = _validate_stage_inputs(source_outputs, prompt)
        prompt, requires_multimodal_data, streaming_context = (
            requires_multimodal_data,
            bool(streaming_context),
            None,
        )
    return source_outputs, prompt, bool(requires_multimodal_data), streaming_context


def _validate_stage_inputs(stage_list: list[Any], engine_input_source: list[int]):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    outputs = stage_list[source_stage_id].engine_outputs
    if outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
    return outputs


def _as_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    return list(value)


def _pick_bridge(mm: OmniPayload, expected_len: int) -> torch.Tensor:
    hidden = mm.get("hidden_states", {}) if isinstance(mm, dict) else {}
    bridge = hidden.get("bridge") if isinstance(hidden, dict) else None
    if bridge is None and isinstance(mm, dict):
        bridge = mm.get("hidden_states.bridge")
    if bridge is None and isinstance(hidden, dict):
        layers = hidden.get("layers", {})
        if layers:
            bridge = layers[max(layers.keys())]
    if bridge is None and isinstance(mm, dict):
        flat_layers = {
            int(key.removeprefix("hidden_states.layer_")): value
            for key, value in mm.items()
            if key.startswith("hidden_states.layer_")
        }
        if flat_layers:
            bridge = flat_layers[max(flat_layers.keys())]
    if bridge is None:
        latent = mm.get("latent") if isinstance(mm, dict) else None
        bridge = latent
    if bridge is None and isinstance(mm, dict):
        bridge = mm.get("hidden")
    if bridge is None:
        keys = sorted(mm.keys()) if isinstance(mm, dict) else []
        raise RuntimeError(
            f"MiniMind thinker output does not contain bridge hidden states for talker. Available keys: {keys}"
        )
    if isinstance(bridge, list):
        bridge = bridge[0]
    if bridge.ndim == 3:
        bridge = bridge.reshape(-1, bridge.shape[-1])
    if expected_len > 0:
        bridge = bridge[-expected_len:]
    return bridge.detach().to(torch.float32)


def thinker2talker(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    thinker_outputs, prompt, _requires_multimodal_data, _streaming_context = _normalise_source_outputs(
        source_outputs,
        prompt,
        requires_multimodal_data,
        streaming_context,
    )
    talker_inputs: list[OmniTokensPrompt] = []

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = _as_list(getattr(thinker_output, "prompt_token_ids", []))
        output_token_ids = _as_list(getattr(output, "cumulative_token_ids", []))
        all_text_ids = prompt_token_ids + output_token_ids
        if not all_text_ids:
            all_text_ids = [AUDIO_PAD_TOKEN_ID]

        mm: OmniPayload = getattr(output, "multimodal_output", None) or {}
        bridge = _pick_bridge(mm, len(all_text_ids))
        additional_information: OmniPayload = {
            "hidden_states": {"bridge": bridge},
            "ids": {"prompt": prompt_token_ids, "output": output_token_ids, "all": all_text_ids},
        }

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[AUDIO_PAD_TOKEN_ID] * len(all_text_ids),
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def talker2code2wav(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    talker_outputs, _prompt, _requires_multimodal_data, _streaming_context = _normalise_source_outputs(
        source_outputs,
        prompt,
        requires_multimodal_data,
        streaming_context,
    )
    code2wav_inputs: list[OmniTokensPrompt] = []

    for talker_output in talker_outputs:
        if not getattr(talker_output, "finished", True):
            continue
        output = talker_output.outputs[0]
        mm: OmniPayload = getattr(output, "multimodal_output", None) or {}
        codes = mm.get("codes", {}) if isinstance(mm, dict) else {}
        audio_codes = codes.get("audio") if isinstance(codes, dict) else None
        if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
            continue
        audio_codes = audio_codes.to(torch.long)
        if audio_codes.ndim != 2:
            raise ValueError(
                f"MiniMind talker audio codes must have shape [frames, codebooks], got {tuple(audio_codes.shape)}"
            )
        num_code_layers = int(audio_codes.shape[-1])
        if num_code_layers <= 0:
            continue

        # Mimi expects codebook-major [codebooks, frames] flattened for Code2Wav.
        codec_codes = audio_codes.transpose(0, 1).cpu().contiguous().reshape(-1).tolist()
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs


def _ids_from_request(request: Any) -> tuple[list[int], list[int], list[int]]:
    prompt_token_ids = _as_list(getattr(request, "prompt_token_ids", []))
    output_token_ids = _as_list(getattr(request, "output_token_ids", []))
    all_token_ids = _as_list(getattr(request, "all_token_ids", None))
    if not all_token_ids:
        all_token_ids = prompt_token_ids + output_token_ids
    if not output_token_ids and len(all_token_ids) >= len(prompt_token_ids):
        output_token_ids = all_token_ids[len(prompt_token_ids) :]
    return prompt_token_ids, output_token_ids, all_token_ids


def _finished_tensor(is_finished: bool) -> torch.Tensor:
    return torch.tensor(bool(is_finished), dtype=torch.bool)


def thinker2talker_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
    is_finished: bool = False,
) -> OmniPayload | None:
    del transfer_manager
    if not isinstance(pooling_output, dict):
        return None

    prompt_token_ids, output_token_ids, all_token_ids = _ids_from_request(request)
    expected_len = len(all_token_ids) or len(prompt_token_ids) + len(output_token_ids)
    bridge = _pick_bridge(pooling_output, expected_len)
    if not all_token_ids:
        all_token_ids = [AUDIO_PAD_TOKEN_ID] * int(bridge.shape[0])

    return {
        "hidden_states": {"bridge": bridge.cpu()},
        "ids": {
            "prompt": prompt_token_ids,
            "output": output_token_ids,
            "all": all_token_ids,
        },
        "meta": {
            "finished": _finished_tensor(is_finished),
            "next_stage_prompt_len": len(all_token_ids),
        },
    }


def talker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
    is_finished: bool = False,
) -> OmniPayload | None:
    del transfer_manager
    if not isinstance(pooling_output, dict):
        return None

    codes = pooling_output.get("codes", {})
    audio_codes = codes.get("audio") if isinstance(codes, dict) else None
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        if is_finished:
            return {"meta": {"finished": _finished_tensor(True)}}
        return None

    audio_codes = audio_codes.to(dtype=torch.long)
    if audio_codes.ndim == 1:
        audio_codes = audio_codes.reshape(1, -1)
    if audio_codes.ndim != 2:
        raise ValueError(
            f"MiniMind talker audio codes must have shape [frames, codebooks], got {tuple(audio_codes.shape)}"
        )

    # Code2Wav receives Mimi codes flattened as [codebook0_frames, codebook1_frames, ...].
    codec_codes = audio_codes.transpose(0, 1).cpu().contiguous().reshape(-1).tolist()
    return {
        "codes": {"audio": codec_codes},
        "code_predictor_codes": codec_codes,
        "meta": {"finished": _finished_tensor(is_finished)},
    }
