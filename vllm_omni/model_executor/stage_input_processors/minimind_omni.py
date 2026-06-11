from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.inputs.data import OmniTokensPrompt

AUDIO_PAD_TOKEN_ID = 2049


def _as_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    return list(value)


def _payload_keys(mm: Any) -> tuple[list[str], list[str]]:
    if not isinstance(mm, dict):
        return [], []
    hidden = mm.get("hidden_states", {})
    hidden_keys = sorted(hidden.keys()) if isinstance(hidden, dict) else []
    return sorted(str(key) for key in mm.keys()), hidden_keys


def _has_explicit_bridge(mm: Any) -> bool:
    if not isinstance(mm, dict):
        return False
    hidden = mm.get("hidden_states", {})
    if isinstance(hidden, dict) and hidden.get("bridge") is not None:
        return True
    return mm.get("hidden_states.bridge") is not None


def _multimodal_output_for_talker(thinker_output: Any, output: Any) -> OmniPayload:
    completion_mm = getattr(output, "multimodal_output", None)
    request_mm = getattr(thinker_output, "multimodal_output", None)
    completion_keys, completion_hidden_keys = _payload_keys(completion_mm)
    request_keys, request_hidden_keys = _payload_keys(request_mm)

    if _has_explicit_bridge(completion_mm):
        return completion_mm

    raise RuntimeError(
        "MiniMind thinker2talker expected explicit bridge in completion multimodal_output. "
        f"completion_keys={completion_keys}; completion_hidden_keys={completion_hidden_keys}; "
        f"request_keys={request_keys}; request_hidden_keys={request_hidden_keys}"
    )


def _pick_bridge(mm: OmniPayload, expected_len: int) -> torch.Tensor:
    hidden = mm.get("hidden_states", {}) if isinstance(mm, dict) else {}
    bridge = hidden.get("bridge") if isinstance(hidden, dict) else None
    if bridge is None and isinstance(mm, dict):
        bridge = mm.get("hidden_states.bridge")

    if bridge is None:
        keys = sorted(mm.keys()) if isinstance(mm, dict) else []
        hidden_keys = sorted(hidden.keys()) if isinstance(hidden, dict) else []
        raise RuntimeError(
            "MiniMind thinker output does not contain explicit bridge hidden states for talker. "
            "Expected hidden_states.bridge. "
            f"Available keys: {keys}; hidden_states keys: {hidden_keys}"
        )

    if isinstance(bridge, list):
        bridge = bridge[0]
    if bridge.ndim == 3:
        bridge = bridge.reshape(-1, bridge.shape[-1])
    if expected_len > 0:
        bridge = bridge[-expected_len:]
    return bridge.detach().to(torch.float32)


def _align_ids_to_bridge(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    bridge_len: int,
) -> tuple[list[int], list[int], list[int]]:
    if bridge_len <= 0:
        all_text_ids = prompt_token_ids + output_token_ids
        return prompt_token_ids, output_token_ids, all_text_ids

    all_text_ids = prompt_token_ids + output_token_ids
    if len(all_text_ids) == bridge_len:
        return prompt_token_ids, output_token_ids, all_text_ids

    if len(all_text_ids) < bridge_len:
        return prompt_token_ids, output_token_ids, all_text_ids

    if len(prompt_token_ids) <= bridge_len:
        output_budget = max(0, bridge_len - len(prompt_token_ids))
        output_token_ids = output_token_ids[:output_budget]
        all_text_ids = prompt_token_ids + output_token_ids
        return prompt_token_ids, output_token_ids, all_text_ids

    prompt_token_ids = prompt_token_ids[-bridge_len:]
    return prompt_token_ids, [], prompt_token_ids


def thinker2talker(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data, streaming_context
    talker_inputs: list[OmniTokensPrompt] = []

    for thinker_output in source_outputs:
        output = thinker_output.outputs[0]
        prompt_token_ids = _as_list(getattr(thinker_output, "prompt_token_ids", []))
        output_token_ids = _as_list(getattr(output, "cumulative_token_ids", []))
        all_text_ids = prompt_token_ids + output_token_ids
        if not all_text_ids:
            all_text_ids = [AUDIO_PAD_TOKEN_ID]

        mm = _multimodal_output_for_talker(thinker_output, output)
        bridge = _pick_bridge(mm, len(all_text_ids))
        prompt_token_ids, output_token_ids, all_text_ids = _align_ids_to_bridge(
            prompt_token_ids,
            output_token_ids,
            int(bridge.shape[0]),
        )
        ids = {"prompt": prompt_token_ids, "output": output_token_ids, "all": all_text_ids}
        additional_information: OmniPayload = {
            "hidden_states": {"bridge": bridge},
            "ids": ids,
        }
        # Match upstream MiniMind stream_generate: the talker prefill consumes
        # only the original text prompt. Generated thinker tokens are consumed
        # later during talker decode via the full bridge hidden-state payload.
        talker_prompt_len = max(1, len(prompt_token_ids))
        talker_prompt_ids = [AUDIO_PAD_TOKEN_ID] * talker_prompt_len
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=talker_prompt_ids,
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
    del prompt, requires_multimodal_data, streaming_context
    code2wav_inputs: list[OmniTokensPrompt] = []

    for talker_output in source_outputs:
        request_id = getattr(talker_output, "request_id", None)
        output = talker_output.outputs[0]
        mm: OmniPayload = getattr(output, "multimodal_output", None) or {}
        codes = mm.get("codes", {}) if isinstance(mm, dict) else {}
        audio_codes = codes.get("audio") if isinstance(codes, dict) else None
        if not isinstance(audio_codes, torch.Tensor):
            raise TypeError(
                "MiniMind talker2code2wav expected multimodal_output['codes']['audio'] "
                f"to be a tensor for request {request_id!r}, got {type(audio_codes).__name__}."
            )
        if audio_codes.numel() == 0:
            raise ValueError(f"MiniMind talker2code2wav received empty audio codes for request {request_id!r}.")
        audio_codes = audio_codes.to(torch.long)
        if audio_codes.ndim != 2:
            raise ValueError(
                "MiniMind talker audio codes must have shape [frames, codebooks], "
                f"got {tuple(audio_codes.shape)} for request {request_id!r}."
            )
        num_code_layers = int(audio_codes.shape[-1])
        if num_code_layers <= 0:
            raise ValueError(
                "MiniMind talker audio codes must contain at least one codebook layer, "
                f"got {num_code_layers} for request {request_id!r}."
            )

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
