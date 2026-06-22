from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.inputs.data import OmniTokensPrompt

AUDIO_PAD_TOKEN_ID = 2049

# MiniMind emits per-step bridge states and codec rows, so full-payload
# connector accumulation must concatenate every tensor key along dim 0.
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()


def _as_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    return list(value)


def _multimodal_output_for_talker(output: Any) -> dict[str, Any]:
    multimodal_output = getattr(output, "multimodal_output", None)
    if not isinstance(multimodal_output, Mapping):
        raise TypeError(
            "MiniMind thinker2talker expected completion multimodal_output "
            f"to be a mapping, got {type(multimodal_output).__name__}."
        )
    return dict(multimodal_output)


def _pick_bridge(mm: dict[str, Any], expected_len: int) -> torch.Tensor:
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
    all_text_ids = prompt_token_ids + output_token_ids
    if bridge_len <= 0 or len(all_text_ids) <= bridge_len:
        return prompt_token_ids, output_token_ids, all_text_ids

    if len(prompt_token_ids) > bridge_len:
        prompt_token_ids = prompt_token_ids[-bridge_len:]
        return prompt_token_ids, [], prompt_token_ids

    output_token_ids = output_token_ids[: bridge_len - len(prompt_token_ids)]
    return prompt_token_ids, output_token_ids, prompt_token_ids + output_token_ids


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

        mm = _multimodal_output_for_talker(output)
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


def thinker2talker_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Build Talker placeholder prompts for synchronous connector transfer."""
    del prompt, requires_multimodal_data, streaming_context
    talker_inputs: list[OmniTokensPrompt] = []

    for thinker_output in source_outputs:
        output = thinker_output.outputs[0]
        prompt_token_ids = _as_list(getattr(thinker_output, "prompt_token_ids", []))
        output_token_ids = _as_list(getattr(output, "cumulative_token_ids", []))
        all_text_ids = prompt_token_ids + output_token_ids or [AUDIO_PAD_TOKEN_ID]

        mm = _multimodal_output_for_talker(output)
        bridge = _pick_bridge(mm, len(all_text_ids))
        prompt_token_ids, _, _ = _align_ids_to_bridge(
            prompt_token_ids,
            output_token_ids,
            int(bridge.shape[0]),
        )
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[AUDIO_PAD_TOKEN_ID] * max(1, len(prompt_token_ids)),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def thinker2talker_full_payload(
    transfer_manager: Any,
    pooling_output: Mapping[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Build the complete Thinker payload for synchronous connector transfer."""
    del transfer_manager
    if not isinstance(pooling_output, Mapping):
        return None

    bridge = pooling_output.get("hidden_states.bridge")
    if bridge is None:
        hidden_states = pooling_output.get("hidden_states", {})
        if isinstance(hidden_states, Mapping):
            bridge = hidden_states.get("bridge")
    if not isinstance(bridge, torch.Tensor):
        return None

    prompt_token_ids = _as_list(getattr(request, "prompt_token_ids", []))
    output_token_ids = _as_list(getattr(request, "output_token_ids", []))
    all_text_ids = prompt_token_ids + output_token_ids or [AUDIO_PAD_TOKEN_ID]
    bridge = _pick_bridge({"hidden_states": {"bridge": bridge}}, len(all_text_ids))
    prompt_token_ids, output_token_ids, all_text_ids = _align_ids_to_bridge(
        prompt_token_ids,
        output_token_ids,
        int(bridge.shape[0]),
    )
    return {
        "hidden_states": {"bridge": bridge.cpu()},
        "ids": {
            "prompt": prompt_token_ids,
            "output": output_token_ids,
            "all": all_text_ids,
        },
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def _audio_codes_tensor(audio_codes: Any, request_id: Any) -> torch.Tensor:
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
    if audio_codes.shape[-1] <= 0:
        raise ValueError(
            "MiniMind talker audio codes must contain at least one codebook layer, "
            f"got {audio_codes.shape[-1]} for request {request_id!r}."
        )
    return audio_codes


def _flatten_audio_codes(audio_codes: torch.Tensor) -> torch.Tensor:
    """Convert frame-major codes to Mimi's flattened codebook-major layout."""
    return audio_codes.transpose(0, 1).cpu().contiguous().reshape(-1)


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
        mm = getattr(output, "multimodal_output", None)
        codes = mm.get("codes", {}) if isinstance(mm, Mapping) else {}
        audio_codes = _audio_codes_tensor(
            codes.get("audio") if isinstance(codes, Mapping) else None,
            request_id,
        )

        # Mimi expects codebook-major [codebooks, frames] flattened for Code2Wav.
        codec_codes = _flatten_audio_codes(audio_codes).tolist()
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs


def talker2code2wav_token_only(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    """Build Code2Wav placeholders sized for connector-delivered codec IDs."""
    del prompt, requires_multimodal_data, streaming_context
    code2wav_inputs: list[OmniTokensPrompt] = []

    for talker_output in source_outputs:
        request_id = getattr(talker_output, "request_id", None)
        output = talker_output.outputs[0]
        mm = getattr(output, "multimodal_output", None)
        codes = mm.get("codes", {}) if isinstance(mm, Mapping) else {}
        audio_codes = _audio_codes_tensor(
            codes.get("audio") if isinstance(codes, Mapping) else None,
            request_id,
        )
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * int(audio_codes.numel()),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs


def talker2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: Mapping[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Build the complete Talker codec payload for synchronous transfer."""
    del transfer_manager
    if not isinstance(pooling_output, Mapping):
        return None

    audio_codes = pooling_output.get("codes.audio")
    if audio_codes is None:
        codes = pooling_output.get("codes", {})
        if isinstance(codes, Mapping):
            audio_codes = codes.get("audio")
    if audio_codes is None:
        return None

    request_id = getattr(request, "request_id", None)
    audio_codes = _audio_codes_tensor(audio_codes, request_id)
    return {
        "codes": {"audio": _flatten_audio_codes(audio_codes)},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }
