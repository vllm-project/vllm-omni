from __future__ import annotations

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.inputs.data import OmniTokensPrompt

AUDIO_PAD_TOKEN_ID = 2049


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
    bridge = hidden.get("bridge")
    if bridge is None:
        layers = hidden.get("layers", {})
        if layers:
            bridge = layers[max(layers.keys())]
    if bridge is None:
        latent = mm.get("latent") if isinstance(mm, dict) else None
        bridge = latent
    if bridge is None:
        raise RuntimeError("MiniMind thinker output does not contain bridge hidden states for talker.")
    if isinstance(bridge, list):
        bridge = bridge[0]
    if bridge.ndim == 3:
        bridge = bridge.reshape(-1, bridge.shape[-1])
    return bridge[-expected_len:].detach().to(torch.float32)


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
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
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[OmniTokensPrompt]:
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
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
                f"MiniMind talker audio codes must have shape [frames, codebooks], "
                f"got {tuple(audio_codes.shape)}"
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
