from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

# Special token ids from the MiniCPM tokenizer config.
TTS_BOS_ID = 151703
TTS_EOS_ID = 151704

logger = init_logger(__name__)


def _validate_stage_inputs(stage_list: list[Any], engine_input_source: list[int]) -> Any:
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


def _extract_token_hidden_states(latent: Any, *, expected_seq_len: int | None = None) -> torch.Tensor:
    """Normalize thinker latent captures into [seq_len, hidden_size]."""
    if isinstance(latent, torch.Tensor):
        if latent.ndim == 2:
            return latent
        if latent.ndim == 3:
            # Handle either [seq, layers, hidden] or [layers, seq, hidden].
            cand_seq_layers = latent[:, -1, :]
            cand_layers_seq = latent[-1]
            if expected_seq_len is not None:
                exp = int(expected_seq_len)
                if abs(int(cand_layers_seq.shape[0]) - exp) < abs(int(cand_seq_layers.shape[0]) - exp):
                    return cand_layers_seq
                return cand_seq_layers
            return cand_seq_layers if latent.shape[0] >= latent.shape[1] else cand_layers_seq
        if latent.ndim == 4:
            # Handle [seq, layers, batch, hidden] or [layers, batch, seq, hidden].
            cand_seq_layers = latent[:, -1, 0, :]
            cand_layers_seq = latent[-1, 0, :, :]
            if expected_seq_len is not None:
                exp = int(expected_seq_len)
                if abs(int(cand_layers_seq.shape[0]) - exp) < abs(int(cand_seq_layers.shape[0]) - exp):
                    return cand_layers_seq
                return cand_seq_layers
            return cand_seq_layers if latent.shape[0] >= latent.shape[1] else cand_layers_seq
        raise ValueError(f"Unsupported latent tensor shape: {tuple(latent.shape)}")

    if isinstance(latent, (list, tuple)):
        rows: list[torch.Tensor] = []
        for token_layers in latent:
            layer_value = token_layers[-1] if isinstance(token_layers, (list, tuple)) else token_layers
            if not isinstance(layer_value, torch.Tensor):
                raise TypeError(f"Unsupported latent element type: {type(layer_value)}")
            if layer_value.ndim == 1:
                rows.append(layer_value)
            elif layer_value.ndim == 2:
                rows.append(layer_value[0])
            elif layer_value.ndim == 3:
                rows.append(layer_value[-1, 0])
            else:
                raise ValueError(f"Unsupported latent element shape: {tuple(layer_value.shape)}")
        return torch.stack(rows, dim=0)

    raise TypeError(f"Unsupported latent type: {type(latent)}")


def thinker2talker(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    from vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5_talker import (
        MiniCPMO4_5TalkerForConditionalGeneration,
    )

    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs: list[OmniTokensPrompt] = []

    for thinker_output in thinker_outputs:
        output = thinker_output.outputs[0]

        prompt_ids = list(thinker_output.prompt_token_ids)
        gen_ids = list(output.token_ids)
        full_sequence = prompt_ids + gen_ids

        hidden_states = _extract_token_hidden_states(
            output.multimodal_output["latent"],
            expected_seq_len=len(full_sequence),
        ).detach()

        tts_bos_idx = None
        tts_eos_idx = None
        for i, tok in enumerate(full_sequence):
            if tok == TTS_BOS_ID:
                tts_bos_idx = i + 1
                tts_eos_idx = None
            elif tok == TTS_EOS_ID and tts_bos_idx is not None:
                tts_eos_idx = i
                break

        if tts_bos_idx is None:
            raise ValueError("MiniCPM thinker output is missing <|tts_bos|>.")
        if tts_eos_idx is None:
            tts_eos_idx = len(full_sequence)
        usable_end = min(tts_eos_idx, int(hidden_states.shape[0]))
        if usable_end <= tts_bos_idx:
            raise ValueError(
                "MiniCPM thinker latent span does not cover the TTS content tokens: "
                f"latent={tuple(hidden_states.shape)} tts_range=({tts_bos_idx}, {tts_eos_idx})"
            )

        llm_tokens = torch.tensor(full_sequence[tts_bos_idx:usable_end], dtype=torch.long)
        tts_hidden_states = hidden_states[tts_bos_idx:usable_end].to(torch.float32)

        info = {
            "llm_tokens": llm_tokens,
            "tts_hidden_states": tts_hidden_states,
        }
        prompt_len = MiniCPMO4_5TalkerForConditionalGeneration.estimate_prompt_len_from_additional_information(info)

        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def _extract_ref_audio_from_prompt(prompt: Any, index: int = 0) -> Any:
    if prompt is None:
        return None
    p = prompt[index] if isinstance(prompt, list) and index < len(prompt) else prompt
    if p is None or not isinstance(p, dict):
        return None
    add_info = p.get("additional_information")
    if not isinstance(add_info, dict):
        return None
    raw_ref_audio = add_info.get("ref_audio")
    if raw_ref_audio is None:
        return None
    return _canonicalize_ref_audio(raw_ref_audio)


def _canonicalize_ref_audio(raw_ref_audio: Any) -> dict[str, Any]:
    if isinstance(raw_ref_audio, list) and len(raw_ref_audio) == 1:
        raw_ref_audio = raw_ref_audio[0]

    if isinstance(raw_ref_audio, dict):
        wav = raw_ref_audio.get("wav")
        sr = raw_ref_audio.get("sr")
    elif isinstance(raw_ref_audio, (list, tuple)) and len(raw_ref_audio) == 2:
        wav, sr = raw_ref_audio
    else:
        raise TypeError(f"Unsupported MiniCPM ref_audio payload at stage boundary: {type(raw_ref_audio)}")

    if isinstance(sr, torch.Tensor):
        if sr.numel() != 1:
            raise ValueError("MiniCPM ref_audio sample rate tensor must be scalar.")
        sr = int(sr.item())
    elif not isinstance(sr, int):
        sr = int(sr)

    if isinstance(wav, torch.Tensor):
        wav_np = wav.detach().cpu().float().numpy()
    else:
        wav_np = np.asarray(wav, dtype=np.float32)

    if wav_np.ndim == 0:
        raise ValueError("MiniCPM ref_audio waveform must be at least 1-D.")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=-1)

    return {
        "wav": np.asarray(wav_np, dtype=np.float32).reshape(-1).tolist(),
        "sr": int(sr),
    }


def talker2code2wav(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Minimal non-async handoff for the future MiniCPM code2wav stage.

    MiniCPM talker emits audio codec token ids directly, so the full finished
    token sequence is enough to seed the next stage once that decoder exists.
    """
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs: list[OmniTokensPrompt] = []

    for i, talker_output in enumerate(talker_outputs):
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        token_ids = list(output.token_ids)
        if not token_ids:
            continue
        additional_information: dict[str, Any] | None = None
        ref_audio = _extract_ref_audio_from_prompt(prompt, index=i)
        if ref_audio is not None:
            additional_information = {"ref_audio": ref_audio}
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
