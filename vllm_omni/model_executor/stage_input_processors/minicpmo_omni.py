from typing import Union

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def thinker2talker(
    stage_list,
    engine_input_source,
    prompt: Union[OmniTokensPrompt, TextPrompt] = None,
    requires_multimodal_data: bool = False,
):
    """Convert thinker stage output to talker stage input for MiniCPMO Omni.

    Extracts from thinker output:
      - Full hidden states (prompt + generated) for speaker embedding extraction
      - Prompt token IDs (for finding spk_bos/spk_eos positions)
      - Generated token IDs (for decoding TTS text)

    The talker model will:
      1. Find <|spk_bos|>/<|spk_eos|> positions in prompt_token_ids
      2. Extract speaker embedding from hidden states at those positions
      3. Decode generated text and extract TTS content
      4. Run ConditionalChatTTS pipeline
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    thinker_outputs = stage_list[source_stage_id].engine_outputs
    talker_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None)
        if isinstance(p, dict)
        else None
        for thinker_output, p in zip(thinker_outputs, prompt)
    }

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)

        latent = output.multimodal_output.get("latent", None)
        if latent is None:
            latent = output.hidden_states if hasattr(output, "hidden_states") else None
            if latent is None:
                raise ValueError("No latent or hidden_states found in thinker output")

        thinker_hidden_states = latent.clone().detach()

        # Split hidden states: prompt portion has speaker embedding,
        # generated portion has the text content
        prompt_hidden = thinker_hidden_states[:prompt_token_ids_len].to(torch.float32)

        # Extract decoded text from thinker output for TTS text extraction
        thinker_text = getattr(output, "text", "") or ""

        full_token_ids = list(prompt_token_ids) + (list(thinker_output_ids) if not isinstance(thinker_output_ids, list) else thinker_output_ids)
        full_hidden = thinker_hidden_states.to(torch.float32)

        tts_bos_id = 151691
        tts_eos_id = 151692
        for _check_id in [151703, 151704]:
            if _check_id in full_token_ids:
                tts_bos_id = 151703
                tts_eos_id = 151704
                break

        tts_bos_idx = None
        tts_eos_idx = None
        for idx_t, tid in enumerate(full_token_ids):
            if tid == tts_bos_id:
                tts_bos_idx = idx_t + 1
            elif tid == tts_eos_id:
                tts_eos_idx = idx_t

        tts_token_ids_slice = None
        tts_hidden_slice = None

        import logging as _log
        _tlog = _log.getLogger("thinker2talker")
        _tlog.warning(
            "DEBUG: full_token_ids len=%d, full_hidden shape=%s, "
            "prompt_len=%d, output_ids_len=%d, "
            "tts_bos_id=%d, tts_bos_idx=%s, tts_eos_idx=%s",
            len(full_token_ids), list(full_hidden.shape),
            prompt_token_ids_len, len(thinker_output_ids),
            tts_bos_id, tts_bos_idx, tts_eos_idx,
        )

        if tts_bos_idx is not None and full_hidden.shape[0] > tts_bos_idx:
            end = tts_eos_idx if tts_eos_idx is not None else full_hidden.shape[0]
            tts_token_ids_slice = torch.tensor(full_token_ids[tts_bos_idx:end], dtype=torch.long)
            tts_hidden_slice = full_hidden[tts_bos_idx:end]
            _tlog.warning("DEBUG: tts_token_ids_slice len=%d, tts_hidden_slice shape=%s", len(tts_token_ids_slice), list(tts_hidden_slice.shape))
        else:
            _tlog.warning("DEBUG: tts_bos_idx=%s not found or hidden too short (%d)", tts_bos_idx, full_hidden.shape[0])

        additional_information = {
            "prompt_embeds": prompt_hidden,
            "prompt_token_ids": list(prompt_token_ids),
            "thinker_output_token_ids": list(thinker_output_ids) if not isinstance(thinker_output_ids, list) else thinker_output_ids,
            "thinker_output_text": [thinker_text],
            "tts_token_ids": tts_token_ids_slice,
            "tts_hidden_states": tts_hidden_slice,
        }

        # Minimal prompt token IDs: the talker's AR framework needs *some* tokens
        # to do a single prefill step. We use [BOS, PAD, EOS] as a dummy.
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[1, 0, 2],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(thinker_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return talker_inputs


def talker2code2wav(
    stage_list,
    engine_input_source,
    prompt: Union[OmniTokensPrompt, TextPrompt] = None,
    requires_multimodal_data: bool = False,
):
    """Convert talker stage output to code2wav stage input for MiniCPMO Omni.

    Extracts mel_spec from talker's multimodal output and passes it to
    the code2wav stage for Vocos vocoder (mel → waveform) conversion.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    talker_outputs = stage_list[source_stage_id].engine_outputs
    code2wav_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {
        talker_output.request_id: p.get("multi_modal_data", None)
        if isinstance(p, dict)
        else None
        for talker_output, p in zip(talker_outputs, prompt)
    }

    for i, talker_output in enumerate(talker_outputs):
        output = talker_output.outputs[0]

        mel_spec = None
        waveform = None
        if hasattr(output, "multimodal_output") and isinstance(output.multimodal_output, dict):
            import logging as _log2
            _log2.getLogger("talker2code2wav").warning(
                "DEBUG: multimodal_output keys=%s, values types=%s",
                list(output.multimodal_output.keys()),
                {k: (type(v).__name__, v.shape if hasattr(v, 'shape') else len(v) if hasattr(v, '__len__') else '?')
                 for k, v in output.multimodal_output.items()},
            )
            mel_spec = output.multimodal_output.get("mel_spec")
            waveform = output.multimodal_output.get("model_outputs")
            if mel_spec is None and waveform is None:
                import torch as _torch
                latent = output.multimodal_output.get("latent")
                if isinstance(latent, _torch.Tensor) and latent.dim() >= 2 and 100 in latent.shape:
                    mel_spec = latent

        if mel_spec is None and waveform is None:
            import logging
            logging.getLogger(__name__).warning(
                "talker2code2wav: no mel_spec/waveform found "
                "(multimodal_output keys: %s)",
                list(output.multimodal_output.keys())
                if hasattr(output, "multimodal_output") and isinstance(output.multimodal_output, dict)
                else "N/A",
            )

        additional_information = {}
        if waveform is not None:
            additional_information["waveform"] = waveform
        elif mel_spec is not None:
            additional_information["mel_spec"] = mel_spec

        # Minimal dummy prompt token IDs for the AR framework (max_tokens=1)
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[1, 0, 2],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[talker_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(talker_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs
