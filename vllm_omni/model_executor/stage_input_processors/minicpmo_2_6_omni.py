from typing import Union

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def llm2tts(
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

    llm_outputs = stage_list[source_stage_id].engine_outputs
    tts_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {
        llm_output.request_id: p.get("multi_modal_data", None)
        if isinstance(p, dict)
        else None
        for llm_output, p in zip(llm_outputs, prompt)
    }

    for i, llm_output in enumerate(llm_outputs):
        output = llm_output.outputs[0]
        prompt_token_ids = llm_output.prompt_token_ids
        llm_output_ids = output.token_ids
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

        additional_information = {
            "prompt_embeds": prompt_hidden,
            "prompt_token_ids": list(prompt_token_ids),
            "llm_output_token_ids": list(llm_output_ids) if not isinstance(llm_output_ids, list) else llm_output_ids,
            "llm_output_text": [thinker_text],
        }

        # Minimal prompt token IDs: the talker's AR framework needs *some* tokens
        # to do a single prefill step. We use [BOS, PAD, EOS] as a dummy.
        tts_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[1, 0, 2],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[llm_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(llm_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return tts_inputs


def tts2t2w(
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

    tts_outputs = stage_list[source_stage_id].engine_outputs
    t2w_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {
        tts_output.request_id: p.get("multi_modal_data", None)
        if isinstance(p, dict)
        else None
        for tts_output, p in zip(tts_outputs, prompt)
    }

    for i, tts_output in enumerate(tts_outputs):
        output = tts_output.outputs[0]

        mel_spec = None
        if hasattr(output, "multimodal_output") and isinstance(output.multimodal_output, dict):
            mel_spec = output.multimodal_output.get("mel_spec")

        if mel_spec is None:
            import logging
            logging.getLogger(__name__).warning(
                "tts2t2w: no mel_spec found in talker output "
                "(multimodal_output keys: %s)",
                list(output.multimodal_output.keys())
                if hasattr(output, "multimodal_output") and isinstance(output.multimodal_output, dict)
                else "N/A",
            )

        additional_information = {}
        if mel_spec is not None:
            additional_information["mel_spec"] = mel_spec

        # Minimal dummy prompt token IDs for the AR framework (max_tokens=1)
        t2w_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[1, 0, 2],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[tts_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(tts_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return t2w_inputs
