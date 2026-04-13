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

        additional_information = {
            "prompt_embeds": prompt_hidden,
            "prompt_token_ids": list(prompt_token_ids),
            "thinker_output_token_ids": list(thinker_output_ids) if not isinstance(thinker_output_ids, list) else thinker_output_ids,
            "thinker_output_text": [thinker_text],
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
        if hasattr(output, "multimodal_output") and isinstance(output.multimodal_output, dict):
            mel_spec = output.multimodal_output.get("mel_spec")
            if mel_spec is None:
                import torch as _torch
                latent = output.multimodal_output.get("latent")
                if isinstance(latent, _torch.Tensor) and latent.dim() >= 2 and 100 in latent.shape:
                    mel_spec = latent

        if mel_spec is None:
            import logging
            logging.getLogger(__name__).warning(
                "talker2code2wav: no mel_spec found in talker output "
                "(multimodal_output keys: %s)",
                list(output.multimodal_output.keys())
                if hasattr(output, "multimodal_output") and isinstance(output.multimodal_output, dict)
                else "N/A",
            )

        additional_information = {}
        if mel_spec is not None:
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
