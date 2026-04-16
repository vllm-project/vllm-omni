"""Stage input processors for HyperCLOVAX-SEED-Omni-8B pipeline.

The thinker generates a mixed token sequence containing:
  - Regular text tokens (< 128606)
  - Discrete audio tokens (128606 ~ 135167)
  - Discrete vision tokens (135168 ~ 135168+255)

These processors extract the relevant discrete tokens and route them
to the appropriate decoder stage.
"""

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

# Token ID boundaries from config.json
DISCRETE_AUDIO_UNIT_0_ID = 128606
DISCRETE_IMAGE_UNIT_0_ID = 135168
DISCRETE_AUDIO_VOCAB_SIZE = 6561  # CosyVoice2 FSQ codebook
DISCRETE_IMAGE_VOCAB_SIZE = 65536  # TA-Tok SimVQ codebook (2^16)
DISCRETE_IMAGE_TOKEN_LENGTH = 729  # 27x27 latent tokens per image


def _extract_discrete_tokens(token_ids: list[int], start_id: int, vocab_size: int) -> list[int]:
    """Extract and remap discrete tokens from a mixed token sequence.

    Returns tokens remapped to [0, vocab_size) range.
    """
    return [tid - start_id for tid in token_ids if start_id <= tid < start_id + vocab_size]


def thinker2vision_decoder(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    """Extract discrete vision tokens from thinker output → vision decoder.

    The vision decoder (HyperCLOVAXVisionPipeline) takes 256 discrete codes
    per image and converts them to pixel images via diffusion.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    thinker_outputs = stage_list[source_stage_id].engine_outputs
    if thinker_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    vision_decoder_inputs = []
    for thinker_output in thinker_outputs:
        # Text-only iterations can produce an empty outputs list.
        if not thinker_output.outputs:
            continue
        output = thinker_output.outputs[0]
        output_token_ids = list(output.token_ids)
        vision_codes = _extract_discrete_tokens(output_token_ids, DISCRETE_IMAGE_UNIT_0_ID, DISCRETE_IMAGE_VOCAB_SIZE)

        if not vision_codes:
            continue

        # Truncate/pad to exact DISCRETE_IMAGE_TOKEN_LENGTH (27x27=729).
        # The LLM may generate slightly more or fewer tokens than expected;
        # the vision decoder rearranges as (h w) → (h, w) so the length must be
        # a perfect square == DISCRETE_IMAGE_TOKEN_LENGTH.
        vision_codes = vision_codes[:DISCRETE_IMAGE_TOKEN_LENGTH]
        vision_codes += [0] * (DISCRETE_IMAGE_TOKEN_LENGTH - len(vision_codes))

        # Pipeline expects vision_tokens key in req.extra
        vision_decoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=vision_codes,
                additional_information={
                    "request_id": thinker_output.request_id,
                    "vision_tokens": vision_codes,
                    "num_images": 1,
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return vision_decoder_inputs


def thinker2audio_decoder(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    """Extract discrete audio tokens from thinker output → audio decoder.

    The audio decoder (Unit-BigVGAN) takes discrete audio codes (6561 vocab)
    and converts them to 24kHz waveforms.
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    thinker_outputs = stage_list[source_stage_id].engine_outputs
    if thinker_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    audio_decoder_inputs = []
    for thinker_output in thinker_outputs:
        # Text-only iterations can produce an empty outputs list.
        if not thinker_output.outputs:
            continue
        output = thinker_output.outputs[0]
        output_token_ids = list(output.token_ids)

        audio_codes = _extract_discrete_tokens(output_token_ids, DISCRETE_AUDIO_UNIT_0_ID, DISCRETE_AUDIO_VOCAB_SIZE)

        if not audio_codes:
            continue

        # Pipeline expects audio_tokens as list[list[int]] (batch),
        # speakers as list[str], formats as list[str], and optional
        # ref_audio_tokens for zero-shot TTS (ECAPA-TDNN speaker embedding).
        # ref_audio_b64 is the raw base64 audio from the user's input message,
        # injected by serving_chat.py into the engine_prompt dict.
        _ref = None
        if isinstance(prompt, dict):
            _ref = prompt.get("ref_audio_b64")
        elif isinstance(prompt, list) and prompt:
            _p = prompt[0]
            if isinstance(_p, dict):
                _ref = _p.get("ref_audio_b64")
        audio_decoder_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_codes,
                additional_information={
                    "request_id": thinker_output.request_id,
                    "audio_tokens": [audio_codes],
                    "speakers": ["fkms"],
                    "ref_audio_tokens": [_ref],
                },
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return audio_decoder_inputs
