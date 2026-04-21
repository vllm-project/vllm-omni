"""Stage input processor for MiniCPM-o-4_5 S2S: LM text tokens → Token2Speech."""

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def text2speech(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[Any]:
    """Package Stage-0 generated text token IDs as Stage-1 OmniTokensPrompt.

    Stage 0 (standard MiniCPMO LM) generates text token IDs available via
    engine_output_type="latent" → output.token_ids. Stage 1 (MiniCPMOToken2Speech)
    receives them as input_ids and runs MiniCPMTTS + Token2wav.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    stage0_outputs = stage_list[engine_input_source[0]].engine_outputs
    result: list[Any] = []
    for out in stage0_outputs:
        if not out.finished:
            continue
        output = out.outputs[0]
        text_token_ids = list(output.token_ids)
        if not text_token_ids:
            logger.warning("minicpmo text2speech: Stage 0 produced empty token_ids")
        logger.info("minicpmo text2speech: %d text tokens → Stage 1", len(text_token_ids))
        result.append(OmniTokensPrompt(prompt_token_ids=text_token_ids))
    return result
