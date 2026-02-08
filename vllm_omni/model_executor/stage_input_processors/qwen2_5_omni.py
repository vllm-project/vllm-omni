import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294

logger = init_logger(__name__)


def _validate_stage_inputs(stage_list, engine_input_source):
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    stage_id = engine_input_source[0]
    if stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {stage_id}")

    stage = stage_list[stage_id]
    if stage.engine_outputs is None:
        raise RuntimeError(f"Stage {stage_id} has no outputs yet")

    return stage.engine_outputs


def thinker2talker(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
    async_chunk_stream: bool = False,
):
    thinker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    talker_inputs = []
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(thinker_outputs, prompt)
    }

    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        latent = output.multimodal_output["latent"]
        if isinstance(latent, list):
            latent = torch.cat(latent, dim=0)
        thinker_hidden_states = latent.clone().detach().to(latent.device)
        additional_information = {
            "thinker_result": thinker_hidden_states[prompt_token_ids_len:].to(torch.float32),
            "prompt_embeds": thinker_hidden_states[:prompt_token_ids_len].to(torch.float32),
            "prompt_token_ids": prompt_token_ids,
            "thinker_output_token_ids": thinker_output_ids,
            "thinker_result_shape": list(thinker_hidden_states[prompt_token_ids_len:].shape),
            "prompt_embeds_shape": list(thinker_hidden_states[:prompt_token_ids_len].shape),
        }
        if async_chunk_stream:
            additional_information["is_prefill"] = [True] if len(thinker_output_ids) <= 1 else [False]
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[TALKER_CODEC_START_TOKEN_ID]
                + [TALKER_CODEC_PAD_TOKEN_ID] * (len(prompt_token_ids))
                + [TALKER_CODEC_END_TOKEN_ID],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[thinker_output.request_id]
                    if requires_multimodal_data and multi_modal_data is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


def talker2codewav(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
    async_chunk_stream: bool = False,
):
    assert async_chunk_stream, (
        "The talker2codewav function should be called only when async_chunk_stream is set to True"
    )
    talker_outputs = _validate_stage_inputs(stage_list, engine_input_source)
    code2wav_inputs = []

    # The number of talker output tokens to accumulate
    # before invoking code2wav stage
    talker_tokens_batch_size = 36

    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        talker_output.request_id: p.get("multi_modal_data", None) for talker_output, p in zip(talker_outputs, prompt)
    }

    for talker_output in talker_outputs:
        talker_output_token_ids = talker_output.outputs[0].token_ids

        additional_information = {
            "is_prefill": [True] if len(talker_output_token_ids) < talker_tokens_batch_size else [False]
        }
        talker_output_token_ids = talker_output_token_ids[:talker_tokens_batch_size]
        engine_input = OmniTokensPrompt(
            prompt_token_ids=talker_output_token_ids,
            additional_information=additional_information,
            multi_modal_data=(
                multi_modal_data[talker_output.request_id] if requires_multimodal_data and multi_modal_data else None
            ),
        )
        code2wav_inputs.append(engine_input)
    return code2wav_inputs
