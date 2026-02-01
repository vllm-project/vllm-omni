import torch
from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294

logger = init_logger(__name__)


def thinker2talker(
    stage_list,
    engine_input_source,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
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
        thinker_output.request_id: p.get("multi_modal_data", None) for thinker_output, p in zip(thinker_outputs, prompt)
    }

    for i, thinker_output in enumerate(thinker_outputs):
        is_prefill = [False]
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        if len(thinker_output_ids) <= 1:
            is_prefill = [True]
        prompt_token_ids_len = len(prompt_token_ids)
        latent = output.multimodal_output["latent"]
        # PR 467 changed multimodal_output["latent"] to be a list
        # and it is concatenated only when the stage is finished
        # for performance gains.
        # But this needs to be concatenated for each chunk to stream to next stage
        # TODO: See if there is a robust approach for this that can preserve
        # the performance gains of PR 467
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
            "is_prefill": is_prefill,
        }
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
):
    engine_inputs = []
    if len(engine_input_source) == 0:
        raise ValueError("engine_input_source is empty")
    source_stage_id = engine_input_source[0]
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
    source_outputs = stage_list[source_stage_id].engine_outputs
    if not isinstance(prompt, list):
        prompt = [prompt]
    multi_modal_data = {
        source_output.request_id: p.get("multi_modal_data", None) for source_output, p in zip(source_outputs, prompt)
    }

    for source_output in source_outputs:
        talker_output_token_ids = source_output.outputs[0].token_ids

        additional_information = {"is_prefill": [True] if len(talker_output_token_ids) < 36 else [False]}
        talker_output_token_ids = talker_output_token_ids[:36]
        engine_input = OmniTokensPrompt(
            prompt_token_ids=talker_output_token_ids,
            additional_information=additional_information,
            multi_modal_data=(
                multi_modal_data[source_output.request_id] if requires_multimodal_data and multi_modal_data else None
            ),
        )
        engine_inputs.append(engine_input)
    return engine_inputs
