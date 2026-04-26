import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.inputs.data import OmniTokensPrompt

TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294


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
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.cumulative_token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        mm: OmniPayload = output.multimodal_output
        latent = mm["latent"]
        thinker_hidden_states = latent.clone().detach().to(latent.device)
        additional_information = {
            "hidden_states": {
                "output": thinker_hidden_states[prompt_token_ids_len:].to(torch.float32),
                "output_shape": list(thinker_hidden_states[prompt_token_ids_len:].shape),
            },
            "embed": {
                "prefill": thinker_hidden_states[:prompt_token_ids_len].to(torch.float32),
                "prefill_shape": list(thinker_hidden_states[:prompt_token_ids_len].shape),
            },
            "ids": {"prompt": prompt_token_ids, "output": thinker_output_ids},
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
