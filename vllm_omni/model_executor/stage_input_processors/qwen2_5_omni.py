import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import OmniPayload
from vllm_omni.inputs.data import OmniTokensPrompt

TALKER_CODEC_PAD_TOKEN_ID = 8292
TALKER_CODEC_START_TOKEN_ID = 8293
TALKER_CODEC_END_TOKEN_ID = 8294


def thinker2talker(
    source_outputs,
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = False,
):
    thinker_outputs = source_outputs
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


def talker2code2wav(
    source_outputs,
    _prompt: OmniTokensPrompt | TextPrompt = None,
    _requires_multimodal_data: bool = False,
):
    code2wav_inputs = []
    for talker_output in source_outputs:
        output = talker_output.outputs[0]
        token_ids = list(output.cumulative_token_ids)
        if token_ids and token_ids[0] == TALKER_CODEC_START_TOKEN_ID:
            token_ids = token_ids[1:]
        if token_ids and token_ids[-1] == TALKER_CODEC_END_TOKEN_ID:
            token_ids = token_ids[:-1]
        if not token_ids:
            continue
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=token_ids,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs
