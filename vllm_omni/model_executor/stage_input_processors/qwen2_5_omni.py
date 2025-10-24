from typing import Union
import torch

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt

def thinker2talker(stage_list, engine_input_source, prompt: Union[OmniTokensPrompt, TextPrompt] = None):
    if not engine_input_source:
      raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
      raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")
    thinker_outputs = stage_list[source_stage_id].engine_outputs
    talker_inputs = []
    multi_modal_data = {thinker_output.request_id: 
            prompt.get('multi_modal_data', None) for thinker_output, prompt in zip(thinker_outputs, prompt)}
    
    for i, thinker_output in enumerate(thinker_outputs):
        output = thinker_output.outputs[0]
        prompt_token_ids = thinker_output.prompt_token_ids
        thinker_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)
        thinker_hidden_states = output.multimodal_output["latent"].clone().detach().cuda()
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * (len(prompt_token_ids) + 2), # add 2 for codec pad and start token
                additional_information={
                    "thinker_result": thinker_hidden_states[prompt_token_ids_len:].to(torch.float32),
                    "prompt_embeds": thinker_hidden_states[:prompt_token_ids_len].to(torch.float32),
                    "prompt_token_ids": prompt_token_ids,
                    "thinker_output_token_ids": thinker_output_ids,
                },
                multi_modal_data=multi_modal_data[thinker_output.request_id] if multi_modal_data is not None else None,
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs