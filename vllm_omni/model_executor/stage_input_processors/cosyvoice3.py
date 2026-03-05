from typing import Any

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def text2flow(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = True,
):
    """Build stage-1 inputs by prefixing stage-0 prompt ids to its outputs."""
    source_stage_id = engine_input_source[0]
    source_outputs = stage_list[source_stage_id].engine_outputs

    if not isinstance(prompt, list):
        prompt = [prompt]

    source_output = source_outputs[0]
    output = source_output.outputs[0]

    multi_modal_data = output.multimodal_output
    if multi_modal_data is None:
        raise RuntimeError(f"Missing multimodal_output for request {source_output.request_id}")

    output_ids = output.token_ids
    prefix_ids = source_output.prompt_token_ids
    multi_modal_data["prefix_ids"] = prefix_ids
    engine_input = OmniTokensPrompt(prompt_token_ids=output_ids, additional_information=multi_modal_data)
    return [engine_input]
