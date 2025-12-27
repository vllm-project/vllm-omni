from typing import Any

from vllm.inputs import TextPrompt
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt

logger = init_logger(__name__)


def text2flow(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt = None,
    requires_multimodal_data: bool = True,
):
    """Build stage-1 inputs by prefixing stage-0 prompt ids to its outputs."""
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")
    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    source_outputs = stage_list[source_stage_id].engine_outputs
    if not isinstance(prompt, list):
        prompt = [prompt]

    if len(source_outputs) != 1:
        raise RuntimeError(f"Expected exactly one source output, got {len(source_outputs)}")
    if len(prompt) != 1:
        raise RuntimeError(f"Expected exactly one prompt, got {len(prompt)}")

    source_output = source_outputs[0]
    prompt_payload = prompt[0]
    if prompt_payload is None:
        raise RuntimeError(f"Missing prompt payload for {source_output.request_id}")
    # print("prompt payload")
    # print(prompt_payload)

    # print("here")
    # print("--------------------------")
    # print(source_output)
    # print(source_output.outputs[0])
    multi_modal_data = source_output.multimodal_output
    output_ids = source_output.outputs[0].token_ids
    prefix_ids = source_output.prompt_token_ids
    multi_modal_data["prefix_ids"] = prefix_ids

    engine_input = OmniTokensPrompt(prompt_token_ids=output_ids, additional_information=multi_modal_data)
    return [engine_input]
