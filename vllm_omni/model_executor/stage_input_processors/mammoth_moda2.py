"""Stage input processor for MammothModa2 (AR -> DiT)."""

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def ar2dit(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompt: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """
    将 AR 阶段输出的隐藏态等信息打包为 DiT 阶段输入。
    """
    if not engine_input_source:
        raise ValueError("engine_input_source cannot be empty")

    source_stage_id = engine_input_source[0]
    if source_stage_id >= len(stage_list):
        raise IndexError(f"Invalid stage_id: {source_stage_id}")
    if stage_list[source_stage_id].engine_outputs is None:
        raise RuntimeError(f"Stage {source_stage_id} has no outputs yet")

    ar_outputs = stage_list[source_stage_id].engine_outputs
    dit_inputs: list[OmniTokensPrompt] = []

    for ar_output in ar_outputs:
        output = ar_output.outputs[0]
        mm_out = getattr(output, "multimodal_outputs", {}) or {}
        captured = mm_out.get("captured_hidden_states")
        hidden_states = mm_out.get("hidden_states")

        additional_information = {}
        if captured is not None:
            additional_information["captured_hidden_states"] = [h.detach().cpu() for h in captured]
        if hidden_states is not None:
            additional_information["hidden_states"] = [h.detach().cpu() for h in hidden_states]

        prompt_ids = getattr(ar_output, "prompt_token_ids", None)
        if prompt_ids is None and hasattr(output, "token_ids"):
            prompt_ids = output.token_ids
        if isinstance(prompt_ids, torch.Tensor):
            prompt_token_ids = prompt_ids.detach().cpu().tolist()
        else:
            prompt_token_ids = prompt_ids or []

        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=prompt_token_ids,
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return dit_inputs
