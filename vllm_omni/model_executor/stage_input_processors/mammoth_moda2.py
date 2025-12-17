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
        hidden = getattr(output, "text_hidden_states", None)
        if hidden is None:
            hidden = mm_out.get("hidden_states")
        captured = hidden

        additional_information: dict[str, object] = {}
        if captured:
            # 将隐藏态转换为 [B, L, H]
            if isinstance(captured, torch.Tensor):
                if captured.ndim == 2:
                    captured = captured.unsqueeze(0)
                text_condition_tokens = captured
            elif isinstance(captured, (list, tuple)) and captured and isinstance(captured[0], torch.Tensor):
                stacked = torch.stack(list(captured), dim=0)  # [K, ...]
                text_condition_tokens = stacked[-1] if stacked.ndim >= 3 else stacked  # 取最后一层
                if text_condition_tokens.ndim == 2:
                    text_condition_tokens = text_condition_tokens.unsqueeze(0)
            else:
                text_condition_tokens = None

            if text_condition_tokens is not None:
                text_condition_attention_mask = torch.ones(
                    text_condition_tokens.shape[:2],
                    dtype=torch.bool,
                )
            additional_information["text_condition_tokens"] = text_condition_tokens.detach().cpu()
            additional_information["text_condition_attention_mask"] = text_condition_attention_mask

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
