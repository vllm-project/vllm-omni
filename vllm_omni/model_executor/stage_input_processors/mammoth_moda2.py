"""Stage input processor for MammothModa2 (AR -> DiT)."""

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def ar2dit(
    stage_list: list[Any],
    engine_input_source: list[int],
    prompts: OmniTokensPrompt | TextPrompt | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Convert AR stage outputs to DiT stage inputs."""

    source_stage_id = engine_input_source[0]
    ar_outputs = stage_list[source_stage_id].engine_outputs

    dit_inputs: list[OmniTokensPrompt] = []
    for ar_output, prompt in zip(ar_outputs, prompts):
        addi_info = prompt["additional_information"]
        image_height = addi_info["image_height"][0]
        image_width = addi_info["image_width"][0]
        text_guidance_scale = addi_info["text_guidance_scale"][0]
        cfg_range = addi_info["cfg_range"]
        num_inference_steps = addi_info["num_inference_steps"][0]
        gen_vocab_start_index = addi_info["visual_token_start_id"][0]

        prompt_token_ids = ar_output.prompt_token_ids
         # exclude the last token because it has no corresponding hidden state
        gen_token_ids = ar_output.outputs[0].token_ids[:-1]
        full_token_ids = prompt_token_ids + gen_token_ids

        full_hidden_states = ar_output.multimodal_output["latent"]
        hidden_total = int(full_hidden_states.shape[0])
        assert hidden_total == len(prompt_token_ids) + len(gen_token_ids), (
            f"Hidden states length mismatch: expected {len(prompt_token_ids) + len(gen_token_ids)}, got {hidden_total}"
        )

        mask_device = full_hidden_states.device
        full_token_ids_t = torch.tensor(full_token_ids, dtype=torch.long, device=mask_device)
        attention_mask = torch.ones_like(full_token_ids_t, dtype=torch.bool)

        L = int(full_token_ids_t.shape[0])
        answer_start_index = max(L - 10, 0) # the last 10 tokens as answer
        pos = torch.arange(L, device=mask_device)
        questions_mask = pos < answer_start_index
        answers_mask = ~questions_mask

        gen_token_mask = full_token_ids_t >= gen_vocab_start_index

        visual_token_mask = torch.zeros_like(gen_token_mask)
        visual_ids = [151655, 151656, 151652, 151653] # ["<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>"]
        visual_token_mask = torch.isin(
            full_token_ids_t,
            torch.tensor(visual_ids, dtype=torch.long, device=mask_device),
        )

        text_condition_token_mask = questions_mask & ~(visual_token_mask | gen_token_mask) & attention_mask
        image_condition_token_mask = answers_mask & gen_token_mask & attention_mask

        text_condition = full_hidden_states[text_condition_token_mask]
        image_condition = full_hidden_states[image_condition_token_mask]

        text_prompt_embeds = text_condition.to(dtype=torch.float32).contiguous()
        image_prompt_embeds = image_condition.to(dtype=torch.float32).contiguous()

        additional_information = {
            "text_prompt_embeds": text_prompt_embeds,
            "text_prompt_embeds_shape": list(text_prompt_embeds.shape),
            "image_prompt_embeds": image_prompt_embeds,
            "image_prompt_embeds_shape": list(image_prompt_embeds.shape),
            "image_height": [int(image_height)],
            "image_width": [int(image_width)],
            "text_guidance_scale": [float(text_guidance_scale)],
            "cfg_range": [float(cfg_range[0]), float(cfg_range[1])],
            "num_inference_steps": [int(num_inference_steps)],
        }

        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0],
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return dit_inputs
