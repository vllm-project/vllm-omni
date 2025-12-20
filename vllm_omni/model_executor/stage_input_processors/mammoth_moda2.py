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

    # MammothModa2: gen token vocab start（来自 llm_config.gen_vocab_start_index）
    # NOTE: 先用固定值跑通 ar→dit；后续可从 stage0 的 hf_config 动态读取并透传。
    gen_vocab_start_index = 152064

    for ar_output in ar_outputs:
        # vllm-omni 会把 stage0 的 engine_output_type=latent 聚合到 RequestOutput.multimodal_output["latent"]
        mm = getattr(ar_output, "multimodal_output", None)
        latent = mm.get("latent") if isinstance(mm, dict) else None
        if not isinstance(latent, dict):
            raise RuntimeError(
                "AR stage did not produce expected latent payload. "
                "Ensure stage0 has engine_output_type=latent and uses vllm-omni GPUARWorker."
            )

        # GPUARModelRunner 侧 payload 使用 "hidden" 存放每个 step 的 hidden slice，
        # OutputProcessor 会按 token 维度把这些 slice 逐步拼接起来。
        hidden_states = latent.get("hidden")
        if hidden_states is None:
            raise RuntimeError("latent payload missing `hidden`")
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"latent['hidden'] expected torch.Tensor, got {type(hidden_states)}")

        # token ids 不在 latent payload 里；这里用 RequestOutput 的 generated token ids 对齐最后一段 hidden。
        gen_token_ids = getattr(ar_output.outputs[0], "token_ids", None) or []
        if not gen_token_ids:
            raise RuntimeError("AR outputs have empty generated token ids.")

        gen_len = len(gen_token_ids)
        if hidden_states.shape[0] < gen_len:
            raise RuntimeError(
                f"latent hidden states shorter than generated token ids: {hidden_states.shape[0]} < {gen_len}"
            )
        gen_hidden_states = hidden_states[-gen_len:]
        token_ids_t = torch.tensor(gen_token_ids, dtype=torch.long)

        # 取 gen vocab 范围内 token 的 hidden states 作为 diffusion condition tokens
        gen_mask = token_ids_t >= gen_vocab_start_index
        cond = gen_hidden_states[gen_mask]
        if cond.numel() == 0:
            raise RuntimeError("No gen tokens found in AR outputs; did you run t2i token generation on AR stage?")

        # DiT stage 通过 prompt_embeds 接收条件 token hidden states（避免依赖额外 kwargs 透传）
        # prompt_token_ids 仅用于对齐长度；值本身在 DiT stage 不使用。
        prompt_embeds = cond.to(dtype=torch.float16).contiguous()
        prompt_token_ids = [0] * int(prompt_embeds.shape[0])

        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=prompt_token_ids,
                prompt_embeds=prompt_embeds,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return dit_inputs
