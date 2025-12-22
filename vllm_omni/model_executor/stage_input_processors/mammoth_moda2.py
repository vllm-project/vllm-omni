"""Stage input processor for MammothModa2 (AR -> DiT)."""

from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def _infer_gen_vocab_start_index(stage: Any, default: int = 152064) -> int:
    """Best-effort infer `gen_vocab_start_index` from stage0 hf_config."""
    try:
        vllm_cfg = getattr(stage, "vllm_config", None)
        model_cfg = getattr(vllm_cfg, "model_config", None)
        hf_cfg = getattr(model_cfg, "hf_config", None)
        llm_cfg = getattr(hf_cfg, "llm_config", None) or getattr(hf_cfg, "text_config", None)
        text_cfg = getattr(llm_cfg, "text_config", None) or llm_cfg
        val = getattr(text_cfg, "gen_vocab_start_index", None) or getattr(hf_cfg, "gen_vocab_start_index", None)
        return int(val) if val is not None else int(default)
    except Exception:
        return int(default)


def _infer_image_hw_from_original_prompt(prompt: Any) -> tuple[int | None, int | None]:
    """从 orchestrator 透传的 original_prompt 里解析输出分辨率（像素）。

    注意：Stage-0 的 text prompt 预处理会丢弃 additional_information，因此这些字段不会进入
    AR stage 的 engine request；但 orchestrator 会把 original_prompt 原样传给 Stage-1，
    我们可以在这里读取并透传给 DiT stage。
    """
    prompt_obj = None
    if isinstance(prompt, list):
        prompt_obj = prompt[0] if prompt else None
    else:
        prompt_obj = prompt

    if not isinstance(prompt_obj, dict):
        return None, None
    info = prompt_obj.get("additional_information")
    if not isinstance(info, dict):
        return None, None

    h = info.get("image_height", info.get("height"))
    w = info.get("image_width", info.get("width"))

    if h is None or w is None:
        size = info.get("image_size")
        if isinstance(size, list) and len(size) >= 2:
            h, w = size[0], size[1]

    if h is None or w is None:
        # best-effort fallback: derive from AR grid
        ar_w = info.get("ar_width")
        ar_h = info.get("ar_height")
        try:
            if ar_w is not None and ar_h is not None:
                w = int(ar_w) * 16
                h = int(ar_h) * 16
        except Exception:
            pass

    try:
        h_i = int(h) if h is not None else None
        w_i = int(w) if w is not None else None
    except Exception:
        return None, None

    if h_i is None or w_i is None:
        return None, None
    return h_i, w_i


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
    gen_vocab_start_index = _infer_gen_vocab_start_index(stage_list[source_stage_id], default=152064)

    image_height, image_width = _infer_image_hw_from_original_prompt(prompt)

    for ar_output in ar_outputs:
        # vllm-omni 会把 stage0 的 engine_output_type=latent 聚合到 multimodal_output["latent"]。
        # 注意：不同输出路径下，multimodal_output 可能在：
        # - RequestOutput.multimodal_output（stage 输出聚合层）
        # - RequestOutput.outputs[0].multimodal_output（单个 completion 输出）
        mm = getattr(ar_output, "multimodal_output", None)
        if not (isinstance(mm, dict) and "latent" in mm):
            try:
                first_out = ar_output.outputs[0]
                mm2 = getattr(first_out, "multimodal_output", None)
                if isinstance(mm2, dict) and "latent" in mm2:
                    mm = mm2
            except Exception:
                pass

        # MultimodalOutputProcessor.add_multimodal_tensor 会把 producer 的 payload
        # 正规化到 mm_accumulated 上：
        # - 如果 payload 是 tensor：mm["latent"] = tensor
        # - 如果 payload 是 dict{"hidden": tensor}：会 remap 为 mm["latent"] = tensor
        # - 旧路径也可能是 dict{"hidden": tensor} 直接挂在 latent 下
        latent = mm.get("latent") if isinstance(mm, dict) else None
        if isinstance(latent, torch.Tensor):
            hidden_states = latent
        elif isinstance(latent, dict):
            hidden_states = latent.get("hidden")
        else:
            raise RuntimeError(
                "AR stage did not produce expected latent payload. "
                "Ensure stage0 has engine_output_type=latent and uses vllm-omni GPUARWorker."
            )

        if hidden_states is None:
            raise RuntimeError("latent payload missing hidden states")
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError(f"latent hidden states expected torch.Tensor, got {type(hidden_states)}")

        # 对齐方式必须与 vllm 输出一致：latent 包含 [prompt, generated] 的完整序列 hidden，
        # 而 CompletionOutput.token_ids 仅包含生成段 token_ids；RequestOutput.prompt_token_ids
        # 给出 prompt 段长度。
        prompt_token_ids = getattr(ar_output, "prompt_token_ids", None) or []
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise RuntimeError("AR RequestOutput missing prompt_token_ids; cannot align hidden states.")

        gen_token_ids = getattr(ar_output.outputs[0], "token_ids", None) or []
        if not isinstance(gen_token_ids, list) or not gen_token_ids:
            raise RuntimeError("AR outputs have empty generated token ids.")

        prompt_len = len(prompt_token_ids)
        gen_len = len(gen_token_ids)

        # 注意：自回归解码下，`CompletionOutput.token_ids` 会包含“最后一次采样出来的 token”，
        # 但该 token 的 hidden state 只有在下一步 forward 才会产生；当请求在该 token 处停止时，
        # hidden_states 往往只覆盖到 gen_len-1 个生成 token。
        #
        # 因此这里以 hidden_states 的实际长度为准，截断 token_ids 对齐。
        hidden_total = int(hidden_states.shape[0])
        if hidden_total < prompt_len:
            raise RuntimeError(
                "latent hidden states shorter than prompt tokens: "
                f"{hidden_total} < {prompt_len} (prompt_len={prompt_len})"
            )
        gen_hidden_len = hidden_total - prompt_len
        if gen_hidden_len <= 0:
            raise RuntimeError(
                "latent hidden states contain no generated segment: "
                f"hidden_total={hidden_total}, prompt_len={prompt_len}"
            )
        if gen_hidden_len > gen_len:
            # Best-effort: should be rare; keep the declared generated length.
            gen_hidden_len = gen_len

        prompt_hidden_states = hidden_states[:prompt_len]
        gen_hidden_states = hidden_states[prompt_len : prompt_len + gen_hidden_len]
        gen_token_ids = gen_token_ids[:gen_hidden_len]

        # 取 gen vocab 范围内 token 的 hidden states 作为 image condition tokens。
        # NOTE: MammothModa2 的 eol_token_id == gen_vocab_start_index，本身属于 gen vocab，
        # 它承载了行边界信息，应当与视觉 token 一起作为条件输入。
        gen_token_ids_t = torch.tensor(gen_token_ids, dtype=torch.long)
        image_mask = gen_token_ids_t >= gen_vocab_start_index
        image_condition_gen = gen_hidden_states[image_mask]
        if image_condition_gen.numel() == 0:
            raise RuntimeError(
                "No image condition tokens found in AR outputs "
                f"(gen_vocab_start_index={gen_vocab_start_index}); did AR generate visual tokens?"
            )

        # 贴近原 MammothModa2：text/image 条件分别构造；refiner 只应该作用在 image 条件上。
        prompt_token_ids_t = torch.tensor(prompt_token_ids, dtype=torch.long)
        prompt_is_gen = prompt_token_ids_t >= gen_vocab_start_index
        text_condition = prompt_hidden_states[~prompt_is_gen]
        image_condition_prompt = prompt_hidden_states[prompt_is_gen]
        if image_condition_prompt.numel() > 0:
            image_condition = torch.cat([image_condition_prompt, image_condition_gen], dim=0)
        else:
            image_condition = image_condition_gen

        # 重要：不要使用 prompt_embeds 字段跨 stage 传递 embedding。
        #
        # 原因：OmniProcessor 会把 prompt_embeds 序列化成 PromptEmbedsPayload（bytes），
        # 但 vLLM V1 在构造 CachedRequestState 时会立刻对 prompt_embeds 调用 len()，
        # 从而触发 `PromptEmbedsPayload has no len()` 并在进入 DiT.forward 前崩溃。
        #
        # 参考 qwen2_5_omni 的做法：把 embedding 放到 additional_information 里，
        # 由 runner 在 runtime_additional_information 中透传给模型。
        text_prompt_embeds = text_condition.to(dtype=torch.float32).contiguous()
        image_prompt_embeds = image_condition.to(dtype=torch.float32).contiguous()
        # DiT stage 不依赖 token ids；用最短 prompt 以减少 vLLM 在 stage-1 的调度与 KV 开销。
        prompt_token_ids = [0]
        # vllm_omni/engine/processor.py 要求 additional_information 的 value 只能是
        # torch.Tensor 或 list（会被序列化到 AdditionalInformationPayload）。
        # 这里透传 text/image 两路条件 embedding，避免把 refiner 错用在 text 上。
        additional_information = {
            "text_prompt_embeds": text_prompt_embeds,
            "text_prompt_embeds_shape": list(text_prompt_embeds.shape),
            "image_prompt_embeds": image_prompt_embeds,
            "image_prompt_embeds_shape": list(image_prompt_embeds.shape),
        }
        if image_height is not None and image_width is not None:
            # vllm_omni/engine/processor.py: list is allowed
            additional_information["image_height"] = [int(image_height)]
            additional_information["image_width"] = [int(image_width)]

        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=prompt_token_ids,
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return dit_inputs
