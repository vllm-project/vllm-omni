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

    def _unwrap_number(val: object) -> int | float | None:
        if isinstance(val, list) and val:
            val = val[0]
        if isinstance(val, torch.Tensor):
            if val.numel() == 0:
                return None
            val = val.flatten()[0].item()
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    h = _unwrap_number(info.get("image_height", info.get("height")))
    w = _unwrap_number(info.get("image_width", info.get("width")))

    if h is None or w is None:
        size = info.get("image_size")
        if isinstance(size, list) and len(size) >= 2:
            h = _unwrap_number(size[0])
            w = _unwrap_number(size[1])

    if h is None or w is None:
        # best-effort fallback: derive from AR grid
        ar_w = _unwrap_number(info.get("ar_width"))
        ar_h = _unwrap_number(info.get("ar_height"))
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


def _infer_dit_sampling_from_original_prompt(
    prompt: Any,
) -> tuple[float | None, tuple[float, float] | None, int | None]:
    """从 original_prompt 解析 DiT CFG/步数参数。"""
    prompt_obj = prompt[0] if isinstance(prompt, list) and prompt else prompt
    if not isinstance(prompt_obj, dict):
        return None, None, None
    info = prompt_obj.get("additional_information")
    if not isinstance(info, dict):
        return None, None, None

    def _unwrap_scalar(val: object) -> float | None:
        if isinstance(val, list) and val:
            val = val[0]
        if isinstance(val, torch.Tensor):
            if val.numel() == 0:
                return None
            val = val.flatten()[0].item()
        if val is None:
            return None
        try:
            return float(val)
        except Exception:
            return None

    def _unwrap_int(val: object) -> int | None:
        out = _unwrap_scalar(val)
        if out is None:
            return None
        try:
            return int(out)
        except Exception:
            return None

    text_guidance_scale = _unwrap_scalar(info.get("text_guidance_scale"))
    num_inference_steps = _unwrap_int(info.get("num_inference_steps"))

    cfg_range = info.get("cfg_range")
    cfg_tuple: tuple[float, float] | None = None
    if isinstance(cfg_range, list) and len(cfg_range) >= 2:
        start = _unwrap_scalar(cfg_range[0])
        end = _unwrap_scalar(cfg_range[1])
        if start is not None and end is not None:
            cfg_tuple = (float(start), float(end))
    elif isinstance(cfg_range, torch.Tensor) and cfg_range.numel() >= 2:
        start = _unwrap_scalar(cfg_range.flatten()[0].item())
        end = _unwrap_scalar(cfg_range.flatten()[1].item())
        if start is not None and end is not None:
            cfg_tuple = (float(start), float(end))

    return text_guidance_scale, cfg_tuple, num_inference_steps


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
    text_guidance_scale, cfg_range, num_inference_steps = _infer_dit_sampling_from_original_prompt(prompt)

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

        # 对齐 mammothmoda 的 encode_full_prompts：
        #   text_condition_token_mask = questions_mask & ~(visual_token_mask | gen_token_mask) & attention_mask
        #   image_condition_token_mask = answers_mask & gen_token_mask
        # 其中 questions/answers 的分界点在上游实现里用 “末尾 10 个 token” 的启发式切分。
        #
        # 注意：mammothmoda 的 decode_diffusion_image 会在序列尾部额外拼接一个 <|image end|> token，
        # 再基于 (L-10) 切分 question/answer。这里我们用同样的方式在 CPU 上补一个 token，
        # 并用零 hidden state 占位，确保切分点一致（该 token 通常不会进入 condition）。
        full_token_ids = prompt_token_ids + gen_token_ids
        full_hidden_states = torch.cat([prompt_hidden_states, gen_hidden_states], dim=0)
        if len(full_token_ids) != int(full_hidden_states.shape[0]):
            raise RuntimeError(
                "AR token ids length mismatch with hidden states after alignment: "
                f"{len(full_token_ids)} != {int(full_hidden_states.shape[0])}"
            )

        mask_device = full_hidden_states.device
        full_token_ids_t = torch.tensor(full_token_ids, dtype=torch.long, device=mask_device)
        attention_mask = torch.ones_like(full_token_ids_t, dtype=torch.bool)

        # Best-effort 获取 <|image end|> token id（用于对齐原实现的切分点）。
        tok = getattr(stage_list[source_stage_id], "tokenizer", None)
        img_end_id = None
        try:
            eoi_token = getattr(tok, "eoi_token", None)
            get_vocab = getattr(tok, "get_vocab", None)
            if callable(get_vocab) and eoi_token is not None:
                img_end_id = get_vocab().get(eoi_token)
        except Exception:
            img_end_id = None

        if isinstance(img_end_id, int):
            full_token_ids_t = torch.cat(
                [full_token_ids_t, torch.tensor([img_end_id], dtype=torch.long, device=mask_device)], dim=0
            )
            attention_mask = torch.cat(
                [attention_mask, torch.tensor([True], dtype=torch.bool, device=mask_device)], dim=0
            )
            pad_h = full_hidden_states.new_zeros((1, int(full_hidden_states.shape[1])))
            full_hidden_states = torch.cat([full_hidden_states, pad_h], dim=0)

        # questions/answers mask：最后 10 个 token 作为 answer（mammothmoda 逻辑）
        L = int(full_token_ids_t.shape[0])
        answer_start_index = max(L - 10, 0)
        pos = torch.arange(L, device=mask_device)
        questions_mask = pos < answer_start_index
        answers_mask = ~questions_mask

        gen_token_mask = full_token_ids_t >= gen_vocab_start_index

        # visual_token_mask：排除 Qwen2.5-VL 的视觉占位符 token（<|vision_start|> 等）
        visual_token_mask = torch.zeros_like(gen_token_mask)
        visual_ids = getattr(tok, "visual_tokens_ids", None)
        if isinstance(visual_ids, list) and visual_ids:
            try:
                visual_token_mask = torch.isin(
                    full_token_ids_t,
                    torch.tensor(visual_ids, dtype=torch.long, device=mask_device),
                )
            except Exception:
                visual_token_mask = torch.zeros_like(gen_token_mask)

        text_condition_token_mask = questions_mask & ~(visual_token_mask | gen_token_mask) & attention_mask
        image_condition_token_mask = answers_mask & gen_token_mask & attention_mask

        text_condition = full_hidden_states[text_condition_token_mask]
        image_condition = full_hidden_states[image_condition_token_mask]
        if image_condition.numel() == 0:
            raise RuntimeError(
                "No image condition tokens found in AR outputs after mask alignment "
                f"(gen_vocab_start_index={gen_vocab_start_index}, answer_tail=10)."
            )

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
        if text_guidance_scale is not None:
            additional_information["text_guidance_scale"] = [float(text_guidance_scale)]
        if cfg_range is not None:
            additional_information["cfg_range"] = [float(cfg_range[0]), float(cfg_range[1])]
        if num_inference_steps is not None:
            additional_information["num_inference_steps"] = [int(num_inference_steps)]

        dit_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=prompt_token_ids,
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return dit_inputs
