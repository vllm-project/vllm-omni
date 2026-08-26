# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_omni.diffusion.diffusion_kv.request import DiffusionKVRequest
from vllm_omni.diffusion.request import OmniDiffusionRequest

from .hunyuan_image3_tokenizer import TokenizerEncodeOutput
from .hunyuan_image3_transformer import ImageInfo, JointImageInfo
from .system_prompt import get_system_prompt

if TYPE_CHECKING:
    from transformers.generation.configuration_utils import GenerationConfig

    from .hunyuan_image3_tokenizer import TokenizerWrapper
    from .hunyuan_image3_transformer import HunyuanImage3ImageProcessor


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_tensor_if_needed(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return torch.tensor(value)
    return value


def _image_info_to_payload(image_info: ImageInfo) -> dict[str, Any]:
    return {
        "image_type": image_info.image_type,
        "image_tensor": image_info.image_tensor,
        "image_width": _to_python_scalar(image_info.image_width),
        "image_height": _to_python_scalar(image_info.image_height),
        "token_width": _to_python_scalar(image_info.token_width),
        "token_height": _to_python_scalar(image_info.token_height),
        "image_token_length": _to_python_scalar(image_info.image_token_length),
        "base_size": _to_python_scalar(image_info.base_size),
        "ratio_index": _to_python_scalar(image_info.ratio_index),
        "add_timestep_token": image_info.add_timestep_token,
        "add_guidance_token": image_info.add_guidance_token,
        "use_front_boi_token": image_info.use_front_boi_token,
        "add_image_shape_token": image_info.add_image_shape_token,
    }


def _image_info_from_payload(payload: dict[str, Any]) -> ImageInfo:
    return ImageInfo(
        image_type=payload.get("image_type"),
        image_tensor=_to_tensor_if_needed(payload.get("image_tensor")),
        image_width=payload.get("image_width"),
        image_height=payload.get("image_height"),
        token_width=payload.get("token_width"),
        token_height=payload.get("token_height"),
        image_token_length=payload.get("image_token_length"),
        base_size=payload.get("base_size"),
        ratio_index=payload.get("ratio_index"),
        add_timestep_token=payload.get("add_timestep_token", True),
        add_guidance_token=payload.get("add_guidance_token", False),
        use_front_boi_token=payload.get("use_front_boi_token", True),
        add_image_shape_token=payload.get("add_image_shape_token", True),
    )


def joint_image_info_to_payload(joint_image_info: JointImageInfo) -> dict[str, Any]:
    return {
        "type": "joint_image_info",
        "vae_image_info": _image_info_to_payload(joint_image_info.vae_image_info),
        "vision_image_info": _image_info_to_payload(joint_image_info.vision_image_info),
        "vision_encoder_kwargs": joint_image_info.vision_encoder_kwargs,
    }


def _joint_image_info_from_payload(payload: Any) -> JointImageInfo:
    if isinstance(payload, JointImageInfo):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict or JointImageInfo for conditional image payload, got {type(payload)}.")

    vae_image_info = _image_info_from_payload(payload["vae_image_info"])
    vision_image_info = _image_info_from_payload(payload["vision_image_info"])
    vision_encoder_kwargs = payload.get("vision_encoder_kwargs") or {}
    if isinstance(vision_encoder_kwargs, dict):
        vision_encoder_kwargs = {key: _to_tensor_if_needed(value) for key, value in vision_encoder_kwargs.items()}
    return JointImageInfo(
        vae_image_info=vae_image_info,
        vision_image_info=vision_image_info,
        vision_encoder_kwargs=vision_encoder_kwargs,
    )


def normalize_hunyuan_single_stage_bot_task(bot_task: Any) -> str:
    if isinstance(bot_task, str) and bot_task.lower() == "none":
        bot_task = None
    tokenizer_bot_task = bot_task
    if tokenizer_bot_task == "think_recaption":
        tokenizer_bot_task = "think"
    elif tokenizer_bot_task == "vanilla":
        tokenizer_bot_task = "image"
    supported_bot_tasks = {"auto", "image", "think", "recaption", "img_ratio"}
    if tokenizer_bot_task is not None and tokenizer_bot_task not in supported_bot_tasks:
        raise ValueError(
            f"Unsupported HunyuanImage3 single-stage bot_task: {tokenizer_bot_task!r}. "
            f"Supported values are: {sorted(supported_bot_tasks)}."
        )
    return tokenizer_bot_task or "auto"


def normalize_hunyuan_cot_text(cot: str | None) -> str | None:
    """Restore an AR generation trigger tag omitted from generated text."""

    if not cot:
        return cot
    if "</think>" in cot and not cot.startswith("<think>"):
        return "<think>" + cot
    if "</recaption>" in cot and not cot.startswith("<recaption>"):
        return "<recaption>" + cot
    return cot


def extract_hunyuan_prompt_inputs(
    prompts: list[Any],
    extra_args: dict[str, Any],
    *,
    request_id: str,
    allow_cond_image: bool,
) -> tuple[list[str], list[str | None], str | None, list[list[JointImageInfo]] | None, str]:
    """Normalize request prompt fields shared by planning and execution."""

    is_dummy_warmup = OmniDiffusionRequest.is_dummy_run_request_id(request_id)
    bot_task = extra_args.get("bot_task")
    use_system_prompt = extra_args.get("use_system_prompt")
    system_prompt = extra_args.get("system_prompt")

    first_prompt = prompts[0] if prompts else None
    if isinstance(first_prompt, dict):
        if bot_task is None:
            bot_task = first_prompt.get("bot_task")
        if use_system_prompt is None:
            use_system_prompt = first_prompt.get("use_system_prompt")
        if system_prompt is None:
            system_prompt = first_prompt.get("system_prompt")
    tokenizer_bot_task = normalize_hunyuan_single_stage_bot_task(bot_task)
    if use_system_prompt is not None:
        system_prompt_bot_task = "image" if tokenizer_bot_task == "auto" else tokenizer_bot_task
        system_prompt = get_system_prompt(use_system_prompt, system_prompt_bot_task, system_prompt)
        system_prompt = system_prompt.strip() if system_prompt is not None else ""

    prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts]
    cot_text_list = [
        (p.get("extra", {}).get("ar_generated_text") if isinstance(p, dict) else None) or None for p in prompts
    ]

    batch_cond_image_info: list[list[JointImageInfo]] | None = None
    if any(not isinstance(p, str) for p in prompts):
        batch_cond_image_info = []
        for prompt_item in prompts:
            if isinstance(prompt_item, str):
                batch_cond_image_info.append([])
                continue
            additional_info = prompt_item.get("additional_information") or {}
            cond_infos = additional_info.get("batch_cond_image_info", [])
            if isinstance(cond_infos, JointImageInfo | dict):
                cond_infos = [cond_infos]
            if cond_infos is None:
                cond_infos = []
            batch_cond_image_info.append([_joint_image_info_from_payload(item) for item in cond_infos])

        has_cond_image = [len(cond_infos) > 0 for cond_infos in batch_cond_image_info]
        if any(has_cond_image) and (not allow_cond_image) and not is_dummy_warmup:
            raise ValueError("HunyuanImage3 step execution does not support image editing requests yet.")
        if allow_cond_image and any(has_cond_image) and not all(has_cond_image):
            raise ValueError("When batching Hunyuan image editing requests, every prompt must include input image(s).")
        if not allow_cond_image or not any(has_cond_image):
            batch_cond_image_info = None

    return prompt, cot_text_list, system_prompt, batch_cond_image_info, tokenizer_bot_task


def resolve_hunyuan_guidance_scale(sampling: Any, default_scale: float = 5.0) -> float:
    if getattr(sampling, "guidance_scale_provided", False):
        return sampling.guidance_scale
    return default_scale


def hunyuan_num_image_tokens(image_info: ImageInfo) -> int:
    """Return the generated-image span overwritten on every denoise step."""

    return int(image_info.image_token_length) + int(image_info.add_timestep_token) + int(image_info.add_guidance_token)


def build_hunyuan_batch_rope_image_info(
    output: TokenizerEncodeOutput,
    sections: list[list[dict[str, Any]]],
) -> list[list[tuple[slice, tuple[int, int]]]]:
    if output.all_image_slices is None:
        raise ValueError("Hunyuan tokenizer output is missing all_image_slices.")
    if len(output.all_image_slices) != len(sections):
        raise ValueError(
            "Hunyuan image-slice rows do not match template section rows: "
            f"image_slices={len(output.all_image_slices)}, sections={len(sections)}"
        )
    rope_image_info: list[list[tuple[slice, tuple[int, int]]]] = []
    for image_slices, sections_i in zip(output.all_image_slices, sections):
        image_shapes: list[tuple[int, int]] = []
        for section in sections_i:
            if "image" not in section["type"]:
                continue
            if isinstance(section["token_height"], list):
                if len(section["token_height"]) != len(section["token_width"]):
                    raise ValueError(
                        "token_height and token_width should have the same length, "
                        f"but got {len(section['token_height'])} and {len(section['token_width'])}"
                    )
                image_shapes.extend(
                    (int(token_height), int(token_width))
                    for token_height, token_width in zip(section["token_height"], section["token_width"])
                )
            else:
                image_shapes.append((int(section["token_height"]), int(section["token_width"])))
        if len(image_slices) != len(image_shapes):
            raise ValueError(f"Image slices({len(image_slices)}) do not match image shapes({len(image_shapes)}).")
        rope_image_info.append(list(zip(image_slices, image_shapes)))
    return rope_image_info


def prepare_hunyuan_layout(
    request: OmniDiffusionRequest,
    *,
    tokenizer_wrapper: TokenizerWrapper,
    image_processor: HunyuanImage3ImageProcessor,
    generation_config: GenerationConfig,
    image_base_size: int,
) -> HunyuanPreparedLayout:
    """Build the CPU token/image layout reused by Scheduler and Worker."""

    sampling = request.sampling_params
    extra_args = getattr(sampling, "extra_args", {}) or {}
    prompt, cot_text_list, system_prompt, batch_cond_image_info, tokenizer_bot_task = extract_hunyuan_prompt_inputs(
        [request.prompt],
        extra_args,
        request_id=request.request_id,
        allow_cond_image=True,
    )
    cot_text = (
        [normalize_hunyuan_cot_text(text) for text in cot_text_list]
        if any(text is not None for text in cot_text_list)
        else None
    )
    height = sampling.height or 1024
    width = sampling.width or 1024
    guidance_scale = resolve_hunyuan_guidance_scale(sampling)
    generated_image_info = image_processor.build_image_info((height, width))
    result = tokenizer_wrapper.apply_chat_template(
        batch_prompt=prompt,
        mode="gen_image",
        batch_gen_image_info=[generated_image_info],
        batch_cond_image_info=batch_cond_image_info,
        batch_system_prompt=[system_prompt],
        batch_cot_text=cot_text,
        max_length=None,
        bot_task=tokenizer_bot_task,
        image_base_size=image_base_size,
        sequence_template=getattr(generation_config, "sequence_template", "pretrain"),
        cfg_factor=1 + int(guidance_scale > 1.0),
        drop_think=getattr(generation_config, "drop_think", False),
    )
    tokenizer_output = result["output"]
    return HunyuanPreparedLayout(
        tokenizer_output=tokenizer_output,
        rope_image_info=build_hunyuan_batch_rope_image_info(tokenizer_output, result["sections"]),
        generated_image_info=generated_image_info,
    )


@dataclass(frozen=True)
class HunyuanPreparedLayout:
    """CPU execution layout prepared once before Scheduler admission."""

    tokenizer_output: TokenizerEncodeOutput
    rope_image_info: list[list[tuple[slice, tuple[int, int]]]]
    generated_image_info: ImageInfo

    def __post_init__(self) -> None:
        tokens = self.tokenizer_output.tokens
        if tokens is None or tokens.ndim != 2:
            shape = None if tokens is None else tuple(tokens.shape)
            raise ValueError(f"Hunyuan prepared tokenizer tokens must be 2-D, got {shape}")
        if len(self.rope_image_info) != int(tokens.shape[0]):
            raise ValueError(
                "Hunyuan prepared RoPE rows do not match tokenizer rows: "
                f"rope_rows={len(self.rope_image_info)}, tokens={int(tokens.shape[0])}"
            )
        prefix_positions = self.tokenizer_output.gen_timestep_scatter_index
        if prefix_positions is None or prefix_positions.ndim != 2 or prefix_positions.shape[1] == 0:
            raise ValueError("Hunyuan prepared layout requires a non-empty 2-D gen_timestep_scatter_index")
        if prefix_positions.shape[0] != int(tokens.shape[0]):
            raise ValueError(
                "Hunyuan prepared prefix-position rows do not match tokenizer rows: "
                f"prefix_positions={prefix_positions.shape[0]}, tokens={int(tokens.shape[0])}"
            )
        real_pos = self.tokenizer_output.real_pos
        if real_pos is None or real_pos.ndim != 2 or real_pos.shape[1] == 0:
            raise ValueError("Hunyuan prepared layout requires a non-empty 2-D real_pos")
        if real_pos.shape[0] != int(tokens.shape[0]):
            raise ValueError(
                "Hunyuan prepared valid-length rows do not match tokenizer rows: "
                f"real_pos={real_pos.shape[0]}, tokens={int(tokens.shape[0])}"
            )
        seq_lens = [int(row[-1].item()) for row in real_pos]
        if any(seq_len <= 0 or seq_len > int(tokens.shape[1]) for seq_len in seq_lens):
            raise ValueError(
                "Hunyuan prepared valid sequence lengths must fit the padded tokenizer width: "
                f"seq_lens={seq_lens}, padded_width={int(tokens.shape[1])}"
            )

    @property
    def num_branches(self) -> int:
        return int(self.tokenizer_output.tokens.shape[0])


def build_hunyuan_diffusion_kv_requests(
    request: OmniDiffusionRequest,
    prepared_layout: HunyuanPreparedLayout,
) -> tuple[DiffusionKVRequest, ...]:
    """Build one persistent Scheduler KV request per Hunyuan execution row."""

    tokenizer_output = prepared_layout.tokenizer_output
    cfg_factor = 1 + int(resolve_hunyuan_guidance_scale(request.sampling_params) > 1.0)
    if prepared_layout.num_branches != cfg_factor:
        raise ValueError(
            "Hunyuan tokenizer sequence count does not match CFG execution: "
            f"rows={prepared_layout.num_branches}, cfg_factor={cfg_factor}"
        )
    prefix_positions = tokenizer_output.gen_timestep_scatter_index
    real_pos = tokenizer_output.real_pos
    assert prefix_positions is not None and real_pos is not None

    target_len = hunyuan_num_image_tokens(prepared_layout.generated_image_info)
    return tuple(
        DiffusionKVRequest(
            f"{request.request_id}/diffusion-kv/{sequence_id}",
            sequence_id=sequence_id,
            # The generated-image timestep position terminates the reusable
            # prompt/reference-image prefix for this execution row.
            prefix_len=int(prefix_row[-1].item()),
            target_len=target_len,
            seq_len=int(valid_row[-1].item()),
            # Prompt and reference-image tokens are already embedded in this
            # row's primary self-attention sequence. Hunyuan therefore has no
            # independently projected cross/joint-attention KV context.
            kv_contexts=(),
        )
        for sequence_id, (prefix_row, valid_row) in enumerate(zip(prefix_positions, real_pos))
    )


def get_hunyuan_prepared_layout(source: Any) -> HunyuanPreparedLayout | None:
    prepared_layout = getattr(source, "prepared_layout", None)
    if prepared_layout is None:
        return None
    if not isinstance(prepared_layout, HunyuanPreparedLayout):
        raise TypeError(
            f"HunyuanImage3 expected prepared_layout to be HunyuanPreparedLayout, got {type(prepared_layout).__name__}"
        )
    return prepared_layout
