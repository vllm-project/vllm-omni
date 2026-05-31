# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for QwenImage disaggregated VAE."""

from __future__ import annotations

from typing import Any

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt


def _read_mm(output: Any, stage_label: str, req_idx: int) -> dict[str, Any]:
    mm = getattr(output, "multimodal_output", None)
    if not mm or not isinstance(mm, dict):
        raise RuntimeError(
            f"[qwen_image.{stage_label}] upstream req#{req_idx} is missing multimodal_output (got {type(mm).__name__})."
        )
    return mm


def _require(mm: dict[str, Any], key: str, stage_label: str, req_idx: int) -> Any:
    if key not in mm or mm[key] is None:
        raise RuntimeError(
            f"[qwen_image.{stage_label}] upstream req#{req_idx} missing required key {key!r}; have {sorted(mm.keys())}."
        )
    return mm[key]


def encode_to_denoise(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | dict | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data
    denoise_inputs: list[OmniTokensPrompt] = []
    for i, enc_out in enumerate(source_outputs):
        mm = _read_mm(enc_out, "encode_to_denoise", i)
        info: dict[str, Any] = {
            "context": _require(mm, "context", "encode_to_denoise", i),
            "context_mask": _require(mm, "context_mask", "encode_to_denoise", i),
            "context_null": mm.get("context_null"),
            "context_null_mask": mm.get("context_null_mask"),
            "latents": _require(mm, "latents", "encode_to_denoise", i),
            "timesteps": _require(mm, "timesteps", "encode_to_denoise", i),
            "sigmas": mm.get("sigmas"),
            "num_inference_steps": mm.get("num_inference_steps"),
            "height": _require(mm, "height", "encode_to_denoise", i),
            "width": _require(mm, "width", "encode_to_denoise", i),
            "img_shapes": _require(mm, "img_shapes", "encode_to_denoise", i),
            "guidance": mm.get("guidance"),
            "guidance_scale": mm.get("guidance_scale", 1.0),
            "true_cfg_scale": mm.get("true_cfg_scale", 1.0),
            "do_true_cfg": bool(mm.get("do_true_cfg", False)),
            "txt_seq_lens": mm.get("txt_seq_lens"),
            "negative_txt_seq_lens": mm.get("negative_txt_seq_lens"),
        }
        if mm.get("image_latents") is not None:
            info["image_latents"] = mm["image_latents"]
        denoise_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[],
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return denoise_inputs


def denoise_to_decode(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | dict | list | None = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    del prompt, requires_multimodal_data
    decode_inputs: list[OmniTokensPrompt] = []
    for i, den_out in enumerate(source_outputs):
        mm = _read_mm(den_out, "denoise_to_decode", i)
        info: dict[str, Any] = {
            "latents": _require(mm, "latents", "denoise_to_decode", i),
            "height": _require(mm, "height", "denoise_to_decode", i),
            "width": _require(mm, "width", "denoise_to_decode", i),
            "output_type": mm.get("output_type", "pil"),
        }
        decode_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[],
                additional_information=info,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return decode_inputs
