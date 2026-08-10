# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Generate raw LTX video and audio outputs with the official or Omni runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("official", "omni"), required=True)
    parser.add_argument(
        "--pipeline-kind",
        choices=("one_stage", "distilled", "two_stage"),
        default="one_stage",
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--model-class-name")
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gemma-root", type=Path)
    parser.add_argument("--spatial-upsampler", type=Path)
    parser.add_argument("--distilled-lora", type=Path)
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    return parser.parse_args()


def _save_outputs(
    output_dir: Path,
    *,
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_sample_rate: int,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video = video.detach().float().cpu().clamp(0.0, 1.0)
    audio = audio.detach().float().cpu()
    np.save(output_dir / "video.npy", video.numpy())
    np.save(output_dir / "audio.npy", audio.numpy())

    frame_indices = sorted({0, video.shape[0] // 2, video.shape[0] - 1})
    for index in frame_indices:
        frame = video[index].mul(255.0).round().to(torch.uint8).numpy()
        Image.fromarray(frame).save(output_dir / f"frame_{index:04d}.png")

    metadata.update(
        {
            "video_shape": list(video.shape),
            "audio_shape": list(audio.shape),
            "audio_sample_rate": int(audio_sample_rate),
            "frame_indices": frame_indices,
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _insert_official_paths(official_root: Path) -> None:
    for relative_path in ("packages/ltx-core/src", "packages/ltx-pipelines/src"):
        path = str((official_root / relative_path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def _configure_official_sdpa(pipeline: Any) -> None:
    """Use PyTorch SDPA for official connector and denoiser attention."""
    from ltx_core.loader.attention_ops import set_attention_module_op
    from ltx_core.model.transformer.attention import PytorchAttention

    class AllValidSDPA(PytorchAttention):
        def __call__(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            heads: int,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if mask is not None and torch.count_nonzero(mask).item():
                raise ValueError("The LTX accuracy guard cannot discard a non-empty attention mask")
            return super().__call__(query, key, value, heads, mask=None)

    attention = AllValidSDPA()
    module_op = set_attention_module_op(
        attention=attention,
        masked_attention=attention,
    )
    owners_and_attributes = [
        (pipeline.prompt_encoder, "_embeddings_processor_builder"),
    ]
    owners_and_attributes.extend(
        (stage, "_transformer_builder")
        for name in ("stage", "stage_1", "stage_2")
        if (stage := getattr(pipeline, name, None)) is not None
    )
    for owner, attribute in owners_and_attributes:
        builder = getattr(owner, attribute)
        setattr(
            owner,
            attribute,
            builder.with_module_ops((*builder.module_ops, module_op)),
        )


def _require_path(path: Path | None, description: str) -> str:
    if path is None or not path.is_file():
        raise ValueError(f"Official {description} is required and must exist: {path}")
    return str(path)


@torch.inference_mode()
def _run_official(args: argparse.Namespace, request: dict[str, Any]) -> None:
    if args.official_root is None or args.checkpoint is None or args.gemma_root is None:
        raise ValueError("Official backend requires --official-root, --checkpoint, and --gemma-root")
    _insert_official_paths(args.official_root)
    # vLLM disables cuDNN SDPA during import; mirror the worker's Gemma dispatch.
    torch.backends.cuda.enable_cudnn_sdp(False)

    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.types import OffloadMode

    if request.get("pipeline_kind", "one_stage") != args.pipeline_kind:
        raise ValueError(
            f"Request pipeline kind {request.get('pipeline_kind')!r} does not match {args.pipeline_kind!r}"
        )
    checkpoint = _require_path(args.checkpoint, "checkpoint")
    gemma_root = str(args.gemma_root)
    if not Path(gemma_root).is_dir():
        raise ValueError(f"Official Gemma root is required and must be a directory: {gemma_root}")
    offload_mode = OffloadMode.CPU if args.enable_layerwise_offload else OffloadMode.NONE
    pipeline: Any
    if args.pipeline_kind == "one_stage":
        pipeline = TI2VidOneStagePipeline(
            checkpoint_path=checkpoint,
            gemma_root=gemma_root,
            loras=(),
            offload_mode=offload_mode,
        )
    elif args.pipeline_kind == "distilled":
        pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint,
            spatial_upsampler_path=_require_path(args.spatial_upsampler, "spatial upsampler"),
            gemma_root=gemma_root,
            loras=(),
            offload_mode=offload_mode,
        )
    else:
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                path=_require_path(args.distilled_lora, "distilled LoRA"),
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=checkpoint,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=_require_path(args.spatial_upsampler, "spatial upsampler"),
            gemma_root=gemma_root,
            loras=(),
            offload_mode=offload_mode,
        )
    _configure_official_sdpa(pipeline)
    image_path = request.get("image")
    images = (
        []
        if image_path is None
        else [
            ImageConditioningInput(
                path=str(image_path),
                frame_idx=0,
                strength=1.0,
                # Both runtimes must receive the same source pixels. The
                # official CLI's optional H.264 preprocessing is not part of
                # the seeded model trajectory under test.
                crf=0,
            )
        ]
    )
    if args.pipeline_kind == "distilled":
        video, audio = pipeline(
            prompt=request["prompt"],
            seed=request["seed"],
            height=request["height"],
            width=request["width"],
            num_frames=request["num_frames"],
            frame_rate=request["fps"],
            images=images,
        )
    else:
        video, audio = pipeline(
            prompt=request["prompt"],
            negative_prompt=request["negative_prompt"],
            seed=request["seed"],
            height=request["height"],
            width=request["width"],
            num_frames=request["num_frames"],
            frame_rate=request["fps"],
            num_inference_steps=request["num_inference_steps"],
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=request["video_cfg_scale"],
                stg_scale=request["video_stg_scale"],
                rescale_scale=request["video_rescale_scale"],
                modality_scale=request["video_modality_scale"],
                skip_step=0,
                stg_blocks=request["video_stg_blocks"],
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=request["audio_cfg_scale"],
                stg_scale=request["audio_stg_scale"],
                rescale_scale=request["audio_rescale_scale"],
                modality_scale=request["audio_modality_scale"],
                skip_step=0,
                stg_blocks=request["audio_stg_blocks"],
            ),
            images=images,
            max_batch_size=4,
        )
    video_tensor = torch.cat([chunk.detach().cpu() for chunk in video], dim=0)
    _save_outputs(
        args.output_dir,
        video=_canonical_video(video_tensor),
        audio=audio.waveform,
        audio_sample_rate=audio.sampling_rate,
        metadata={
            "backend": "official",
            "attention_backend": "torch_sdpa",
            "official_revision": os.environ.get("VLLM_TEST_LTX_OFFICIAL_REVISION"),
            "checkpoint": str(args.checkpoint),
            "pipeline_kind": args.pipeline_kind,
        },
    )


def _unwrap_omni_output(output: Any) -> tuple[Any, Any, int]:
    from vllm_omni.outputs import OmniRequestOutput

    audio = None
    audio_sample_rate = None
    frames = output[0] if isinstance(output, list) and output else output
    if isinstance(frames, OmniRequestOutput):
        multimodal_output = frames.multimodal_output or {}
        audio = multimodal_output.get("audio")
        audio_sample_rate = multimodal_output.get("audio_sample_rate")
        if frames.is_pipeline_output and isinstance(frames.request_output, OmniRequestOutput):
            frames = frames.request_output
            multimodal_output = frames.multimodal_output or {}
            audio = multimodal_output.get("audio", audio)
            audio_sample_rate = multimodal_output.get("audio_sample_rate", audio_sample_rate)
        if isinstance(frames, OmniRequestOutput):
            if not frames.images:
                raise ValueError("No video frames found in OmniRequestOutput")
            frames = frames.images

    if isinstance(frames, list) and len(frames) == 1:
        frames = frames[0]
    if isinstance(frames, tuple) and len(frames) == 2:
        frames, audio = frames
    elif isinstance(frames, dict):
        audio = frames.get("audio", audio)
        audio_sample_rate = frames.get("audio_sample_rate", audio_sample_rate)
        frames = frames.get("frames", frames.get("video"))

    if frames is None or audio is None or audio_sample_rate is None:
        raise ValueError("Omni output did not contain video, audio, and audio_sample_rate")
    return frames, audio, int(audio_sample_rate)


def _canonical_video(video: Any) -> torch.Tensor:
    if isinstance(video, list):
        if len(video) == 1:
            return _canonical_video(video[0])
        tensors = [torch.as_tensor(np.asarray(item) if not isinstance(item, torch.Tensor) else item) for item in video]
        if all(item.ndim == 3 for item in tensors):
            video = torch.stack(tensors)
        elif all(item.ndim == 4 for item in tensors):
            video = torch.cat(tensors)
        else:
            raise ValueError(f"Cannot combine video items with dimensions {[item.ndim for item in tensors]}")
    tensor = torch.as_tensor(np.asarray(video) if not isinstance(video, torch.Tensor) else video).detach().cpu()
    if tensor.ndim == 5:
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(f"Expected a 4D video tensor, got {tuple(tensor.shape)}")
    if tensor.shape[-1] in (3, 4):
        tensor = tensor[..., :3]
    elif tensor.shape[1] in (3, 4):
        tensor = tensor[:, :3].permute(0, 2, 3, 1)
    elif tensor.shape[0] in (3, 4):
        tensor = tensor[:3].permute(1, 2, 3, 0)
    else:
        raise ValueError(f"Cannot infer video channel dimension from {tuple(tensor.shape)}")
    tensor = tensor.float()
    if tensor.numel() and tensor.max() > 1:
        tensor = tensor / 255.0
    elif tensor.numel() and tensor.min() < 0:
        tensor = tensor.clamp(-1.0, 1.0).add(1.0).mul(0.5)
    return tensor.clamp(0.0, 1.0)


@torch.inference_mode()
def _run_omni(args: argparse.Namespace, request: dict[str, Any]) -> None:
    if args.model is None or args.model_class_name is None:
        raise ValueError("Omni backend requires --model and --model-class-name")
    if request.get("pipeline_kind", "one_stage") != args.pipeline_kind:
        raise ValueError(
            f"Request pipeline kind {request.get('pipeline_kind')!r} does not match {args.pipeline_kind!r}"
        )

    attention_config = {"default": {"backend": "TORCH_SDPA"}}

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import get_extra_body_params, get_model_class_name
    from vllm_omni.platforms import current_omni_platform

    generator = torch.Generator(device=current_omni_platform.device_type).manual_seed(request["seed"])
    omni = Omni(
        model=args.model,
        model_class_name=args.model_class_name,
        enforce_eager=True,
        enable_layerwise_offload=args.enable_layerwise_offload,
        diffusion_attention_config=attention_config,
        parallel_config=DiffusionParallelConfig(),
    )
    try:
        detected_model_class_name = get_model_class_name(omni)
        model_class_name = args.model_class_name
        sampling_params = OmniDiffusionSamplingParams(
            height=request["height"],
            width=request["width"],
            generator=generator,
            guidance_scale=None,
            num_inference_steps=request["num_inference_steps"],
            num_frames=request["num_frames"],
            fps=request["fps"],
            frame_rate=float(request["fps"]),
            output_type="np",
        )
        guidance = {
            key: value for key, value in request.items() if key.startswith("video_") or key.startswith("audio_")
        }
        apply_declared_extra_args(sampling_params, get_extra_body_params(model_class_name), guidance)
        prompt: dict[str, Any] = {"prompt": request["prompt"]}
        if request.get("negative_prompt"):
            prompt["negative_prompt"] = request["negative_prompt"]
        image_path = request.get("image")
        if image_path is not None:
            with Image.open(str(image_path)) as source_image:
                image = source_image.convert("RGB")
                image.load()
            prompt["multi_modal_data"] = {"image": image}
        output = omni.generate(prompt, sampling_params)
        video, audio, audio_sample_rate = _unwrap_omni_output(output)
        audio_tensor = torch.as_tensor(np.asarray(audio) if not isinstance(audio, torch.Tensor) else audio)
        _save_outputs(
            args.output_dir,
            video=_canonical_video(video),
            audio=audio_tensor,
            audio_sample_rate=audio_sample_rate,
            metadata={
                "backend": "omni",
                "attention_backend": "torch_sdpa",
                "model": args.model,
                "model_class_name": model_class_name,
                "model_config_class_name": detected_model_class_name,
                "pipeline_kind": args.pipeline_kind,
            },
        )
    finally:
        omni.shutdown()


def main() -> None:
    args = _parse_args()
    request = json.loads(args.request.read_text())
    if args.backend == "official":
        _run_official(args, request)
    else:
        _run_omni(args, request)


if __name__ == "__main__":
    main()
