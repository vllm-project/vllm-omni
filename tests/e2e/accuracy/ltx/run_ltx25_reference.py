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
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--model-class-name")
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gemma-root", type=Path)
    parser.add_argument("--transformer-path", type=Path)
    parser.add_argument("--text-encoder-path", type=Path)
    parser.add_argument("--video-vae-path", type=Path)
    parser.add_argument("--audio-vae-path", type=Path)
    parser.add_argument("--spatial-upsampler-path", type=Path)
    parser.add_argument("--distilled-lora-path", type=Path)
    parser.add_argument("--connector-model", type=Path)
    parser.add_argument(
        "--official-pipeline",
        choices=("distilled_two_stage", "full_one_stage", "full_two_stage"),
        default="distilled_two_stage",
    )
    parser.add_argument("--enable-layerwise-offload", action="store_true")
    return parser.parse_args()


def _save_outputs(
    output_dir: Path,
    *,
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_sample_rate: int,
    fps: float,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    video = video.detach().float().cpu().clamp(0.0, 1.0)
    audio = audio.detach().float().cpu()
    np.save(output_dir / "video.npy", video.numpy())
    np.save(output_dir / "audio.npy", audio.numpy())

    from vllm_omni.diffusion.utils.media_utils import mux_video_audio_bytes

    frames_u8 = video.mul(255.0).round().to(torch.uint8).numpy()
    audio_array = audio.numpy()
    while audio_array.ndim > 2 and audio_array.shape[0] == 1:
        audio_array = audio_array[0]
    (output_dir / "output.mp4").write_bytes(
        mux_video_audio_bytes(
            frames_u8,
            audio_array,
            fps=float(fps),
            audio_sample_rate=int(audio_sample_rate),
            video_codec_options={"preset": "ultrafast"},
        )
    )

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
            "mp4": "output.mp4",
        }
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def _reset_accelerator_peak_memory_stats() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.reset_peak_memory_stats()


def _accelerator_peak_memory_stats() -> dict[str, float]:
    if not torch.accelerator.is_available():
        return {}
    torch.accelerator.synchronize()
    mib = 1024**2
    return {
        "peak_memory_allocated_mb": torch.accelerator.max_memory_allocated() / mib,
        "peak_memory_reserved_mb": torch.accelerator.max_memory_reserved() / mib,
    }


def _insert_official_paths(official_root: Path) -> None:
    for relative_path in ("packages/ltx-core/src", "packages/ltx-pipelines/src"):
        path = str((official_root / relative_path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def _configure_official_sdpa(pipeline: Any) -> None:
    """Pin official connector and denoiser attention to cuDNN, with MATH fallback."""
    from ltx_core.loader.attention_ops import set_attention_module_op
    from ltx_core.model.transformer.attention import PytorchAttention
    from torch.nn.attention import SDPBackend

    attention = PytorchAttention(priority=[SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH])
    module_op = set_attention_module_op(
        attention=attention,
        masked_attention=attention,
    )
    transformer_owners = [
        owner
        for attribute in ("stage", "stage_1", "stage_2")
        if (owner := getattr(pipeline, attribute, None)) is not None
    ]
    owners_and_attributes = [
        *((owner, "_transformer_builder") for owner in transformer_owners),
        (pipeline.prompt_encoder, "_embeddings_processor_builder"),
    ]
    for owner, attribute in owners_and_attributes:
        builder = getattr(owner, attribute)
        setattr(
            owner,
            attribute,
            builder.with_module_ops((*builder.module_ops, module_op)),
        )


def _use_diffusers_connector_weights(prompt_encoder: Any, model: Path) -> None:
    """Run official connector code with the checkpoint under test."""
    from safetensors import safe_open

    connector_root = model / "connectors"
    shards = sorted(connector_root.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"No connector safetensors found under {connector_root}")

    original_build_embeddings_processor = prompt_encoder._build_embeddings_processor

    def build_embeddings_processor():
        processor = original_build_embeddings_processor()
        parameters = dict(processor.named_parameters())
        expected = {name for name in parameters if name.startswith(("video_connector.", "audio_connector."))}
        loaded: set[str] = set()
        with torch.no_grad():
            for shard in shards:
                with safe_open(str(shard), framework="pt", device="cpu") as weights:
                    for source_name in weights.keys():
                        if not source_name.startswith(("video_connector.", "audio_connector.")):
                            continue
                        target_name = (
                            source_name.replace(".transformer_blocks.", ".transformer_1d_blocks.")
                            .replace(".attn1.norm_q.", ".attn1.q_norm.")
                            .replace(".attn1.norm_k.", ".attn1.k_norm.")
                        )
                        parameter = parameters[target_name]
                        parameter.copy_(weights.get_tensor(source_name).to(parameter.device, parameter.dtype))
                        loaded.add(target_name)
        if loaded != expected:
            missing = sorted(expected - loaded)
            unexpected = sorted(loaded - expected)
            raise ValueError(f"Connector weight mismatch: missing={missing}, unexpected={unexpected}")
        return processor

    prompt_encoder._build_embeddings_processor = build_embeddings_processor


@torch.inference_mode()
def _run_official(args: argparse.Namespace, request: dict[str, Any]) -> None:
    if args.official_root is None:
        raise ValueError("Official backend requires --official-root")
    _insert_official_paths(args.official_root)
    # Keep Gemma on the same SDPA path in both subprocesses. DiT attention is
    # still pinned independently to the explicit cuDNN backend below.
    torch.backends.cuda.enable_cudnn_sdp(False)
    _reset_accelerator_peak_memory_stats()

    from ltx_pipelines.utils.args import ImageConditioningInput
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import OffloadMode

    offload_mode = OffloadMode.CPU if args.enable_layerwise_offload else OffloadMode.NONE
    image_path = request.get("image")
    images = (
        []
        if image_path is None
        else [
            ImageConditioningInput(
                path=str(image_path),
                frame_idx=0,
                strength=1.0,
                crf=request.get("image_crf", 18),
            )
        ]
    )

    if args.transformer_path is not None:
        required = {
            "--text-encoder-path": args.text_encoder_path,
            "--video-vae-path": args.video_vae_path,
            "--audio-vae-path": args.audio_vae_path,
        }
        if args.official_pipeline in {"distilled_two_stage", "full_two_stage"}:
            required["--spatial-upsampler-path"] = args.spatial_upsampler_path
        if args.official_pipeline == "full_two_stage":
            required["--distilled-lora-path"] = args.distilled_lora_path
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"Official LTX-2.5 split backend is missing: {', '.join(missing)}")

        model_paths = ModelPaths.from_split(
            transformer_path=str(args.transformer_path),
            text_encoder_path=str(args.text_encoder_path),
            video_vae_path=str(args.video_vae_path),
            audio_vae_path=str(args.audio_vae_path),
        )
        if args.official_pipeline == "distilled_two_stage":
            from ltx_pipelines.distilled import DistilledPipeline

            pipeline = DistilledPipeline(
                model_paths=model_paths,
                spatial_upsampler_path=str(args.spatial_upsampler_path),
                loras=(),
                offload_mode=offload_mode,
            )
            if args.connector_model is not None:
                _use_diffusers_connector_weights(pipeline.prompt_encoder, args.connector_model)
            _configure_official_sdpa(pipeline)
            video, audio, _, _ = pipeline(
                prompt=request["prompt"],
                seed=request["seed"],
                height=request["height"],
                width=request["width"],
                num_frames=request["num_frames"],
                frame_rate=request["fps"],
                images=images,
            )
            pipeline_name = "DistilledPipeline"
        elif args.official_pipeline == "full_two_stage":
            from ltx_core.components.guiders import MultiModalGuiderParams
            from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

            pipeline = TI2VidTwoStagesPipeline(
                model_paths=model_paths,
                distilled_lora=[
                    LoraPathStrengthAndSDOps(
                        path=str(args.distilled_lora_path),
                        strength=1.0,
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                ],
                spatial_upsampler_path=str(args.spatial_upsampler_path),
                loras=[],
                offload_mode=offload_mode,
            )
            if args.connector_model is not None:
                _use_diffusers_connector_weights(pipeline.prompt_encoder, args.connector_model)
            _configure_official_sdpa(pipeline)
            video, audio, _, _ = pipeline(
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
                stage_1_sigmas=torch.tensor(request["stage_1_sigmas"], dtype=torch.float32),
                stage_2_sigmas=torch.tensor(request["stage_2_sigmas"], dtype=torch.float32),
            )
            pipeline_name = "TI2VidTwoStagesPipeline"
        else:
            from ltx_core.components.guiders import MultiModalGuiderParams
            from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

            pipeline = TI2VidOneStagePipeline(
                model_paths=model_paths,
                loras=(),
                offload_mode=offload_mode,
            )
            if args.connector_model is not None:
                _use_diffusers_connector_weights(pipeline.prompt_encoder, args.connector_model)
            _configure_official_sdpa(pipeline)
            request_sigmas = request.get("sigmas")
            video, audio, _ = pipeline(
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
                sigmas=None if request_sigmas is None else torch.tensor(request_sigmas, dtype=torch.float32),
            )
            pipeline_name = "TI2VidOneStagePipeline"
        checkpoint_metadata = {
            "transformer_path": str(args.transformer_path),
            "video_vae_path": str(args.video_vae_path),
            "connector_model": None if args.connector_model is None else str(args.connector_model),
            "pipeline": pipeline_name,
        }
    else:
        if args.checkpoint is None or args.gemma_root is None:
            raise ValueError("Official monolith backend requires --checkpoint and --gemma-root")

        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

        pipeline = TI2VidOneStagePipeline(
            model_paths=ModelPaths.from_monolith(str(args.checkpoint), str(args.gemma_root)),
            loras=(),
            offload_mode=offload_mode,
        )
        _configure_official_sdpa(pipeline)
        video, audio, _ = pipeline(
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
        checkpoint_metadata = {
            "checkpoint": str(args.checkpoint),
            "pipeline": "TI2VidOneStagePipeline",
        }

    video_tensor = torch.cat([chunk.detach().cpu() for chunk in video], dim=0)
    _save_outputs(
        args.output_dir,
        video=_canonical_video(video_tensor),
        audio=audio.waveform,
        audio_sample_rate=audio.sampling_rate,
        fps=float(request["fps"]),
        metadata={
            "backend": "official",
            "attention_backend": "torch_sdpa_cudnn_bf16",
            "official_revision": os.environ.get("VLLM_TEST_LTX_OFFICIAL_REVISION"),
            "seed": request["seed"],
            "sigmas": request.get("sigmas"),
            "stage_1_sigmas": request.get("stage_1_sigmas"),
            "stage_2_sigmas": request.get("stage_2_sigmas"),
            **_accelerator_peak_memory_stats(),
            **checkpoint_metadata,
        },
    )


def _omni_worker_peak_memory_mb(output: Any) -> float:
    result = output[0] if isinstance(output, list) and output else output
    return float(getattr(result, "peak_memory_mb", 0.0) or 0.0)


def _unwrap_omni_output(output: Any) -> tuple[Any, Any, int]:
    from vllm_omni.outputs import OmniRequestOutput

    audio = None
    audio_sample_rate = None
    frames = output[0] if isinstance(output, list) and output else output
    if isinstance(frames, OmniRequestOutput):
        multimodal_output = frames.multimodal_output or {}
        audio = multimodal_output.get("audio")
        audio_sample_rate = multimodal_output.get("audio_sample_rate")
        nested_output = getattr(frames, "request_output", None)
        if frames.is_pipeline_output and isinstance(nested_output, OmniRequestOutput):
            frames = nested_output
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

    # vLLM imports can otherwise change the process-wide Gemma dispatch.
    torch.backends.cuda.enable_cudnn_sdp(False)
    _reset_accelerator_peak_memory_stats()

    attention_config = {"default": {"backend": "CUDNN_ATTN"}}

    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams
    from vllm_omni.model_extras import get_extra_body_params
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
        # The explicit public class argument selects the worker pipeline. The
        # parent Omni proxy can only expose the checkpoint's base class, which
        # is insufficient to distinguish LTX execution modes.
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
        extra_args = {
            key: value for key, value in request.items() if key.startswith("video_") or key.startswith("audio_")
        }
        for key in ("sigmas", "stage_1_sigmas", "stage_2_sigmas", "image_crf"):
            if key in request:
                extra_args[key] = request[key]
        apply_declared_extra_args(sampling_params, get_extra_body_params(model_class_name), extra_args)
        prompt: dict[str, Any] = {"prompt": request["prompt"]}
        if "negative_prompt" in request:
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
            fps=float(request["fps"]),
            metadata={
                "backend": "omni",
                "attention_backend": "torch_sdpa_cudnn_bf16",
                "seed": request["seed"],
                "sigmas": request.get("sigmas"),
                "model": args.model,
                "stage_1_sigmas": request.get("stage_1_sigmas"),
                "stage_2_sigmas": request.get("stage_2_sigmas"),
                "model_class_name": model_class_name,
                "worker_peak_memory_mb": _omni_worker_peak_memory_mb(output),
                **_accelerator_peak_memory_stats(),
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
