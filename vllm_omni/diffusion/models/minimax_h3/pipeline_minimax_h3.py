# SPDX-License-Identifier: Apache-2.0
"""vLLM-Omni pipeline for MiniMax H3 FL2VA and Ref2VA partitions."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3VLProcessor
from vllm.logger import init_logger

from vllm_omni.diffusion import envs
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    get_dit_group,
    init_world_group,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import (
    DiffusersPipelineLoader,
)
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportAudioOutput,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.platforms import current_omni_platform

from .condition_noise import (
    minimax_h3_audio_cond_noise_aug_rows,
    minimax_h3_imgvid_cond_noise_aug_rows,
)
from .denoise_loop import MiniMaxH3DenoiseBranch, minimax_h3_denoise_loop
from .encoder import MiniMaxH3Qwen3VLEncoder
from .minimax_h3_transformer import MiniMaxH3DiTModel
from .packed_sequence import (
    minimax_h3_packed_sequence,
    minimax_h3_packed_sequence_ref2va_blocks,
)
from .packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from .presentation import (
    minimax_h3_multi_image_presentation_ids,
    minimax_h3_multi_image_presentation_token_tags,
    minimax_h3_ref2va_presentation,
    minimax_h3_ref2va_video_presentation,
    minimax_h3_text_only_ids,
)
from .reference_video import (
    load_audio_file,
    load_video_audio,
    load_video_frames,
    prepare_reference_videos,
    sample_reference_video_frames,
)
from .time_request import (
    MINIMAX_H3_SHAPE_PLANNER,
    minimax_h3_align_frame_count,
    minimax_h3_time_shift_sigmas,
)
from .vae import MiniMaxH3AudioVAE, MiniMaxH3VideoVAE

logger = init_logger(__name__)

MINIMAX_H3_FPS = 24
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_IMGVID_COND_TIMESTEP = 0.999
MINIMAX_H3_AUDIO_REF_COND_TIMESTEP = 1.0
MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32


def _minimax_h3_post_process(output, output_type: str = "np"):
    """Convert the joint video/audio output without capturing worker state.

    The callable crosses the multiprocessing result queue, so it must remain a
    module-level function that the standard pickle module can resolve.
    """
    if not isinstance(output, tuple) or len(output) != 2:
        return output
    video, audio = output
    if output_type == "latent":
        return output
    if output_type == "np":
        video = video.detach().float().cpu().permute(0, 2, 3, 4, 1).clamp(0, 1).numpy()
        audio = audio.detach().float().cpu().numpy()
        video = [sample for sample in video]
    return {
        "video": video,
        "audio": audio,
        "audio_sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE,
        "fps": MINIMAX_H3_FPS,
    }


def get_minimax_h3_post_process_func(
    od_config: OmniDiffusionConfig,
):
    del od_config
    return _minimax_h3_post_process


def _align_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _load_image(value: Any) -> Image.Image:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("MiniMax H3 currently supports exactly one image")
        value = value[0]
    if isinstance(value, (str, os.PathLike)):
        return Image.open(value).convert("RGB")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().cpu()
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        if tensor.ndim != 3:
            raise ValueError(f"image tensor must be [C,H,W], got {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        return Image.fromarray(array.clip(0, 255).astype(np.uint8)).convert("RGB")
    raise TypeError(f"unsupported MiniMax H3 image input {type(value)!r}")


def _load_audio(value: Any) -> tuple[torch.Tensor, int]:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError("MiniMax H3 currently supports exactly one audio")
        value = value[0]
    if isinstance(value, (str, os.PathLike)):
        return load_audio_file(str(value))
    if isinstance(value, tuple) and len(value) == 2:
        waveform, sample_rate = value
        waveform = torch.as_tensor(waveform).float()
        return waveform, int(sample_rate)
    if isinstance(value, dict):
        waveform = value.get("waveform", value.get("array"))
        sample_rate = value.get("sample_rate", value.get("sampling_rate"))
        if waveform is not None and sample_rate is not None:
            return torch.as_tensor(waveform).float(), int(sample_rate)
    raise TypeError("MiniMax H3 audio input must be a path, (waveform, sample_rate), or a waveform mapping")


def _dit_rank_world() -> tuple[Any, int, int]:
    if not dist.is_initialized():
        return None, 0, 1
    group = get_dit_group()
    return group, dist.get_rank(group), dist.get_world_size(group)


def _broadcast_tensor(
    tensor: torch.Tensor | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    group, rank, world_size = _dit_rank_world()
    if world_size == 1:
        if tensor is None:
            raise ValueError("source tensor is required for single-rank execution")
        return tensor.to(device=device, dtype=dtype)

    shape = torch.zeros(5, dtype=torch.long, device=device)
    if rank == 0:
        if tensor is None:
            raise ValueError("rank 0 must provide a tensor to broadcast")
        shape[0] = tensor.ndim
        shape[1 : tensor.ndim + 1] = torch.tensor(
            tensor.shape,
            device=device,
        )
    dist.broadcast(shape, src=0, group=group)
    ndim = int(shape[0].item())
    tensor_shape = tuple(int(v) for v in shape[1 : ndim + 1].tolist())
    if rank == 0:
        output = tensor.to(device=device, dtype=dtype).contiguous()
    else:
        output = torch.empty(tensor_shape, device=device, dtype=dtype)
    dist.broadcast(output, src=0, group=group)
    return output


def _reference_image_shape(image: Image.Image) -> tuple[int, int]:
    width, height = image.size
    if width > 4 * height or height > 4 * width:
        raise ValueError(f"reference image aspect ratio must be in [1:4, 4:1], got {width}x{height}")
    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(width, height)
    return (
        _align_multiple(
            width * scale,
            MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
        ),
        _align_multiple(
            height * scale,
            MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
        ),
    )


class _SingleRankEncoderGroup:
    """Lightweight encoder group for ``text_encoder_tp_size == 1``.

    Avoids creating a distributed ``GroupCoordinator`` with a single-member
    rank set, which would assert on every other DiT rank that is not part of
    the group.  The pipeline and encoder only use the attributes below, and
    all ``world_size == 1`` code paths short-circuit before any collective.
    """

    world_size: int = 1
    ranks: list[int] = [0]

    def __init__(self, rank: int) -> None:
        self.rank_in_group = 0 if rank == 0 else -1
        self.device_group = None


class MiniMaxH3Pipeline(
    nn.Module,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    SupportImageInput,
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
):
    """CFG-distilled joint video/audio generation for MiniMax H3."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder"]
    _vae_modules: ClassVar[list[str]] = ["video_vae", "audio_vae"]
    _PROFILER_TARGETS: ClassVar[list[str]] = [
        "_prepare_reference_videos",
        "encode_prompt",
        "_encode_video_conditions",
        "_encode_video_audio_conditions",
        "diffuse",
        "decode",
    ]
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ) -> None:
        del prefix
        super().__init__()
        self.od_config = od_config
        self.parallel_config = od_config.parallel_config
        if int(self.parallel_config.cfg_parallel_size) != 1:
            raise ValueError("MiniMax-H3 is CFG-distilled and has no negative branch; cfg_parallel_size must be 1")
        self.device = get_local_device()
        model_path = str(od_config.model)
        model_index = json.loads((Path(model_path) / "model_index.json").read_text(encoding="utf-8"))
        release = model_index.get("_minimax_h3") or {}
        self.partition = str(release.get("partition", ""))
        self.supported_tasks = frozenset(release.get("tasks") or ())
        shifts = release.get("sigma_shift_scales") or {}
        self.default_video_shift = float(shifts.get("video", 12.0))
        self.default_audio_shift = float(shifts.get("audio", 3.0))

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]
        self.transformer = MiniMaxH3DiTModel(
            od_config,
            quant_config=od_config.quantization_config,
        )

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_path,
            subfolder="tokenizer",
            local_files_only=os.path.isdir(model_path),
        )
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            subfolder="processor",
            local_files_only=os.path.isdir(model_path),
        )

        _, rank, dit_world = _dit_rank_world()
        self._dit_rank = rank
        text_encoder_tp_size = int(getattr(self.parallel_config, "text_encoder_tp_size", 1))
        if text_encoder_tp_size < 1:
            raise ValueError(f"text_encoder_tp_size must be >= 1, got {text_encoder_tp_size}")
        if text_encoder_tp_size > dit_world:
            raise ValueError(
                f"text_encoder_tp_size must not exceed the DiT group size ({dit_world}), got {text_encoder_tp_size}"
            )
        # The Qwen3-VL text model uses 64 attention heads / 8 KV heads; the
        # encoder shards them across the encoder TP ranks.
        if 64 % text_encoder_tp_size or 8 % text_encoder_tp_size:
            raise ValueError(
                "text_encoder_tp_size must divide both Qwen3-VL "
                f"num_attention_heads (64) and num_key_value_heads (8), "
                f"got {text_encoder_tp_size}"
            )
        self.text_encoder_tp_size = text_encoder_tp_size
        self.text_encoder_group = self._build_text_encoder_group(text_encoder_tp_size)
        self.text_encoder = MiniMaxH3Qwen3VLEncoder(
            os.path.join(model_path, "text_encoder"),
            device=self.device,
            load_model=rank < text_encoder_tp_size,
            encoder_group=self.text_encoder_group,
        )
        self.video_vae = MiniMaxH3VideoVAE(
            os.path.join(model_path, "video_vae"),
            device=self.device,
        )
        self.audio_vae = MiniMaxH3AudioVAE(
            os.path.join(model_path, "audio_vae"),
            device=self.device,
        )
        # Registry-side VAE patch-parallel discovery uses ``pipeline.vae``.
        self.vae = self.video_vae

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=(od_config.enable_diffusion_pipeline_profiler)
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        prefix = "transformer."

        def transformer_weights():
            for name, tensor in weights:
                if name.startswith(prefix):
                    yield name[len(prefix) :], tensor

        loaded = self.transformer.load_weights(transformer_weights())
        self.transformer.post_load_weights()
        loaded_with_prefix = {prefix + name for name in loaded}
        # The text encoder and both VAEs load eagerly in ``__init__`` rather
        # than through ``weights_sources``. Record them for the runner's strict
        # missing-parameter check.
        for component_name in ("text_encoder", "video_vae", "audio_vae"):
            component = getattr(self, component_name)
            loaded_with_prefix.update(f"{component_name}.{name}" for name, _ in component.named_parameters())
        return loaded_with_prefix

    def _resolve_task(
        self,
        requested: str | None,
        multi_modal_data: dict[str, Any],
    ) -> str:
        if requested is None:
            if self.partition == "ref2va":
                requested = "ref2va"
            elif multi_modal_data.get("image") is not None:
                requested = "fl2va"
            else:
                requested = "t2va"
        task = str(requested).lower()
        if task not in self.supported_tasks:
            raise ValueError(
                f"checkpoint partition {self.partition!r} supports {sorted(self.supported_tasks)}, got task={task!r}"
            )
        return task

    def _resolve_shape(
        self,
        task: str,
        sampling: Any,
        image: Image.Image | None,
    ) -> tuple[int, int, int, int, int]:
        fps = int(sampling.fps or MINIMAX_H3_FPS)
        if fps != MINIMAX_H3_FPS:
            raise ValueError(f"MiniMax H3 output fps is fixed at {MINIMAX_H3_FPS}")
        extra = sampling.extra_args or {}
        duration = extra.get("duration")
        if duration is not None:
            requested_frames = int(round(float(duration) * fps))
        elif int(sampling.num_frames or 1) > 1:
            requested_frames = int(sampling.num_frames)
        else:
            requested_frames = 124 if task == "ref2va" else 209
        num_frames = minimax_h3_align_frame_count(requested_frames)

        height = sampling.height
        width = sampling.width
        if height is None or width is None:
            if task == "fl2va" and image is not None:
                ratio = image.width / image.height
                if ratio >= 1:
                    height = 768
                    width = _align_multiple(768 * ratio)
                else:
                    width = 768
                    height = _align_multiple(768 / ratio)
            else:
                height, width = 768, 1344
        height = int(height) // 32 * 32
        width = int(width) // 32 * 32
        if min(height, width) <= 0:
            raise ValueError(f"invalid MiniMax H3 canvas {width}x{height}")
        if width > 4 * height or height > 4 * width:
            raise ValueError("MiniMax H3 canvas aspect ratio must be in [1:4, 4:1]")

        latent_t = MINIMAX_H3_SHAPE_PLANNER.video_latent_t(num_frames)
        audio_t = MINIMAX_H3_SHAPE_PLANNER.audio_latent_t(num_frames / fps)
        return height, width, num_frames, latent_t, audio_t

    def encode_prompt(
        self,
        *,
        task: str,
        prompt: str,
        image: Image.Image | None,
        prepared_videos: list[dict[str, Any]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, rank, _ = _dit_rank_world()
        hidden = None
        tags = None
        ids = None
        vision_kwargs: dict[str, torch.Tensor] = {}
        if rank == 0:
            if task == "t2va":
                ids = minimax_h3_text_only_ids(self.tokenizer, prompt)
                tags = torch.ones(ids.shape[0], dtype=torch.long)
                vision_kwargs = {}
            elif prepared_videos:
                videos = []
                sampled_videos = []
                for index, item in enumerate(prepared_videos):
                    sampled = sample_reference_video_frames(
                        item["prepared_path"],
                        workdir=str(Path(item["prepared_path"]).parent / f"qwen_frames_{index}"),
                    )
                    videos.append(np.stack(sampled["frames"]))
                    sampled_videos.append(sampled)
                vision = self.processor.video_processor(
                    videos=videos,
                    do_sample_frames=False,
                    return_tensors="pt",
                )
                video_grid = vision["video_grid_thw"]
                merge = int(self.processor.image_processor.merge_size) ** 2
                block_counts = []
                block_timestamps = []
                for index, sampled in enumerate(sampled_videos):
                    blocks = int(video_grid[index, 0])
                    per_block = int(video_grid[index, 1]) * int(video_grid[index, 2]) // merge
                    timestamps = sampled["block_timestamps"]
                    if len(timestamps) != blocks:
                        raise ValueError(
                            f"video block count mismatch: processor={blocks}, timestamps={len(timestamps)}"
                        )
                    block_counts.append([per_block] * blocks)
                    block_timestamps.append(timestamps)
                condition_labels: list[tuple[str, int]] = []
                audio_index = 0
                for video_index, item in enumerate(prepared_videos, start=1):
                    if item["input_has_audio"]:
                        audio_index += 1
                        condition_labels.append(("audio", audio_index))
                    condition_labels.append(("video", video_index))
                ids, tags = minimax_h3_ref2va_video_presentation(
                    self.tokenizer,
                    prompt=prompt,
                    condition_labels=condition_labels,
                    image_token_count=None,
                    video_block_token_counts=block_counts,
                    video_block_timestamps=block_timestamps,
                )
                vision_kwargs = {
                    "pixel_values_videos": vision["pixel_values_videos"],
                    "video_grid_thw": video_grid,
                }
            else:
                if image is None:
                    raise ValueError(f"{task} requires one image")
                vision = self.processor.image_processor(
                    images=[image],
                    return_tensors="pt",
                )
                image_grid = vision["image_grid_thw"]
                merge = int(self.processor.image_processor.merge_size) ** 2
                image_tokens = int(image_grid[0].prod().item()) // merge
                if task == "fl2va":
                    ids = minimax_h3_multi_image_presentation_ids(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=[image_tokens],
                    )
                    tags = minimax_h3_multi_image_presentation_token_tags(
                        self.tokenizer,
                        prompt=prompt,
                        image_token_counts=[image_tokens],
                    )
                else:
                    ids, tags = minimax_h3_ref2va_presentation(
                        self.tokenizer,
                        prompt=prompt,
                        condition_labels=[("image", 1), ("audio", 1)],
                        image_token_count=image_tokens,
                    )
                vision_kwargs = {
                    "pixel_values": vision["pixel_values"],
                    "image_grid_thw": image_grid,
                }

            logger.info(
                "MiniMax H3 %s Qwen presentation: %d tokens%s",
                task,
                int(ids.shape[0]),
                (f", {len(prepared_videos)} reference videos" if prepared_videos else ""),
            )

        if rank < self.text_encoder_tp_size:
            # Distribute the encode inputs from the DiT main rank to the other
            # encoder TP ranks, then run the distributed encode on all of them.
            ids = self._distribute_encode_inputs(ids, vision_kwargs)
            hidden = self._encode_text_hidden(ids, vision_kwargs)

        hidden = _broadcast_tensor(
            hidden,
            dtype=torch.bfloat16,
            device=self.device,
        )
        tags = _broadcast_tensor(
            tags,
            dtype=torch.long,
            device=self.device,
        )
        return hidden, tags

    def _build_text_encoder_group(self, text_encoder_tp_size: int) -> Any:
        """Create the encoder tensor-parallel process group.

        The encoder group covers the first ``text_encoder_tp_size`` DiT ranks
        (the DiT group is always global ranks ``[0, dit_world)``).  Every rank
        participates in ``new_group`` so the collective completes; ranks
        outside the group never run encoder collectives.  For a single-rank
        encoder we return a lightweight placeholder so non-encoder ranks do
        not need to join a ``GroupCoordinator`` that would assert on ranks
        outside the group.
        """
        if text_encoder_tp_size == 1:
            return _SingleRankEncoderGroup(rank=self._dit_rank)
        ranks = list(range(text_encoder_tp_size))
        return init_world_group(
            ranks=ranks,
            local_rank=envs.LOCAL_RANK,
            backend=current_omni_platform.dist_backend,
        )

    def _encoder_group_broadcast_tensor(
        self,
        tensor: torch.Tensor | None,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Broadcast a tensor from encoder rank 0 over the encoder TP group."""
        group = self.text_encoder_group
        if group.world_size == 1:
            if tensor is None:
                raise ValueError("source tensor is required for single-rank execution")
            return tensor.to(device=device, dtype=dtype)

        shape = torch.zeros(8, dtype=torch.long, device=device)
        if group.rank_in_group == 0:
            if tensor is None:
                raise ValueError("encoder rank 0 must provide a tensor to broadcast")
            shape[0] = tensor.ndim
            shape[1 : tensor.ndim + 1] = torch.tensor(tensor.shape, device=device)
        torch.distributed.broadcast(shape, src=group.ranks[0], group=group.device_group)
        ndim = int(shape[0].item())
        tensor_shape = tuple(int(value) for value in shape[1 : ndim + 1].tolist())
        if group.rank_in_group == 0:
            output = tensor.to(device=device, dtype=dtype).contiguous()
        else:
            output = torch.empty(tensor_shape, device=device, dtype=dtype)
        torch.distributed.broadcast(output, src=group.ranks[0], group=group.device_group)
        return output

    def _distribute_encode_inputs(
        self,
        ids: torch.Tensor | None,
        vision_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fan out encode inputs from encoder rank 0 to the encoder TP ranks.

        Mutates ``vision_kwargs`` in place so every encoder rank ends up with
        the same vision tensors, and returns the broadcast ``input_ids``.
        """
        keys = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")
        key_dtypes = {
            "pixel_values": torch.bfloat16,
            "pixel_values_videos": torch.bfloat16,
            "image_grid_thw": torch.long,
            "video_grid_thw": torch.long,
        }
        group = self.text_encoder_group
        device = self.device
        if group.world_size == 1:
            if ids is None:
                raise ValueError("encoder rank 0 must produce input ids")
            return ids.to(device=device, dtype=torch.long)

        mask = torch.zeros(len(keys), dtype=torch.long, device=device)
        if group.rank_in_group == 0:
            for index, key in enumerate(keys):
                mask[index] = 1 if key in vision_kwargs else 0
        torch.distributed.broadcast(mask, src=group.ranks[0], group=group.device_group)

        if group.rank_in_group == 0:
            ids = self._encoder_group_broadcast_tensor(ids, dtype=torch.long, device=device)
        else:
            ids = self._encoder_group_broadcast_tensor(None, dtype=torch.long, device=device)
        for index, key in enumerate(keys):
            if mask[index].item() == 0:
                continue
            source = vision_kwargs.get(key) if group.rank_in_group == 0 else None
            vision_kwargs[key] = self._encoder_group_broadcast_tensor(
                source,
                dtype=key_dtypes[key],
                device=device,
            )
        return ids

    def _prepare_reference_videos(
        self,
        values: Any,
        *,
        target_frame_count: int,
        workdir: str,
    ) -> list[dict[str, Any]] | None:
        _, rank, _ = _dit_rank_world()
        if rank != 0:
            return None
        return prepare_reference_videos(
            values,
            target_frame_count=target_frame_count,
            workdir=workdir,
        )

    def _encode_text_hidden(
        self,
        input_ids: torch.Tensor,
        vision_kwargs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if self.od_config.enable_cpu_offload:
            # Invoke nn.Module.__call__ so the generic model-level offloader
            # swaps the resident DiT and encoder.
            return self.text_encoder(input_ids, **vision_kwargs)

        if self.od_config.enable_layerwise_offload:
            # Layerwise DiT offload already provides the low-residency encoder
            # phase used by the checkpoint reference.
            self.text_encoder.load_to_device()
            try:
                return self.text_encoder.encode_ids(input_ids, **vision_kwargs)
            finally:
                self.text_encoder.offload_to_cpu()

        # Keep both Qwen and DiT resident across requests. Moving either model
        # here makes encoder latency include a tens-of-gigabytes PCIe transfer,
        # which defeats the no-offload contract.
        self.text_encoder.load_to_device()
        return self.text_encoder.encode_ids(input_ids, **vision_kwargs)

    def _encode_visual_condition(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        _, rank, _ = _dit_rank_world()
        rows = self.video_vae.encode_image(image) if rank == 0 else None
        return _broadcast_tensor(
            rows,
            dtype=torch.float32,
            device=self.device,
        )

    def _encode_audio_condition(
        self,
        audio: tuple[torch.Tensor, int],
    ) -> tuple[torch.Tensor, int]:
        _, rank, _ = _dit_rank_world()
        rows = None
        audio_t = 0
        if rank == 0:
            rows, audio_t = self.audio_vae.encode_waveform(*audio)
        audio_t_tensor = torch.tensor(
            [audio_t],
            dtype=torch.long,
            device=self.device,
        )
        group, _, world_size = _dit_rank_world()
        if world_size > 1:
            dist.broadcast(audio_t_tensor, src=0, group=group)
        rows = _broadcast_tensor(
            rows,
            dtype=torch.float32,
            device=self.device,
        )
        return rows, int(audio_t_tensor.item())

    def _encode_video_conditions(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        count: int,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        group, rank, world_size = _dit_rank_world()
        distributed_encode = self.video_vae.is_distributed_enabled()
        if distributed_encode:
            # Native tiled encode uses collectives, so every VPP rank must
            # enter each reference encode in the same input order.
            prepared_videos_list = [prepared_videos]
            dist.broadcast_object_list(
                prepared_videos_list,
                src=0,
                group=group,
                device=self.device,
            )
            prepared_videos = prepared_videos_list[0]

        rows = None
        shapes = torch.zeros((count, 3), dtype=torch.long, device=self.device)
        if rank == 0 or distributed_encode:
            if prepared_videos is None or len(prepared_videos) != count:
                raise ValueError("reference-video preparation is incomplete")
            encoded = [
                self.video_vae.encode_video(load_video_frames(item["prepared_path"])) for item in prepared_videos
            ]
            rows = torch.cat([item[0] for item in encoded])
            shapes = torch.tensor(
                [item[1] for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        if distributed_encode:
            return (
                rows.to(device=self.device, dtype=torch.float32),
                [tuple(int(value) for value in item) for item in shapes.tolist()],
            )

        if world_size > 1:
            dist.broadcast(shapes, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [tuple(int(value) for value in item) for item in shapes.tolist()],
        )

    def _encode_video_audio_conditions(
        self,
        prepared_videos: list[dict[str, Any]] | None,
        *,
        has_audio: list[bool],
    ) -> tuple[torch.Tensor | None, list[int]]:
        _, rank, _ = _dit_rank_world()
        count = sum(has_audio)
        if count == 0:
            return None, []
        rows = None
        lengths = torch.zeros(count, dtype=torch.long, device=self.device)
        if rank == 0:
            if prepared_videos is None:
                raise ValueError("rank 0 reference-video preparation is incomplete")
            encoded = [
                self.audio_vae.encode_waveform(*load_video_audio(item["original_path"]))
                for item in prepared_videos
                if item["input_has_audio"]
            ]
            rows = torch.cat([item[0] for item in encoded])
            lengths = torch.tensor(
                [item[1] for item in encoded],
                dtype=torch.long,
                device=self.device,
            )
        group, _, world_size = _dit_rank_world()
        if world_size > 1:
            dist.broadcast(lengths, src=0, group=group)
        return (
            _broadcast_tensor(rows, dtype=torch.float32, device=self.device),
            [int(value) for value in lengths.tolist()],
        )

    def _initial_noise(
        self,
        *,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_generator = torch.Generator(device="cpu").manual_seed(seed)
        video = torch.randn(
            1,
            24,
            latent_t,
            latent_h,
            latent_w,
            generator=video_generator,
            dtype=torch.float32,
        )
        video_rows = minimax_h3_patchify_video_latent(
            video,
            patch_size=(1, 2, 2),
        )
        audio_generator = torch.Generator(device="cpu").manual_seed(seed)
        audio_rows = torch.randn(
            audio_t * 2,
            32,
            generator=audio_generator,
            dtype=torch.float32,
        )
        return video_rows, audio_rows

    def diffuse(
        self,
        *,
        task: str,
        text_embeddings: torch.Tensor,
        text_tags: torch.Tensor,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
        num_frames: int,
        num_steps: int,
        video_shift: float,
        audio_shift: float,
        visual_condition: torch.Tensor | None,
        visual_condition_shape: tuple[int, int, int] | None,
        audio_condition: torch.Tensor | None,
        ref_audio_t: int | None,
        ref_blocks: list[dict[str, Any]] | None = None,
        visual_condition_shapes: list[tuple[int, int, int]] | None = None,
        audio_condition_lengths: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        initial_video, initial_audio = self._initial_noise(
            seed=seed,
            latent_t=latent_t,
            latent_h=latent_h,
            latent_w=latent_w,
            audio_t=audio_t,
        )
        if task == "ref2va":
            if ref_blocks is None:
                if visual_condition_shape is None or ref_audio_t is None:
                    raise ValueError("ref2va condition metadata is missing")
                _, ref_h, ref_w = visual_condition_shape
                ref_blocks = [
                    {"kind": "image", "latent_h": ref_h, "latent_w": ref_w},
                    {"kind": "audio", "ref_audio_t": ref_audio_t},
                ]
            packed = minimax_h3_packed_sequence_ref2va_blocks(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                ref_blocks=ref_blocks,
            )
        else:
            packed = minimax_h3_packed_sequence(
                text_len=int(text_embeddings.shape[0]),
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
                include_keyframe_cond=task == "fl2va",
                keyframe_frame_indices=[0] if task == "fl2va" else None,
                frame_count=num_frames if task == "fl2va" else None,
            )

        tags = packed["token_tags"].clone()
        tags[packed["text_pos"]] = text_tags.cpu()
        branch = MiniMaxH3DenoiseBranch(
            packed=packed,
            text_embeddings=text_embeddings,
            token_tags=tags,
            device=self.device,
        )

        visual_anchor = visual_condition
        if visual_anchor is not None:
            condition_shapes = visual_condition_shapes
            if condition_shapes is None and visual_condition_shape is not None:
                condition_shapes = [visual_condition_shape]
            if not condition_shapes:
                raise ValueError("visual condition shape is missing")
            visual_anchor = minimax_h3_imgvid_cond_noise_aug_rows(
                visual_anchor,
                condition_shapes=condition_shapes,
                target_latent_t=latent_t,
                imgvid_cond_num_frames=len(condition_shapes),
                seed=seed,
                noise_aug=MINIMAX_H3_IMGVID_COND_TIMESTEP,
            )
            full_video = torch.zeros(
                branch.img_pos.shape[0],
                96,
                dtype=torch.float32,
            )
            full_video[branch.update_mask] = initial_video
            initial_video = full_video

        audio_anchor = audio_condition
        if audio_anchor is not None:
            condition_audio_t = audio_condition_lengths
            if condition_audio_t is None and ref_audio_t is not None:
                condition_audio_t = [ref_audio_t]
            if not condition_audio_t:
                raise ValueError("reference audio length is missing")
            audio_anchor = minimax_h3_audio_cond_noise_aug_rows(
                audio_anchor,
                condition_audio_t=condition_audio_t,
                seed=seed,
                noise_aug=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
            )
            full_audio = torch.zeros(
                branch.audio_pos.shape[0],
                32,
                dtype=torch.float32,
            )
            full_audio[branch.audio_update_mask] = initial_audio
            initial_audio = full_audio

        video_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=video_shift,
        )
        audio_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=num_steps,
            shift_scale=audio_shift,
        )
        with self.progress_bar(total=len(video_sigmas) - 1) as progress:
            video_rows, audio_rows = minimax_h3_denoise_loop(
                model=self.transformer,
                positive=branch,
                initial_video_rows=initial_video,
                initial_audio_rows=initial_audio,
                keyframe_cond_rows=visual_anchor,
                audio_ref_rows=audio_anchor,
                sigmas_video=video_sigmas,
                sigmas_audio=audio_sigmas,
                device=self.device,
                imgvid_cond_noise_aug_for_inference=(MINIMAX_H3_IMGVID_COND_TIMESTEP),
                audio_cond_noise_aug_for_inference=(MINIMAX_H3_AUDIO_REF_COND_TIMESTEP),
                on_step=lambda step, video, audio: progress.update(),
            )

        target_video = video_rows[branch.update_mask_dev]
        video_latent = minimax_h3_unpatchify_video_tokens(
            target_video,
            latent_shape=(
                latent_t,
                latent_h // 2,
                latent_w // 2,
                24,
            ),
            patch_size=(1, 2, 2),
        )
        target_audio = audio_rows[branch.audio_update_mask_dev]
        audio_latent = minimax_h3_unpack_audio_tokens(
            target_audio,
            audio_t=audio_t * 2,
            audio_channel=2,
        )
        return video_latent, audio_latent

    def decode(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with current_omni_platform.create_autocast_context(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=True,
        ):
            video = self.video_vae.decode_latent(video_latent)
        video = video[..., :height, :width].contiguous()
        audio = self.audio_vae.decode_latent(audio_latent)
        return video, audio

    @torch.no_grad()
    def forward(self, request: DiffusionRequestBatch) -> DiffusionOutput:
        if len(request.prompts) != 1:
            raise ValueError("MiniMax H3 supports one request at a time")
        raw_prompt = request.prompts[0]
        if isinstance(raw_prompt, str):
            prompt = raw_prompt
            multi_modal_data: dict[str, Any] = {}
        else:
            prompt = str(raw_prompt.get("prompt") or "")
            multi_modal_data = raw_prompt.get("multi_modal_data") or {}
        if not prompt:
            raise ValueError("MiniMax H3 requires a non-empty prompt")

        sampling = request.sampling_params
        extra = sampling.extra_args or {}
        task = self._resolve_task(extra.get("task"), multi_modal_data)

        raw_image = multi_modal_data.get("image")
        raw_videos = multi_modal_data.get("video")
        image = _load_image(raw_image) if raw_image is not None else None
        if task == "fl2va" and image is None:
            raise ValueError(f"{task} requires multi_modal_data.image")
        if task == "ref2va" and image is None and raw_videos is None:
            raise ValueError("ref2va requires multi_modal_data.image or multi_modal_data.video")
        if task == "ref2va" and image is not None and raw_videos is not None:
            raise ValueError("ref2va currently accepts image+audio or one or more videos, not both")
        if task != "ref2va" and raw_videos is not None:
            raise ValueError(f"{task} does not accept a video condition")
        if task == "ref2va" and raw_videos is not None and multi_modal_data.get("audio") is not None:
            raise ValueError(
                "video Ref2VA uses the reference-video soundtracks and does not accept a separate audio condition"
            )
        if task == "t2va" and image is not None:
            raise ValueError("t2va does not accept an image condition")

        height, width, num_frames, latent_t, audio_t = self._resolve_shape(task, sampling, image)
        prepared_image = image
        if task == "fl2va" and image is not None:
            prepared_image = image.resize(
                (width, height),
                Image.Resampling.LANCZOS,
            )
        elif task == "ref2va" and image is not None:
            ref_width, ref_height = _reference_image_shape(image)
            prepared_image = image.resize(
                (ref_width, ref_height),
                Image.Resampling.LANCZOS,
            )

        visual_condition = None
        visual_shape = None
        visual_shapes = None
        audio_condition = None
        ref_audio_t = None
        audio_lengths = None
        ref_blocks = None
        with tempfile.TemporaryDirectory(prefix="minimax_h3_ref2va_") as workdir:
            prepared_videos = None
            has_audio: list[bool] = []
            video_count = 0
            if raw_videos is not None:
                video_count = len(raw_videos) if isinstance(raw_videos, (list, tuple)) else 1
                prepared_videos = self._prepare_reference_videos(
                    raw_videos,
                    target_frame_count=num_frames,
                    workdir=workdir,
                )
                has_audio_tensor = torch.zeros(
                    video_count,
                    dtype=torch.long,
                    device=self.device,
                )
                _, rank, world_size = _dit_rank_world()
                if rank == 0:
                    has_audio_tensor = torch.tensor(
                        [int(item["input_has_audio"]) for item in prepared_videos or []],
                        dtype=torch.long,
                        device=self.device,
                    )
                if world_size > 1:
                    dist.broadcast(
                        has_audio_tensor,
                        src=0,
                        group=get_dit_group(),
                    )
                has_audio = [bool(value) for value in has_audio_tensor.tolist()]

            text_embeddings, text_tags = self.encode_prompt(
                task=task,
                prompt=prompt,
                image=prepared_image,
                prepared_videos=prepared_videos,
            )

            if prepared_videos is not None or raw_videos is not None:
                visual_condition, visual_shapes = self._encode_video_conditions(
                    prepared_videos,
                    count=video_count,
                )
                audio_condition, audio_lengths = self._encode_video_audio_conditions(
                    prepared_videos,
                    has_audio=has_audio,
                )
                audio_iterator = iter(audio_lengths)
                ref_blocks = []
                for shape, contributes_audio in zip(
                    visual_shapes,
                    has_audio,
                    strict=True,
                ):
                    ref_audio = next(audio_iterator) if contributes_audio else 0
                    ref_blocks.append(
                        {
                            "kind": "video",
                            "ref_audio_t": ref_audio,
                            "latent_t": shape[0],
                            "latent_h": shape[1],
                            "latent_w": shape[2],
                        }
                    )
            elif prepared_image is not None:
                visual_condition = self._encode_visual_condition(prepared_image)
                visual_shape = (
                    1,
                    prepared_image.height // 16,
                    prepared_image.width // 16,
                )

            if task == "ref2va" and raw_videos is None:
                raw_audio = multi_modal_data.get("audio")
                if raw_audio is None:
                    raise ValueError("image Ref2VA requires multi_modal_data.audio")
                audio_condition, ref_audio_t = self._encode_audio_condition(_load_audio(raw_audio))
            elif task != "ref2va" and multi_modal_data.get("audio") is not None:
                raise ValueError(f"{task} does not accept an audio condition")

        seed = int(sampling.seed if sampling.seed is not None else 42)
        num_steps = int(sampling.num_inference_steps or 50)
        video_shift = float(extra.get("flow_shift", self.default_video_shift))
        audio_shift = float(extra.get("audio_flow_shift", self.default_audio_shift))
        video_latent, audio_latent = self.diffuse(
            task=task,
            text_embeddings=text_embeddings,
            text_tags=text_tags,
            seed=seed,
            latent_t=latent_t,
            latent_h=height // 16,
            latent_w=width // 16,
            audio_t=audio_t,
            num_frames=num_frames,
            num_steps=num_steps,
            video_shift=video_shift,
            audio_shift=audio_shift,
            visual_condition=visual_condition,
            visual_condition_shape=visual_shape,
            audio_condition=audio_condition,
            ref_audio_t=ref_audio_t,
            ref_blocks=ref_blocks,
            visual_condition_shapes=visual_shapes,
            audio_condition_lengths=audio_lengths,
        )
        video, audio = self.decode(
            video_latent,
            audio_latent,
            height=height,
            width=width,
        )
        return DiffusionOutput(
            output=(video, audio),
            post_process_func=get_minimax_h3_post_process_func(self.od_config),
            stage_durations=(self.stage_durations if hasattr(self, "_stage_durations") else {}),
        )


__all__ = [
    "MiniMaxH3Pipeline",
    "get_minimax_h3_post_process_func",
]
