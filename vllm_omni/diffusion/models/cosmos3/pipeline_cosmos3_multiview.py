# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 Multiview-AV pipeline.

The checkpoint uses the regular Cosmos3 Nano weights.  Multiview behavior is
entirely request-, VAE-, position-, and attention-mask-side: one clean WSM
item and one RGB target item are packed camera-major and denoised together.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .multiview_flex_attention import (
    DEFAULT_MAX_UND_TOKENS,
    MultiviewLayout,
    expand_multiview_condition_frame_indexes,
    validate_multiview_backend,
)
from .pipeline_cosmos3 import (
    COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE,
    COSMOS3_T2V_DEFAULT_NUM_INFERENCE_STEPS,
    COSMOS3_TRANSFER_SYSTEM_PROMPT,
    COSMOS3_VIDEO_DEFAULT_FLOW_SHIFT,
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_ir_op_priority_func,
    get_cosmos3_post_process_func,
)
from .transfer import (
    IMAGE_EXTENSIONS,
    TRANSFER_HINT_KEYS,
    as_bool,
    media_to_uint8_cthw,
    uint8_cthw_to_normalized_5d,
)
from .transformer_cosmos3 import _tf_config_get
from .transformer_cosmos3_multiview import Cosmos3MultiviewVFMTransformer
from .utils import VIDEO_RES_SIZE_INFO

# Overrides transformer config multiview.backend, so the Triton and FA4 sparse
# attention paths can be compared without editing the checkpoint.
COSMOS3_MULTIVIEW_BACKEND_ENV = "VLLM_OMNI_COSMOS3_MULTIVIEW_BACKEND"

COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES = 93
COSMOS3_MULTIVIEW_DEFAULT_FPS = 10.0

# The tokenizer appends eos and vision_start after truncating. Derive the
# request ceiling from the sparse attention's single fixed UND capacity so the
# two cannot drift and accidentally trigger shape-specific recompilation.
COSMOS3_MULTIVIEW_PROMPT_FRAMING_TOKENS = 2
COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH = DEFAULT_MAX_UND_TOKENS - COSMOS3_MULTIVIEW_PROMPT_FRAMING_TOKENS
COSMOS3_MULTIVIEW_EMPHASIS = (
    "Follow the wsm control videos precisely for every camera view: shape, contour, position, and motion must "
    "align with the wsm signal at every frame."
)
COSMOS3_MADS_CAMERAS = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
    "camera_front_fisheye_200fov",
    "camera_left_fisheye_200fov",
    "camera_right_fisheye_200fov",
    "camera_rear_fisheye_200fov",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Cosmos3 multiview {name} must be an object, got {type(value).__name__}.")
    return value


def _media_kind(value: Any) -> str:
    if isinstance(value, str | Path):
        return "image" if Path(value).suffix.lower() in IMAGE_EXTENSIONS else "video"
    if isinstance(value, PIL.Image.Image):
        return "image"
    if isinstance(value, torch.Tensor):
        tensor = value
        if tensor.ndim == 5:
            return "image" if tensor.shape[2] == 1 else "video"
        if tensor.ndim == 4:
            temporal_dim = 1 if tensor.shape[0] in (3, 4) else 0
            return "image" if tensor.shape[temporal_dim] == 1 else "video"
        if tensor.ndim == 3:
            return "image"
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return "image" if len(value) == 1 else "video"
    raise TypeError(f"Unsupported Cosmos3 multiview media payload type: {type(value).__name__}.")


def _normalize_local_condition_indexes(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Sequence[Any] = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, int):
        values = [value]
    elif isinstance(value, Sequence):
        values = value
    else:
        raise TypeError("Cosmos3 multiview condition_frame_indexes_vision must be an int, list, or CSV string.")
    return sorted({int(index) for index in values})


def _pad_multiview_view_video(
    frames: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Pad one camera to ``num_frames`` by replicating its last frame.

    The reference pads a view into a mid-gray canvas and then replicates the
    last decoded frame over the tail, so gray only ever survives for a view
    with no media at all.  Admission requires a control clip for every camera
    and vision for all cameras or none, so that case cannot reach here; an
    empty clip is a decode failure and is reported rather than silently
    generating a gray camera.
    """
    if frames.ndim != 4 or tuple(frames.shape[:1]) != (3,) or tuple(frames.shape[2:]) != (height, width):
        raise ValueError(
            "Cosmos3 multiview view frames must have shape [3, T, H, W] at the output resolution, "
            f"got {tuple(frames.shape)}."
        )
    fill = min(int(frames.shape[1]), num_frames)
    if fill <= 0:
        raise ValueError("Cosmos3 multiview view media decoded to zero frames.")
    video = frames[:, :fill]
    if fill < num_frames:
        video = torch.cat([video, video[:, -1:].expand(-1, num_frames - fill, -1, -1)], dim=1)
    return video.contiguous()


def _resolve_multiview_resolution(sp: Any, multiview: Mapping[str, Any]) -> str:
    """Resolve the variant-owned resolution without inheriting the image default."""
    resolution = multiview.get("resolution")
    if resolution is None:
        extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
        resolution = extra.get("resolution", "480")
    return str(resolution)


class Cosmos3MultiviewPipeline(Cosmos3OmniDiffusersPipeline):
    """Bidirectional one-shot 11-view RGB generation with WSM control."""

    # The generic engine warmup has no per-camera WSM inputs and uses image
    # geometry that is invalid for this fixed-layout pipeline. Compile the
    # model on its first real request instead of weakening request validation.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        parallel_config = od_config.parallel_config
        sequence_parallel_size = int(getattr(parallel_config, "sequence_parallel_size", 1) or 1)
        cfg_parallel_size = int(getattr(parallel_config, "cfg_parallel_size", 1) or 1)
        tensor_parallel_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
        pipeline_parallel_size = int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)
        vae_parallel_size = int(getattr(parallel_config, "vae_patch_parallel_size", 1) or 1)
        if sequence_parallel_size > 1:
            raise ValueError("Cosmos3 multiview v1 does not support sequence parallelism.")
        if cfg_parallel_size > 1:
            raise ValueError("Cosmos3 multiview v1 uses single-GPU sequential CFG; cfg_parallel_size must be 1.")
        if tensor_parallel_size > 1 or pipeline_parallel_size > 1 or vae_parallel_size > 1:
            raise ValueError(
                "Cosmos3 multiview v1 is single-GPU: tensor, pipeline, and VAE parallel sizes must all be 1."
            )
        if bool(getattr(parallel_config, "use_hsdp", False)):
            raise ValueError("Cosmos3 multiview v1 does not support HSDP.")
        if od_config.enable_session_state_manager:
            raise ValueError("Cosmos3 multiview v1 does not support enable_session_state_manager.")
        super().__init__(od_config=od_config, prefix=prefix)
        if self.device.type != "cuda":
            raise ValueError("Cosmos3 multiview v1 requires CUDA for its sparse attention backends.")
        if not isinstance(self.transformer, Cosmos3MultiviewVFMTransformer):
            raise ValueError(
                "Cosmos3MultiviewPipeline requires transformer/config.json backbone_type='cosmos3_multiview'."
            )

        model_config = od_config.tf_model_config
        multiview_config = _tf_config_get(model_config, "multiview", None)
        if multiview_config is None:
            raise ValueError("Cosmos3 multiview transformer config must contain a 'multiview' object.")
        if hasattr(multiview_config, "to_dict"):
            multiview_config = multiview_config.to_dict()
        multiview_config = _mapping(multiview_config, "transformer config")
        cameras = multiview_config.get("cameras")
        if not isinstance(cameras, Sequence) or isinstance(cameras, str | bytes) or not cameras:
            raise ValueError("Cosmos3 multiview transformer config must contain a non-empty cameras list.")
        self.multiview_cameras = tuple(str(camera) for camera in cameras)
        if self.multiview_cameras != COSMOS3_MADS_CAMERAS:
            raise ValueError(
                "Cosmos3 Multiview-AV v1 requires the fixed 11-camera MADS order: "
                f"expected={list(COSMOS3_MADS_CAMERAS)}, got={list(self.multiview_cameras)}."
            )
        max_views = int(multiview_config.get("max_views", len(self.multiview_cameras)))
        if max_views != len(self.multiview_cameras):
            raise ValueError(
                "Cosmos3 multiview max_views must equal the exported camera list length: "
                f"max_views={max_views}, cameras={len(self.multiview_cameras)}."
            )
        if not as_bool(multiview_config.get("share_vision_temporal_positions", True), True):
            raise ValueError("Cosmos3 multiview requires share_vision_temporal_positions=true.")
        self.multiview_attention_scope = str(multiview_config.get("attention_scope", "same_view_or_frame"))
        self.multiview_backend = self._resolve_attention_backend(multiview_config)

    @staticmethod
    def _resolve_attention_backend(multiview_config: Mapping[str, Any]) -> str:
        """Pick the sparse attention backend, env override winning over the checkpoint.

        Unlike ``attention_scope``, the backend does not change what the model
        computes -- both backends project the same visibility predicate -- so it
        is safe to override without editing the checkpoint.  That matters for
        A/B measurement, which is the reason the second backend exists.

        Validated here rather than in ``MultiviewLayout`` so a bad name fails at
        load time instead of on the first generated frame.
        """
        override = os.environ.get(COSMOS3_MULTIVIEW_BACKEND_ENV)
        backend = str(override if override else multiview_config.get("backend", "triton"))
        try:
            return validate_multiview_backend(backend)
        except ValueError as exc:
            source = (
                f"{COSMOS3_MULTIVIEW_BACKEND_ENV}={override!r}" if override else "transformer config multiview.backend"
            )
            raise ValueError(f"{exc} (from {source})") from exc

    def _parse_multiview_request(self, sp: Any) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
        extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
        multiview = _mapping(extra.get("multiview"), "extra_args['multiview']")
        unknown_multiview_fields = set(multiview) - {
            "views",
            "condition_video_as_image",
            "condition_frame_indexes_vision",
            "num_frames",
            "resolution",
        }
        if unknown_multiview_fields:
            raise ValueError(f"Unsupported Cosmos3 multiview fields: {sorted(unknown_multiview_fields)}.")
        raw_views = multiview.get("views")
        if not isinstance(raw_views, Sequence) or isinstance(raw_views, str | bytes) or not raw_views:
            raise ValueError("Cosmos3 multiview.views must contain at least one camera view.")
        views = [_mapping(view, f"view {index}") for index, view in enumerate(raw_views)]
        allowed_view_fields = {"camera_key", "vision_path", "control_path", "vision", "control"}
        for index, view in enumerate(views):
            unknown_view_fields = set(view) - allowed_view_fields
            if unknown_view_fields:
                raise ValueError(f"Unsupported Cosmos3 multiview view {index} fields: {sorted(unknown_view_fields)}.")
        camera_keys = [str(view.get("camera_key", "")) for view in views]
        if any(not key for key in camera_keys) or len(set(camera_keys)) != len(camera_keys):
            raise ValueError(f"Cosmos3 multiview camera_key values must be non-empty and unique: {camera_keys}.")
        if tuple(camera_keys) != self.multiview_cameras:
            raise ValueError(
                "Cosmos3 multiview camera order must exactly match the exported checkpoint order: "
                f"expected={list(self.multiview_cameras)}, got={camera_keys}."
            )
        if any("lidar" in key.lower() for key in camera_keys) or extra.get("lidar") is not None:
            raise ValueError("Cosmos3 multiview v1 does not support LiDAR items.")

        selected_hints = [key for key in TRANSFER_HINT_KEYS if extra.get(key) is not None]
        if selected_hints != ["wsm"]:
            raise ValueError(
                f"Cosmos3 multiview requires exactly one top-level precomputed WSM hint; selected={selected_hints}."
            )
        wsm = extra.get("wsm")
        if wsm is not True and (not isinstance(wsm, Mapping) or len(wsm) != 0):
            raise ValueError(
                "Cosmos3 multiview WSM controls must be supplied per view; "
                "top-level wsm must be true or an empty object."
            )

        vision_present = [view.get("vision_path", view.get("vision")) is not None for view in views]
        control_present = [view.get("control_path", view.get("control")) is not None for view in views]
        if any(vision_present) and not all(vision_present):
            raise ValueError("Cosmos3 multiview vision inputs must be supplied for every camera or none.")
        if not all(control_present):
            raise ValueError("Cosmos3 multiview requires a precomputed control input for every camera.")

        for field, present in (("vision", vision_present), ("control", control_present)):
            if not any(present):
                continue
            values = [view.get(f"{field}_path", view.get(field)) for view in views]
            kinds = {_media_kind(value) for value in values}
            if len(kinds) != 1:
                raise ValueError(f"Cosmos3 multiview {field} inputs must be all images or all videos, got {kinds}.")
        return multiview, views

    @staticmethod
    def _view_value(view: Mapping[str, Any], field: str) -> Any:
        return view.get(f"{field}_path", view.get(field))

    def _prepare_camera_major_pixels(
        self,
        views: Sequence[Mapping[str, Any]],
        *,
        field: str,
        height: int,
        width: int,
        num_frames: int,
        keep_first: bool,
    ) -> torch.Tensor:
        prepared = []
        for view in views:
            value = self._view_value(view, field)
            if value is None:
                raise ValueError(f"Cosmos3 multiview camera {view['camera_key']!r} is missing {field} input.")
            frames = media_to_uint8_cthw(
                value,
                height=height,
                width=width,
                max_frames=1 if keep_first else num_frames,
            )
            prepared.append(_pad_multiview_view_video(frames, num_frames=num_frames, height=height, width=width))
        camera_major = torch.cat(prepared, dim=1)
        return uint8_cthw_to_normalized_5d(camera_major, dtype=self.dtype)

    def _encode_multiview_video(
        self,
        camera_major_video: torch.Tensor,
        *,
        num_views: int,
        frames_per_view: int,
    ) -> torch.Tensor:
        expected_frames = num_views * frames_per_view
        if camera_major_video.ndim != 5 or camera_major_video.shape[2] != expected_frames:
            raise ValueError(
                "Cosmos3 multiview pixel video must be camera-major [1, 3, V*F, H, W]: "
                f"shape={tuple(camera_major_video.shape)}, V={num_views}, F={frames_per_view}."
            )
        per_view = [
            self._encode_video_tensor(camera_major_video[:, :, view * frames_per_view : (view + 1) * frames_per_view])
            for view in range(num_views)
        ]
        latent_frames = {int(latent.shape[2]) for latent in per_view}
        if len(latent_frames) != 1:
            raise ValueError(f"Cosmos3 multiview per-camera VAE encodes have unequal lengths: {latent_frames}.")
        return torch.cat(per_view, dim=2)

    def _decode_multiview_latents(
        self,
        camera_major_latents: torch.Tensor,
        *,
        num_views: int,
        latent_frames_per_view: int,
    ) -> torch.Tensor:
        if camera_major_latents.shape[2] != num_views * latent_frames_per_view:
            raise ValueError(
                "Cosmos3 multiview latents must be camera-major before decode: "
                f"shape={tuple(camera_major_latents.shape)}, V={num_views}, F={latent_frames_per_view}."
            )
        decoded = [
            self._decode_latents(
                camera_major_latents[
                    :,
                    :,
                    view * latent_frames_per_view : (view + 1) * latent_frames_per_view,
                ]
            )
            for view in range(num_views)
        ]
        return torch.cat(decoded, dim=2)

    def _prepare_multiview_latents(
        self,
        *,
        target_pixels: torch.Tensor | None,
        condition_indexes: Sequence[int],
        num_views: int,
        num_frames: int,
        height: int,
        width: int,
        generator: torch.Generator,
        injected_latents: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent_frames_per_view = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            1,
            self.transformer.latent_channel_size,
            num_views * latent_frames_per_view,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if injected_latents is None:
            noise = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)
        else:
            noise = injected_latents.to(device=self.device, dtype=self.dtype)
            if tuple(noise.shape) != shape:
                raise ValueError(
                    "Cosmos3 multiview injected latents have the wrong shape: "
                    f"expected={shape}, got={tuple(noise.shape)}."
                )
        condition_mask = torch.zeros(1, 1, shape[2], 1, 1, device=self.device, dtype=self.dtype)
        condition_latents = torch.zeros_like(noise)
        if condition_indexes:
            if target_pixels is None:
                raise ValueError("Cosmos3 multiview condition indexes require per-camera vision inputs.")
            encoded = self._encode_multiview_video(
                target_pixels,
                num_views=num_views,
                frames_per_view=num_frames,
            )
            if tuple(encoded.shape) != shape:
                raise ValueError(
                    f"Cosmos3 multiview target VAE latent shape mismatch: expected={shape}, got={tuple(encoded.shape)}."
                )
            for index in condition_indexes:
                condition_mask[:, :, index] = 1
                condition_latents[:, :, index : index + 1] = encoded[:, :, index : index + 1]
        latents = condition_mask * condition_latents + (1.0 - condition_mask) * noise
        return latents, 1.0 - condition_mask, condition_latents

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("Cosmos3MultiviewPipeline supports exactly one prompt per request.")
        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            prompt = prompt_data
            request_negative_prompt = None
        elif isinstance(prompt_data, Mapping):
            prompt = str(prompt_data.get("prompt", ""))
            request_negative_prompt = prompt_data.get("negative_prompt")
        else:
            raise TypeError(f"Unsupported Cosmos3 multiview prompt type: {type(prompt_data).__name__}.")

        sp = req.sampling_params
        multiview, views = self._parse_multiview_request(sp)
        num_views = len(views)
        requested_num_frames = multiview.get("num_frames")
        if requested_num_frames is None:
            requested_num_frames = sp.num_frames
        # OmniDiffusionSamplingParams retains a legacy image-oriented default
        # of one frame; this video-only variant owns the 93-frame default.
        if requested_num_frames in (None, 1):
            requested_num_frames = COSMOS3_MULTIVIEW_DEFAULT_NUM_FRAMES
        num_frames = int(requested_num_frames)
        if num_frames <= 0 or (num_frames - 1) % self.vae_scale_factor_temporal:
            raise ValueError(f"Cosmos3 multiview num_frames must satisfy 4k+1 for the Wan VAE, got {num_frames}.")
        resolution = _resolve_multiview_resolution(sp, multiview)
        if resolution != "480":
            raise ValueError(f"Cosmos3 multiview v1 supports only resolution='480', got {resolution!r}.")
        width, height = VIDEO_RES_SIZE_INFO["480"]["16,9"]
        if sp.height is not None and int(sp.height) != height:
            raise ValueError(f"Cosmos3 multiview height is fixed at {height}, got {sp.height}.")
        if sp.width is not None and int(sp.width) != width:
            raise ValueError(f"Cosmos3 multiview width is fixed at {width}, got {sp.width}.")
        frame_rate = float(
            self._get_sp_param(sp, "resolved_frame_rate", None)
            or self._get_sp_param(sp, "frame_rate", None)
            or self._get_sp_param(sp, "fps", None)
            or COSMOS3_MULTIVIEW_DEFAULT_FPS
        )
        if frame_rate != COSMOS3_MULTIVIEW_DEFAULT_FPS:
            raise ValueError(f"Cosmos3 multiview v1 is pinned to 10 FPS, got {frame_rate}.")

        condition_video_as_image = as_bool(multiview.get("condition_video_as_image"), False)
        has_vision = self._view_value(views[0], "vision") is not None
        vision_kind = _media_kind(self._view_value(views[0], "vision")) if has_vision else None
        target_pixels = None
        if has_vision:
            target_pixels = self._prepare_camera_major_pixels(
                views,
                field="vision",
                height=height,
                width=width,
                num_frames=num_frames,
                keep_first=condition_video_as_image,
            )
        control_pixels = self._prepare_camera_major_pixels(
            views,
            field="control",
            height=height,
            width=width,
            num_frames=num_frames,
            keep_first=False,
        )

        latent_frames_per_view = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_t = num_views * latent_frames_per_view
        raw_indexes = multiview.get("condition_frame_indexes_vision")
        if raw_indexes is None:
            if not has_vision:
                local_indexes = []
            elif condition_video_as_image or vision_kind == "image":
                local_indexes = [0]
            else:
                local_indexes = [0, 1]
        else:
            local_indexes = _normalize_local_condition_indexes(raw_indexes)
        condition_indexes = expand_multiview_condition_frame_indexes(local_indexes, num_views, latent_t)

        generator = sp.generator
        seed = self._resolve_seed(sp, generator)
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        injected_latents = sp.latents if isinstance(sp.latents, torch.Tensor) else None
        latents, velocity_mask, condition_latents = self._prepare_multiview_latents(
            target_pixels=target_pixels,
            condition_indexes=condition_indexes,
            num_views=num_views,
            num_frames=num_frames,
            height=height,
            width=width,
            generator=generator,
            injected_latents=injected_latents,
        )
        control_latents = self._encode_multiview_video(
            control_pixels,
            num_views=num_views,
            frames_per_view=num_frames,
        )
        if control_latents.shape != latents.shape:
            raise ValueError(
                "Cosmos3 multiview WSM and target latent shapes must match: "
                f"control={tuple(control_latents.shape)}, target={tuple(latents.shape)}."
            )

        max_sequence_length = int(
            self._get_sp_param(sp, "max_sequence_length", COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH)
            or COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH
        )
        if max_sequence_length > COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "Cosmos3 multiview max_sequence_length cannot exceed the variant ceiling the sparse "
                f"attention is sized for: requested={max_sequence_length}, "
                f"ceiling={COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH}."
            )
        patch_h, patch_w, _, _ = self.transformer._pad_to_patch_size(latents.shape[3], latents.shape[4])
        layout = MultiviewLayout(
            num_views=num_views,
            latent_frames=latent_t,
            patch_height=patch_h,
            patch_width=patch_w,
            condition_frame_indexes=tuple(condition_indexes),
            attention_scope=self.multiview_attention_scope,  # type: ignore[arg-type]
            backend=self.multiview_backend,
        )

        # Same contract as the other Cosmos3 pipelines: no packaged default, an
        # unsupplied negative prompt is empty. Reference-parity runs must pass
        # the reference negative prompt explicitly (see the recipe); serializing
        # it with default json separators is the caller's job, exactly as it is
        # for Cosmos3-Nano and Cosmos3-Super.
        negative_prompt = request_negative_prompt
        if negative_prompt is None:
            negative_prompt = self._get_sp_param(sp, "negative_prompt", None)
        if negative_prompt is None:
            negative_prompt = ""
        negative_prompt = str(negative_prompt)
        cond_ids, cond_mask, uncond_ids, uncond_mask = self._format_and_tokenize_prompts(
            prompt,
            negative_prompt,
            num_frames,
            frame_rate,
            height,
            width,
            max_sequence_length,
            sp,
            use_system_prompt=True,
            system_prompt=COSMOS3_TRANSFER_SYSTEM_PROMPT,
            prompt_suffix=COSMOS3_MULTIVIEW_EMPHASIS,
            use_duration_template=True,
            use_resolution_template=True,
            negative_metadata_mode="none",
            aspect_ratio_override="16,9",
        )

        guidance_scale = min(
            7.0,
            max(0.0, self._resolve_guidance_scale(sp, COSMOS3_T2V_DEFAULT_GUIDANCE_SCALE)),
        )
        num_inference_steps = int(sp.num_inference_steps or COSMOS3_T2V_DEFAULT_NUM_INFERENCE_STEPS)
        flow_shift = float(self._get_sp_param(sp, "flow_shift", COSMOS3_VIDEO_DEFAULT_FLOW_SHIFT))
        self._guidance_scale = guidance_scale
        self._num_timesteps = num_inference_steps
        self._set_flow_shift(flow_shift)
        self._set_timesteps(num_inference_steps, device=self.device, shift=flow_shift)

        video_shape = tuple(int(dim) for dim in latents.shape[2:])
        shared_kwargs = {
            "video_shape": video_shape,
            "fps": frame_rate,
            "noisy_frame_mask": velocity_mask,
            "control_latents": [control_latents],
            "transfer_share_vision_temporal_positions": True,
            "multiview_layout": layout,
        }
        latents = self.diffuse(
            latents=latents,
            timesteps=self.scheduler.timesteps,
            cond_ids=cond_ids,
            cond_mask=cond_mask,
            uncond_ids=uncond_ids,
            uncond_mask=uncond_mask,
            guidance_scale=guidance_scale,
            shared_kwargs=shared_kwargs,
            velocity_mask=velocity_mask,
            condition_latents=condition_latents,
            generator=generator,
            session_id=getattr(req, "request_id", None),
        )
        video = self._decode_multiview_latents(
            latents,
            num_views=num_views,
            latent_frames_per_view=latent_frames_per_view,
        ).clamp(-1, 1)
        return DiffusionOutput(
            output={
                "payload": {"video": video},
                "metadata": {
                    "multiview": {
                        "cameras": list(self.multiview_cameras),
                        "frames_per_view": num_frames,
                        "fps": frame_rate,
                    }
                },
            }
        )


__all__ = [
    "COSMOS3_MADS_CAMERAS",
    "Cosmos3MultiviewPipeline",
    "get_cosmos3_ir_op_priority_func",
    "get_cosmos3_post_process_func",
]
