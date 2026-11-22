# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3-Nano-Sim-Auto pipeline: causal rolling multiview RGB (+ LiDAR) generation.

One request is one full rollout: per-camera WSM control videos, per-camera
captions, an optional first-frame RGB condition and, for joint checkpoints, a
numeric LiDAR HD-map control (plus an optional measured first sweep). The
pipeline prepares those inputs once per session, then generates chunk by chunk
with the four-step student schedule, attending the two-slot KV ring through the
causal maskless planner and committing each chunk with a clean refresh. On the
AR-Diffusion engine the ring lives in the runner's paged pool (one padded chunk
per frame unit, sink 1 / window ``cache_chunks - 1``); on the plain engine a
model-owned dense ring serves as the numerical oracle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, ClassVar

import torch

from vllm.logger import init_logger
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3.fixed_step_sde import fixed_step_sample, initial_noise
from vllm_omni.diffusion.models.cosmos3.lidar import Cosmos3LidarDecoder, Cosmos3LidarEncoder
from vllm_omni.diffusion.models.cosmos3.multiview_maskless_attention import resolve_maskless_kernel
from vllm_omni.diffusion.models.cosmos3.multiview_prompts import (
    COSMOS3_AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
    COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
    format_rig_view_captions,
)
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_ir_op_priority_func,
)
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3_multiview import (
    COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH,
    _media_kind,
    _pad_multiview_view_video,
    _resolve_multiview_frame_rate,
    _resolve_multiview_geometry,
    _resolve_multiview_num_frames,
    get_cosmos3_multiview_post_process_func,
    get_cosmos3_multiview_pre_process_func,
)
from vllm_omni.diffusion.models.cosmos3.transfer import as_bool, media_to_uint8_cthw, uint8_cthw_to_normalized_5d
from vllm_omni.diffusion.models.cosmos3.utils import VIDEO_RES_SIZE_INFO
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionRequestKVSpec,
    ARDiffusionRequestRejectedError,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import compute_slot_mapping
from vllm_omni.model_extras.cosmos3 import (
    COSMOS3_MULTIVIEW_UNSUPPORTED_NEGATIVE_FIELDS,
    COSMOS3_TRANSFER_HINT_KEYS,
    reject_multiview_negative_fields,
    validate_multiview_request,
)
from vllm_omni.model_extras.cosmos3_lidar import load_lidar_frames

from .causal_plan import CausalMasklessPlan, build_causal_maskless_plan
from .config import Cosmos3NanoSimAutoManifest, validate_sim_auto_parallel_config
from .geometry import (
    Chunk,
    SensorGrid,
    build_chunk_schedule,
    chunk_token_count,
    latent_frames_from_pixel_frames,
    lidar_grid,
    lidar_num_sweeps,
    max_chunk_tokens,
    paged_frame_unit,
    rgb_grid,
)
from .layout import (
    CaptionDocument,
    CaptionMetadata,
    StreamItem,
    TokenMetadata,
    build_chunk_items,
    caption_metadata,
    chunk_token_metadata,
    committed_metadata,
)
from .state_cosmos3_nano_sim_auto import Cosmos3NanoSimAutoSessionState, DenseReplayRing, ring_slot_for_step
from .transformer_cosmos3_nano_sim_auto import CausalMasklessRuntime, Cosmos3NanoSimAutoTransformer

logger = init_logger(__name__)

_SYSTEM_PROMPTS = {
    "av_multiview_transfer": COSMOS3_AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
    "av_joint_camera_lidar_transfer": COSMOS3_AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
}


def get_cosmos3_nano_sim_auto_pre_process_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_multiview_pre_process_func(od_config)


def get_cosmos3_nano_sim_auto_post_process_func(od_config: OmniDiffusionConfig):
    process_video = get_cosmos3_multiview_post_process_func(od_config)

    def post_process(output: dict[str, Any], output_type: str = "np", sampling_params=None):
        processed = process_video(output, output_type=output_type, sampling_params=sampling_params)
        if (latents := output.get("payload", {}).get("latents")) is not None:
            processed["payload"]["latents"] = latents
        return processed

    return post_process


def get_cosmos3_nano_sim_auto_ir_op_priority_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_ir_op_priority_func(od_config)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _reject(message: str) -> ARDiffusionRequestRejectedError:
    return ARDiffusionRequestRejectedError(f"Cosmos3-Nano-Sim-Auto: {message}")


class Cosmos3NanoSimAutoPipeline(Cosmos3OmniDiffusersPipeline):
    """Cosmos3-Nano-Sim-Auto rollout on the AR-Diffusion paged ring or the dense oracle ring."""

    dummy_run_num_frames: ClassVar[int] = 0
    _transformer_cls_override: ClassVar[type[Cosmos3NanoSimAutoTransformer]] = Cosmos3NanoSimAutoTransformer
    _MAIN_BRANCH = "main"
    _TEXT_CACHE = "text"
    _SESSION_CAPACITY = 1
    _ar_diffusion_kv_state = None
    _bound_session_id: str | None = None

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        validate_sim_auto_parallel_config(od_config)
        super().__init__(od_config=od_config, prefix=prefix)
        self.manifest = Cosmos3NanoSimAutoManifest.from_od_config(od_config)
        if not isinstance(self.transformer, Cosmos3NanoSimAutoTransformer):
            raise TypeError(
                f"Cosmos3-Nano-Sim-Auto resolved the wrong transformer type: {type(self.transformer).__name__}."
            )
        if not self.is_distilled_model:
            raise ValueError("Cosmos3-Nano-Sim-Auto requires a distilled fixed-step checkpoint.")
        scheduler_t_list = tuple(float(value) for value in self._scheduler_init_t_list)
        if len(scheduler_t_list) != len(self.manifest.t_list) or any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8)
            for a, b in zip(scheduler_t_list, self.manifest.t_list, strict=True)
        ):
            raise ValueError(
                "Cosmos3-Nano-Sim-Auto scheduler and transformer manifests define different fixed-step schedules: "
                f"scheduler={scheduler_t_list}, transformer={self.manifest.t_list}."
            )
        if int(self.scheduler.config.num_train_timesteps) != self.manifest.num_train_timesteps:
            raise ValueError("Cosmos3-Nano-Sim-Auto scheduler and manifest disagree on num_train_timesteps.")
        self.lidar_encoder: Cosmos3LidarEncoder | None = None
        self.lidar_decoder: Cosmos3LidarDecoder | None = None
        self.lidar_grid: SensorGrid | None = None
        if self.manifest.lidar is not None:
            lidar_config = dict(self.manifest.lidar)
            self.lidar_encoder = Cosmos3LidarEncoder.from_pretrained(od_config.model, lidar_config, self.device)
            self.lidar_decoder = Cosmos3LidarDecoder.from_pretrained(od_config.model, lidar_config, self.device)
            network = lidar_config["network_config"]
            assert self.manifest.lidar_latent_patch_size_hw is not None
            self.lidar_grid = lidar_grid(
                model_height=network["resolution"][0],
                model_width=network["resolution"][1],
                spatial_compression=tuple(lidar_config["spatial_compression"]),
                patch_size=self.manifest.lidar_latent_patch_size_hw,
            )
        self._maskless_kernel: str | None = None
        self._plan_cache: dict[tuple, CausalMasklessPlan] = {}
        self._states: OrderedDict[str, Cosmos3NanoSimAutoSessionState] = OrderedDict()

    # -- weights -----------------------------------------------------------

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Reject artifact tensors that do not map into this exact model."""
        allowed = set(self.state_dict())
        tp_aware = {name for name, parameter in self.named_parameters() if hasattr(parameter, "weight_loader")}
        unexpected: list[str] = []

        def is_export_only_tensor(name: str) -> bool:
            key = name.removeprefix("transformer.").removeprefix("model.")
            return key.startswith("lm_head.")

        def checked_weights():
            for name, tensor in weights:
                remapped = self._remap_ckpt_key(name)
                if (
                    not is_export_only_tensor(name)
                    and name not in allowed
                    and name not in tp_aware
                    and (remapped is None or (remapped not in allowed and remapped not in tp_aware))
                ):
                    unexpected.append(name)
                yield name, tensor

        loaded = super().load_weights(checked_weights())
        if unexpected:
            preview = ", ".join(sorted(unexpected)[:12])
            suffix = "" if len(unexpected) <= 12 else f" (and {len(unexpected) - 12} more)"
            raise ValueError(
                f"Cosmos3-Nano-Sim-Auto checkpoint contains unexpected transformer tensors: {preview}{suffix}"
            )
        return loaded

    # -- geometry ----------------------------------------------------------

    def _rgb_grid(self, height: int, width: int) -> SensorGrid:
        return rgb_grid(
            height,
            width,
            vae_spatial_compression=self.manifest.vae_spatial_compression_factor,
            patch_size=self.manifest.latent_patch_size,
        )

    def _rgb_seconds_per_frame(self, fps: float) -> float:
        return self.manifest.temporal_compression_factor / fps

    def _frame_unit(self, *, num_views: int, joint: bool, height: int, width: int, fps: float) -> int:
        """Paged frame unit for a geometry: the largest (regular) chunk, padded to the page alignment."""
        lidar_spf = self.manifest.lidar_seconds_per_frame
        rgb_spf = self._rgb_seconds_per_frame(fps)
        sweeps_per_chunk = round(self.manifest.frames_per_chunk * rgb_spf / lidar_spf) if joint and lidar_spf else 0
        regular = Chunk(
            step=1,
            rgb_frames=tuple(range(1, 1 + self.manifest.frames_per_chunk)),
            lidar_sweeps=tuple(range(1, 1 + sweeps_per_chunk)) if joint else (),
            rgb_prefix_end=1 + self.manifest.frames_per_chunk,
            lidar_prefix_end=1 + sweeps_per_chunk if joint else 0,
        )
        tokens = chunk_token_count(
            regular,
            num_views=num_views,
            rgb_tokens_per_frame=self._rgb_grid(height, width).tokens_per_frame,
            lidar_tokens_per_sweep=self.lidar_grid.tokens_per_frame if joint and self.lidar_grid else None,
        )
        return paged_frame_unit(tokens)

    # -- AR-Diffusion capability ---------------------------------------------

    def _kv_spec(self, *, num_views: int, joint: bool, height: int, width: int, fps: float) -> ARDiffusionKVCacheSpec:
        if joint and self.lidar_grid is None:
            raise ValueError("Joint requests require a checkpoint with LiDAR metadata.")
        return ARDiffusionKVCacheSpec(
            num_layers=self.transformer.num_hidden_layers,
            num_kv_heads=self.transformer.num_kv_heads_local,
            head_size=self.transformer.head_dim,
            tokens_per_frame=self._frame_unit(num_views=num_views, joint=joint, height=height, width=width, fps=fps),
            frames_per_block=1,
            window_frames=self.manifest.cache_chunks - self.manifest.sink_chunks,
            sink_frames=self.manifest.sink_chunks,
            kv_branches=(ARDiffusionKVBranchSpec(self._MAIN_BRANCH, 0),),
            session_capacity=self._SESSION_CAPACITY,
            cross_attention=(ARDiffusionCrossAttentionKVSpec(self._TEXT_CACHE, self.manifest.text_cache_max_len),),
            max_scratch_frames_per_branch=0,
            max_scratch_tokens_per_branch=0,
        )

    def _default_geometry(self) -> tuple[int, bool, int, int, float]:
        width, height = VIDEO_RES_SIZE_INFO[self.manifest.default_resolution]["16,9"]
        return len(self.manifest.cameras), self.manifest.joint, height, width, self.manifest.default_fps

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        num_views, joint, height, width, fps = self._default_geometry()
        return self._kv_spec(num_views=num_views, joint=joint, height=height, width=width, fps=fps)

    def ar_diffusion_default_request_spec(self) -> ARDiffusionRequestKVSpec:
        num_views, joint, height, width, fps = self._default_geometry()
        return ARDiffusionRequestKVSpec(
            kv_spec=self._kv_spec(num_views=num_views, joint=joint, height=height, width=width, fps=fps),
            geometry_key=(num_views, joint, height, width, fps),
        )

    def ar_diffusion_request_spec(self, request: Any) -> ARDiffusionRequestKVSpec:
        sp = request.sampling_params
        extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
        multiview, views = validate_multiview_request(extra, self.manifest.cameras, media_kind=_media_kind)
        joint = extra.get("lidar") is not None
        _, _, width, height = _resolve_multiview_geometry(
            sp, multiview, views, default_resolution=self.manifest.default_resolution
        )
        fps = self._request_fps(sp)
        key = (len(views), joint, height, width, fps)
        return ARDiffusionRequestKVSpec(
            kv_spec=self._kv_spec(num_views=len(views), joint=joint, height=height, width=width, fps=fps),
            geometry_key=key,
        )

    def ar_diffusion_worst_case_request_specs(self) -> Iterable[ARDiffusionRequestKVSpec]:
        sizes = VIDEO_RES_SIZE_INFO[self.manifest.default_resolution]
        width, height = max(sizes.values(), key=lambda size: size[0] * size[1])
        num_views = len(self.manifest.cameras)
        yield ARDiffusionRequestKVSpec(
            kv_spec=self._kv_spec(
                num_views=num_views,
                joint=self.manifest.joint,
                height=height,
                width=width,
                fps=self.manifest.default_fps,
            ),
            geometry_key=(num_views, self.manifest.joint, height, width, self.manifest.default_fps),
        )

    def validate_ar_diffusion_effective_spec(self, spec: ARDiffusionKVCacheSpec) -> None:
        expected = self.ar_diffusion_kv_cache_spec()
        fields = (
            "num_layers",
            "num_kv_heads",
            "head_size",
            "frames_per_block",
            "window_frames",
            "sink_frames",
            "reset_at_boundary",
            "kv_branches",
            "session_capacity",
            "cross_attention",
            "max_scratch_frames_per_branch",
            "max_scratch_tokens_per_branch",
        )
        mismatches = {
            name: (getattr(expected, name), getattr(spec, name))
            for name in fields
            if getattr(expected, name) != getattr(spec, name)
        }
        if mismatches:
            detail = ", ".join(f"{name}=expected {exp!r}, got {got!r}" for name, (exp, got) in mismatches.items())
            raise ValueError(f"Cosmos3-Nano-Sim-Auto AR-Diffusion structural specification is invalid ({detail}).")

    @contextmanager
    def bind_ar_diffusion_state(self, session_id, state):
        if self._ar_diffusion_kv_state is not None:
            raise RuntimeError("Cosmos3-Nano-Sim-Auto AR-Diffusion state is already bound.")
        if state.session_id != session_id:
            raise ValueError(f"Cosmos3-Nano-Sim-Auto bound session mismatch: {state.session_id!r} != {session_id!r}.")
        self._ar_diffusion_kv_state = state
        self._bound_session_id = str(session_id)
        try:
            yield
        finally:
            self._ar_diffusion_kv_state = None
            self._bound_session_id = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        self._drop_session(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        self._drop_session(session_id)

    def _drop_session(self, session_id: str) -> None:
        state = self._states.pop(str(session_id or "default"), None)
        if state is not None:
            state.reset()

    def _get_or_create_state(self, session_id: str) -> Cosmos3NanoSimAutoSessionState:
        state = self._states.get(session_id)
        if state is None:
            while len(self._states) >= self._SESSION_CAPACITY:
                _, evicted = self._states.popitem(last=False)
                evicted.reset()
            state = Cosmos3NanoSimAutoSessionState(session_id=session_id)
            self._states[session_id] = state
        self._states.move_to_end(session_id)
        return state

    # -- request helpers ----------------------------------------------------

    def _request_fps(self, sp: Any) -> float:
        return _resolve_multiview_frame_rate(
            _first_not_none(
                self._get_sp_param(sp, "resolved_frame_rate", None),
                self._get_sp_param(sp, "frame_rate", None),
                self._get_sp_param(sp, "fps", None),
                self.manifest.default_fps,
            )
        )

    def _maskless_kernel_name(self) -> str:
        if self._maskless_kernel is None:
            self._maskless_kernel = resolve_maskless_kernel() if torch.device(self.device).type == "cuda" else "torch"
        return self._maskless_kernel

    @staticmethod
    def _view_value(view: Mapping[str, Any], field: str) -> Any:
        return view.get(f"{field}_path", view.get(field))

    def _encode_camera_major(
        self,
        views: Sequence[Mapping[str, Any]],
        *,
        field: str,
        height: int,
        width: int,
        num_frames: int,
    ) -> torch.Tensor:
        """Encode one media field of every camera independently: ``[1, C, V*T_lat, h, w]``."""
        latents = []
        for view in views:
            value = self._view_value(view, field)
            if value is None:
                raise ValueError(f"Camera {view['camera_key']!r} is missing {field} input.")
            frames = media_to_uint8_cthw(value, height=height, width=width, max_frames=num_frames)
            frames = _pad_multiview_view_video(frames, num_frames=num_frames, height=height, width=width)
            latents.append(self._encode_video_tensor(uint8_cthw_to_normalized_5d(frames, dtype=self.dtype)))
        lengths = {int(latent.shape[2]) for latent in latents}
        if len(lengths) != 1:
            raise ValueError(f"Per-camera VAE encodes have unequal latent lengths: {lengths}.")
        return torch.cat(latents, dim=2)

    def _encode_first_frames(self, views: Sequence[Mapping[str, Any]], *, height: int, width: int) -> torch.Tensor:
        """Latent 0 of every camera from its first RGB frame (the causal Wan encoder makes it exact)."""
        latents = []
        for view in views:
            value = self._view_value(view, "vision")
            frames = media_to_uint8_cthw(value, height=height, width=width, max_frames=1)
            frames = _pad_multiview_view_video(frames, num_frames=1, height=height, width=width)
            latents.append(self._encode_video_tensor(uint8_cthw_to_normalized_5d(frames, dtype=self.dtype))[:, :, :1])
        return torch.cat(latents, dim=2)

    def _encode_lidar_condition(self, lidar_request: Mapping[str, Any], *, num_sweeps: int) -> torch.Tensor:
        assert self.lidar_encoder is not None
        count = int(lidar_request.get("num_conditional_sweeps", 1))
        if count != 1:
            raise ValueError("Cosmos3-Nano-Sim-Auto supports exactly one measured LiDAR condition sweep.")
        if num_sweeps <= count:
            raise ValueError("The request must generate at least one LiDAR sweep beyond the condition.")
        frames = load_lidar_frames(lidar_request["condition_path"], num_sweeps=count)
        chunk = int(self.lidar_encoder.config["streaming_chunk_frames"])
        padded = min(math.ceil(count / chunk) * chunk, num_sweeps)
        if padded > count:
            frames = torch.cat([frames, frames.new_zeros(frames.shape[0], padded - count, *frames.shape[2:])], dim=1)
        return self.lidar_encoder(frames)[:, :, :count].to(device=self.device, dtype=self.sampling_dtype)

    # -- forward --------------------------------------------------------------

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        try:
            return self._forward_impl(req)
        except ARDiffusionRequestRejectedError:
            raise
        except Exception:
            extra = req.sampling_params.extra_args or {}
            self._drop_session(str(extra.get("session_id") or self._bound_session_id or "default"))
            raise

    def _forward_impl(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        # ---- Admission (pure; raises ARDiffusionRequestRejectedError only) ---------
        if len(req.prompts) != 1:
            raise _reject("exactly one prompt per request is supported.")
        prompt_data = req.prompts[0]
        sp = req.sampling_params
        extra = sp.extra_args if isinstance(sp.extra_args, Mapping) else {}
        if sp.extra_args is not None and not isinstance(sp.extra_args, Mapping):
            raise _reject(f"extra_args must be a mapping, got {type(sp.extra_args).__name__}.")
        if extra.get("ar_diffusion_tick") is not None:
            raise _reject("typed tick sessions are not supported yet; submit a full rollout request.")
        try:
            if isinstance(prompt_data, Mapping):
                reject_multiview_negative_fields(prompt_data, "prompt")
            reject_multiview_negative_fields(
                {field: getattr(sp, field, None) for field in COSMOS3_MULTIVIEW_UNSUPPORTED_NEGATIVE_FIELDS},
                "sampling_params",
            )
            multiview, views = validate_multiview_request(extra, self.manifest.cameras, media_kind=_media_kind)
            selected_hints = [key for key in COSMOS3_TRANSFER_HINT_KEYS if extra.get(key) is not None]
            if selected_hints != ["wsm"]:
                raise ValueError(
                    "Cosmos3-Nano-Sim-Auto is a WSM transfer student: supply `wsm` and a control per camera."
                )
            cameras = [str(view["camera_key"]) for view in views]
            self.manifest.validate_request_cameras(cameras)
            lidar_request = extra.get("lidar")
            joint = lidar_request is not None
            if joint and not self.manifest.joint:
                raise ValueError("Joint camera+LiDAR requests require a joint checkpoint with LiDAR metadata.")
            num_frames = _resolve_multiview_num_frames(
                _first_not_none(multiview.get("num_frames"), sp.num_frames), self.manifest.temporal_compression_factor
            )
            resolution, aspect_ratio, width, height = _resolve_multiview_geometry(
                sp, multiview, views, default_resolution=self.manifest.default_resolution
            )
            fps = self._request_fps(sp)
            if joint and not math.isclose(fps, self.manifest.default_fps):
                raise ValueError(
                    f"Joint inference requires fps={self.manifest.default_fps:g} to match the LiDAR source clock, "
                    f"got {fps:g}."
                )
            guidance = _first_not_none(self._get_sp_param(sp, "guidance_scale", None), extra.get("guidance"), 1.0)
            if float(guidance) != 1.0:
                raise ValueError(f"the distilled student runs guidance 1.0 only, got {guidance}.")
            if sp.num_inference_steps not in (None, len(self.manifest.t_list)):
                raise ValueError(
                    f"the student uses its fixed {len(self.manifest.t_list)}-step schedule; "
                    f"got {sp.num_inference_steps}."
                )
            known = [self._view_value(view, "vision") is not None for view in views]
            if any(known) and not all(known):
                raise ValueError("RGB conditions must cover every camera or none.")
            has_condition = all(known)
            if has_condition and not as_bool(multiview.get("condition_video_as_image"), True):
                raise ValueError(
                    "Cosmos3-Nano-Sim-Auto conditions on each camera's first frame only (condition_video_as_image)."
                )
            lidar_condition = bool(joint and lidar_request.get("condition_path") is not None)
            if joint and lidar_request.get("num_conditional_sweeps", 1) != 1:
                raise ValueError("Cosmos3-Nano-Sim-Auto supports exactly one measured LiDAR condition sweep.")
        except (TypeError, ValueError) as exc:
            raise _reject(str(exc)) from exc

        session_id = str(extra.get("session_id") or "default")
        reset = as_bool(extra.get("reset"), True)
        close_session = as_bool(extra.get("close_session"), True)
        if self._bound_session_id is not None and session_id != self._bound_session_id:
            raise _reject(f"request session {session_id!r} does not match bound session {self._bound_session_id!r}.")
        existing = self._states.get(session_id)
        if existing is not None and existing.fingerprint is not None and not reset:
            raise _reject("continuing a live session needs tick requests, which are not supported yet; set reset=true.")

        # Captions are pure text work: tokenize before any side effect so an oversize prompt rejects cleanly.
        captions = format_rig_view_captions([view["prompt"] for view in views], cameras)
        # The reference's explicit system prompt overrides vlm_config.use_system_prompt=False.
        # Our formatter gates on the Boolean, so keep it True to include that exact message.
        system_prompt = _SYSTEM_PROMPTS[self.manifest.system_prompt_joint if joint else self.manifest.system_prompt_rgb]
        caption_ids: list[torch.Tensor] = []
        for caption in captions:
            ids, mask, _, _ = self._format_and_tokenize_prompts(
                caption,
                "",
                num_frames,
                fps,
                height,
                width,
                COSMOS3_MULTIVIEW_MAX_SEQUENCE_LENGTH,
                sp,
                use_system_prompt=True,
                system_prompt=system_prompt,
                prompt_suffix=None,
                use_duration_template=True,
                use_resolution_template=True,
                negative_metadata_mode="none",
                aspect_ratio_override=aspect_ratio,
                truncate_duration=not self.manifest.fractional_duration,
            )
            caption_ids.append(ids[mask.bool()].reshape(1, -1))
        sink_tokens = min(self.manifest.text_sink_tokens, min(int(ids.shape[1]) for ids in caption_ids))
        und_tokens = sum(int(ids.shape[1]) + sink_tokens for ids in caption_ids)
        if und_tokens > self.manifest.text_cache_max_len:
            raise _reject(
                f"per-camera captions need {und_tokens} UND tokens but "
                f"text_cache_max_len={self.manifest.text_cache_max_len}."
            )
        seed = self._resolve_seed(sp, sp.generator if isinstance(sp.generator, torch.Generator) else None)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "manifest": self.manifest.digest,
                    "cameras": cameras,
                    "joint": joint,
                    "height": height,
                    "width": width,
                    "fps": fps,
                    "num_frames": num_frames,
                    "captions": captions,
                    "condition": has_condition,
                    "lidar_condition": lidar_condition,
                    "seed": seed,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        # ---- Side effects begin ---------------------------------------------------
        if reset and existing is not None:
            self._drop_session(session_id)
        state = self._get_or_create_state(session_id)
        bound = self._ar_diffusion_kv_state
        try:
            self._prepare_session(
                state,
                views=views,
                cameras=cameras,
                joint=joint,
                lidar_request=lidar_request,
                lidar_condition=lidar_condition,
                has_condition=has_condition,
                height=height,
                width=width,
                num_frames=num_frames,
                fps=fps,
                seed=seed,
                caption_ids=caption_ids,
                sink_tokens=sink_tokens,
                bound=bound,
            )
            state.fingerprint = fingerprint
            for chunk in state.schedule[state.next_chunk :]:
                self._run_chunk(state, chunk, bound)
            payload, metadata = self._build_payload(
                state,
                lidar_request,
                views,
                resolution,
                aspect_ratio,
                return_latents=as_bool(extra.get("return_latents"), False),
            )
        finally:
            state.terminal = True
        if close_session and bound is None:
            self._drop_session(session_id)
        return DiffusionOutput(
            output={"payload": payload, "metadata": metadata},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    # -- preparation ------------------------------------------------------------

    def _prepare_session(
        self,
        state: Cosmos3NanoSimAutoSessionState,
        *,
        views: Sequence[Mapping[str, Any]],
        cameras: list[str],
        joint: bool,
        lidar_request: Mapping[str, Any] | None,
        lidar_condition: bool,
        has_condition: bool,
        height: int,
        width: int,
        num_frames: int,
        fps: float,
        seed: int,
        caption_ids: list[torch.Tensor],
        sink_tokens: int,
        bound: Any,
    ) -> None:
        num_views = len(views)
        rgb = self._rgb_grid(height, width)
        latents_per_view = latent_frames_from_pixel_frames(num_frames, self.manifest.temporal_compression_factor)
        sweeps = None
        if joint:
            assert self.manifest.lidar_fps is not None
            sweeps = lidar_num_sweeps(num_frames, camera_fps=fps, lidar_fps=self.manifest.lidar_fps)
        schedule = build_chunk_schedule(
            rgb_latents=latents_per_view,
            lidar_sweeps=sweeps,
            frames_per_chunk=self.manifest.frames_per_chunk,
            rgb_seconds_per_frame=self._rgb_seconds_per_frame(fps),
            lidar_seconds_per_frame=self.manifest.lidar_seconds_per_frame if joint else None,
        )
        frame_unit = paged_frame_unit(
            max_chunk_tokens(
                schedule,
                num_views=num_views,
                rgb_tokens_per_frame=rgb.tokens_per_frame,
                lidar_tokens_per_sweep=self.lidar_grid.tokens_per_frame if joint and self.lidar_grid else None,
            )
        )
        if bound is not None:
            spec = bound.kv_cache.spec
            if spec.chunk_size != frame_unit or bound.kv_cache.block_size != frame_unit:
                raise RuntimeError(
                    "Cosmos3-Nano-Sim-Auto bound KV pool geometry does not match the request: "
                    f"pool chunk={spec.chunk_size}, block={bound.kv_cache.block_size}, request frame unit={frame_unit}."
                )

        control = self._encode_camera_major(views, field="control", height=height, width=width, num_frames=num_frames)
        if control.shape[2] != num_views * latents_per_view:
            raise RuntimeError(
                f"WSM control latents hold {control.shape[2]} frames, expected {num_views * latents_per_view}."
            )
        target = torch.zeros_like(control)
        condition_frames: frozenset[int] = frozenset()
        if has_condition:
            first = self._encode_first_frames(views, height=height, width=width)  # [1, C, V, h, w]
            target.view(1, target.shape[1], num_views, latents_per_view, *target.shape[3:])[:, :, :, 0] = first.to(
                target.dtype
            )
            condition_frames = frozenset({0})

        lidar_control = lidar_target = None
        lidar_condition_sweeps: frozenset[int] = frozenset()
        if joint:
            assert self.lidar_encoder is not None and lidar_request is not None and sweeps is not None
            frames = load_lidar_frames(lidar_request["control_path"], num_sweeps=sweeps)
            lidar_control = self.lidar_encoder(frames).to(device=self.device, dtype=self.sampling_dtype)
            del frames
            lidar_target = torch.zeros_like(lidar_control)
            if lidar_condition:
                lidar_target[:, :, :1] = self._encode_lidar_condition(lidar_request, num_sweeps=sweeps)
                lidar_condition_sweeps = frozenset({0})

        # Captions: one causal UND pass per camera (positions rewound to zero); the
        # sink document is the caption's own first tokens, so its K/V are a prefix slice.
        documents: list[CaptionDocument] = []
        per_view_kv: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        horizon_seconds = num_frames / fps
        for view_index, ids in enumerate(caption_ids):
            ids = ids.to(self.device)
            kv, length = self.transformer.encode_und_kv(ids, torch.ones_like(ids))
            parts = [(k[:, :sink_tokens], v[:, :sink_tokens]) for k, v in kv] if sink_tokens else []
            layers = []
            for layer_idx, (k, v) in enumerate(kv):
                if sink_tokens:
                    layers.append(
                        (torch.cat([parts[layer_idx][0], k], dim=1), torch.cat([parts[layer_idx][1], v], dim=1))
                    )
                else:
                    layers.append((k, v))
            per_view_kv.append(layers)
            if sink_tokens:
                documents.append(CaptionDocument(view_index, sink_tokens, is_sink=True))
            documents.append(CaptionDocument(view_index, length, start_seconds=0.0, end_seconds=horizon_seconds))
        und_kv = [
            (
                torch.cat([per_view_kv[view][layer][0] for view in range(num_views)], dim=1),
                torch.cat([per_view_kv[view][layer][1] for view in range(num_views)], dim=1),
            )
            for layer in range(self.transformer.num_hidden_layers)
        ]
        if bound is not None:
            padded = self.transformer.pad_text_kv(und_kv, max_len=self.manifest.text_cache_max_len)
            bound.populate_cross_attention(self._MAIN_BRANCH, self._TEXT_CACHE, padded)
            state.und_kv = None
        else:
            state.und_kv = und_kv
            state.dense_ring = DenseReplayRing(
                num_layers=self.transformer.num_hidden_layers,
                cache_chunks=self.manifest.cache_chunks,
                frame_unit=frame_unit,
                num_kv_heads=self.transformer.num_kv_heads_local,
                head_dim=self.transformer.head_dim,
                device=self.device,
                dtype=und_kv[0][0].dtype,
            )

        state.cameras = tuple(cameras)
        state.joint = joint
        state.height, state.width, state.fps, state.num_frames = height, width, fps, num_frames
        state.schedule = schedule
        state.frame_unit = frame_unit
        state.text_extent = max(int(ids.shape[1]) for ids in caption_ids)
        state.seed = seed
        state.rgb_control_latents = control
        state.rgb_target_latents = target
        state.rgb_condition_frames = condition_frames
        state.lidar_control_latents = lidar_control
        state.lidar_target_latents = lidar_target
        state.lidar_condition_sweeps = lidar_condition_sweeps
        state.caption_documents = tuple(documents)
        state.next_chunk = 0

    # -- chunk execution ------------------------------------------------------------

    def _items_for_chunk(self, state: Cosmos3NanoSimAutoSessionState, chunk: Chunk) -> tuple[StreamItem, ...]:
        return build_chunk_items(
            chunk,
            num_views=state.num_views,
            rgb_grid=self._rgb_grid(state.height, state.width),
            rgb_seconds_per_frame=self._rgb_seconds_per_frame(state.fps),
            lidar_grid=self.lidar_grid if state.joint else None,
            lidar_seconds_per_frame=self.manifest.lidar_seconds_per_frame if state.joint else None,
            with_controls=True,
            rgb_condition_frames=state.rgb_condition_frames,
            lidar_condition_sweeps=state.lidar_condition_sweeps,
        )

    @staticmethod
    def _camera_major_slice(
        latents: torch.Tensor, *, num_views: int, frames_per_view: int, frames: Sequence[int]
    ) -> torch.Tensor:
        """Select ``frames`` of every view from camera-major ``[1, C, V*T, h, w]`` -> ``[1, C, V*n, h, w]``."""
        grid = latents.view(1, latents.shape[1], num_views, frames_per_view, *latents.shape[3:])
        index = torch.tensor(list(frames), device=latents.device, dtype=torch.long)
        return grid.index_select(3, index).reshape(1, latents.shape[1], num_views * len(frames), *latents.shape[3:])

    @staticmethod
    def _camera_major_write(
        latents: torch.Tensor, values: torch.Tensor, *, num_views: int, frames_per_view: int, frames: Sequence[int]
    ) -> None:
        grid = latents.view(1, latents.shape[1], num_views, frames_per_view, *latents.shape[3:])
        index = torch.tensor(list(frames), device=latents.device, dtype=torch.long)
        grid.index_copy_(3, index, values.view(1, values.shape[1], num_views, len(frames), *values.shape[3:]))

    def _item_latents(self, state: Cosmos3NanoSimAutoSessionState, items: Sequence[StreamItem]) -> list[torch.Tensor]:
        assert state.rgb_control_latents is not None and state.rgb_target_latents is not None
        frames_per_view = state.rgb_control_latents.shape[2] // state.num_views
        latents = []
        for item in items:
            if item.is_lidar:
                source = state.lidar_control_latents if item.is_control else state.lidar_target_latents
                assert source is not None
                latents.append(source[:, :, list(item.frames)])
            else:
                source = state.rgb_control_latents if item.is_control else state.rgb_target_latents
                latents.append(
                    self._camera_major_slice(
                        source, num_views=state.num_views, frames_per_view=frames_per_view, frames=item.frames
                    )
                )
        return latents

    def _caption_metadata(self, state: Cosmos3NanoSimAutoSessionState) -> CaptionMetadata:
        return caption_metadata(state.caption_documents)

    def _runtime_plan(
        self,
        state: Cosmos3NanoSimAutoSessionState,
        chunk: Chunk,
        metadata: TokenMetadata,
        pass_kind: str,
        bound: Any,
    ) -> CausalMasklessRuntime:
        history = state.history_slots(frame_unit=state.frame_unit)
        # Plans depend on the chunk shape, on which ring slots are populated and on
        # whether chunk 0 is still inside the cross-view window (only for chunk 1);
        # absolute time otherwise only shifts every stream together.
        key = (
            pass_kind,
            (len(chunk.rgb_frames), len(chunk.lidar_sweeps)),
            min(chunk.step, 2),
            tuple(
                (slot.slot, slot.metadata.num_tokens, int(slot.metadata.role.min()), int(slot.metadata.role.max()))
                for slot in history
            ),
            tuple(sorted(state.rgb_condition_frames & set(chunk.rgb_frames))),
            tuple(sorted(state.lidar_condition_sweeps & set(chunk.lidar_sweeps))),
            state.text_extent,
            tuple(doc.num_tokens for doc in state.caption_documents),
            state.num_views,
            state.joint,
        )
        plan = self._plan_cache.get(key)
        if plan is None:
            plan = build_causal_maskless_plan(
                current=metadata,
                history=history,
                captions=self._caption_metadata(state),
                window_seconds=self.manifest.cross_view_past_window_seconds,
                device=self.device,
            )
            self._plan_cache[key] = plan
        if bound is not None and history:
            adapter = bound.adapter(self._MAIN_BRANCH)
            block_ids = bound.kv_cache.window_block_ids(adapter)
            if len(block_ids) != len(history):
                raise RuntimeError(
                    f"Cosmos3-Nano-Sim-Auto ring has {len(history)} committed slots but the pool holds "
                    f"{len(block_ids)} blocks."
                )
            # Table order is commit order: the pinned sink first, then the most recent chunk.
            bases = [block_id * bound.kv_cache.block_size for block_id in block_ids]
            plan = plan.with_history_offsets(bases, state.frame_unit)
        return CausalMasklessRuntime(
            plan=plan.flatten(), num_passes=len(plan.passes), kernel=self._maskless_kernel_name()
        )

    def _history_kv(
        self, state: Cosmos3NanoSimAutoSessionState, bound: Any
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        if not state.ring_metadata:
            return None
        if bound is not None:
            return [
                (bound.kv_cache.key_cache(layer), bound.kv_cache.value_cache(layer))
                for layer in range(self.transformer.num_hidden_layers)
            ]
        assert state.dense_ring is not None
        return state.dense_ring.layer_kv()

    def _und_kv(self, state: Cosmos3NanoSimAutoSessionState, bound: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if bound is None:
            assert state.und_kv is not None
            return state.und_kv
        total = sum(doc.num_tokens for doc in state.caption_documents)
        pooled = bound.get_cross_attention_kv(self._MAIN_BRANCH, self._TEXT_CACHE)
        return [(entry["k"][:, :total], entry["v"][:, :total]) for entry in pooled]

    def _run_chunk(self, state: Cosmos3NanoSimAutoSessionState, chunk: Chunk, bound: Any) -> None:
        assert state.rgb_target_latents is not None
        items = self._items_for_chunk(state, chunk)
        num_views = state.num_views
        frames_per_view = state.rgb_target_latents.shape[2] // num_views
        rgb_target = next(item for item in items if item.kind == "rgb_target")
        lidar_target = next((item for item in items if item.kind == "lidar_target"), None)
        noisy_rgb_frames = [frame for frame in rgb_target.frames if frame not in rgb_target.condition_frames]
        noisy_sweeps = (
            [s for s in lidar_target.frames if s not in lidar_target.condition_frames] if lidar_target else []
        )
        und_kv = self._und_kv(state, bound)
        history_kv = self._history_kv(state, bound)

        if noisy_rgb_frames or noisy_sweeps:
            latents = self._item_latents(state, items)
            target_index = items.index(rgb_target)
            lidar_index = items.index(lidar_target) if lidar_target is not None else None
            rgb_base = latents[target_index]
            rgb_noisy_shape = (1, rgb_base.shape[1], num_views * len(noisy_rgb_frames), *rgb_base.shape[3:])
            rgb_size = math.prod(rgb_noisy_shape)
            local_rgb = [rgb_target.frames.index(frame) for frame in noisy_rgb_frames]
            lidar_size, lidar_noisy_shape, local_sweeps = 0, (1, 0), []
            if lidar_index is not None and noisy_sweeps:
                lidar_base = latents[lidar_index]
                lidar_noisy_shape = (1, lidar_base.shape[1], len(noisy_sweeps), *lidar_base.shape[3:])
                lidar_size = math.prod(lidar_noisy_shape)
                local_sweeps = [lidar_target.frames.index(s) for s in noisy_sweeps]
            metadata, _ = chunk_token_metadata(items, step=chunk.step, pass_kind="noisy")
            runtime = self._runtime_plan(state, chunk, metadata, "noisy", bound)
            # The reference seeds RGB rollouts by the chunk's first latent and joint rollouts by the step.
            sample_index = chunk.step if state.joint else chunk.rgb_frames[0]

            def velocity(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
                current = list(latents)
                if rgb_size:
                    noisy_rgb = rgb_base.clone()
                    self._camera_major_write(
                        noisy_rgb,
                        x[:, :rgb_size].reshape(rgb_noisy_shape).to(noisy_rgb.dtype),
                        num_views=num_views,
                        frames_per_view=len(rgb_target.frames),
                        frames=local_rgb,
                    )
                    current[target_index] = noisy_rgb
                if lidar_index is not None and lidar_size:
                    noisy_lidar = latents[lidar_index].clone()
                    noisy_lidar[:, :, local_sweeps] = x[:, rgb_size:].reshape(lidar_noisy_shape).to(noisy_lidar.dtype)
                    current[lidar_index] = noisy_lidar
                output = self.transformer(
                    items=items,
                    latents=current,
                    timestep=timestep.reshape(1).to(self.device),
                    text_extent=state.text_extent,
                    und_kv=und_kv,
                    history_kv=history_kv,
                    runtime=runtime,
                )
                targets = [item for item in items if not item.is_control]
                preds = []
                if rgb_size:
                    preds.append(
                        self._camera_major_slice(
                            output.targets[targets.index(rgb_target)],
                            num_views=num_views,
                            frames_per_view=len(rgb_target.frames),
                            frames=local_rgb,
                        ).reshape(1, -1)
                    )
                if lidar_index is not None and lidar_size:
                    preds.append(output.targets[targets.index(lidar_target)][:, :, local_sweeps].reshape(1, -1))
                return torch.cat(preds, dim=1).float()

            noise = initial_noise(
                (1, rgb_size + lidar_size), seed=state.seed, sample_index=sample_index, device=self.device
            )
            denoised = fixed_step_sample(
                velocity,
                noise,
                t_list=self.manifest.t_list,
                num_train_timesteps=self.manifest.num_train_timesteps,
                seed=state.seed,
                sample_index=sample_index,
                sample_type=self.manifest.sample_type,
            )
            if not torch.isfinite(denoised).all():
                raise FloatingPointError(f"Cosmos3-Nano-Sim-Auto produced non-finite latents in chunk {chunk.step}.")
            if rgb_size:
                self._camera_major_write(
                    state.rgb_target_latents,
                    denoised[:, :rgb_size].reshape(rgb_noisy_shape).to(state.rgb_target_latents.dtype),
                    num_views=num_views,
                    frames_per_view=frames_per_view,
                    frames=noisy_rgb_frames,
                )
            if lidar_target is not None and lidar_size:
                assert state.lidar_target_latents is not None
                state.lidar_target_latents[:, :, noisy_sweeps] = (
                    denoised[:, rgb_size:].reshape(lidar_noisy_shape).to(state.lidar_target_latents.dtype)
                )

        if chunk.step != state.schedule[-1].step:
            self._commit_chunk(state, chunk, items, und_kv, history_kv, bound)
        state.next_chunk = chunk.step + 1

    def _commit_chunk(
        self,
        state: Cosmos3NanoSimAutoSessionState,
        chunk: Chunk,
        items: Sequence[StreamItem],
        und_kv: list[tuple[torch.Tensor, torch.Tensor]],
        history_kv: list[tuple[torch.Tensor, torch.Tensor]] | None,
        bound: Any,
    ) -> None:
        """Clean refresh forward that publishes the chunk's K/V into its ring slot."""
        metadata, _ = chunk_token_metadata(items, step=chunk.step, pass_kind="clean")
        runtime = self._runtime_plan(state, chunk, metadata, "clean", bound)
        latents = self._item_latents(state, items)
        num_real = metadata.num_tokens
        slot = ring_slot_for_step(
            chunk.step, cache_chunks=self.manifest.cache_chunks, sink_chunks=self.manifest.sink_chunks
        )
        if bound is not None:
            adapter = bound.adapter(self._MAIN_BRANCH)
            kv_cache = bound.kv_cache
            start = int(adapter.num_computed_tokens)
            table = kv_cache.allocate_token_slots(adapter, state.frame_unit)
            slots = compute_slot_mapping(table, torch.arange(start, start + state.frame_unit), kv_cache.block_size).to(
                self.device
            )
            pools = [
                (kv_cache.key_cache(layer), kv_cache.value_cache(layer))
                for layer in range(self.transformer.num_hidden_layers)
            ]
            if num_real < state.frame_unit:
                padding = slots[num_real:]
                for key_pool, value_pool in pools:
                    key_pool.index_fill_(0, padding, 0)
                    value_pool.index_fill_(0, padding, 0)
            write_slots = slots[:num_real]
        else:
            assert state.dense_ring is not None
            # History still aliases the slot being replaced. Collect fresh K/V without
            # modifying the ring until every layer has finished reading the old history.
            pools = None
            write_slots = None
        refreshed = self.transformer(
            items=items,
            latents=latents,
            timestep=None,
            text_extent=state.text_extent,
            und_kv=und_kv,
            history_kv=history_kv,
            runtime=runtime,
            write_slots=write_slots,
            kv_pools=pools,
            collect_kv=bound is None,
        )
        if bound is not None:
            bound.kv_cache.commit_chunk(adapter)
        else:
            assert state.dense_ring is not None
            state.dense_ring.publish(slot, refreshed.current_kv, num_real)
        state.record_commit(slot=slot, step=chunk.step, metadata=committed_metadata(metadata))

    # -- outputs ------------------------------------------------------------------

    def _build_payload(
        self,
        state: Cosmos3NanoSimAutoSessionState,
        lidar_request: Mapping[str, Any] | None,
        views: Sequence[Mapping[str, Any]],
        resolution: str,
        aspect_ratio: str,
        *,
        return_latents: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        assert state.rgb_target_latents is not None
        num_views = state.num_views
        frames_per_view = state.rgb_target_latents.shape[2] // num_views
        decoded = None
        for view in range(num_views):
            clip = self._decode_latents(
                state.rgb_target_latents.narrow(2, view * frames_per_view, frames_per_view)
            ).clamp_(-1, 1)
            if decoded is None:
                decoded = torch.empty(
                    (clip.shape[0], clip.shape[1], num_views * clip.shape[2], *clip.shape[3:]),
                    dtype=clip.dtype,
                    device="cpu",
                )
            decoded.narrow(2, view * clip.shape[2], clip.shape[2]).copy_(clip)
            del clip
        assert decoded is not None
        payload: dict[str, Any] = {"video": decoded}
        if return_latents:
            # Preserve sampling dtype and camera-major layout for reference comparisons.
            payload["latents"] = {"vision_latent": state.rgb_target_latents.detach().cpu().contiguous()}
            if state.lidar_target_latents is not None:
                payload["latents"]["lidar_latent"] = state.lidar_target_latents.detach().cpu().contiguous()
        metadata: dict[str, Any] = {
            "multiview": {
                "cameras": list(state.cameras),
                "frames_per_view": state.num_frames,
                "fps": state.fps,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "width": state.width,
                "height": state.height,
            },
            "cosmos3_nano_sim_auto": {
                "chunks": len(state.schedule),
                "frames_per_chunk": self.manifest.frames_per_chunk,
                "frame_unit": state.frame_unit,
                "checkpoint_id": self.manifest.checkpoint_id,
            },
        }
        if state.joint and lidar_request is not None and as_bool(lidar_request.get("return_output"), False):
            assert self.lidar_decoder is not None and state.lidar_target_latents is not None
            lidar_output = self.lidar_decoder(state.lidar_target_latents)
            payload["lidar"] = lidar_output
            config = self.lidar_decoder.config
            metadata["lidar"] = {
                "fps": config["fps"],
                "num_frames": lidar_output.shape[2],
                "shape": list(lidar_output.shape),
                "dtype": "float32",
                "channels": ["range", "intensity", "validity"],
                "units": ["metres", "unit", "binary" if config["apply_validity_mask"] else "probability"],
                "validity_threshold": config["range_projection"].get("validity_threshold", 0.5),
                "apply_validity_mask": config["apply_validity_mask"],
                "start_time_seconds": 0.0,
                "range_projection": dict(config["range_projection"]),
            }
        return payload, metadata


__all__ = [
    "Cosmos3NanoSimAutoPipeline",
    "get_cosmos3_nano_sim_auto_ir_op_priority_func",
    "get_cosmos3_nano_sim_auto_post_process_func",
    "get_cosmos3_nano_sim_auto_pre_process_func",
]
