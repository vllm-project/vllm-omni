# SPDX-License-Identifier: Apache-2.0
"""Offline dense-oracle pipeline for Cosmos3-Nano-Sim-Transfer."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral
from typing import Any, ClassVar

import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3.action import find_closest_target_size
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    COSMOS3_DURATION_TEMPLATE,
    COSMOS3_RESOLUTION_TEMPLATE,
    COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE,
    COSMOS3_TRANSFER_SYSTEM_PROMPT,
    _format_json_object_prompt,
    _json_object_aspect_ratio,
    get_cosmos3_pre_process_func,
)
from vllm_omni.diffusion.models.cosmos3.transfer import (
    Cosmos3TransferHint,
    load_or_compute_control_frames,
    media_hw,
    normalized_video_to_uint8_cthw,
    parse_transfer_hint,
    uint8_cthw_to_normalized_5d,
)
from vllm_omni.diffusion.models.cosmos3.utils import VIDEO_RES_SIZE_INFO
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.control_contract import (
    TRANSFER_HINTS,
    TransferHint,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import (
    Cosmos3NanoSimBimanualGeometry,
    Cosmos3NanoSimBimanualResolutionPolicy,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualPipeline,
    _admission_float,
    _admission_int,
    _resolution_policy,
    get_cosmos3_nano_sim_bimanual_ir_op_priority_func,
    get_cosmos3_nano_sim_bimanual_post_process_func,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualSessionState,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_transfer import (
    Cosmos3NanoSimTransferTransformer,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.utils import iter_clean_commit_frames
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionRequestRejectedError


@dataclass(frozen=True, slots=True)
class _TransferRequestContract:
    hint: TransferHint
    hint_config: Mapping[str, Any]
    control_video: Any
    num_pixel_frames: int
    window_frames: int
    sink_frames: int
    emphasize_control_in_prompt: bool


@dataclass(frozen=True, slots=True)
class _TransferConditioning:
    request: _TransferRequestContract
    control_latents: torch.Tensor


def _prompt_value(prompt_data: Any, key: str) -> Any:
    if not isinstance(prompt_data, Mapping):
        return None
    if prompt_data.get(key) is not None:
        return prompt_data[key]
    additional = prompt_data.get("additional_information")
    if isinstance(additional, Mapping):
        return additional.get(key)
    return None


def _request_value(sampling_params: Any, prompt_data: Any, key: str, default: Any = None) -> Any:
    extra = getattr(sampling_params, "extra_args", None)
    if isinstance(extra, Mapping) and extra.get(key) is not None:
        return extra[key]
    value = getattr(sampling_params, key, None)
    if value is not None:
        return value
    value = _prompt_value(prompt_data, key)
    return default if value is None else value


def _transfer_media_source(sampling_params: Any, prompt_data: Any) -> Any:
    """Select the same bucket-driving source used by Transfer execution."""

    if isinstance(prompt_data, Mapping):
        additional = prompt_data.get("additional_information")
        if isinstance(additional, Mapping) and additional.get("preprocessed_transfer_video") is not None:
            return additional["preprocessed_transfer_video"]
        multi_modal = prompt_data.get("multi_modal_data")
        if isinstance(multi_modal, Mapping):
            for key in ("video", "image"):
                if multi_modal.get(key) is not None:
                    return multi_modal[key]

    control_video = _request_value(sampling_params, prompt_data, "control_video")
    if control_video is not None:
        return control_video
    for hint in TRANSFER_HINTS:
        raw_hint = _request_value(sampling_params, prompt_data, hint)
        if isinstance(raw_hint, Mapping):
            for key in ("control", "control_path"):
                if raw_hint.get(key) is not None:
                    return raw_hint[key]
    return None


def _transfer_media_hw(value: Any) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        height, width = value.get("height"), value.get("width")
        if height is not None and width is not None:
            return int(height), int(width)
        for key in ("video", "frames", "data", "image", "control", "control_path"):
            if value.get(key) is not None:
                resolved = _transfer_media_hw(value[key])
                if resolved is not None:
                    return resolved
        return None
    return media_hw(value)


def _default_transfer_resolution(policy: Cosmos3NanoSimBimanualResolutionPolicy) -> str:
    height, width = policy.default_resolution
    matching = [key for key, sizes in VIDEO_RES_SIZE_INFO.items() if (width, height) in sizes.values()]
    if not matching:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Transfer default_resolution must be a canonical Cosmos3 bucket, got {height}x{width}."
        )
    return max(matching, key=int)


def resolve_cosmos3_nano_sim_transfer_geometry(
    sampling_params: Any,
    prompt_data: Any,
    policy: Cosmos3NanoSimBimanualResolutionPolicy,
) -> Cosmos3NanoSimBimanualGeometry:
    """Snap the prioritized Transfer source to a policy-valid Cosmos3 bucket."""

    resolution = _request_value(
        sampling_params,
        prompt_data,
        "resolution",
        _request_value(
            sampling_params,
            prompt_data,
            "image_size",
            _default_transfer_resolution(policy),
        ),
    )
    try:
        aspect_ratio = _request_value(sampling_params, prompt_data, "aspect_ratio")
        if aspect_ratio is not None:
            target_width, target_height = VIDEO_RES_SIZE_INFO[str(resolution)][str(aspect_ratio)]
        else:
            source_hw = _transfer_media_hw(_transfer_media_source(sampling_params, prompt_data))
            if source_hw is None:
                source_hw = policy.default_resolution
            target_width, target_height = find_closest_target_size(*source_hw, resolution)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Cosmos3-Nano-Sim-Transfer invalid resolution/aspect_ratio: {resolution!r}/{aspect_ratio!r}."
        ) from exc
    geometry = policy.resolve(target_height, target_width)

    requested_height = getattr(sampling_params, "height", None)
    requested_width = getattr(sampling_params, "width", None)
    if (requested_height is None) != (requested_width is None):
        raise ValueError("Cosmos3-Nano-Sim-Transfer height and width must be supplied together.")
    if requested_height is not None and (int(requested_height), int(requested_width)) != geometry.session_key:
        raise ValueError(
            "Cosmos3-Nano-Sim-Transfer serialized dimensions do not match the control-selected bucket: "
            f"requested {requested_height}x{requested_width}, selected {geometry.height}x{geometry.width}."
        )
    return geometry


def _strict_integer(value: Any, name: str = "num_frames") -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ARDiffusionRequestRejectedError(
            f"Cosmos3-Nano-Sim-Transfer {name} must be an integer without coercion, got {value!r}."
        )
    return int(value)


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ARDiffusionRequestRejectedError(
            f"Cosmos3-Nano-Sim-Transfer {name} must be a JSON boolean, got {value!r}."
        )
    return value


def format_cosmos3_nano_sim_transfer_prompt(
    prompt: str,
    *,
    hint: str,
    num_frames: int,
    fps: float,
    height: int,
    width: int,
    emphasize_control_in_prompt: bool = True,
) -> str:
    """Replicate the reference non-AR full-clip Transfer prompt."""

    prompt_fps = int(round(fps))
    if prompt_fps <= 0:
        raise ValueError(f"Cosmos3-Nano-Sim-Transfer prompt FPS must round to a positive integer, got {fps}.")
    formatted = _format_json_object_prompt(
        prompt,
        num_frames=num_frames,
        frame_rate=prompt_fps,
        height=height,
        width=width,
        aspect_ratio=_json_object_aspect_ratio(prompt),
    )
    if formatted is None:
        formatted = prompt.strip()
        duration_text = COSMOS3_DURATION_TEMPLATE.format(duration=num_frames / prompt_fps, fps=prompt_fps)
        formatted = formatted.rstrip(".") + ". " + duration_text
        formatted = formatted.strip()
        resolution_text = COSMOS3_RESOLUTION_TEMPLATE.format(height=height, width=width)
        formatted = formatted.rstrip(".") + ". " + resolution_text
    if emphasize_control_in_prompt:
        suffix = COSMOS3_TRANSFER_CONTROL_DIRECTIVE_TEMPLATE.format(hint_names=hint)
        return f"{formatted.rstrip()} {suffix}"
    return formatted


def _trim_transfer_history(
    history: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    tokens_per_frame: int,
    window_frames: int,
    sink_frames: int,
) -> None:
    """Select paired sinks and recent past before the next control forward.

    W includes S sink frames and the current temporal frame. A past frame
    occupies two entries (control, generated latent). The current control is
    appended afterwards, so denoising and clean commit additionally see C_t.
    Positions in the retained K/V remain absolute; eviction never renumbers RoPE.
    """
    sink_tokens = 2 * sink_frames * tokens_per_frame
    recent_tokens = 2 * (window_frames - sink_frames - 1) * tokens_per_frame
    for layer_idx, (key, value) in enumerate(history):
        if key.shape[1] <= sink_tokens + recent_tokens:
            continue

        def select(tensor: torch.Tensor) -> torch.Tensor:
            prefix = tensor[:, :sink_tokens]
            return torch.cat((prefix, tensor[:, -recent_tokens:]), dim=1) if recent_tokens else prefix.clone()

        history[layer_idx] = (select(key), select(value))


def get_cosmos3_nano_sim_transfer_pre_process_func(od_config: OmniDiffusionConfig):
    """Resolve a Transfer bucket, preprocess to it, and serialize final H/W."""

    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import Cosmos3NanoSimBimanualManifest

    manifest = Cosmos3NanoSimBimanualManifest.from_od_config(od_config)
    manifest.require_control_video_conditioning()
    policy = _resolution_policy(od_config, manifest)

    def transfer_target_size(request) -> tuple[int, int]:
        geometry = resolve_cosmos3_nano_sim_transfer_geometry(
            request.sampling_params,
            request.prompt,
            policy,
        )
        return geometry.height, geometry.width

    cosmos3_pre_process = get_cosmos3_pre_process_func(
        od_config,
        transfer_target_size=transfer_target_size,
    )

    def pre_process_func(request):
        sp = request.sampling_params
        extra = sp.extra_args
        if extra is None:
            extra = {}
            sp.extra_args = extra
        if not isinstance(extra, dict):
            raise ValueError("Cosmos3-Nano-Sim-Transfer extra_args must be a mutable mapping during preprocessing.")
        prompt_data = request.prompt
        geometry = resolve_cosmos3_nano_sim_transfer_geometry(sp, prompt_data, policy)
        sp.height, sp.width = geometry.height, geometry.width

        def request_param(key: str) -> Any:
            if extra.get(key) is not None:
                return extra[key]
            value = getattr(sp, key, None)
            if value is not None:
                return value
            return _prompt_value(prompt_data, key)

        generic_hint = request_param("control_hint")
        named_hints = [hint for hint in (*TRANSFER_HINTS, "wsm") if request_param(hint) is not None]
        injected_hint: str | None = None
        injected_hint_was_present = False
        injected_hint_previous: Any = None
        if generic_hint is not None and not named_hints:
            normalized_hint = str(generic_hint).strip().lower()
            if normalized_hint in TRANSFER_HINTS:
                injected_hint = normalized_hint
                injected_hint_was_present = injected_hint in extra
                injected_hint_previous = extra.get(injected_hint)
                # Cosmos3 preprocessing detects named hints. Temporarily expose
                # the generic contract so a vision video enters the Transfer slot.
                extra[injected_hint] = True
        try:
            processed = cosmos3_pre_process(request)
        finally:
            if injected_hint is not None:
                if injected_hint_was_present:
                    extra[injected_hint] = injected_hint_previous
                else:
                    extra.pop(injected_hint, None)
        final_sp = processed.sampling_params
        final_geometry = resolve_cosmos3_nano_sim_transfer_geometry(final_sp, processed.prompt, policy)
        final_sp.height, final_sp.width = final_geometry.height, final_geometry.width
        return processed

    return pre_process_func


def get_cosmos3_nano_sim_transfer_post_process_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_nano_sim_bimanual_post_process_func(od_config)


def get_cosmos3_nano_sim_transfer_ir_op_priority_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_nano_sim_bimanual_ir_op_priority_func(od_config)


class Cosmos3NanoSimTransferPipeline(Cosmos3NanoSimBimanualPipeline):
    """Transfer inference with dense full history or paired sliding history."""

    _transformer_cls_override: ClassVar[type[Cosmos3NanoSimTransferTransformer]] = Cosmos3NanoSimTransferTransformer

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        if not isinstance(self.transformer, Cosmos3NanoSimTransferTransformer):
            raise TypeError(
                "Cosmos3-Nano-Sim-Transfer pipeline resolved the wrong transformer type: "
                f"{type(self.transformer).__name__}."
            )

    def _init_conditioning(self, od_config: OmniDiffusionConfig) -> None:
        contract = self.manifest.require_control_video_conditioning()
        if self.manifest.chunk_size != 1 and (self.manifest.sink_frames != 0 or not contract.no_eviction):
            raise ValueError("Cosmos3-Nano-Sim-Transfer sliding history requires chunk_size=1.")
        if not bool(getattr(od_config, "enforce_eager", False)):
            raise ValueError(
                "Cosmos3-Nano-Sim-Transfer requires enforce_eager=True; compiled execution is unsupported."
            )
        kv_cache_dtype = getattr(od_config, "diffusion_kv_cache_dtype", None)
        if kv_cache_dtype not in (None, "auto"):
            raise ValueError("Cosmos3-Nano-Sim-Transfer does not support a quantized diffusion KV cache.")

    @staticmethod
    def _is_action_weight(name: str) -> bool:
        key = name.removeprefix("transformer.").removeprefix("model.")
        return key.startswith(
            ("action2llm.", "llm2action.", "action_proj_in.", "action_proj_out.", "action_pos_embed.")
        ) or key in {
            "action_modality_embed",
            "action_modality_embed.weight",
        }

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Require an action-free Transfer checkpoint inventory."""

        action_weights: list[str] = []

        def checked_weights():
            for name, tensor in weights:
                if self._is_action_weight(name):
                    action_weights.append(name)
                    continue
                yield name, tensor

        loaded = super().load_weights(checked_weights())
        if action_weights:
            preview = ", ".join(sorted(action_weights)[:12])
            raise ValueError(f"Cosmos3-Nano-Sim-Transfer checkpoint contains forbidden action weights: {preview}.")
        return loaded

    def ar_diffusion_kv_cache_spec(self):
        raise NotImplementedError(
            "Cosmos3-Nano-Sim-Transfer paged-cache topology is Phase T2; use the default dense diffusion engine."
        )

    def _parse_tick(self, tick):
        del tick
        raise ValueError("Cosmos3-Nano-Sim-Transfer tick transport is Phase T3 and is not supported by this pipeline.")

    def _request_param(self, sp: Any, prompt_data: Any, key: str, default: Any = None) -> Any:
        value = self._get_sp_param(sp, key, None)
        if value is None:
            value = _prompt_value(prompt_data, key)
        return default if value is None else value

    def _resolve_request_fps(self, sp: Any, prompt_data: Any) -> float:
        input_fps = _prompt_value(prompt_data, "transfer_input_fps")
        if input_fps is not None:
            try:
                resolved_input_fps = float(input_fps)
            except (TypeError, ValueError, OverflowError):
                resolved_input_fps = 0.0
            if math.isfinite(resolved_input_fps) and resolved_input_fps > 0:
                return resolved_input_fps
        return super()._resolve_request_fps(sp, prompt_data)

    def _resolve_request_geometry(self, sp: Any, prompt_data: Any) -> Cosmos3NanoSimBimanualGeometry:
        return resolve_cosmos3_nano_sim_transfer_geometry(sp, prompt_data, self.resolution_policy)

    def _resolve_requested_pixel_frames(
        self,
        sp: Any,
        prompt_data: Any,
        conditioning_request: _TransferRequestContract,
    ) -> int:
        del sp, prompt_data
        return conditioning_request.num_pixel_frames

    def _prompt_token_limit(self, sp: Any, prompt_data: Any) -> int:
        # Dense text K/V is sized to the prompt, unlike the fixed paged pool.
        limit = _strict_integer(
            self._request_param(sp, prompt_data, "max_prompt_tokens", self.manifest.text_cache_max_len),
            "max_prompt_tokens",
        )
        if limit <= 0:
            raise ARDiffusionRequestRejectedError("max_prompt_tokens must be positive.")
        return limit

    def _append_dense_kv(
        self,
        state: Cosmos3NanoSimBimanualSessionState,
        current_kv: list[tuple[torch.Tensor, torch.Tensor]],
        geometry: Cosmos3NanoSimBimanualGeometry,
    ) -> None:
        # Transfer trims complete pairs before C_t, not individual entries at
        # each append. Keep C_t for all denoise steps and the clean commit.
        history = state.dense_kv_by_branch.get(self._MAIN_BRANCH)
        if history is None:
            state.dense_kv_by_branch[self._MAIN_BRANCH] = [(k.detach(), v.detach()) for k, v in current_kv]
        else:
            for idx, ((old_k, old_v), (new_k, new_v)) in enumerate(zip(history, current_kv, strict=True)):
                history[idx] = (torch.cat((old_k, new_k), dim=1), torch.cat((old_v, new_v), dim=1))

    def _validate_conditioning_request(
        self,
        sp,
        typed_inputs: Any | None,
        *,
        prompt_data: Any = None,
    ) -> _TransferRequestContract:
        if typed_inputs is not None or bool(self._get_sp_param(sp, "chunk_only", False)):
            raise ARDiffusionRequestRejectedError(
                "Cosmos3-Nano-Sim-Transfer T1 supports only offline full-clip requests; tick transport is Phase T3."
            )
        for action_key in ("action", "domain_id", "domain_name", "embodiment"):
            if self._request_param(sp, prompt_data, action_key, None) is not None:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos3-Nano-Sim-Transfer does not accept action conditioning field {action_key!r}."
                )

        named_hints = []
        hint_values: dict[str, Any] = {}
        for hint in (*TRANSFER_HINTS, "wsm"):
            value = self._request_param(sp, prompt_data, hint, None)
            if value is not None:
                named_hints.append(hint)
                hint_values[hint] = value
        generic_hint = self._request_param(sp, prompt_data, "control_hint", None)
        control_video = self._request_param(sp, prompt_data, "control_video", None)
        if generic_hint is not None and named_hints:
            raise ARDiffusionRequestRejectedError(
                "Cosmos3-Nano-Sim-Transfer accepts either control_hint/control_video or one named hint, not both."
            )
        if generic_hint is not None:
            hint = str(generic_hint).strip().lower()
            named_hints = [hint]
            hint_values[hint] = {}
        if len(named_hints) != 1:
            raise ARDiffusionRequestRejectedError(
                "Cosmos3-Nano-Sim-Transfer requires exactly one edge, blur, depth, or seg control hint."
            )
        hint = named_hints[0]
        if hint not in TRANSFER_HINTS:
            raise ARDiffusionRequestRejectedError(
                f"Unsupported Cosmos3-Nano-Sim-Transfer control hint {hint!r}; expected one of {list(TRANSFER_HINTS)}."
            )

        try:
            parsed_hint = parse_transfer_hint(hint, hint_values[hint])
        except (TypeError, ValueError) as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if control_video is not None and (parsed_hint.control is not None or parsed_hint.control_path is not None):
            raise ARDiffusionRequestRejectedError(
                "Cosmos3-Nano-Sim-Transfer control_video cannot be combined with a named control/control_path."
            )
        if parsed_hint.control_weight != 1.0:
            raise ARDiffusionRequestRejectedError("Cosmos3-Nano-Sim-Transfer single-control weight must equal 1.0.")
        hint_config: dict[str, Any] = {"control_weight": parsed_hint.control_weight}
        if parsed_hint.control_path is not None:
            hint_config["control_path"] = parsed_hint.control_path
        if parsed_hint.control is not None:
            hint_config["control"] = parsed_hint.control
        if hint == "edge":
            hint_config["preset_edge_threshold"] = parsed_hint.preset_edge_threshold
        elif hint == "blur":
            hint_config["preset_blur_strength"] = parsed_hint.preset_blur_strength

        control_guidance = _admission_float(
            self._request_param(sp, prompt_data, "control_guidance", 1.0),
            "control_guidance",
        )
        if control_guidance != 1.0:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos3-Nano-Sim-Transfer distilled inference requires control_guidance=1.0, got {control_guidance}."
            )
        first_conditional = _admission_int(
            self._request_param(sp, prompt_data, "num_first_chunk_conditional_frames", 0),
            "num_first_chunk_conditional_frames",
        )
        if first_conditional != 0:
            raise ARDiffusionRequestRejectedError(
                "Cosmos3-Nano-Sim-Transfer requires num_first_chunk_conditional_frames=0."
            )
        share_positions = _strict_bool(
            self._request_param(sp, prompt_data, "share_vision_temporal_positions", True),
            "share_vision_temporal_positions",
        )
        if not share_positions:
            raise ARDiffusionRequestRejectedError(
                "Cosmos3-Nano-Sim-Transfer requires share_vision_temporal_positions=True."
            )
        emphasize = _strict_bool(
            self._request_param(
                sp,
                prompt_data,
                "emphasize_control_in_prompt",
                self.manifest.require_control_video_conditioning().emphasize_control_in_prompt,
            ),
            "emphasize_control_in_prompt",
        )

        window = _strict_integer(
            self._request_param(sp, prompt_data, "kv_cache_inference_size", self.manifest.window_frames),
            "kv_cache_inference_size",
        )
        sink = _strict_integer(
            self._request_param(sp, prompt_data, "attention_sink_size", self.manifest.sink_frames),
            "attention_sink_size",
        )
        if window <= 0 or not 0 <= sink < window:
            raise ARDiffusionRequestRejectedError("Require kv_cache_inference_size > attention_sink_size >= 0.")
        if self.manifest.chunk_size != 1 and sink != 0:
            raise ARDiffusionRequestRejectedError("Transfer sink history requires chunk_size=1.")

        num_pixel_frames = _strict_integer(self._request_param(sp, prompt_data, "num_frames", 1))
        frame_stride = self.manifest.temporal_compression_factor * self.manifest.chunk_size
        minimum_frames = 1 if self.manifest.chunk_size == 1 else 1 + frame_stride
        if num_pixel_frames < minimum_frames or (num_pixel_frames - 1) % frame_stride != 0:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos3-Nano-Sim-Transfer requires F >= {minimum_frames} and (F - 1) % {frame_stride} == 0 "
                f"pixel frames; got F={num_pixel_frames}."
            )
        latent_frames = (num_pixel_frames - 1) // self.manifest.temporal_compression_factor + 1
        required_history_frames = 2 * latent_frames + 1
        # Framewise inference uses window/sink retention regardless of
        # conditioning.no_eviction. Chunkwise inference requires full-history capacity.
        if self.manifest.chunk_size != 1 and required_history_frames > window:
            raise ARDiffusionRequestRejectedError(
                "Cosmos3-Nano-Sim-Transfer full history exceeds the artifact's no-eviction window: "
                f"required {required_history_frames}, configured {window}."
            )
        return _TransferRequestContract(
            hint=hint,
            hint_config=hint_config,
            control_video=control_video,
            num_pixel_frames=num_pixel_frames,
            window_frames=window,
            sink_frames=sink,
            emphasize_control_in_prompt=emphasize,
        )

    def _conditioning_fingerprint(self, request: _TransferRequestContract) -> tuple[tuple[str, Any], ...]:
        return (
            ("control_hint", request.hint),
            ("clip_num_frames", request.num_pixel_frames),
            ("control_contract_sha256", self.manifest.conditioning_digest),
            ("window_frames", request.window_frames),
            ("sink_frames", request.sink_frames),
            ("emphasize_control_in_prompt", request.emphasize_control_in_prompt),
        )

    def _build_prompt_tokens(
        self,
        prompt: str,
        *,
        sampling_params: Any,
        prompt_data: Any,
        geometry: Cosmos3NanoSimBimanualGeometry,
        fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        request = self._validate_conditioning_request(
            sampling_params,
            None,
            prompt_data=prompt_data,
        )
        formatted = format_cosmos3_nano_sim_transfer_prompt(
            prompt,
            hint=request.hint,
            num_frames=request.num_pixel_frames,
            fps=fps,
            height=geometry.height,
            width=geometry.width,
            emphasize_control_in_prompt=request.emphasize_control_in_prompt,
        )
        return self._tokenize_prompt(
            formatted,
            max_sequence_length=1 << 30,
            use_system_prompt=True,
            system_prompt=COSMOS3_TRANSFER_SYSTEM_PROMPT,
        )

    def _prepare_conditioning(
        self,
        sp,
        *,
        typed_inputs: Any | None,
        request: _TransferRequestContract,
        geometry: Cosmos3NanoSimBimanualGeometry,
        start_frame: int,
        target_frame: int,
        prompt_data: Any = None,
    ) -> _TransferConditioning:
        if typed_inputs is not None or start_frame != 0:
            raise ValueError("Cosmos3-Nano-Sim-Transfer T1 requires a fresh offline full-clip session.")
        additional = prompt_data.get("additional_information", {}) if isinstance(prompt_data, Mapping) else {}
        source_video = additional.get("preprocessed_transfer_video") if isinstance(additional, Mapping) else None
        input_frames = normalized_video_to_uint8_cthw(source_video) if source_video is not None else None

        control_value = request.control_video
        if control_value is None:
            control_value = request.hint_config.get("control")
        control_path = request.hint_config.get("control_path")
        hint = Cosmos3TransferHint(
            key=request.hint,
            control_path=str(control_path) if control_path is not None else None,
            control=control_value,
            control_weight=1.0,
            preset_edge_threshold=str(request.hint_config.get("preset_edge_threshold", "medium")).lower(),
            preset_blur_strength=str(request.hint_config.get("preset_blur_strength", "medium")).lower(),
        )
        control_frames = load_or_compute_control_frames(
            hint,
            height=geometry.height,
            width=geometry.width,
            max_frames=request.num_pixel_frames,
            input_frames=input_frames,
        )
        if control_frames.shape[1] != request.num_pixel_frames:
            raise ValueError(
                "Cosmos3-Nano-Sim-Transfer control video must cover the complete requested clip: "
                f"expected {request.num_pixel_frames} frames, got {control_frames.shape[1]}."
            )
        control_video = uint8_cthw_to_normalized_5d(control_frames, dtype=torch.float32)
        control_latents = self._encode_video_tensor(control_video)
        expected = (
            1,
            self.transformer.latent_channel_size,
            target_frame,
            geometry.latent_height,
            geometry.latent_width,
        )
        if tuple(control_latents.shape) != expected:
            raise ValueError(
                "Cosmos3-Nano-Sim-Transfer control/target latent shape mismatch: "
                f"expected {expected}, got {tuple(control_latents.shape)}."
            )
        return _TransferConditioning(request=request, control_latents=control_latents)

    def _initial_condition_latent(self, prompt_data: Any, sp, geometry: Cosmos3NanoSimBimanualGeometry) -> None:
        del geometry
        if (
            self._get_sp_param(sp, "initial_latent", None) is not None
            or _prompt_value(prompt_data, "initial_latent") is not None
        ):
            raise ValueError("Cosmos3-Nano-Sim-Transfer does not accept initial_latent or seed images.")
        if isinstance(prompt_data, Mapping):
            additional = prompt_data.get("additional_information", {}) or {}
            multi_modal = prompt_data.get("multi_modal_data", {}) or {}
            if (
                prompt_data.get("seed_image") is not None
                or (isinstance(additional, Mapping) and additional.get("preprocessed_image") is not None)
                or (isinstance(multi_modal, Mapping) and multi_modal.get("image") is not None)
            ):
                raise ValueError("Cosmos3-Nano-Sim-Transfer does not accept initial_latent or seed images.")
        return None

    def _prefill_first_frame(
        self, state: Cosmos3NanoSimBimanualSessionState, initial_latent: torch.Tensor | None, **kwargs
    ):
        del state, kwargs
        if initial_latent is not None:
            raise ValueError("Cosmos3-Nano-Sim-Transfer does not accept an RGB prefix latent.")
        return None

    def _run_chunk(
        self,
        state: Cosmos3NanoSimBimanualSessionState,
        *,
        geometry: Cosmos3NanoSimBimanualGeometry,
        chunk_start: int,
        chunk_end: int,
        target_frame: int,
        terminal_request: bool,
        request_start_frame: int,
        seed: int,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        fps: float,
        conditioning: _TransferConditioning,
        tick_durations: dict[str, float],
        measure_tick_latency: bool,
    ) -> torch.Tensor:
        del request_start_frame
        if self.manifest.chunk_size == 1:
            history = state.dense_kv_by_branch.get(self._MAIN_BRANCH)
            if history is not None:
                _trim_transfer_history(
                    history,
                    tokens_per_frame=geometry.vision_tokens_per_frame,
                    window_frames=conditioning.request.window_frames,
                    sink_frames=conditioning.request.sink_frames,
                )
        control_chunk = conditioning.control_latents[:, :, chunk_start:chunk_end]
        with self._timed_tick_stage(
            tick_durations,
            "control_cache_commit_s",
            enabled=measure_tick_latency,
        ):
            self._transformer_forward(
                state,
                control_chunk.to(self.dtype),
                torch.zeros(1, device=self.device, dtype=torch.float32),
                geometry=geometry,
                text_kv=text_kv,
                real_text_kv_len=real_text_kv_len,
                frame_start=chunk_start,
                fps=fps,
                conditioning_kwargs={},
                condition_vision=True,
                commit_current=True,
            )

        clean_chunk = self._denoise_chunk(
            state,
            geometry=geometry,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            seed=seed,
            text_kv=text_kv,
            real_text_kv_len=real_text_kv_len,
            fps=fps,
            conditioning_kwargs={},
            tick_durations=tick_durations,
            measure_tick_latency=measure_tick_latency,
        )
        with self._timed_tick_stage(
            tick_durations,
            "clean_cache_commit_s",
            enabled=measure_tick_latency,
        ):
            for local_idx, frame_idx in iter_clean_commit_frames(
                chunk_start,
                chunk_end,
                target_frame=target_frame,
                terminal_request=terminal_request,
            ):
                self._commit_clean_frame(
                    state,
                    clean_chunk[:, :, local_idx : local_idx + 1],
                    geometry=geometry,
                    frame_idx=frame_idx,
                    text_kv=text_kv,
                    real_text_kv_len=real_text_kv_len,
                    fps=fps,
                    conditioning_kwargs={},
                )
        return clean_chunk


__all__ = [
    "Cosmos3NanoSimTransferPipeline",
    "format_cosmos3_nano_sim_transfer_prompt",
    "get_cosmos3_nano_sim_transfer_ir_op_priority_func",
    "get_cosmos3_nano_sim_transfer_post_process_func",
    "get_cosmos3_nano_sim_transfer_pre_process_func",
    "resolve_cosmos3_nano_sim_transfer_geometry",
]
