# SPDX-License-Identifier: Apache-2.0
"""Causal autoregressive pipeline for Cosmos-Dreams checkpoints."""

from __future__ import annotations

import logging
import math
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar

import torch

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3.action import (
    load_action_tensor,
    pad_action_to_dim,
)
from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
    Cosmos3OmniDiffusersPipeline,
    get_cosmos3_ir_op_priority_func,
    get_cosmos3_post_process_func,
    get_cosmos3_pre_process_func,
)
from vllm_omni.diffusion.models.cosmos_dreams.config import CosmosDreamsManifest
from vllm_omni.diffusion.models.cosmos_dreams.normalizer import ActionAffineNormalizer
from vllm_omni.diffusion.models.cosmos_dreams.sampler import CosmosDreamsDistilledSampler
from vllm_omni.diffusion.models.cosmos_dreams.state_cosmos_dreams import (
    CosmosDreamsSessionFingerprint,
    CosmosDreamsSessionState,
    append_dense_kv_history,
)
from vllm_omni.diffusion.models.cosmos_dreams.streaming_vae import decode_wan_causal_chunk
from vllm_omni.diffusion.models.cosmos_dreams.tick_adapter import parse_cosmos_dreams_tick
from vllm_omni.diffusion.models.cosmos_dreams.transformer_cosmos_dreams import (
    CosmosDreamsTransformer,
    CosmosDreamsTransformerOutput,
)
from vllm_omni.diffusion.models.cosmos_dreams.utils import (
    estimate_kv_memory_bytes,
    iter_ar_chunk_ranges,
    iter_clean_commit_frames,
    prompt_token_hash,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
    ARDiffusionRequestRejectedError,
)
from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionTickRequest,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _ActionRequestContract:
    domain_id: int
    embodiment: str


@dataclass(frozen=True, slots=True)
class _ActionConditioning:
    request: _ActionRequestContract
    raw_action: torch.Tensor | None
    layout: str | None
    domain_ids: torch.Tensor


def _nested_config_value(config: Any, key: str, default: Any = None) -> Any:
    """Read one deploy/checkpoint option from the known config envelopes."""

    roots = [
        getattr(config, "custom_pipeline_args", None),
        getattr(config, "model_config", None),
        getattr(config, "tf_model_config", None),
    ]
    nested_names = (
        "cosmos_dreams",
        "causal_manifest",
        "interactive_config",
        "diffusion_expert_config",
    )
    for root in roots:
        if not isinstance(root, dict):
            to_dict = getattr(root, "to_dict", None)
            root = to_dict() if callable(to_dict) else getattr(root, "params", None)
        if not isinstance(root, dict):
            continue
        for nested_name in nested_names:
            nested = root.get(nested_name)
            if isinstance(nested, dict) and nested.get(key) is not None:
                return nested[key]
        if root.get(key) is not None:
            return root[key]
    return default


def _first_not_none(*values: Any) -> Any:
    """Return the first explicitly supplied value, preserving falsey inputs."""

    return next((value for value in values if value is not None), None)


def _admission_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams {name} must be an integer, got {value!r}.") from exc


def _admission_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams {name} must be numeric, got {value!r}.") from exc


def get_cosmos_dreams_pre_process_func(od_config: OmniDiffusionConfig):
    """Use Cosmos3 media preprocessing with the deployment-fixed resolution."""

    manifest = CosmosDreamsManifest.from_od_config(od_config, require_explicit=True)
    cosmos3_pre_process = get_cosmos3_pre_process_func(od_config)

    def pre_process_func(request):
        sp = request.sampling_params
        if sp.height is not None and int(sp.height) != manifest.height:
            raise ValueError(
                "Cosmos-Dreams resolution is fixed per deployment: "
                f"requested height={sp.height}, configured height={manifest.height}."
            )
        if sp.width is not None and int(sp.width) != manifest.width:
            raise ValueError(
                "Cosmos-Dreams resolution is fixed per deployment: "
                f"requested width={sp.width}, configured width={manifest.width}."
            )
        sp.height = manifest.height
        sp.width = manifest.width
        return cosmos3_pre_process(request)

    return pre_process_func


def get_cosmos_dreams_post_process_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_post_process_func(od_config)


def get_cosmos_dreams_ir_op_priority_func(od_config: OmniDiffusionConfig):
    return get_cosmos3_ir_op_priority_func(od_config)


class CosmosDreamsPipeline(Cosmos3OmniDiffusersPipeline):
    """Cosmos3-Interactive inference with dense-or-paged persistent GEN K/V.

    The default diffusion engine exercises the dense numerical-oracle path.
    When the AR-Diffusion runner binds a state, the exact same attention uses
    paged storage and gathers one layer of immutable history at a time.
    """

    # The engine's generic warmup request is 512x512 with a one-step sampler,
    # while Cosmos-Dreams has artifact-fixed geometry and a four-step sampler.
    # Skip that incompatible request; AR-Diffusion owns any model-valid rollout
    # warmup when CUDA graphs are enabled.
    dummy_run_num_frames: ClassVar[int] = 0
    _transformer_cls_override: ClassVar[type[CosmosDreamsTransformer]] = CosmosDreamsTransformer
    _MAIN_BRANCH = "main"
    _SESSION_CAPACITY = 1
    _ar_diffusion_kv_state = None
    _bound_session_id: str | None = None

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.manifest = self._load_manifest(od_config)
        self.manifest.require_exported_artifact()
        if not isinstance(self.transformer, CosmosDreamsTransformer):
            raise TypeError(
                f"Cosmos-Dreams pipeline resolved the wrong transformer type: {type(self.transformer).__name__}."
            )
        if not self.is_distilled_model:
            raise ValueError("Cosmos-Dreams requires a distilled fixed-step checkpoint.")
        if len(self.manifest.t_list) != 4:
            raise ValueError(
                f"Cosmos-Dreams requires exactly four distilled denoise steps, got {len(self.manifest.t_list)}."
            )
        if od_config.parallel_config.sequence_parallel_size > 1:
            raise ValueError(
                "Cosmos-Dreams supports tensor parallelism but not sequence parallelism; "
                f"got sequence_parallel_size={od_config.parallel_config.sequence_parallel_size}."
            )
        self.distilled_sampler = CosmosDreamsDistilledSampler(
            self.manifest.t_list,
            sample_type=self.manifest.sample_type,
            num_train_timesteps=self.manifest.num_train_timesteps,
        )
        self._init_conditioning(od_config)
        self.default_fps = float(_nested_config_value(od_config, "default_fps", 15.0))
        if not math.isfinite(self.default_fps) or self.default_fps <= 0:
            raise ValueError(f"Cosmos-Dreams default_fps must be positive, got {self.default_fps}.")
        self.checkpoint_id = (
            self.manifest.checkpoint_id if self.manifest.checkpoint_id != "unknown" else str(od_config.model)
        )
        self._states: OrderedDict[str, CosmosDreamsSessionState] = OrderedDict()

        estimate = estimate_kv_memory_bytes(
            self.manifest,
            num_layers=self.transformer.num_hidden_layers,
            num_kv_heads=self.transformer.num_kv_heads_local,
            head_size=self.transformer.head_dim,
            dtype=self.dtype,
            session_capacity=self._SESSION_CAPACITY,
        )
        logger.info(
            "Cosmos-Dreams KV floor estimate: %.2f GiB (managed %.2f, scratch %.2f, text %.2f)",
            estimate.total_bytes / 2**30,
            estimate.self_attention_bytes / 2**30,
            estimate.scratch_bytes / 2**30,
            estimate.cross_attention_bytes / 2**30,
        )

    def _load_manifest(self, od_config: OmniDiffusionConfig) -> CosmosDreamsManifest:
        return CosmosDreamsManifest.from_od_config(od_config, require_explicit=True)

    def _init_conditioning(self, od_config: OmniDiffusionConfig) -> None:
        """Initialize the Humanoid action adapter from the v2 payload."""

        action_schema = self.manifest.action_schema
        if action_schema is None:
            raise ValueError("Cosmos-Dreams requires a validated v2 action_schema.")
        action_schema.validate_temporal_compression_factor(self.manifest.temporal_compression_factor)
        self.action_normalizers = {
            embodiment: ActionAffineNormalizer.from_contract(contract)
            for embodiment, contract in action_schema.normalizers.items()
        }
        configured_domain_id = _nested_config_value(od_config, "default_domain_id")
        artifact_domain_id = action_schema.embodiment_to_domain[action_schema.default_embodiment]
        self.default_domain_id = int(artifact_domain_id if configured_domain_id is None else configured_domain_id)
        self.default_embodiment = action_schema.resolve_embodiment(
            action_schema.default_embodiment,
            self.default_domain_id,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Reject artifact tensors that do not map into this exact model."""

        allowed = set(self.state_dict())
        tp_aware = {name for name, parameter in self.named_parameters() if hasattr(parameter, "weight_loader")}
        unexpected: list[str] = []

        def is_export_only_tensor(name: str) -> bool:
            key = name.removeprefix("transformer.").removeprefix("model.")
            return key.startswith(("lm_head.", "action_pos_embed."))

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
            raise ValueError(f"Cosmos-Dreams checkpoint contains unexpected transformer tensors: {preview}{suffix}")
        return loaded

    # -- AR-Diffusion pipeline capability ---------------------------------

    def _validate_ar_diffusion_deploy_overrides(self) -> None:
        od_config = getattr(self, "od_config", None)
        raw = getattr(od_config, "ar_diffusion_kv_config", None)
        if raw is None:
            model_config = getattr(od_config, "model_config", None)
            if isinstance(model_config, dict):
                raw = model_config.get("ar_diffusion_kv_config")
        if raw is None:
            return

        def configured(name: str, default: Any) -> Any:
            if isinstance(raw, dict):
                return raw.get(name, default)
            return getattr(raw, name, default)

        actual_window = configured("window_chunks", None) or self.manifest.window_frames
        actual_sink = configured("sink_chunks", 0) or self.manifest.sink_frames
        actual_reset = bool(configured("reset_at_boundary", False))
        mismatches = []
        if int(actual_window) != self.manifest.window_frames:
            mismatches.append(f"window_chunks={actual_window} (manifest={self.manifest.window_frames})")
        if int(actual_sink) != self.manifest.sink_frames:
            mismatches.append(f"sink_chunks={actual_sink} (manifest={self.manifest.sink_frames})")
        if actual_reset:
            mismatches.append("reset_at_boundary=True (manifest=False)")
        if mismatches:
            raise ValueError(
                "Cosmos-Dreams AR-Diffusion window geometry is fixed by the model manifest; "
                "deployment overrides would diverge from the dense oracle: " + ", ".join(mismatches)
            )

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        self._validate_ar_diffusion_deploy_overrides()
        return ARDiffusionKVCacheSpec(
            num_layers=self.transformer.num_hidden_layers,
            num_kv_heads=self.transformer.num_kv_heads_local,
            head_size=self.transformer.head_dim,
            tokens_per_frame=self.manifest.tokens_per_frame,
            frames_per_block=1,
            window_frames=self.manifest.window_frames,
            sink_frames=self.manifest.sink_frames,
            kv_branches=(ARDiffusionKVBranchSpec(self._MAIN_BRANCH, 0),),
            session_capacity=self._SESSION_CAPACITY,
            cross_attention=(ARDiffusionCrossAttentionKVSpec("text", self.manifest.text_cache_max_len),),
            max_scratch_tokens_per_branch=0,
        )

    def _validate_bound_kv_geometry(self, state) -> None:
        """Reject engine overrides that would diverge from the dense oracle.

        ``window_frames`` and ``sink_frames`` are checkpoint-manifest semantics
        for Cosmos-Dreams, not performance-only engine knobs. The generic AR
        runner permits deployment overrides for other models, so validate the
        resolved cache here before it can be bound to a model session.
        """

        cache = state.kv_cache
        actual = {
            "num_layers": int(cache.num_layers),
            "num_kv_heads": int(cache.num_kv_heads),
            "head_size": int(cache.head_size),
            "tokens_per_frame": int(cache.block_size),
            "window_frames": int(cache.spec.window_chunks),
            "sink_frames": int(cache.spec.sink_chunks),
            "reset_at_boundary": bool(cache.spec.reset_at_boundary),
            "text_cache_max_len": int(cache.cross_attention_lengths.get("text", -1)),
        }
        expected = {
            "num_layers": int(self.transformer.num_hidden_layers),
            "num_kv_heads": int(self.transformer.num_kv_heads_local),
            "head_size": int(self.transformer.head_dim),
            "tokens_per_frame": int(self.manifest.tokens_per_frame),
            "window_frames": int(self.manifest.window_frames),
            "sink_frames": int(self.manifest.sink_frames),
            "reset_at_boundary": False,
            "text_cache_max_len": int(self.manifest.text_cache_max_len),
        }
        mismatches = {name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]}
        if mismatches:
            details = ", ".join(
                f"{name}=expected {expected_value}, got {actual_value}"
                for name, (expected_value, actual_value) in mismatches.items()
            )
            raise RuntimeError(
                "Cosmos-Dreams AR-Diffusion KV geometry differs from the immutable "
                f"model manifest ({details}). Fix the deployment configuration and restart."
            )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id, state):
        if self._ar_diffusion_kv_state is not None:
            raise RuntimeError("Cosmos-Dreams AR-Diffusion state is already bound.")
        if state.session_id != session_id:
            raise ValueError(f"Cosmos-Dreams bound session mismatch: {state.session_id!r} != {session_id!r}.")
        self._validate_bound_kv_geometry(state)
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
            if state.vae_decoder_initialized:
                self.vae.clear_cache()
            state.reset()

    def _sync_for_tick_timing(self, enabled: bool) -> None:
        if not enabled or self.device.type != "cuda":
            return
        torch.accelerator.synchronize(self.device)

    @contextmanager
    def _timed_tick_stage(
        self,
        durations: dict[str, float],
        name: str,
        *,
        enabled: bool,
    ):
        if not enabled:
            yield
            return
        self._sync_for_tick_timing(True)
        started = time.perf_counter()
        try:
            yield
        finally:
            self._sync_for_tick_timing(True)
            durations[name] = durations.get(name, 0.0) + (time.perf_counter() - started)

    @staticmethod
    def _validate_session_mode(
        *,
        tick: bool,
        state_was_new: bool,
        session_id: str,
        reset: bool,
        close_session: bool,
    ) -> None:
        """Prevent accidental history reuse by ordinary offline requests."""

        if not tick and state_was_new and not reset and not close_session:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams full rollout session {session_id!r} requires reset=True at start "
                "or close_session=True at end."
            )

    # -- Session and conditioning -----------------------------------------

    def _parse_tick(self, tick: ARDiffusionTickRequest):
        """Parse the Humanoid-owned typed action track."""

        return parse_cosmos_dreams_tick(tick)

    def _resolve_request_fps(self, sp: Any, prompt_data: Any) -> float:
        """Resolve the request FPS used by prompts, mRoPE, and fingerprints."""

        del prompt_data
        return _admission_float(
            _first_not_none(
                self._get_sp_param(sp, "resolved_frame_rate", None),
                self._get_sp_param(sp, "frame_rate", None),
                self._get_sp_param(sp, "fps", None),
                self.default_fps,
            ),
            "FPS",
        )

    def _resolve_requested_pixel_frames(
        self,
        sp: Any,
        prompt_data: Any,
        conditioning_request: Any,
    ) -> int:
        """Return the authoritative pixel-frame count for a full rollout."""

        del prompt_data, conditioning_request
        return _admission_int(_first_not_none(sp.num_frames, 1), "num_frames")

    def _validate_conditioning_request(
        self,
        sp,
        typed_inputs: Any | None,
        *,
        prompt_data: Any = None,
    ) -> _ActionRequestContract:
        """Resolve Humanoid domain and embodiment without mutating state."""

        del prompt_data
        domain_name = (
            typed_inputs.domain_name if typed_inputs is not None else self._get_sp_param(sp, "domain_name", None)
        )
        domain_value = typed_inputs.domain_id if typed_inputs is not None else self._get_sp_param(sp, "domain_id", None)
        if domain_value is None and domain_name is None:
            domain_value = self.default_domain_id
        if domain_value is not None:
            domain_id = _admission_int(domain_value, "domain_id")
            if domain_id < 0:
                raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams domain_id must be non-negative, got {domain_id}.")
        else:
            try:
                domain_id = self.manifest.resolve_domain_name(str(domain_name))
            except ValueError as exc:
                raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if domain_id >= self.manifest.num_embodiment_domains:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams domain_id is outside the exported embodiment table: "
                f"{domain_id} not in [0, {self.manifest.num_embodiment_domains})."
            )
        try:
            embodiment = self.manifest.resolve_embodiment(domain_name, domain_id)
        except ValueError as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        return _ActionRequestContract(domain_id=domain_id, embodiment=embodiment)

    def _prepare_conditioning(
        self,
        sp,
        *,
        typed_inputs: Any | None,
        request: _ActionRequestContract,
        start_frame: int,
        target_frame: int,
        prompt_data: Any = None,
    ) -> _ActionConditioning:
        """Normalize action rows and freeze their request-wide indexing mode."""

        del prompt_data
        raw_action = self._prepare_raw_action(
            sp,
            embodiment=request.embodiment,
            action_value=typed_inputs.action if typed_inputs is not None else None,
        )
        layout = self._resolve_action_layout(
            raw_action,
            start_frame=start_frame,
            target_frame=target_frame,
        )
        domain_ids = torch.tensor([request.domain_id], device=self.device, dtype=torch.long)
        return _ActionConditioning(
            request=request,
            raw_action=raw_action,
            layout=layout,
            domain_ids=domain_ids,
        )

    def _conditioning_fingerprint(self, request: _ActionRequestContract) -> tuple[tuple[str, Any], ...]:
        return (
            ("domain_id", request.domain_id),
            ("embodiment", request.embodiment),
            ("action_contract_sha256", self.manifest.action_contract_sha256),
        )

    def _build_prompt_tokens(
        self,
        prompt: str,
        *,
        sampling_params: Any,
        prompt_data: Any,
        fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the Humanoid AR prompt without a system prompt."""

        del sampling_params, prompt_data, fps
        return self._tokenize_prompt(
            prompt,
            max_sequence_length=1 << 30,
            use_system_prompt=False,
        )

    def _get_or_create_state(self, session_id: str) -> CosmosDreamsSessionState:
        state = self._states.get(session_id)
        if state is None:
            while len(self._states) >= self._SESSION_CAPACITY:
                _, evicted = self._states.popitem(last=False)
                evicted.reset()
            state = CosmosDreamsSessionState(session_id=session_id)
            self._states[session_id] = state
        self._states.move_to_end(session_id)
        return state

    def _ensure_text_kv(
        self,
        state: CosmosDreamsSessionState,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        cached = state.text_kv_by_branch.get(self._MAIN_BRANCH)
        if cached is not None:
            return cached

        paged_state = self._ar_diffusion_kv_state
        if paged_state is not None and paged_state.is_cross_attention_populated(self._MAIN_BRANCH, "text"):
            pooled = paged_state.get_cross_attention_kv(self._MAIN_BRANCH, "text")
            cached = [(entry["k"], entry["v"]) for entry in pooled]
        else:
            raw_kv, real_len = self.transformer.encode_text_kv(text_ids, text_mask)
            if real_len > self.manifest.text_cache_max_len:
                raise ValueError(
                    f"Cosmos-Dreams prompt exceeds text_cache_max_len: {real_len} > {self.manifest.text_cache_max_len}."
                )
            if paged_state is None:
                cached = raw_kv
            else:
                padded = self.transformer.pad_text_kv(
                    raw_kv,
                    max_len=self.manifest.text_cache_max_len,
                )
                paged_state.populate_cross_attention(self._MAIN_BRANCH, "text", padded)
                pooled = paged_state.get_cross_attention_kv(self._MAIN_BRANCH, "text")
                cached = [(entry["k"], entry["v"]) for entry in pooled]
        state.text_kv_by_branch[self._MAIN_BRANCH] = cached
        return cached

    def _fingerprint(
        self,
        text_ids: torch.Tensor,
        *,
        real_text_kv_len: int,
        height: int,
        width: int,
        fps: float,
        conditioning_request: Any,
    ) -> CosmosDreamsSessionFingerprint:
        return CosmosDreamsSessionFingerprint(
            prompt_hash=prompt_token_hash(text_ids),
            real_text_kv_lengths=((self._MAIN_BRANCH, real_text_kv_len),),
            height=height,
            width=width,
            fps=fps,
            conditioning=self._conditioning_fingerprint(conditioning_request),
            checkpoint_id=self.checkpoint_id,
            manifest_id=self.manifest.digest,
            sampler_id=self.manifest.sampler_id,
        )

    # -- Dense/paged transformer bridge -----------------------------------

    def _append_dense_kv(
        self,
        state: CosmosDreamsSessionState,
        current_kv: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        history = state.dense_kv_by_branch.get(self._MAIN_BRANCH)
        state.dense_kv_by_branch[self._MAIN_BRANCH] = append_dense_kv_history(
            history,
            current_kv,
            tokens_per_frame=self.manifest.tokens_per_frame,
            sink_frames=self.manifest.sink_frames,
            window_frames=self.manifest.window_frames,
        )

    def _transformer_forward(
        self,
        state: CosmosDreamsSessionState,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        *,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        frame_start: int,
        fps: float,
        conditioning_kwargs: Mapping[str, Any],
        condition_vision: bool,
        commit_current: bool,
    ) -> CosmosDreamsTransformerOutput:
        paged_state = self._ar_diffusion_kv_state
        seq_len = hidden_states.shape[2] * self.manifest.tokens_per_frame
        paged_kv = None
        dense_history = None
        if paged_state is not None:
            paged_kv = paged_state.get_kv_caches(
                self._MAIN_BRANCH,
                seq_len=seq_len,
                commit_current=commit_current,
            )
        else:
            dense_history = state.dense_kv_by_branch.get(self._MAIN_BRANCH)

        output = self.transformer(
            hidden_states,
            timestep,
            text_kv=text_kv,
            real_text_kv_len=real_text_kv_len,
            frame_start=frame_start,
            fps=fps,
            paged_kv=paged_kv,
            dense_history=dense_history,
            condition_vision=condition_vision,
            **conditioning_kwargs,
        )
        if paged_state is not None:
            paged_state.commit_paged_context(self._MAIN_BRANCH)
        elif commit_current:
            self._append_dense_kv(state, output.current_kv)
        return output

    # -- Action and latent preparation ------------------------------------

    def _prepare_raw_action(
        self,
        sp,
        *,
        embodiment: str,
        action_value: Any = None,
    ) -> torch.Tensor | None:
        if action_value is None:
            action_value = self._get_sp_param(sp, "action", None)
        if action_value is None:
            return None
        action = self.action_normalizers[embodiment].normalize(load_action_tensor(action_value))
        action = pad_action_to_dim(action, self.manifest.max_action_dim)
        return action.to(device=self.device, dtype=self.dtype)

    def _resolve_action_layout(
        self,
        raw_action: torch.Tensor | None,
        *,
        start_frame: int,
        target_frame: int,
    ) -> str | None:
        """Decide once per request how raw action rows are indexed.

        ``global``: row block ``[(f-1)*A, f*A)`` conditions latent frame ``f``
        ``local``:
        rows cover exactly this request's non-prefix frames in order (the
        chunk-per-request tick layout). Resolving once keeps the
        interpretation stable across every chunk of the request and turns
        insufficient coverage into an admission rejection instead of a
        mid-rollout failure after frames were already committed.
        """
        if raw_action is None:
            return None
        action_count = self.manifest.action_tokens_per_frame
        global_rows = max((target_frame - 1) * action_count, 0)
        local_rows = sum(action_count for frame in range(start_frame, target_frame) if frame > 0)
        rows = raw_action.shape[0]
        if rows >= global_rows:
            return "global"
        if rows == local_rows:
            return "local"
        raise ARDiffusionRequestRejectedError(
            "Cosmos-Dreams action length cannot cover the requested latent frames: "
            f"rows={rows}, frame_range=[{start_frame}, {target_frame}), "
            f"expected local rows={local_rows} or at least global rows={global_rows}."
        )

    def _actions_for_frames(
        self,
        raw_action: torch.Tensor | None,
        *,
        layout: str | None,
        request_start_frame: int,
        frame_start: int,
        frame_end: int,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        action_count = self.manifest.action_tokens_per_frame
        frame_count = frame_end - frame_start
        if raw_action is None or layout is None:
            zeros = torch.zeros(
                1,
                frame_count * action_count,
                self.manifest.max_action_dim,
                device=self.device,
                dtype=self.dtype,
            )
            return zeros, tuple(range(frame_count))

        # First raw row conditions the first non-prefix frame of the request.
        local_base_frame = max(request_start_frame, 1)
        rows: list[torch.Tensor] = []
        null_indexes: list[int] = []
        for local_idx, frame_idx in enumerate(range(frame_start, frame_end)):
            if frame_idx == 0:
                rows.append(raw_action.new_zeros(action_count, self.manifest.max_action_dim))
                null_indexes.append(local_idx)
                continue
            if layout == "global":
                start = (frame_idx - 1) * action_count
            else:
                start = (frame_idx - local_base_frame) * action_count
            rows.append(raw_action[start : start + action_count])
        return torch.cat(rows, dim=0).unsqueeze(0), tuple(null_indexes)

    def _initial_condition_latent(self, prompt_data: Any, sp) -> torch.Tensor | None:
        explicit = self._get_sp_param(sp, "initial_latent", None)
        if explicit is not None:
            latent = explicit if isinstance(explicit, torch.Tensor) else torch.as_tensor(explicit)
            if latent.ndim == 4:
                latent = latent.unsqueeze(2)
            expected = (
                1,
                self.transformer.latent_channel_size,
                1,
                self.manifest.latent_height,
                self.manifest.latent_width,
            )
            if tuple(latent.shape) != expected:
                raise ValueError(f"Cosmos-Dreams initial_latent must have shape {expected}, got {tuple(latent.shape)}.")
            return latent.to(device=self.device, dtype=self.dtype)

        if isinstance(prompt_data, str):
            return None
        additional = prompt_data.get("additional_information", {}) or {}
        image = additional.get("preprocessed_image")
        video = additional.get("preprocessed_video")
        if image is None and isinstance(video, torch.Tensor):
            if video.ndim != 5:
                raise ValueError(
                    f"Cosmos-Dreams preprocessed video must have shape [1,3,T,H,W], got {tuple(video.shape)}."
                )
            image = video[:, :, 0]
        if image is None:
            return None
        if not isinstance(image, torch.Tensor):
            raise TypeError("Cosmos-Dreams preprocessed image must be a torch.Tensor.")
        return self._encode_conditioning_image_latent(image)

    def _commit_clean_frame(
        self,
        state: CosmosDreamsSessionState,
        latent: torch.Tensor,
        *,
        frame_idx: int,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        fps: float,
        conditioning_kwargs: Mapping[str, Any],
    ) -> None:
        self._transformer_forward(
            state,
            latent.to(self.dtype),
            torch.zeros(1, device=self.device, dtype=torch.float32),
            text_kv=text_kv,
            real_text_kv_len=real_text_kv_len,
            frame_start=frame_idx,
            fps=fps,
            conditioning_kwargs=conditioning_kwargs,
            condition_vision=True,
            commit_current=True,
        )

    def _denoise_chunk(
        self,
        state: CosmosDreamsSessionState,
        *,
        chunk_start: int,
        chunk_end: int,
        seed: int,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        fps: float,
        conditioning_kwargs: Mapping[str, Any],
        tick_durations: dict[str, float],
        measure_tick_latency: bool,
    ) -> torch.Tensor:
        """Denoise one canonical AR chunk without committing its noisy K/V."""

        chunk_frames = chunk_end - chunk_start
        noise_generator = torch.Generator(device=self.device).manual_seed(seed + chunk_start)
        initial_noise = torch.randn(
            1,
            self.transformer.latent_channel_size,
            chunk_frames,
            self.manifest.latent_height,
            self.manifest.latent_width,
            generator=noise_generator,
            device=self.device,
            dtype=self.dtype,
        )

        def velocity_fn(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            output = self._transformer_forward(
                state,
                x.to(self.dtype),
                timestep,
                text_kv=text_kv,
                real_text_kv_len=real_text_kv_len,
                frame_start=chunk_start,
                fps=fps,
                conditioning_kwargs=conditioning_kwargs,
                condition_vision=False,
                commit_current=False,
            )
            return output.video.float()

        with self._timed_tick_stage(
            tick_durations,
            "denoise_s",
            enabled=measure_tick_latency,
        ):
            return self.distilled_sampler.sample(
                velocity_fn,
                initial_noise,
                seed=seed,
                frame_idx=chunk_start,
            ).to(self.dtype)

    def _run_chunk(
        self,
        state: CosmosDreamsSessionState,
        *,
        chunk_start: int,
        chunk_end: int,
        target_frame: int,
        terminal_request: bool,
        request_start_frame: int,
        seed: int,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        fps: float,
        conditioning: _ActionConditioning,
        tick_durations: dict[str, float],
        measure_tick_latency: bool,
    ) -> torch.Tensor:
        """Run Humanoid action conditioning, denoise, then clean-refresh."""

        action_chunk, null_action_indexes = self._actions_for_frames(
            conditioning.raw_action,
            layout=conditioning.layout,
            request_start_frame=request_start_frame,
            frame_start=chunk_start,
            frame_end=chunk_end,
        )
        conditioning_kwargs = {
            "action_latents": action_chunk,
            "action_domain_ids": conditioning.domain_ids,
            "null_action_frame_indexes": null_action_indexes,
        }
        clean_chunk = self._denoise_chunk(
            state,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            seed=seed,
            text_kv=text_kv,
            real_text_kv_len=real_text_kv_len,
            fps=fps,
            conditioning_kwargs=conditioning_kwargs,
            tick_durations=tick_durations,
            measure_tick_latency=measure_tick_latency,
        )

        action_count = self.manifest.action_tokens_per_frame
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
                action_start = local_idx * action_count
                action_frame = action_chunk[:, action_start : action_start + action_count]
                self._commit_clean_frame(
                    state,
                    clean_chunk[:, :, local_idx : local_idx + 1],
                    frame_idx=frame_idx,
                    text_kv=text_kv,
                    real_text_kv_len=real_text_kv_len,
                    fps=fps,
                    conditioning_kwargs={
                        "action_latents": action_frame,
                        "action_domain_ids": conditioning.domain_ids,
                        "null_action_frame_indexes": (0,) if local_idx in null_action_indexes else (),
                    },
                )
        return clean_chunk

    def _prefill_first_frame(
        self,
        state: CosmosDreamsSessionState,
        initial_latent: torch.Tensor | None,
        *,
        target_frame: int,
        terminal_request: bool,
        request_start_frame: int,
        text_kv: list[tuple[torch.Tensor, torch.Tensor]],
        real_text_kv_len: int,
        fps: float,
        conditioning: _ActionConditioning,
        tick_durations: dict[str, float],
        measure_tick_latency: bool,
        retain_latent: bool,
    ) -> torch.Tensor | None:
        """Seed and commit the optional Humanoid first-frame condition."""

        if initial_latent is None or state.next_frame_idx != 0:
            return None
        initial_action, initial_null = self._actions_for_frames(
            conditioning.raw_action,
            layout=conditioning.layout,
            request_start_frame=request_start_frame,
            frame_start=0,
            frame_end=1,
        )
        if target_frame > 1 or not terminal_request:
            with self._timed_tick_stage(
                tick_durations,
                "clean_cache_commit_s",
                enabled=measure_tick_latency,
            ):
                self._commit_clean_frame(
                    state,
                    initial_latent,
                    frame_idx=0,
                    text_kv=text_kv,
                    real_text_kv_len=real_text_kv_len,
                    fps=fps,
                    conditioning_kwargs={
                        "action_latents": initial_action,
                        "action_domain_ids": conditioning.domain_ids,
                        "null_action_frame_indexes": (0,) if initial_null else (),
                    },
                )
        state.append_chunk(initial_latent, frame_start=0, retain_latent=retain_latent)
        return initial_latent

    def _denormalize_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.vae.dtype)
        if hasattr(self.vae.config, "latents_mean") and hasattr(self.vae.config, "latents_std"):
            latents_mean, latents_std = self._get_latents_mean_std(latents.device, latents.dtype)
            return (latents * latents_std) + latents_mean
        return latents / float(getattr(self.vae.config, "scaling_factor", 1.0))

    def _decode_live_latents(
        self,
        state: CosmosDreamsSessionState,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Decode only the new tick block with session-owned Wan features."""

        execution_context = getattr(self.vae, "_execution_context", None)
        with execution_context() if callable(execution_context) else nullcontext():
            result = decode_wan_causal_chunk(
                self.vae,
                self._denormalize_vae_latents(latents),
                feature_cache=state.vae_decoder_feat_cache,
                initialized=state.vae_decoder_initialized,
            )
        state.record_incremental_decode(
            input_frames=int(latents.shape[2]),
            feature_cache=result.feature_cache,
        )
        return result.video

    # -- Generation --------------------------------------------------------

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        try:
            return self._forward_impl(req)
        except ARDiffusionRequestRejectedError:
            # Admission rejection: guaranteed to be raised before any session
            # or KV side effect, so the session (and its paid-for history)
            # survives for a corrected retry or an explicit reset.
            raise
        except Exception:
            extra = req.sampling_params.extra_args or {}
            session_id = str(extra.get("session_id") or self._bound_session_id or "default")
            self._drop_session(session_id)
            raise

    def _forward_impl(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        # ---- Admission (pure) ------------------------------------------------
        # Everything in this section only reads request and session state and
        # raises ARDiffusionRequestRejectedError on invalid input. No session may be
        # created, initialized, evicted, or written before it completes: the
        # rejection contract promises the client an unchanged session.
        if len(req.prompts) != 1:
            raise ARDiffusionRequestRejectedError("CosmosDreamsPipeline supports exactly one prompt per request.")
        prompt_data = req.prompts[0]
        if isinstance(prompt_data, str):
            prompt = prompt_data
        elif isinstance(prompt_data, Mapping):
            prompt = str(prompt_data.get("prompt", ""))
        else:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams prompt input must be a string or a mapping with a 'prompt' field."
            )
        sp = req.sampling_params
        extra = {} if sp.extra_args is None else sp.extra_args
        if not isinstance(extra, Mapping):
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams extra_args must be a mapping, got {type(extra).__name__}."
            )
        try:
            typed_tick = ARDiffusionTickRequest.from_extra_args(extra)
            typed_inputs = self._parse_tick(typed_tick) if typed_tick is not None else None
        except ValueError as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if typed_tick is not None:
            if self._ar_diffusion_kv_state is None:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams typed ticks require ARDiffusionEngine session binding."
                )
            if typed_tick.prompt is not None and typed_tick.prompt != prompt:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams ar_diffusion_tick.prompt must match the standard request prompt."
                )
        session_id = str(
            typed_tick.session_id
            if typed_tick is not None
            else extra.get("session_id") or self._bound_session_id or "default"
        )
        if self._bound_session_id is not None and session_id != self._bound_session_id:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams request session {session_id!r} does not match bound session {self._bound_session_id!r}."
            )

        reset = typed_tick.reset if typed_tick is not None else bool(extra.get("reset", False))
        close_session = typed_tick.close_session if typed_tick is not None else bool(extra.get("close_session", False))
        tick = typed_tick is not None or bool(extra.get("chunk_only", False))
        measure_tick_latency = tick and (
            typed_inputs.measure_tick_latency
            if typed_inputs is not None
            else bool(extra.get("measure_tick_latency", False))
        )
        tick_output_type = "latent" if sp.output_type == "latent" else "video"
        existing_state = self._states.get(session_id)
        if reset or existing_state is None or existing_state.fingerprint is None:
            existing_state = None
        state_was_new = existing_state is None
        self._validate_session_mode(
            tick=tick,
            state_was_new=state_was_new,
            session_id=session_id,
            reset=reset,
            close_session=close_session,
        )

        height = _admission_int(_first_not_none(sp.height, self.manifest.height), "height")
        width = _admission_int(_first_not_none(sp.width, self.manifest.width), "width")
        if (height, width) != (self.manifest.height, self.manifest.width):
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams resolution is fixed per deployment: "
                f"requested {height}x{width}, configured {self.manifest.height}x{self.manifest.width}."
            )
        fps = self._resolve_request_fps(sp, prompt_data)
        if not math.isfinite(fps) or fps <= 0:
            raise ARDiffusionRequestRejectedError(f"Cosmos-Dreams FPS must be positive, got {fps}.")
        conditioning_request = self._validate_conditioning_request(
            sp,
            typed_inputs,
            prompt_data=prompt_data,
        )

        guidance_scale = _admission_float(
            _first_not_none(self._get_sp_param(sp, "guidance_scale", None), 1.0),
            "guidance_scale",
        )
        if guidance_scale != 1.0:
            raise ARDiffusionRequestRejectedError(
                f"Cosmos-Dreams distilled inference requires guidance_scale=1.0, got {guidance_scale}."
            )
        if sp.num_inference_steps not in (None, len(self.manifest.t_list)):
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams distilled inference uses the checkpoint-defined four-step schedule; "
                f"got num_inference_steps={sp.num_inference_steps}."
            )

        text_ids, text_mask = self._build_prompt_tokens(
            prompt,
            sampling_params=sp,
            prompt_data=prompt_data,
            fps=fps,
        )
        real_text_kv_len = int(text_mask[0].sum().item())
        if real_text_kv_len > self.manifest.text_cache_max_len:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams prompt exceeds text_cache_max_len: "
                f"{real_text_kv_len} > {self.manifest.text_cache_max_len}."
            )
        fingerprint = self._fingerprint(
            text_ids,
            real_text_kv_len=real_text_kv_len,
            height=height,
            width=width,
            fps=fps,
            conditioning_request=conditioning_request,
        )
        start_frame = 0 if state_was_new else existing_state.next_frame_idx
        requested_frame_idx = _admission_int(
            typed_inputs.frame_idx if typed_inputs is not None else extra.get("frame_idx", start_frame),
            "frame_idx",
        )
        if state_was_new:
            if requested_frame_idx != 0:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos-Dreams new sessions must start at latent frame 0; got {requested_frame_idx}."
                )
        else:
            try:
                existing_state.validate_request(fingerprint, frame_idx=requested_frame_idx)
            except (ValueError, RuntimeError) as exc:
                raise ARDiffusionRequestRejectedError(str(exc)) from exc
            if tick and existing_state.tick_output_type not in (None, tick_output_type):
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams tick output_type cannot change within a session; session reset required."
                )

        try:
            initial_latent = self._initial_condition_latent(prompt_data, sp)
        except (TypeError, ValueError) as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc
        if start_frame > 0 and initial_latent is not None:
            raise ARDiffusionRequestRejectedError(
                "Cosmos-Dreams initial media may only be supplied at frame 0; session reset required."
            )

        requested_pixel_frames: int | None = None
        if tick:
            tick_frames = _admission_int(
                (
                    typed_inputs.num_latent_frames
                    if typed_inputs is not None
                    else extra.get("num_latent_frames", self.manifest.chunk_size)
                ),
                "num_latent_frames",
            )
            if tick_frames <= 0:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos-Dreams num_latent_frames must be positive, got {tick_frames}."
                )
            target_frame = start_frame + tick_frames
            # Frame zero is the singleton causal prefix. A normal first tick
            # therefore advances through [0, 1) and then one [1, 5) chunk,
            # regardless of whether frame zero is supplied or generated.
            if start_frame == 0:
                target_frame += 1
            if not close_session and (target_frame - 1) % self.manifest.chunk_size != 0:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams non-terminal ticks must end on a canonical [1,4,4,...] "
                    f"chunk boundary, got target latent frame {target_frame}."
                )
        else:
            requested_pixel_frames = self._resolve_requested_pixel_frames(
                sp,
                prompt_data,
                conditioning_request,
            )
            if requested_pixel_frames <= 0:
                raise ARDiffusionRequestRejectedError(
                    f"Cosmos-Dreams num_frames must be positive, got {requested_pixel_frames}."
                )
            target_frame = (requested_pixel_frames - 1) // self.manifest.temporal_compression_factor + 1
            if target_frame < start_frame:
                raise ARDiffusionRequestRejectedError(
                    "Cosmos-Dreams full rollout target precedes existing session state; session reset required."
                )
        try:
            conditioning = self._prepare_conditioning(
                sp,
                typed_inputs=typed_inputs,
                request=conditioning_request,
                start_frame=start_frame,
                target_frame=target_frame,
                prompt_data=prompt_data,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ARDiffusionRequestRejectedError(str(exc)) from exc

        # ---- Side effects begin ---------------------------------------------
        tick_durations: dict[str, float] = {}
        self._sync_for_tick_timing(measure_tick_latency)
        tick_total_started = time.perf_counter() if measure_tick_latency else 0.0
        if reset:
            self._drop_session(session_id)
        state = self._get_or_create_state(session_id)
        if state.fingerprint is None:
            state.initialize(fingerprint)
        if tick and state.tick_output_type is None:
            state.tick_output_type = tick_output_type
        text_kv = self._ensure_text_kv(state, text_ids, text_mask)

        terminal_request = close_session or not tick
        seed = self._resolve_seed(sp, sp.generator if isinstance(sp.generator, torch.Generator) else None)
        request_latent_chunks: list[torch.Tensor] = []

        prefilled = self._prefill_first_frame(
            state,
            initial_latent,
            target_frame=target_frame,
            terminal_request=terminal_request,
            request_start_frame=start_frame,
            text_kv=text_kv,
            real_text_kv_len=real_text_kv_len,
            fps=fps,
            conditioning=conditioning,
            tick_durations=tick_durations,
            measure_tick_latency=measure_tick_latency,
            retain_latent=not tick,
        )
        if prefilled is not None:
            request_latent_chunks.append(prefilled)

        generation_start = state.next_frame_idx
        for chunk_start, chunk_end in iter_ar_chunk_ranges(
            generation_start,
            target_frame,
            self.manifest.chunk_size,
        ):
            clean_chunk = self._run_chunk(
                state,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                target_frame=target_frame,
                terminal_request=terminal_request,
                request_start_frame=start_frame,
                seed=seed,
                text_kv=text_kv,
                real_text_kv_len=real_text_kv_len,
                fps=fps,
                conditioning=conditioning,
                tick_durations=tick_durations,
                measure_tick_latency=measure_tick_latency,
            )
            state.append_chunk(clean_chunk, frame_start=chunk_start, retain_latent=not tick)
            request_latent_chunks.append(clean_chunk)

        if not request_latent_chunks:
            raise RuntimeError("Cosmos-Dreams request produced no new latent frames.")
        request_latents = torch.cat(request_latent_chunks, dim=2)
        accumulated = state.accumulated_latents
        if not tick and accumulated is None:
            raise RuntimeError("Cosmos-Dreams full rollout produced no accumulated latent frames.")
        if sp.output_type == "latent":
            output_value = request_latents if tick else accumulated
        else:
            with self._timed_tick_stage(
                tick_durations,
                "vae_decode_s",
                enabled=measure_tick_latency,
            ):
                if tick:
                    output_value = self._decode_live_latents(state, request_latents).clamp(-1, 1)
                else:
                    output_value = self._decode_latents(accumulated).clamp(-1, 1)
            if not tick:
                assert requested_pixel_frames is not None
                output_value = output_value[:, :, :requested_pixel_frames]

        if not tick:
            state.terminal = True
        if measure_tick_latency:
            self._sync_for_tick_timing(True)
            tick_durations["tick_model_total_s"] = time.perf_counter() - tick_total_started
        output: dict[str, Any] = {"video": output_value}
        if typed_tick is not None:
            output = {
                "payload": output,
                "metadata": {
                    "ar_diffusion": ARDiffusionChunkMetadata.from_tick(typed_tick).to_dict(),
                },
            }
        result = DiffusionOutput(output=output, stage_durations=tick_durations)
        if close_session and self._ar_diffusion_kv_state is None:
            self._drop_session(session_id)
        return result


# Export-manifest aliases used while the upstream name settles.
CosmosDreamsOmniPipeline = CosmosDreamsPipeline
Cosmos3InteractivePipeline = CosmosDreamsPipeline
