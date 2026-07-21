# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero pipeline for vllm-omni.

Entry point for DiffusionEngine.step() -> pipeline.forward(req)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections import OrderedDict
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.cache.stepcache import (
    get_stepcache_state,
    is_stepcache_active,
)
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import (
    DistributedAutoencoderKLWan,
)
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.parallel_state import get_classifier_free_guidance_world_size
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.dreamzero.causal_wan_model import CausalWanModel
from vllm_omni.diffusion.models.dreamzero.image_encoder import DreamZeroImageEncoder
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import (
    DreamZeroStageCarrier,
    DreamZeroState,
)
from vllm_omni.diffusion.models.dreamzero.transform import (
    DEFAULT_EMBODIMENT,
    ensure_transforms_loaded,
)
from vllm_omni.diffusion.models.dreamzero.transform.base import get_transform
from vllm_omni.diffusion.models.dreamzero.utils import (
    DEFAULT_CFG_SCALE,
    DEFAULT_EMBODIMENT_NAME_TO_ID,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_SIGMA_SHIFT,
)
from vllm_omni.diffusion.models.interface import StageBoundary, StagePayload
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.stage_payload import StagePayloadError
from vllm_omni.diffusion.stage_roles import (
    ALL_COMPONENTS,
    DECODE,
    DENOISE,
    ENCODE,
    MONOLITHIC,
    StageComponentSpec,
    normalize_stage_role,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

logger = logging.getLogger(__name__)
MAX_DREAMZERO_SESSIONS = 64


def _wan_vae_latents_mean_std() -> tuple[list[float], list[float]]:
    """Wan 2.1 VAE (16-channel) latent normalization constants.

    Mirrors ``AutoencoderKLWan.config.latents_mean`` / ``latents_std``. Used only
    to build DreamZero's ``vae_latents_mean`` / ``vae_latents_inv_std`` buffers on
    the disaggregated *denoise* stage, which skips constructing the VAE module
    (and therefore cannot read ``self.vae.config``). Encode/decode stages and the
    monolithic path read these directly from the constructed VAE, so this helper
    is a stage-partial-loading fallback, not a second source of truth for them.
    """
    latents_mean = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    latents_std = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ]
    return latents_mean, latents_std


class VideoActionScheduler:
    """Wraps video + action schedulers into single .step() interface."""

    def __init__(self, video_scheduler, action_scheduler):
        self.video_scheduler = video_scheduler
        self.action_scheduler = action_scheduler

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(
            noise_pred[0],
            t[0],
            latents[0],
            return_dict=False,
            generator=generator,
        )[0]
        action_out = self.action_scheduler.step(
            noise_pred[1],
            t[1],
            latents[1],
            return_dict=False,
            generator=generator,
        )[0]
        return ((video_out, action_out),)


# ---------------------------------------------------------------------------
# DreamZeroPipeline
# ---------------------------------------------------------------------------


class DreamZeroPipeline(nn.Module, CFGParallelMixin):
    """DreamZero world model pipeline.

    Multi-output: predict_noise() returns (video_pred, action_pred).
    CFG: video gets standard CFG, action takes positive branch only.

    KV is managed by the AR-Diffusion engine: ``self._ar_diffusion_kv_state`` is set by the
    runner before ``forward()`` and the pipeline routes all KV access (get / update /
    commit / reset) through the pool-backed state. Purely duck-typed (no engine import).
    """

    _ar_diffusion_kv_state = None  # set by the runner before each forward

    # DreamZero requires a real robot_obs (raw camera frames) for every
    # request; the engine's synthetic dummy-warmup request has no robot_obs
    # and would fail check_inputs/encode on every role (encode, denoise,
    # decode -- not just the monolithic path, which special-cases it in
    # forward()). Skip the warmup entirely rather than special-case each
    # disaggregated atom.
    dummy_run_num_frames = 0

    def _kv_get(self, state, is_negative, seq_len=None, update_kv_cache=False):
        return self._ar_diffusion_kv_state.get_kv_caches(
            is_negative,
            seq_len=seq_len,
            commit_current=update_kv_cache,
        )

    def _kv_create(self, state, batch_size, dtype, device, num_layers, num_heads, head_dim):
        # The engine owns all KV allocation: self-attn is allocated lazily from
        # paged contexts, and cross-attn is populated eagerly in _kv_populate_cross.
        # Nothing is created model-side.
        return

    def _kv_commit(self, is_negative: bool):
        self._ar_diffusion_kv_state.commit_paged_context(is_negative)

    def _kv_get_cross(self, state, is_negative):
        """Cross-attn cache from the engine pool (text k/v + I2V image k_img/v_img)."""
        return self._ar_diffusion_kv_state.get_cross_kv_caches(is_negative)

    @staticmethod
    def _layerwise_offload_hook(block):
        """Return the layerwise-offload hook attached to a DiT block, or None.

        When layerwise CPU offload is active each block carries a HookRegistry
        with a LayerwiseOffloadHook; the hook materializes/frees the block's
        weights around its forward(). Code paths that use block weights outside
        forward() (e.g. eager cross-attn KV population) must onload/offload via
        this hook. Returns None when offload is not enabled.
        """
        registry = getattr(block, "_hook_registry", None)
        if registry is None:
            return None
        try:
            from vllm_omni.diffusion.offloader.layerwise_backend import LayerwiseOffloadHook

            return registry._hooks.get(LayerwiseOffloadHook._HOOK_NAME)
        except Exception:
            return None

    def _kv_populate_cross(self, context: torch.Tensor, clip_feature, is_negative: bool) -> None:
        """Eagerly project cross-attn K/V for all layers into the AR-Diffusion pool.

        Caches the session-invariant cross-attn projections once, per half: the text
        ``k``/``v`` from ``text_embedding(context)`` survive window-boundary resets
        (prompt unchanged within a session — only session resets clear them), while
        the I2V image-token ``k_img``/``v_img`` from ``img_emb(clip_feature)`` (the
        257 image tokens the forward splits off, cached model-side by #4154) are
        re-projected on every window restart from the fresh CLIP features. Must run
        after the image is encoded so ``clip_feature`` is available.
        """
        s = self._ar_diffusion_kv_state
        need_text = not s._cross_text_populated.get(is_negative, False)
        need_img = (
            clip_feature is not None
            and getattr(self.transformer, "model_type", "t2v") == "i2v"
            and not s._cross_img_populated.get(is_negative, False)
        )
        if not need_text and not need_img:
            return
        projected = self.transformer.text_embedding(context) if need_text else None
        img_ctx = self.transformer.img_emb(clip_feature) if need_img else None
        for i, block in enumerate(self.transformer.blocks):
            # Layerwise-offload compatibility: this eager cross-KV precompute reaches
            # into block.cross_attn.{k,v,...} WITHOUT calling block.forward(), so the
            # offload pre_forward/post_forward hooks (which materialize block weights
            # on device and free them after) never fire and the projection weights
            # stay as CPU/empty placeholders -> "mat1 on xpu, weight on cpu" addmm
            # error. Mirror the hook's self-heal path: materialize this block before
            # use and offload it after. No-op when offload is disabled (no hook).
            _off_hook = self._layerwise_offload_hook(block)
            if _off_hook is not None and not _off_hook.is_materialized and _off_hook._prev_hook is not None:
                _off_hook._prev_hook.prefetch_layer(non_blocking=False)
            try:
                ca = block.cross_attn
                n, d = ca.tp_num_heads, ca.head_dim
                k = v = None
                if projected is not None:
                    k = ca.norm_k(ca.k(projected)).unflatten(2, (n, d))
                    v = ca.v(projected).unflatten(2, (n, d))
                k_img = v_img = None
                if img_ctx is not None:
                    k_img = ca.norm_k_img(ca.k_img(img_ctx)).unflatten(2, (n, d))
                    v_img = ca.v_img(img_ctx).unflatten(2, (n, d))
                s.kv_cache.write_cross_kv(i, is_negative, k, v, k_img, v_img)
            finally:
                if _off_hook is not None:
                    _off_hook.offload_layer()
        if need_text:
            s._cross_text_populated[is_negative] = True
        if need_img:
            s._cross_img_populated[is_negative] = True
        logger.info(
            "AR-Diffusion CROSS POPULATE [%s]: %d layers, text=%s img=%s",
            "neg" if is_negative else "pos",
            len(self.transformer.blocks),
            "kept" if projected is None else tuple(context.shape),
            None if img_ctx is None else tuple(img_ctx.shape),
        )

    def _kv_reset(self, state, *, clear_video_latents: bool = True):
        """Reset the engine's pooled session window plus the model's non-KV state.

        DreamZero resets at the attention-window boundary; the engine pool drops the
        same window so the next forward starts fresh. ``clear_video_latents=False``
        keeps the accumulated video latents for export.

        ``clear_video_latents=False`` also marks a window ("inference") reset: the
        prompt is unchanged, so the pool keeps the text cross-attn K/V and only the
        image half repopulates on the restart forward.

        On the encode worker ``_ar_diffusion_kv_state`` is unset (encode owns no
        pool), so the engine-side reset is a no-op there; only the model-local
        state resets.
        """
        state.reset(clear_video_latents=clear_video_latents)
        if self._ar_diffusion_kv_state is not None:
            self._ar_diffusion_kv_state.reset(keep_cross_text=not clear_video_latents)

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        """Initialize pipeline components.

        DreamZero root checkpoint layout (GEAR-Dreams/DreamZero-DROID):
          config.json                     -- root config (action_head_cfg, architectures, etc.)
          model-*.safetensors             -- all learned weights (action_head.{model,text_encoder,image_encoder,vae}.*)
          experiment_cfg/metadata.json    -- per-embodiment action normalization stats
          vae/                            -- symlink to Wan2.1 VAE (diffusers-compatible)

        Components are instantiated from config (not from_pretrained), then filled
        by load_weights() which reads root safetensors and remaps key prefixes.
        Exceptions:
        - tokenizer loads from `google/umt5-xxl`
        - VAE uses `DistributedAutoencoderKLWan` as the local execution module.
          It can be bootstrapped either from an explicit diffusers source
          (`od_config.model_paths["vae"]`) or directly from constructor defaults
          that match Wan2.1 VAE, after which DreamZero root
          `action_head.vae.*` weights are remapped onto that module in
          `load_weights()`
        """
        super().__init__()

        # DreamZero's DiT-owning roles (denoise, and monolithic which owns
        # everything) are engine-only: every KV access in forward() routes
        # through the AR-Diffusion engine's pool-backed state. Fail fast here
        # — a stale or programmatic config that leaves engine_backend="default"
        # would otherwise only crash mid-forward on the first KV access.
        # Encode/decode own no DiT/KV (see required_components_for_stage) and
        # run on the standard diffusion engine, so they are exempt.
        model_stage_role = normalize_stage_role(getattr(od_config, "model_stage", None))
        if model_stage_role in (DENOISE, MONOLITHIC):
            engine_backend = str(getattr(od_config, "engine_backend", "") or "")
            if "ar_diffusion" not in engine_backend.lower().replace("-", "_"):
                raise ValueError(
                    "DreamZeroPipeline requires the AR-Diffusion engine for the "
                    f"{model_stage_role!r} role; set "
                    "engine_backend: vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine "
                    f"in the deploy config (got engine_backend={engine_backend!r})."
                )

        model_path = od_config.model
        model_config = od_config.model_config
        local_files_only = os.path.exists(model_path)
        self.od_config = od_config
        ensure_transforms_loaded()
        self.default_robot_embodiment = model_config.get(
            "default_robot_embodiment",
            DEFAULT_EMBODIMENT,
        )

        root_cfg = self._load_repo_json(model_path, "config.json", local_files_only)
        if root_cfg is None:
            raise ValueError(f"DreamZero requires root config.json in {model_path}.")
        action_head_cfg = root_cfg["action_head_cfg"]
        ah_config = action_head_cfg["config"]
        diffusion_model_cfg = ah_config["diffusion_model_cfg"]

        # ---- Stage-specific partial construction (RFC #4590 §8) ----
        # Build only the components this stage's role needs. The monolithic role
        # (diffusion / None) builds everything, so single-stage DreamZero is
        # unchanged. Weight loading self-gates: load_weights only fills params
        # present on constructed submodules. Components skipped are set to None
        # and guarded at their use sites (setup_compile, the phase methods).
        self._model_stage = model_stage_role
        self._component_spec = self.required_components_for_stage(self._model_stage)
        spec = self._component_spec
        logger.info(
            "DreamZeroPipeline building for stage role %r: components=%s",
            self._model_stage,
            spec.describe(),
        )

        # ---- Tokenizer ----
        self.tokenizer = None
        if spec.tokenizer:
            tokenizer_source = od_config.model_paths.get("tokenizer", "google/umt5-xxl")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        # ---- Text encoder (UMT5) ----
        # Instantiate from config; weights load through `load_weights()`.
        self.text_encoder = None
        if spec.text_encoder:
            umt5_config = UMT5Config(
                d_model=4096,
                d_ff=10240,
                num_heads=64,
                num_layers=24,
                vocab_size=256384,
                relative_attention_num_buckets=32,
                relative_attention_max_distance=128,
                dense_act_fn="gelu_new",
                feed_forward_proj="gated-gelu",
                is_encoder_decoder=False,
            )
            self.text_encoder = UMT5EncoderModel(umt5_config)

        # ---- Image encoder (CLIP) ----
        self.image_encoder = None
        if spec.image_encoder:
            self.image_encoder = DreamZeroImageEncoder()

        # ---- VAE ----
        # The diffusers AutoencoderKLWan constructor is monolithic (encoder +
        # decoder in one __init__), so any stage needing EITHER half builds the
        # full module (audit option 3). The denoise stage needs NEITHER half, so
        # it skips the VAE entirely and derives the latents_mean/std buffers from
        # the Wan VAE default config constants instead of the module.
        self.vae = None
        needs_vae = spec.vae_encoder or spec.vae_decoder
        if needs_vae:
            vae_source = od_config.model_paths.get("vae")
            if vae_source:
                self.vae = DistributedAutoencoderKLWan.from_pretrained(
                    vae_source,
                    torch_dtype=torch.float32,
                )
            elif local_files_only and os.path.isdir(os.path.join(model_path, "vae")):
                self.vae = DistributedAutoencoderKLWan.from_pretrained(
                    model_path,
                    subfolder="vae",
                    torch_dtype=torch.float32,
                )
            else:
                self.vae = DistributedAutoencoderKLWan()
                self.vae.init_distributed()
            if not (
                getattr(od_config, "enable_cpu_offload", False)
                or getattr(od_config, "enable_layerwise_offload", False)
            ):
                self.vae = self.vae.to(device=get_local_device(), dtype=od_config.dtype)
            latents_mean = self.vae.config.latents_mean
            latents_std = self.vae.config.latents_std
        else:
            # Wan 2.1 VAE latent normalization constants (16 channels). Needed by
            # encode/decode math; carried here so the denoise stage's buffers
            # exist even without the VAE module (they are simply unused there).
            latents_mean, latents_std = _wan_vae_latents_mean_std()
        self.register_buffer(
            "vae_latents_mean",
            torch.tensor(latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "vae_latents_inv_std",
            (1.0 / torch.tensor(latents_std, dtype=torch.float32)).view(1, -1, 1, 1, 1),
            persistent=False,
        )

        # ---- Transformer (CausalWan DiT) + action head ----
        self.transformer = None
        if spec.dit:
            # Filter out keys not accepted by `CausalWanModel.__init__`.
            transformer_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in ("_convert_", "_target_")}
            transformer_kwargs["action_dim"] = ah_config["action_dim"]
            transformer_kwargs["max_state_dim"] = ah_config["max_state_dim"]
            transformer_kwargs["num_frame_per_block"] = ah_config["num_frame_per_block"]
            self.transformer = CausalWanModel(**transformer_kwargs)

        # ---- Diffusion scheduler ----
        self.scheduler = None
        if spec.scheduler:
            self.scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False,
            )

        # Config-derived scalars the encode phase needs WITHOUT the DiT built
        # (the encode stage skips constructing self.transformer). These mirror
        # CausalWanModel's own derivations so the encode stage produces the same
        # action-noise shape and reset decision as the monolithic path.
        self._action_dim: int = ah_config["action_dim"]
        _max_chunk_size = int(diffusion_model_cfg.get("max_chunk_size", -1))
        _nfpb = int(ah_config["num_frame_per_block"])
        self._local_attn_size: int = _max_chunk_size * _nfpb + 1 if _max_chunk_size != -1 else -1

        self._states: OrderedDict[str, DreamZeroState] = OrderedDict()
        self._max_session_states = MAX_DREAMZERO_SESSIONS
        self.state = self._get_or_create_state("default")

        # DiT step cache is configured by StepCacheBackend
        # (cache_backend="step_cache") via pipeline._stepcache_config.

        # Keep runtime inference settings separate from the training-time config.
        self.num_inference_steps: int = model_config.get(
            "num_inference_steps",
            DEFAULT_NUM_INFERENCE_STEPS,
        )
        self.cfg_scale: float = model_config.get("cfg_scale", DEFAULT_CFG_SCALE)
        self.sigma_shift: float = model_config.get("sigma_shift", DEFAULT_SIGMA_SHIFT)
        self.num_frames: int = ah_config["num_frames"]
        self.num_frame_per_block: int = ah_config["num_frame_per_block"]
        self.action_horizon: int = ah_config["action_horizon"]

        self.decouple_inference_noise: bool = ah_config["decouple_inference_noise"]
        self.video_inference_final_noise: float = ah_config["video_inference_final_noise"]

        self.seed: int = model_config.get("seed", DEFAULT_SEED)

        # Model-level constants for state/action padding.
        self.max_state_dim: int = ah_config["max_state_dim"]
        self.max_action_dim: int = ah_config["max_action_dim"]

        self.negative_prompt: str = model_config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
        # The negative prompt is a model constant: encode it once, lazily, and
        # reuse across every forward/session (UMT5 encode is deterministic).
        self._negative_prompt_embeds_cache: torch.Tensor | None = None

        # Embodiment name -> numeric ID mapping (model knowledge)
        self.embodiment_name_to_id: dict[str, int] = model_config.get(
            "embodiment_name_to_id",
            DEFAULT_EMBODIMENT_NAME_TO_ID,
        )

        # Prefer root `experiment_cfg/metadata.json`, then `model_config`.
        stats_path = model_config.get("action_norm_stats_path")
        metadata = self._load_repo_json(model_path, "experiment_cfg/metadata.json", local_files_only)
        if metadata is not None:
            self.action_norm_stats = self._parse_action_norm_stats(metadata)
            self.state_norm_stats = self._parse_state_norm_stats(metadata)
        elif stats_path:
            self.action_norm_stats = self._load_action_norm_stats(stats_path)
            self.state_norm_stats = {}
        else:
            self.action_norm_stats: dict[str, dict[str, torch.Tensor]] = {}
            self.state_norm_stats: dict[str, dict[str, torch.Tensor]] = {}

        # Whether model uses relative actions (need to add back last state)
        self.relative_action: bool = model_config.get("relative_action", True)
        # Number of action dims that are relative (DROID: 7 = joint only, gripper is absolute)
        self.relative_action_dim: int = model_config.get("relative_action_dim", 7)

        self._weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=None,
                revision=None,
                prefix="",
                fall_back_to_pt=False,
                allow_patterns_overrides=[
                    "model-*.safetensors",
                    "model.safetensors",
                ],
            ),
        ]

    def _get_or_create_state(self, session_id: str | None) -> DreamZeroState:
        session_key = str(session_id or "default")
        state = self._states.get(session_key)
        if state is None:
            state = DreamZeroState()
            self._states[session_key] = state
            max_states = getattr(self, "_max_session_states", MAX_DREAMZERO_SESSIONS)
            while len(self._states) > max_states:
                self._states.popitem(last=False)
        else:
            self._states.move_to_end(session_key)
        return state

    # -----------------------------------------------------------------------
    # Root config loading
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_repo_json(model_path: str, relative_path: str, local_files_only: bool) -> dict | None:
        """Load a JSON file from a local checkpoint directory or HF repo."""
        if local_files_only and os.path.isdir(model_path):
            json_path = os.path.join(model_path, relative_path)
            if not os.path.exists(json_path):
                return None
            with open(json_path) as f:
                return json.load(f)

        try:
            json_path = hf_hub_download(model_path, relative_path)
            with open(json_path) as f:
                return json.load(f)
        except Exception:
            logger.warning("Failed to load %s from %s", relative_path, model_path)
            return None

    # -----------------------------------------------------------------------
    # CFGParallelMixin overrides
    # -----------------------------------------------------------------------

    def predict_noise(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Call CausalWanModel, return (video_pred, action_pred)."""
        video_pred, action_pred = self._predict_noise_eager(kwargs)

        if is_stepcache_active(self):
            video_pred = video_pred.clone()
            if action_pred is not None:
                action_pred = action_pred.clone()

        if action_pred is None:
            batch_size = kwargs["hidden_states"].shape[0]
            action_pred = torch.empty(
                batch_size,
                0,
                self.transformer.action_dim,
                device=video_pred.device,
                dtype=video_pred.dtype,
            )
        return (video_pred, action_pred)

    def _predict_noise_eager(self, kwargs: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Eager DiT forward; also handles KV-cache write-back on prefill."""
        self._cudagraph_mark_step_begin()
        video_pred, action_pred, updated_kv_caches = self.transformer(
            x=kwargs["hidden_states"],
            timestep=kwargs["timestep_video"],
            context=kwargs["encoder_hidden_states"],
            seq_len=kwargs["seq_len"],
            kv_cache=kwargs["kv_cache"],
            crossattn_cache=kwargs["crossattn_cache"],
            current_start_frame=kwargs["current_start_frame"],
            y=kwargs.get("y"),
            clip_feature=kwargs.get("clip_feature"),
            action=kwargs.get("action"),
            timestep_action=kwargs.get("timestep_action"),
            state=kwargs.get("state_features"),
            embodiment_id=kwargs.get("embodiment_id"),
        )
        if kwargs.get("update_kv_cache", False):
            is_neg = kwargs.get("is_negative", False)
            logger.debug(
                "AR-Diffusion pipeline predict_noise -> commit paged context: "
                "is_neg=%s seq_len=%s current_start_frame=%s layers=%d",
                is_neg,
                kwargs.get("seq_len"),
                kwargs.get("current_start_frame"),
                len(updated_kv_caches),
            )
            self._kv_commit(is_neg)

        return video_pred, action_pred

    # -----------------------------------------------------------------------
    # torch.compile setup (paper D.2 extended: encoders + VAE + DiT blocks)
    # -----------------------------------------------------------------------

    def setup_compile(self) -> None:
        """Compile DreamZero encoders, VAE, and per-block DiT for inference.

        Paper D.2 uses ``mode=reduce-overhead``, ``fullgraph=True``, ``dynamic=False``
        on text/image/VAE and DiT. VAE decode uses a tensor feat_cache patch and
        compiles ``decoder.forward`` (not ``_decode``, which has a Python frame loop).
        Incremental VAE encode (``_vae_encode_encoder_chunk``) stays eager because
        Wan ``feat_cache`` mutation is incompatible with CUDAGraph capture.
        DiT blocks use per-block ``fullgraph=True``.
        """
        if not torch.cuda.is_available():
            logger.info("DreamZero setup_compile skipped: CUDA not available.")
            return

        from vllm_omni.diffusion.models.dreamzero.wan_vae_feat_cache_patch import (
            apply_wan_vae_feat_cache_tensor_patch,
        )

        apply_wan_vae_feat_cache_tensor_patch()

        compile_ro = {"mode": "reduce-overhead", "fullgraph": True, "dynamic": False}
        # DiT blocks: default avoids CUDAGraph overwrite on modulation tensors; encoders use reduce-overhead.
        # The AR-Diffusion paged self-attention is a registered custom op, so the
        # block stays fullgraph even on that path.
        dit_compile = {"mode": "default", "fullgraph": True, "dynamic": False}

        logger.info(
            "DreamZero: torch.compile text/image/VAE encode + per-block DiT (encoders reduce-overhead, DiT default)."
        )

        if self.text_encoder is not None:
            try:
                self.text_encoder.forward = torch.compile(self.text_encoder.forward, **compile_ro)
            except Exception as exc:
                logger.warning("DreamZero: text_encoder compile failed (%s); skipping.", exc)

        if self.image_encoder is not None:
            try:
                self.image_encoder.model.visual.forward = torch.compile(
                    self.image_encoder.model.visual.forward,
                    **compile_ro,
                )
            except Exception as exc:
                logger.warning("DreamZero: image_encoder compile failed (%s); skipping.", exc)

        if self.vae is not None:
            try:
                self.vae._encode = torch.compile(self.vae._encode, **compile_ro)
            except Exception as exc:
                logger.warning("DreamZero: vae._encode compile failed (%s); skipping.", exc)

        if self.transformer is None:
            self.warmup_compile()
            return

        compiled_blocks = 0
        for block in self.transformer.blocks:
            try:
                block.forward = torch.compile(block.forward, **dit_compile)
                compiled_blocks += 1
            except Exception as exc:
                logger.warning(
                    "DreamZero: transformer block %d compile failed (%s); leaving remaining eager.",
                    compiled_blocks,
                    exc,
                )
                break
        if compiled_blocks:
            logger.info(
                "DreamZero: compiled %d/%d transformer blocks.",
                compiled_blocks,
                len(self.transformer.blocks),
            )

        self.warmup_compile()

    def warmup_compile(self) -> None:
        """Warm up compiled text/image/VAE paths before timed inference."""
        if not torch.cuda.is_available():
            return

        device = next(self.text_encoder.parameters()).device
        with torch.inference_mode():
            try:
                text_tokens = torch.zeros(1, 16, dtype=torch.long, device=device)
                attention_mask = torch.ones_like(text_tokens)
                self._encode_text(text_tokens, attention_mask)
            except Exception as exc:
                logger.warning("DreamZero compile warmup (text_encoder) skipped: %s", exc)

            try:
                image = torch.zeros(1, 1, 3, 180, 320, dtype=torch.bfloat16, device=device)
                self._encode_image(image, self.num_frames, 180, 320, state=self.state)
            except Exception as exc:
                logger.warning("DreamZero compile warmup (image_encoder) skipped: %s", exc)

            try:
                self.state.reset_vae_encoder_stream()
                dummy_video = torch.zeros(1, 3, 1, 180, 320, dtype=torch.bfloat16, device=device)
                self._vae_stream_seed(self.state, dummy_video[:, :, :1])
                for _ in range(4):
                    self._vae_stream_append_frame(self.state, dummy_video[:, :, :1])
                self._vae_stream_get_observation_latents(
                    self.state,
                    self.num_frame_per_block,
                    dtype=torch.bfloat16,
                )
            except Exception as exc:
                logger.warning("DreamZero compile warmup (vae encode stream) skipped: %s", exc)

            try:
                latent_h, latent_w = 180 // 8, 320 // 8
                dummy_latent = torch.zeros(
                    1,
                    16,
                    self.num_frame_per_block,
                    latent_h * 2,
                    latent_w * 2,
                    dtype=self.vae.dtype,
                    device=device,
                )
                mean, inv_std = self._vae_latents_mean_inv_std(device=device, dtype=self.vae.dtype)
                dummy_latent_denorm = dummy_latent / inv_std + mean
                self._cudagraph_mark_step_begin()
                self.vae.decode(dummy_latent_denorm, return_dict=False)
            except Exception as exc:
                logger.warning("DreamZero compile warmup (vae decode) skipped: %s", exc)

        torch.accelerator.synchronize(device)
        logger.info("DreamZero compile warmup finished (text / image / vae decode).")

    def combine_cfg_noise(
        self,
        positive_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        negative_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        true_cfg_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Video: standard CFG. Action: positive only (no CFG).
        action = cond only (no uncond blending)
        """
        (video_pos, action_pos) = positive_noise_pred
        (video_neg, _) = negative_noise_pred
        video_combined = super().combine_cfg_noise(video_pos, video_neg, true_cfg_scale, cfg_normalize)
        return (video_combined, action_pos)

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------

    def _synchronize_cfg_parallel_step_output(
        self,
        latents: tuple[torch.Tensor, torch.Tensor],
        do_true_cfg: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Post-step sync: .contiguous() + cuda.synchronize()"""
        latents = tuple(t.contiguous() for t in latents)
        if do_true_cfg and get_classifier_free_guidance_world_size() > 1:
            device = next((t.device for t in latents if t.is_cuda), None)
            if device is not None:
                torch.cuda.current_stream(device).synchronize()
        return latents

    # -----------------------------------------------------------------------
    # Video preprocessing
    # -----------------------------------------------------------------------

    def _preprocess_video(self, videos: torch.Tensor) -> torch.Tensor:
        """uint8 [B,T,H,W,C] -> bfloat16 [B,C,T,H,W] normalized to [-1,1]."""
        videos = videos.permute(0, 4, 1, 2, 3)
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            # Cast to bf16 before normalization to preserve input rounding.
            videos = videos.to(dtype=torch.bfloat16)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)
            videos = videos.reshape(b * t, c, h, w)
            videos = videos * 2.0 - 1.0
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return videos.to(dtype=torch.bfloat16)

    @staticmethod
    def _cudagraph_mark_step_begin() -> None:
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Text encoding
    # -----------------------------------------------------------------------

    def _encode_text(self, text_tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text prompt via UMT5."""
        self._cudagraph_mark_step_begin()
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(
            text_tokens,
            attention_mask,
        ).last_hidden_state
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    # -----------------------------------------------------------------------
    # Image encoding
    # -----------------------------------------------------------------------

    def _encode_image(
        self,
        image: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        *,
        state: DreamZeroState | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode first frame via CLIP + VAE.
        Returns: (clip_feas, ys, image_latent)
        """
        device = image.device
        batch_size = image.shape[0]

        with torch.amp.autocast(dtype=torch.bfloat16, device_type=device.type):
            self._cudagraph_mark_step_begin()
            clip_context = self.image_encoder.encode_image(image)

            msk = torch.ones(batch_size, num_frames, height // 8, width // 8, device=device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(batch_size, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)

            latent_dtype = image.dtype
            image_input = image.transpose(1, 2)
            image_zeros = torch.zeros(
                batch_size,
                3,
                num_frames - 1,
                height,
                width,
                dtype=latent_dtype,
                device=device,
            )
            vae_input = torch.concat([image_input, image_zeros], dim=2)
            y = self._encode_vae_latents(vae_input)
            y = y.to(dtype=latent_dtype)

            new_image = y[:, :, 0:1]
            y = torch.concat([msk, y], dim=1)

            if state is not None:
                if not state.vae_stream_initialized:
                    # Seed the AR streaming encoder after the compiled full encode above;
                    # ``cudagraph_mark_step_begin`` isolates eager feat_cache work from CUDAGraph.
                    self._vae_stream_seed(state, image_input[:, :, :1])
                else:
                    # Window ("inference") restart with a live stream: keep the real
                    # frame history in the Wan feat_cache and append the restart
                    # observation instead of reseeding from scratch.
                    self._cudagraph_mark_step_begin()
                    self._vae_stream_append_frame(state, image_input[:, :, :1])

        return clip_context, y, new_image

    def _vae_patchify(self, videos: torch.Tensor) -> torch.Tensor:
        if self.vae.config.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import patchify

            return patchify(videos, patch_size=self.vae.config.patch_size)
        return videos

    @staticmethod
    def _vae_clone_feat_map(feat_map: list[torch.Tensor | None]) -> list[torch.Tensor | None]:
        return [entry.clone() if isinstance(entry, torch.Tensor) else entry for entry in feat_map]

    def _vae_init_enc_feat_map(self) -> list[torch.Tensor | None]:
        self.vae.clear_cache()
        return self._vae_clone_feat_map(self.vae._enc_feat_map)

    def _vae_encode_encoder_chunk(
        self,
        chunk: torch.Tensor,
        feat_map: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        """Run one Wan encoder chunk while mutating ``feat_map`` in place.

        Must stay eager (not ``torch.compile``): Wan causal ``feat_cache`` updates
        conflict with ``reduce-overhead`` CUDAGraph capture.
        """
        self.vae._enc_feat_map = feat_map
        self.vae._enc_conv_idx = [0]
        out = self.vae.encoder(
            chunk,
            feat_cache=self.vae._enc_feat_map,
            feat_idx=self.vae._enc_conv_idx,
        )
        return out, self._vae_clone_feat_map(self.vae._enc_feat_map)

    def _vae_latents_mean_inv_std(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (mean, inv_std) VAE-latent normalization buffers on a target device/dtype.

        Single accessor for the ``vae_latents_mean`` / ``vae_latents_inv_std``
        registered buffers so the (de)normalization sites don't each hand-cast
        both buffers.
        """
        return (
            self.vae_latents_mean.to(device=device, dtype=dtype),
            self.vae_latents_inv_std.to(device=device, dtype=dtype),
        )

    def _vae_quantize_encoder_out(self, encoder_out: torch.Tensor) -> torch.Tensor:
        enc = self.vae.quant_conv(encoder_out)
        mu, _ = enc.chunk(2, dim=1)
        mean, inv_std = self._vae_latents_mean_inv_std(device=mu.device, dtype=mu.dtype)
        return (mu - mean) * inv_std

    def _vae_stream_seed(self, state: DreamZeroState, first_frame: torch.Tensor) -> None:
        """Seed incremental VAE encode with the first observation frame."""
        state.reset_vae_encoder_stream()
        feat_map = self._vae_init_enc_feat_map()
        chunk = self._vae_patchify(first_frame.to(dtype=self.vae.dtype))
        self._cudagraph_mark_step_begin()
        encoder_out, feat_map = self._vae_encode_encoder_chunk(chunk[:, :, :1], feat_map)
        state.vae_enc_feat_map = feat_map
        state.vae_encoder_out = encoder_out
        state.vae_pending_body_frames = None
        state.vae_stream_initialized = True

    def _vae_stream_append_frame(self, state: DreamZeroState, new_frame: torch.Tensor) -> None:
        """Append one pixel frame and encode a 4-frame body chunk when ready."""
        if not state.vae_stream_initialized or state.vae_enc_feat_map is None or state.vae_encoder_out is None:
            raise RuntimeError("VAE encoder stream is not initialized.")

        frame = new_frame.to(dtype=self.vae.dtype)
        if state.vae_pending_body_frames is None:
            state.vae_pending_body_frames = frame
        else:
            state.vae_pending_body_frames = torch.cat([state.vae_pending_body_frames, frame], dim=2)

        while state.vae_pending_body_frames is not None and state.vae_pending_body_frames.shape[2] >= 4:
            body = state.vae_pending_body_frames[:, :, :4]
            if state.vae_pending_body_frames.shape[2] > 4:
                state.vae_pending_body_frames = state.vae_pending_body_frames[:, :, 4:]
            else:
                state.vae_pending_body_frames = None
            chunk = self._vae_patchify(body)
            encoder_chunk, feat_map = self._vae_encode_encoder_chunk(chunk, state.vae_enc_feat_map)
            state.vae_encoder_out = torch.cat([state.vae_encoder_out, encoder_chunk], dim=2)
            state.vae_enc_feat_map = feat_map

    def _vae_stream_get_observation_latents(
        self,
        state: DreamZeroState,
        num_latent_frames: int,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if state.vae_encoder_out is None:
            raise RuntimeError("VAE encoder stream has no accumulated encoder output.")
        latents = self._vae_quantize_encoder_out(state.vae_encoder_out).to(dtype=dtype)
        if latents.shape[2] >= num_latent_frames:
            return latents[:, :, -num_latent_frames:]
        pad_count = num_latent_frames - latents.shape[2]
        pad = latents[:, :, -1:].expand(-1, -1, pad_count, -1, -1)
        return torch.cat([pad, latents], dim=2)

    def _preprocess_vae_observation_window(self, videos: torch.Tensor) -> torch.Tensor:
        _, _, num_frames_raw, _, _ = videos.shape
        if (num_frames_raw - 1) // 4 == self.num_frame_per_block:
            return videos
        if num_frames_raw // 4 != self.num_frame_per_block:
            repeat_factor = self.num_frame_per_block // (num_frames_raw // 4)
            videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
            first_frame = videos[:, :, 0:1]
            return torch.cat([first_frame, videos], dim=2)
        first_frame = videos[:, :, 0:1]
        return torch.cat([first_frame, videos], dim=2)

    def _encode_observation_latents(
        self,
        state: DreamZeroState,
        videos: torch.Tensor,
        *,
        latent_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode current robot observation into normalized VAE latents."""
        if state.vae_stream_initialized:
            self._cudagraph_mark_step_begin()
            self._vae_stream_append_frame(state, videos[:, :, -1:])
            return self._vae_stream_get_observation_latents(
                state,
                self.num_frame_per_block,
                dtype=latent_dtype,
            )

        videos = self._preprocess_vae_observation_window(videos)
        return self._encode_vae_latents(videos).to(dtype=latent_dtype)

    def _encode_vae_latents(self, videos: torch.Tensor) -> torch.Tensor:
        """Encode videos into normalized VAE latents."""
        input_dtype = videos.dtype
        self._cudagraph_mark_step_begin()
        hidden = self.vae._encode(videos.to(dtype=self.vae.dtype))
        mu, _ = hidden.chunk(2, dim=1)
        mean, inv_std = self._vae_latents_mean_inv_std(device=mu.device, dtype=mu.dtype)
        mu = (mu - mean) * inv_std
        return mu.to(dtype=input_dtype)

    def decode_video_latents(self, video_latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized VAE latents into RGB video tensors."""
        vae_dtype = self.vae.dtype
        vae_device = next(self.vae.parameters()).device
        latents = video_latents.to(device=vae_device, dtype=vae_dtype)
        mean, inv_std = self._vae_latents_mean_inv_std(device=vae_device, dtype=vae_dtype)
        latents = latents / inv_std + mean
        with torch.no_grad():
            self._cudagraph_mark_step_begin()
            return self.vae.decode(latents, return_dict=False)[0]

    def decode_accumulated_video_latents(self, session_id: str | None = None) -> torch.Tensor:
        """Decode all AR-chunk latents accumulated for ``session_id``."""
        state = self._get_or_create_state(session_id)
        latents = state.get_concatenated_video_latents()
        if latents is None:
            session_key = str(session_id or "default")
            raise RuntimeError(f"No accumulated video latents for session {session_key!r}.")
        return self.decode_video_latents(latents)

    def clear_accumulated_video_latents(self, session_id: str | None = None) -> None:
        """Clear accumulated video latents for ``session_id`` without resetting KV state."""
        state = self._get_or_create_state(session_id)
        state.clear_video_latents()

    # -----------------------------------------------------------------------
    # KV cache prefill
    # -----------------------------------------------------------------------

    def _prefill_kv_cache(
        self,
        image_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        frame_seqlen: int,
        seq_len: int,
        do_true_cfg: bool,
        state: DreamZeroState,
    ) -> None:
        """Prefill KV cache with first frame and/or current observation.

        Uses predict_noise_maybe_with_cfg() for CFG parallel -- same path as
        the denoise loop. The mixin handles rank dispatch automatically.
        KV cache update happens as a side effect inside predict_noise().
        """
        batch_size = image_latents.shape[0]
        device = image_latents.device
        dtype = image_latents.dtype
        num_heads = getattr(self.transformer.blocks[0].self_attn, "tp_num_heads", self.transformer.num_heads)
        head_dim = self.transformer.dim // self.transformer.num_heads

        if state.current_start_frame == 0:
            self._kv_create(
                state,
                batch_size,
                dtype,
                device,
                self.transformer.num_layers,
                num_heads,
                head_dim,
            )

            zero_t = torch.zeros([batch_size, 1], device=device, dtype=torch.long)
            y_first = state.ys[:, :, 0:1] if state.ys is not None else None

            # KV cache update is a side effect in predict_noise()
            common = dict(
                hidden_states=image_latents.transpose(1, 2),
                timestep_video=zero_t,
                seq_len=frame_seqlen,
                current_start_frame=0,
                y=y_first,
                clip_feature=state.clip_feas,
                update_kv_cache=True,
                dreamzero_state=state,
            )
            positive_kwargs = dict(
                encoder_hidden_states=prompt_embeds,
                kv_cache=self._kv_get(state, False, seq_len=frame_seqlen, update_kv_cache=True),
                crossattn_cache=self._kv_get_cross(state, False),
                is_negative=False,
                **common,
            )
            negative_kwargs = (
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    kv_cache=self._kv_get(state, True, seq_len=frame_seqlen, update_kv_cache=True),
                    crossattn_cache=self._kv_get_cross(state, True),
                    is_negative=True,
                    **common,
                )
                if negative_prompt_embeds is not None
                else None
            )

            self.predict_noise_maybe_with_cfg(
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                do_true_cfg=do_true_cfg,
                true_cfg_scale=self.cfg_scale,
                cfg_normalize=False,
            )
            state.current_start_frame = 1

        if state.current_start_frame != 1:
            csf = state.current_start_frame
            nfpb = self.num_frame_per_block
            current_ref = image_latents[:, -nfpb:]
            if state.ys is not None and csf <= state.ys.shape[2]:
                y = state.ys[:, :, csf - nfpb : csf]
            elif state.ys is not None:
                y = state.ys[:, :, -nfpb:]
            else:
                y = None

            zero_t = torch.zeros([batch_size, nfpb], device=device, dtype=torch.long)
            common = dict(
                hidden_states=current_ref.transpose(1, 2),
                timestep_video=zero_t,
                seq_len=seq_len,
                current_start_frame=csf - nfpb,
                y=y,
                clip_feature=state.clip_feas,
                update_kv_cache=True,
                dreamzero_state=state,
            )
            positive_kwargs = dict(
                encoder_hidden_states=prompt_embeds,
                kv_cache=self._kv_get(state, False, seq_len=seq_len, update_kv_cache=True),
                crossattn_cache=self._kv_get_cross(state, False),
                is_negative=False,
                **common,
            )
            negative_kwargs = (
                dict(
                    encoder_hidden_states=negative_prompt_embeds,
                    kv_cache=self._kv_get(state, True, seq_len=seq_len, update_kv_cache=True),
                    crossattn_cache=self._kv_get_cross(state, True),
                    is_negative=True,
                    **common,
                )
                if negative_prompt_embeds is not None
                else None
            )

            self.predict_noise_maybe_with_cfg(
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                do_true_cfg=do_true_cfg,
                true_cfg_scale=self.cfg_scale,
                cfg_normalize=False,
            )

    def _run_dit_loop(
        self,
        video_latents: torch.Tensor,
        action_latents: torch.Tensor,
        timesteps_video: torch.Tensor,
        timesteps_action: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        video_action_scheduler: VideoActionScheduler,
        do_true_cfg: bool,
        state: DreamZeroState,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denoising loop with CFG parallel support.

        Internal per-chunk DiT loop over video+action latents. Renamed from the
        former ``diffuse`` to avoid colliding with the ``DiffusionV2Atoms.diffuse``
        whole-request denoise atom (see :meth:`diffuse`).

        For each timestep:
          1. Build positive_kwargs / negative_kwargs
          2. predict_noise_maybe_with_cfg()    -> (video_pred, action_pred)
          3. scheduler_step_maybe_with_cfg()   -> VideoActionScheduler
          4. _synchronize_cfg_parallel_step_output()
        """
        seq_len = kwargs["seq_len"]
        state_features = kwargs.get("state_features")
        embodiment_id = kwargs.get("embodiment_id")

        # Shared kwargs for predict_noise (both cond & uncond branches)
        common_kwargs = dict(
            seq_len=seq_len,
            current_start_frame=state.current_start_frame,
            state_features=state_features,
            embodiment_id=embodiment_id,
            update_kv_cache=False,
            dreamzero_state=state,
        )

        noisy_input = video_latents
        noisy_input_action = action_latents
        # ---- step_cache (StepCacheBackend / dreamzero.git) ----
        _cached_flow_pred: torch.Tensor | None = None
        _cached_flow_pred_action: torch.Tensor | None = None
        _prev_predictions: list[tuple[torch.Tensor]] = []
        _step_cache = get_stepcache_state(self) if is_stepcache_active(self) else None

        for index in range(len(timesteps_video)):
            video_timestep = timesteps_video[index]
            action_timestep = timesteps_action[index]
            batch_size = noisy_input.shape[0]

            timestep = (
                torch.ones(
                    [batch_size, self.num_frame_per_block],
                    device=noisy_input.device,
                    dtype=torch.int64,
                )
                * video_timestep
            )
            timestep_action = (
                torch.ones(
                    [batch_size, self.action_horizon],
                    device=noisy_input.device,
                    dtype=torch.int64,
                )
                * action_timestep
            )

            csf = state.current_start_frame
            if csf + self.num_frame_per_block <= state.ys.shape[2]:
                y = state.ys[:, :, csf : csf + self.num_frame_per_block]
            else:
                y = state.ys[:, :, -self.num_frame_per_block :]

            run_dit = _step_cache is None or _step_cache.should_run_step(_prev_predictions)
            if run_dit:
                positive_kwargs = dict(
                    hidden_states=noisy_input.transpose(1, 2),
                    timestep_video=timestep,
                    encoder_hidden_states=prompt_embeds,
                    kv_cache=self._kv_get(state, False, seq_len=seq_len, update_kv_cache=False),
                    crossattn_cache=self._kv_get_cross(state, False),
                    y=y,
                    clip_feature=state.clip_feas,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    is_negative=False,
                    **common_kwargs,
                )

                if do_true_cfg and negative_prompt_embeds is not None:
                    negative_kwargs = dict(
                        hidden_states=noisy_input.transpose(1, 2),
                        timestep_video=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        kv_cache=self._kv_get(state, True, seq_len=seq_len, update_kv_cache=False),
                        crossattn_cache=self._kv_get_cross(state, True),
                        y=y,
                        clip_feature=state.clip_feas,
                        action=noisy_input_action,
                        timestep_action=timestep_action,
                        is_negative=True,
                        **common_kwargs,
                    )
                else:
                    negative_kwargs = None

                noise_pred = self.predict_noise_maybe_with_cfg(
                    positive_kwargs=positive_kwargs,
                    negative_kwargs=negative_kwargs,
                    do_true_cfg=do_true_cfg,
                    true_cfg_scale=self.cfg_scale,
                    cfg_normalize=False,
                )
                flow_pred, flow_pred_action = noise_pred
                _cached_flow_pred = flow_pred
                _cached_flow_pred_action = flow_pred_action

                _prev_predictions.append((flow_pred,))
                if _step_cache is not None:
                    _step_cache.trim_history(_prev_predictions)
            else:
                # Reuse previous prediction (DiT step-skipping cache).
                assert _cached_flow_pred is not None, "DiT cache: no cached prediction available for step skip."
                flow_pred = _cached_flow_pred
                flow_pred_action = _cached_flow_pred_action

            latents = (noisy_input, noisy_input_action)
            t = (video_timestep, action_timestep)
            noise_pred_tuple = (flow_pred.transpose(1, 2), flow_pred_action)
            step_output = video_action_scheduler.step(
                noise_pred_tuple,
                t,
                latents,
                generator=kwargs.get("generator"),
            )
            noisy_input, noisy_input_action = step_output[0]

            noisy_input, noisy_input_action = self._synchronize_cfg_parallel_step_output(
                (noisy_input, noisy_input_action),
                do_true_cfg,
            )

        return noisy_input, noisy_input_action

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def _transform_robot_obs(self, robot_obs: dict):
        """Select DreamZero robot transform and convert raw obs to model input."""
        embodiment = robot_obs.get("embodiment", self.default_robot_embodiment)
        transform = get_transform(embodiment)
        return transform, transform.transform_input(robot_obs)

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch, **kwargs) -> DiffusionOutput:
        """Full inference step. Called by DiffusionEngine.step().

        Golden reference path (RFC #4590 §3): composes the SAME three phase
        methods the disaggregated encode/denoise/decode stages call, on one
        shared session state and one carrier, in one process. Because the atoms
        reuse these exact methods, the disaggregated path is numerically
        equivalent to this monolithic forward by construction.
        """
        extra_args = req.sampling_params.extra_args or {}
        robot_obs = extra_args.get("robot_obs")
        if robot_obs is None:
            first_prompt = req.prompts[0] if req.prompts else ""
            prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
            is_dummy_warmup = prompt == "dummy run" and req.sampling_params.num_inference_steps == 1
            if is_dummy_warmup:
                logger.info("Skipping DreamZero dummy warmup request without robot_obs.")
                return DiffusionOutput(
                    output={
                        "actions": np.zeros(
                            (self.action_horizon, self.max_action_dim),
                            dtype=np.float32,
                        ),
                    },
                )
            raise KeyError("robot_obs")
        session_id = str(extra_args.get("session_id") or "default")
        state = self._get_or_create_state(session_id)
        self.state = state

        # Optional phase timing (DZ_PHASE_TIMING=1): logs per-forward stage costs
        # (text encode / obs VAE encode / KV prefill / denoise) at INFO. Each mark
        # synchronizes CUDA, so leave it off for timed benchmark runs.
        _pt = None
        if os.environ.get("DZ_PHASE_TIMING"):
            import time as _time

            torch.accelerator.synchronize()
            _pt = {"time": _time, "t0": _time.perf_counter(), "marks": []}

        def _pt_mark(name: str) -> None:
            if _pt is not None:
                torch.accelerator.synchronize()
                _pt["marks"].append((name, _pt["time"].perf_counter()))

        # ---- Phase 1: encode (conditions + latents/timesteps params) ----
        carrier = self._run_encode_phase(robot_obs, state, session_id, explicit_reset=extra_args.get("reset", False))
        _pt_mark("text_encode")
        _pt_mark("obs_vae_encode")

        # ---- Phase 2: denoise (KV prefill + DiT loop; owns AR-Diffusion KV) ----
        self._run_denoise_phase(carrier, state)
        _pt_mark("prefill_kv")
        if _pt is not None:
            _pt_mark("diffuse")
            prev_t = _pt["t0"]
            parts = []
            for name, t in _pt["marks"]:
                parts.append(f"{name}={1000 * (t - prev_t):.1f}ms")
                prev_t = t
            logger.info(
                "DZ_PHASE_TIMING csf=%s total=%.1fms %s",
                state.current_start_frame,
                1000 * (prev_t - _pt["t0"]),
                " ".join(parts),
            )

        # ---- Phase 3: decode (action denorm + video export) ----
        return self._run_decode_phase(carrier)

    # -----------------------------------------------------------------------
    # Disaggregated phase methods (RFC #4590). forward() composes all three;
    # the disaggregated stages call them one per worker. They share the SAME
    # math helpers (_encode_text / _encode_image / _prefill_kv_cache / diffuse /
    # _denormalize_action / decode) — no duplication of the numerical path.
    # -----------------------------------------------------------------------

    def _run_encode_phase(
        self,
        robot_obs: dict,
        state: DreamZeroState,
        session_id: str,
        *,
        explicit_reset: bool = False,
    ) -> DreamZeroStageCarrier:
        """Encode conditions + prepare initial latents/timestep params.

        Runs on the encode stage (and as forward()'s first phase). Produces every
        stable condition the denoise stage consumes, plus the deterministic
        initial noise, WITHOUT touching the AR-Diffusion KV pool or the DiT. The
        engine-KV reset is DECIDED here (from the encode-session state) and
        recorded on the carrier so the denoise worker can apply the same reset.
        """
        transform, unified_obs = self._transform_robot_obs(robot_obs)
        # Capture the exact key get_transform() was selected by, so the decode
        # phase reselects the identical transform even across a process boundary.
        transform_embodiment = robot_obs.get("embodiment", self.default_robot_embodiment)
        device = get_local_device()

        # ---- Step 1: Extract inputs from unified observation ----
        prompt_str = unified_obs["prompt"]  # str (templated)
        stitched = unified_obs["images"]  # ndarray (T,H,W,C) from transform
        if not isinstance(stitched, np.ndarray):
            stitched = np.asarray(stitched)
        embodiment_name = unified_obs["embodiment_name"]
        embodiment_id = torch.tensor(  # (B,) tensor for CategorySpecificMLP
            [self.embodiment_name_to_id[embodiment_name]],
            dtype=torch.long,
            device=device,
        )

        # State: raw from transform -> pad to (B, state_horizon=1, max_state_dim)
        raw_state = unified_obs["state"]
        state_for_postprocess = None
        if raw_state is not None:
            if not isinstance(raw_state, np.ndarray):
                raw_state = np.asarray(raw_state, dtype=np.float64)
            raw_state = raw_state.flatten()
            padded = np.zeros(self.max_state_dim, dtype=np.float64)
            n = min(len(raw_state), self.max_state_dim)
            padded[:n] = raw_state[:n]
            state_for_postprocess = (
                torch.from_numpy(padded)
                .reshape(1, 1, self.max_state_dim)
                .to(
                    device=device,
                    dtype=torch.float32,
                )
            )
            state_features = self._normalize_state(
                state_for_postprocess,
                embodiment_name,
            ).to(dtype=torch.bfloat16)
        else:
            state_features = None

        # ---- Step 1b: Tokenize ---- (wan2_2 convention: pipeline owns tokenizer)
        text_inputs = self.tokenizer(
            prompt_str,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        text_tokens = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Decide reset (see original forward). The reset is APPLIED to model-local
        # state here; the engine-KV window reset is applied on the denoise worker
        # (which owns the pool) via the reason recorded on the carrier.
        if explicit_reset:
            reset_reason = "session"
        else:
            reset_reason = state.reset_reason(text_tokens, 0, self._local_attn_size)
        self._apply_model_local_reset(state, reset_reason)
        state.language = text_tokens

        # Frame accumulation: stitched single frame -> multi-frame video
        video_frames = state.accumulate_frames(stitched)  # (T, H, W, C)
        videos = torch.from_numpy(video_frames).unsqueeze(0).to(device)  # (B=1, T, H, W, C)

        videos = self._preprocess_video(videos)  # -> [B,C,T,H,W] bf16
        _, _, num_frames_raw, height, width = videos.shape

        # Prompt embeds are constant within a session (a prompt change triggers a
        # "session" reset above, which clears this cache alongside state.language).
        if state.prompt_embeds is None:
            state.prompt_embeds = self._encode_text(text_tokens, attention_mask)
        prompt_embeds = state.prompt_embeds
        # Negative prompt for CFG uncond branch (model constant)
        negative_prompt_embeds = None
        if self.cfg_scale > 1.0:
            if self._negative_prompt_embeds_cache is None:
                neg_inputs = self.tokenizer(
                    self.negative_prompt,
                    max_length=512,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                self._negative_prompt_embeds_cache = self._encode_text(
                    neg_inputs["input_ids"].to(device),
                    neg_inputs["attention_mask"].to(device),
                )
            negative_prompt_embeds = self._negative_prompt_embeds_cache

        # Extract first/last frame for CLIP + VAE encoding
        if num_frames_raw == 4 or num_frames_raw == 9:
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if state.current_start_frame == 0:
            clip_feas, ys, image = self._encode_image(
                image,
                self.num_frames,
                height,
                width,
                state=state,
            )
            state.clip_feas = clip_feas.to(dtype=image.dtype)
            state.ys = ys.to(dtype=image.dtype)

        if state.current_start_frame != 0:
            latent_dtype = videos.dtype
            with torch.no_grad():
                image = self._encode_observation_latents(state, videos, latent_dtype=latent_dtype)

        batch_size = image.shape[0]
        generator = torch.Generator(device=device).manual_seed(self.seed)
        noise_obs = torch.randn(
            batch_size,
            16,
            self.num_frame_per_block,
            height // 8,
            width // 8,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        generator = torch.Generator(device=device).manual_seed(self.seed)
        noise_action = torch.randn(
            batch_size,
            self.action_horizon,
            self._action_dim,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )

        _, num_channels, num_frames, h_latent, w_latent = noise_obs.shape
        frame_seqlen = int(h_latent * w_latent / 4)
        seq_len = frame_seqlen * num_frames

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        do_true_cfg = self.cfg_scale > 1.0 and negative_prompt_embeds is not None

        return DreamZeroStageCarrier(
            session_id=session_id,
            embodiment_name=embodiment_name,
            transform_embodiment=str(transform_embodiment),
            reset_reason=reset_reason,
            explicit_reset=explicit_reset,
            do_true_cfg=do_true_cfg,
            current_start_frame=state.current_start_frame,
            height=height,
            width=width,
            seq_len=seq_len,
            frame_seqlen=frame_seqlen,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            clip_feas=state.clip_feas,
            ys=state.ys,
            image_latent=image,
            state_features=state_features,
            embodiment_id=embodiment_id,
            noise_obs=noise_obs,
            noise_action=noise_action,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            state_for_postprocess=state_for_postprocess,
        )

    def _apply_model_local_reset(self, state: DreamZeroState, reset_reason: str | None) -> None:
        """Apply the model-local + engine-KV reset for the decided reason.

        On the monolithic and denoise workers ``_ar_diffusion_kv_state`` is set,
        so this drops the pool window too (parity with the original forward). On
        the encode worker (no KV state attached) only the model-local reset runs;
        the engine reset is a no-op there because encode owns no pool.
        """
        if reset_reason == "session":
            self._kv_reset(state)
        elif reset_reason == "inference":
            self._kv_reset(state, clear_video_latents=False)

    def _run_denoise_phase(self, carrier: DreamZeroStageCarrier, state: DreamZeroState) -> None:
        """KV prefill + DiT denoise loop. Owns the AR-Diffusion KV/session state.

        Consumes the encode carrier's stable conditions read-only, populates the
        cross-attn KV, prefills, runs ``diffuse``, and writes ``video_out`` /
        ``action_out`` back onto the carrier. This is the ONLY phase that touches
        the DiT or the AR-Diffusion pool (RFC §9.1).
        """
        prompt_embeds = carrier.prompt_embeds
        negative_prompt_embeds = carrier.negative_prompt_embeds

        # Eager cross-attn population (AR-Diffusion only): cache text + image-token
        # K/V into the pool now that conditions are available. Runs on the first
        # forward of a session and after each window-boundary reset (csf == 0).
        if carrier.current_start_frame == 0:
            self._kv_populate_cross(prompt_embeds, carrier.clip_feas, is_negative=False)
            if negative_prompt_embeds is not None:
                self._kv_populate_cross(negative_prompt_embeds, carrier.clip_feas, is_negative=True)

        device = get_local_device()
        image = carrier.image_latent
        self._prefill_kv_cache(
            image,
            prompt_embeds,
            negative_prompt_embeds,
            carrier.frame_seqlen,
            carrier.seq_len,
            carrier.do_true_cfg,
            state,
        )

        sample_scheduler = copy.deepcopy(self.scheduler)
        sample_scheduler_action = copy.deepcopy(self.scheduler)
        sample_scheduler.set_timesteps(
            carrier.num_inference_steps,
            device=device,
            shift=carrier.sigma_shift,
        )
        sample_scheduler_action.set_timesteps(
            carrier.num_inference_steps,
            device=device,
            shift=carrier.sigma_shift,
        )

        if self.decouple_inference_noise:
            video_final_noise = self.video_inference_final_noise
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = (
                sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
            )
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)

        video_action_scheduler = VideoActionScheduler(
            sample_scheduler,
            sample_scheduler_action,
        )

        video_out, action_out = self._run_dit_loop(
            video_latents=carrier.noise_obs,
            action_latents=carrier.noise_action,
            timesteps_video=sample_scheduler.timesteps,
            timesteps_action=sample_scheduler_action.timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            video_action_scheduler=video_action_scheduler,
            do_true_cfg=carrier.do_true_cfg,
            state=state,
            seq_len=carrier.seq_len,
            state_features=carrier.state_features,
            embodiment_id=carrier.embodiment_id,
        )

        if state.current_start_frame == 1:
            video_out = torch.cat([image, video_out], dim=1)
        state.current_start_frame += self.num_frame_per_block

        state.append_video_latents(video_out)

        carrier.video_out = video_out
        carrier.action_out = action_out

    def _run_decode_phase(self, carrier: DreamZeroStageCarrier) -> DiffusionOutput:
        """Action denormalization + video export. Owns no DiT / KV / scheduler."""
        # Reselect the identical transform the encode phase used (by the same key
        # passed to get_transform), not by the derived embodiment_name.
        transform = get_transform(carrier.transform_embodiment or carrier.embodiment_name)
        video_out = carrier.video_out
        action_out = carrier.action_out

        # q99 denorm: [-1,1] → real values
        action_out = self._denormalize_action(action_out.float(), carrier.embodiment_name)

        # Relative -> absolute: only for relative_action_keys (joint_position only)
        # gripper_position is NOT relative, so don't add state back to it
        if self.relative_action and carrier.state_for_postprocess is not None:
            n_relative = self.relative_action_dim  # 7 for DROID (joint only)
            # Use original state precision for post-denorm absolute recovery.
            # Upstream adds obs state after `eval_transform.unapply()`
            # the bf16 denoising path.
            last_state = carrier.state_for_postprocess[:, 0, :n_relative]  # (B, n_relative)
            action_out[..., :n_relative] = (
                action_out[..., :n_relative] + last_state.unsqueeze(1)  # broadcast over horizon
            )

        # Squeeze batch dim for output: (B, horizon, dim) -> (horizon, dim)
        actions_np = action_out.squeeze(0).float().cpu().numpy()  # (horizon, max_action_dim)
        actions_np = transform.transform_action_output(actions_np)

        return DiffusionOutput(
            output={
                "actions": actions_np,
                # Source `video_pred` is normalized VAE latent output, not RGB.
                # Use `decode_video_latents()` for DreamZero-equivalent debug
                # video decoding.
                "video": video_out.transpose(1, 2).cpu(),
            },
        )

    # -----------------------------------------------------------------------
    # Disaggregated diffusion protocol (DiffusionV2Atoms — #4948 contract)
    # -----------------------------------------------------------------------

    #: DreamZero supports the three-stage encode/denoise/decode topology. The
    #: runner selects the path from ``od_config.model_stage``; the pipeline
    #: marshals its private carrier through :class:`StagePayload` here. This flag
    #: (plus the DiffusionV2Atoms method surface) is what
    #: ``supports_disaggregated_execution(pipeline)`` gates on.
    supports_disaggregated_execution: bool = True

    #: DreamZero runs only as a request-mode / disaggregated pipeline (it drives
    #: the whole-request ``diffuse`` atom with model-owned AR-Diffusion KV, not
    #: the single-process step loop). Kept ``False`` so the worker never routes
    #: it to the single-process DiffusionModelRunnerV2 step runner.
    supports_step_execution: bool = False

    #: Key under which the DreamZero carrier lives on DiffusionRequestState.extra.
    _CARRIER_KEY = "dreamzero_carrier"

    #: Carrier fields packed into a StagePayload per stage boundary. Everything
    #: DreamZero carries is model-private, so tensors go in
    #: ``private_tensor_fields`` and scalars in ``private_scalar_fields``; only
    #: ``session_id`` is public (the transition processor + AR runner read it to
    #: attach the right AR-Diffusion KV session without decoding private schema).
    _PAYLOAD_TENSOR_FIELDS: ClassVar[dict[StageBoundary, tuple[str, ...]]] = {
        StageBoundary.ENCODE_TO_DIT: (
            "prompt_embeds",
            "negative_prompt_embeds",
            "clip_feas",
            "ys",
            "image_latent",
            "state_features",
            "embodiment_id",
            "noise_obs",
            "noise_action",
            "state_for_postprocess",
        ),
        StageBoundary.DIT_TO_DECODE: ("video_out", "action_out", "state_for_postprocess"),
    }
    _PAYLOAD_SCALAR_FIELDS: ClassVar[tuple[str, ...]] = (
        "embodiment_name",
        "transform_embodiment",
        "reset_reason",
        "explicit_reset",
        "do_true_cfg",
        "current_start_frame",
        "height",
        "width",
        "seq_len",
        "frame_seqlen",
        "num_inference_steps",
        "sigma_shift",
    )

    @classmethod
    def required_components_for_stage(cls, model_stage: str) -> StageComponentSpec:
        """Declare which components each DreamZero stage must construct/load.

        Derived from the real data dependencies of the phase methods:
        encode owns the tokenizer/text+image encoders and the VAE ENCODER;
        denoise owns the CausalWan DiT + schedulers (+ action head, part of the
        transformer); decode owns the VAE DECODER for video export. The action
        norm stats / transforms are lightweight metadata built in __init__ for
        every stage, so they are not gated here.
        """
        role = normalize_stage_role(model_stage)
        if role == ENCODE:
            return StageComponentSpec(
                tokenizer=True,
                text_encoder=True,
                image_encoder=True,
                vae_encoder=True,
            )
        if role == DENOISE:
            return StageComponentSpec(
                dit=True,
                scheduler=True,
                action_modules=True,
            )
        if role == DECODE:
            return StageComponentSpec(vae_decoder=True)
        # Monolithic / unknown: everything. Reuse the shared all-True singleton so
        # this stays in step with StageComponentSpec if a component is ever added.
        return ALL_COMPONENTS

    # -- DiffusionV2Atoms state-based atoms ----------------------------------
    # The runner drives these explicitly (encode: init_state -> check_inputs ->
    # encode -> prepare; denoise: unpack -> diffuse; decode: unpack -> decode ->
    # postprocess). Each stores/reads the DreamZero carrier on state.extra so the
    # generic DiffusionRequestState never grows model-private fields.

    def init_state(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Initialize a fresh request state before encode (clear stale carrier)."""
        state.extra.pop(self._CARRIER_KEY, None)
        state.extra.pop("decoded_output", None)
        return state

    def check_inputs(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Validate that a raw request carries a robot observation."""
        extra_args = getattr(state.sampling, "extra_args", None) or {}
        if extra_args.get("robot_obs") is None:
            raise ValueError(
                f"DreamZero request {state.request_id!r} has no robot_obs in sampling_params.extra_args."
            )
        return state

    def encode(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Run the encode phase and stash the DreamZero carrier on state.extra."""
        extra_args = getattr(state.sampling, "extra_args", None) or {}
        robot_obs = extra_args["robot_obs"]
        session_id = str(extra_args.get("session_id") or state.request_id or "default")
        # Encode owns no AR-Diffusion KV: resolve a model-local session state only.
        dz_state = self._get_or_create_state(session_id)
        self.state = dz_state
        carrier = self._run_encode_phase(
            robot_obs,
            dz_state,
            session_id,
            explicit_reset=bool(extra_args.get("reset", False)),
        )
        state.extra[self._CARRIER_KEY] = carrier
        return state

    def prepare(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """No-op atom: the encode phase already prepared initial noise + schedule.

        DreamZero's timestep schedule is rebuilt on the denoise worker from the
        carried ``num_inference_steps`` / ``sigma_shift`` (the scheduler object
        is process-local and never transported), and the initial noise is carried
        explicitly. Kept as a distinct atom for protocol symmetry.
        """
        return state

    def diffuse(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """Whole-request denoise atom: run the DiT loop on the restored carrier.

        Called by the runner's denoise stage. The AR-Diffusion KV/session state
        was attached to the pipeline by ARDiffusionModelRunner before this call.
        DreamZero runs the entire per-request denoise here (its dual video+action
        DiT loop with model-owned KV), so it does NOT implement the single-step
        ``denoise_step`` / ``step_scheduler`` contract.
        """
        carrier = state.extra.get(self._CARRIER_KEY)
        if carrier is None:
            raise StagePayloadError(
                f"DreamZero denoise for {state.request_id!r} has no carrier on state.extra "
                f"[{self._CARRIER_KEY!r}] — unpack_stage_state must run first."
            )
        dz_state = self._get_or_create_state(carrier.session_id)
        self.state = dz_state
        if self._ar_diffusion_kv_state is None:
            raise StagePayloadError(
                f"DreamZero denoise for session {carrier.session_id!r} has no AR-Diffusion KV state; "
                "the denoise stage must run on the AR-Diffusion engine."
            )
        # Apply the same engine-KV window reset the encode worker decided. On the
        # denoise worker the KV state is attached, so this drops the pool window.
        if carrier.current_start_frame == 0:
            self._apply_model_local_reset(dz_state, carrier.reset_reason)
        # Realign the model-local window position + conditions with the encode
        # carrier (the denoise DreamZeroState is a fresh per-session shadow).
        dz_state.current_start_frame = carrier.current_start_frame
        dz_state.clip_feas = carrier.clip_feas
        dz_state.ys = carrier.ys
        if dz_state.prompt_embeds is None:
            dz_state.prompt_embeds = carrier.prompt_embeds
        self._run_denoise_phase(carrier, dz_state)
        return state

    def decode(self, state: DiffusionRequestState) -> DiffusionRequestState:
        """No-op atom: DreamZero's decode work happens in :meth:`postprocess`
        (action denorm + video export from the carrier); kept for symmetry."""
        return state

    def postprocess(self, state: DiffusionRequestState) -> DiffusionOutput:
        """Run the decode phase from the restored carrier -> user-visible output."""
        carrier = state.extra.get(self._CARRIER_KEY)
        if carrier is None:
            raise StagePayloadError(
                f"DreamZero decode for {state.request_id!r} has no carrier on state.extra."
            )
        return self._run_decode_phase(carrier)

    # -- step-only atoms (unused: DreamZero is request-mode / disaggregated) --
    # DreamZero's dual video+action denoise with model-owned AR-Diffusion KV does
    # not fit the single-tensor step contract; it overrides ``diffuse`` instead.
    # These stubs exist so ``isinstance(pipeline, DiffusionV2Atoms)`` holds while
    # making the request-mode-only intent explicit.

    def build_step_batch(self, states, *, cached_batch=None):
        raise NotImplementedError("DreamZero does not support single-process step batching.")

    def build_step_attention_metadata(self, input_batch):
        return None

    def denoise_step(self, input_batch):
        raise NotImplementedError("DreamZero runs a whole-request denoise via diffuse(); no per-step denoise_step.")

    def step_scheduler(self, state, noise_pred):
        raise NotImplementedError("DreamZero advances its scheduler inside diffuse(); no per-step step_scheduler.")

    # -- payload marshalling (StagePayload) ----------------------------------

    def pack_stage_state(self, state: DiffusionRequestState, boundary: StageBoundary) -> StagePayload:
        """Pack the DreamZero carrier into a transportable :class:`StagePayload`.

        The carrier is entirely model-private, so its tensors go in
        ``private_tensor_fields`` and its scalars in ``private_scalar_fields``
        (both sanitized/validated by ``StagePayload.create``). Only ``session_id``
        is exposed in the public ``scalar_fields`` so the generic transition
        processor and the AR-Diffusion runner can attach the right KV session
        without decoding model-private schema. The AR-Diffusion KV / scheduler /
        DreamZeroState are NEVER packed — only stable data.
        """
        carrier = state.extra.get(self._CARRIER_KEY)
        if carrier is None:
            raise StagePayloadError(
                f"DreamZero pack for {state.request_id!r} (boundary={boundary}) "
                f"has no carrier on state.extra[{self._CARRIER_KEY!r}]."
            )
        tensor_names = self._PAYLOAD_TENSOR_FIELDS.get(boundary)
        if tensor_names is None:
            raise StagePayloadError(f"DreamZero has no pack for stage boundary {boundary!r}.")

        private_tensor_fields = {}
        for name in tensor_names:
            value = getattr(carrier, name, None)
            if value is not None:
                private_tensor_fields[name] = value

        private_scalar_fields = {name: getattr(carrier, name) for name in self._PAYLOAD_SCALAR_FIELDS}

        return StagePayload.create(
            request_id=state.request_id,
            boundary=boundary,
            scalar_fields={"session_id": carrier.session_id},
            private_tensor_fields=private_tensor_fields,
            private_scalar_fields=private_scalar_fields,
        )

    def unpack_stage_state(self, payload: StagePayload, state: DiffusionRequestState) -> DiffusionRequestState:
        """Apply a received :class:`StagePayload` to the existing request state.

        Mutates the runner-created ``state`` in place (the runner already built it
        from the incoming request, preserving request-level sampling/generator
        plumbing): rebuilds the DreamZero carrier from the payload's scalar/tensor
        dicts and stashes it on ``state.extra``. The live AR-Diffusion session is
        acquired separately by the denoise runner via the normal engine mechanism
        (keyed by the transported ``session_id``) — never from the payload.
        """
        payload.validate()
        meta = payload.private_scalar_fields
        device = get_local_device()

        def _to_device(t):
            return t.to(device=device) if isinstance(t, torch.Tensor) else t

        carrier = DreamZeroStageCarrier(
            session_id=str(payload.scalar_fields.get("session_id", "default")),
            embodiment_name=str(meta.get("embodiment_name", "")),
            transform_embodiment=str(meta.get("transform_embodiment", "")),
            reset_reason=meta.get("reset_reason"),
            explicit_reset=bool(meta.get("explicit_reset", False)),
            do_true_cfg=bool(meta.get("do_true_cfg", False)),
            current_start_frame=int(meta.get("current_start_frame", 0)),
            height=int(meta.get("height", 0)),
            width=int(meta.get("width", 0)),
            seq_len=int(meta.get("seq_len", 0)),
            frame_seqlen=int(meta.get("frame_seqlen", 0)),
            num_inference_steps=int(meta.get("num_inference_steps", 0)),
            sigma_shift=float(meta.get("sigma_shift", 1.0)),
        )
        for name, tensor in payload.private_tensor_fields.items():
            if hasattr(carrier, name):
                setattr(carrier, name, _to_device(tensor))

        state.extra[self._CARRIER_KEY] = carrier
        return state

    # -----------------------------------------------------------------------
    # Action denormalization
    # -----------------------------------------------------------------------

    def _load_action_norm_stats(self, stats_path: str) -> dict[str, dict[str, torch.Tensor]]:
        """Load per-embodiment action normalization stats from metadata.json.

        Returns: {embodiment_name: {"q01": Tensor(action_dim,), "q99": Tensor(action_dim,)}}
        """
        with open(stats_path) as f:
            metadata = json.load(f)
        return self._parse_action_norm_stats(metadata)

    @staticmethod
    def _parse_norm_stats(metadata: dict, stats_kind: str) -> dict[str, dict[str, torch.Tensor]]:
        """Parse per-embodiment q01/q99 normalization stats of one kind.

        ``stats_kind`` selects the ``statistics`` sub-block ("action" or
        "state"); the joint_position + gripper_position q01/q99 vectors are
        concatenated into per-embodiment tensors. Shared by the action and state
        parsers, which differ only in that key.

        Returns: {embodiment_name: {"q01": Tensor(dim,), "q99": Tensor(dim,)}}
        """
        result = {}
        for emb_name, emb_data in metadata.items():
            stats = emb_data.get("statistics", {}).get(stats_kind, {})
            q01_parts, q99_parts = [], []
            for key in ["joint_position", "gripper_position"]:
                if key in stats:
                    q01_parts.extend(stats[key]["q01"])
                    q99_parts.extend(stats[key]["q99"])
            if q01_parts:
                result[emb_name] = {
                    "q01": torch.tensor(q01_parts, dtype=torch.float32),
                    "q99": torch.tensor(q99_parts, dtype=torch.float32),
                }
        return result

    @staticmethod
    def _parse_action_norm_stats(metadata: dict) -> dict[str, dict[str, torch.Tensor]]:
        """Load per-embodiment action normalization stats from metadata.json."""
        return DreamZeroPipeline._parse_norm_stats(metadata, "action")

    @staticmethod
    def _parse_state_norm_stats(metadata: dict) -> dict[str, dict[str, torch.Tensor]]:
        """Load per-embodiment state normalization stats from metadata.json."""
        return DreamZeroPipeline._parse_norm_stats(metadata, "state")

    def _normalize_state(
        self,
        state: torch.Tensor,
        embodiment_name: str,
    ) -> torch.Tensor:
        """Normalize state with q99 stats before feeding the model."""
        state_norm_stats = getattr(self, "state_norm_stats", {})
        if embodiment_name not in state_norm_stats:
            return state
        stats = state_norm_stats[embodiment_name]
        q01 = stats["q01"].to(device=state.device, dtype=state.dtype)
        q99 = stats["q99"].to(device=state.device, dtype=state.dtype)
        actual_dim = q01.shape[0]
        normalized = state.clone()
        range_vals = q99 - q01
        mask = range_vals != 0
        normalized_slice = normalized[..., :actual_dim]
        normalized_slice[..., mask] = 2 * (normalized_slice[..., mask] - q01[mask]) / range_vals[mask] - 1
        normalized_slice = torch.clamp(normalized_slice, -1, 1)
        normalized[..., :actual_dim] = normalized_slice
        return normalized

    def _denormalize_action(
        self,
        action: torch.Tensor,
        embodiment_name: str,
    ) -> torch.Tensor:
        """Denormalize action from [-1,1] to real values using q99 mode.

        Formula: real = (normalized + 1) / 2 * (q99 - q01) + q01
        """
        if embodiment_name not in self.action_norm_stats:
            return action
        stats = self.action_norm_stats[embodiment_name]
        q01 = stats["q01"].to(device=action.device, dtype=action.dtype)
        q99 = stats["q99"].to(device=action.device, dtype=action.dtype)
        # action shape: (B, horizon, action_dim) or (B, horizon, max_action_dim)
        # q01/q99 shape: (actual_action_dim,) -- only denorm actual dims
        actual_dim = q01.shape[0]
        action_real = action.clone()
        action_real[..., :actual_dim] = (action[..., :actual_dim] + 1) / 2 * (q99 - q01) + q01
        return action_real

    # -----------------------------------------------------------------------
    # Weight loading
    # -----------------------------------------------------------------------

    @property
    def weights_sources(self):
        """ComponentSource list for DiffusersPipelineLoader."""
        return self._weights_sources

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load checkpoint weights with key remapping."""
        loaded: set[str] = set()
        params = dict(self.named_parameters())
        buffers = dict(self.named_buffers())

        for name, tensor in weights:
            if name.startswith("action_head.model."):
                new_name = "transformer." + name[len("action_head.model.") :]
                new_name = (
                    new_name.replace("img_emb.proj.0.", "img_emb.norm1.")
                    .replace("img_emb.proj.1.", "img_emb.fc1.")
                    .replace("img_emb.proj.3.", "img_emb.fc2.")
                    .replace("img_emb.proj.4.", "img_emb.norm2.")
                )

                # Self-attn q/k/v are fused into a single QKVParallelLinear; route
                # each separate checkpoint weight/bias to the packed `qkv` param
                # with its shard id. cross_attn keeps separate q/k/v (q from x,
                # k/v from context — not fusible), so it is left untouched here.
                qkv_shard_id: str | None = None
                for shard_id in ("q", "k", "v"):
                    needle = f".self_attn.{shard_id}."
                    if needle in new_name:
                        new_name = new_name.replace(needle, ".self_attn.qkv.")
                        qkv_shard_id = shard_id
                        break

                if new_name in params:
                    param = params[new_name]
                    if qkv_shard_id is not None:
                        # QKVParallelLinear.weight_loader needs the shard id.
                        param.weight_loader(param, tensor, qkv_shard_id)
                    else:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, tensor)
                    loaded.add(new_name)
                elif new_name in buffers:
                    buffers[new_name].data.copy_(tensor)
                    loaded.add(new_name)

            elif name.startswith("action_head.text_encoder."):
                mapped = self._remap_text_encoder_key(name)
                if mapped is None:
                    continue
                for new_name in mapped if isinstance(mapped, list) else [mapped]:
                    full_name = "text_encoder." + new_name
                    if full_name in params:
                        params[full_name].data.copy_(tensor)
                        loaded.add(full_name)

            elif name.startswith("action_head.image_encoder."):
                self._remap_image_encoder_key(name, tensor, params, loaded)

            elif name.startswith("action_head.vae."):
                mapped = self._remap_vae_key(name)
                if mapped is None:
                    continue
                full_name = "vae." + mapped
                if full_name in params:
                    params[full_name].data.copy_(tensor)
                    loaded.add(full_name)

        logger.info(
            "DreamZero load_weights: loaded %d parameters from root checkpoint",
            len(loaded),
        )
        return loaded

    # -----------------------------------------------------------------------
    # Text encoder key remapping
    # -----------------------------------------------------------------------

    @staticmethod
    def _remap_text_encoder_key(name: str) -> str | list[str] | None:
        """Remap a single text encoder key."""
        subkey = name[len("action_head.text_encoder.") :]

        if subkey == "token_embedding.weight":
            return "shared.weight"
        if subkey == "norm.weight":
            return "encoder.final_layer_norm.weight"

        m = re.match(r"blocks\.(\d+)\.(.*)", subkey)
        if not m:
            return None
        block_idx = m.group(1)
        rest = m.group(2)

        prefix = f"encoder.block.{block_idx}"

        if rest == "attn.q.weight":
            return f"{prefix}.layer.0.SelfAttention.q.weight"
        if rest == "attn.k.weight":
            return f"{prefix}.layer.0.SelfAttention.k.weight"
        if rest == "attn.v.weight":
            return f"{prefix}.layer.0.SelfAttention.v.weight"
        if rest == "attn.o.weight":
            return f"{prefix}.layer.0.SelfAttention.o.weight"
        if rest == "pos_embedding.embedding.weight":
            return f"{prefix}.layer.0.SelfAttention.relative_attention_bias.weight"
        if rest == "norm1.weight":
            return f"{prefix}.layer.0.layer_norm.weight"

        if rest == "ffn.gate.0.weight":
            return f"{prefix}.layer.1.DenseReluDense.wi_0.weight"
        if rest == "ffn.fc1.weight":
            return f"{prefix}.layer.1.DenseReluDense.wi_1.weight"
        if rest == "ffn.fc2.weight":
            return f"{prefix}.layer.1.DenseReluDense.wo.weight"
        if rest == "norm2.weight":
            return f"{prefix}.layer.1.layer_norm.weight"

        return None

    # -----------------------------------------------------------------------
    # VAE key remapping
    # -----------------------------------------------------------------------

    @staticmethod
    def _remap_vae_key(name: str) -> str | None:
        """Remap DreamZero VAE keys to `DistributedAutoencoderKLWan` keys."""
        if not name.startswith("action_head.vae.model."):
            return None

        rest = name[len("action_head.vae.model.") :]

        direct_prefix_map = {
            "encoder.conv1.": "encoder.conv_in.",
            "encoder.head.0.": "encoder.norm_out.",
            "encoder.head.2.": "encoder.conv_out.",
            "decoder.conv1.": "decoder.conv_in.",
            "decoder.head.0.": "decoder.norm_out.",
            "decoder.head.2.": "decoder.conv_out.",
            "conv1.": "quant_conv.",
            "conv2.": "post_quant_conv.",
        }
        for src_prefix, dst_prefix in direct_prefix_map.items():
            if rest.startswith(src_prefix):
                return dst_prefix + rest[len(src_prefix) :]

        resnet_leaf_map = {
            "residual.0.gamma": "norm1.gamma",
            "residual.2.weight": "conv1.weight",
            "residual.2.bias": "conv1.bias",
            "residual.3.gamma": "norm2.gamma",
            "residual.6.weight": "conv2.weight",
            "residual.6.bias": "conv2.bias",
        }
        block_leaf_map = {
            **resnet_leaf_map,
            "shortcut.weight": "conv_shortcut.weight",
            "shortcut.bias": "conv_shortcut.bias",
            "resample.1.weight": "resample.1.weight",
            "resample.1.bias": "resample.1.bias",
            "time_conv.weight": "time_conv.weight",
            "time_conv.bias": "time_conv.bias",
        }

        m = re.match(r"encoder\.middle\.(\d+)\.(.*)", rest)
        if m:
            idx = int(m.group(1))
            tail = m.group(2)
            if idx in (0, 2) and tail in resnet_leaf_map:
                res_idx = 0 if idx == 0 else 1
                return f"encoder.mid_block.resnets.{res_idx}.{resnet_leaf_map[tail]}"
            if idx == 1:
                return f"encoder.mid_block.attentions.0.{tail}"
            return None

        m = re.match(r"decoder\.middle\.(\d+)\.(.*)", rest)
        if m:
            idx = int(m.group(1))
            tail = m.group(2)
            if idx in (0, 2) and tail in resnet_leaf_map:
                res_idx = 0 if idx == 0 else 1
                return f"decoder.mid_block.resnets.{res_idx}.{resnet_leaf_map[tail]}"
            if idx == 1:
                return f"decoder.mid_block.attentions.0.{tail}"
            return None

        m = re.match(r"encoder\.downsamples\.(\d+)\.(.*)", rest)
        if m:
            idx = int(m.group(1))
            tail = m.group(2)
            if tail in block_leaf_map:
                return f"encoder.down_blocks.{idx}.{block_leaf_map[tail]}"
            return None

        m = re.match(r"decoder\.upsamples\.(\d+)\.(.*)", rest)
        if m:
            idx = int(m.group(1))
            tail = m.group(2)
            if tail not in block_leaf_map:
                return None

            if idx <= 2:
                prefix = f"decoder.up_blocks.0.resnets.{idx}."
            elif idx == 3:
                prefix = "decoder.up_blocks.0.upsamplers.0."
            elif 4 <= idx <= 6:
                prefix = f"decoder.up_blocks.1.resnets.{idx - 4}."
            elif idx == 7:
                prefix = "decoder.up_blocks.1.upsamplers.0."
            elif 8 <= idx <= 10:
                prefix = f"decoder.up_blocks.2.resnets.{idx - 8}."
            elif idx == 11:
                prefix = "decoder.up_blocks.2.upsamplers.0."
            elif 12 <= idx <= 14:
                prefix = f"decoder.up_blocks.3.resnets.{idx - 12}."
            else:
                return None
            return prefix + block_leaf_map[tail]

        return None

    # -----------------------------------------------------------------------
    # Image encoder key remapping
    # -----------------------------------------------------------------------

    def _remap_image_encoder_key(
        self,
        name: str,
        tensor: torch.Tensor,
        params: dict[str, torch.nn.Parameter],
        loaded: set[str],
    ) -> None:
        """Map an image encoder key onto the local module."""
        if not name.startswith("action_head.image_encoder."):
            return

        full_name = "image_encoder." + name[len("action_head.image_encoder.") :]
        if full_name in params:
            params[full_name].data.copy_(tensor)
            loaded.add(full_name)
