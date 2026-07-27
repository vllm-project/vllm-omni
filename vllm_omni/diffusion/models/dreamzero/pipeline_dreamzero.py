# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""DreamZero pipeline for vllm-omni.

Entry point for DiffusionEngine.step_streaming() -> pipeline.forward(req)
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re as re_module
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import contextmanager

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
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState
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
from vllm_omni.diffusion.models.schedulers.scheduling_flow_unipc_multistep import FlowUniPCMultistepScheduler
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionCrossAttentionKVSpec,
    ARDiffusionKVBranchSpec,
    ARDiffusionKVCacheSpec,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = logging.getLogger(__name__)
MAX_DREAMZERO_SESSIONS = 64


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
    CFG: video gets standard CFG, action takes the positive KV branch only.

    KV is managed by the AR-Diffusion engine through the explicit capability
    methods below. The runner binds one session state only for ``forward()``.
    """

    _POSITIVE_BRANCH = "positive"
    _NEGATIVE_BRANCH = "negative"
    _ar_diffusion_kv_state = None
    state: DreamZeroState | None

    def ar_diffusion_kv_cache_spec(self) -> ARDiffusionKVCacheSpec:
        """Describe DreamZero's local KV geometry to the generic runner."""
        transformer = self.transformer
        frame_tokens = int(transformer.frame_seqlen)
        max_attention_tokens = int(transformer.blocks[0].self_attn.max_attention_size)
        cfg_world = int(get_classifier_free_guidance_world_size())
        negative_local_index = 0 if cfg_world >= 2 else 1
        cross_attention = [
            ARDiffusionCrossAttentionKVSpec("text", int(transformer.text_len)),
        ]
        if transformer.model_type == "i2v":
            cross_attention.append(ARDiffusionCrossAttentionKVSpec("image", 257))
        return ARDiffusionKVCacheSpec(
            num_layers=int(transformer.num_layers),
            num_kv_heads=int(transformer.blocks[0].self_attn.tp_num_heads),
            head_size=int(transformer.dim // transformer.num_heads),
            tokens_per_frame=frame_tokens,
            frames_per_block=int(transformer.num_frame_per_block),
            window_frames=max_attention_tokens // frame_tokens,
            kv_branches=(
                ARDiffusionKVBranchSpec(self._POSITIVE_BRANCH, 0),
                ARDiffusionKVBranchSpec(self._NEGATIVE_BRANCH, negative_local_index),
            ),
            session_capacity=MAX_DREAMZERO_SESSIONS,
            cross_attention=tuple(cross_attention),
            max_scratch_tokens_per_branch=int(transformer.num_action_per_block + transformer.num_state_per_block),
        )

    @contextmanager
    def bind_ar_diffusion_state(self, session_id, state):
        """Bind runner-owned KV only for the duration of one forward."""
        if self._ar_diffusion_kv_state is not None:
            raise RuntimeError("DreamZero AR-Diffusion state is already bound")
        if state.session_id != session_id:
            raise ValueError(f"DreamZero bound session mismatch: {state.session_id!r} != {session_id!r}")
        self._ar_diffusion_kv_state = state
        try:
            yield
        finally:
            self._ar_diffusion_kv_state = None

    def reset_ar_diffusion_session(self, session_id: str) -> None:
        """Reset DreamZero-owned state after the runner releases session KV."""
        self._drop_ar_diffusion_session_state(session_id)

    def close_ar_diffusion_session(self, session_id: str) -> None:
        """Drop DreamZero-owned state after close, eviction, or failed forward."""
        self._drop_ar_diffusion_session_state(session_id)

    def _drop_ar_diffusion_session_state(self, session_id: str) -> None:
        """Remove model state and clear the compatibility alias when it points there."""
        removed = self._states.pop(str(session_id or "default"), None)
        if removed is not None and getattr(self, "state", None) is removed:
            self.state = None

    @staticmethod
    def _ar_warmup_robot_obs(height: int, width: int, n_frames: int, session_id: str) -> dict:
        image = (
            np.zeros((height, width, 3), dtype=np.uint8)
            if n_frames == 1
            else np.zeros((n_frames, height, width, 3), dtype=np.uint8)
        )
        return {
            "observation/exterior_image_0_left": image,
            "observation/exterior_image_1_left": image,
            "observation/wrist_image_left": image,
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/cartesian_position": np.zeros(6, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": "warmup",
            "session_id": session_id,
        }

    def ar_diffusion_warmup_requests(self, session_id: str) -> Iterable[OmniDiffusionRequest]:
        """Yield DreamZero-valid requests covering each resident window shape."""
        spec = self.ar_diffusion_kv_cache_spec()
        n_forwards = 1 + math.ceil(max(0, spec.window_frames - 1) / spec.frames_per_block)
        raw_kv_config = getattr(self.od_config, "ar_diffusion_kv_config", None)
        if raw_kv_config is None and isinstance(self.od_config.model_config, dict):
            raw_kv_config = self.od_config.model_config.get("ar_diffusion_kv_config")
        capture_reset = (
            bool(raw_kv_config.get("warmup_capture_reset", False))
            if isinstance(raw_kv_config, dict)
            else bool(getattr(raw_kv_config, "warmup_capture_reset", False))
        )
        if capture_reset:
            n_forwards += 1
        policy_config = self.od_config.model_config.get("policy_server_config", {})
        height, width = (int(value) for value in policy_config.get("image_resolution", [180, 320]))
        for index in range(n_forwards):
            n_frames = 1 if index == 0 else 4
            robot_obs = self._ar_warmup_robot_obs(height, width, n_frames, session_id)
            sampling_params = OmniDiffusionSamplingParams(
                extra_args={
                    "reset": index == 0,
                    "session_id": session_id,
                    "robot_obs": robot_obs,
                }
            )
            yield OmniDiffusionRequest(
                prompt="warmup",
                sampling_params=sampling_params,
                request_id=f"ardiffusion-warmup-{index}",
            )

    def _ar_branch(self, is_negative: bool) -> str:
        return self._NEGATIVE_BRANCH if is_negative else self._POSITIVE_BRANCH

    def _kv_get(self, state, is_negative, seq_len=None, update_kv_cache=False):
        return self._ar_diffusion_kv_state.get_kv_caches(
            self._ar_branch(is_negative),
            seq_len=seq_len,
            commit_current=update_kv_cache,
        )

    def _kv_create(self, state, batch_size, dtype, device, num_layers, num_heads, head_dim):
        # The engine owns all KV allocation: self-attn is allocated lazily from
        # paged contexts, and cross-attn is populated eagerly in _kv_populate_cross.
        # Nothing is created model-side.
        return

    def _kv_commit(self, is_negative: bool):
        self._ar_diffusion_kv_state.commit_paged_context(self._ar_branch(is_negative))

    def _kv_get_cross(self, state, is_negative):
        """Cross-attn cache from the engine pool (text k/v + I2V image k_img/v_img)."""
        kv_branch = self._ar_branch(is_negative)
        text_caches = self._ar_diffusion_kv_state.get_cross_attention_kv(kv_branch, "text")
        if "image" in self._ar_diffusion_kv_state.kv_cache.cross_attention_lengths:
            image_caches = self._ar_diffusion_kv_state.get_cross_attention_kv(kv_branch, "image")
            for text_cache, image_cache in zip(text_caches, image_caches, strict=True):
                text_cache["k_img"] = image_cache["k"]
                text_cache["v_img"] = image_cache["v"]
        return text_caches

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
        kv_branch = self._ar_branch(is_negative)
        need_text = not s.is_cross_attention_populated(kv_branch, "text")
        need_img = (
            clip_feature is not None
            and self.transformer.model_type == "i2v"
            and not s.is_cross_attention_populated(kv_branch, "image")
        )
        if not need_text and not need_img:
            return
        projected = self.transformer.text_embedding(context) if need_text else None
        img_ctx = self.transformer.img_emb(clip_feature) if need_img else None
        if projected is not None:

            def text_layer_kv():
                for block in self.transformer.blocks:
                    ca = block.cross_attn
                    n, d = ca.tp_num_heads, ca.head_dim
                    yield (
                        ca.norm_k(ca.k(projected)).unflatten(2, (n, d)),
                        ca.v(projected).unflatten(2, (n, d)),
                    )

            s.populate_cross_attention(kv_branch, "text", text_layer_kv())
        if img_ctx is not None:

            def image_layer_kv():
                for block in self.transformer.blocks:
                    ca = block.cross_attn
                    n, d = ca.tp_num_heads, ca.head_dim
                    yield (
                        ca.norm_k_img(ca.k_img(img_ctx)).unflatten(2, (n, d)),
                        ca.v_img(img_ctx).unflatten(2, (n, d)),
                    )

            s.populate_cross_attention(kv_branch, "image", image_layer_kv())
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
        """
        state.reset(clear_video_latents=clear_video_latents)
        keep_cross = ("text",) if not clear_video_latents else ()
        self._ar_diffusion_kv_state.reset(keep_cross_attention=keep_cross)

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

        # DreamZero is engine-only: every KV access in forward() routes through
        # the AR-Diffusion engine's pool-backed state. Fail fast here — a stale
        # or programmatic config that leaves engine_backend="default" would
        # otherwise only crash mid-forward on the first KV access.
        engine_backend = str(getattr(od_config, "engine_backend", "") or "")
        if "ar_diffusion" not in engine_backend.lower().replace("-", "_"):
            raise ValueError(
                "DreamZeroPipeline requires the AR-Diffusion engine; set "
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

        # ---- Tokenizer ----
        tokenizer_source = od_config.model_paths.get("tokenizer", "google/umt5-xxl")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        # Instantiate from config; weights load through `load_weights()`.
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

        self.image_encoder = DreamZeroImageEncoder()

        # Build a compatible VAE module, then fill it through `load_weights()`.
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
            getattr(od_config, "enable_cpu_offload", False) or getattr(od_config, "enable_layerwise_offload", False)
        ):
            self.vae = self.vae.to(device=get_local_device(), dtype=od_config.dtype)
        self.register_buffer(
            "vae_latents_mean",
            torch.tensor(self.vae.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "vae_latents_inv_std",
            (1.0 / torch.tensor(self.vae.config.latents_std, dtype=torch.float32)).view(1, -1, 1, 1, 1),
            persistent=False,
        )

        # Filter out keys not accepted by `CausalWanModel.__init__`.
        transformer_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in ("_convert_", "_target_")}
        transformer_kwargs["action_dim"] = ah_config["action_dim"]
        transformer_kwargs["max_state_dim"] = ah_config["max_state_dim"]
        transformer_kwargs["num_frame_per_block"] = ah_config["num_frame_per_block"]
        self.transformer = CausalWanModel(**transformer_kwargs)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )

        self._states: OrderedDict[str, DreamZeroState] = OrderedDict()
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

        try:
            self.text_encoder.forward = torch.compile(self.text_encoder.forward, **compile_ro)
        except Exception as exc:
            logger.warning("DreamZero: text_encoder compile failed (%s); skipping.", exc)

        try:
            self.image_encoder.model.visual.forward = torch.compile(
                self.image_encoder.model.visual.forward,
                **compile_ro,
            )
        except Exception as exc:
            logger.warning("DreamZero: image_encoder compile failed (%s); skipping.", exc)

        try:
            self.vae._encode = torch.compile(self.vae._encode, **compile_ro)
        except Exception as exc:
            logger.warning("DreamZero: vae._encode compile failed (%s); skipping.", exc)

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

        state = self.state
        if state is None:
            state = self._get_or_create_state("default")
            self.state = state
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
                self._encode_image(image, self.num_frames, 180, 320, state=state)
            except Exception as exc:
                logger.warning("DreamZero compile warmup (image_encoder) skipped: %s", exc)

            try:
                state.reset_vae_encoder_stream()
                dummy_video = torch.zeros(1, 3, 1, 180, 320, dtype=torch.bfloat16, device=device)
                self._vae_stream_seed(state, dummy_video[:, :, :1])
                for _ in range(4):
                    self._vae_stream_append_frame(state, dummy_video[:, :, :1])
                self._vae_stream_get_observation_latents(
                    state,
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
                mean = self.vae_latents_mean.to(device=device, dtype=self.vae.dtype)
                inv_std = self.vae_latents_inv_std.to(device=device, dtype=self.vae.dtype)
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

    def _vae_quantize_encoder_out(self, encoder_out: torch.Tensor) -> torch.Tensor:
        enc = self.vae.quant_conv(encoder_out)
        mu, _ = enc.chunk(2, dim=1)
        mean = self.vae_latents_mean.to(device=mu.device, dtype=mu.dtype)
        inv_std = self.vae_latents_inv_std.to(device=mu.device, dtype=mu.dtype)
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
        mean = self.vae_latents_mean.to(device=mu.device, dtype=mu.dtype)
        inv_std = self.vae_latents_inv_std.to(device=mu.device, dtype=mu.dtype)
        mu = (mu - mean) * inv_std
        return mu.to(dtype=input_dtype)

    def decode_video_latents(self, video_latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized VAE latents into RGB video tensors."""
        vae_dtype = self.vae.dtype
        vae_device = next(self.vae.parameters()).device
        latents = video_latents.to(device=vae_device, dtype=vae_dtype)
        mean = self.vae_latents_mean.to(device=vae_device, dtype=vae_dtype)
        inv_std = self.vae_latents_inv_std.to(device=vae_device, dtype=vae_dtype)
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

    def diffuse(
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

        For each timestep:
          1. Build positive_kwargs / negative_kwargs
          2. predict_noise_maybe_with_cfg()    -> (video_pred, action_pred)
          3. scheduler_step_maybe_with_cfg()   -> VideoActionScheduler
          4. _synchronize_cfg_parallel_step_output()
        """
        seq_len = kwargs["seq_len"]
        state_features = kwargs.get("state_features")
        embodiment_id = kwargs.get("embodiment_id")

        # Shared kwargs for predict_noise (both conditional and unconditional branches)
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
        """Full inference step. Called by DiffusionEngine.step_streaming()."""
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
        transform, unified_obs = self._transform_robot_obs(robot_obs)
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

        # Explicit request reset is handled by ARDiffusionModelRunner before it
        # binds this forward. Model-detected prompt/window resets still happen
        # here because they depend on tokenized DreamZero state.
        reset_reason = state.reset_reason(text_tokens, 0, self.transformer.local_attn_size)
        if reset_reason == "session":
            self._kv_reset(state)
        elif reset_reason == "inference":
            self._kv_reset(state, clear_video_latents=False)
        state.language = text_tokens

        # Frame accumulation: stitched single frame -> multi-frame video
        video_frames = state.accumulate_frames(stitched)  # (T, H, W, C)
        videos = torch.from_numpy(video_frames).unsqueeze(0).to(device)  # (B=1, T, H, W, C)

        videos = self._preprocess_video(videos)  # -> [B,C,T,H,W] bf16
        _, _, num_frames_raw, height, width = videos.shape

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

        # Prompt embeds are constant within a session (a prompt change triggers a
        # "session" reset above, which clears this cache alongside state.language).
        if state.prompt_embeds is None:
            state.prompt_embeds = self._encode_text(text_tokens, attention_mask)
        prompt_embeds = state.prompt_embeds
        # Negative prompt for the unconditional CFG branch (model constant)
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

        _pt_mark("text_encode")

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

            # Eager cross-attn population (AR-Diffusion only): cache text + image-token K/V
            # into the pool now that the image is encoded (clip_feas available).
            # Runs on the first forward of a session and after each window-boundary
            # reset (current_start_frame returns to 0); the populate guards (text/img halves) gate re-entry.
            self._kv_populate_cross(prompt_embeds, state.clip_feas, is_negative=False)
            if negative_prompt_embeds is not None:
                self._kv_populate_cross(negative_prompt_embeds, state.clip_feas, is_negative=True)

        if state.current_start_frame != 0:
            latent_dtype = videos.dtype
            with torch.no_grad():
                image = self._encode_observation_latents(state, videos, latent_dtype=latent_dtype)
        _pt_mark("obs_vae_encode")

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
            self.transformer.action_dim,
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
        self._prefill_kv_cache(
            image,
            prompt_embeds,
            negative_prompt_embeds,
            frame_seqlen,
            seq_len,
            do_true_cfg,
            state,
        )
        _pt_mark("prefill_kv")

        sample_scheduler = copy.deepcopy(self.scheduler)
        sample_scheduler_action = copy.deepcopy(self.scheduler)
        sample_scheduler.set_timesteps(
            self.num_inference_steps,
            device=device,
            shift=self.sigma_shift,
        )
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps,
            device=device,
            shift=self.sigma_shift,
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

        video_out, action_out = self.diffuse(
            video_latents=noise_obs,
            action_latents=noise_action,
            timesteps_video=sample_scheduler.timesteps,
            timesteps_action=sample_scheduler_action.timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            video_action_scheduler=video_action_scheduler,
            do_true_cfg=do_true_cfg,
            state=state,
            seq_len=seq_len,
            state_features=state_features,
            embodiment_id=embodiment_id,
        )
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

        if state.current_start_frame == 1:
            video_out = torch.cat([image, video_out], dim=1)
        state.current_start_frame += self.num_frame_per_block

        state.append_video_latents(video_out)

        # q99 denorm: [-1,1] → real values
        action_out = self._denormalize_action(action_out.float(), embodiment_name)

        # Relative -> absolute: only for relative_action_keys (joint_position only)
        # gripper_position is NOT relative, so don't add state back to it
        if self.relative_action and state_for_postprocess is not None:
            n_relative = self.relative_action_dim  # 7 for DROID (joint only)
            # Use original state precision for post-denorm absolute recovery.
            # Upstream adds obs state after `eval_transform.unapply()`
            # the bf16 denoising path.
            last_state = state_for_postprocess[:, 0, :n_relative]  # (B, n_relative)
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
    def _parse_action_norm_stats(metadata: dict) -> dict[str, dict[str, torch.Tensor]]:
        result = {}
        for emb_name, emb_data in metadata.items():
            action_stats = emb_data.get("statistics", {}).get("action", {})
            q01_parts, q99_parts = [], []
            # Concatenate joint_position + gripper_position stats
            for key in ["joint_position", "gripper_position"]:
                if key in action_stats:
                    q01_parts.extend(action_stats[key]["q01"])
                    q99_parts.extend(action_stats[key]["q99"])
            if q01_parts:
                result[emb_name] = {
                    "q01": torch.tensor(q01_parts, dtype=torch.float32),
                    "q99": torch.tensor(q99_parts, dtype=torch.float32),
                }
        return result

    @staticmethod
    def _parse_state_norm_stats(metadata: dict) -> dict[str, dict[str, torch.Tensor]]:
        """Load per-embodiment state normalization stats from metadata.json."""
        result = {}
        for emb_name, emb_data in metadata.items():
            state_stats = emb_data.get("statistics", {}).get("state", {})
            q01_parts, q99_parts = [], []
            for key in ["joint_position", "gripper_position"]:
                if key in state_stats:
                    q01_parts.extend(state_stats[key]["q01"])
                    q99_parts.extend(state_stats[key]["q99"])
            if q01_parts:
                result[emb_name] = {
                    "q01": torch.tensor(q01_parts, dtype=torch.float32),
                    "q99": torch.tensor(q99_parts, dtype=torch.float32),
                }
        return result

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

        m = re_module.match(r"blocks\.(\d+)\.(.*)", subkey)
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

        m = re_module.match(r"encoder\.middle\.(\d+)\.(.*)", rest)
        if m:
            idx = int(m.group(1))
            tail = m.group(2)
            if idx in (0, 2) and tail in resnet_leaf_map:
                res_idx = 0 if idx == 0 else 1
                return f"encoder.mid_block.resnets.{res_idx}.{resnet_leaf_map[tail]}"
            if idx == 1:
                return f"encoder.mid_block.attentions.0.{tail}"
            return None

        m = re_module.match(r"decoder\.middle\.(\d+)\.(.*)", rest)
        if m:
            idx = int(m.group(1))
            tail = m.group(2)
            if idx in (0, 2) and tail in resnet_leaf_map:
                res_idx = 0 if idx == 0 else 1
                return f"decoder.mid_block.resnets.{res_idx}.{resnet_leaf_map[tail]}"
            if idx == 1:
                return f"decoder.mid_block.attentions.0.{tail}"
            return None

        m = re_module.match(r"encoder\.downsamples\.(\d+)\.(.*)", rest)
        if m:
            idx = int(m.group(1))
            tail = m.group(2)
            if tail in block_leaf_map:
                return f"encoder.down_blocks.{idx}.{block_leaf_map[tail]}"
            return None

        m = re_module.match(r"decoder\.upsamples\.(\d+)\.(.*)", rest)
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
