# SPDX-License-Identifier: Apache-2.0
"""ARDiffusionModelRunner — the diffusion model runner for the AR-Diffusion engine.

Subclasses ``DiffusionModelRunner`` and owns the engine-level KV cache
(:class:`ARDiffusionKVCache`). It brackets a request's rollout with the KV lifecycle
(``begin_request`` / ``end_request``) and exposes the live ``ARDiffusionKVCache`` so the
model pipeline (DreamZero) can drive the per-chunk allocate / slot-mapping /
gather / commit operations during ``pipeline.forward``.

Selecting the AR-Diffusion engine enables KV management (no env gate); optional KV
parameters come from the deploy yaml's ``model_config["ar_diffusion_kv_config"]``.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import OrderedDict

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import MAX_DREAMZERO_SESSIONS
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

logger = init_logger(__name__)


def resolve_ar_diffusion_kv_config(od_config: OmniDiffusionConfig) -> ARDiffusionKVConfig:
    """Resolve the AR-Diffusion KV config from the deploy/command arg.

    Selecting the AR-Diffusion engine implies KV management, so the result is always
    enabled. Optional overrides (``window_chunks`` / ``gpu_memory_fraction`` /
    ``sink_chunks`` / ``reset_at_boundary``) may be supplied via the deploy yaml as
    ``model_config["ar_diffusion_kv_config"]``. ``chunk_size`` / ``window_chunks`` are
    finalized from the model geometry at load (see ``_preallocate_kv_cache``).
    """
    raw = getattr(od_config, "ar_diffusion_kv_config", None)
    if raw is None:
        model_config = getattr(od_config, "model_config", None)
        if isinstance(model_config, dict):
            raw = model_config.get("ar_diffusion_kv_config")
    if isinstance(raw, ARDiffusionKVConfig):
        return dataclasses.replace(raw, enable=True)
    if isinstance(raw, dict):
        return ARDiffusionKVConfig(**{**raw, "enable": True})
    return ARDiffusionKVConfig(enable=True)


class ARDiffusionModelRunner(DiffusionModelRunner):
    def __init__(self, vllm_config, od_config: OmniDiffusionConfig, device) -> None:
        super().__init__(vllm_config, od_config, device)
        self.ar_diffusion_kv_config = resolve_ar_diffusion_kv_config(od_config)
        # Built after the model is loaded (dimensions known); stays None while KV
        # management is disabled.
        self.kv_cache: ARDiffusionKVCache | None = None
        # DreamZero KV is session-scoped (persists across a session's forwards),
        # so AR-Diffusion KV state is keyed by session_id and reused, not created per request.
        # Bounded by the same LRU cap as DreamZeroPipeline._states: an evicted
        # session's ARDiffusionKVState owns pool blocks (two CFG adapters), so without a
        # bound, session-id churn in a long-running server would leak pool ownership
        # until the KV pool is exhausted. Evicting frees those blocks (state.close()).
        self._ar_diffusion_states: OrderedDict[str, ARDiffusionKVState] = OrderedDict()
        self._max_ar_diffusion_states = MAX_DREAMZERO_SESSIONS
        # Per-request server-side forward E2E times (seconds), recorded in
        # execute_model. This is the worker-side compute time — the analog of the
        # upstream server's per-request E2E — and excludes the engine<->worker IPC
        # that the client's omni.generate wall time includes. Read (and cleared)
        # by the ar_diffusion_perf_stats worker RPC for the perf-compare summary.
        self._perf_e2e_times: list[float] = []

    def build_kv_cache(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype,
        block_size: int,
        max_model_len: int,
        available_bytes: int,
        cross_attn_length: int = 0,
        cross_attn_img_length: int = 0,
        local_branches: int = 2,
        num_frame_per_block: int = 1,
    ) -> None:
        """Construct the ARDiffusionKVCache (preallocating GPU pools on ``self.device``).

        No-op when KV management is disabled.
        """
        if not self.ar_diffusion_kv_config.enable:
            return
        self.kv_cache = ARDiffusionKVCache(
            self.ar_diffusion_kv_config,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
            block_size=block_size,
            max_model_len=max_model_len,
            available_bytes=available_bytes,
            cross_attn_length=cross_attn_length,
            cross_attn_img_length=cross_attn_img_length,
            local_branches=local_branches,
            num_frame_per_block=num_frame_per_block,
            device=self.device,
        )
        logger.info(
            "AR-Diffusion KV cache enabled: %d blocks, chunk_size=%d, window_chunks=%s, "
            "layers=%d kv_heads=%d head_dim=%d block_size=%d cross_attn_len=%d device=%s",
            self.kv_cache.num_blocks,
            self.ar_diffusion_kv_config.chunk_size,
            self.ar_diffusion_kv_config.window_chunks,
            num_layers,
            num_kv_heads,
            head_size,
            block_size,
            cross_attn_length,
            self.device,
        )

    # -- preallocation at load -------------------------------------------------

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        if self.ar_diffusion_kv_config.enable and self.pipeline is not None:
            self._preallocate_kv_cache()
            # Pre-capture the CUDA graphs for every window-fill shape at load time
            # (only when compiling/cuda-graph is on) so serving is fast from chunk 0.
            if not self.od_config.enforce_eager and self.ar_diffusion_kv_config.warmup_cudagraph:
                self._warmup_ar_rollout()

    def _infer_frame_seqlen(self) -> int:
        """frame_seqlen = (H//8)*(W//8)//4 from the configured image_resolution."""
        mc = getattr(self.od_config, "model_config", None) or {}
        psc = (mc.get("policy_server_config") if isinstance(mc, dict) else None) or {}
        res = psc.get("image_resolution", [180, 320])
        h, w = int(res[0]), int(res[1])
        return (h // 8) * (w // 8) // 4

    def _preallocate_kv_cache(self) -> None:
        """Build the KV pool once, from the loaded DreamZero transformer geometry."""
        # The paged self-attention path carries one rollout per request: slot
        # mappings/adapters are single-sequence and K/V writes address batch
        # index 0 only. Reject a multi-sequence config at init instead of
        # failing (or silently dropping sequences) mid-forward.
        max_num_seqs = int(getattr(self.od_config, "max_num_seqs", 1) or 1)
        if max_num_seqs > 1:
            raise ValueError(
                "AR-Diffusion paged KV supports max_num_seqs=1 (single-sequence "
                f"rollouts); got max_num_seqs={max_num_seqs}. Per-sequence "
                "adapters/slot maps are not implemented."
            )
        t = self.pipeline.transformer
        num_layers = int(t.num_layers)
        num_kv_heads = int(getattr(t.blocks[0].self_attn, "tp_num_heads", t.num_heads))
        head_size = int(t.dim // t.num_heads)
        num_frame_per_block = int(t.num_frame_per_block)
        local_attn_size = int(t.local_attn_size)
        # The loaded transformer is the source of truth for frame_seqlen (it derives
        # max_attention_size from it). The deploy-config inference is only a
        # cross-check — warn on drift rather than silently sizing chunks wrong.
        frame_seqlen = int(getattr(t, "frame_seqlen", 0)) or self._infer_frame_seqlen()
        inferred = self._infer_frame_seqlen()
        if frame_seqlen != inferred:
            logger.warning(
                "AR-Diffusion frame_seqlen mismatch: transformer=%d deploy-config=%d; using transformer",
                frame_seqlen,
                inferred,
            )
        # The model's own attention window, in tokens (= the [-max_attention_size:]
        # slice it applies). Read it directly rather than recomputing.
        max_attention_size = int(t.blocks[0].self_attn.max_attention_size)
        # Cross-attn text length — static pool for the text-encoder output.
        # The cached k/v spans the full text sequence (the image tokens
        # prepended in _forward_blocks are stripped inside the cross-attn
        # forward but are not part of the cached k/v).
        cross_attn_length = int(getattr(t, "text_len", 0))
        # I2V image-token cross-attn length: the model splits context[:, :257] as the
        # image tokens (the img_emb(clip_feature) output), caching k_img/v_img. T2V: 0.
        cross_attn_img_length = 257 if getattr(t, "model_type", "t2v") == "i2v" else 0

        # Frame-granular paging: 1 block = 1 frame = frame_seqlen tokens, so the
        # resident window matches max_attention_size exactly (it need not be a whole
        # number of num_frame_per_block causal blocks).
        chunk_size = frame_seqlen
        window_chunks = self.ar_diffusion_kv_config.window_chunks or (max_attention_size // frame_seqlen)

        self.ar_diffusion_kv_config = dataclasses.replace(
            self.ar_diffusion_kv_config, chunk_size=chunk_size, window_chunks=window_chunks
        )
        free_bytes = torch.cuda.mem_get_info(self.device)[0]
        # Under CFG-parallel each rank executes exactly ONE branch (rank0 pos,
        # rank1 neg; the other branch's lazy contexts never allocate on this
        # rank), so its pool only needs one branch's capacity. A single-process
        # run (cfg world 1) executes both branches from one pool.
        try:
            from vllm_omni.diffusion.distributed.parallel_state import (
                get_classifier_free_guidance_world_size,
            )

            cfg_world = int(get_classifier_free_guidance_world_size())
        except Exception:
            cfg_world = 1
        local_branches = 1 if cfg_world >= 2 else 2
        logger.info(
            "AR-Diffusion preallocating (paged): frame_seqlen=%d num_frame_per_block=%d "
            "local_attn_size=%d -> chunk_size=%d window_chunks=%d (window=%d tokens)",
            frame_seqlen,
            num_frame_per_block,
            local_attn_size,
            chunk_size,
            window_chunks,
            window_chunks * chunk_size,
        )
        self.build_kv_cache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=self.od_config.dtype,
            block_size=chunk_size,  # one pool block per chunk
            max_model_len=1 << 20,
            available_bytes=free_bytes,
            cross_attn_length=cross_attn_length,
            cross_attn_img_length=cross_attn_img_length,
            local_branches=local_branches,
            num_frame_per_block=num_frame_per_block,
        )

    def execute_model(self, req: OmniDiffusionRequest, kv_prefetch_jobs: dict | None = None) -> DiffusionOutput:
        # KV disabled -> base behavior, unchanged.
        if self.kv_cache is None:
            return super().execute_model(req, kv_prefetch_jobs=kv_prefetch_jobs)

        kv = self.kv_cache
        # DreamZero KV is session-scoped (the model state persists across a
        # session's forwards), so the AR-Diffusion KV state is keyed by session_id and
        # reused — matching how pipeline.forward resolves the model-local state.
        extra_args = req.sampling_params.extra_args or {}
        session_id = str(extra_args.get("session_id") or "default")
        state = self._ar_diffusion_states.get(session_id)
        if state is None:
            pos = kv.begin_request(f"bde__{session_id}")
            neg = kv.begin_request(f"bde__{session_id}__neg")
            state = ARDiffusionKVState(kv, pos, neg, num_layers=kv.num_layers)
            self._ar_diffusion_states[session_id] = state
            # Evict the least-recently-used session(s) past the cap, freeing their
            # pool blocks — mirrors DreamZeroPipeline._states so the AR-Diffusion pool can't
            # outlive the model-local state map it shadows.
            while len(self._ar_diffusion_states) > self._max_ar_diffusion_states:
                old_id, old_state = self._ar_diffusion_states.popitem(last=False)
                old_state.close()
                logger.debug("AR-Diffusion evicted session=%s (LRU); freed pool blocks", old_id)
        # Track recency on every access (hit or miss) so the LRU order is correct.
        self._ar_diffusion_states.move_to_end(session_id)
        logger.debug(
            "AR-Diffusion execute_model: req=%s session=%s chunk_size=%d window_chunks=%s num_blocks=%d",
            req.request_id,
            session_id,
            kv.spec.chunk_size,
            kv.config.window_chunks,
            kv.num_blocks,
        )
        self.pipeline._ar_diffusion_kv_state = state
        # Time the worker-side forward (true per-request compute E2E). The CUDA
        # sync below makes the stop reflect completed GPU work, not just dispatch.
        _e2e_t0 = time.perf_counter()
        try:
            out = super().execute_model(req, kv_prefetch_jobs=kv_prefetch_jobs)
        except Exception:
            # Transactional containment: a forward that died partway may have
            # written some layers' K/V into allocated-but-uncommitted blocks
            # and advanced model-local state. Tear the whole session down
            # (engine pool blocks freed via close(); model-local state dropped)
            # so the next request with this session id starts clean instead of
            # tripping the pending-commit guard or reading half-written KV.
            broken = self._ar_diffusion_states.pop(session_id, None)
            if broken is not None:
                broken.close()
            states = getattr(self.pipeline, "_states", None)
            if states is not None:
                states.pop(session_id, None)
            logger.warning(
                "AR-Diffusion execute_model failed for session=%s; session state torn down "
                "(pool blocks freed) — the next request starts a fresh session",
                session_id,
            )
            raise
        finally:
            self.pipeline._ar_diffusion_kv_state = None
        if self.device is not None and torch.cuda.is_available():
            torch.accelerator.synchronize(self.device)
        self._perf_e2e_times.append(time.perf_counter() - _e2e_t0)
        return out

    # -- cuda-graph warm-up ----------------------------------------------------

    #: Dedicated session id for the warm-up rollout; cleaned from both state maps.
    _WARMUP_SID = "__ardiffusion_warmup__"

    def _synth_robot_obs(self, h: int, w: int, n_frames: int) -> dict:
        """Zero-filled roboarena observation matching the client's obs layout.

        Mirrors ``examples/.../client_schedule.make_obs_from_video`` (2 ext + wrist
        cameras, 7/6/1 state vectors) so it survives ``_transform_robot_obs`` exactly
        like a real request. One frame -> ``(H,W,3)``; chunk -> ``(n_frames,H,W,3)``.
        """
        img = np.zeros((h, w, 3), dtype=np.uint8) if n_frames == 1 else np.zeros((n_frames, h, w, 3), dtype=np.uint8)
        return {
            "observation/exterior_image_0_left": img,
            "observation/exterior_image_1_left": img,
            "observation/wrist_image_left": img,
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/cartesian_position": np.zeros(6, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": "warmup",
            "session_id": self._WARMUP_SID,
        }

    def _warmup_ar_rollout(self) -> None:
        """Drive a synthetic rollout through one window-fill so every DiT graph is
        captured at load (off the serving hot path). Frees the synthetic session and
        leaves the KV pool exactly as found. Never raises — falls back to lazy capture.
        """
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        kv = self.kv_cache
        if kv is None:
            return
        sid = self._WARMUP_SID
        t = self.pipeline.transformer
        nfpb = int(t.num_frame_per_block)
        window_chunks = int(kv.config.window_chunks or 0)
        if window_chunks <= 0:
            return
        # Distinct DiT window shapes are walked by one fill: prefill + ceil((W-1)/nfpb)
        # chunk forwards. +1 optionally to also capture the reset-cycle forward.
        n_forwards = 1 + math.ceil(max(0, window_chunks - 1) / nfpb)
        if self.ar_diffusion_kv_config.warmup_capture_reset:
            n_forwards += 1
        res = self._image_resolution()
        h, w = int(res[0]), int(res[1])

        free_before = kv.manager.block_pool.get_num_free_blocks()
        # The step cache is left active: the DiT input shape is per-chunk (the resident
        # window length), and the first denoise step of every chunk always computes, so
        # every window shape is captured regardless of how many later steps are skipped.
        logger.info(
            "AR-Diffusion cudagraph warm-up: up to %d synthetic forwards (window_chunks=%d, nfpb=%d)",
            n_forwards,
            window_chunks,
            nfpb,
        )
        try:
            for i in range(n_forwards):
                n_frames = 1 if i == 0 else 4  # client convention: 1-frame prefill, 4-frame chunks
                obs = self._synth_robot_obs(h, w, n_frames)
                sp = OmniDiffusionSamplingParams(extra_args={"reset": i == 0, "session_id": sid, "robot_obs": obs})
                req = OmniDiffusionRequest(prompts=["warmup"], sampling_params=sp, request_id=f"ardiffusion-warmup-{i}")
                self.execute_model(req)
                # Stop once the resident window is full — the remaining shapes are
                # already captured (the window caps/resets through the same set).
                ar_state = self._ar_diffusion_states.get(sid)
                if (
                    not self.ar_diffusion_kv_config.warmup_capture_reset
                    and ar_state is not None
                    and len(kv.window_block_ids(ar_state.pos)) >= window_chunks
                ):
                    break
        except Exception as e:  # noqa: BLE001 — warm-up must never break model load
            logger.warning("AR-Diffusion cudagraph warm-up failed (%s); using lazy capture.", e)
        finally:
            # Free the synthetic session from BOTH maps (KV pool + pipeline state).
            ar_state = self._ar_diffusion_states.pop(sid, None)
            if ar_state is not None:
                try:
                    ar_state.close()  # return KV blocks to the BlockPool
                except Exception as e:  # noqa: BLE001
                    logger.warning("AR-Diffusion warm-up: state.close() failed (%s)", e)
            try:
                self.pipeline._states.pop(sid, None)
            except Exception:  # noqa: BLE001
                pass
            # Drop any per-forward timings the warm-up recorded so serving stats start clean.
            self._perf_e2e_times.clear()
            free_after = kv.manager.block_pool.get_num_free_blocks()
            if free_after != free_before:
                logger.warning(
                    "AR-Diffusion warm-up: KV pool not restored (free %d -> %d) — possible leak",
                    free_before,
                    free_after,
                )
            else:
                logger.info(
                    "AR-Diffusion cudagraph warm-up complete; KV pool restored (%d free blocks)",
                    free_after,
                )

    def _image_resolution(self) -> list[int]:
        mc = getattr(self.od_config, "model_config", None) or {}
        psc = (mc.get("policy_server_config") if isinstance(mc, dict) else None) or {}
        return psc.get("image_resolution", [180, 320])
