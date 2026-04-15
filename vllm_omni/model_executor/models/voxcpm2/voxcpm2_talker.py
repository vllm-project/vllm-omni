# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VoxCPM2 AR talker — PagedAttention pipeline with per-request state.

Architecture:
  MiniCPM4PagedForVoxCPM2 (base_lm, 28 layers, PagedAttention + fp32 RoPE)
  → FSQ → MiniCPM4PagedResidualLM (8 layers, PagedAttention, no RoPE)
  → LocDiT (CFM solver) → AudioVAE → 48kHz waveform
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import time
from collections.abc import Iterable
from typing import Any

import librosa
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .minicpm4_paged import MiniCPM4PagedForVoxCPM2, MiniCPM4PagedResidualLM
from .voxcpm2_import_utils import import_voxcpm2_core

logger = init_logger(__name__)

_ENABLE_PROFILING = os.environ.get("VOXCPM2_PROFILE", "0") == "1"


def _encode_raw_audio(
    tts: nn.Module,
    samples: list[float] | torch.Tensor,
    sr: int,
    padding_mode: str = "right",
) -> torch.Tensor:
    """Encode raw audio samples using the native VoxCPM2 AudioVAE.

    Mirrors ``VoxCPM2Model._encode_wav`` but accepts in-memory samples
    instead of a file path (needed for the OpenAI speech API).
    """
    if isinstance(samples, list):
        audio = torch.tensor(samples, dtype=torch.float32)
    else:
        audio = samples.float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    encode_sr = tts._encode_sample_rate
    if sr != encode_sr:
        audio_np = audio.squeeze(0).numpy()
        audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=encode_sr)
        audio = torch.from_numpy(audio_np).unsqueeze(0)

    patch_len = tts.patch_size * tts.chunk_size
    if audio.size(1) % patch_len != 0:
        padding_size = patch_len - audio.size(1) % patch_len
        pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
        audio = torch.nn.functional.pad(audio, pad)

    feat = tts.audio_vae.encode(audio.to(tts.device), encode_sr).cpu()
    return feat.view(tts.audio_vae.latent_dim, -1, tts.patch_size).permute(1, 2, 0)


# ===================================================================
#  Per-request state
# ===================================================================


@dataclasses.dataclass
class _RequestState:
    request_id: str
    curr_embed_for_next: torch.Tensor | None = None
    prev_feat_embed: torch.Tensor | None = None
    curr_prefix_feat_cond: torch.Tensor | None = None
    last_audio_patch_gpu: torch.Tensor | None = None
    precomputed_stop_logits: torch.Tensor | None = None
    # Rolling tail of previously-decoded latents used as VAE receptive-field context.
    # Shape (n_pad_frames, feat_dim) on GPU. None before first decode.
    decode_pad: torch.Tensor | None = None
    # Audio chunks already emitted (CPU float32), concatenated for cumulative output.
    audio_chunks: list[torch.Tensor] = dataclasses.field(default_factory=list)
    decode_step_count: int = 0
    request_start_time: float = 0.0
    prefill_completed: bool = False
    prefill_text: str = ""
    prompt_cache: dict | None = None
    prefill_masks: tuple | None = None
    is_stopping: bool = False
    last_decoded_audio: torch.Tensor | None = None


@dataclasses.dataclass
class _CapturedGraph:
    graph: torch.cuda.CUDAGraph
    input_embeds: torch.Tensor
    positions: torch.Tensor
    output: torch.Tensor


# ===================================================================
#  Profiling timer
# ===================================================================


class _PerfTimer:
    __slots__ = ("_enabled", "_timers", "_counts", "_starts", "_pairs")

    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._timers: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._starts: dict[str, torch.cuda.Event] = {}
        self._pairs: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    def start(self, name: str) -> None:
        if not self._enabled:
            return
        evt = torch.cuda.Event(enable_timing=True)
        evt.record()
        self._starts[name] = evt

    def stop(self, name: str) -> None:
        if not self._enabled or name not in self._starts:
            return
        start_evt = self._starts.pop(name)
        end_evt = torch.cuda.Event(enable_timing=True)
        end_evt.record()
        self._pairs.append((name, start_evt, end_evt))

    def _resolve(self) -> None:
        if not self._pairs:
            return
        torch.cuda.synchronize()
        for name, s, e in self._pairs:
            self._timers[name] = self._timers.get(name, 0.0) + s.elapsed_time(e)
            self._counts[name] = self._counts.get(name, 0) + 1
        self._pairs.clear()

    def breakdown(self) -> str:
        if not self._enabled:
            return ""
        self._resolve()
        if not self._timers:
            return ""
        total = self._timers.get("decode_step", sum(self._timers.values()))
        lines = [
            "=== VoxCPM2 Decode Step Breakdown ===",
            f"{'Component':<30} | {'ms':>10} | {'%':>6} | {'N':>5} | {'avg':>8}",
            "-" * 70,
        ]
        for name in sorted(self._timers):
            t, c = self._timers[name], self._counts[name]
            lines.append(f"{name:<30} | {t:>10.2f} | {t / total * 100:>5.1f}% | {c:>5} | {t / c:>8.3f}")
        lines.append(f"{'TOTAL':<30} | {total:>10.2f} |")
        return "\n".join(lines)

    def reset(self) -> None:
        self._timers.clear()
        self._counts.clear()
        self._starts.clear()
        self._pairs.clear()


# ===================================================================
#  CFM pre-allocated buffers + optimized Euler solver
# ===================================================================


class _CFMBufferManager:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        feat_dim: int,
        patch_size: int,
        dit_hidden_size: int,
        max_batch_size: int = 1,
        sway_sampling_coef: float = 1.0,
    ):
        n = 2 * max_batch_size  # CFG doubles the batch
        self.x_in = torch.zeros(n, feat_dim, patch_size, device=device, dtype=dtype)
        self.mu_in = torch.zeros(n, dit_hidden_size, device=device, dtype=dtype)
        self.t_in = torch.zeros(n, device=device, dtype=dtype)
        self.dt_in = torch.zeros(n, device=device, dtype=dtype)
        self.cond_in = torch.zeros(n, feat_dim, patch_size, device=device, dtype=dtype)
        self.noise = torch.zeros(max_batch_size, feat_dim, patch_size, device=device, dtype=dtype)
        self._sway_coef = sway_sampling_coef
        self._device = device
        self._dtype = dtype
        self.t_span_10 = self._make_t_span(10)

    def _make_t_span(self, n: int) -> torch.Tensor:
        t = torch.linspace(1, 0, n + 1, device=self._device, dtype=self._dtype)
        return t + self._sway_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

    def get_t_span(self, n: int) -> torch.Tensor:
        return self.t_span_10 if n == 10 else self._make_t_span(n)


def _optimized_solve_euler(
    cfm_module: nn.Module,
    mu: torch.Tensor,
    patch_size: int,
    cond: torch.Tensor,
    n_timesteps: int,
    cfg_value: float,
    buffers: _CFMBufferManager,
    use_cfg_zero_star: bool = True,
    cfg_cutoff_ratio: float = 1.0,
    perf: _PerfTimer | None = None,
) -> torch.Tensor:
    estimator = cfm_module.estimator
    mean_mode = getattr(cfm_module, "mean_mode", False)
    b = mu.size(0)

    buffers.noise[:b].normal_()
    x = buffers.noise[:b].clone()

    t_span = buffers.get_t_span(n_timesteps)
    t, dt = t_span[0], t_span[0] - t_span[1]
    zero_init_steps = max(1, int(len(t_span) * 0.04))
    cfg_cutoff_step = max(zero_init_steps + 1, int(len(t_span) * cfg_cutoff_ratio))

    for step in range(1, len(t_span)):
        if use_cfg_zero_star and step <= zero_init_steps:
            dphi_dt = torch.zeros_like(x)
        elif step <= cfg_cutoff_step:
            buffers.x_in[:b].copy_(x)
            buffers.x_in[b : 2 * b].copy_(x)
            buffers.mu_in[:b].copy_(mu)
            buffers.mu_in[b : 2 * b].zero_()
            # Broadcast the 0-dim GPU scalar directly instead of
            # ``.fill_(t.item())`` — ``.item()`` forces a GPU->CPU sync.
            buffers.t_in[: 2 * b].copy_(t)
            if mean_mode:
                buffers.dt_in[: 2 * b].copy_(dt)
            else:
                buffers.dt_in.zero_()
            buffers.cond_in[:b].copy_(cond[:b])
            buffers.cond_in[b : 2 * b].copy_(cond[:b])

            if perf:
                perf.start("  cfm.estimator_cfg")
            raw_out = estimator(
                buffers.x_in[: 2 * b],
                buffers.mu_in[: 2 * b],
                buffers.t_in[: 2 * b],
                buffers.cond_in[: 2 * b],
                buffers.dt_in[: 2 * b],
            )
            if perf:
                perf.stop("  cfm.estimator_cfg")

            dphi_dt, cfg_dphi_dt = raw_out[:b], raw_out[b : 2 * b]
            if use_cfg_zero_star:
                pos = dphi_dt.reshape(b, -1)
                neg = cfg_dphi_dt.reshape(b, -1)
                st = torch.sum(pos * neg, 1, keepdim=True) / (torch.sum(neg**2, 1, keepdim=True) + 1e-8)
                st = st.view(b, *([1] * (len(dphi_dt.shape) - 1)))
            else:
                st = 1.0
            dphi_dt = cfg_dphi_dt * st + cfg_value * (dphi_dt - cfg_dphi_dt * st)
        else:
            buffers.x_in[:b].copy_(x)
            buffers.mu_in[:b].copy_(mu)
            # Broadcast the 0-dim GPU scalar; ``.fill_(t.item())`` would sync.
            buffers.t_in[:b].copy_(t)
            if mean_mode:
                buffers.dt_in[:b].copy_(dt)
            else:
                buffers.dt_in[:b].zero_()
            buffers.cond_in[:b].copy_(cond[:b])
            if perf:
                perf.start("  cfm.estimator_nocfg")
            dphi_dt = estimator(
                buffers.x_in[:b], buffers.mu_in[:b], buffers.t_in[:b], buffers.cond_in[:b], buffers.dt_in[:b]
            )
            if perf:
                perf.stop("  cfm.estimator_nocfg")

        x = x - dt * dphi_dt
        t = t - dt
        if step < len(t_span) - 1:
            dt = t - t_span[step + 1]
    return x


# ===================================================================
#  Main talker model
# ===================================================================


class VoxCPM2TalkerForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = True

        self.model = MiniCPM4PagedForVoxCPM2(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        self.residual_model = MiniCPM4PagedResidualLM(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "residual_model"),
        )
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        self._tts: nn.Module | None = None
        self._device = "cuda"
        self._side_dtype = torch.bfloat16

        self._patch_size = getattr(self.config, "patch_size", 4)
        self._feat_dim = getattr(self.config, "feat_dim", 64)
        self._sample_rate = getattr(self.config, "sample_rate", 48000)

        self._inference_timesteps = 10
        self._cfg_value = 2.0
        self._cfg_cutoff_ratio = 1.0
        # Number of trailing latent frames to keep as VAE receptive-field context
        # for sliding-window streaming decode. 12 matches the nanovllm reference
        # implementation and covers the longest VAE decoder receptive field.
        self._n_decode_pad_frames = 12
        self._enable_torch_compile = True
        self._compile_vae = True
        self._max_decode_steps = 2000
        self._max_batch_size = getattr(vllm_config.scheduler_config, "max_num_seqs", 4)

        self._perf = _PerfTimer(enabled=_ENABLE_PROFILING)
        self._cfm_buffers: _CFMBufferManager | None = None
        self._enable_cuda_graph = True
        self._scaffold_graphs: dict[int, _CapturedGraph] = {}
        self._residual_graphs: dict[int, _CapturedGraph] = {}
        self._max_cached_graphs = self._max_batch_size
        self._cuda_graph_pool: tuple | None = None
        self._cuda_graph_warmup_steps = 0
        self._cuda_graph_warmup_threshold = 3

        self._active_states: dict[str, _RequestState] = {}
        self._current_request_id: str | None = None
        self._pending_requests: list[tuple[str, bool, torch.Tensor | None, int]] = []
        self._results_queue: list[tuple[str, torch.Tensor | None]] = []
        self._audio_queue: list[tuple[str, Any]] = []
        self._deferred_cleanup_ids: set[str] = set()

    @property
    def tts(self) -> nn.Module:
        assert self._tts is not None, "Model not loaded yet"
        return self._tts

    # -------------------- request state management --------------------

    def _get_or_create_state(self, request_id: str) -> _RequestState:
        if request_id not in self._active_states:
            self._active_states[request_id] = _RequestState(request_id=request_id)
        return self._active_states[request_id]

    def _switch_to_request(self, request_id: str) -> _RequestState:
        if request_id != self._current_request_id:
            self._current_request_id = request_id
        return self._get_or_create_state(request_id)

    def _cleanup_request(self, request_id: str) -> None:
        self._active_states.pop(request_id, None)
        if self._current_request_id == request_id:
            self._current_request_id = None

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        # Defer cleanup: on_requests_finished is called before forward(),
        # so we must not delete state that the current step may still need.
        self._deferred_cleanup_ids.update(finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        for req_id in self._deferred_cleanup_ids:
            self._cleanup_request(req_id)
        self._deferred_cleanup_ids.clear()

    def _build_prompt_cache(
        self,
        ref_audio: Any = None,
        prompt_audio: Any = None,
        prompt_text: str | None = None,
    ) -> dict | None:
        """Build prompt cache, handling both file paths and raw audio data.

        The OpenAI speech API sends decoded audio as [samples_list, sr]
        via ``_resolve_ref_audio``, while offline usage sends file paths.
        """
        tts = self.tts

        def _is_raw_audio(v: Any) -> bool:
            import numbers

            return (
                isinstance(v, (list, tuple))
                and len(v) == 2
                and isinstance(v[1], numbers.Integral)
                and isinstance(v[0], (list, torch.Tensor))
            )

        if not _is_raw_audio(ref_audio) and not _is_raw_audio(prompt_audio):
            return tts.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_audio,
                reference_wav_path=ref_audio,
            )

        cache: dict[str, Any] = {}
        if ref_audio is not None:
            if _is_raw_audio(ref_audio):
                samples, sr = ref_audio
                cache["ref_audio_feat"] = _encode_raw_audio(tts, samples, sr)
            else:
                cache["ref_audio_feat"] = tts._encode_wav(ref_audio, padding_mode="right")

        if prompt_audio is not None and prompt_text is not None:
            cache["prompt_text"] = prompt_text
            if _is_raw_audio(prompt_audio):
                samples, sr = prompt_audio
                cache["audio_feat"] = _encode_raw_audio(tts, samples, sr, padding_mode="left")
            else:
                cache["audio_feat"] = tts._encode_wav(prompt_audio, padding_mode="left")

        has_ref = "ref_audio_feat" in cache
        has_prompt = "audio_feat" in cache
        if has_ref and has_prompt:
            cache["mode"] = "ref_continuation"
        elif has_ref:
            cache["mode"] = "reference"
        else:
            cache["mode"] = "continuation"

        return cache

    # -------------------- compile setup --------------------

    def _setup_cfm_buffers(self) -> None:
        if self._cfm_buffers is not None:
            return
        tts = self.tts
        dit_hidden = tts.lm_to_dit_proj.out_features + tts.res_to_dit_proj.out_features
        self._cfm_buffers = _CFMBufferManager(
            device=torch.device(self._device),
            dtype=self._side_dtype,
            feat_dim=self._feat_dim,
            patch_size=self._patch_size,
            dit_hidden_size=dit_hidden,
            max_batch_size=self._max_batch_size,
        )

    def _setup_torch_compile(self) -> None:
        if not self._enable_torch_compile:
            return
        tts = self.tts
        estimator = tts.feat_decoder.estimator
        if hasattr(estimator, "_compiled"):
            return

        targets: list[str] = []

        try:
            tts.feat_decoder.estimator = torch.compile(estimator, mode="reduce-overhead", fullgraph=False)
            tts.feat_decoder.estimator._compiled = True
            targets.append("LocDiT")
        except Exception as e:
            logger.warning("torch.compile LocDiT failed: %s", e)

        try:
            if not hasattr(tts.feat_encoder, "_compiled"):
                tts.feat_encoder = torch.compile(tts.feat_encoder, mode="reduce-overhead", fullgraph=False)
                tts.feat_encoder._compiled = True
                targets.append("feat_encoder")
        except Exception as e:
            logger.warning("torch.compile feat_encoder failed: %s", e)

        if self._compile_vae:
            try:
                if not hasattr(tts.audio_vae, "_compiled"):
                    tts.audio_vae.decode = torch.compile(tts.audio_vae.decode, mode="reduce-overhead", fullgraph=False)
                    tts.audio_vae._compiled = True
                    targets.append("AudioVAE")
            except Exception as e:
                logger.warning("torch.compile AudioVAE failed: %s", e)

        if not self._enable_cuda_graph:
            if not getattr(self.model, "_selective_compiled", False):
                try:
                    targets.extend(f"scaffold.{t}" for t in self.model.compile_selective())
                    self.model._selective_compiled = True
                except Exception as e:
                    logger.warning("scaffold compile failed: %s", e)

            if not getattr(self.residual_model, "_selective_compiled", False):
                try:
                    targets.extend(f"residual.{t}" for t in self.residual_model.compile_selective())
                    self.residual_model._selective_compiled = True
                except Exception as e:
                    logger.warning("residual compile failed: %s", e)
        else:
            self.model.precompute_fused_qkv()
            self.residual_model.precompute_fused_qkv()
            targets.append("scaffold+residual (CUDA Graph, skipping compile)")

        if not getattr(self, "_projections_compiled", False):
            try:
                self._compiled_dit_proj = torch.compile(self._dit_proj_fn, mode="default", fullgraph=True)
                self._compiled_stop_fn = torch.compile(self._stop_fn, mode="default", fullgraph=True)
                self._projections_compiled = True
                targets.append("projections")
            except Exception as e:
                self._compiled_dit_proj = self._compiled_stop_fn = None
                logger.warning("projections compile failed: %s", e)

        if targets:
            logger.info("VoxCPM2: torch.compile applied to: %s", ", ".join(targets))

    def _dit_proj_fn(self, lm_h: torch.Tensor, res_h: torch.Tensor) -> torch.Tensor:
        tts = self.tts
        return torch.cat([tts.lm_to_dit_proj(lm_h), tts.res_to_dit_proj(res_h)], dim=-1)

    def _stop_fn(self, lm_h: torch.Tensor) -> torch.Tensor:
        tts = self.tts
        return tts.stop_head(tts.stop_actn(tts.stop_proj(lm_h)))

    def _get_cuda_graph_pool(self) -> tuple:
        if self._cuda_graph_pool is None:
            self._cuda_graph_pool = torch.cuda.graph_pool_handle()
        return self._cuda_graph_pool

    @staticmethod
    def _nullify_volatile_metadata(ctx: Any) -> Any:
        """Set ``scheduler_metadata`` to None on all attention layers.

        This is the only tensor FA3 reallocates each step (variable shape).
        All other metadata tensors are persistent model-runner buffers.
        Setting it to None makes FA3 use default scheduling (~0.1ms cost).
        """
        if not isinstance(ctx.attn_metadata, dict):
            return ctx

        ctx = copy.copy(ctx)
        new_meta: dict[str, Any] = {}
        for layer_name, meta in ctx.attn_metadata.items():
            if getattr(meta, "scheduler_metadata", None) is not None:
                meta = copy.copy(meta)
                meta.scheduler_metadata = None
            new_meta[layer_name] = meta
        ctx.attn_metadata = new_meta
        return ctx

    def _capture_graph(
        self,
        model: nn.Module,
        batch_size: int,
        label: str,
        is_residual: bool = False,
    ) -> _CapturedGraph:
        """Capture a CUDA Graph for *model* at *batch_size*."""
        hidden_size = self.config.hidden_size
        dtype = self._side_dtype
        dev = torch.device(self._device)
        pool = self._get_cuda_graph_pool()

        model.precompute_fused_qkv()

        g = _CapturedGraph(
            graph=torch.cuda.CUDAGraph(),
            input_embeds=torch.zeros(batch_size, hidden_size, device=dev, dtype=dtype),
            positions=torch.zeros(batch_size, device=dev, dtype=torch.long),
            output=torch.zeros(batch_size, hidden_size, device=dev, dtype=dtype),
        )

        if is_residual:
            call_kwargs = dict(positions=g.positions, inputs_embeds=g.input_embeds)
        else:
            call_kwargs = dict(input_ids=None, positions=g.positions, inputs_embeds=g.input_embeds)

        ctx = get_forward_context()
        patched_ctx = self._nullify_volatile_metadata(ctx)

        with override_forward_context(patched_ctx):
            for _ in range(3):
                _ = model(**call_kwargs)

            with torch.cuda.graph(g.graph, pool=pool):
                g.output = model(**call_kwargs)

        logger.info("CUDA Graph captured for %s (batch_size=%d)", label, batch_size)
        return g

    def _replay_graph(
        self,
        g: _CapturedGraph,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Copy fresh inputs into static buffers, then replay.

        No metadata copy needed: persistent buffers (seq_lens, slot_mapping,
        etc.) are updated in-place by the model runner.  scheduler_metadata
        was nullified at capture time so no kernel references it.
        """
        g.input_embeds[:batch_size].copy_(inputs_embeds[:batch_size])
        g.positions[:batch_size].copy_(positions[:batch_size])
        g.graph.replay()
        return g.output[:batch_size].clone()

    # -------------------- vllm hooks --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        self._perf.start("forward_total")
        dev = input_ids.device

        num_reqs = len(self._pending_requests)
        num_decode = sum(1 for _, is_p, _, n in self._pending_requests if not is_p and n == 1)
        is_all_decode = num_decode == num_reqs and num_reqs > 0

        tts_compiled = getattr(self.tts.feat_decoder.estimator, "_compiled", False) if self._tts is not None else False
        graph_ready = tts_compiled and self._cuda_graph_warmup_steps >= self._cuda_graph_warmup_threshold
        if num_decode > 0:
            self._cuda_graph_warmup_steps += 1

        can_use_graph = (
            self._enable_cuda_graph and graph_ready and intermediate_tensors is None and inputs_embeds is not None
        )

        if can_use_graph and is_all_decode and num_reqs <= self._max_cached_graphs:
            self._perf.start("scaffold_fwd")
            if num_reqs not in self._scaffold_graphs:
                self._scaffold_graphs[num_reqs] = self._capture_graph(self.model, num_reqs, "scaffold")
            scaffold_hidden = self._replay_graph(self._scaffold_graphs[num_reqs], inputs_embeds, positions, num_reqs)
            self._perf.stop("scaffold_fwd")

        else:
            self._perf.start("scaffold_fwd")
            model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
            self._perf.stop("scaffold_fwd")
            if isinstance(model_output, IntermediateTensors):
                return model_output
            scaffold_hidden = model_output
            if isinstance(scaffold_hidden, tuple):
                scaffold_hidden = scaffold_hidden[0]

        # Phase 1: per-request FSQ + residual input
        token_offset = 0
        residual_inputs: list[torch.Tensor] = []
        residual_positions: list[torch.Tensor] = []
        req_metas: list[tuple] = []

        for req_id, is_prefill, _req_embeds, n in self._pending_requests:
            state = self._switch_to_request(req_id)
            req_hidden = scaffold_hidden[token_offset : token_offset + n]
            req_pos = positions[token_offset : token_offset + n]

            if is_prefill:
                res_input, meta = self._prepare_residual_prefill(state, req_hidden, dev)
            elif state.prefill_completed:
                res_input, meta = self._prepare_residual_decode(state, req_hidden, dev)
            else:
                token_offset += n
                self._results_queue.append((req_id, None))
                self._audio_queue.append((req_id, None))
                continue

            residual_inputs.append(res_input)
            residual_positions.append(req_pos)
            req_metas.append((state, is_prefill, meta))
            token_offset += n

        # Phase 2: batch residual_model forward
        if residual_inputs:
            batch_in = torch.cat(residual_inputs, dim=0)
            batch_pos = torch.cat(residual_positions, dim=0)

            residual_batch_size = batch_in.shape[0]
            use_residual_graph = (
                self._enable_cuda_graph
                and is_all_decode
                and graph_ready
                and residual_batch_size == num_reqs  # 1 token per request
                and residual_batch_size <= self._max_cached_graphs
            )

            self._perf.start("residual_fwd")
            if use_residual_graph:
                if residual_batch_size not in self._residual_graphs:
                    self._residual_graphs[residual_batch_size] = self._capture_graph(
                        self.residual_model, residual_batch_size, "residual", is_residual=True
                    )
                batch_out = self._replay_graph(
                    self._residual_graphs[residual_batch_size], batch_in, batch_pos, residual_batch_size
                )
            else:
                batch_out = self.residual_model(batch_pos, batch_in)
            self._perf.stop("residual_fwd")

            # Phase 3: per-request LocDiT + update
            offset = 0
            for idx, (state, is_prefill, meta) in enumerate(req_metas):
                n = residual_inputs[idx].shape[0]
                res_out = batch_out[offset : offset + n]
                offset += n

                if is_prefill:
                    self._finish_prefill(state, meta, res_out, dev)
                else:
                    self._finish_decode(state, meta, res_out, dev)

                self._results_queue.append((state.request_id, state.precomputed_stop_logits))
                self._audio_queue.append((state.request_id, self._collect_audio(state)))

        self._pending_requests.clear()
        self._flush_deferred_cleanup()
        self._perf.stop("forward_total")
        return scaffold_hidden

    # -------------------- prefill / decode helpers --------------------

    def _prepare_residual_prefill(self, state: _RequestState, base_lm_out: torch.Tensor, dev: Any):
        tts = self.tts
        text_mask, feat_mask, feat, feat_embed = state.prefill_masks
        state.prefill_masks = None

        tts_len = text_mask.shape[1]
        scaffold_len = base_lm_out.shape[0]

        if scaffold_len < tts_len:
            # Voice clone / continuation: scaffold only processed vllm tokens.
            # Pad to match TTS sequence length (extra positions are masked out).
            pad = torch.zeros(
                tts_len - scaffold_len,
                base_lm_out.shape[-1],
                device=base_lm_out.device,
                dtype=base_lm_out.dtype,
            )
            enc_out = torch.cat([base_lm_out, pad], dim=0).unsqueeze(0)
        else:
            enc_out = base_lm_out.unsqueeze(0)

        prefix_feat_cond = (
            feat[:, -1, ...]
            if feat.shape[1] > 0
            else torch.zeros(1, self._patch_size, self._feat_dim, device=dev, dtype=self._side_dtype)
        )
        enc_outputs = tts.fsq_layer(enc_out) * feat_mask.unsqueeze(-1) + enc_out * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

        residual_input = tts.fusion_concat_proj(torch.cat([enc_outputs, feat_mask.unsqueeze(-1) * feat_embed], dim=-1))
        meta = {"lm_hidden": lm_hidden, "prefix_feat_cond": prefix_feat_cond}
        return residual_input.squeeze(0), meta

    def _prepare_residual_decode(self, state: _RequestState, base_lm_out: torch.Tensor, dev: Any):
        tts = self.tts
        state.decode_step_count += 1

        if state.decode_step_count >= self._max_decode_steps:
            logger.warning("MAX_DECODE_STEPS for %s (%d), forcing stop", state.request_id, state.decode_step_count)
            state.is_stopping = True

        h = base_lm_out.unsqueeze(0) if base_lm_out.ndim == 1 else base_lm_out
        lm_h = tts.fsq_layer(h)
        if lm_h.ndim == 1:
            lm_h = lm_h.unsqueeze(0)

        prev = state.prev_feat_embed.to(self._side_dtype)
        if prev.ndim == 1:
            prev = prev.unsqueeze(0)
        res_input = tts.fusion_concat_proj(torch.cat([lm_h, prev], dim=-1))
        return res_input, {"new_lm_hidden": lm_h}

    def _run_cfm(self, dit_h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self._cfm_buffers is not None:
            return _optimized_solve_euler(
                self.tts.feat_decoder,
                dit_h,
                self._patch_size,
                cond,
                self._inference_timesteps,
                self._cfg_value,
                self._cfm_buffers,
                cfg_cutoff_ratio=self._cfg_cutoff_ratio,
                perf=self._perf,
            ).transpose(1, 2)
        return self.tts.feat_decoder(
            mu=dit_h,
            patch_size=self._patch_size,
            cond=cond,
            n_timesteps=self._inference_timesteps,
            cfg_value=self._cfg_value,
        ).transpose(1, 2)

    def _finish_prefill(self, state: _RequestState, meta: dict, res_out: torch.Tensor, dev: Any):
        tts = self.tts
        lm_hidden = meta["lm_hidden"]
        prefix_feat_cond = meta["prefix_feat_cond"]
        residual_hidden = res_out[-1:, :]

        state.precomputed_stop_logits = tts.stop_head(tts.stop_actn(tts.stop_proj(lm_hidden))).detach()
        dit_h = torch.cat([tts.lm_to_dit_proj(lm_hidden), tts.res_to_dit_proj(residual_hidden)], dim=-1)

        self._setup_cfm_buffers()
        if self._enable_torch_compile:
            self._setup_torch_compile()

        pred_feat = self._run_cfm(dit_h, prefix_feat_cond.transpose(1, 2).contiguous())

        with torch.no_grad():
            curr_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)

        state.curr_embed_for_next = curr_embed.detach()
        state.prev_feat_embed = curr_embed.detach()
        state.curr_prefix_feat_cond = pred_feat[0].detach()
        state.last_audio_patch_gpu = pred_feat.detach()
        state.decode_step_count = 0
        state.request_start_time = time.perf_counter()
        state.prefill_completed = True

        if logger.isEnabledFor(logging.DEBUG):
            # Only compute the norm (which forces a GPU->CPU sync) if we will log it.
            logger.debug("PREFILL[%s]: patch norm=%.4f", state.request_id, pred_feat.norm().item())
        self._perf.reset()

    def _finish_decode(self, state: _RequestState, meta: dict, res_out: torch.Tensor, dev: Any):
        self._perf.start("decode_step")
        tts = self.tts

        lm_h = meta["new_lm_hidden"]
        res_h = res_out.unsqueeze(0) if res_out.ndim == 1 else res_out

        dit_proj = getattr(self, "_compiled_dit_proj", None) or self._dit_proj_fn
        stop_fn = getattr(self, "_compiled_stop_fn", None) or self._stop_fn

        dit_h = dit_proj(lm_h, res_h)
        pfc = state.curr_prefix_feat_cond.to(self._side_dtype)
        if pfc.ndim == 2:
            pfc = pfc.unsqueeze(0)

        pred_feat = self._run_cfm(dit_h, pfc.transpose(1, 2).contiguous())
        next_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)

        state.precomputed_stop_logits = stop_fn(lm_h).detach()
        state.curr_embed_for_next = next_embed.detach()
        state.prev_feat_embed = next_embed.detach()
        state.curr_prefix_feat_cond = pred_feat[0].detach()
        state.last_audio_patch_gpu = pred_feat.detach()

        self._perf.stop("decode_step")
        if _ENABLE_PROFILING and state.decode_step_count % 20 == 0:
            logger.info("Step %d[%s]:\n%s", state.decode_step_count, state.request_id, self._perf.breakdown())

    # -------------------- audio collection --------------------

    def _collect_audio(self, state: _RequestState) -> torch.Tensor | None:
        """Per-step sliding-window VAE decode (nanovllm pattern).

        Each decode step feeds ``[decode_pad, new_patch]`` through the VAE
        and slices out only the audio region corresponding to the new patch.
        The pad buffer (last ``_n_decode_pad_frames`` latent frames) provides
        the receptive-field context needed by the VAE's transposed convolutions,
        eliminating boundary artifacts between chunks.

        Returns the delta audio chunk (not cumulative) so the output processor
        can stream each chunk to the client independently.
        """
        patch = state.last_audio_patch_gpu
        if patch is None:
            return None
        state.last_audio_patch_gpu = None

        # patch shape: (patch_size, feat_dim) or (1, patch_size, feat_dim)
        new_latent = patch.reshape(-1, self._feat_dim).to(torch.float32)
        n_new = new_latent.shape[0]  # = patch_size (typically 4)

        self._perf.start("vae_decode")

        # Build VAE input: [pad_frames | new_latent]
        if state.decode_pad is not None:
            vae_input = torch.cat([state.decode_pad, new_latent], dim=0)
            pad_frames = state.decode_pad.shape[0]
        else:
            vae_input = new_latent
            pad_frames = 0

        # VAE decode: (1, feat_dim, T_frames) -> (1, 1, T_samples)
        feat = vae_input.unsqueeze(0).transpose(1, 2).contiguous()
        with torch.no_grad():
            audio = self.tts.audio_vae.decode(feat.to(self._device)).reshape(-1)

        # Slice out only the new audio (after the pad region).
        # Each latent frame maps to decoder_chunk_size audio samples.
        dcs = int(getattr(self.tts.audio_vae, "decode_chunk_size", audio.numel() // vae_input.shape[0]))
        new_audio = audio[pad_frames * dcs : (pad_frames + n_new) * dcs].detach().cpu().float()

        # Roll the pad buffer: keep last N latent frames as context for next step.
        all_latents = vae_input  # [pad + new]
        state.decode_pad = all_latents[-self._n_decode_pad_frames :].detach()

        state.audio_chunks.append(new_audio)
        state.last_decoded_audio = new_audio
        self._perf.stop("vae_decode")
        return new_audio

    # -------------------- compute_logits --------------------

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        bsz = hidden_states.shape[0]
        logits = torch.full(
            (bsz, self.config.vocab_size), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype
        )

        if self._results_queue:
            for i, (req_id, stop_logits) in enumerate(self._results_queue):
                if i >= bsz:
                    break
                state = self._active_states.get(req_id)
                if stop_logits is not None:
                    if state is not None and state.is_stopping:
                        logits[i, 0] = 0.0
                        logits[i, 1] = 1.0
                        state.precomputed_stop_logits = None
                    else:
                        logits[i, 0] = stop_logits[0, 0]
                        logits[i, 1] = stop_logits[0, 1]
                        if state is not None:
                            state.is_stopping = bool(stop_logits[0, 1] > stop_logits[0, 0])
                            state.precomputed_stop_logits = None
                elif state and state.prefill_completed:
                    logits[i, 1] = 1.0
                else:
                    logits[i, 0] = 1.0
            self._results_queue.clear()
        else:
            logits[:, 0] = 1.0
        return logits

    # -------------------- omni output --------------------

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        mm: dict[str, Any] = {}
        if self._audio_queue:
            audio_by_req = {rid: audio for rid, audio in self._audio_queue}
            order = [r for r, _ in self._audio_queue]
            mm["model_outputs"] = [audio_by_req.get(r) for r in order]
            mm["sr"] = [torch.tensor(self._sample_rate, dtype=torch.int32) for _ in order]
            self._audio_queue.clear()

        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=mm)

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info_dict: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        additional = info_dict.get("additional_information")
        if isinstance(additional, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in additional.items():
                merged.setdefault(k, v)
            info_dict = merged

        span_len = int(input_ids.shape[0])
        dev = input_ids.device
        req_id = info_dict.get("request_id", "default")
        is_prefill = span_len > 1

        if is_prefill:
            # Evict stale states
            pending_ids = {rid for rid, *_ in self._pending_requests}
            pending_ids.add(req_id)
            if self._current_request_id:
                pending_ids.add(self._current_request_id)
            for rid in [r for r, s in self._active_states.items() if r not in pending_ids and s.prefill_completed]:
                self._cleanup_request(rid)

            # VoxCPM2Tokenizer does char-level Chinese splitting, so use input_ids directly
            token_ids = input_ids.tolist()
            if token_ids and token_ids[0] == self.config.bos_token_id:
                token_ids = token_ids[1:]

            state = self._get_or_create_state(req_id)
            state.prefill_text = ""
            state.decode_pad = None
            state.audio_chunks = []
            state.prefill_completed = False
            state.decode_step_count = 0
            state.precomputed_stop_logits = None
            state.last_audio_patch_gpu = None
            state.curr_embed_for_next = None
            state.prev_feat_embed = None
            state.curr_prefix_feat_cond = None
            state.is_stopping = False
            state.last_decoded_audio = None

            # Voice clone / continuation
            ref_audio = info_dict.get("reference_audio") or info_dict.get("ref_audio")
            prompt_audio = info_dict.get("prompt_audio")
            prompt_text = info_dict.get("prompt_text")
            if isinstance(ref_audio, list):
                ref_audio = ref_audio[0] if ref_audio else None
            if isinstance(prompt_audio, list):
                prompt_audio = prompt_audio[0] if prompt_audio else None
            if isinstance(prompt_text, list):
                prompt_text = prompt_text[0] if prompt_text else None

            state.prompt_cache = None
            if ref_audio or (prompt_audio and prompt_text):
                try:
                    state.prompt_cache = self._build_prompt_cache(
                        ref_audio=ref_audio,
                        prompt_audio=prompt_audio,
                        prompt_text=prompt_text,
                    )
                except Exception as e:
                    logger.warning("build_prompt_cache failed: %s", e)

            inputs = self._build_prefill_inputs(token_ids, dev, req_id)
            tts = self.tts
            feat_embed = tts.enc_to_lm_proj(tts.feat_encoder(inputs["audio_feat"]))
            text_embed = self.model.embed_input_ids(inputs["text_token"].to(dev))
            text_mask, feat_mask = inputs["text_mask"], inputs["audio_mask"]
            embeds = (text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed).squeeze(0)
            state.prefill_masks = (text_mask, feat_mask, inputs["audio_feat"], feat_embed)
        else:
            state = self._active_states.get(req_id)
            curr = state.curr_embed_for_next if state else None
            if curr is not None:
                embeds = curr.to(dev, dtype=self._side_dtype).reshape(1, -1)
            else:
                embeds = torch.zeros(1, self.config.hidden_size, device=dev, dtype=self._side_dtype)

        self._pending_requests.append((req_id, is_prefill, embeds, span_len))
        return input_ids, embeds, {}

    def postprocess(self, hidden_states: torch.Tensor, **info: Any) -> dict[str, Any]:
        req_id = info.get("request_id", self._current_request_id or "default")
        if _ENABLE_PROFILING:
            state = self._active_states.get(req_id)
            if state and state.decode_step_count > 0:
                logger.info(
                    "REQUEST DONE[%s]: %d steps, %.2fs\n%s",
                    req_id,
                    state.decode_step_count,
                    time.perf_counter() - state.request_start_time,
                    self._perf.breakdown(),
                )
        return {}

    # -------------------- build prefill inputs --------------------

    def _build_prefill_inputs(self, token_ids: list[int], dev: Any, req_id: str = "default") -> dict:
        tts = self.tts
        dtype = self._side_dtype
        state = self._active_states.get(req_id)
        cache = state.prompt_cache if state else None
        mode = cache.get("mode", "continuation") if cache else "zero_shot"

        if cache and mode in ("continuation", "ref_continuation"):
            prompt_text = cache.get("prompt_text", "")
            prompt_ids = list(tts.text_tokenizer(prompt_text)) if prompt_text else []
            all_ids = prompt_ids + token_ids
        else:
            all_ids = token_ids

        text_token = torch.tensor(all_ids, dtype=torch.int32)
        text_token = torch.cat([text_token, torch.tensor([tts.audio_start_token], dtype=torch.int32)], dim=-1)
        text_len = text_token.shape[0]
        latent_dim = tts.audio_vae.latent_dim
        ps = self._patch_size

        if mode in ("zero_shot", "continuation"):
            audio_feat = cache["audio_feat"] if cache else torch.empty((0, ps, latent_dim), dtype=torch.float32)
            a_len = audio_feat.size(0)
            text_token = torch.cat([text_token, torch.zeros(a_len, dtype=torch.int32)])
            audio_feat = torch.cat([torch.zeros((text_len, ps, latent_dim), dtype=torch.float32), audio_feat])
            text_mask = torch.cat([torch.ones(text_len, dtype=torch.int32), torch.zeros(a_len, dtype=torch.int32)])
            audio_mask = torch.cat([torch.zeros(text_len, dtype=torch.int32), torch.ones(a_len, dtype=torch.int32)])
        elif mode == "reference":
            ref = cache["ref_audio_feat"]
            rt, rf, rtm, ram = tts._make_ref_prefix(ref, text_token.device)
            text_token = torch.cat([rt.cpu(), text_token])
            audio_feat = torch.cat([rf.cpu(), torch.zeros((text_len, ps, latent_dim), dtype=torch.float32)])
            text_mask = torch.cat([rtm.cpu(), torch.ones(text_len, dtype=torch.int32)])
            audio_mask = torch.cat([ram.cpu(), torch.zeros(text_len, dtype=torch.int32)])
        else:  # ref_continuation
            ref = cache["ref_audio_feat"]
            prompt = cache["audio_feat"]
            p_len = prompt.size(0)
            rt, rf, rtm, ram = tts._make_ref_prefix(ref, text_token.device)
            text_token = torch.cat([rt.cpu(), text_token, torch.zeros(p_len, dtype=torch.int32)])
            audio_feat = torch.cat([rf.cpu(), torch.zeros((text_len, ps, latent_dim), dtype=torch.float32), prompt])
            ones_t = torch.ones(text_len, dtype=torch.int32)
            zeros_p = torch.zeros(p_len, dtype=torch.int32)
            zeros_t = torch.zeros(text_len, dtype=torch.int32)
            ones_p = torch.ones(p_len, dtype=torch.int32)
            text_mask = torch.cat([rtm.cpu(), ones_t, zeros_p])
            audio_mask = torch.cat([ram.cpu(), zeros_t, ones_p])

        return {
            "text_token": text_token.unsqueeze(0).to(dev),
            "audio_feat": audio_feat.unsqueeze(0).to(dev).to(dtype),
            "text_mask": text_mask.unsqueeze(0).to(dev),
            "audio_mask": audio_mask.unsqueeze(0).to(dev),
        }

    # -------------------- weight loading --------------------

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"base_lm.": "model."})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _base_lm_only(ws):
            for name, tensor in ws:
                if name.startswith("base_lm."):
                    yield name, tensor

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(_base_lm_only(weights), mapper=self.hf_to_vllm_mapper)

        model_path = self.vllm_config.model_config.model
        VoxCPM = import_voxcpm2_core()
        native = VoxCPM.from_pretrained(model_path, load_denoiser=False, optimize=False)
        self._tts = native.tts_model.to("cuda")
        self._side_dtype = self._tts.fusion_concat_proj.weight.dtype
        self._device = "cuda"
        self._patch_size = self._tts.patch_size
        self._feat_dim = self._tts.feat_dim

        n = self.residual_model.load_weights_from_native(self._tts.residual_lm)
        for name, _ in self.residual_model.named_parameters():
            loaded.add(f"residual_model.{name}")
        logger.info("VoxCPM2: loaded %d params into paged residual_model", n)

        del self._tts.base_lm
        self._tts.base_lm = None
        del self._tts.residual_lm
        self._tts.residual_lm = None
        torch.cuda.empty_cache()

        logger.info(
            "Loaded VoxCPM2 (patch=%d, feat_dim=%d, dtype=%s)", self._patch_size, self._feat_dim, self._side_dtype
        )
        return loaded
