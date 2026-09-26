# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2-3B text-to-music model for vllm-omni: single-stage native-AR.

The checkpoint is one Mixture-of-Transformers: a Qwen3-shaped AR path (which
is exactly a Qwen3-1.7B-class backbone over an extended vocabulary) plus a
parallel NAR path (``nar_self_attn``/``nar_mlp`` per layer) that a 32-step
midpoint flow-matching solver walks over 64-dim VAE latents. Both paths and
the projection heads live in the same ``model.safetensors``, so a single
vLLM stage loads everything once and no weights are duplicated.

Sampling is model-owned (``prefer_model_sampler``), reproducing the upstream
request-local arithmetic: per-phase vocabulary masking, windowed repetition
penalty, seeded multinomial, ``min_tokens`` end-token suppression. One song
is one or two engine requests — the abc phase (``cot=full|melody``) first,
then the semantic phase; the driver stitches them. When the semantic phase's
end token is drawn (or its frame budget is hit), the model solves the ODE and
decodes 48 kHz stereo audio inside the step and ships it as the request's
final multimodal payload, so the whole song arrives in one piece.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .constants import (
    ABC_END,
    ABC_SAMPLING,
    CODEC_OFFSET,
    DEFAULT_VAE_ID,
    KEY_MAX_AUDIO_FRAMES,
    KEY_MIN_TOKENS,
    KEY_PENALTY_WINDOW,
    KEY_PHASE,
    KEY_PREFIX_IDS,
    KEY_REPETITION_PENALTY,
    KEY_SEED,
    KEY_SKIP_SYNTHESIS,
    KEY_TEMPERATURE,
    KEY_TOP_K,
    KEY_TOP_P,
    LATENT_DIM,
    MUSIC_END,
    ODE_STEPS,
    SAMPLE_RATE,
    SEMANTIC_SAMPLING,
    VAE_CORE_FRAMES,
    VAE_HALO_FRAMES,
)
from .nar import synthesize
from .sampling import distribution, sample_row
from .vae import YuE2VAE
from .weights import partition_checkpoint_weights

logger = init_logger(__name__)

__all__ = ["Yue2ForCausalLM"]

DEFAULT_SEED = 831001


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class _RotaryEmbedding(nn.Module):
    """Checkpoint-compatible RoPE: cos/sin over [B, T, head_dim/2]."""

    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._inv_freq: torch.Tensor | None = None

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        if self._inv_freq is None or self._inv_freq.device != device:
            self._inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim)
            )
        angles = position_ids.float().unsqueeze(-1) * self._inv_freq
        return angles.cos(), angles.sin()


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class _NARAttention(nn.Module):
    """NAR-path attention with checkpoint-native separate projections."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, config.rms_norm_eps)

    def project_qkv(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        return _apply_rotary(q, rc, rs), _apply_rotary(k, rc, rs), v


class _NARMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class _NARLayer(nn.Module):
    """One NAR path layer; checkpoint ``model.layers.N.nar_*`` remaps onto it."""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = _NARAttention(config)
        self.pre_mlp_layernorm = _RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = _NARMLP(config)


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb.to(next(self.parameters()).dtype))


class _AudioPositionEmbedding(nn.Module):
    def __init__(self, max_frames: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(max_frames, hidden_size)
        position = torch.arange(0, max_frames, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, position_ids):
        return self.pe[position_ids]


def _request_key(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


@dataclass
class _RowConstants:
    """Per-request values fixed at admission (a prefill step, never replayed)."""

    request_id: str
    phase: str
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    penalty_window: int
    min_tokens: int
    max_audio_frames: int
    seed: int
    skip_synthesis: bool


@dataclass
class _RequestState:
    """Everything one song request accumulates across decode steps."""

    request_id: str
    constants: _RowConstants
    prompt_tokens: int = 0
    prompt_len: int = 0  # full prompt length; a completing prefill row reaches it
    prefix_ids: list[int] = field(default_factory=list)
    history: list[int] = field(default_factory=list)
    generator: torch.Generator | None = None
    finished: bool = False
    truncated: bool = False


class Yue2ForCausalLM(nn.Module):
    """YuE2-3B: vLLM-native Qwen3 AR backbone + NAR flow-matching + VAE."""

    have_multimodal_outputs = True
    prefer_model_sampler = True
    has_postprocess = False
    # The song is decoded in-model and shipped via make_omni_output; per-step
    # hidden states must NOT ride the pooler payload, or the output pipeline
    # remaps the accumulated "hidden" rows to the audio modality key and the
    # driver receives hidden states instead of the decoded waveform
    # (single-stage precedent: minimax_music3 talker).
    omni_pooler_payload_include_hidden: bool = False
    # No downstream stage consumes prefix-cached hidden/mm tensors for YuE2;
    # opting out keeps the merged prefix-cache view from rebuilding "hidden"
    # payloads (KV-block prefix caching for cot=full stays on).
    requires_full_prefix_cached_hidden_states: bool = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        config = vllm_config.model_config.hf_config
        self.config = config
        hidden = int(config.hidden_size)

        self.model = Qwen3Model(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                int(config.vocab_size),
                hidden,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(int(config.vocab_size))

        self.nar_layers = nn.ModuleList([_NARLayer(config) for _ in range(int(config.num_hidden_layers))])
        self.vae2llm = nn.Linear(int(getattr(config, "latent_dim", LATENT_DIM)), hidden)
        self.llm2vae = nn.Linear(hidden, int(getattr(config, "latent_dim", LATENT_DIM)))
        self.time_embedder = _TimestepEmbedder(hidden)
        self.latent_pos_embed = _AudioPositionEmbedding(
            int(getattr(config, "max_latent_frames", config.max_position_embeddings)),
            hidden,
        )
        self.rotary_emb = _RotaryEmbedding(int(config.head_dim), float(config.rope_theta))

        self.vocab_size = int(config.vocab_size)
        self.max_position_embeddings = int(config.max_position_embeddings)
        self.max_latent_frames = int(getattr(config, "max_latent_frames", self.max_position_embeddings))
        self._timestep_shift = float(getattr(config, "timestep_shift", 1.0))

        self._states: dict[str, _RequestState] = {}
        self._row_constants: dict[str, _RowConstants] = {}
        self._audio_queue: list[tuple[str, torch.Tensor, bool]] = []
        self._deferred_cleanup_ids: set[str] = set()
        self._step_rows: list[tuple[str, int, int]] = []  # (req_id, computed, scheduled)
        self._decode_t0: dict[str, float] = {}  # first decode-step wall time, for AR tok/s
        self._last_mm: dict[str, Any] | None = None  # current step's make_omni_output payload
        self._vae: YuE2VAE | None = None
        self._vae_device: torch.device | None = None

    # ------------------------------------------------------------ sampling

    def shift_t(self, raw_t: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t_sig = torch.sigmoid(torch.tensor(raw_t, dtype=dtype, device=device))
        shift = self._timestep_shift
        return shift * t_sig / (1 + (shift - 1) * t_sig)

    def rotary(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(positions)

    def ar_project(self, layer, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """AR-path Q/K/V from the vLLM backbone's fused projection."""
        attn = layer.self_attn
        qkv = attn.qkv_proj(x)
        if isinstance(qkv, tuple):
            qkv = qkv[0]
        num_heads, num_kv_heads, head_dim = attn.num_heads, attn.num_kv_heads, attn.head_dim
        q, k, v = torch.split(
            qkv,
            [num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim],
            dim=-1,
        )
        T = x.shape[-2]
        q = attn.q_norm(q.view(T, num_heads, head_dim))
        k = attn.k_norm(k.view(T, num_kv_heads, head_dim))
        rc, rs = cos.unsqueeze(2), sin.unsqueeze(2)
        # q/k are [T, heads, head_dim] (3D); rc/rs are [1, T, 1, head_dim/2]
        # (4D). Broadcasting them directly promotes q/k to 4D, which the NAR
        # attention rejects. Add the batch axis explicitly, then drop it.
        q = _apply_rotary(q.unsqueeze(0), rc, rs).squeeze(0)
        k = _apply_rotary(k.unsqueeze(0), rc, rs).squeeze(0)
        return q, k, v.view(T, num_kv_heads, head_dim)

    # ------------------------------------------------------------ weights

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ar_pairs, remapped = partition_checkpoint_weights(weights)
        missing, unexpected = self.load_state_dict(dict(remapped), strict=False)
        missing = [k for k in missing if not k.startswith(("model.", "lm_head"))]
        if missing or unexpected:
            raise RuntimeError(f"YuE2 NAR/projection weight mismatch: missing={missing}, unexpected={unexpected}")

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(iter(ar_pairs))
        # The NAR/projection modules were populated by hand above;
        # DefaultModelLoader.track_weights_loading diffs every named
        # parameter against the returned set, so the side keys must be
        # reported too or the loader flags them as uninitialized.
        return loaded | {name for name, _ in remapped}

    # ------------------------------------------------------------ hooks

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def prepare_runner_inputs(
        self,
        *,
        req_ids: list[str],
        num_computed_tokens: Any,
        num_scheduled_tokens: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Bind rows to requests for the step; prompt tokens are captured in
        ``forward``, where the state is created (see ``_capture_constants``).
        Decode rows feed tokens the model itself sampled, so they never
        contribute to the prompt prefix."""
        computed = [int(v) for v in num_computed_tokens]
        scheduled = [int(v) for v in num_scheduled_tokens]
        self._step_rows = [(str(rid), computed[i], scheduled[i]) for i, rid in enumerate(req_ids)]
        return input_ids, positions

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del intermediate_tensors
        self._capture_constants(kwargs, input_ids)
        hidden = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            positions=positions,
            inputs_embeds=inputs_embeds,
        )
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        self._flush_deferred_cleanup()
        return hidden

    def _capture_constants(self, kwargs: dict[str, Any], input_ids: torch.Tensor | None) -> None:
        """Create per-request state on the request's FIRST scheduled step.

        With KV prefix caching the first step may arrive with ``comp > 0``
        (a shared head was cached by an earlier request), so ``comp == 0``
        cannot be the trigger. The scheduled ``input_ids`` slice then holds
        only the uncached tail, so the driver ships the full prompt ids in
        the request's extra args (KEY_PREFIX_IDS); they are the source of
        truth for the NAR conditioning prefix and for the prompt length the
        sampler uses to tell a completing prefill row from a mid-chunk one.
        """
        extra_args = kwargs.get("sampling_extra_args")
        if extra_args is None or input_ids is None:
            return
        offset = 0
        for row, (req_id, comp, span) in enumerate(self._step_rows):
            if row >= len(extra_args):
                break
            if req_id in self._states:
                offset += span
                continue
            args = extra_args[row] or {}
            prefix = args.get(KEY_PREFIX_IDS)
            if isinstance(prefix, (list, tuple)) and prefix:
                prefix_ids = [int(v) for v in prefix]
                prompt_len = len(prefix_ids)
            elif comp == 0 and span >= 1:
                # Legacy path (no ids in extra args): only correct when the
                # whole prompt is scheduled in one chunk with no cache hit.
                prefix_ids = input_ids[offset : offset + span].tolist()
                prompt_len = len(prefix_ids)
            else:
                logger.warning(
                    "YuE2 request %s arrived with %d cached tokens but no "
                    "yue2_prefix_ids in its extra args; cannot rebuild its "
                    "full prompt (prefix-cache hit on the first request?).",
                    req_id,
                    comp,
                )
                offset += span
                continue
            phase = str(args.get(KEY_PHASE, "semantic"))
            preset = ABC_SAMPLING if phase == "abc" else SEMANTIC_SAMPLING
            seed = args.get(KEY_SEED, DEFAULT_SEED)
            constants = _RowConstants(
                request_id=req_id,
                phase=phase,
                temperature=float(args.get(KEY_TEMPERATURE, preset["temperature"])),
                top_p=float(args.get(KEY_TOP_P, preset["top_p"])),
                top_k=int(args.get(KEY_TOP_K, preset["top_k"])),
                repetition_penalty=float(args.get(KEY_REPETITION_PENALTY, preset["repetition_penalty"])),
                penalty_window=int(args.get(KEY_PENALTY_WINDOW, preset["penalty_window"])),
                min_tokens=int(args.get(KEY_MIN_TOKENS, preset["min_tokens"])),
                max_audio_frames=int(args.get(KEY_MAX_AUDIO_FRAMES, preset["max_tokens"])),
                seed=int(seed),
                skip_synthesis=bool(args.get(KEY_SKIP_SYNTHESIS, phase == "abc")),
            )
            device = input_ids.device
            generator = torch.Generator(device=device if device.type != "mps" else "cpu")
            generator.manual_seed(constants.seed)
            self._row_constants[req_id] = constants
            self._states[req_id] = _RequestState(
                request_id=req_id,
                constants=constants,
                prompt_tokens=prompt_len,
                prompt_len=prompt_len,
                prefix_ids=prefix_ids,
                generator=generator,
            )
            offset += span

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    # ------------------------------------------------------------ sampler

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> SamplerOutput | None:
        """Draw one token per decoding row with the request's own arithmetic."""
        del sampling_metadata
        rows = int(logits.shape[0])
        token_ids = torch.zeros((rows, 1), dtype=torch.long, device=logits.device)
        if not self._step_rows:
            return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

        for row, (req_id, comp, span) in enumerate(self._step_rows):
            if row >= rows:
                break
            state = self._states.get(req_id)
            if state is None or state.finished:
                continue
            if comp + span < state.prompt_len:
                # Mid-chunk prefill rows: their sampled token is discarded.
                # (A completing prefill row reaches the full prompt length
                # even when a prefix-cache hit left comp > 0.)
                continue
            constants = state.constants
            if state.request_id not in self._decode_t0:
                self._decode_t0[state.request_id] = time.perf_counter()
            scores = distribution(
                logits[row],
                temperature=constants.temperature,
                top_p=constants.top_p,
                top_k=constants.top_k,
                repetition_penalty=constants.repetition_penalty,
                penalty_window=constants.penalty_window,
                history=state.history,
                step=len(state.history),
                min_tokens=constants.min_tokens,
                phase=constants.phase,
            )
            token = sample_row(scores, state.generator, greedy=constants.temperature == 0)
            end = ABC_END if constants.phase == "abc" else MUSIC_END
            if token == end:
                state.finished = True
                state.truncated = False
                self._finish_request(state, hit_end=True)
                token_ids[row, 0] = end
            else:
                state.history.append(token)
                budget = constants.max_audio_frames
                if constants.phase == "semantic" and len(state.history) >= budget:
                    state.finished = True
                    state.truncated = True
                    self._finish_request(state, hit_end=False)
                    token_ids[row, 0] = end
                else:
                    token_ids[row, 0] = token
        return SamplerOutput(sampled_token_ids=token_ids, logprobs_tensors=None)

    # ------------------------------------------------------------ audio

    def _finish_request(self, state: _RequestState, *, hit_end: bool) -> None:
        """Solve the ODE and decode the song; abc-phase requests skip this."""
        constants = state.constants
        if constants.skip_synthesis or not state.history:
            return
        from .constants import CODEC_SIZE

        codec = [t - CODEC_OFFSET for t in state.history]
        if min(codec) < 0 or max(codec) >= CODEC_SIZE:
            raise RuntimeError("semantic history contains non-codec tokens")
        t0 = self._decode_t0.pop(state.request_id, None)
        t_nar0 = time.perf_counter()
        latents = synthesize(self, state.prefix_ids, codec, constants.seed, steps=ODE_STEPS)
        logger.info(
            "YUE2_LATENTS frames=%d absmax=%.4f mean=%.5f std=%.4f",
            latents.shape[0],
            float(latents.abs().max()),
            float(latents.mean()),
            float(latents.std()),
        )
        t_nar = time.perf_counter() - t_nar0
        t_vae0 = time.perf_counter()
        audio = self._decode_latents(latents)
        t_vae = time.perf_counter() - t_vae0
        t_ar = (t_nar0 - t0) if t0 is not None else float("nan")
        logger.info(
            "YUE2_TIMING req=%s phase=%s ar_tokens=%d ar_s=%.2f ar_tps=%.1f nar_s=%.2f vae_s=%.2f",
            state.request_id,
            constants.phase,
            len(state.history),
            t_ar,
            len(state.history) / t_ar if t_ar else float("nan"),
            t_nar,
            t_vae,
        )
        self._ship_audio(state.request_id, audio, state.truncated)
        del hit_end

    def _ship_audio(self, req_id: str, audio: torch.Tensor, truncated: bool) -> None:
        """Append the finished song to the CURRENT step's mm payload in place.

        make_omni_output already ran for this step (it ships the sparse marker
        with empty lists), and the runner assembles per-request payloads only
        after sample(), so mutating the same dict here is picked up by the
        sparse routing on the request's last step in the output batch. Leaving
        the payload for a later step would drop it: the request is gone by
        then (gepard_talker flushes at its last step for the same reason).
        """
        mm = self._last_mm
        if mm is None:
            # No forward built a payload this step; fall back to the queue.
            self._audio_queue.append((req_id, audio, truncated))
            return
        mm["model_outputs"].append(audio)
        mm["sr"].append(torch.tensor(SAMPLE_RATE, dtype=torch.int32))
        meta = mm["meta"]
        meta["req_id"].append(req_id)
        meta["truncated"].append(str(int(truncated)))

    def _vae_model(self, device: torch.device) -> YuE2VAE:
        if self._vae is None:
            vae_path = os.environ.get("YUE2_VAE", DEFAULT_VAE_ID)
            logger.info("Loading YuE2 VAE decoder from %s", vae_path)
            self._vae = YuE2VAE.from_pretrained(vae_path, decoder_only=True, device="cpu")
        if self._vae_device != device:
            self._vae.to(device)
            self._vae_device = device
        return self._vae

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """[frames, 64] FP32 CPU latents -> interleaved stereo [samples, 2]."""
        device = f"cuda:{torch.accelerator.current_device_index()}" if torch.cuda.is_available() else "cpu"
        model = self._vae_model(torch.device(device))
        z = latents.T.unsqueeze(0)  # [1, 64, T]
        with torch.inference_mode():
            audio = model.decode_tiled(
                z,
                core_frames=VAE_CORE_FRAMES,
                halo_frames=VAE_HALO_FRAMES,
                output_device="cpu",
            )
        if not torch.isfinite(audio).all():
            raise RuntimeError("VAE produced non-finite audio")
        logger.info("YUE2_AUDIO_PRECLAMP absmax=%.4f", float(audio.abs().max()))
        return audio[0].float().clamp(-1, 1).T.contiguous().reshape(-1)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        by_req: dict[str, torch.Tensor] = {}
        truncated: dict[str, bool] = {}
        for req_id, audio, is_truncated in self._audio_queue:
            by_req[req_id] = audio
            truncated[req_id] = is_truncated
        self._audio_queue.clear()
        ready = list(by_req)
        sr = torch.tensor(SAMPLE_RATE, dtype=torch.int32)
        # The sparse marker rides EVERY step, even ones that decoded nothing
        # (req_id=[] is the legal zero-audio shape): without it the runner
        # falls back to the dense pooler payload, whose per-step hidden states
        # land under this stage's "audio" key and the driver receives hidden
        # states instead of the decoded waveform (gepard_talker precedent).
        mm: dict[str, Any] = {
            "model_outputs": [by_req[r] for r in ready],
            "sr": [sr for _ in ready],
            "meta": {
                "req_id": ready,
                "sparse_audio": ["1"],
                "truncated": [str(int(truncated[r])) for r in ready],
            },
        }
        # sample() runs after this each step and appends finished songs into
        # this very dict (see _ship_audio); keep the handle for it.
        self._last_mm = mm
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=mm)

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        # Fires before forward; defer the free so the in-flight step can read.
        for rid in finished_req_ids:
            self._deferred_cleanup_ids.add(_request_key(rid))

    def _flush_deferred_cleanup(self) -> None:
        for req_id in self._deferred_cleanup_ids:
            state = self._states.pop(req_id, None)
            if state is not None and not state.finished:
                logger.warning(
                    "YuE2 request %s ended without an end token or its frame "
                    "budget (aborted or preempted); no audio was produced.",
                    req_id,
                )
            self._row_constants.pop(req_id, None)
        self._deferred_cleanup_ids.clear()
