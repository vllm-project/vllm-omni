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
import hashlib
import logging
import math
import time
from collections.abc import Callable, Iterable, Sequence
from types import MethodType
from typing import Any, NamedTuple, Protocol, TypedDict

import torch
import torch.nn as nn
from typing_extensions import Unpack
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, override_forward_context
from vllm.inputs import tokens_input
from vllm.logger import init_logger
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.multimodal.audio import AudioResampler
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.speaker_cache import (
    get_speaker_cache,
    iter_custom_voice_profiles,
    load_validated_profile_tensors,
)
from vllm_omni.worker.runner_assisted_metadata import RunnerAssistedFullAttentionMetadataRequest

from .minicpm4_paged import MiniCPM4PagedForVoxCPM2, MiniCPM4PagedResidualLM
from .runtime_config import _VoxCPM2RuntimeConfig
from .voxcpm2_import_utils import import_voxcpm2_core

logger = init_logger(__name__)

_ENABLE_NVTX_PROFILE = False

# Lower bound for the _active_states leak-warn threshold.  The effective
# threshold is max(_ACTIVE_STATE_LEAK_WARN_MIN, 4 * max_batch_size) so small
# deployments still get a usable floor instead of a tiny noisy one.
_ACTIVE_STATE_LEAK_WARN_MIN = 512


class _VoxCPM2PromptConfigLike(Protocol):
    audio_vae_config: dict[str, Any]
    patch_size: int


class _AttentionMetadataLike(Protocol):
    scheduler_metadata: Any


class _ForwardContextLike(Protocol):
    attn_metadata: dict[str, _AttentionMetadataLike] | Any


class VoxCPM2PreprocessInput(TypedDict, total=False):
    additional_information: dict[str, Any]
    request_id: str
    text_token_ids: list[list[int]]
    reference_audio: object
    ref_audio: object
    prompt_audio: object
    prompt_text: str | list[str] | None
    voice_profile: dict[str, Any] | list[dict[str, Any]] | None
    voice_name: str | list[str] | None
    voice_created_at: int | str | None


class VoxCPM2PostprocessInput(TypedDict, total=False):
    request_id: str


class _PrefillInputs(NamedTuple):
    text_token: torch.Tensor
    audio_feat: torch.Tensor
    text_mask: torch.Tensor
    audio_mask: torch.Tensor


class _PrefillResidualMeta(NamedTuple):
    lm_hidden: torch.Tensor
    prefix_feat_cond: torch.Tensor


class _DecodeResidualMeta(NamedTuple):
    new_lm_hidden: torch.Tensor


def is_cjk_char(c: str) -> bool:
    """Check if a character is a CJK ideograph."""
    cp = ord(c)
    return (
        0x4E00 <= cp <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # Extension A
        or 0xF900 <= cp <= 0xFAFF  # Compatibility Ideographs
        or 0x20000 <= cp <= 0x2A6DF  # Extension B
        or 0x2A700 <= cp <= 0x2B73F  # Extension C
        or 0x2B740 <= cp <= 0x2B81F  # Extension D
        or 0x2F800 <= cp <= 0x2FA1F  # Compatibility Supplement
    )


def build_cjk_split_map(tokenizer: Any) -> dict[int, list[int]]:
    """Build {multichar_cjk_token_id: [single_char_ids]} from tokenizer vocab."""
    vocab = tokenizer.get_vocab()
    split_map: dict[int, list[int]] = {}
    for token, token_id in vocab.items():
        clean = token.replace("\u2581", "")
        if len(clean) >= 2 and all(is_cjk_char(c) for c in clean):
            char_ids = tokenizer.convert_tokens_to_ids(list(clean))
            if all(cid != tokenizer.unk_token_id for cid in char_ids):
                split_map[token_id] = char_ids
    return split_map


def split_multichar_chinese(token_ids: list[int], split_map: dict[int, list[int]]) -> list[int]:
    """Replace multichar Chinese token IDs with single-char IDs (idempotent)."""
    result: list[int] = []
    for tid in token_ids:
        expansion = split_map.get(tid)
        if expansion is not None:
            result.extend(expansion)
        else:
            result.append(tid)
    return result


def build_voxcpm2_prompt(
    hf_config: _VoxCPM2PromptConfigLike,
    tokenizer: Any,
    split_map: dict[int, list[int]],
    text: str,
    ref_audio: Sequence[float] | torch.Tensor | None = None,
    ref_sr: int | None = None,
    ref_text: str | None = None,
    voice_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a VoxCPM2 prefill prompt whose ``prompt_token_ids`` length matches
    the talker-side prefill length.

    Used by both online serving (``serving_speech._build_voxcpm2_prompt``) and
    the offline example, so the talker-side length assertion never fires.
    """
    ids = split_multichar_chinese(tokenizer.encode(text, add_special_tokens=True), split_map)
    bos = tokenizer.bos_token_id
    if ids and ids[0] == bos:
        ids = ids[1:]
    prefill_len = len(ids) + 1  # + audio_start
    additional: dict[str, Any] = {"text_token_ids": [ids]}
    if voice_profile is not None and ref_audio is None:
        mode = str(voice_profile.get("mode") or "reference").lower()
        ref_audio_feat_len = int(voice_profile.get("ref_audio_feat_len") or 0)
        audio_feat_len = int(voice_profile.get("audio_feat_len") or 0)
        prompt_text = voice_profile.get("prompt_text") or voice_profile.get("ref_text")
        additional["voice_profile"] = dict(voice_profile)

        if mode in ("reference", "ref_continuation"):
            prefill_len += ref_audio_feat_len + 2  # ref_start / ref_end
        if mode in ("continuation", "ref_continuation"):
            if isinstance(prompt_text, str) and prompt_text:
                additional["prompt_text"] = [prompt_text]
                prompt_ids = split_multichar_chinese(tokenizer.encode(prompt_text, add_special_tokens=True), split_map)
                if prompt_ids and prompt_ids[0] == bos:
                    prompt_ids = prompt_ids[1:]
                prefill_len += len(prompt_ids)
            prefill_len += audio_feat_len
    elif ref_audio is not None:
        if ref_sr is None:
            raise ValueError("VoxCPM2 ref_sr is required when ref_audio is provided.")
        vae = hf_config.audio_vae_config
        patch_samples = hf_config.patch_size * math.prod(vae["encoder_rates"])
        ref_len = math.ceil(math.ceil(len(ref_audio) * vae["sample_rate"] / ref_sr) / patch_samples)
        if ref_text is not None:
            additional["prompt_audio"] = [[ref_audio, ref_sr]]
            additional["prompt_text"] = [ref_text]
            ref_ids = split_multichar_chinese(tokenizer.encode(ref_text, add_special_tokens=True), split_map)
            if ref_ids and ref_ids[0] == bos:
                ref_ids = ref_ids[1:]
            prefill_len += ref_len + len(ref_ids)
        else:
            additional["reference_audio"] = [[ref_audio, ref_sr]]
            prefill_len += ref_len + 2  # ref_start / ref_end
    prompt = tokens_input(prompt_token_ids=[1] * prefill_len)
    prompt["additional_information"] = additional
    return prompt


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
        resampler = AudioResampler(target_sr=encode_sr)
        audio_np = resampler.resample(audio_np, orig_sr=sr)
        audio = torch.from_numpy(audio_np).unsqueeze(0)

    patch_len = tts.patch_size * tts.chunk_size
    if audio.size(1) % patch_len != 0:
        padding_size = patch_len - audio.size(1) % patch_len
        pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
        audio = torch.nn.functional.pad(audio, pad)

    vae_device = next(tts.audio_vae.parameters()).device
    feat = tts.audio_vae.encode(audio.to(vae_device), encode_sr).cpu()
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
    cfm_output_gpu: torch.Tensor | None = None
    cfm_noise_step: int = 0
    precomputed_stop_logits: torch.Tensor | None = None
    pending_audio_chunks_gpu: list[torch.Tensor] = dataclasses.field(default_factory=list)
    pending_audio_copies: list[_PendingAudioCopy] = dataclasses.field(default_factory=list)
    pending_vae_latents_gpu: list[torch.Tensor] = dataclasses.field(default_factory=list)
    # Rolling tail of previously-decoded latents used as VAE receptive-field context.
    # Shape (n_pad_frames, feat_dim) on GPU. None before first decode.
    decode_pad: torch.Tensor | None = None
    decode_step_count: int = 0
    request_start_time: float = 0.0
    prefill_completed: bool = False
    prompt_cache: dict | None = None
    prefill_masks: tuple | None = None
    is_stopping: bool = False
    precomputed_is_stopping: bool | None = None


@dataclasses.dataclass
class _CapturedGraph:
    graph: torch.cuda.CUDAGraph
    input_embeds: torch.Tensor
    positions: torch.Tensor
    output: torch.Tensor


@dataclasses.dataclass
class _CapturedVAEGraph:
    graph: torch.cuda.CUDAGraph
    input_feat: torch.Tensor
    output: torch.Tensor


@dataclasses.dataclass
class _CapturedCFMGraph:
    graph: torch.cuda.CUDAGraph
    mu: torch.Tensor
    cond: torch.Tensor
    noise: torch.Tensor
    output: torch.Tensor
    buffers: _CFMBufferManager


@dataclasses.dataclass
class _CapturedUnifiedDecodeGraph:
    graph: torch.cuda.CUDAGraph
    batch_size: int
    # Static input buffers (copied before replay)
    input_embeds: torch.Tensor
    positions: torch.Tensor
    prev_feat_embed: torch.Tensor
    prefix_feat_cond: torch.Tensor
    # Static output buffers (read after replay)
    next_feat_embed: torch.Tensor
    cfm_output: torch.Tensor
    lm_hidden: torch.Tensor
    # Internal buffers for CFM solver
    cfm_buffers: _CFMBufferManager
    cfm_noise: torch.Tensor


@dataclasses.dataclass
class _UnifiedDecodeGraphStats:
    captures: int = 0
    replays: int = 0
    skips: dict[str, int] = dataclasses.field(default_factory=dict)
    real_batch_sizes: dict[int, int] = dataclasses.field(default_factory=dict)
    graph_bucket_sizes: dict[int, int] = dataclasses.field(default_factory=dict)
    logged_replays: int = 0
    logged_skips: int = 0

    def record_skip(self, reason: str) -> None:
        self.skips[reason] = self.skips.get(reason, 0) + 1

    def record_replay_batch(self, *, real_batch_size: int, graph_bucket_size: int) -> None:
        self.real_batch_sizes[real_batch_size] = self.real_batch_sizes.get(real_batch_size, 0) + 1
        self.graph_bucket_sizes[graph_bucket_size] = self.graph_bucket_sizes.get(graph_bucket_size, 0) + 1

    @property
    def total_skips(self) -> int:
        return sum(self.skips.values())


@dataclasses.dataclass
class _PendingAudioCopy:
    host: torch.Tensor
    event: torch.cuda.Event | None = None
    source: torch.Tensor | None = None
    async_copy: bool = False


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
        torch.accelerator.synchronize()
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


class _NvtxRange:
    __slots__ = ("_enabled", "_entered")

    def __init__(self, name: str):
        self._enabled = _ENABLE_NVTX_PROFILE and torch.cuda.is_available()
        self._entered = False
        if self._enabled:
            torch.cuda.nvtx.range_push(name)
            self._entered = True

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._entered:
            torch.cuda.nvtx.range_pop()


def _install_locdit_fused_qkv(
    estimator: nn.Module,
    *,
    enable_fast_rope: bool,
    skip_qkv_contig: bool,
) -> int:
    """Patch native VoxCPM LocDiT attention modules to use one QKV projection.

    Native LocDiT uses separate q/k/v Linear layers before PyTorch SDPA.  This
    keeps the attention backend unchanged and only replaces the projection
    front-end with a pre-concatenated weight, mirroring the AR-side wrapper.
    """
    decoder = getattr(estimator, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        return 0
    patched = 0

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary_pos_emb_fast_dtype(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)

    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "_voxcpm2_fused_qkv", False):
            continue
        apply_rotary = type(attn).forward.__globals__.get("apply_rotary_pos_emb")
        if apply_rotary is None:
            continue
        weight = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0).detach()
        attn.register_buffer("_voxcpm2_fused_qkv_weight", weight, persistent=False)

        def _forward(
            self,
            hidden_states: torch.Tensor,
            position_emb: tuple[torch.Tensor, torch.Tensor] | None,
            is_causal: bool,
            _apply_rotary=apply_rotary,
        ):
            bsz, q_len, _ = hidden_states.size()
            qkv = nn.functional.linear(hidden_states, self._voxcpm2_fused_qkv_weight)
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim
            query_states, key_states, value_states = qkv.split([q_size, kv_size, kv_size], dim=-1)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            if position_emb is not None:
                cos, sin = position_emb
                if enable_fast_rope and query_states.dtype != torch.float32:
                    query_states, key_states = _apply_rotary_pos_emb_fast_dtype(query_states, key_states, cos, sin)
                else:
                    query_states, key_states = _apply_rotary(query_states, key_states, cos, sin)
            if not skip_qkv_contig:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                is_causal=is_causal,
                enable_gqa=True,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
            attn_output = self.o_proj(attn_output)
            return attn_output, (key_states, value_states)

        attn.forward = MethodType(_forward, attn)
        attn._voxcpm2_fused_qkv = True
        patched += 1
    return patched


def _install_locdit_fused_mlp(estimator: nn.Module) -> int:
    """Patch LocDiT MLP gate/up projections into one Linear call."""
    decoder = getattr(estimator, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        return 0
    patched = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or getattr(mlp, "_voxcpm2_fused_mlp", False):
            continue
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            continue
        gate_bias = getattr(gate_proj, "bias", None)
        up_bias = getattr(up_proj, "bias", None)
        if (gate_bias is None) != (up_bias is None):
            continue
        weight = torch.cat([gate_proj.weight, up_proj.weight], dim=0).detach()
        mlp.register_buffer("_voxcpm2_fused_gate_up_weight", weight, persistent=False)
        if gate_bias is not None and up_bias is not None:
            bias = torch.cat([gate_bias, up_bias], dim=0).detach()
            mlp.register_buffer("_voxcpm2_fused_gate_up_bias", bias, persistent=False)
        else:
            mlp._voxcpm2_fused_gate_up_bias = None

        def _forward(self, x: torch.Tensor):
            gate_up = nn.functional.linear(
                x,
                self._voxcpm2_fused_gate_up_weight,
                self._voxcpm2_fused_gate_up_bias,
            )
            gate, up = gate_up.chunk(2, dim=-1)
            return self.down_proj(nn.functional.silu(gate) * up)

        mlp.forward = MethodType(_forward, mlp)
        mlp._voxcpm2_fused_mlp = True
        patched += 1
    return patched


def _install_locdit_zero_dt_cache(estimator: nn.Module) -> bool:
    """Cache the constant delta-time embedding used when CFM mean_mode is off."""
    if getattr(estimator, "_voxcpm2_zero_dt_cache", False):
        return False
    time_embeddings = getattr(estimator, "time_embeddings", None)
    delta_time_mlp = getattr(estimator, "delta_time_mlp", None)
    out_proj = getattr(estimator, "out_proj", None)
    decoder = getattr(estimator, "decoder", None)
    if time_embeddings is None or delta_time_mlp is None or out_proj is None or decoder is None:
        return False

    device = out_proj.weight.device
    dtype = out_proj.weight.dtype
    with torch.no_grad():
        zero_dt = torch.zeros(1, device=device, dtype=dtype)
        zero_dt_emb = delta_time_mlp(time_embeddings(zero_dt).to(dtype)).detach()
    estimator.register_buffer("_voxcpm2_zero_dt_emb", zero_dt_emb, persistent=False)

    def _forward(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        dt: torch.Tensor,
    ):
        x = self.in_proj(x.transpose(1, 2).contiguous())
        cond = self.cond_proj(cond.transpose(1, 2).contiguous())
        prefix = cond.size(1)

        t = self.time_embeddings(t).to(x.dtype)
        t = self.time_mlp(t)
        t = t + self._voxcpm2_zero_dt_emb.expand(t.shape[0], -1)

        mu = mu.view(x.size(0), -1, x.size(-1))
        x = torch.cat([mu, t.unsqueeze(1), cond, x], dim=1)

        hidden, _ = self.decoder(x, is_causal=False)
        hidden = hidden[:, prefix + mu.size(1) + 1 :, :]
        hidden = self.out_proj(hidden)

        return hidden.transpose(1, 2).contiguous()

    estimator.forward = MethodType(_forward, estimator)
    estimator._voxcpm2_zero_dt_cache = True
    return True


def _install_locdit_layer_nvtx(estimator: nn.Module) -> int:
    decoder = getattr(estimator, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        return 0
    patched = 0
    for idx, layer in enumerate(layers):
        if getattr(layer, "_voxcpm2_layer_nvtx", False):
            continue

        def _forward(
            self,
            hidden_states: torch.Tensor,
            position_emb: tuple[torch.Tensor, torch.Tensor] | None,
            is_causal: bool,
            _idx: int = idx,
        ):
            residual = hidden_states
            with _NvtxRange(f"voxcpm2.cfm.estimator.layer{_idx}.norm1"):
                hidden_states = self.input_layernorm(hidden_states)
            with _NvtxRange(f"voxcpm2.cfm.estimator.layer{_idx}.attn"):
                hidden_states, present_key_value = self.self_attn(
                    hidden_states=hidden_states,
                    position_emb=position_emb,
                    is_causal=is_causal,
                )
            with _NvtxRange(f"voxcpm2.cfm.estimator.layer{_idx}.resid1"):
                if self.use_mup:
                    hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
                else:
                    hidden_states = residual + hidden_states

            residual = hidden_states
            with _NvtxRange(f"voxcpm2.cfm.estimator.layer{_idx}.norm2"):
                hidden_states = self.post_attention_layernorm(hidden_states)
            with _NvtxRange(f"voxcpm2.cfm.estimator.layer{_idx}.mlp"):
                hidden_states = self.mlp(hidden_states)
            with _NvtxRange(f"voxcpm2.cfm.estimator.layer{_idx}.resid2"):
                if self.use_mup:
                    hidden_states = residual + hidden_states * (self.scale_depth / math.sqrt(self.num_hidden_layers))
                else:
                    hidden_states = residual + hidden_states
            return hidden_states, present_key_value

        layer.forward = MethodType(_forward, layer)
        layer._voxcpm2_layer_nvtx = True
        patched += 1
    return patched


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
        self.x_work = torch.zeros_like(self.noise)
        self._zeroed_mu_negative_batch: int | None = None
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
    b = mu.size(0)
    buffers.noise[:b].normal_()
    return _optimized_solve_euler_with_noise(
        cfm_module,
        mu,
        patch_size,
        cond,
        buffers.noise[:b],
        n_timesteps,
        cfg_value,
        buffers,
        use_cfg_zero_star=use_cfg_zero_star,
        cfg_cutoff_ratio=cfg_cutoff_ratio,
        perf=perf,
    )


def _optimized_solve_euler_with_noise(
    cfm_module: nn.Module,
    mu: torch.Tensor,
    patch_size: int,
    cond: torch.Tensor,
    noise: torch.Tensor,
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

    with _NvtxRange("voxcpm2.cfm.init_copy"):
        buffers.x_work[:b].copy_(noise[:b])
        x = buffers.x_work[:b]

    t_span = buffers.get_t_span(n_timesteps)
    t, dt = t_span[0], t_span[0] - t_span[1]
    zero_init_steps = max(1, int(len(t_span) * 0.04))
    cfg_cutoff_step = max(zero_init_steps + 1, int(len(t_span) * cfg_cutoff_ratio))
    if use_cfg_zero_star and zero_init_steps > 0:
        t = t_span[zero_init_steps]
        if zero_init_steps < len(t_span) - 1:
            dt = t - t_span[zero_init_steps + 1]
        start_step = zero_init_steps + 1
    else:
        start_step = 1

    with _NvtxRange("voxcpm2.cfm.static_inputs"):
        buffers.mu_in[:b].copy_(mu)
        buffers.cond_in[:b].copy_(cond[:b])
        if cfg_cutoff_step >= start_step:
            if buffers._zeroed_mu_negative_batch != b:
                buffers.mu_in[b : 2 * b].zero_()
                buffers._zeroed_mu_negative_batch = b
            buffers.cond_in[b : 2 * b].copy_(cond[:b])

    for step in range(start_step, len(t_span)):
        if step <= cfg_cutoff_step:
            with _NvtxRange("voxcpm2.cfm.step_cfg_inputs"):
                buffers.x_in[:b].copy_(x)
                buffers.x_in[b : 2 * b].copy_(x)
                buffers.t_in[: 2 * b].copy_(t)
                if mean_mode:
                    buffers.dt_in[: 2 * b].copy_(dt)

            if perf:
                perf.start("  cfm.estimator_cfg")
            with _NvtxRange("voxcpm2.cfm.estimator_cfg"):
                raw_out = estimator(
                    buffers.x_in[: 2 * b],
                    buffers.mu_in[: 2 * b],
                    buffers.t_in[: 2 * b],
                    buffers.cond_in[: 2 * b],
                    buffers.dt_in[: 2 * b],
                )
            if perf:
                perf.stop("  cfm.estimator_cfg")

            with _NvtxRange("voxcpm2.cfm.cfg_post"):
                dphi_dt, cfg_dphi_dt = raw_out[:b], raw_out[b : 2 * b]
                if use_cfg_zero_star:
                    pos = dphi_dt.reshape(b, -1)
                    neg = cfg_dphi_dt.reshape(b, -1)
                    st = torch.sum(pos * neg, 1, keepdim=True) / (torch.sum(neg**2, 1, keepdim=True) + 1e-8)
                    st = st.view(b, *([1] * (len(dphi_dt.shape) - 1)))
                    dphi_dt = cfg_dphi_dt * st + cfg_value * (dphi_dt - cfg_dphi_dt * st)
                else:
                    st = 1.0
                    dphi_dt = cfg_dphi_dt * st + cfg_value * (dphi_dt - cfg_dphi_dt * st)
        else:
            with _NvtxRange("voxcpm2.cfm.step_nocfg_inputs"):
                buffers.x_in[:b].copy_(x)
                buffers.t_in[:b].copy_(t)
                if mean_mode:
                    buffers.dt_in[:b].copy_(dt)
            if perf:
                perf.start("  cfm.estimator_nocfg")
            with _NvtxRange("voxcpm2.cfm.estimator_nocfg"):
                dphi_dt = estimator(
                    buffers.x_in[:b],
                    buffers.mu_in[:b],
                    buffers.t_in[:b],
                    buffers.cond_in[:b],
                    buffers.dt_in[:b],
                )
            if perf:
                perf.stop("  cfm.estimator_nocfg")

        with _NvtxRange("voxcpm2.cfm.euler_update"):
            x.sub_(dt * dphi_dt)
            t = t - dt
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]
    return x.clone()


# ===================================================================
#  Main talker model
# ===================================================================


class VoxCPM2TalkerForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self._runtime_config = _VoxCPM2RuntimeConfig.from_vllm_config(vllm_config)
        global _ENABLE_NVTX_PROFILE
        _ENABLE_NVTX_PROFILE = self._runtime_config.enable_nvtx_profile

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

        # Eager-init tts_model so it registers in self.state_dict() before vLLM's
        # post-__init__ profiling claims the remaining GPU memory for KV cache.
        # Required for load_format=dummy: DummyModelLoader only randomizes
        # already-registered nn.Parameters.
        # NOTE: from_pretrained() is unconditional, so load_format=dummy still pays
        # the checkpoint download/read cost at construction time; DummyModelLoader
        # will then randomize the just-loaded _tts params — this is intended.
        model_path = vllm_config.model_config.model
        self._device = current_omni_platform.get_torch_device()
        VoxCPM = import_voxcpm2_core()
        native = VoxCPM.from_pretrained(model_path, load_denoiser=False, optimize=False)
        self._tts: nn.Module = native.tts_model.to(self._device)
        self._side_dtype = self._tts.fusion_concat_proj.weight.dtype
        self._patch_size = self._tts.patch_size
        self._feat_dim = self._tts.feat_dim
        self._sample_rate = getattr(self.config, "sample_rate", 48000)

        # base_lm/residual_lm in native tts_model duplicate self.model and
        # self.residual_model: copy residual weights over, drop both submodules.
        self.residual_model.load_weights_from_native(self._tts.residual_lm)
        del self._tts.base_lm
        self._tts.base_lm = None
        del self._tts.residual_lm
        self._tts.residual_lm = None
        torch.accelerator.empty_cache()

        self._inference_timesteps = 10
        self._cfg_value = 2.0
        self._cfg_cutoff_ratio = self._runtime_config.cfg_cutoff_ratio
        # Number of trailing latent frames to keep as VAE receptive-field context
        # for sliding-window streaming decode. 12 matches the nanovllm reference
        # implementation and covers the longest VAE decoder receptive field.
        self._n_decode_pad_frames = 12
        use_cuda_graph = current_omni_platform.is_cuda()
        self._enable_torch_compile = current_omni_platform.supports_torch_inductor()
        self._compile_vae = self._enable_torch_compile
        self._max_decode_steps = 2000
        self._max_batch_size = getattr(vllm_config.scheduler_config, "max_num_seqs", 4)

        # Speaker cache for ref_audio_feat across requests
        self._speaker_cache = get_speaker_cache()
        self._load_custom_voice_profiles()

        self._enable_profiling = self._runtime_config.enable_profiling
        self._perf = _PerfTimer(enabled=self._enable_profiling)
        self._cfm_buffers: _CFMBufferManager | None = None
        self._enable_cuda_graph = use_cuda_graph
        self._scaffold_graphs: dict[int, _CapturedGraph] = {}
        self._residual_graphs: dict[int, _CapturedGraph] = {}
        self._decode_graph_capture_policy = self._runtime_config.decode_graph_capture_policy
        self._vae_graphs: dict[tuple[int, int], _CapturedVAEGraph] = {}
        self._enable_vae_cuda_graph = use_cuda_graph and self._runtime_config.enable_vae_cuda_graph
        self._cfm_graphs: dict[int, _CapturedCFMGraph] = {}
        self._enable_cfm_cuda_graph = use_cuda_graph and self._runtime_config.enable_cfm_cuda_graph
        self._enable_cfm_prealloc_output = use_cuda_graph and self._runtime_config.enable_cfm_prealloc_output
        self._enable_batched_cfm = self._runtime_config.enable_batched_cfm
        self._deterministic_cfm_noise = self._runtime_config.deterministic_cfm_noise
        self._deterministic_cfm_seed = self._runtime_config.deterministic_cfm_seed
        self._vae_decode_sr_cond: torch.Tensor | None = None
        self._audio_emit_every = self._runtime_config.audio_emit_every
        self._vae_decode_every = self._runtime_config.vae_decode_every
        self._enable_delayed_audio_copy = self._runtime_config.enable_delayed_audio_copy
        self._delayed_audio_copy_use_events = use_cuda_graph and self._runtime_config.delayed_audio_copy_use_events
        self._coalesce_audio_d2h = self._runtime_config.coalesce_audio_d2h
        self._enable_batched_vae_decode = self._runtime_config.enable_batched_vae_decode
        self._enable_batched_fsq_fusion = self._runtime_config.enable_batched_fsq_fusion
        self._batched_fsq_fusion_max_batch = self._runtime_config.batched_fsq_fusion_max_batch
        self._audio_copy_stream: torch.cuda.Stream | None = None
        self._max_cached_graphs = self._max_batch_size
        self._cuda_graph_warmup_steps = 0
        self._cuda_graph_warmup_threshold = 3
        self._unified_graphs: dict[int, _CapturedUnifiedDecodeGraph] = {}
        self._enable_unified_decode_graph = self._runtime_config.unified_decode_graph_available(
            use_cuda_graph=use_cuda_graph
        )
        self._unified_decode_graph_max_batch_size = self._runtime_config.unified_decode_graph_max_batch_size
        self._unified_graph_bucket_sizes: frozenset[int] = self._build_unified_graph_bucket_sizes(
            min(self._max_batch_size, self._unified_decode_graph_max_batch_size)
        )
        self._unified_graph_stats = _UnifiedDecodeGraphStats()
        self._runner_assisted_unified_decode_graph_active = False
        self._runner_assisted_unified_decode_graph_batch_size = 0

        self._multichar_zh_split: dict[int, list[int]] | None = None

        self._active_states: dict[str, _RequestState] = {}
        self._current_request_id: str | None = None
        self._pending_requests: list[tuple[str, bool, torch.Tensor | None, int]] = []
        self._results_queue: list[tuple[str, torch.Tensor | None]] = []
        self._audio_queue: list[tuple[str, Any]] = []
        self._last_audio_output_req_ids: list[str] = []
        self._deferred_cleanup_ids: set[str] = set()
        self._active_state_warn_threshold = max(_ACTIVE_STATE_LEAK_WARN_MIN, 4 * self._max_batch_size)
        # one-shot by design: fires at most once per process to avoid log spam.
        self._active_state_warned = False

    # -------------------- custom voice profiles --------------------

    @staticmethod
    def _clone_prompt_cache(cache: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in cache.items():
            cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
        return cloned

    def _load_custom_voice_profiles(self) -> None:
        """Preload offline VoxCPM2 prompt caches into the shared speaker cache."""
        custom_voice_dir = getattr(self.config, "custom_voice_dir", None)
        if not custom_voice_dir:
            return

        loaded = 0
        for profile in iter_custom_voice_profiles(custom_voice_dir, expected_model_type="voxcpm2"):
            tensors = load_validated_profile_tensors(profile, expected_model_type="voxcpm2")
            if tensors is None:
                continue

            ref_audio_feat = tensors.get("ref_audio_feat")
            audio_feat = tensors.get("audio_feat")
            mode = str(profile.get("mode") or "").lower()

            prompt_cache: dict[str, Any] = {"mode": mode}
            if ref_audio_feat is not None:
                prompt_cache["ref_audio_feat"] = ref_audio_feat.contiguous().cpu()
            if audio_feat is not None:
                prompt_cache["audio_feat"] = audio_feat.contiguous().cpu()
            prompt_text = profile.get("prompt_text") or profile.get("ref_text")
            if isinstance(prompt_text, str) and prompt_text:
                prompt_cache["prompt_text"] = prompt_text

            key = self._speaker_cache.make_cache_key(
                profile["voice_name_lower"],
                model_type="voxcpm2",
                created_at=0,
            )
            self._speaker_cache.put(key, prompt_cache)
            loaded += 1

        if loaded:
            logger.info("Loaded %d precomputed VoxCPM2 custom voice profile(s) from %s", loaded, custom_voice_dir)

    @property
    def tts(self) -> nn.Module:
        return self._tts

    # -------------------- request state management --------------------

    def _get_or_create_state(self, request_id: str) -> _RequestState:
        state = self._active_states.get(request_id)
        if state is None:
            state = _RequestState(request_id=request_id)
            self._active_states[request_id] = state
            if len(self._active_states) > self._active_state_warn_threshold and not self._active_state_warned:
                logger.warning(
                    "VoxCPM2: _active_states size=%d exceeds threshold %d "
                    "(max_batch_size=%d); possible cleanup path leak",
                    len(self._active_states),
                    self._active_state_warn_threshold,
                    self._max_batch_size,
                )
                self._active_state_warned = True
        return state

    def _switch_to_request(self, request_id: str) -> _RequestState:
        if request_id != self._current_request_id:
            self._current_request_id = request_id
        return self._get_or_create_state(request_id)

    def _cleanup_request(self, request_id: str) -> None:
        state = self._active_states.pop(request_id, None)
        if state is not None:
            state.pending_audio_chunks_gpu.clear()
            state.pending_audio_copies.clear()
            state.pending_vae_latents_gpu.clear()
            state.cfm_output_gpu = None
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
        ref_audio: object = None,
        prompt_audio: object = None,
        prompt_text: str | None = None,
    ) -> dict[str, Any] | None:
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

    @staticmethod
    def _voxcpm2_unwrap_torch_compile(module: nn.Module | Callable) -> nn.Module | Callable:
        """Return the eager module behind torch.compile wrappers."""
        seen: set[int] = set()
        while hasattr(module, "_orig_mod") and id(module) not in seen:
            seen.add(id(module))
            module = getattr(module, "_orig_mod")
        return module

    @staticmethod
    def _voxcpm2_compile_without_inductor_cudagraphs(
        module: nn.Module | Callable,
        *,
        mode: str | None,
        fullgraph: bool,
    ) -> Callable:
        """Compile VoxCPM2 LocDiT for capture inside VoxCPM2-owned CUDA graphs."""
        kwargs: dict[str, Any] = {
            "fullgraph": fullgraph,
            "options": {
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
            },
        }
        if mode is not None:
            kwargs["mode"] = mode
        return torch.compile(module, **kwargs)

    @staticmethod
    def _raise_if_cuda_runtime_error(error: BaseException) -> None:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            raise error
        if isinstance(error, RuntimeError):
            message = str(error).lower()
            cuda_markers = (
                "cuda error",
                "device-side assert",
                "illegal memory access",
                "out of memory",
                "cublas",
                "cudnn",
                "cufft",
            )
            if any(marker in message for marker in cuda_markers):
                raise error

    def _voxcpm2_compile_unified_capture_estimator(self, module: nn.Module | Callable) -> Callable:
        eager_module = self._voxcpm2_unwrap_torch_compile(module)
        if getattr(self, "_estimator_for_unified_capture_source", None) is eager_module and hasattr(
            self, "_estimator_for_unified_capture"
        ):
            return self._estimator_for_unified_capture
        compiled = self._voxcpm2_compile_without_inductor_cudagraphs(
            eager_module,
            mode=None,
            fullgraph=False,
        )
        compiled._compiled = True
        self._estimator_for_unified_capture_source = eager_module
        self._estimator_for_unified_capture = compiled
        return compiled

    def _voxcpm2_compile_unified_capture_feat_encoder(self, module: nn.Module | Callable) -> Callable:
        eager_module = self._voxcpm2_unwrap_torch_compile(module)
        if getattr(self, "_feat_encoder_for_unified_capture_source", None) is eager_module and hasattr(
            self, "_feat_encoder_for_unified_capture"
        ):
            return self._feat_encoder_for_unified_capture
        compiled = self._voxcpm2_compile_without_inductor_cudagraphs(
            eager_module,
            mode=None,
            fullgraph=False,
        )
        compiled._compiled = True
        self._feat_encoder_for_unified_capture_source = eager_module
        self._feat_encoder_for_unified_capture = compiled
        return compiled

    def _setup_torch_compile(self) -> None:
        if not self._enable_torch_compile:
            return
        tts = self.tts
        estimator = tts.feat_decoder.estimator
        if hasattr(estimator, "_compiled"):
            return

        targets: list[str] = []
        cfg = self._runtime_config

        if cfg.enable_loc_dit_fused_qkv:
            patched = _install_locdit_fused_qkv(
                estimator,
                enable_fast_rope=cfg.enable_loc_dit_fast_rope,
                skip_qkv_contig=cfg.enable_loc_dit_skip_qkv_contig,
            )
            if patched:
                targets.append(f"LocDiT fused-qkv attention ({patched} layers)")
        if cfg.enable_loc_dit_fused_mlp:
            patched = _install_locdit_fused_mlp(estimator)
            if patched:
                targets.append(f"LocDiT fused gate-up MLP ({patched} layers)")
        if cfg.enable_loc_dit_zero_dt_cache and not getattr(tts.feat_decoder, "mean_mode", False):
            if _install_locdit_zero_dt_cache(estimator):
                targets.append("LocDiT zero-dt embedding cache")

        external_cfm_capture = self._enable_cfm_cuda_graph or self._enable_unified_decode_graph
        if cfg.enable_loc_dit_layer_nvtx:
            patched = _install_locdit_layer_nvtx(estimator)
            if patched:
                targets.append(f"LocDiT layer NVTX ({patched} layers, compile skipped)")
            estimator._compiled = True
        elif self._enable_unified_decode_graph:
            try:
                compiled_ro = torch.compile(estimator, mode="reduce-overhead", fullgraph=False)
                compiled_ro._compiled = True
                tts.feat_decoder.estimator = compiled_ro
                targets.append("LocDiT (reduce-overhead serving + lazy no-cg unified capture)")
            except Exception as e:
                self._raise_if_cuda_runtime_error(e)
                logger.warning("torch.compile LocDiT dual-mode failed: %s", e)
                try:
                    tts.feat_decoder.estimator = self._voxcpm2_compile_without_inductor_cudagraphs(
                        estimator,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    tts.feat_decoder.estimator._compiled = True
                    self._estimator_for_unified_capture = tts.feat_decoder.estimator
                    targets.append("LocDiT (no-cg only, unified capture)")
                except Exception as inner_e:
                    self._raise_if_cuda_runtime_error(inner_e)
                    logger.warning("torch.compile LocDiT failed completely: %s", inner_e)
        elif external_cfm_capture:
            try:
                if cfg.enable_loc_dit_reduce_overhead_no_cg:
                    tts.feat_decoder.estimator = self._voxcpm2_compile_without_inductor_cudagraphs(
                        estimator,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    targets.append("LocDiT (reduce-overhead no-inductor-cudagraph + external CUDA Graph)")
                else:
                    tts.feat_decoder.estimator = torch.compile(
                        estimator,
                        fullgraph=cfg.enable_loc_dit_fullgraph_no_cg,
                        options={
                            "triton.cudagraphs": False,
                            "triton.cudagraph_trees": False,
                        },
                    )
                    targets.append("LocDiT (no-inductor-cudagraph + external CUDA Graph)")
                tts.feat_decoder.estimator._compiled = True
            except Exception as e:
                self._raise_if_cuda_runtime_error(e)
                logger.warning("torch.compile LocDiT for external CUDA Graph failed: %s", e)
        else:
            try:
                tts.feat_decoder.estimator = torch.compile(
                    estimator,
                    mode="reduce-overhead",
                    fullgraph=False,
                )
                tts.feat_decoder.estimator._compiled = True
                targets.append("LocDiT")
            except Exception as e:
                self._raise_if_cuda_runtime_error(e)
                logger.warning("torch.compile LocDiT failed: %s", e)

        try:
            if not hasattr(tts.feat_encoder, "_compiled"):
                if self._enable_unified_decode_graph:
                    feat_encoder = tts.feat_encoder
                    compiled_ro = torch.compile(feat_encoder, mode="reduce-overhead", fullgraph=False)
                    compiled_ro._compiled = True
                    tts.feat_encoder = compiled_ro
                    targets.append("feat_encoder (reduce-overhead serving + lazy no-cg unified capture)")
                elif external_cfm_capture:
                    tts.feat_encoder = torch.compile(
                        tts.feat_encoder,
                        fullgraph=False,
                        options={
                            "triton.cudagraphs": False,
                            "triton.cudagraph_trees": False,
                        },
                    )
                    tts.feat_encoder._compiled = True
                    targets.append("feat_encoder (no-inductor-cudagraph)")
                else:
                    tts.feat_encoder = torch.compile(tts.feat_encoder, mode="reduce-overhead", fullgraph=False)
                    tts.feat_encoder._compiled = True
                    targets.append("feat_encoder")
        except Exception as e:
            self._raise_if_cuda_runtime_error(e)
            logger.warning("torch.compile feat_encoder failed: %s", e)

        if self._compile_vae and not self._enable_vae_cuda_graph:
            try:
                if not hasattr(tts.audio_vae, "_compiled"):
                    tts.audio_vae.decode = torch.compile(tts.audio_vae.decode, mode="reduce-overhead", fullgraph=False)
                    tts.audio_vae._compiled = True
                    targets.append("AudioVAE")
            except Exception as e:
                self._raise_if_cuda_runtime_error(e)
                logger.warning("torch.compile AudioVAE failed: %s", e)

        if not self._enable_cuda_graph:
            if not getattr(self.model, "_selective_compiled", False):
                try:
                    targets.extend(f"scaffold.{t}" for t in self.model.compile_selective())
                    self.model._selective_compiled = True
                except Exception as e:
                    self._raise_if_cuda_runtime_error(e)
                    logger.warning("scaffold compile failed: %s", e)

            if not getattr(self.residual_model, "_selective_compiled", False):
                try:
                    targets.extend(f"residual.{t}" for t in self.residual_model.compile_selective())
                    self.residual_model._selective_compiled = True
                except Exception as e:
                    self._raise_if_cuda_runtime_error(e)
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
                self._raise_if_cuda_runtime_error(e)
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

    @staticmethod
    def _nullify_volatile_metadata(ctx: _ForwardContextLike) -> _ForwardContextLike:
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

            with torch.cuda.graph(g.graph, pool=current_platform.get_global_graph_pool()):
                g.output = model(**call_kwargs)

        logger.info("CUDA Graph captured for %s (batch_size=%d)", label, batch_size)
        return g

    def _replay_graph(
        self,
        g: _CapturedGraph,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        batch_size: int,
        *,
        clone_output: bool = True,
    ) -> torch.Tensor:
        """Copy fresh inputs into static buffers, then replay.

        No metadata copy needed: persistent buffers (seq_lens, slot_mapping,
        etc.) are updated in-place by the model runner.  scheduler_metadata
        was nullified at capture time so no kernel references it.
        """
        g.input_embeds[:batch_size].copy_(inputs_embeds[:batch_size])
        g.positions[:batch_size].copy_(positions[:batch_size])
        g.graph.replay()
        output = g.output[:batch_size]
        return output.clone() if clone_output else output

    def _should_use_decode_graph(self, batch_size: int) -> bool:
        if batch_size > self._max_cached_graphs:
            return False
        policy = self._decode_graph_capture_policy
        if policy == "all":
            return True
        if policy == "power2":
            return batch_size > 0 and (batch_size & (batch_size - 1)) == 0
        if policy == "power2_or_cached":
            return (
                batch_size in self._scaffold_graphs
                or batch_size in self._residual_graphs
                or (batch_size > 0 and (batch_size & (batch_size - 1)) == 0)
            )
        return True

    @staticmethod
    def _build_unified_graph_bucket_sizes(max_size: int) -> frozenset[int]:
        if max_size <= 0:
            return frozenset()
        sizes = {1}
        size = 2
        while size <= min(max_size, 16):
            sizes.add(size)
            size <<= 1
        bucket = 24
        while bucket <= max_size:
            sizes.add(bucket)
            bucket += 8
        sizes.add(max_size)
        return frozenset(size for size in sizes if 0 < size <= max_size)

    def _select_unified_graph_bucket_size(self, batch_size: int) -> int | None:
        if (
            batch_size <= 0
            or batch_size > self._unified_decode_graph_max_batch_size
            or batch_size > self._max_cached_graphs
        ):
            return None
        return min((size for size in self._unified_graph_bucket_sizes if size >= batch_size), default=None)

    def _capture_vae_graph(self, feat: torch.Tensor) -> _CapturedVAEGraph:
        batch_size = feat.shape[0]
        num_frames = feat.shape[-1]
        input_feat = torch.zeros_like(feat)
        sr_cond = self._get_vae_decode_sr_cond(feat.device)

        with torch.no_grad():
            for _ in range(3):
                output = self.tts.audio_vae.decode(input_feat, sr_cond=sr_cond)

            g = _CapturedVAEGraph(
                graph=torch.cuda.CUDAGraph(),
                input_feat=input_feat,
                output=output,
            )
            with torch.cuda.graph(g.graph, pool=current_platform.get_global_graph_pool()):
                g.output = self.tts.audio_vae.decode(g.input_feat, sr_cond=sr_cond)

        logger.info("CUDA Graph captured for AudioVAE decode (batch_size=%d, frames=%d)", batch_size, num_frames)
        return g

    def _get_vae_decode_sr_cond(self, device: torch.device) -> torch.Tensor:
        sr_cond = self._vae_decode_sr_cond
        if sr_cond is None or sr_cond.device != device:
            sr_cond = torch.tensor(
                [self.tts.audio_vae.out_sample_rate],
                device=device,
                dtype=torch.int32,
            )
            self._vae_decode_sr_cond = sr_cond
        return sr_cond

    def _run_vae_decode(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.device.type != current_omni_platform.device_type:
            return self.tts.audio_vae.decode(feat)

        sr_cond = self._get_vae_decode_sr_cond(feat.device)
        if not self._enable_vae_cuda_graph:
            return self.tts.audio_vae.decode(feat, sr_cond=sr_cond)

        graph_key = (feat.shape[0], feat.shape[-1])
        graph = self._vae_graphs.get(graph_key)
        if graph is None:
            graph = self._capture_vae_graph(feat)
            self._vae_graphs[graph_key] = graph

        graph.input_feat.copy_(feat)
        graph.graph.replay()
        return graph.output

    def _capture_cfm_graph(
        self,
        mu: torch.Tensor | None,
        cond: torch.Tensor,
        *,
        batch_size: int | None = None,
        dit_hidden: int | None = None,
    ) -> _CapturedCFMGraph:
        if mu is not None:
            batch_size = mu.shape[0]
            dit_hidden = mu.shape[-1]
            device = mu.device
            dtype = mu.dtype
        else:
            assert batch_size is not None
            assert dit_hidden is not None
            device = cond.device
            dtype = cond.dtype
        graph_buffers = _CFMBufferManager(
            device=device,
            dtype=dtype,
            feat_dim=self._feat_dim,
            patch_size=self._patch_size,
            dit_hidden_size=dit_hidden,
            max_batch_size=batch_size,
        )
        static_mu = torch.zeros(batch_size, dit_hidden, device=device, dtype=dtype)
        static_cond = torch.zeros_like(cond)
        static_noise = torch.zeros(batch_size, self._feat_dim, self._patch_size, device=device, dtype=dtype)

        with torch.no_grad():
            for _ in range(3):
                output = _optimized_solve_euler_with_noise(
                    self.tts.feat_decoder,
                    static_mu,
                    self._patch_size,
                    static_cond,
                    static_noise,
                    self._inference_timesteps,
                    self._cfg_value,
                    graph_buffers,
                    cfg_cutoff_ratio=self._cfg_cutoff_ratio,
                )

            graph = _CapturedCFMGraph(
                graph=torch.cuda.CUDAGraph(),
                mu=static_mu,
                cond=static_cond,
                noise=static_noise,
                output=output,
                buffers=graph_buffers,
            )
            with torch.cuda.graph(graph.graph, pool=current_platform.get_global_graph_pool()):
                graph.output = _optimized_solve_euler_with_noise(
                    self.tts.feat_decoder,
                    graph.mu,
                    self._patch_size,
                    graph.cond,
                    graph.noise,
                    self._inference_timesteps,
                    self._cfg_value,
                    graph.buffers,
                    cfg_cutoff_ratio=self._cfg_cutoff_ratio,
                )

        logger.info("CUDA Graph captured for VoxCPM2 CFM solver (batch_size=%d)", batch_size)
        return graph

    def _get_cfm_cuda_graph(
        self,
        mu: torch.Tensor | None,
        cond: torch.Tensor,
        *,
        batch_size: int | None = None,
        dit_hidden: int | None = None,
    ) -> _CapturedCFMGraph:
        if mu is not None:
            graph_batch_size = mu.shape[0]
        else:
            assert batch_size is not None
            graph_batch_size = batch_size
        graph = self._cfm_graphs.get(graph_batch_size)
        if graph is None:
            graph = self._capture_cfm_graph(mu, cond, batch_size=batch_size, dit_hidden=dit_hidden)
            self._cfm_graphs[graph_batch_size] = graph
        return graph

    def _run_cfm_cuda_graph(self, mu: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        graph = self._get_cfm_cuda_graph(mu, cond)
        with _NvtxRange("voxcpm2.cfm.graph_copy_mu"):
            graph.mu.copy_(mu)
        return self._run_cfm_cuda_graph_from_static_mu(graph, cond)

    def _run_cfm_cuda_graph_from_static_mu(self, graph: _CapturedCFMGraph, cond: torch.Tensor) -> torch.Tensor:
        with _NvtxRange("voxcpm2.cfm.graph_copy_cond"):
            graph.cond.copy_(cond)
        with _NvtxRange("voxcpm2.cfm.graph_noise"):
            graph.noise.normal_()
        with _NvtxRange("voxcpm2.cfm.graph_replay"):
            graph.graph.replay()
        with _NvtxRange("voxcpm2.cfm.graph_output_clone"):
            return graph.output.clone()

    def _run_cfm_cuda_graph_to_state_buffer(
        self,
        state: _RequestState,
        mu: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        graph = self._get_cfm_cuda_graph(mu, cond)
        with _NvtxRange("voxcpm2.cfm.graph_copy_mu"):
            graph.mu.copy_(mu)
        with _NvtxRange("voxcpm2.cfm.graph_copy_cond"):
            graph.cond.copy_(cond)
        with _NvtxRange("voxcpm2.cfm.graph_noise"):
            if self._deterministic_cfm_noise:
                self._fill_deterministic_cfm_noise(state, graph.noise)
            else:
                graph.noise.normal_()
        with _NvtxRange("voxcpm2.cfm.graph_replay"):
            graph.graph.replay()

        output_shape = graph.output.shape
        if (
            state.cfm_output_gpu is None
            or state.cfm_output_gpu.shape != output_shape
            or state.cfm_output_gpu.device != graph.output.device
            or state.cfm_output_gpu.dtype != graph.output.dtype
        ):
            state.cfm_output_gpu = torch.empty(output_shape, device=graph.output.device, dtype=graph.output.dtype)
        with _NvtxRange("voxcpm2.cfm.graph_output_copy"):
            state.cfm_output_gpu.copy_(graph.output)
        return state.cfm_output_gpu.transpose(1, 2)

    # -------------------- unified decode graph --------------------

    def _capture_unified_decode_graph(self, batch_size: int) -> _CapturedUnifiedDecodeGraph:
        H = self.config.hidden_size
        D = self._feat_dim
        P = self._patch_size
        dev = torch.device(self._device)
        dtype = self._side_dtype

        self._setup_cfm_buffers()
        if self._enable_torch_compile:
            self._setup_torch_compile()
        self.model.precompute_fused_qkv()
        self.residual_model.precompute_fused_qkv()

        tts = self.tts
        dit_hidden = tts.lm_to_dit_proj.out_features + tts.res_to_dit_proj.out_features
        cfm_buffers = _CFMBufferManager(
            device=dev,
            dtype=dtype,
            feat_dim=D,
            patch_size=P,
            dit_hidden_size=dit_hidden,
            max_batch_size=batch_size,
        )

        g = _CapturedUnifiedDecodeGraph(
            graph=torch.cuda.CUDAGraph(),
            batch_size=batch_size,
            input_embeds=torch.zeros(batch_size, H, device=dev, dtype=dtype),
            positions=torch.zeros(batch_size, device=dev, dtype=torch.long),
            prev_feat_embed=torch.zeros(batch_size, H, device=dev, dtype=dtype),
            prefix_feat_cond=torch.zeros(batch_size, P, D, device=dev, dtype=dtype),
            next_feat_embed=torch.zeros(batch_size, H, device=dev, dtype=dtype),
            cfm_output=torch.zeros(batch_size, D, P, device=dev, dtype=dtype),
            lm_hidden=torch.zeros(batch_size, H, device=dev, dtype=dtype),
            cfm_buffers=cfm_buffers,
            cfm_noise=torch.zeros(batch_size, D, P, device=dev, dtype=dtype),
        )

        dit_proj = self._dit_proj_fn

        def unified_fwd():
            scaffold_h = self.model(input_ids=None, positions=g.positions, inputs_embeds=g.input_embeds)
            if isinstance(scaffold_h, tuple):
                scaffold_h = scaffold_h[0]
            lm_h = tts.fsq_layer(scaffold_h)
            res_input = tts.fusion_concat_proj(torch.cat([lm_h, g.prev_feat_embed], dim=-1))
            res_h = self.residual_model(positions=g.positions, inputs_embeds=res_input)
            dit_h = dit_proj(lm_h, res_h)
            cond = g.prefix_feat_cond.transpose(1, 2).contiguous()
            cfm_out = _optimized_solve_euler_with_noise(
                tts.feat_decoder,
                dit_h,
                P,
                cond,
                g.cfm_noise,
                self._inference_timesteps,
                self._cfg_value,
                cfm_buffers,
                cfg_cutoff_ratio=self._cfg_cutoff_ratio,
            )
            feat_enc = tts.feat_encoder(cfm_out.transpose(1, 2).unsqueeze(1)).squeeze(1)
            next_embed = tts.enc_to_lm_proj(feat_enc)
            return next_embed, cfm_out, lm_h

        ctx = get_forward_context()
        capture_context = override_forward_context(self._nullify_volatile_metadata(ctx))

        original_estimator = tts.feat_decoder.estimator
        capture_estimator = self._voxcpm2_compile_unified_capture_estimator(original_estimator)
        tts.feat_decoder.estimator = capture_estimator

        original_feat_encoder = tts.feat_encoder
        capture_feat_encoder = self._voxcpm2_compile_unified_capture_feat_encoder(original_feat_encoder)
        tts.feat_encoder = capture_feat_encoder

        try:
            with capture_context:
                with torch.no_grad():
                    for _ in range(3):
                        g.cfm_noise.normal_()
                        unified_fwd()

                    g.cfm_noise.normal_()
                    with torch.cuda.graph(g.graph, pool=current_platform.get_global_graph_pool()):
                        g.next_feat_embed, g.cfm_output, g.lm_hidden = unified_fwd()
        finally:
            tts.feat_decoder.estimator = original_estimator
            tts.feat_encoder = original_feat_encoder
        self._unified_graph_stats.captures += 1
        logger.info(
            "CUDA Graph captured for unified decode (batch_size=%d, captures=%d)",
            batch_size,
            self._unified_graph_stats.captures,
        )
        return g

    def _unified_decode_graph_skip_reason(
        self,
        *,
        can_use_graph: bool,
        is_all_decode: bool,
        num_reqs: int,
    ) -> str | None:
        if not self._enable_unified_decode_graph:
            return "disabled"
        if not can_use_graph:
            return "graph_not_ready"
        if not is_all_decode:
            return "not_all_decode"
        if num_reqs > self._unified_decode_graph_max_batch_size:
            return "batch_exceeds_unified_max"
        if num_reqs > self._max_cached_graphs:
            return "batch_too_large"
        graph_size = self._select_unified_graph_bucket_size(num_reqs)
        if graph_size is None:
            return "capture_policy"
        if num_reqs > 1 and not self._runner_assisted_unified_decode_graph_active:
            return "runner_full_metadata_missing"
        for req_id, _, _, _ in self._pending_requests:
            state = self._active_states[req_id]
            if not state.prefill_completed:
                return "prefill_incomplete"
            if state.prev_feat_embed is None or state.curr_prefix_feat_cond is None:
                return "state_not_ready"
        return None

    def _maybe_log_unified_graph_stats(self) -> None:
        if not self._enable_profiling:
            return
        stats = self._unified_graph_stats
        total_skips = stats.total_skips
        should_log = (stats.replays - stats.logged_replays >= 100) or (total_skips - stats.logged_skips >= 50)
        if not should_log:
            return
        stats.logged_replays = stats.replays
        stats.logged_skips = total_skips
        logger.info(
            "VoxCPM2 unified graph stats: captures=%d replays=%d skips=%s real_batch_sizes=%s graph_bucket_sizes=%s",
            stats.captures,
            stats.replays,
            stats.skips,
            dict(sorted(stats.real_batch_sizes.items())),
            dict(sorted(stats.graph_bucket_sizes.items())),
        )

    def _forward_unified_decode(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        num_reqs: int,
    ) -> torch.Tensor:
        graph_size = self._select_unified_graph_bucket_size(num_reqs)
        if graph_size is None:
            raise RuntimeError(f"No unified decode graph bucket for batch size {num_reqs}")
        g = self._unified_graphs.get(graph_size)
        if g is None:
            g = self._capture_unified_decode_graph(graph_size)
            self._unified_graphs[graph_size] = g

        self._unified_graph_stats.replays += 1
        if self._enable_profiling:
            self._unified_graph_stats.record_replay_batch(real_batch_size=num_reqs, graph_bucket_size=graph_size)
        self._maybe_log_unified_graph_stats()
        states: list[_RequestState] = []
        commit_mask: list[bool] = []
        for req_id, _is_prefill, _embeds, _n in self._pending_requests:
            state = self._active_states[req_id]
            states.append(state)
            already_stopping = state.is_stopping
            commit_mask.append(not already_stopping)
            if not already_stopping:
                state.decode_step_count += 1
                if state.decode_step_count >= self._max_decode_steps:
                    state.is_stopping = True

        self._perf.start("unified.copy_inputs")
        for i, state in enumerate(states):
            pfe = state.prev_feat_embed
            g.prev_feat_embed[i].copy_(pfe.squeeze(0) if pfe.ndim > 1 else pfe)
            pfc = state.curr_prefix_feat_cond
            if pfc.ndim == 2:
                g.prefix_feat_cond[i].copy_(pfc)
            else:
                g.prefix_feat_cond[i].copy_(pfc.squeeze(0))

        g.input_embeds[:num_reqs].copy_(inputs_embeds[:num_reqs])
        g.positions[:num_reqs].copy_(positions[:num_reqs])
        if graph_size > num_reqs:
            # Runner-assisted FULL metadata pads attention to graph_size. Keep
            # padded rows as valid duplicates instead of position-0 zero rows;
            # some attention backends may still touch padded slots before
            # honoring seq_lens=0.
            last = num_reqs - 1
            g.input_embeds[num_reqs:graph_size].copy_(g.input_embeds[last : last + 1].expand(graph_size - num_reqs, -1))
            g.positions[num_reqs:graph_size].copy_(g.positions[last : last + 1].expand(graph_size - num_reqs))
            g.prev_feat_embed[num_reqs:graph_size].copy_(
                g.prev_feat_embed[last : last + 1].expand(graph_size - num_reqs, -1)
            )
            g.prefix_feat_cond[num_reqs:graph_size].copy_(
                g.prefix_feat_cond[last : last + 1].expand(graph_size - num_reqs, -1, -1)
            )
        g.cfm_noise.normal_()
        self._perf.stop("unified.copy_inputs")

        self._perf.start("unified.replay")
        g.graph.replay()
        self._perf.stop("unified.replay")

        self._perf.start("unified.commit")
        with torch.no_grad():
            all_stop_logits = self._stop_fn(g.lm_hidden[:num_reqs])
        for i, state in enumerate(states):
            stop_logits_i = all_stop_logits[i : i + 1]
            if not commit_mask[i]:
                state.precomputed_stop_logits = stop_logits_i
                continue
            next_embed_i = g.next_feat_embed[i : i + 1].clone()
            cfm_out_i = g.cfm_output[i : i + 1].transpose(1, 2)
            self._commit_decode_state(state, stop_logits_i, next_embed_i, cfm_out_i)
        self._perf.stop("unified.commit")

        self._perf.start("unified.audio")
        self._precompute_stop_flags_for_audio_collect(states)
        ready_audio = self._drain_ready_audio_copies_for_states(states)
        audio_by_req = self._collect_audio_batch(states, initial_delayed_chunks_by_req=ready_audio)
        for state in states:
            self._results_queue.append((state.request_id, state.precomputed_stop_logits))
            self._audio_queue.append((state.request_id, audio_by_req.get(state.request_id)))

        self._pending_requests.clear()
        self._flush_deferred_cleanup()
        self._perf.stop("unified.audio")
        return g.next_feat_embed[:num_reqs]

    # -------------------- vllm hooks --------------------

    def get_runner_assisted_full_attention_metadata_request(
        self,
        *,
        req_ids: Sequence[str],
        num_reqs: int,
        num_scheduled_tokens: Sequence[int],
        num_computed_tokens: Sequence[int],
        max_num_scheduled_tokens: int,
    ) -> RunnerAssistedFullAttentionMetadataRequest | None:
        if not self._enable_unified_decode_graph or not self._enable_cuda_graph:
            return None
        if num_reqs <= 1 or num_reqs > self._unified_decode_graph_max_batch_size:
            return None
        if num_reqs > self._max_cached_graphs or self._select_unified_graph_bucket_size(num_reqs) is None:
            return None
        if max_num_scheduled_tokens != 1 or len(num_scheduled_tokens) != num_reqs:
            return None
        if any(int(n) != 1 for n in num_scheduled_tokens):
            return None
        if len(req_ids) != num_reqs or len(num_computed_tokens) != num_reqs:
            return None
        if any(int(n) <= 0 for n in num_computed_tokens):
            return None
        tts_compiled = getattr(self.tts.feat_decoder.estimator, "_compiled", False) if self._tts is not None else False
        if not tts_compiled or self._cuda_graph_warmup_steps < self._cuda_graph_warmup_threshold:
            return None
        for req_id in req_ids:
            state = self._active_states.get(req_id)
            if (
                state is None
                or not state.prefill_completed
                or state.prev_feat_embed is None
                or state.curr_prefix_feat_cond is None
            ):
                return None
        bucket_size = self._select_unified_graph_bucket_size(num_reqs)
        if bucket_size is None:
            return None
        return RunnerAssistedFullAttentionMetadataRequest(
            num_reqs_padded=bucket_size,
            for_cudagraph_capture=bucket_size not in self._unified_graphs,
        )

    def set_runner_assisted_full_attention_metadata_context(
        self,
        *,
        enabled: bool,
        num_reqs: int = 0,
    ) -> None:
        enabled = bool(enabled)
        self._runner_assisted_unified_decode_graph_active = enabled
        self._runner_assisted_unified_decode_graph_batch_size = int(num_reqs) if enabled else 0

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

        self._last_audio_output_req_ids = [req_id for req_id, _, _, _ in self._pending_requests]
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

        unified_skip_reason = self._unified_decode_graph_skip_reason(
            can_use_graph=can_use_graph,
            is_all_decode=is_all_decode,
            num_reqs=num_reqs,
        )
        if unified_skip_reason is None:
            self._perf.start("unified_decode")
            result = self._forward_unified_decode(inputs_embeds, positions, num_reqs)
            self._perf.stop("unified_decode")
            self._perf.stop("forward_total")
            return result
        if self._enable_unified_decode_graph and num_reqs > 0 and unified_skip_reason is not None:
            self._unified_graph_stats.record_skip(unified_skip_reason)
            self._maybe_log_unified_graph_stats()

        use_segmented_decode_graph = (
            can_use_graph
            and is_all_decode
            and num_reqs <= self._max_cached_graphs
            and (not self._enable_unified_decode_graph or unified_skip_reason is not None)
            and self._should_use_decode_graph(num_reqs)
        )

        if use_segmented_decode_graph:
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

        token_offset = 0
        residual_inputs: list[torch.Tensor] = []
        residual_positions: list[torch.Tensor] = []
        req_metas: list[tuple[_RequestState, bool, _PrefillResidualMeta | _DecodeResidualMeta]] = []
        pending_decode_fsq: list[tuple[_RequestState, torch.Tensor, torch.Tensor]] = []

        def flush_decode_fsq_batch() -> None:
            if not pending_decode_fsq:
                return
            if len(pending_decode_fsq) == 1:
                state, req_hidden, req_pos = pending_decode_fsq[0]
                res_input, meta = self._prepare_residual_decode(state, req_hidden, dev)
                residual_inputs.append(res_input)
                residual_positions.append(req_pos)
                req_metas.append((state, False, meta))
            else:
                states = [state for state, _, _ in pending_decode_fsq]
                hidden_list = [req_hidden for _, req_hidden, _ in pending_decode_fsq]
                batched = self._prepare_residual_decode_batch(states, hidden_list, dev)
                for (state, _, req_pos), (res_input, meta) in zip(pending_decode_fsq, batched):
                    residual_inputs.append(res_input)
                    residual_positions.append(req_pos)
                    req_metas.append((state, False, meta))
            pending_decode_fsq.clear()

        for req_id, is_prefill, _req_embeds, n in self._pending_requests:
            state = self._switch_to_request(req_id)
            req_hidden = scaffold_hidden[token_offset : token_offset + n]
            req_pos = positions[token_offset : token_offset + n]

            if is_prefill:
                flush_decode_fsq_batch()
                res_input, meta = self._prepare_residual_prefill(state, req_hidden, dev)
            elif state.prefill_completed:
                if (
                    self._enable_batched_fsq_fusion
                    and n == 1
                    and state.prev_feat_embed is not None
                    and req_hidden.ndim == 2
                    and req_hidden.shape[0] == 1
                ):
                    pending_decode_fsq.append((state, req_hidden, req_pos))
                    token_offset += n
                    if len(pending_decode_fsq) >= self._batched_fsq_fusion_max_batch:
                        flush_decode_fsq_batch()
                    continue
                flush_decode_fsq_batch()
                res_input, meta = self._prepare_residual_decode(state, req_hidden, dev)
            else:
                flush_decode_fsq_batch()
                token_offset += n
                self._results_queue.append((req_id, None))
                self._audio_queue.append((req_id, None))
                continue

            residual_inputs.append(res_input)
            residual_positions.append(req_pos)
            req_metas.append((state, is_prefill, meta))
            token_offset += n
        flush_decode_fsq_batch()

        # Phase 2: batch residual_model forward
        if residual_inputs:
            batch_in = torch.cat(residual_inputs, dim=0)
            batch_pos = torch.cat(residual_positions, dim=0)

            residual_batch_size = batch_in.shape[0]
            use_residual_graph = (
                self._enable_cuda_graph
                and use_segmented_decode_graph
                and is_all_decode
                and graph_ready
                and residual_batch_size == num_reqs  # 1 token per request
                and self._should_use_decode_graph(residual_batch_size)
            )

            self._perf.start("residual_fwd")
            with _NvtxRange("voxcpm2.residual_fwd"):
                if use_residual_graph:
                    if residual_batch_size not in self._residual_graphs:
                        self._residual_graphs[residual_batch_size] = self._capture_graph(
                            self.residual_model, residual_batch_size, "residual", is_residual=True
                        )
                    batch_out = self._replay_graph(
                        self._residual_graphs[residual_batch_size],
                        batch_in,
                        batch_pos,
                        residual_batch_size,
                    )
                else:
                    batch_out = self.residual_model(batch_pos, batch_in)
            self._perf.stop("residual_fwd")

            can_finish_decode_batch = (
                bool(req_metas)
                and all(not is_prefill for _, is_prefill, _ in req_metas)
                and all(x.shape[0] == 1 for x in residual_inputs)
            )
            if can_finish_decode_batch:
                self._finish_decode_batch(req_metas, batch_out)
                self._precompute_stop_flags_for_audio_collect([state for state, _, _ in req_metas])
                ready_audio_by_req = self._drain_ready_audio_copies_for_states([state for state, _, _ in req_metas])
                audio_by_req = self._collect_audio_batch(
                    [state for state, _, _ in req_metas],
                    initial_delayed_chunks_by_req=ready_audio_by_req,
                )
                for state, _, _ in req_metas:
                    self._results_queue.append((state.request_id, state.precomputed_stop_logits))
                    self._audio_queue.append((state.request_id, audio_by_req.get(state.request_id)))
            else:
                offset = 0
                decoded_states: list[_RequestState] = []
                prefill_batch: list[tuple[_RequestState, _PrefillResidualMeta, torch.Tensor]] = []
                for idx, (state, is_prefill, meta) in enumerate(req_metas):
                    n = residual_inputs[idx].shape[0]
                    res_out = batch_out[offset : offset + n]
                    offset += n

                    if is_prefill:
                        prefill_batch.append((state, meta, res_out))
                    else:
                        self._finish_decode(state, meta, res_out)
                        decoded_states.append(state)

                if prefill_batch and self._runtime_config.enable_batched_prefill_tail:
                    self._finish_prefill_batch(prefill_batch)
                else:
                    for state, meta, res_out in prefill_batch:
                        self._finish_prefill(state, meta, res_out, dev)

                collect_states = [state for state, _, _ in req_metas]
                self._precompute_stop_flags_for_audio_collect(collect_states)
                ready_audio_by_req = self._drain_ready_audio_copies_for_states(collect_states)
                audio_by_req = self._collect_audio_batch(
                    collect_states,
                    initial_delayed_chunks_by_req={
                        state.request_id: ready_audio_by_req.get(state.request_id)
                        for state, is_prefill, _ in req_metas
                        if not is_prefill
                    },
                )
                for state, is_prefill, _ in req_metas:
                    self._results_queue.append((state.request_id, state.precomputed_stop_logits))
                    self._audio_queue.append((state.request_id, audio_by_req.get(state.request_id)))

        self._pending_requests.clear()
        self._flush_deferred_cleanup()
        self._perf.stop("forward_total")
        return scaffold_hidden

    # -------------------- prefill / decode helpers --------------------

    def _prepare_residual_prefill(
        self,
        state: _RequestState,
        base_lm_out: torch.Tensor,
        dev: torch.device,
    ) -> tuple[torch.Tensor, _PrefillResidualMeta]:
        tts = self.tts
        text_mask, feat_mask, feat, feat_embed = state.prefill_masks
        state.prefill_masks = None

        tts_len = text_mask.shape[1]
        scaffold_len = base_lm_out.shape[0]
        assert scaffold_len == tts_len, (
            f"voxcpm2 prefill length mismatch: scaffold_len={scaffold_len} tts_len={tts_len}; "
            "caller must pad prompt_token_ids to the full prefill length "
            "(see serving_speech._build_voxcpm2_prompt or the offline example)."
        )
        enc_out = base_lm_out.unsqueeze(0)

        prefix_feat_cond = (
            feat[:, -1, ...]
            if feat.shape[1] > 0
            else torch.zeros(1, self._patch_size, self._feat_dim, device=dev, dtype=self._side_dtype)
        )
        enc_outputs = tts.fsq_layer(enc_out) * feat_mask.unsqueeze(-1) + enc_out * text_mask.unsqueeze(-1)
        lm_hidden = enc_outputs[:, -1, :]

        residual_input = tts.fusion_concat_proj(torch.cat([enc_outputs, feat_mask.unsqueeze(-1) * feat_embed], dim=-1))
        meta = _PrefillResidualMeta(lm_hidden=lm_hidden, prefix_feat_cond=prefix_feat_cond)
        return residual_input.squeeze(0), meta

    def _prepare_residual_decode(
        self,
        state: _RequestState,
        base_lm_out: torch.Tensor,
        dev: torch.device,
    ) -> tuple[torch.Tensor, _DecodeResidualMeta]:
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
        return res_input, _DecodeResidualMeta(new_lm_hidden=lm_h)

    def _prepare_residual_decode_batch(
        self,
        states: list[_RequestState],
        base_lm_outs: list[torch.Tensor],
        dev: torch.device,
    ) -> list[tuple[torch.Tensor, _DecodeResidualMeta]]:
        tts = self.tts

        for state in states:
            state.decode_step_count += 1
            if state.decode_step_count >= self._max_decode_steps:
                logger.warning(
                    "MAX_DECODE_STEPS for %s (%d), forcing stop",
                    state.request_id,
                    state.decode_step_count,
                )
                state.is_stopping = True

        with _NvtxRange("voxcpm2.fsq_fusion_batch"):
            hidden_batch = torch.cat(
                [h.reshape(1, -1) if h.ndim == 1 else h for h in base_lm_outs],
                dim=0,
            )
            lm_h_batch = tts.fsq_layer(hidden_batch)
            if lm_h_batch.ndim == 1:
                lm_h_batch = lm_h_batch.unsqueeze(0)
            prev_batch = torch.cat(
                [state.prev_feat_embed.to(dev, dtype=self._side_dtype).reshape(1, -1) for state in states],
                dim=0,
            )
            res_batch = tts.fusion_concat_proj(torch.cat([lm_h_batch, prev_batch], dim=-1))

        return [
            (
                res_batch[i : i + 1],
                _DecodeResidualMeta(new_lm_hidden=lm_h_batch[i : i + 1]),
            )
            for i in range(len(states))
        ]

    def _run_cfm(self, dit_h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        with _NvtxRange("voxcpm2.cfm"):
            if self._cfm_buffers is not None:
                if self._enable_cfm_cuda_graph and dit_h.device.type == current_omni_platform.device_type:
                    return self._run_cfm_cuda_graph(dit_h, cond).transpose(1, 2)
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

    def _run_cfm_for_state(self, state: _RequestState, dit_h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        with _NvtxRange("voxcpm2.cfm"):
            if self._cfm_buffers is not None:
                if self._enable_cfm_cuda_graph and dit_h.device.type == current_omni_platform.device_type:
                    if self._deterministic_cfm_noise and not self._enable_cfm_prealloc_output:
                        graph = self._get_cfm_cuda_graph(dit_h, cond)
                        with _NvtxRange("voxcpm2.cfm.graph_copy_mu"):
                            graph.mu.copy_(dit_h)
                        with _NvtxRange("voxcpm2.cfm.graph_copy_cond"):
                            graph.cond.copy_(cond)
                        with _NvtxRange("voxcpm2.cfm.graph_noise"):
                            self._fill_deterministic_cfm_noise(state, graph.noise)
                        with _NvtxRange("voxcpm2.cfm.graph_replay"):
                            graph.graph.replay()
                        with _NvtxRange("voxcpm2.cfm.graph_output_clone"):
                            return graph.output.clone().transpose(1, 2)
                    if not self._enable_cfm_prealloc_output:
                        return self._run_cfm_cuda_graph(dit_h, cond).transpose(1, 2)
                    return self._run_cfm_cuda_graph_to_state_buffer(state, dit_h, cond)
                if self._deterministic_cfm_noise:
                    noise = self._cfm_buffers.noise[: dit_h.shape[0]]
                    self._fill_deterministic_cfm_noise(state, noise)
                    return _optimized_solve_euler_with_noise(
                        self.tts.feat_decoder,
                        dit_h,
                        self._patch_size,
                        cond,
                        noise,
                        self._inference_timesteps,
                        self._cfg_value,
                        self._cfm_buffers,
                        cfg_cutoff_ratio=self._cfg_cutoff_ratio,
                        perf=self._perf,
                    ).transpose(1, 2)
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

    def _fill_deterministic_cfm_noise(self, state: _RequestState, out: torch.Tensor) -> None:
        """Fill CFM noise deterministically for benchmark replay only."""
        request_key = state.request_id.split("_", 1)[0]
        if not request_key.isdigit():
            request_key = state.request_id
        key = f"{self._deterministic_cfm_seed}:{request_key}:{state.cfm_noise_step}".encode()
        digest = hashlib.blake2b(key, digest_size=8).digest()
        seed = int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF
        gen = torch.Generator(device=out.device)
        gen.manual_seed(seed)
        out.normal_(generator=gen)
        state.cfm_noise_step += 1

    def _finish_prefill_batch(self, batch: list[tuple[_RequestState, _PrefillResidualMeta, torch.Tensor]]) -> None:
        if len(batch) == 1:
            state, meta, res_out = batch[0]
            self._finish_prefill(state, meta, res_out, None)
            return

        self._perf.start("prefill_tail_batch")
        tts = self.tts
        self._setup_cfm_buffers()
        if self._enable_torch_compile:
            self._setup_torch_compile()

        dit_hs: list[torch.Tensor] = []
        conds: list[torch.Tensor] = []
        stop_logits_list: list[torch.Tensor] = []

        for state, meta, res_out in batch:
            lm_hidden = meta.lm_hidden
            prefix_feat_cond = meta.prefix_feat_cond
            residual_hidden = res_out[-1:, :]
            stop_logits = tts.stop_head(tts.stop_actn(tts.stop_proj(lm_hidden))).detach()
            dit_h = torch.cat([tts.lm_to_dit_proj(lm_hidden), tts.res_to_dit_proj(residual_hidden)], dim=-1)
            cond = prefix_feat_cond.transpose(1, 2).contiguous()
            dit_hs.append(dit_h)
            conds.append(cond)
            stop_logits_list.append(stop_logits)

        batched_dit_h = torch.cat(dit_hs, dim=0)
        batched_cond = torch.cat(conds, dim=0)
        b = batched_dit_h.shape[0]

        if self._deterministic_cfm_noise and self._cfm_buffers is not None:
            noise = self._cfm_buffers.noise[:b]
            for i, (state, _, _) in enumerate(batch):
                self._fill_deterministic_cfm_noise(state, noise[i : i + 1])
        else:
            noise = torch.randn(
                b,
                self._feat_dim,
                self._patch_size,
                device=batched_dit_h.device,
                dtype=batched_dit_h.dtype,
            )
        pred_feats = _optimized_solve_euler_with_noise(
            tts.feat_decoder,
            batched_dit_h,
            self._patch_size,
            batched_cond,
            noise,
            self._inference_timesteps,
            self._cfg_value,
            self._cfm_buffers,
            cfg_cutoff_ratio=self._cfg_cutoff_ratio,
        ).transpose(1, 2)

        with torch.no_grad():
            curr_embeds = tts.enc_to_lm_proj(tts.feat_encoder(pred_feats.unsqueeze(1))).squeeze(1)

        for i, (state, meta, res_out) in enumerate(batch):
            pred_feat_i = pred_feats[i : i + 1]
            curr_embed_i = curr_embeds[i : i + 1]
            self._commit_decode_state(state, stop_logits_list[i], curr_embed_i, pred_feat_i)
            state.decode_step_count = 0
            state.request_start_time = time.perf_counter()
            state.prefill_completed = True

        self._perf.stop("prefill_tail_batch")
        if self._enable_profiling:
            logger.info("PREFILL_BATCH[%d requests] tail breakdown:\n%s", len(batch), self._perf.breakdown())
            self._perf.reset()

    def _finish_prefill(
        self,
        state: _RequestState,
        meta: _PrefillResidualMeta,
        res_out: torch.Tensor,
        dev: torch.device | None,
    ) -> None:
        self._perf.start("prefill_tail")
        tts = self.tts
        lm_hidden = meta.lm_hidden
        prefix_feat_cond = meta.prefix_feat_cond
        residual_hidden = res_out[-1:, :]

        self._perf.start("prefill.stop_fn")
        stop_logits = tts.stop_head(tts.stop_actn(tts.stop_proj(lm_hidden))).detach()
        self._perf.stop("prefill.stop_fn")
        self._perf.start("prefill.dit_proj")
        dit_h = torch.cat([tts.lm_to_dit_proj(lm_hidden), tts.res_to_dit_proj(residual_hidden)], dim=-1)
        self._perf.stop("prefill.dit_proj")

        self._setup_cfm_buffers()
        if self._enable_torch_compile:
            self._setup_torch_compile()

        pred_feat = self._run_cfm_for_state(state, dit_h, prefix_feat_cond.transpose(1, 2).contiguous())

        self._perf.start("prefill.feat_encoder")
        with torch.no_grad():
            curr_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)
        self._perf.stop("prefill.feat_encoder")

        self._commit_decode_state(state, stop_logits, curr_embed, pred_feat)
        state.decode_step_count = 0
        state.request_start_time = time.perf_counter()
        state.prefill_completed = True
        self._perf.stop("prefill_tail")

        if logger.isEnabledFor(logging.DEBUG):
            # Only compute the norm (which forces a GPU->CPU sync) if we will log it.
            logger.debug("PREFILL[%s]: patch norm=%.4f", state.request_id, pred_feat.norm().item())
        if self._enable_profiling:
            logger.info("PREFILL[%s] tail breakdown:\n%s", state.request_id, self._perf.breakdown())
        self._perf.reset()

    def _finish_decode(self, state: _RequestState, meta: _DecodeResidualMeta, res_out: torch.Tensor) -> None:
        self._perf.start("decode_step")
        tts = self.tts

        lm_h = meta.new_lm_hidden
        res_h = res_out.unsqueeze(0) if res_out.ndim == 1 else res_out

        dit_proj = getattr(self, "_compiled_dit_proj", None) or self._dit_proj_fn
        stop_fn = getattr(self, "_compiled_stop_fn", None) or self._stop_fn

        pfc = state.curr_prefix_feat_cond.to(self._side_dtype)
        if pfc.ndim == 2:
            pfc = pfc.unsqueeze(0)
        cond = pfc.transpose(1, 2).contiguous()

        dit_h = dit_proj(lm_h, res_h)
        pred_feat = self._run_cfm_for_state(state, dit_h, cond)
        next_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)
        stop_logits = stop_fn(lm_h).detach()
        self._commit_decode_state(state, stop_logits, next_embed, pred_feat)

        self._perf.stop("decode_step")
        if self._enable_profiling and state.decode_step_count % 20 == 0:
            logger.info("Step %d[%s]:\n%s", state.decode_step_count, state.request_id, self._perf.breakdown())

    def _finish_decode_batch(
        self,
        req_metas: list[tuple[_RequestState, bool, _DecodeResidualMeta]],
        batch_out: torch.Tensor,
    ) -> None:
        self._perf.start("decode_step")
        tts = self.tts

        dit_proj = getattr(self, "_compiled_dit_proj", None) or self._dit_proj_fn
        stop_fn = getattr(self, "_compiled_stop_fn", None) or self._stop_fn

        if self._enable_batched_cfm and not self._enable_cfm_cuda_graph:
            states = [state for state, _, _ in req_metas]
            lm_h = torch.cat([meta.new_lm_hidden for _, _, meta in req_metas], dim=0)
            pfc = torch.cat(
                [
                    state.curr_prefix_feat_cond.to(self._side_dtype).unsqueeze(0)
                    if state.curr_prefix_feat_cond.ndim == 2
                    else state.curr_prefix_feat_cond.to(self._side_dtype)
                    for state in states
                ],
                dim=0,
            )
            with _NvtxRange("voxcpm2.dit_proj"):
                dit_h = dit_proj(lm_h, batch_out)
            cond = pfc.transpose(1, 2).contiguous()
            if self._deterministic_cfm_noise and self._cfm_buffers is not None:
                noise = self._cfm_buffers.noise[: dit_h.size(0)]
                for i, state in enumerate(states):
                    self._fill_deterministic_cfm_noise(state, noise[i : i + 1])
                pred_feat = _optimized_solve_euler_with_noise(
                    self.tts.feat_decoder,
                    dit_h,
                    self._patch_size,
                    cond,
                    noise,
                    self._inference_timesteps,
                    self._cfg_value,
                    self._cfm_buffers,
                    cfg_cutoff_ratio=self._cfg_cutoff_ratio,
                    perf=self._perf,
                ).transpose(1, 2)
            else:
                pred_feat = self._run_cfm(dit_h, cond)
            with _NvtxRange("voxcpm2.feat_encoder_feedback"):
                next_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)
            with _NvtxRange("voxcpm2.stop_fn"):
                stop_logits = stop_fn(lm_h).detach()

            for i, state in enumerate(states):
                self._commit_decode_state(
                    state,
                    stop_logits[i : i + 1],
                    next_embed[i : i + 1],
                    pred_feat[i : i + 1],
                )

            self._perf.stop("decode_step")
            return

        for i, (state, _, meta) in enumerate(req_metas):
            lm_h = meta.new_lm_hidden
            res_h = batch_out[i : i + 1]
            pfc = state.curr_prefix_feat_cond.to(self._side_dtype)
            if pfc.ndim == 2:
                pfc = pfc.unsqueeze(0)
            cond = pfc.transpose(1, 2).contiguous()
            with _NvtxRange("voxcpm2.dit_proj"):
                dit_h = dit_proj(lm_h, res_h)
            pred_feat = self._run_cfm_for_state(state, dit_h, cond)
            with _NvtxRange("voxcpm2.feat_encoder_feedback"):
                next_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)
            with _NvtxRange("voxcpm2.stop_fn"):
                stop_logits = stop_fn(lm_h).detach()
            self._commit_decode_state(state, stop_logits, next_embed, pred_feat)

        self._perf.stop("decode_step")

    def _commit_decode_state(
        self,
        state: _RequestState,
        stop_logits: torch.Tensor,
        next_embed: torch.Tensor,
        pred_feat: torch.Tensor,
    ) -> None:
        state.precomputed_stop_logits = stop_logits
        state.precomputed_is_stopping = None
        state.curr_embed_for_next = next_embed.detach()
        state.prev_feat_embed = next_embed.detach()
        state.curr_prefix_feat_cond = pred_feat[0].detach()
        state.last_audio_patch_gpu = pred_feat.detach()

    # -------------------- audio collection --------------------

    def _uses_sparse_audio_outputs(self) -> bool:
        return (
            self._audio_emit_every > 1 or self._enable_delayed_audio_copy or getattr(self, "_vae_decode_every", 1) > 1
        )

    def _vae_output_storage_may_be_reused(self) -> bool:
        return self._enable_vae_cuda_graph or bool(getattr(self.tts.audio_vae, "_compiled", False))

    def _enqueue_delayed_audio_copy(self, state: _RequestState, audio: torch.Tensor) -> None:
        src = audio.detach().contiguous()
        if src.device.type != current_omni_platform.device_type:
            state.pending_audio_copies.append(_PendingAudioCopy(host=src.cpu().contiguous()))
            return

        # Compiled/graph VAE output storage can be reused on the next token.
        if self._vae_output_storage_may_be_reused():
            src = src.clone()
        host = torch.empty(src.shape, dtype=src.dtype, device="cpu", pin_memory=True)
        if self._audio_copy_stream is None:
            self._audio_copy_stream = torch.cuda.Stream(device=src.device)
        copy_stream = self._audio_copy_stream
        copy_stream.wait_stream(torch.cuda.current_stream(src.device))
        with torch.cuda.stream(copy_stream):
            host.copy_(src, non_blocking=True)
            src.record_stream(copy_stream)
            event = None
            if self._delayed_audio_copy_use_events:
                event = torch.cuda.Event()
                event.record(copy_stream)
        state.pending_audio_copies.append(_PendingAudioCopy(host=host, event=event, source=src, async_copy=True))

    def _pending_audio_copy_ready(self, copy: _PendingAudioCopy) -> bool:
        if copy.event is not None:
            return copy.event.query()
        if not copy.async_copy:
            return True
        return self._audio_copy_stream is not None and self._audio_copy_stream.query()

    def _drain_ready_audio_copies_for_states(self, states: list[_RequestState]) -> dict[str, list[torch.Tensor]]:
        if (
            not self._enable_delayed_audio_copy
            or self._audio_emit_every != 1
            or self._delayed_audio_copy_use_events
            or self._audio_copy_stream is None
            or not any(state.pending_audio_copies for state in states)
            or not self._audio_copy_stream.query()
        ):
            return {}

        ready_by_req: dict[str, list[torch.Tensor]] = {}
        for state in states:
            ready = self._drain_pending_audio_copies(state, force=False)
            if ready:
                ready_by_req[state.request_id] = ready
        return ready_by_req

    def _drain_pending_audio_copies(self, state: _RequestState, *, force: bool) -> list[torch.Tensor]:
        ready: list[torch.Tensor] = []
        if (
            force
            and not self._delayed_audio_copy_use_events
            and state.pending_audio_copies
            and any(copy.async_copy for copy in state.pending_audio_copies)
            and self._audio_copy_stream is not None
            and not self._audio_copy_stream.query()
        ):
            self._audio_copy_stream.synchronize()
        while state.pending_audio_copies:
            pending = state.pending_audio_copies[0]
            if pending.event is not None and force:
                pending.event.synchronize()
            elif not self._pending_audio_copy_ready(pending):
                break
            state.pending_audio_copies.pop(0)
            ready.append(pending.host.float())
        return ready

    @staticmethod
    def _merge_audio_chunks(chunks: list[torch.Tensor]) -> torch.Tensor | None:
        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0]
        return torch.cat([chunk.reshape(-1) for chunk in chunks], dim=0)

    def _precompute_stop_flags_for_audio_collect(self, states: list[_RequestState]) -> None:
        if (
            self._audio_emit_every == 1
            and not self._enable_delayed_audio_copy
            and getattr(self, "_vae_decode_every", 1) == 1
        ):
            return

        pending: list[tuple[_RequestState, torch.Tensor]] = []
        for state in states:
            if state.is_stopping or state.precomputed_is_stopping is not None:
                continue
            stop_logits = state.precomputed_stop_logits
            if stop_logits is None or stop_logits.device.type != current_omni_platform.device_type:
                continue
            pending.append((state, stop_logits))
        if not pending:
            return

        stacked = torch.stack([stop_logits[0] for _, stop_logits in pending], dim=0)
        stop_mask = stacked[:, 1] > stacked[:, 0]
        stop_mask_cpu = stop_mask.cpu()
        for i, (state, _) in enumerate(pending):
            is_stopping = bool(stop_mask_cpu[i])
            state.precomputed_is_stopping = is_stopping
            if is_stopping:
                state.is_stopping = True

    @staticmethod
    def _should_stop_from_cached_logits(state: _RequestState) -> bool:
        if state.is_stopping:
            return True
        cached = state.precomputed_is_stopping
        if cached is not None:
            return cached
        stop_logits = state.precomputed_stop_logits
        if stop_logits is None:
            return False
        is_stopping = bool(stop_logits[0, 1] > stop_logits[0, 0])
        state.precomputed_is_stopping = is_stopping
        if is_stopping:
            state.is_stopping = True
        return is_stopping

    def _collect_audio(
        self, state: _RequestState, initial_delayed_chunks: list[torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """Per-step sliding-window VAE decode (nanovllm pattern).

        Each decode step feeds ``[decode_pad, new_patch]`` through the VAE
        and slices out only the audio region corresponding to the new patch.
        The pad buffer (last ``_n_decode_pad_frames`` latent frames) provides
        the receptive-field context needed by the VAE's transposed convolutions,
        eliminating boundary artifacts between chunks.

        Returns the delta audio chunk (not cumulative) so the output processor
        can stream each chunk to the client independently.
        """
        delayed_chunks: list[torch.Tensor] = list(initial_delayed_chunks or [])
        if self._enable_delayed_audio_copy and self._audio_emit_every == 1:
            if initial_delayed_chunks is None:
                delayed_chunks.extend(self._drain_pending_audio_copies(state, force=False))

        patch = state.last_audio_patch_gpu
        if patch is None:
            return self._merge_audio_chunks(delayed_chunks)
        state.last_audio_patch_gpu = None

        # patch shape: (patch_size, feat_dim) or (1, patch_size, feat_dim)
        new_latent = patch.reshape(-1, self._feat_dim).to(torch.float32)
        vae_decode_every = getattr(self, "_vae_decode_every", 1)
        if vae_decode_every > 1:
            is_stopping = self._should_stop_from_cached_logits(state)
            state.pending_vae_latents_gpu.append(new_latent.detach())
            if not is_stopping and len(state.pending_vae_latents_gpu) < vae_decode_every:
                return self._merge_audio_chunks(delayed_chunks)
            new_latent = torch.cat(state.pending_vae_latents_gpu, dim=0)
            state.pending_vae_latents_gpu.clear()

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
        with _NvtxRange("voxcpm2.vae_prepare"):
            feat = vae_input.unsqueeze(0).transpose(1, 2).contiguous()
        vae_frames = vae_input.shape[0]
        with _NvtxRange("voxcpm2.vae_decode"):
            with torch.no_grad():
                audio = self._run_vae_decode(feat.to(self._device)).reshape(-1)

        # Slice out only the new audio (after the pad region).
        # Each latent frame maps to decoder_chunk_size audio samples.
        with _NvtxRange("voxcpm2.vae_slice"):
            dcs = int(getattr(self.tts.audio_vae, "decode_chunk_size", audio.numel() // vae_frames))
            new_audio = audio[pad_frames * dcs : (pad_frames + n_new) * dcs]

        # Roll the pad buffer: keep last N latent frames as context for next step.
        with _NvtxRange("voxcpm2.vae_pad_update"):
            all_latents = vae_input  # [pad + new]
            state.decode_pad = all_latents[-self._n_decode_pad_frames :].detach()

        self._perf.stop("vae_decode")
        if self._enable_delayed_audio_copy and self._audio_emit_every == 1:
            is_stopping = self._should_stop_from_cached_logits(state)
            self._enqueue_delayed_audio_copy(state, new_audio)
            if is_stopping:
                delayed_chunks.extend(self._drain_pending_audio_copies(state, force=True))
            return self._merge_audio_chunks(delayed_chunks)

        if self._audio_emit_every > 1:
            is_stopping = self._should_stop_from_cached_logits(state)
            audio_chunk = new_audio.detach()
            if self._vae_output_storage_may_be_reused():
                audio_chunk = audio_chunk.clone()
            state.pending_audio_chunks_gpu.append(audio_chunk)
            if not is_stopping and len(state.pending_audio_chunks_gpu) < self._audio_emit_every:
                return None
            if len(state.pending_audio_chunks_gpu) == 1:
                merged_audio = state.pending_audio_chunks_gpu[0]
            else:
                merged_audio = torch.cat([chunk.reshape(-1) for chunk in state.pending_audio_chunks_gpu], dim=0)
            state.pending_audio_chunks_gpu.clear()
            return merged_audio.detach().cpu().float()
        if self._coalesce_audio_d2h:
            audio_chunk = new_audio.detach()
            if self._vae_output_storage_may_be_reused():
                audio_chunk = audio_chunk.clone()
            return audio_chunk
        return new_audio.detach().cpu().float()

    def _can_collect_audio_batch(
        self,
        states: list[_RequestState],
        initial_delayed_chunks_by_req: dict[str, list[torch.Tensor] | None] | None,
    ) -> bool:
        if not getattr(self, "_enable_batched_vae_decode", False):
            return False
        if self._enable_delayed_audio_copy or self._audio_emit_every > 1 or self._enable_vae_cuda_graph:
            return False
        if initial_delayed_chunks_by_req and any(initial_delayed_chunks_by_req.values()):
            return False
        return len(states) > 1

    def _collect_audio_batch(
        self,
        states: list[_RequestState],
        initial_delayed_chunks_by_req: dict[str, list[torch.Tensor] | None] | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if not self._can_collect_audio_batch(states, initial_delayed_chunks_by_req):
            return {
                state.request_id: self._collect_audio(
                    state,
                    initial_delayed_chunks=None
                    if initial_delayed_chunks_by_req is None
                    else initial_delayed_chunks_by_req.get(state.request_id),
                )
                for state in states
            }

        outputs: dict[str, torch.Tensor | None] = {state.request_id: None for state in states}
        pending_by_shape: dict[
            tuple[torch.device, torch.dtype, int, int],
            list[tuple[_RequestState, torch.Tensor, torch.Tensor, int, int]],
        ] = {}
        vae_decode_every = getattr(self, "_vae_decode_every", 1)

        for state in states:
            patch = state.last_audio_patch_gpu
            if patch is None:
                continue
            state.last_audio_patch_gpu = None

            new_latent = patch.reshape(-1, self._feat_dim).to(torch.float32)
            if vae_decode_every > 1:
                is_stopping = self._should_stop_from_cached_logits(state)
                state.pending_vae_latents_gpu.append(new_latent.detach())
                if not is_stopping and len(state.pending_vae_latents_gpu) < vae_decode_every:
                    outputs[state.request_id] = None
                    continue
                new_latent = torch.cat(state.pending_vae_latents_gpu, dim=0)
                state.pending_vae_latents_gpu.clear()

            n_new = new_latent.shape[0]
            if state.decode_pad is not None:
                vae_input = torch.cat([state.decode_pad, new_latent], dim=0)
                pad_frames = state.decode_pad.shape[0]
            else:
                vae_input = new_latent
                pad_frames = 0

            with _NvtxRange("voxcpm2.vae_prepare"):
                feat = vae_input.unsqueeze(0).transpose(1, 2).contiguous()
            key = (feat.device, feat.dtype, feat.shape[1], feat.shape[2])
            pending_by_shape.setdefault(key, []).append((state, feat, vae_input, pad_frames, n_new))

        for group in pending_by_shape.values():
            with _NvtxRange("voxcpm2.vae_decode"):
                with torch.no_grad():
                    feat_batch = torch.cat([feat for _, feat, _, _, _ in group], dim=0).to(self._device)
                    audio_batch = self._run_vae_decode(feat_batch)

            for i, (state, _feat, vae_input, pad_frames, n_new) in enumerate(group):
                audio = audio_batch[i].reshape(-1)
                vae_frames = vae_input.shape[0]
                with _NvtxRange("voxcpm2.vae_slice"):
                    dcs = int(getattr(self.tts.audio_vae, "decode_chunk_size", audio.numel() // vae_frames))
                    new_audio = audio[pad_frames * dcs : (pad_frames + n_new) * dcs]

                with _NvtxRange("voxcpm2.vae_pad_update"):
                    state.decode_pad = vae_input[-self._n_decode_pad_frames :].detach()

                if self._coalesce_audio_d2h:
                    audio_chunk = new_audio.detach()
                    if self._vae_output_storage_may_be_reused():
                        audio_chunk = audio_chunk.clone()
                    outputs[state.request_id] = audio_chunk
                else:
                    outputs[state.request_id] = new_audio.detach().cpu().float()

        return outputs

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
                        state.precomputed_is_stopping = None
                    else:
                        logits[i, 0] = stop_logits[0, 0]
                        logits[i, 1] = stop_logits[0, 1]
                        if state is not None:
                            if state.precomputed_is_stopping is not None:
                                state.is_stopping = state.precomputed_is_stopping
                            state.precomputed_stop_logits = None
                            state.precomputed_is_stopping = None
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
            audio_by_req: dict[str, torch.Tensor] = {}
            for req_id, audio in self._audio_queue:
                if audio is None:
                    continue
                if req_id in audio_by_req:
                    audio_by_req[req_id] = torch.cat([audio_by_req[req_id].reshape(-1), audio.reshape(-1)], dim=0)
                else:
                    audio_by_req[req_id] = audio
            if audio_by_req:
                sr = torch.tensor(self._sample_rate, dtype=torch.int32)
                if self._uses_sparse_audio_outputs():
                    ready_req_ids = list(audio_by_req)
                    chunks = [audio_by_req[req_id].reshape(-1) for req_id in ready_req_ids]
                    if self._coalesce_audio_d2h and any(
                        chunk.device.type == current_omni_platform.device_type for chunk in chunks
                    ):
                        sizes = [int(chunk.numel()) for chunk in chunks]
                        merged = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
                        merged_cpu = merged.detach().cpu().contiguous()
                        mm["model_outputs"] = list(merged_cpu.split(sizes))
                    else:
                        mm["model_outputs"] = chunks
                    mm["sr"] = [sr for _ in ready_req_ids]
                    mm["meta"] = {"req_id": ready_req_ids, "sparse_audio": ["1"]}
                elif self._coalesce_audio_d2h and any(
                    audio.device.type == current_omni_platform.device_type for audio in audio_by_req.values()
                ):
                    ready_req_ids = list(audio_by_req)
                    chunks = [audio_by_req[req_id].reshape(-1) for req_id in ready_req_ids]
                    sizes = [int(chunk.numel()) for chunk in chunks]
                    merged = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
                    merged_cpu = merged.detach().cpu().float()
                    mm["model_outputs"] = list(merged_cpu.split(sizes))
                    mm["sr"] = [sr for _ in ready_req_ids]
                else:
                    mm["model_outputs"] = list(audio_by_req.values())
                    mm["sr"] = [sr for _ in audio_by_req]
            elif self._uses_sparse_audio_outputs():
                mm["model_outputs"] = []
                mm["sr"] = []
                mm["meta"] = {"req_id": [], "sparse_audio": ["1"]}
            self._audio_queue.clear()

        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=mm)

    # -------------------- Chinese token splitting --------------------

    def _get_multichar_zh_split(self) -> dict[int, list[int]]:
        """Lazy-build {multichar_chinese_token_id: [char_id, ...]} map."""
        if self._multichar_zh_split is not None:
            return self._multichar_zh_split
        base_tokenizer = self.tts.text_tokenizer.tokenizer
        self._multichar_zh_split = build_cjk_split_map(base_tokenizer)
        logger.info("VoxCPM2: built multichar Chinese split map (%d entries)", len(self._multichar_zh_split))
        return self._multichar_zh_split

    # -------------------- preprocess / postprocess --------------------

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Unpack[VoxCPM2PreprocessInput],
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
            # Do not evict state here: _pending_requests is a per-step prefix,
            # not the full batch. Cleanup is driven by on_requests_finished ->
            # _flush_deferred_cleanup (fed by vLLM scheduler._free_request via
            # gpu_ar_model_runner.py).
            real = info_dict.get("text_token_ids")
            token_ids = input_ids.tolist() if real is None else real[0]
            # Fail-fast: unsplit multichar Chinese IDs in input_ids means the
            # serving layer didn't pre-split.  Silent fixup here would cause
            # input_ids/embeds length mismatch (scheduler slot count is fixed).
            split_map = self._get_multichar_zh_split()
            if split_map and any(tid in split_map for tid in token_ids):
                raise ValueError(
                    "VoxCPM2 preprocess received unsplit multichar Chinese "
                    "token IDs. The serving layer must send prompt_token_ids "
                    "with single-char CJK IDs (see _voxcpm2_encode)."
                )
            if token_ids and token_ids[0] == self.config.bos_token_id:
                token_ids = token_ids[1:]

            state = self._get_or_create_state(req_id)
            state.decode_pad = None
            state.prefill_completed = False
            state.decode_step_count = 0
            state.precomputed_stop_logits = None
            state.precomputed_is_stopping = None
            state.last_audio_patch_gpu = None
            if not hasattr(state, "pending_audio_chunks_gpu"):
                state.pending_audio_chunks_gpu = []
            if not hasattr(state, "pending_audio_copies"):
                state.pending_audio_copies = []
            state.pending_audio_chunks_gpu.clear()
            state.pending_audio_copies.clear()
            state.curr_embed_for_next = None
            state.prev_feat_embed = None
            state.curr_prefix_feat_cond = None
            state.is_stopping = False

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
            voice_profile = info_dict.get("voice_profile")
            if isinstance(voice_profile, list):
                voice_profile = voice_profile[0] if voice_profile else None
            requires_precomputed_cache = isinstance(voice_profile, dict)

            state.prompt_cache = None
            voice_name = info_dict.get("voice_name")
            if isinstance(voice_name, list):
                voice_name = voice_name[0] if voice_name else None
            _created_at = int(info_dict.get("voice_created_at") or 0)

            if voice_name:
                _cache_key = self._speaker_cache.make_cache_key(
                    voice_name, model_type="voxcpm2", created_at=_created_at
                )
                cached = self._speaker_cache.get(_cache_key)
                if cached is not None:
                    if "mode" in cached:
                        state.prompt_cache = self._clone_prompt_cache(cached)
                    elif "ref_audio_feat" in cached:
                        state.prompt_cache = {
                            "mode": "reference",
                            "ref_audio_feat": cached["ref_audio_feat"].clone(),
                        }
                    logger.debug("Speaker cache HIT for VoxCPM2 speaker '%s'", voice_name)

            if state.prompt_cache is None and requires_precomputed_cache:
                requested_mode = voice_profile.get("mode") if isinstance(voice_profile, dict) else None
                raise ValueError(
                    f"Precomputed VoxCPM2 voice '{voice_name}' was accepted by serving but is not loaded "
                    f"in the model cache (voice_created_at={_created_at}, mode={requested_mode})"
                )

            if state.prompt_cache is None and (ref_audio or (prompt_audio and prompt_text)):
                state.prompt_cache = self._build_prompt_cache(
                    ref_audio=ref_audio,
                    prompt_audio=prompt_audio,
                    prompt_text=prompt_text,
                )
                if (
                    voice_name
                    and state.prompt_cache is not None
                    and state.prompt_cache.get("mode") == "reference"
                    and "ref_audio_feat" in state.prompt_cache
                ):
                    _key = self._speaker_cache.make_cache_key(voice_name, model_type="voxcpm2", created_at=_created_at)
                    self._speaker_cache.put(_key, {"ref_audio_feat": state.prompt_cache["ref_audio_feat"].cpu()})
                    logger.debug("Speaker cache STORE for VoxCPM2 speaker '%s'", voice_name)

            inputs = self._build_prefill_inputs(token_ids, dev, req_id)
            tts = self.tts
            feat_embed = tts.enc_to_lm_proj(tts.feat_encoder(inputs.audio_feat))
            text_embed = self.model.embed_input_ids(inputs.text_token.to(dev))
            text_mask, feat_mask = inputs.text_mask, inputs.audio_mask
            embeds = (text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed).squeeze(0)
            state.prefill_masks = (text_mask, feat_mask, inputs.audio_feat, feat_embed)
        else:
            state = self._active_states.get(req_id)
            curr = state.curr_embed_for_next if state else None
            if curr is not None:
                embeds = curr.to(dev, dtype=self._side_dtype).reshape(1, -1)
            else:
                embeds = torch.zeros(1, self.config.hidden_size, device=dev, dtype=self._side_dtype)

        self._pending_requests.append((req_id, is_prefill, embeds, span_len))
        return input_ids, embeds, {}

    def postprocess(self, hidden_states: torch.Tensor, **info: Unpack[VoxCPM2PostprocessInput]) -> dict[str, Any]:
        req_id = info.get("request_id", self._current_request_id or "default")
        if self._enable_profiling:
            state = self._active_states.get(req_id)
            if state and state.decode_step_count > 0:
                logger.info(
                    "REQUEST DONE[%s]: %d steps, %.2fs\n%s\nUnified graph: captures=%d replays=%d skips=%s",
                    req_id,
                    state.decode_step_count,
                    time.perf_counter() - state.request_start_time,
                    self._perf.breakdown(),
                    self._unified_graph_stats.captures,
                    self._unified_graph_stats.replays,
                    self._unified_graph_stats.skips,
                )
        return {}

    # -------------------- build prefill inputs --------------------

    def _build_prefill_inputs(
        self,
        token_ids: list[int],
        dev: torch.device,
        req_id: str = "default",
    ) -> _PrefillInputs:
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

        return _PrefillInputs(
            text_token=text_token.unsqueeze(0).to(dev),
            audio_feat=audio_feat.unsqueeze(0).to(dev).to(dtype),
            text_mask=text_mask.unsqueeze(0).to(dev),
            audio_mask=audio_mask.unsqueeze(0).to(dev),
        )

    # -------------------- weight loading --------------------

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"base_lm.": "model."})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def _base_lm_only(ws):
            for name, tensor in ws:
                if name.startswith("base_lm."):
                    yield name, tensor

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(_base_lm_only(weights), mapper=self.hf_to_vllm_mapper)

        # _tts and residual_model are constructed and populated eagerly in
        # __init__ via VoxCPM.from_pretrained; here we only need to mark their
        # params as loaded so AutoWeightsLoader's strict check doesn't flag
        # them as missing from the checkpoint.
        loaded |= {name for name, _ in self.named_parameters() if name.startswith(("_tts.", "residual_model."))}

        logger.info(
            "Loaded VoxCPM2 (patch=%d, feat_dim=%d, dtype=%s)", self._patch_size, self._feat_dim, self._side_dtype
        )
        return loaded
