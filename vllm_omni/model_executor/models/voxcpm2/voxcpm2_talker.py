# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
import os
import threading
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from types import MethodType
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypedDict

import numpy as np
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
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

import vllm_omni.platforms as omni_platform
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.speaker_cache import (
    get_speaker_cache,
    iter_custom_voice_profiles,
    load_validated_profile_tensors,
    validate_voxcpm2_profile,
)
from vllm_omni.worker.runner_assisted_metadata import RunnerAssistedFullAttentionMetadataRequest

from .lora import merge_voxcpm2_lora
from .minicpm4_paged import MiniCPM4PagedForVoxCPM2, MiniCPM4PagedResidualLM
from .runtime_config import _VoxCPM2RuntimeConfig
from .voxcpm2_import_utils import import_voxcpm2_core

if TYPE_CHECKING:
    from .model_state import VoxCPM2BatchSlots, VoxCPM2RequestState

logger = init_logger(__name__)


# RLock: _disable_npu_internal_format_once acquires this same lock, and the
# first VAE decode calls it from inside _vae_npu_graph_runner_for's critical
# section. threading.Lock is not reentrant and deadlocks that path.
_VAE_NPU_GRAPH_RUNNER_LOCK = threading.RLock()
_NPU_INTERNAL_FORMAT_MUTATED = False


def _disable_npu_internal_format_once() -> None:
    """One-shot process-global CANN requirement for NPUGraph capture.

    ``allow_internal_format=False`` must hold process-wide before any
    NPUGraph capture of the VAE's Conv2D ops; it is idempotent, so several
    AudioVAE instances each calling this are harmless. CANN provides no
    per-module scope for this switch, which is why the mutation lives in a
    single guarded helper instead of being scattered across call sites.
    """
    global _NPU_INTERNAL_FORMAT_MUTATED
    if _NPU_INTERNAL_FORMAT_MUTATED:
        return
    with _VAE_NPU_GRAPH_RUNNER_LOCK:
        if not _NPU_INTERNAL_FORMAT_MUTATED:
            torch.npu.config.allow_internal_format = False
            _NPU_INTERNAL_FORMAT_MUTATED = True
            logger.info(
                "Set torch.npu.config.allow_internal_format=False (process-wide; "
                "required before AudioVAE NPUGraph capture and visible to every "
                "module in this stage process)"
            )


def _vae_npu_graph_runner_for(audio_vae: torch.nn.Module):
    """Return this AudioVAE instance's own NPUGraph runner, lazily created.

    The runner is attached to the AudioVAE instance and never shared between
    instances: each instance captures its own weights into its own dedicated
    graph pool, so two models with identical input signatures can never
    replay each other's captured weights. ``None`` means not yet initialized;
    ``False`` means NPU graphs are disabled for this instance (unsupported
    platform or the operator escape hatch below).
    """
    runner = getattr(audio_vae, "_npu_graph_runner", None)
    if runner is not None:
        return runner
    with _VAE_NPU_GRAPH_RUNNER_LOCK:
        runner = getattr(audio_vae, "_npu_graph_runner", None)
        if runner is not None:
            return runner
        # Escape hatch: after a failed capture the runner refuses further
        # in-process replay, so operators disable the AudioVAE NPUGraph by
        # restarting the stage process with VLLM_OMNI_DISABLE_VAE_NPU_GRAPH=1.
        # Accepts the usual truthy spellings, matching the patch-level
        # VLLM_OMNI_DISABLE_VOXCPM2_NPU_PATCH switches (see #8313).
        if os.environ.get("VLLM_OMNI_DISABLE_VAE_NPU_GRAPH", "0").strip().lower() in {"1", "true", "yes", "on"}:
            logger.info("AudioVAE NPUGraph disabled via VLLM_OMNI_DISABLE_VAE_NPU_GRAPH")
            audio_vae._npu_graph_runner = False
            return False
        # Only construct the runner on NPU platforms; CUDA runtimes without
        # torch.npu would raise AttributeError on the config access below.
        # Module-attribute access (not from-import): this helper can run while
        # the platforms package is still initializing, where from-imports of
        # the lazily-resolved platform singleton fail.
        if not omni_platform.current_omni_platform.is_npu():
            audio_vae._npu_graph_runner = False
            return False
        from vllm_omni.platforms.npu.graph_tools import NPUExactGraphRunner

        _disable_npu_internal_format_once()
        # Dedicated pool per instance: isolate this component's graphs from
        # other components and other model instances sharing the global pool.
        # Prevents cross-graph memory reuse corruption when many graphs are
        # cached.
        runner = NPUExactGraphRunner(
            # Production serving sees 13-17 distinct AudioVAE decode shapes per
            # instance-day (streaming chunks plus tails), so 8 entries would
            # push most traffic to eager. 32 = the runner default, already
            # exercised end-to-end by daily NPU serving; overflow shapes still
            # fall back to eager decode with a warning.
            max_graphs=32,
            component_name="VoxCPM2 AudioVAE",
            disable_config_hint="disable the AudioVAE NPUGraph (set VLLM_OMNI_DISABLE_VAE_NPU_GRAPH=1)",
            use_shared_pool=False,
        )
        if not runner.is_supported():
            logger.warning("AudioVAE NPUGraph not supported; using eager execution")
            runner = False
        audio_vae._npu_graph_runner = runner
        return runner


def _install_dtype_aware_encode(tts: nn.Module, vae_dtype: torch.dtype) -> None:
    """Wrap ``audio_vae.encode`` so every caller is dtype-safe under a bf16 VAE.

    Inputs are cast to ``vae_dtype`` (prevents fp32/bf16 mismatch errors on
    Ascend NPU for callers like ``_encode_raw_audio`` and the native
    ``_encode_wav`` / ``build_prompt_cache``). The result is cast back to
    float32 to preserve the historical contract of the cached prompt
    features: ``_build_prefill_inputs`` concatenates them with fp32 zero
    padding (voice-clone / continuation / ICL prefill). ``torch.cat`` would
    silently type-promote a bf16 feature to float32 on torch 2.x, so without
    this cast the cache would drift to bf16-precision values with no error
    signal; the explicit float32 restore keeps the pre-PR cache precision.
    """
    original_encode = tts.audio_vae.encode

    def _dtype_aware_encode(audio_data, sample_rate, *args, **kwargs):
        audio_data = audio_data.to(dtype=vae_dtype)
        result = original_encode(audio_data, sample_rate, *args, **kwargs)
        return result.float()

    tts.audio_vae.encode = _dtype_aware_encode


def _remove_weight_norm_from_module(module: nn.Module) -> None:
    """Recursively remove weight_norm from all submodules.

    This materializes the final weight (weight_g * weight_v / ||weight_v||)
    so it is computed once at load time instead of on every forward pass,
    avoiding fp32 upcast from the norm computation.
    """
    removed = 0
    for m in module.modules():
        if hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            try:
                nn.utils.remove_weight_norm(m)
                removed += 1
            except (ValueError, RuntimeError) as exc:
                # A failed materialization must not be silently skipped: the
                # module would keep the per-forward norm path (the aten::to
                # upcast this PR removes), and WeightNorm.remove can leave a
                # module half-broken (weight deleted, hook still attached).
                # Fail loudly at load time with the offending module named.
                logger.error(
                    "weight_norm removal failed for %s in audio_vae: %s",
                    type(m).__name__,
                    exc,
                )
                raise
    if removed:
        logger.info("Removed weight_norm from %d modules in audio_vae", removed)


class _VoxCPM2PromptConfigLike(Protocol):
    audio_vae_config: dict[str, Any]
    patch_size: int


class _AttentionMetadataLike(Protocol):
    scheduler_metadata: Any


class _ForwardContextLike(Protocol):
    attn_metadata: dict[str, _AttentionMetadataLike] | Any


class VoxCPM2PreprocessInput(TypedDict, total=False):
    # Shared input data.
    additional_information: dict[str, Any]
    _omni_seed: int | None
    _omni_num_computed_tokens: int
    _omni_is_prefill: bool
    text_token_ids: list[list[int]]
    reference_audio: object
    ref_audio: object
    prompt_audio: object
    prompt_text: str | list[str] | None
    voice_profile: dict[str, Any] | list[dict[str, Any]] | None
    voice_name: str | list[str] | None
    voice_created_at: int | str | None

    # V2 runner identity, slot ownership, batch position, and prepared prefill data.
    req_id: str
    slot_index: int
    batch_row: int
    prefill_text_embed: torch.Tensor
    prepared_prefill: _PreparedPrefill


class _PrefillInputs(NamedTuple):
    text_token: torch.Tensor
    audio_feat: torch.Tensor
    text_mask: torch.Tensor
    audio_mask: torch.Tensor


class _PreparedPrefill(NamedTuple):
    """V2 batched voice prefill data reused by per-request preprocess."""

    prompt_cache: dict[str, Any]
    embeds: torch.Tensor
    text_mask: torch.Tensor
    audio_mask: torch.Tensor
    audio_feat: torch.Tensor
    feat_embed: torch.Tensor


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
    ref_audio: Sequence[float] | torch.Tensor | bytes | None = None,
    ref_sr: int | None = None,
    ref_text: str | None = None,
    voice_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a VoxCPM2 prefill prompt whose ``prompt_token_ids`` length matches
    the talker-side prefill length.

    Used by both online serving (``VoxCPM2Adapter._build_prompt``) and
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
        if isinstance(ref_audio, bytes) and len(ref_audio) % np.dtype(np.float32).itemsize:
            raise ValueError("VoxCPM2 ref_audio bytes must contain float32 samples")
        num_samples = len(ref_audio) // 4 if isinstance(ref_audio, bytes) else len(ref_audio)
        ref_len = math.ceil(math.ceil(num_samples * vae["sample_rate"] / ref_sr) / patch_samples)
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


@lru_cache(maxsize=32)
def _audio_resampler_on_device(orig_sr: int, target_sr: int, device: torch.device) -> nn.Module:
    import torchaudio

    return torchaudio.transforms.Resample(orig_sr, target_sr).to(device)


def _encode_raw_audio(
    tts: nn.Module,
    samples: list[float] | torch.Tensor | bytes,
    sr: int,
    padding_mode: str = "right",
) -> torch.Tensor:
    """Encode raw audio samples using the native VoxCPM2 AudioVAE.

    Mirrors ``VoxCPM2Model._encode_wav`` but accepts in-memory samples
    instead of a file path (needed for the OpenAI speech API).
    """
    if isinstance(samples, bytes):
        if len(samples) % np.dtype(np.float32).itemsize:
            raise ValueError("VoxCPM2 audio bytes must contain float32 samples")
        audio = torch.from_numpy(np.frombuffer(samples, dtype=np.float32).copy())
    elif isinstance(samples, list):
        audio = torch.from_numpy(np.asarray(samples, dtype=np.float32))
    else:
        audio = samples.float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    vae_device = next(tts.audio_vae.parameters()).device
    encode_sr = tts._encode_sample_rate
    if audio.device.type == "cpu" and vae_device.type == "cuda":
        audio = audio.pin_memory()
    audio = audio.to(vae_device, non_blocking=True)
    if sr != encode_sr:
        audio = _audio_resampler_on_device(sr, encode_sr, vae_device)(audio)

    patch_len = tts.patch_size * tts.chunk_size
    if audio.size(1) % patch_len != 0:
        padding_size = patch_len - audio.size(1) % patch_len
        pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
        audio = torch.nn.functional.pad(audio, pad)

    vae_dtype = next(tts.audio_vae.parameters()).dtype
    feat = tts.audio_vae.encode(audio.to(device=vae_device, dtype=vae_dtype), encode_sr)
    return feat.view(tts.audio_vae.latent_dim, -1, tts.patch_size).permute(1, 2, 0)


def patch_is_valid(state, is_prefill: bool) -> bool:
    # Intermediate prefill chunks still write LM KV, but cannot commit a patch.
    # A newly stopping step remains valid; only previously stopped rows are masked.
    return not state.is_stopping and (not is_prefill or state.is_last_prefill_chunk)


def mixed_request_is_ready(state, is_prefill: bool, per_patch_audio: bool) -> bool:
    if is_prefill:
        return per_patch_audio or state.is_last_prefill_chunk
    return state.decode_state_ready


def finish_graph_prefills(pending_requests) -> None:
    """Release prefill inputs only after the final chunk produced its patch."""
    for state, is_prefill, _, _ in pending_requests:
        if is_prefill and state.is_last_prefill_chunk:
            state.decode_step_count = 0
            state.prefill_completed = True
            state.prefill_embeds = None
            state.prefill_masks = None
            state.prompt_cache = None


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
    text_mask: torch.Tensor | None = None
    last_token_indices: torch.Tensor | None = None
    patch_indices: torch.Tensor | None = None
    scaffold_hidden: torch.Tensor | None = None
    pcm: torch.Tensor | None = None
    pcm_valid: torch.Tensor | None = None
    stop_logits: torch.Tensor | None = None
    audio_slots: torch.Tensor | None = None
    audio_rows_valid: torch.Tensor | None = None
    audio_copy_event: torch.cuda.Event | None = None


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
) -> torch.Tensor:
    estimator = cfm_module.estimator
    mean_mode = getattr(cfm_module, "mean_mode", False)
    b = mu.size(0)

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

    buffers.mu_in[:b].copy_(mu)
    buffers.cond_in[:b].copy_(cond[:b])
    if cfg_cutoff_step >= start_step:
        if buffers._zeroed_mu_negative_batch != b:
            buffers.mu_in[b : 2 * b].zero_()
            buffers._zeroed_mu_negative_batch = b
        buffers.cond_in[b : 2 * b].copy_(cond[:b])

    for step in range(start_step, len(t_span)):
        if step <= cfg_cutoff_step:
            buffers.x_in[:b].copy_(x)
            buffers.x_in[b : 2 * b].copy_(x)
            buffers.t_in[: 2 * b].copy_(t)
            if mean_mode:
                buffers.dt_in[: 2 * b].copy_(dt)

            raw_out = estimator(
                buffers.x_in[: 2 * b],
                buffers.mu_in[: 2 * b],
                buffers.t_in[: 2 * b],
                buffers.cond_in[: 2 * b],
                buffers.dt_in[: 2 * b],
            )

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
            buffers.x_in[:b].copy_(x)
            buffers.t_in[:b].copy_(t)
            if mean_mode:
                buffers.dt_in[:b].copy_(dt)
            dphi_dt = estimator(
                buffers.x_in[:b],
                buffers.mu_in[:b],
                buffers.t_in[:b],
                buffers.cond_in[:b],
                buffers.dt_in[:b],
            )

        x.sub_(dt * dphi_dt)
        t = t - dt
        if step < len(t_span) - 1:
            dt = t - t_span[step + 1]
    return x.clone()


# ===================================================================
#  Main talker model
# ===================================================================


def _resolve_cfg_value(config_value: float) -> float:
    """cfg_value precedence: env (VLLM_OMNI_SPEECH_CFG_VALUE) > the deploy
    yaml's ``voxcpm2_runtime_config.cfg_value`` > the dataclass default.

    An unset or empty env value falls back to the config value; a non-numeric
    env value names the variable in the error; the resolved value must be a
    positive finite float, and the error names the source it came from.
    """
    cfg_from_env = os.environ.get("VLLM_OMNI_SPEECH_CFG_VALUE")
    if cfg_from_env and cfg_from_env.strip():
        try:
            value = float(cfg_from_env)
        except ValueError as exc:
            raise ValueError(f"VLLM_OMNI_SPEECH_CFG_VALUE={cfg_from_env!r} is not a valid float") from exc
        source = f"VLLM_OMNI_SPEECH_CFG_VALUE={cfg_from_env!r}"
    else:
        # yaml booleans are rejected upstream in _coerce_value, so this
        # helper only ever sees the already-coerced config value.
        try:
            value = float(config_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"voxcpm2_runtime_config.cfg_value={config_value!r} is not a valid float") from exc
        source = f"voxcpm2_runtime_config.cfg_value={value!r}"
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{source} must be a positive finite float (env VLLM_OMNI_SPEECH_CFG_VALUE overrides the deploy yaml)"
        )
    return value


class VoxCPM2TalkerForConditionalGeneration(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self._runtime_config = _VoxCPM2RuntimeConfig.from_vllm_config(vllm_config)
        self._startup_lora_applied = False
        if self._runtime_config.startup_lora_path and vllm_config.load_config.load_format == "dummy":
            raise ValueError("VoxCPM2 startup LoRA requires real base weights; load_format=dummy is unsupported")

        self.have_multimodal_outputs = True
        self.has_preprocess = True

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
        self._device = omni_platform.current_omni_platform.get_torch_device()
        VoxCPM = import_voxcpm2_core()
        native = VoxCPM.from_pretrained(model_path, load_denoiser=False, optimize=False)
        self._tts: nn.Module = native.tts_model.to(self._device)

        # Remove weight_norm from audio_vae: materialize final weights so
        # forward passes avoid recomputing norm (which forces fp32 aten::to).
        _remove_weight_norm_from_module(self._tts.audio_vae)
        # Follow the deployment dtype (--dtype) instead of hardcoding: the
        # AudioVAE loads fp32 from audiovae.pth outside vLLM's dtype
        # machinery, so without this it would stay fp32 under --dtype
        # bfloat16. Deriving from model_config.dtype keeps the pipeline
        # single-dtype for any configuration (float32 restores the original
        # behavior; bfloat16 keeps the optimization).
        vae_dtype = vllm_config.model_config.dtype
        self._tts.audio_vae.to(dtype=vae_dtype)

        self._side_dtype = self._tts.fusion_concat_proj.weight.dtype
        self._vae_dtype = next(self._tts.audio_vae.parameters()).dtype

        # Wrap audio_vae.encode so all callers (_encode_raw_audio, native
        # _encode_wav / build_prompt_cache) are dtype-safe under the bf16
        # cast; see _install_dtype_aware_encode for the fp32 output contract.
        _install_dtype_aware_encode(self._tts, self._vae_dtype)
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

        self._inference_timesteps = self._runtime_config.inference_timesteps
        self._cfg_value = _resolve_cfg_value(self._runtime_config.cfg_value)
        self._cfg_cutoff_ratio = self._runtime_config.cfg_cutoff_ratio
        # Number of trailing latent frames to keep as VAE receptive-field context
        # for sliding-window streaming decode. 12 matches the nanovllm reference
        # implementation and covers the longest VAE decoder receptive field.
        self._n_decode_pad_frames = 12
        use_cuda_graph = omni_platform.current_omni_platform.is_cuda()
        self._enable_torch_compile = (
            self._runtime_config.enable_torch_compile and omni_platform.current_omni_platform.supports_torch_inductor()
        )
        self._compile_vae = self._enable_torch_compile
        self._max_decode_steps = 2000
        self._max_batch_size = getattr(vllm_config.scheduler_config, "max_num_seqs", 4)

        # Speaker cache for ref_audio_feat across requests
        self._speaker_cache = get_speaker_cache()
        self._load_custom_voice_profiles()

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
        self._coalesce_audio_d2h = self._runtime_config.coalesce_audio_d2h
        self._enable_batched_vae_decode = self._runtime_config.enable_batched_vae_decode
        self._enable_batched_fsq_fusion = self._runtime_config.enable_batched_fsq_fusion
        self._batched_fsq_fusion_max_batch = self._runtime_config.batched_fsq_fusion_max_batch
        self._max_cached_graphs = self._max_batch_size
        self._mixed_graph_cache_limit = self._runtime_config.mixed_graph_cache_limit
        self._cuda_graph_warmup_steps = 0
        self._cuda_graph_warmup_threshold = 3
        self._unified_graphs: dict[int | tuple[int, int], _CapturedUnifiedDecodeGraph] = {}
        self._unified_graph_stats = _UnifiedDecodeGraphStats()
        self._enable_unified_decode_graph = self._runtime_config.unified_decode_graph_available(
            use_cuda_graph=use_cuda_graph
        )
        self._unified_decode_graph_max_batch_size = self._runtime_config.unified_decode_graph_max_batch_size
        self._unified_graph_bucket_sizes: frozenset[int] = self._build_unified_graph_bucket_sizes(
            min(self._max_batch_size, self._unified_decode_graph_max_batch_size)
        )
        self._mixed_graphs: dict[tuple[int, int] | tuple[int, int, int], _CapturedUnifiedDecodeGraph] = {}
        self._runner_assisted_unified_decode_graph_active = False
        self._runner_assisted_unified_decode_graph_batch_size = 0

        self._multichar_zh_split: dict[int, list[int]] | None = None

        self._active_states: dict[str, VoxCPM2RequestState] = {}
        self._mrv2_model_state: Any | None = None
        self._pending_requests: list[tuple[str, bool, torch.Tensor | None, int]] = []
        self._pending_batch_rows: list[int] | None = None
        self._results_queue: list[tuple[str, torch.Tensor | None]] = []
        self._audio_queue: list[tuple[str, Any]] = []
        self._last_audio_output_req_ids: list[str] = []

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

        work_device = self._device
        loaded = 0
        for profile in iter_custom_voice_profiles(custom_voice_dir, expected_model_type="voxcpm2"):
            tensors = load_validated_profile_tensors(
                profile,
                expected_model_type="voxcpm2",
                validate_profile=validate_voxcpm2_profile,
            )
            if tensors is None:
                continue

            ref_audio_feat = tensors.get("ref_audio_feat")
            audio_feat = tensors.get("audio_feat")
            mode = str(profile.get("mode") or "").lower()

            prompt_cache: dict[str, Any] = {"mode": mode}
            if ref_audio_feat is not None:
                prompt_cache["ref_audio_feat"] = ref_audio_feat.contiguous().to(work_device)
            if audio_feat is not None:
                prompt_cache["audio_feat"] = audio_feat.contiguous().to(work_device)
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

    def create_omni_model_state(self, vllm_config: VllmConfig, encoder_cache: Any, device: torch.device) -> Any:
        from .model_state import VoxCPM2ModelState

        if omni_platform.current_omni_platform.is_npu():
            from vllm_omni.platforms.npu.models.voxcpm2_model_state import NPUVoxCPM2ModelState

            return NPUVoxCPM2ModelState(vllm_config, self, encoder_cache, device)
        return VoxCPM2ModelState(vllm_config, self, encoder_cache, device)

    def _get_or_create_state(self, request_id: str) -> VoxCPM2RequestState:
        state = self._active_states.get(request_id)
        if state is None:
            raise RuntimeError(f"VoxCPM2 MRV2 request {request_id} has no runner slot")
        return state

    def _build_prompt_cache(
        self,
        ref_audio: object = None,
        prompt_audio: object = None,
        prompt_text: str | None = None,
    ) -> dict[str, Any] | None:
        """Build prompt cache, handling both file paths and raw audio data.

        The OpenAI speech adapter sends float32 waveform bytes with the sample
        rate; offline usage may send a sample list or file path.
        """
        tts = self.tts
        work_device = self._device

        def _is_raw_audio(v: Any) -> bool:
            import numbers

            return (
                isinstance(v, (list, tuple))
                and len(v) == 2
                and isinstance(v[1], numbers.Integral)
                and isinstance(v[0], (list, torch.Tensor, bytes))
            )

        if not _is_raw_audio(ref_audio) and not _is_raw_audio(prompt_audio):
            cache = tts.build_prompt_cache(
                prompt_text=prompt_text,
                prompt_wav_path=prompt_audio,
                reference_wav_path=ref_audio,
            )
            # File-path inputs use the native helper, which returns CPU features.
            # MRV2 prefill keeps them on GPU; raw waveform inputs below already
            # encode on device and do not pass through this conversion.
            if cache is not None:
                cache = {
                    key: value.to(work_device) if isinstance(value, torch.Tensor) else value
                    for key, value in cache.items()
                }
            return cache

        cache: dict[str, Any] = {}
        if ref_audio is not None:
            if _is_raw_audio(ref_audio):
                samples, sr = ref_audio
                cache["ref_audio_feat"] = _encode_raw_audio(tts, samples, sr)
            else:
                cache["ref_audio_feat"] = tts._encode_wav(ref_audio, padding_mode="right").to(work_device)

        if prompt_audio is not None and prompt_text is not None:
            cache["prompt_text"] = prompt_text
            if _is_raw_audio(prompt_audio):
                samples, sr = prompt_audio
                cache["audio_feat"] = _encode_raw_audio(
                    tts,
                    samples,
                    sr,
                    padding_mode="left",
                )
            else:
                cache["audio_feat"] = tts._encode_wav(prompt_audio, padding_mode="left").to(work_device)

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

    def _setup_execution_optimizations(self) -> None:
        if getattr(self, "_eager_optimizations_applied", False):
            return
        tts = self.tts
        estimator = tts.feat_decoder.estimator
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

        # The rewrites above are regular eager PyTorch operations. Ascend does
        # not support TorchInductor, but still benefits from fused projections
        # and the cached delta-time embedding.
        self._eager_optimizations_applied = True
        if targets:
            logger.info("VoxCPM2: eager optimizations applied to: %s", ", ".join(targets))
        if self._enable_torch_compile:
            self._setup_torch_compile()

    def _setup_torch_compile(self) -> None:
        tts = self.tts
        estimator = tts.feat_decoder.estimator
        if hasattr(estimator, "_compiled"):
            return

        targets: list[str] = []
        cfg = self._runtime_config

        external_cfm_capture = self._enable_cfm_cuda_graph or self._enable_unified_decode_graph
        if self._enable_unified_decode_graph:
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

        if self._compile_vae and not self._enable_vae_cuda_graph and not self._runtime_config.enable_incremental_vae:
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
        if feat.device.type != omni_platform.current_omni_platform.device_type:
            return self.tts.audio_vae.decode(feat)

        feat = feat.to(dtype=self._vae_dtype)
        sr_cond = self._get_vae_decode_sr_cond(feat.device)

        audio_vae = self.tts.audio_vae
        graph_runner = _vae_npu_graph_runner_for(audio_vae)
        if graph_runner and feat.dim() == 3:
            result = graph_runner.run(
                "decode",
                (feat, sr_cond),
                (),
                lambda z, sr: (audio_vae.decode(z, sr_cond=sr),),
            )
            # fp32 PCM boundary: same contract as the eager path below —
            # decode_ready_slots / make_audio_output copy_ this waveform into
            # the fp32 audio_buffer, and a bf16 waveform raises there under a
            # bf16 deployment. The graph path must honor the same boundary.
            return result[0].float()

        if not self._enable_vae_cuda_graph:
            # fp32 PCM boundary: decode_ready_slots / make_audio_output copy_
            # this waveform into the fp32 audio_buffer; a bf16 return raises
            # there on the non-per-patch MRV2 path under a bf16 deployment.
            return self.tts.audio_vae.decode(feat, sr_cond=sr_cond).float()

        graph_key = (feat.shape[0], feat.shape[-1])
        graph = self._vae_graphs.get(graph_key)
        if graph is None:
            graph = self._capture_vae_graph(feat)
            self._vae_graphs[graph_key] = graph

        graph.input_feat.copy_(feat)
        graph.graph.replay()
        return graph.output.float()

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
        graph.mu.copy_(mu)
        return self._run_cfm_cuda_graph_from_static_mu(graph, cond)

    def _run_cfm_cuda_graph_from_static_mu(self, graph: _CapturedCFMGraph, cond: torch.Tensor) -> torch.Tensor:
        graph.cond.copy_(cond)
        graph.noise.normal_()
        graph.graph.replay()
        return graph.output.clone()

    def _run_cfm_cuda_graph_to_state_buffer(
        self,
        state: VoxCPM2RequestState,
        mu: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        graph = self._get_cfm_cuda_graph(mu, cond)
        graph.mu.copy_(mu)
        graph.cond.copy_(cond)
        if self._has_deterministic_cfm_noise(state):
            self._fill_deterministic_cfm_noise_for_state(state, graph.noise)
        else:
            graph.noise.normal_()
        graph.graph.replay()

        output_shape = graph.output.shape
        if (
            state.cfm_output_gpu is None
            or state.cfm_output_gpu.shape != output_shape
            or state.cfm_output_gpu.device != graph.output.device
            or state.cfm_output_gpu.dtype != graph.output.dtype
        ):
            state.cfm_output_gpu = torch.empty(output_shape, device=graph.output.device, dtype=graph.output.dtype)
        state.cfm_output_gpu.copy_(graph.output)
        return state.cfm_output_gpu.transpose(1, 2)

    # -------------------- unified decode graph --------------------

    def _capture_unified_decode_graph(
        self, batch_size: int, token_capacity: int | None = None, patch_batch_size: int | None = None
    ) -> _CapturedUnifiedDecodeGraph:
        mixed = token_capacity is not None
        token_capacity = token_capacity if mixed else batch_size
        patch_count = batch_size if patch_batch_size is None else patch_batch_size
        H = self.config.hidden_size
        D = self._feat_dim
        P = self._patch_size
        dev = torch.device(self._device)
        dtype = self._side_dtype

        self._setup_cfm_buffers()
        self._setup_execution_optimizations()
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
            max_batch_size=max(1, patch_count),
        )

        g = _CapturedUnifiedDecodeGraph(
            graph=torch.cuda.CUDAGraph(),
            batch_size=batch_size,
            input_embeds=torch.zeros(token_capacity, H, device=dev, dtype=dtype),
            positions=torch.zeros(token_capacity, device=dev, dtype=torch.long),
            prev_feat_embed=torch.zeros(token_capacity, H, device=dev, dtype=dtype),
            prefix_feat_cond=torch.zeros(batch_size, P, D, device=dev, dtype=dtype),
            next_feat_embed=torch.zeros(batch_size, H, device=dev, dtype=dtype),
            cfm_output=torch.zeros(batch_size, D, P, device=dev, dtype=dtype),
            lm_hidden=torch.zeros(batch_size, H, device=dev, dtype=dtype),
            cfm_buffers=cfm_buffers,
            cfm_noise=torch.zeros(batch_size, D, P, device=dev, dtype=dtype),
        )

        if patch_batch_size is not None:
            g.patch_indices = torch.zeros(patch_count, device=dev, dtype=torch.long)
        if mixed:
            g.text_mask = torch.ones(token_capacity, 1, device=dev, dtype=dtype)
            g.last_token_indices = torch.zeros(batch_size, device=dev, dtype=torch.long)
        model_state = getattr(self, "_mrv2_model_state", None)
        per_patch_audio = model_state.per_patch_audio
        if patch_batch_size is not None and (not per_patch_audio or not model_state.incremental_vae):
            raise ValueError("Exact patch graphs require MRV2 incremental per-patch audio")
        if per_patch_audio:
            g.audio_slots = torch.full((batch_size,), len(model_state.slots), device=dev, dtype=torch.long)
            g.audio_rows_valid = torch.zeros(batch_size, device=dev, dtype=torch.bool)
        dit_proj = self._dit_proj_fn

        def unified_fwd():
            scaffold_h = self.model(input_ids=None, positions=g.positions, inputs_embeds=g.input_embeds)
            if isinstance(scaffold_h, tuple):
                scaffold_h = scaffold_h[0]
            lm_h = tts.fsq_layer(scaffold_h)
            if mixed:
                lm_h = lm_h * (1 - g.text_mask) + scaffold_h * g.text_mask
            res_input = tts.fusion_concat_proj(torch.cat([lm_h, g.prev_feat_embed], dim=-1))
            res_h = self.residual_model(positions=g.positions, inputs_embeds=res_input)
            if mixed:
                g.scaffold_hidden = scaffold_h
                lm_h = lm_h.index_select(0, g.last_token_indices)
                res_h = res_h.index_select(0, g.last_token_indices)
            output_lm_h = lm_h
            prefix_cond = g.prefix_feat_cond
            noise = g.cfm_noise
            if g.patch_indices is not None:
                if patch_count == 0:
                    g.stop_logits = self._stop_fn(output_lm_h)
                    g.pcm = torch.zeros(batch_size, P * int(tts.audio_vae.decode_chunk_size), device=dev)
                    g.pcm_valid = torch.zeros(batch_size, device=dev, dtype=torch.bool)
                    return torch.zeros_like(g.next_feat_embed), torch.zeros_like(g.cfm_output), output_lm_h
                lm_h = lm_h.index_select(0, g.patch_indices)
                res_h = res_h.index_select(0, g.patch_indices)
                prefix_cond = prefix_cond.index_select(0, g.patch_indices)
                noise = noise.index_select(0, g.patch_indices)
            dit_h = dit_proj(lm_h, res_h)
            cond = prefix_cond.transpose(1, 2).contiguous()
            cfm_out = _optimized_solve_euler_with_noise(
                tts.feat_decoder,
                dit_h,
                P,
                cond,
                noise,
                self._inference_timesteps,
                self._cfg_value,
                cfm_buffers,
                cfg_cutoff_ratio=self._cfg_cutoff_ratio,
            )
            feat_enc = tts.feat_encoder(cfm_out.transpose(1, 2).unsqueeze(1)).squeeze(1)
            next_embed = tts.enc_to_lm_proj(feat_enc)
            if per_patch_audio:
                g.stop_logits = self._stop_fn(output_lm_h)
                if getattr(model_state, "incremental_vae", False):
                    slots, rows_valid, scores = g.audio_slots, g.audio_rows_valid, g.stop_logits
                    if g.patch_indices is not None:
                        slots = slots.index_select(0, g.patch_indices)
                        rows_valid = rows_valid.index_select(0, g.patch_indices)
                        scores = scores.index_select(0, g.patch_indices)
                    pcm, valid = model_state.decode_patch(slots, rows_valid, cfm_out.transpose(1, 2))
                    model_state.commit_patch(slots, valid, next_embed, cfm_out.transpose(1, 2), scores)
                    if g.patch_indices is not None:
                        g.pcm = pcm.new_zeros(batch_size, pcm.shape[1]).index_copy_(0, g.patch_indices, pcm)
                        g.pcm_valid = valid.new_zeros(batch_size).index_copy_(0, g.patch_indices, valid)
                    else:
                        g.pcm, g.pcm_valid = pcm, valid
            if g.patch_indices is not None:
                next_embed = next_embed.new_zeros(batch_size, H).index_copy_(0, g.patch_indices, next_embed)
                cfm_out = cfm_out.new_zeros(batch_size, D, P).index_copy_(0, g.patch_indices, cfm_out)
            return next_embed, cfm_out, output_lm_h

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
        if patch_batch_size is not None:
            logger.info(
                "Exact patch graph captured: request_bucket=%d token_bucket=%d patch_count=%d",
                batch_size,
                token_capacity,
                patch_count,
            )
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
            if (
                self._state_tensor(state, "prev_feat_embed") is None
                or self._state_tensor(state, "curr_prefix_feat_cond") is None
            ):
                return "state_not_ready"
        return None

    def _maybe_log_unified_graph_stats(self) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        stats = self._unified_graph_stats
        total_skips = stats.total_skips
        if stats.replays - stats.logged_replays < 100 and total_skips - stats.logged_skips < 50:
            return
        stats.logged_replays = stats.replays
        stats.logged_skips = total_skips
        logger.debug(
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
        batch_context: VoxCPM2BatchSlots | None = None,
        mixed_token_capacity: int | None = None,
    ) -> torch.Tensor:
        model_state = getattr(self, "_mrv2_model_state", None)
        if model_state.per_patch_audio and model_state._pcm_copy_event is not None:
            # The shared capture pool can alias outputs across different buckets.
            torch.cuda.current_stream(self._device).wait_event(model_state._pcm_copy_event)
        graph_size = self._select_unified_graph_bucket_size(num_reqs)
        if graph_size is None:
            raise RuntimeError(f"No unified decode graph bucket for batch size {num_reqs}")
        exact = getattr(getattr(self, "_runtime_config", None), "enable_exact_patch_batch", False)
        patch_rows = (
            [
                i
                for i, (req_id, is_prefill, _, _) in enumerate(self._pending_requests)
                if patch_is_valid(self._active_states[req_id], is_prefill)
            ]
            if exact
            else None
        )
        graph_cache = self._unified_graphs if mixed_token_capacity is None else self._mixed_graphs
        graph_key = graph_size if mixed_token_capacity is None else (graph_size, mixed_token_capacity)
        if exact:
            graph_key = (
                (graph_size, len(patch_rows))
                if mixed_token_capacity is None
                else (graph_size, mixed_token_capacity, len(patch_rows))
            )
        g = graph_cache.get(graph_key)
        if g is None:
            g = (
                self._capture_unified_decode_graph(graph_size, mixed_token_capacity, len(patch_rows))
                if exact
                else self._capture_unified_decode_graph(graph_size, mixed_token_capacity)
            )
            graph_cache[graph_key] = g

        self._unified_graph_stats.replays += 1
        self._unified_graph_stats.record_replay_batch(real_batch_size=num_reqs, graph_bucket_size=graph_size)
        self._maybe_log_unified_graph_stats()

        states: list[VoxCPM2RequestState] = []
        # Requests already stopping must not commit another generated patch.
        commit_mask: list[bool] = []
        for req_id, _is_prefill, _embeds, _n in self._pending_requests:
            state = self._active_states[req_id]
            states.append(state)
            already_stopping = state.is_stopping
            commit_mask.append(patch_is_valid(state, _is_prefill))
            if not already_stopping and not _is_prefill:
                state.decode_step_count += 1
                if state.decode_step_count >= self._max_decode_steps:
                    state.is_stopping = True

        model_state = getattr(self, "_mrv2_model_state", None)
        slot_indices = model_state.indices_for(states, batch_context)
        if mixed_token_capacity is not None:
            self._prepare_mixed_graph_inputs(g, num_reqs)
        else:
            g.prev_feat_embed[:num_reqs].copy_(model_state.next_embed.index_select(0, slot_indices))
            g.prefix_feat_cond[:num_reqs].copy_(model_state.prefix_feat.index_select(0, slot_indices))

        num_tokens = sum(count for _, _, _, count in self._pending_requests) if mixed_token_capacity else num_reqs
        g.input_embeds[:num_tokens].copy_(inputs_embeds[:num_tokens])
        g.positions[:num_tokens].copy_(positions[:num_tokens])
        if mixed_token_capacity is not None:
            g.input_embeds[num_tokens:].zero_()
            g.positions[num_tokens:].zero_()
        elif graph_size > num_reqs:
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
        # Seeded rows overwrite their slice from their own generator, so the
        # noise a seeded request sees does not depend on batch composition.
        for i, state in enumerate(states[:num_reqs]):
            if mixed_token_capacity is not None and self._has_deterministic_cfm_noise(state):
                self._fill_deterministic_cfm_noise_for_state(state, g.cfm_noise[i : i + 1])
            elif state.cfm_generator is not None:
                g.cfm_noise[i : i + 1].normal_(generator=state.cfm_generator)

        if g.audio_slots is not None:
            g.audio_slots[:num_reqs].copy_(slot_indices)
            g.audio_slots[num_reqs:].fill_(len(model_state.slots))
            host = torch.tensor(commit_mask + [False] * (graph_size - num_reqs), dtype=torch.bool, pin_memory=True)
            g.audio_rows_valid.copy_(host, non_blocking=True)
        if g.patch_indices is not None:
            host_rows = torch.tensor(patch_rows, dtype=torch.long, pin_memory=True)
            g.patch_indices.copy_(host_rows, non_blocking=True)
        g.graph.replay()

        if g.audio_slots is not None and not getattr(model_state, "incremental_vae", False):
            with torch.no_grad():
                patch = g.cfm_output.transpose(1, 2)
                g.pcm, g.pcm_valid = model_state.decode_patch(g.audio_slots, g.audio_rows_valid, patch)
                model_state.commit_patch(g.audio_slots, g.pcm_valid, g.next_feat_embed, patch, g.stop_logits)

        self._commit_graph_decode_state(g, states, commit_mask, slot_indices, batch_context)

        if mixed_token_capacity is not None:
            requests = [
                (state, entry[1], entry[2], entry[3])
                for state, entry in zip(states, self._pending_requests, strict=True)
            ]
            finish_graph_prefills(requests)
        if g.pcm is not None:
            self._queue_graph_pcm(g, states, commit_mask)
            self._pending_requests.clear()
            return g.scaffold_hidden[:num_tokens] if mixed_token_capacity is not None else g.next_feat_embed[:num_reqs]
        self._precompute_stop_flags_for_audio_collect(states)
        audio_by_req = self._collect_audio_batch(states, batch_context=batch_context)
        for state in states:
            self._results_queue.append((state.request_id, self._state_tensor(state, "precomputed_stop_logits")))
            self._audio_queue.append((state.request_id, audio_by_req.get(state.request_id)))

        self._pending_requests.clear()
        return g.scaffold_hidden[:num_tokens] if mixed_token_capacity is not None else g.next_feat_embed[:num_reqs]

    def _commit_graph_decode_state(self, g, states, commit_mask, slot_indices, batch_context) -> None:
        """Commit valid patches while retaining the legacy stop-logit contract."""
        num_reqs = len(states)
        model_state = getattr(self, "_mrv2_model_state", None)
        with torch.no_grad():
            all_stop_logits = (
                g.stop_logits[:num_reqs] if g.stop_logits is not None else self._stop_fn(g.lm_hidden[:num_reqs])
            )
        if g.pcm is not None:
            # GPU valid/active masks own tensor commits. Host ready flags describe
            # initialized slot data, not whether this step committed a new patch.
            for state, committed in zip(states, commit_mask, strict=True):
                if not committed:
                    continue
                state.decode_state_ready = True
                state.stop_logits_ready = True
            return
        # store_decode_batch writes committed rows. Only stopping rows
        # need their logits refreshed separately.
        if not all(commit_mask):
            model_state.stop_logits.index_copy_(
                0,
                slot_indices,
                all_stop_logits.reshape(num_reqs, -1)[:, :2].to(dtype=model_state.stop_logits.dtype),
            )
        committed = [i for i, should_commit in enumerate(commit_mask) if should_commit]
        if committed:
            if len(committed) == num_reqs:
                # The usual decode step commits every row in graph order.
                # Avoid uploading a Python row list to CUDA and waiting
                # for the preceding graph replay to finish.
                model_state.store_decode_batch(
                    states,
                    all_stop_logits,
                    g.next_feat_embed[:num_reqs],
                    g.cfm_output[:num_reqs].transpose(1, 2),
                    batch_context,
                )
            else:
                rows = torch.tensor(committed, device=self._device, dtype=torch.long)
                model_state.store_decode_batch(
                    [states[i] for i in committed],
                    all_stop_logits.index_select(0, rows),
                    g.next_feat_embed.index_select(0, rows),
                    g.cfm_output.index_select(0, rows).transpose(1, 2),
                    batch_context,
                )
        for state in states:
            state.stop_logits_ready = True

    def _queue_graph_pcm(self, graph, states, commit_mask) -> None:
        """Keep PCM alive until async copy and preserve original batch rows."""
        model_state = self._mrv2_model_state
        num_reqs = len(states)
        # The output copy stream records this event before another replay.
        graph.audio_copy_event = torch.cuda.Event()
        model_state._pcm_copy_event = graph.audio_copy_event
        model_state._step_pcm = (graph.pcm[:num_reqs], graph.pcm_valid[:num_reqs], graph.audio_copy_event)
        for state, committed in zip(states, commit_mask, strict=True):
            state.audio_patch_ready = False
            logits = self._state_tensor(state, "precomputed_stop_logits") if committed else None
            # An already-scheduled extra step may enqueue stale stop logits.
            # GPU active_slots prevents patch commits and valid PCM output;
            # the scheduler ignores repeated stop results for finished requests.
            self._results_queue.append((state.request_id, logits))
            self._audio_queue.append((state.request_id, None))

    # -------------------- vllm hooks --------------------

    def _scheduled_patch_count(
        self, req_ids, num_scheduled_tokens, num_computed_tokens, num_prompt_tokens
    ) -> int | None:
        """Predict CPU eligibility before preprocess; GPU active remains authoritative."""
        if num_prompt_tokens is None or len(num_prompt_tokens) != len(req_ids):
            return None
        patch_count = 0
        for req_id, num_scheduled_token, num_computed_token, num_prompt_token in zip(
            req_ids, num_scheduled_tokens, num_computed_tokens, num_prompt_tokens, strict=True
        ):
            state = self._active_states.get(req_id)
            if state is None:
                return None
            if state.is_stopping:
                continue
            is_prefill = int(num_computed_token) < int(num_prompt_token) or not state.prefill_completed
            patch_count += not is_prefill or int(num_computed_token) + int(num_scheduled_token) == int(num_prompt_token)
        return patch_count

    def get_runner_assisted_full_attention_metadata_request(
        self,
        *,
        req_ids: Sequence[str],
        num_reqs: int,
        num_scheduled_tokens: Sequence[int],
        num_computed_tokens: Sequence[int],
        max_num_scheduled_tokens: int,
        num_prompt_tokens: Sequence[int] | None = None,
    ) -> RunnerAssistedFullAttentionMetadataRequest | None:
        if not self._enable_unified_decode_graph or not self._enable_cuda_graph:
            return None
        if len(req_ids) != num_reqs or len(num_computed_tokens) != num_reqs:
            return None
        has_prefill = any(int(n) == 0 for n in num_computed_tokens) or any(
            not getattr(self._active_states.get(req_id), "prefill_completed", False) for req_id in req_ids
        )
        exact = getattr(getattr(self, "_runtime_config", None), "enable_exact_patch_batch", False)
        patch_count = None
        if exact:
            if len(num_scheduled_tokens) != num_reqs:
                return None
            patch_count = self._scheduled_patch_count(
                req_ids, num_scheduled_tokens, num_computed_tokens, num_prompt_tokens
            )
            if patch_count is None:
                return None
            has_prefill |= any(
                int(num_computed_token) < int(num_prompt_token)
                for num_computed_token, num_prompt_token in zip(num_computed_tokens, num_prompt_tokens, strict=True)
            )
        if self._mixed_unified_graph_enabled() and has_prefill:
            tts_compiled = (
                getattr(self.tts.feat_decoder.estimator, "_compiled", False) if self._tts is not None else False
            )
            if not tts_compiled or self._cuda_graph_warmup_steps < self._cuda_graph_warmup_threshold:
                return None
            bucket = self._select_unified_graph_bucket_size(num_reqs)
            if (
                bucket is None
                or num_reqs > self._unified_decode_graph_max_batch_size
                or len(req_ids) != num_reqs
                or len(num_scheduled_tokens) != num_reqs
                or any(int(n) <= 0 for n in num_scheduled_tokens)
            ):
                return None
            tokens = self._mixed_token_capacity(sum(int(n) for n in num_scheduled_tokens), bucket)
            if tokens is None:
                return None
            key = (bucket, tokens, patch_count) if exact else (bucket, tokens)
            if (
                key not in self._mixed_graphs
                and self._mixed_graph_cache_limit > 0
                and len(self._mixed_graphs) >= self._mixed_graph_cache_limit
            ):
                return None
            return RunnerAssistedFullAttentionMetadataRequest(bucket, key not in self._mixed_graphs, tokens)
        model_state = getattr(self, "_mrv2_model_state", None)
        per_patch_single = num_reqs == 1 and getattr(model_state, "per_patch_audio", False)
        if (num_reqs <= 1 and not per_patch_single) or num_reqs > self._unified_decode_graph_max_batch_size:
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
                or self._state_tensor(state, "prev_feat_embed") is None
                or self._state_tensor(state, "curr_prefix_feat_cond") is None
            ):
                return None
        bucket_size = self._select_unified_graph_bucket_size(num_reqs)
        if bucket_size is None:
            return None
        key = (bucket_size, patch_count) if exact else bucket_size
        return RunnerAssistedFullAttentionMetadataRequest(
            num_reqs_padded=bucket_size,
            for_cudagraph_capture=key not in self._unified_graphs,
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
        dev = input_ids.device

        self._last_audio_output_req_ids = [req_id for req_id, _, _, _ in self._pending_requests]
        assert len(self._last_audio_output_req_ids) == len(set(self._last_audio_output_req_ids)), (
            "VoxCPM2 pending batch contains duplicate request IDs"
        )
        num_reqs = len(self._pending_requests)
        from .model_state import VoxCPM2BatchSlots

        batch_slots = kwargs.get("batch_slots")
        batch_slot_rows = kwargs.get("batch_slot_rows", {})
        batch_context = (
            VoxCPM2BatchSlots(batch_slots, batch_slot_rows) if isinstance(batch_slots, torch.Tensor) else None
        )
        num_decode = sum(1 for _, is_p, _, n in self._pending_requests if not is_p and n == 1)
        is_all_decode = num_decode == num_reqs and num_reqs > 0

        tts_compiled = getattr(self.tts.feat_decoder.estimator, "_compiled", False) if self._tts is not None else False
        graph_ready = tts_compiled and self._cuda_graph_warmup_steps >= self._cuda_graph_warmup_threshold
        if num_decode > 0 or (num_reqs > 0 and self._mixed_unified_graph_enabled()):
            self._cuda_graph_warmup_steps += 1

        can_use_graph = (
            self._enable_cuda_graph and graph_ready and intermediate_tensors is None and inputs_embeds is not None
        )

        if (
            self._mixed_unified_graph_enabled()
            and any(is_prefill for _, is_prefill, _, _ in self._pending_requests)
            and self._runner_assisted_unified_decode_graph_active
            and can_use_graph
            and num_reqs > 0
            and all(
                mixed_request_is_ready(self._active_states[req_id], is_prefill, self._mrv2_model_state.per_patch_audio)
                for req_id, is_prefill, _, _ in self._pending_requests
            )
        ):
            bucket = self._select_unified_graph_bucket_size(num_reqs)
            capacity = (
                self._mixed_token_capacity(sum(n for _, _, _, n in self._pending_requests), bucket)
                if bucket is not None
                else None
            )
            if capacity is not None and getattr(
                getattr(self, "_runtime_config", None), "enable_exact_patch_batch", False
            ):
                patch_count = sum(
                    patch_is_valid(self._active_states[req_id], is_prefill)
                    for req_id, is_prefill, _, _ in self._pending_requests
                )
                key = (bucket, capacity, patch_count)
                if (
                    key not in self._mixed_graphs
                    and self._mixed_graph_cache_limit > 0
                    and len(self._mixed_graphs) >= self._mixed_graph_cache_limit
                ):
                    capacity = None
            if capacity is not None:
                result = self._forward_unified_decode(inputs_embeds, positions, num_reqs, batch_context, capacity)
                return result

        unified_skip_reason = self._unified_decode_graph_skip_reason(
            can_use_graph=can_use_graph,
            is_all_decode=is_all_decode,
            num_reqs=num_reqs,
        )
        if unified_skip_reason is None:
            result = self._forward_unified_decode(inputs_embeds, positions, num_reqs, batch_context)
            return result
        if self._enable_unified_decode_graph and num_reqs > 0:
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
            if num_reqs not in self._scaffold_graphs:
                self._scaffold_graphs[num_reqs] = self._capture_graph(self.model, num_reqs, "scaffold")
            scaffold_hidden = self._replay_graph(self._scaffold_graphs[num_reqs], inputs_embeds, positions, num_reqs)

        else:
            model_output = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
            if isinstance(model_output, IntermediateTensors):
                return model_output
            scaffold_hidden = model_output
            if isinstance(scaffold_hidden, tuple):
                scaffold_hidden = scaffold_hidden[0]

        token_offset = 0
        residual_inputs: list[torch.Tensor] = []
        residual_positions: list[torch.Tensor] = []
        req_metas: list[tuple[VoxCPM2RequestState, bool, _PrefillResidualMeta | _DecodeResidualMeta]] = []
        pending_decode_fsq: list[tuple[VoxCPM2RequestState, torch.Tensor, torch.Tensor]] = []

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
                batched = self._prepare_residual_decode_batch(states, hidden_list, dev, batch_context)
                for (state, _, req_pos), (res_input, meta) in zip(pending_decode_fsq, batched):
                    residual_inputs.append(res_input)
                    residual_positions.append(req_pos)
                    req_metas.append((state, False, meta))
            pending_decode_fsq.clear()

        for req_id, is_prefill, _req_embeds, n in self._pending_requests:
            state = self._get_or_create_state(req_id)
            req_hidden = scaffold_hidden[token_offset : token_offset + n]
            req_pos = positions[token_offset : token_offset + n]

            if is_prefill:
                flush_decode_fsq_batch()
                res_input, meta = self._prepare_residual_prefill(state, req_hidden, dev, req_pos)
            elif state.prefill_completed:
                if (
                    self._enable_batched_fsq_fusion
                    and n == 1
                    and self._state_tensor(state, "prev_feat_embed") is not None
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

            can_finish_decode_batch = (
                bool(req_metas)
                and all(not is_prefill for _, is_prefill, _ in req_metas)
                and all(x.shape[0] == 1 for x in residual_inputs)
            )
            if can_finish_decode_batch:
                self._finish_decode_batch(req_metas, batch_out, batch_context)
                self._precompute_stop_flags_for_audio_collect([state for state, _, _ in req_metas])
                audio_by_req = self._collect_audio_batch(
                    [state for state, _, _ in req_metas],
                    batch_context=batch_context,
                )
                for state, _, _ in req_metas:
                    self._results_queue.append((state.request_id, self._state_tensor(state, "precomputed_stop_logits")))
                    self._audio_queue.append((state.request_id, audio_by_req.get(state.request_id)))
            else:
                offset = 0
                decoded_states: list[VoxCPM2RequestState] = []
                prefill_batch: list[tuple[VoxCPM2RequestState, _PrefillResidualMeta, torch.Tensor]] = []
                for idx, (state, is_prefill, meta) in enumerate(req_metas):
                    n = residual_inputs[idx].shape[0]
                    res_out = batch_out[offset : offset + n]
                    offset += n

                    if is_prefill:
                        if state.is_last_prefill_chunk:
                            prefill_batch.append((state, meta, res_out))
                    else:
                        self._finish_decode(state, meta, res_out)
                        decoded_states.append(state)

                if prefill_batch and self._runtime_config.enable_batched_prefill_tail:
                    self._finish_prefill_batch(prefill_batch)
                else:
                    for state, meta, res_out in prefill_batch:
                        self._finish_prefill(state, meta, res_out, dev)

                collect_states = [
                    state for state, is_prefill, _ in req_metas if not is_prefill or state.is_last_prefill_chunk
                ]
                audio_by_req: dict[str, torch.Tensor | None] = {}
                if collect_states:
                    self._precompute_stop_flags_for_audio_collect(collect_states)
                    audio_by_req = self._collect_audio_batch(
                        collect_states,
                        batch_context=batch_context,
                    )
                for state, is_prefill, _ in req_metas:
                    if is_prefill and not state.is_last_prefill_chunk:
                        self._results_queue.append((state.request_id, None))
                        self._audio_queue.append((state.request_id, None))
                    else:
                        self._results_queue.append(
                            (state.request_id, self._state_tensor(state, "precomputed_stop_logits"))
                        )
                        self._audio_queue.append((state.request_id, audio_by_req.get(state.request_id)))

        self._pending_requests.clear()
        return scaffold_hidden

    # -------------------- prefill / decode helpers --------------------

    def _prepare_residual_prefill(
        self,
        state: VoxCPM2RequestState,
        base_lm_out: torch.Tensor,
        dev: torch.device,
        req_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, _PrefillResidualMeta]:
        tts = self.tts
        text_mask, feat_mask, feat, feat_embed = state.prefill_masks
        text_mask = text_mask[:, req_pos]
        feat_mask = feat_mask[:, req_pos]
        feat = feat[:, req_pos]
        feat_embed = feat_embed[:, req_pos]
        tts_len = text_mask.shape[1]
        scaffold_len = base_lm_out.shape[0]
        assert scaffold_len == tts_len, (
            f"voxcpm2 prefill length mismatch: scaffold_len={scaffold_len} tts_len={tts_len}; "
            "caller must pad prompt_token_ids to the full prefill length "
            "(see VoxCPM2Adapter._build_prompt or the offline example)."
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
        state: VoxCPM2RequestState,
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

        prev = self._state_tensor(state, "prev_feat_embed").to(self._side_dtype)
        if prev.ndim == 1:
            prev = prev.unsqueeze(0)
        res_input = tts.fusion_concat_proj(torch.cat([lm_h, prev], dim=-1))
        return res_input, _DecodeResidualMeta(new_lm_hidden=lm_h)

    def _prepare_residual_decode_batch(
        self,
        states: list[VoxCPM2RequestState],
        base_lm_outs: list[torch.Tensor],
        dev: torch.device,
        batch_context: VoxCPM2BatchSlots | None = None,
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

        hidden_batch = torch.cat(
            [h.reshape(1, -1) if h.ndim == 1 else h for h in base_lm_outs],
            dim=0,
        )
        lm_h_batch = tts.fsq_layer(hidden_batch)
        if lm_h_batch.ndim == 1:
            lm_h_batch = lm_h_batch.unsqueeze(0)
        model_state = getattr(self, "_mrv2_model_state", None)
        prev_batch = model_state.gather_embeddings(states, batch_context).to(dtype=self._side_dtype)
        res_batch = tts.fusion_concat_proj(torch.cat([lm_h_batch, prev_batch], dim=-1))

        return [
            (
                res_batch[i : i + 1],
                _DecodeResidualMeta(new_lm_hidden=lm_h_batch[i : i + 1]),
            )
            for i in range(len(states))
        ]

    def _run_cfm(self, dit_h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self._cfm_buffers is not None:
            if self._enable_cfm_cuda_graph and dit_h.device.type == omni_platform.current_omni_platform.device_type:
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
            ).transpose(1, 2)
        return self.tts.feat_decoder(
            mu=dit_h,
            patch_size=self._patch_size,
            cond=cond,
            n_timesteps=self._inference_timesteps,
            cfg_value=self._cfg_value,
        ).transpose(1, 2)

    def _run_cfm_for_state(self, state: VoxCPM2RequestState, dit_h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self._cfm_buffers is not None:
            if self._enable_cfm_cuda_graph and dit_h.device.type == omni_platform.current_omni_platform.device_type:
                if self._has_deterministic_cfm_noise(state) and not self._enable_cfm_prealloc_output:
                    graph = self._get_cfm_cuda_graph(dit_h, cond)
                    graph.mu.copy_(dit_h)
                    graph.cond.copy_(cond)
                    self._fill_deterministic_cfm_noise_for_state(state, graph.noise)
                    graph.graph.replay()
                    return graph.output.clone().transpose(1, 2)
                if not self._enable_cfm_prealloc_output:
                    return self._run_cfm_cuda_graph(dit_h, cond).transpose(1, 2)
                return self._run_cfm_cuda_graph_to_state_buffer(state, dit_h, cond)
            if self._has_deterministic_cfm_noise(state):
                noise = self._cfm_buffers.noise[: dit_h.shape[0]]
                self._fill_deterministic_cfm_noise_for_state(state, noise)
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
            ).transpose(1, 2)
        return self.tts.feat_decoder(
            mu=dit_h,
            patch_size=self._patch_size,
            cond=cond,
            n_timesteps=self._inference_timesteps,
            cfg_value=self._cfg_value,
        ).transpose(1, 2)

    def _has_deterministic_cfm_noise(self, state: VoxCPM2RequestState) -> bool:
        """Whether this request's CFM noise is drawn deterministically."""
        return state.cfm_generator is not None or self._deterministic_cfm_noise

    def _fill_deterministic_cfm_noise_for_state(self, state: VoxCPM2RequestState, out: torch.Tensor) -> None:
        """Deterministically fill ``out`` for a request with deterministic noise.

        A request-level seed wins: the noise stream is a pure function of the
        seed, so identical text + seed reproduces byte-identical audio
        regardless of batch composition or request id. The replay-only
        ``deterministic_cfm_noise`` hash applies otherwise.
        """
        if state.cfm_generator is not None:
            out.normal_(generator=state.cfm_generator)
        else:
            self._fill_deterministic_cfm_noise(state, out)

    def _fill_deterministic_cfm_noise(self, state: VoxCPM2RequestState, out: torch.Tensor) -> None:
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

    def _finish_prefill_batch(
        self, batch: list[tuple[VoxCPM2RequestState, _PrefillResidualMeta, torch.Tensor]]
    ) -> None:
        if len(batch) == 1:
            state, meta, res_out = batch[0]
            self._finish_prefill(state, meta, res_out, None)
            return

        tts = self.tts
        self._setup_cfm_buffers()
        self._setup_execution_optimizations()

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

        any_seeded = any(state.cfm_generator is not None for state, _, _ in batch)
        if (self._deterministic_cfm_noise or any_seeded) and self._cfm_buffers is not None:
            noise = self._cfm_buffers.noise[:b]
            for i, (state, _, _) in enumerate(batch):
                if self._has_deterministic_cfm_noise(state):
                    self._fill_deterministic_cfm_noise_for_state(state, noise[i : i + 1])
                else:
                    noise[i : i + 1].normal_()
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
            state.prefill_completed = True
            state.prefill_masks = None
            state.prefill_embeds = None
            state.prompt_cache = None

    def _finish_prefill(
        self,
        state: VoxCPM2RequestState,
        meta: _PrefillResidualMeta,
        res_out: torch.Tensor,
        dev: torch.device | None,
    ) -> None:
        tts = self.tts
        lm_hidden = meta.lm_hidden
        prefix_feat_cond = meta.prefix_feat_cond
        residual_hidden = res_out[-1:, :]

        stop_logits = tts.stop_head(tts.stop_actn(tts.stop_proj(lm_hidden))).detach()
        dit_h = torch.cat([tts.lm_to_dit_proj(lm_hidden), tts.res_to_dit_proj(residual_hidden)], dim=-1)

        self._setup_cfm_buffers()
        self._setup_execution_optimizations()

        pred_feat = self._run_cfm_for_state(state, dit_h, prefix_feat_cond.transpose(1, 2).contiguous())

        with torch.no_grad():
            curr_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)

        self._commit_decode_state(state, stop_logits, curr_embed, pred_feat)
        state.decode_step_count = 0
        state.prefill_completed = True
        state.prefill_embeds = None
        state.prefill_masks = None
        state.prompt_cache = None

    def _finish_decode(self, state: VoxCPM2RequestState, meta: _DecodeResidualMeta, res_out: torch.Tensor) -> None:
        tts = self.tts

        lm_h = meta.new_lm_hidden
        res_h = res_out.unsqueeze(0) if res_out.ndim == 1 else res_out

        dit_proj = getattr(self, "_compiled_dit_proj", None) or self._dit_proj_fn
        stop_fn = getattr(self, "_compiled_stop_fn", None) or self._stop_fn

        pfc = self._state_tensor(state, "curr_prefix_feat_cond").to(self._side_dtype)
        if pfc.ndim == 2:
            pfc = pfc.unsqueeze(0)
        cond = pfc.transpose(1, 2).contiguous()

        dit_h = dit_proj(lm_h, res_h)
        pred_feat = self._run_cfm_for_state(state, dit_h, cond)
        next_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)
        stop_logits = stop_fn(lm_h).detach()
        self._commit_decode_state(state, stop_logits, next_embed, pred_feat)

    def _finish_decode_batch(
        self,
        req_metas: list[tuple[VoxCPM2RequestState, bool, _DecodeResidualMeta]],
        batch_out: torch.Tensor,
        batch_context: VoxCPM2BatchSlots | None = None,
    ) -> None:
        tts = self.tts

        dit_proj = getattr(self, "_compiled_dit_proj", None) or self._dit_proj_fn
        stop_fn = getattr(self, "_compiled_stop_fn", None) or self._stop_fn

        if self._enable_batched_cfm and not self._enable_cfm_cuda_graph:
            states = [state for state, _, _ in req_metas]
            lm_h = torch.cat([meta.new_lm_hidden for _, _, meta in req_metas], dim=0)
            model_state = getattr(self, "_mrv2_model_state", None)
            indices = model_state.indices_for(states, batch_context)
            pfc = model_state.prefix_feat.index_select(0, indices).to(self._side_dtype)
            dit_h = dit_proj(lm_h, batch_out)
            cond = pfc.transpose(1, 2).contiguous()
            any_seeded = any(state.cfm_generator is not None for state in states)
            if (self._deterministic_cfm_noise or any_seeded) and self._cfm_buffers is not None:
                noise = self._cfm_buffers.noise[: dit_h.size(0)]
                for i, state in enumerate(states):
                    if self._has_deterministic_cfm_noise(state):
                        self._fill_deterministic_cfm_noise_for_state(state, noise[i : i + 1])
                    else:
                        noise[i : i + 1].normal_()
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
                ).transpose(1, 2)
            else:
                pred_feat = self._run_cfm(dit_h, cond)
            next_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)
            stop_logits = stop_fn(lm_h).detach()

            model_state.store_decode_batch(states, stop_logits, next_embed, pred_feat, batch_context)

            return

        for i, (state, _, meta) in enumerate(req_metas):
            lm_h = meta.new_lm_hidden
            res_h = batch_out[i : i + 1]
            pfc = self._state_tensor(state, "curr_prefix_feat_cond").to(self._side_dtype)
            if pfc.ndim == 2:
                pfc = pfc.unsqueeze(0)
            cond = pfc.transpose(1, 2).contiguous()
            dit_h = dit_proj(lm_h, res_h)
            pred_feat = self._run_cfm_for_state(state, dit_h, cond)
            next_embed = tts.enc_to_lm_proj(tts.feat_encoder(pred_feat.unsqueeze(1))).squeeze(1)
            stop_logits = stop_fn(lm_h).detach()
            self._commit_decode_state(state, stop_logits, next_embed, pred_feat)

    def _commit_decode_state(
        self,
        state: VoxCPM2RequestState,
        stop_logits: torch.Tensor,
        next_embed: torch.Tensor,
        pred_feat: torch.Tensor,
    ) -> None:
        model_state = getattr(self, "_mrv2_model_state", None)
        model_state.store_decode_state(state, stop_logits, next_embed, pred_feat)

    def _state_tensor(self, state: VoxCPM2RequestState, name: str) -> torch.Tensor | None:
        model_state = getattr(self, "_mrv2_model_state", None)
        # MRV2 owns tensors in persistent slot buffers.
        return model_state.tensor_for(state, name)

    def _clear_state_tensor(self, state: VoxCPM2RequestState, name: str) -> None:
        model_state = getattr(self, "_mrv2_model_state", None)
        model_state.clear_tensor(state, name)

    # -------------------- audio collection --------------------

    def _vae_output_storage_may_be_reused(self) -> bool:
        return self._enable_vae_cuda_graph or bool(getattr(self.tts.audio_vae, "_compiled", False))

    def _precompute_stop_flags_for_audio_collect(self, states: list[VoxCPM2RequestState]) -> None:
        # Local import: upstream has both `from vllm_omni.platforms import
        # current_omni_platform` and `import vllm_omni.platforms as
        # omni_platform` styles across revisions; a function-local import
        # resolves against either.
        from vllm_omni.platforms import current_omni_platform

        model_state = getattr(self, "_mrv2_model_state", None)
        if model_state.per_patch_audio:
            return
        if self._audio_emit_every == 1 and getattr(self, "_vae_decode_every", 1) == 1:
            return

        # If a previous batch's copy is still in flight, sync it up front —
        # an empty-pending early return below must not skip this wait, or the
        # in-flight copy can land in pinned pages the allocator has reused.
        prev_event = getattr(self, "_pending_stop_mask_event", None)
        if prev_event is not None:
            prev_event.synchronize()
            self._pending_stop_mask_event = None

        pending: list[tuple[VoxCPM2RequestState, torch.Tensor]] = []
        for state in states:
            if state.is_stopping or state.precomputed_is_stopping is not None:
                continue
            stop_logits = self._state_tensor(state, "precomputed_stop_logits")
            if stop_logits is None or stop_logits.device.type != omni_platform.current_omni_platform.device_type:
                continue
            pending.append((state, stop_logits))
        if not pending:
            return

        model_state = getattr(self, "_mrv2_model_state", None)
        indices = model_state.indices_for([state for state, _ in pending])
        stacked = model_state.stop_logits.index_select(0, indices)

        device_module = torch.get_device_module(current_omni_platform.get_torch_device())

        # Compute the mask and stage the D2H copy on a dedicated side stream:
        # the compute stream (VAE decode etc.) never queues behind the copy,
        # and the consumer's event wait only covers the tiny transfer.
        if getattr(self, "_stop_mask_stream", None) is None:
            self._stop_mask_stream = device_module.Stream()
        fork = device_module.Event()
        fork.record()
        copy_done = device_module.Event()
        with device_module.stream(self._stop_mask_stream):
            self._stop_mask_stream.wait_event(fork)
            stop_mask_gpu = stacked[:, 1] > stacked[:, 0]
            # The side stream reads tensors that live on the compute stream;
            # record_stream defers allocator reuse until the read completes
            # even if the states drop their references.
            for _, stop_logits in pending:
                stop_logits.record_stream(self._stop_mask_stream)
            stacked.record_stream(self._stop_mask_stream)
            # pin_memory=True raises on recent PyTorch builds without an
            # accelerator; gate it like the host-tensor path below. The
            # pending states were filtered to this device's tensors above,
            # so the side-stream copy only ever runs with a device present.
            on_device = (
                omni_platform.current_omni_platform.device_type == "cuda"
                or getattr(omni_platform.current_omni_platform, "is_npu", lambda: False)()
            )
            stop_mask_cpu = torch.empty(len(pending), dtype=torch.bool, pin_memory=on_device)
            # Canary: a host read that races ahead of the DMA observes True —
            # a deterministic wrong value for keep-going requests, caught by
            # the latch test's expected-value assertion (instead of
            # uninitialized garbage that can accidentally match).
            stop_mask_cpu.fill_(True)
            if on_device:
                stop_mask_cpu.copy_(stop_mask_gpu, non_blocking=True)
            else:
                stop_mask_cpu.copy_(stop_mask_gpu)
            copy_done.record(self._stop_mask_stream)

        for i, (state, _) in enumerate(pending):
            state.pending_stop_mask_cpu = stop_mask_cpu[i : i + 1]
            state.pending_stop_mask_event = copy_done
        # Keep the GPU source tensors alive until the side-stream copy lands.
        self._pending_stop_mask_source = (stacked, stop_mask_gpu)
        self._pending_stop_mask_event = copy_done

    def _discard_pending_stop_mask(self, state: VoxCPM2RequestState) -> None:
        """Synchronize any in-flight stop-mask copy before dropping the
        pending view, so the single-slot device source can be safely
        replaced and no copy is left reading soon-to-be-reused memory."""
        event = state.pending_stop_mask_event
        if event is not None:
            event.synchronize()
        state.pending_stop_mask_cpu = None
        state.pending_stop_mask_event = None

    def _should_stop_from_cached_logits(self, state: VoxCPM2RequestState) -> bool:
        if state.is_stopping:
            return True
        if state.audio_length_limit_reached:
            # The request stops at its length limit; discard any pending
            # stop-mask copy (with its event synchronized) so the in-flight
            # D2H is not left writing into soon-to-be-reused memory.
            self._discard_pending_stop_mask(state)
            return True
        cached = state.precomputed_is_stopping
        if cached is not None:
            return cached
        if state.pending_stop_mask_cpu is not None:
            # Block the host until the async D2H copy has landed, mirroring
            # the delayed-audio path's event wait before host-buffer reads.
            event = state.pending_stop_mask_event
            if event is not None:
                event.synchronize()
                state.pending_stop_mask_event = None
            is_stopping = bool(state.pending_stop_mask_cpu[0])
            state.pending_stop_mask_cpu = None
            state.precomputed_is_stopping = is_stopping
            if is_stopping:
                state.is_stopping = True
            return is_stopping
        stop_logits = self._state_tensor(state, "precomputed_stop_logits")
        if stop_logits is None:
            return False
        is_stopping = bool(stop_logits[0, 1] > stop_logits[0, 0])
        state.precomputed_is_stopping = is_stopping
        if is_stopping:
            state.is_stopping = True
        return is_stopping

    def _audio_length_limit_reached(
        self,
        num_computed_tokens: int | np.ndarray,
        num_scheduled_tokens: int | np.ndarray,
        max_seq_len: int | np.ndarray | None,
    ) -> bool | np.ndarray:
        """Whether this span plus its sampled token exhausts the request budget."""
        model_config = getattr(getattr(self, "vllm_config", None), "model_config", None)
        max_model_len = getattr(model_config, "max_model_len", None)
        if max_seq_len is None:
            limit = max_model_len
        else:
            limit = np.minimum(max_seq_len, max_model_len) if max_model_len is not None else max_seq_len
        return num_computed_tokens + num_scheduled_tokens + 1 >= limit if limit is not None else False

    def _collect_audio(self, state: VoxCPM2RequestState) -> torch.Tensor | None:
        """Per-step sliding-window VAE decode (nanovllm pattern).

        Each decode step feeds ``[decode_pad, new_patch]`` through the VAE
        and slices out only the audio region corresponding to the new patch.
        The pad buffer (last ``_n_decode_pad_frames`` latent frames) provides
        the receptive-field context needed by the VAE's transposed convolutions,
        eliminating boundary artifacts between chunks.

        Returns the delta audio chunk (not cumulative) so the output processor
        can stream each chunk to the client independently.
        """

        patch = self._state_tensor(state, "last_audio_patch_gpu")
        if patch is None:
            return None
        self._clear_state_tensor(state, "last_audio_patch_gpu")

        # patch shape: (patch_size, feat_dim) or (1, patch_size, feat_dim)
        # Keep latents fp32: they feed append_pending_latent / store_decode_pad
        # which copy_ into fp32 pending_latents / decode_pad, and the VAE
        # decode casts to the deployment dtype itself in _run_vae_decode.
        new_latent = patch.reshape(-1, self._feat_dim).to(torch.float32)
        model_state = getattr(self, "_mrv2_model_state", None)
        if model_state.per_patch_audio:
            indices = model_state.indices_for([state])
            pcm, valid = model_state.decode_patch(
                indices, torch.ones(1, dtype=torch.bool, device=indices.device), new_latent.unsqueeze(0)
            )
            model_state.record_eager_pcm(state, pcm[0], valid[0])
            return None
        vae_decode_every = getattr(self, "_vae_decode_every", 1)
        if vae_decode_every > 1:
            is_stopping = self._should_stop_from_cached_logits(state)
            model_state = getattr(self, "_mrv2_model_state", None)
            pending = model_state.append_pending_latent(state, new_latent)
            pending_count = state.pending_vae_count
            if not is_stopping and pending_count < vae_decode_every:
                return None
            new_latent = pending
            state.pending_vae_count = 0
            state.pending_vae_latents_gpu.clear()

        n_new = new_latent.shape[0]  # = patch_size (typically 4)

        # Build VAE input: [pad_frames | new_latent]
        if self._state_tensor(state, "decode_pad") is not None:
            vae_input = torch.cat([self._state_tensor(state, "decode_pad"), new_latent], dim=0)
            pad_frames = self._state_tensor(state, "decode_pad").shape[0]
        else:
            vae_input = new_latent
            pad_frames = 0

        # VAE decode: (1, feat_dim, T_frames) -> (1, 1, T_samples)
        feat = vae_input.unsqueeze(0).transpose(1, 2).contiguous()
        vae_frames = vae_input.shape[0]
        with torch.no_grad():
            audio = self._run_vae_decode(feat.to(self._device)).reshape(-1)

        # Slice out only the new audio (after the pad region).
        # Each latent frame maps to decoder_chunk_size audio samples.
        dcs = int(getattr(self.tts.audio_vae, "decode_chunk_size", audio.numel() // vae_frames))
        new_audio = audio[pad_frames * dcs : (pad_frames + n_new) * dcs]

        # Roll the pad buffer: keep last N latent frames as context for next step.
        all_latents = vae_input  # [pad + new]
        model_state = getattr(self, "_mrv2_model_state", None)
        model_state.store_decode_pad(state, all_latents)

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
            return merged_audio
        if self._coalesce_audio_d2h:
            audio_chunk = new_audio.detach()
            if self._vae_output_storage_may_be_reused():
                audio_chunk = audio_chunk.clone()
            return audio_chunk
        return new_audio

    def _can_collect_audio_batch(
        self,
        states: list[VoxCPM2RequestState],
    ) -> bool:
        model_state = getattr(self, "_mrv2_model_state", None)
        if model_state.per_patch_audio:
            return False  # Graph path owns VAE; warmup falls back per request.
        if not getattr(self, "_enable_batched_vae_decode", False):
            return False
        if self._audio_emit_every > 1 or self._enable_vae_cuda_graph:
            return False
        return len(states) > 1

    def _collect_audio_batch(
        self,
        states: list[VoxCPM2RequestState],
        batch_context: VoxCPM2BatchSlots | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if not self._can_collect_audio_batch(states):
            return {
                state.request_id: self._collect_audio(
                    state,
                )
                for state in states
            }

        outputs: dict[str, torch.Tensor | None] = {state.request_id: None for state in states}
        model_state = getattr(self, "_mrv2_model_state", None)
        vae_decode_every = getattr(self, "_vae_decode_every", 1)

        # Keep metadata on the host as before, but pack tensor data once
        # per shape directly from slots and append decoded PCM in place.
        cohorts: dict[int, list[tuple[Any, int, int]]] = {}
        patch_states = [state for state in states if state.audio_patch_ready]
        if vae_decode_every > 1:
            model_state.append_pending_batch(patch_states, batch_context)
        for state in patch_states:
            state.audio_patch_ready = False
            if vae_decode_every > 1:
                if state.pending_vae_count < vae_decode_every and not self._should_stop_from_cached_logits(state):
                    continue
                n_new = state.pending_vae_count * self._patch_size
            else:
                n_new = self._patch_size
            frames = state.decode_pad_len + n_new
            cohorts.setdefault(frames, []).append((state, state.decode_pad_len, n_new))
        for group in cohorts.values():
            model_state.decode_ready_slots(group, batch_context)
        return outputs

    # -------------------- compute_logits --------------------

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> torch.Tensor | None:
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None

        model_state = getattr(self, "_mrv2_model_state", None)
        if hidden_states.is_cuda:
            # The MRV2 path decides stops via model_state directly, but a
            # pending stop-mask D2H may still carry an unresolved positive
            # decision. RESOLVE it here (synchronize + latch into
            # is_stopping) — build_stop_logits only forces the stop when
            # state.is_stopping is already true (metadata mode 2) — rather
            # than discarding the mask and silently dropping the stop.
            for state in self._active_states.values():
                if state.pending_stop_mask_cpu is not None:
                    self._should_stop_from_cached_logits(state)
            return model_state.build_stop_logits(hidden_states)

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
                        self._clear_state_tensor(state, "precomputed_stop_logits")
                        state.precomputed_is_stopping = None
                        self._discard_pending_stop_mask(state)
                    else:
                        logits[i, 0] = stop_logits[0, 0]
                        logits[i, 1] = stop_logits[0, 1]
                        if state is not None:
                            if state.pending_stop_mask_cpu is not None:
                                self._should_stop_from_cached_logits(state)
                            if state.precomputed_is_stopping is not None:
                                state.is_stopping = state.precomputed_is_stopping
                            self._clear_state_tensor(state, "precomputed_stop_logits")
                            state.precomputed_is_stopping = None
                            self._discard_pending_stop_mask(state)
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

        model_state = getattr(self, "_mrv2_model_state", None)
        return model_state.make_audio_output(model_outputs)

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

    def preprocess_batch_mrv2(self, *, req_infos: list[dict[str, Any]], device: torch.device) -> None:
        """MRV2 batch embeddings; each reference waveform is encoded separately."""
        split_map = self._get_multichar_zh_split()
        candidates: list[tuple[dict[str, Any], list[int]]] = []
        for info in req_infos:
            extra = info.get("additional_information")
            data = {**extra, **info} if isinstance(extra, dict) else info
            slot = data.get("slot_index")
            if slot is None:
                continue
            state = self._mrv2_model_state.slots[slot]
            if state.request_id != data.get("req_id") or state.prefill_embeds is not None:
                continue
            if state.prompt_cache is not None or any(
                data.get(key) is not None
                for key in ("reference_audio", "ref_audio", "prompt_audio", "voice_profile", "voice_name")
            ):
                continue
            ids = data.get("text_token_ids")
            if not ids or not ids[0]:
                continue
            token_ids = list(ids[0])
            if split_map and any(token in split_map for token in token_ids):
                # Preserve the normal prefill validation path.
                continue
            if token_ids[0] == self.config.bos_token_id:
                token_ids = token_ids[1:]
            token_ids.append(self.tts.audio_start_token)
            candidates.append((info, token_ids))

        if len(candidates) >= 2:
            lengths = [len(tokens) for _, tokens in candidates]
            host_tokens = torch.tensor(
                [token for _, tokens in candidates for token in tokens],
                dtype=torch.long,
                pin_memory=device.type == "cuda",
            )
            embedded = self.model.embed_input_ids(host_tokens.to(device, non_blocking=True))
            for (info, _), text_embed in zip(candidates, embedded.split(lengths), strict=True):
                info["prefill_text_embed"] = text_embed

        # Encode waveforms individually; batch the downstream prefill embeddings.
        voice: list[tuple[dict[str, Any], VoxCPM2RequestState, list[int], object, object, str | None]] = []
        for info in req_infos:
            extra = info.get("additional_information")
            data = {**extra, **info} if isinstance(extra, dict) else info
            slot = data.get("slot_index")
            if slot is None or data.get("voice_profile") is not None:
                continue
            state = self._mrv2_model_state.slots[slot]
            if (
                state.request_id != data.get("req_id")
                or state.prefill_embeds is not None
                or state.prompt_cache is not None
            ):
                continue
            ref = data.get("reference_audio")
            if ref is None:
                ref = data.get("ref_audio")
            prompt = data.get("prompt_audio")
            if isinstance(ref, list):
                ref = ref[0] if ref else None
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else None
            text = data.get("prompt_text")
            if isinstance(text, list):
                text = text[0] if text else None
            if ref is None and (prompt is None or text is None):
                continue
            voice_name = data.get("voice_name")
            if isinstance(voice_name, list):
                voice_name = voice_name[0] if voice_name else None
            if voice_name:
                key = self._speaker_cache.make_cache_key(
                    voice_name, model_type="voxcpm2", created_at=int(data.get("voice_created_at") or 0)
                )
                if self._speaker_cache.get(key) is not None:
                    # The normal per-request path reuses this cache entry and
                    # gives it precedence over freshly supplied audio.
                    continue
            if any(
                not isinstance(value, (list, tuple)) or len(value) != 2 or not isinstance(value[1], int)
                for value in (ref, prompt if text is not None else None)
                if value is not None
            ):
                continue
            ids = data.get("text_token_ids")
            if not ids or not ids[0]:
                continue
            token_ids = list(ids[0])
            if split_map and any(token in split_map for token in token_ids):
                continue
            if token_ids[0] == self.config.bos_token_id:
                token_ids = token_ids[1:]
            voice.append((info, state, token_ids, ref, prompt, text))

        if len(voice) < 2:
            return
        caches = [
            self._build_prompt_cache(ref_audio=ref, prompt_audio=prompt, prompt_text=text)
            for _, _, _, ref, prompt, text in voice
        ]
        prepared: list[_PrefillInputs] = []
        for (_, state, token_ids, *_), cache in zip(voice, caches, strict=True):
            state.prompt_cache = cache
            try:
                prepared.append(self._build_prefill_inputs(token_ids, device, state=state))
            finally:
                state.prompt_cache = None

        text_lengths = [item.text_token.shape[1] for item in prepared]
        text_embeds = self.model.embed_input_ids(torch.cat([item.text_token.reshape(-1) for item in prepared])).split(
            text_lengths
        )
        feat_embeds: list[torch.Tensor | None] = [None] * len(prepared)
        shape_groups: dict[tuple[int, ...], list[int]] = {}
        for row, item in enumerate(prepared):
            shape_groups.setdefault(tuple(item.audio_feat.shape[1:]), []).append(row)
        for rows in shape_groups.values():
            batch_feat = torch.cat([prepared[row].audio_feat for row in rows], dim=0)
            encoded = self.tts.enc_to_lm_proj(self.tts.feat_encoder(batch_feat))
            for i, row in enumerate(rows):
                feat_embeds[row] = encoded[i : i + 1]
        for (info, *_), cache, inputs, text_embed, feat_embed in zip(
            voice, caches, prepared, text_embeds, feat_embeds, strict=True
        ):
            assert feat_embed is not None
            text_mask, audio_mask = inputs.text_mask, inputs.audio_mask
            embeds = (
                text_mask.unsqueeze(-1) * text_embed.unsqueeze(0) + audio_mask.unsqueeze(-1) * feat_embed
            ).squeeze(0)
            info["prepared_prefill"] = _PreparedPrefill(
                cache, embeds, text_mask, audio_mask, inputs.audio_feat, feat_embed
            )

    def preprocess_decode_batch_mrv2(
        self,
        *,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor,
        req_infos: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        model_state = self._mrv2_model_state
        states: list[VoxCPM2RequestState] = []
        rows = self._pending_batch_rows
        if rows is None:
            raise RuntimeError("VoxCPM2 MRv2 decode batch requires a batch row context")
        for info in req_infos:
            slot = info["slot_index"]
            state = model_state.slots[slot]
            if state.request_id != info["req_id"]:
                raise RuntimeError(f"VoxCPM2 slot {slot} does not belong to {info['req_id']}")
            states.append(state)

        if any(state.decode_state_ready for state in states):
            embeds = model_state.gather_embeddings(states, model_state.preprocess_batch_slots).to(
                device=input_embeds.device, dtype=self._side_dtype
            )
            if not all(state.decode_state_ready for state in states):
                ready = torch.tensor([state.decode_state_ready for state in states], device=embeds.device)
                embeds = embeds.masked_fill(~ready[:, None], 0)
        else:
            embeds = input_embeds.new_zeros((len(states), self.config.hidden_size), dtype=self._side_dtype)

        for state, info in zip(states, req_infos, strict=True):
            self._pending_requests.append((state.request_id, False, None, 1))
            rows.append(info["batch_row"])
        empty = input_embeds.new_empty((len(states), 0))
        return input_ids, embeds, empty, empty, [{} for _ in states]

    @staticmethod
    def _resolve_request_id(info: dict[str, Any], fallback: str = "default") -> str:
        return str(info.get("req_id") or fallback)

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
        req_id = self._resolve_request_id(info_dict)
        model_state = getattr(self, "_mrv2_model_state", None)
        slot_index = info_dict.get("slot_index")
        if slot_index is None:
            raise RuntimeError("VoxCPM2 MRv2 preprocess requires a runner slot")
        state = model_state.slots[slot_index]
        if state.request_id != req_id:
            raise RuntimeError(f"VoxCPM2 slot {slot_index} does not belong to {req_id}")
        is_prefill = bool(info_dict.get("_omni_is_prefill", span_len > 1))

        if is_prefill:
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

            num_computed_tokens = int(info_dict.get("_omni_num_computed_tokens", 0))
            is_first_prefill_chunk = num_computed_tokens == 0
            if is_first_prefill_chunk:
                state.decode_pad = None
                state.decode_pad_len = 0
                state.pending_vae_count = 0
                state.decode_state_ready = False
                state.audio_patch_ready = False
                state.stop_logits_ready = False
                state.prefill_completed = False
                state.is_last_prefill_chunk = False
                state.decode_step_count = 0
                state.precomputed_stop_logits = None
                state.precomputed_is_stopping = None
                self._discard_pending_stop_mask(state)
                state.last_audio_patch_gpu = None
                # SamplingParams.seed reaches vLLM's own sampler but never the
                # CFM noise draws below, so a seeded request threads its seed
                # into a dedicated generator here (same pattern as gepard).
                state.cfm_noise_step = 0
                state.cfm_generator = None
                seed = info_dict.get("_omni_seed")
                if seed is not None:
                    state.cfm_generator = torch.Generator(device=self._device)
                    state.cfm_generator.manual_seed(int(seed))
                if not hasattr(state, "pending_audio_chunks_gpu"):
                    state.pending_audio_chunks_gpu = []
                state.pending_audio_chunks_gpu.clear()
                state.curr_embed_for_next = None
                state.prev_feat_embed = None
                state.curr_prefix_feat_cond = None
                state.is_stopping = False
                state.prompt_cache = None
                state.prefill_embeds = None
                state.prefill_masks = None

            prepared_prefill = info_dict.get("prepared_prefill")
            if prepared_prefill is not None:
                state.prompt_cache = prepared_prefill.prompt_cache
                state.prefill_embeds = prepared_prefill.embeds
                state.prefill_masks = (
                    prepared_prefill.text_mask,
                    prepared_prefill.audio_mask,
                    prepared_prefill.audio_feat,
                    prepared_prefill.feat_embed,
                )

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

            voice_name = info_dict.get("voice_name")
            if isinstance(voice_name, list):
                voice_name = voice_name[0] if voice_name else None
            _created_at = int(info_dict.get("voice_created_at") or 0)

            if (
                prepared_prefill is not None
                and voice_name
                and state.prompt_cache.get("mode") == "reference"
                and "ref_audio_feat" in state.prompt_cache
            ):
                key = self._speaker_cache.make_cache_key(voice_name, model_type="voxcpm2", created_at=_created_at)
                self._speaker_cache.put(key, {"ref_audio_feat": state.prompt_cache["ref_audio_feat"].clone()})

            if state.prompt_cache is None and voice_name:
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
                    cached_feat = state.prompt_cache["ref_audio_feat"]
                    self._speaker_cache.put(
                        _key,
                        {"ref_audio_feat": cached_feat.clone()},
                    )
                    logger.debug("Speaker cache STORE for VoxCPM2 speaker '%s'", voice_name)
            if state.prefill_embeds is None or state.prefill_masks is None:
                inputs = self._build_prefill_inputs(token_ids, dev, state=state)
                tts = self.tts
                text_embed = info_dict.get("prefill_text_embed")
                if text_embed is not None and text_embed.shape[0] == inputs.text_token.shape[1]:
                    # Zero-shot has no audio positions: its audio embedding is
                    # always masked out, so skip the redundant feature encoder.
                    text_embed = text_embed.unsqueeze(0)
                    feat_embed = torch.zeros_like(text_embed)
                else:
                    text_embed = self.model.embed_input_ids(inputs.text_token.to(dev))
                    feat_embed = tts.enc_to_lm_proj(tts.feat_encoder(inputs.audio_feat))
                text_mask, feat_mask = inputs.text_mask, inputs.audio_mask
                embeds = (text_mask.unsqueeze(-1) * text_embed + feat_mask.unsqueeze(-1) * feat_embed).squeeze(0)
                state.prefill_masks = (text_mask, feat_mask, inputs.audio_feat, feat_embed)
                state.prefill_embeds = embeds
            state.prefill_chunk_start = num_computed_tokens
            chunk_end = num_computed_tokens + span_len
            # span_len equals num_scheduled_tokens. Agreement with
            # _scheduled_patch_count assumes prefill_embeds.shape[0] equals the
            # runner's num_prompt_tokens. build_voxcpm2_prompt must ensure this
            # prerequisite, including reference-audio positions.
            state.is_last_prefill_chunk = chunk_end == state.prefill_embeds.shape[0]
            if chunk_end > state.prefill_embeds.shape[0]:
                raise ValueError(
                    "VoxCPM2 chunked prefill exceeds the constructed prompt: "
                    f"request_id={req_id} start={num_computed_tokens} end={chunk_end} "
                    f"prompt_len={state.prefill_embeds.shape[0]}"
                )
            embeds = state.prefill_embeds[num_computed_tokens:chunk_end]
        else:
            curr = self._state_tensor(state, "curr_embed_for_next")
            if curr is not None:
                embeds = curr.to(dev, dtype=self._side_dtype).reshape(1, -1)
            else:
                embeds = torch.zeros(1, self.config.hidden_size, device=dev, dtype=self._side_dtype)

        self._pending_requests.append((req_id, is_prefill, embeds, span_len))
        if self._pending_batch_rows is None or "batch_row" not in info_dict:
            raise RuntimeError("VoxCPM2 MRv2 preprocess requires a batch row context")
        self._pending_batch_rows.append(info_dict["batch_row"])
        return input_ids, embeds, {}

    # -------------------- build prefill inputs --------------------

    def _build_prefill_inputs(
        self,
        token_ids: list[int],
        dev: torch.device,
        *,
        state: VoxCPM2RequestState,
    ) -> _PrefillInputs:
        tts = self.tts
        dtype = self._side_dtype
        cache = state.prompt_cache
        mode = cache.get("mode", "continuation") if cache else "zero_shot"

        if cache and mode in ("continuation", "ref_continuation"):
            prompt_text = cache.get("prompt_text", "")
            prompt_ids = list(tts.text_tokenizer(prompt_text)) if prompt_text else []
            all_ids = prompt_ids + token_ids
        else:
            all_ids = token_ids

        work_device = dev
        use_async_h2d = work_device.type == "cuda"
        text_token = torch.tensor([*all_ids, tts.audio_start_token], dtype=torch.int32, pin_memory=use_async_h2d).to(
            work_device, non_blocking=use_async_h2d
        )
        text_len = text_token.shape[0]
        latent_dim = tts.audio_vae.latent_dim
        ps = self._patch_size

        if mode in ("zero_shot", "continuation"):
            audio_feat = (
                cache["audio_feat"]
                if cache
                else torch.empty((0, ps, latent_dim), dtype=torch.float32, device=work_device)
            )
            a_len = audio_feat.size(0)
            text_token = torch.cat([text_token, torch.zeros(a_len, dtype=torch.int32, device=work_device)])
            audio_feat = torch.cat(
                [torch.zeros((text_len, ps, latent_dim), dtype=torch.float32, device=work_device), audio_feat]
            )
            text_mask = torch.cat(
                [
                    torch.ones(text_len, dtype=torch.int32, device=work_device),
                    torch.zeros(a_len, dtype=torch.int32, device=work_device),
                ]
            )
            audio_mask = torch.cat(
                [
                    torch.zeros(text_len, dtype=torch.int32, device=work_device),
                    torch.ones(a_len, dtype=torch.int32, device=work_device),
                ]
            )
        elif mode == "reference":
            ref = cache["ref_audio_feat"]
            rt, rf, rtm, ram = tts._make_ref_prefix(ref, text_token.device)
            text_token = torch.cat([rt.to(work_device), text_token])
            audio_feat = torch.cat(
                [rf.to(work_device), torch.zeros((text_len, ps, latent_dim), dtype=torch.float32, device=work_device)]
            )
            text_mask = torch.cat([rtm.to(work_device), torch.ones(text_len, dtype=torch.int32, device=work_device)])
            audio_mask = torch.cat([ram.to(work_device), torch.zeros(text_len, dtype=torch.int32, device=work_device)])
        else:  # ref_continuation
            ref = cache["ref_audio_feat"]
            prompt = cache["audio_feat"]
            p_len = prompt.size(0)
            rt, rf, rtm, ram = tts._make_ref_prefix(ref, text_token.device)
            text_token = torch.cat(
                [rt.to(work_device), text_token, torch.zeros(p_len, dtype=torch.int32, device=work_device)]
            )
            audio_feat = torch.cat(
                [
                    rf.to(work_device),
                    torch.zeros((text_len, ps, latent_dim), dtype=torch.float32, device=work_device),
                    prompt,
                ]
            )
            ones_t = torch.ones(text_len, dtype=torch.int32, device=work_device)
            zeros_p = torch.zeros(p_len, dtype=torch.int32, device=work_device)
            zeros_t = torch.zeros(text_len, dtype=torch.int32, device=work_device)
            ones_p = torch.ones(p_len, dtype=torch.int32, device=work_device)
            text_mask = torch.cat([rtm.to(work_device), ones_t, zeros_p])
            audio_mask = torch.cat([ram.to(work_device), zeros_t, ones_p])

        return _PrefillInputs(
            text_token=text_token.unsqueeze(0).to(dev),
            audio_feat=audio_feat.unsqueeze(0).to(dev).to(dtype),
            text_mask=text_mask.unsqueeze(0).to(dev),
            audio_mask=audio_mask.unsqueeze(0).to(dev),
        )

    # -------------------- weight loading --------------------

    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"base_lm.": "model."})

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self._startup_lora_applied:
            raise ValueError("Reloading VoxCPM2 weights with a startup LoRA is unsupported; restart the server")

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

        if adapter_path := self._runtime_config.startup_lora_path:
            merged = merge_voxcpm2_lora(
                adapter_path, base_lm=self.model, residual_lm=self.residual_model, tts=self._tts
            )
            self._startup_lora_applied = True
            logger.info("Merged VoxCPM2 startup LoRA from %s into %d linear layers", adapter_path, merged)

        logger.info(
            "Loaded VoxCPM2 (patch=%d, feat_dim=%d, dtype=%s)", self._patch_size, self._feat_dim, self._side_dtype
        )
        return loaded

    def _mixed_token_capacity(self, num_tokens: int, request_bucket: int) -> int | None:
        limit = self.vllm_config.scheduler_config.max_num_batched_tokens
        required = max(num_tokens, request_bucket)
        if required > limit:
            return None
        return min(limit, 1 << (required - 1).bit_length())

    def _mixed_unified_graph_enabled(self) -> bool:
        return self._enable_unified_decode_graph

    def _prepare_mixed_graph_inputs(self, g: _CapturedUnifiedDecodeGraph, num_reqs: int) -> None:
        """Map packed token inputs to per-request patch conditions."""
        offset = 0
        last_indices = []
        g.text_mask.zero_()
        g.prev_feat_embed.zero_()
        g.prefix_feat_cond.zero_()
        for row, (req_id, is_prefill, _embeds, count) in enumerate(self._pending_requests):
            state = self._active_states[req_id]
            if is_prefill:
                text_mask, audio_mask, audio_feat, feat_embed = state.prefill_masks
                start = state.prefill_chunk_start
                end = start + count
                text_mask = text_mask[:, start:end]
                audio_mask = audio_mask[:, start:end]
                feat_embed = feat_embed[:, start:end]
                g.text_mask[offset : offset + count].copy_(text_mask.reshape(-1, 1))
                g.prev_feat_embed[offset : offset + count].copy_(
                    (audio_mask.unsqueeze(-1) * feat_embed).reshape(count, -1)
                )
                g.prefix_feat_cond[row].copy_(audio_feat[0, end - 1])
            else:
                g.prev_feat_embed[offset].copy_(self._state_tensor(state, "prev_feat_embed").reshape(-1))
                g.prefix_feat_cond[row].copy_(self._state_tensor(state, "curr_prefix_feat_cond"))
            offset += count
            last_indices.append(offset - 1)
        last_indices.extend([last_indices[-1]] * (g.batch_size - num_reqs))
        host = torch.tensor(last_indices, dtype=torch.long, pin_memory=g.last_token_indices.is_cuda)
        g.last_token_indices.copy_(host, non_blocking=True)
