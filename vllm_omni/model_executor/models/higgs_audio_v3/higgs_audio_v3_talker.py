# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage-0 talker for higgs-audio v3 (Qwen3 backbone, fused multi-codebook).

Architecture:
- Backbone: Qwen3 (~4B, 36 layers, 2560 hidden, GQA 32/8). No DualFFN.
- Fused multi-codebook embedding: [N*V, D] weight, offset lookup, sum across N
- Fused multi-codebook head: same weight (tied), reshape to [L, N, V]
- MusicGen-style delay pattern [0,1,...,7] with BOC/EOC
- Audio feedback: replace continuation-token embedding with fused codebook embed

Weight loading maps from the HF checkpoint's prefixes:
  tied.embedding.text_embedding. -> model.embed_tokens.
  body.layers.                   -> model.layers.
  body.norm.                     -> model.norm.
  tied.head.text_head.           -> lm_head.
  tied.embedding.modality_embeddings.0.embedding. -> multimodal_embedding.
  tied.embedding.modality_embeddings.0.model.*    -> skipped (codec for code2wav)
  tied.head.modality_heads.0.*                    -> skipped when tied
"""

from __future__ import annotations

import copy
import os
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.platforms import current_platform
from vllm.v1.outputs import SamplerOutput

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.transformers_utils.configs.higgs_audio_v3 import (
    HiggsAudioV3Config,
)

try:
    from flashinfer.sampling import top_k_renorm_probs as _flashinfer_top_k_renorm_probs
    from flashinfer.sampling import top_p_renorm_probs as _flashinfer_top_p_renorm_probs
except (ImportError, RuntimeError):
    _flashinfer_top_k_renorm_probs = None
    _flashinfer_top_p_renorm_probs = None

_higgs_flashinfer_sampling_module: Any | None = None


def _get_higgs_flashinfer_sampling_module() -> Any:
    global _higgs_flashinfer_sampling_module
    if _higgs_flashinfer_sampling_module is None:
        from flashinfer.sampling import get_sampling_module

        _higgs_flashinfer_sampling_module = get_sampling_module()
    return _higgs_flashinfer_sampling_module


@torch.library.custom_op(
    "vllm_omni::higgs_flashinfer_top_k_renorm",
    mutates_args=("row_states",),
)
def _higgs_flashinfer_top_k_renorm_op(
    probs: torch.Tensor,
    row_states: torch.Tensor,
) -> torch.Tensor:
    module = _get_higgs_flashinfer_sampling_module()
    return module.top_k_renorm_probs(probs, None, 50, row_states)


@_higgs_flashinfer_top_k_renorm_op.register_fake
def _fake_higgs_flashinfer_top_k_renorm_op(
    probs: torch.Tensor,
    row_states: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(probs)


@torch.library.custom_op(
    "vllm_omni::higgs_flashinfer_top_p_renorm",
    mutates_args=("workspace",),
)
def _higgs_flashinfer_top_p_renorm_op(
    probs: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    module = _get_higgs_flashinfer_sampling_module()
    return module.top_p_renorm_probs(probs, None, 0.95, True, workspace)


@_higgs_flashinfer_top_p_renorm_op.register_fake
def _fake_higgs_flashinfer_top_p_renorm_op(
    probs: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(probs)


def _flashinfer_top_p_workspace_size(batch_rows: int, vocab_size: int) -> int:
    """Workspace bytes required by FlashInfer deterministic AIR top-p."""

    def align256(value: int) -> int:
        return ((value + 255) // 256) * 256

    buf_len = max(align256(vocab_size // 32), 256)
    return (
        align256(384 * batch_rows)
        + align256(8 * 2048 * batch_rows)
        + align256(4 * 2048 * batch_rows)
        + align256(4 * buf_len * batch_rows)
        + align256(4 * buf_len * batch_rows)
    )


def _resolve_audio_runtime_mode(
    env_var: str,
    config_value: str,
    allowed: set[str],
) -> str:
    """Resolve a runtime mode with an explicit environment override."""
    raw_value = os.getenv(env_var)
    value = str(config_value if raw_value is None else raw_value).strip().lower()
    if value not in allowed:
        choices = "/".join(sorted(allowed))
        raise ValueError(f"{env_var} must be one of {choices}, got {value!r}")
    return value


__all__ = ["HiggsAudioV3PendingStep", "HiggsAudioV3TalkerForConditionalGeneration"]

logger = init_logger(__name__)


# Delay pattern constants
BOC_ID = 1024  # beginning of codebook
EOC_ID = 1025  # end of codebook

# Checkpoint prefix mapping
_BACKBONE_PREFIX_MAP = {
    "tied.embedding.text_embedding.": "model.embed_tokens.",
    "body.layers.": "model.layers.",
    "body.norm.": "model.norm.",
    "tied.head.text_head.": "lm_head.",
}
_MODALITY_EMBEDDING_PREFIX = "tied.embedding.modality_embeddings.0.embedding."
_MODALITY_HEAD_PREFIX = "tied.head.modality_heads.0."
_CODEC_PREFIX = "tied.embedding.modality_embeddings.0.model."


class _HiggsAudioTrajectoryRecorder:
    """Opt-in request trajectory JSONL recorder for determinism diagnosis."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = os.fspath(path)
        self._lock = threading.Lock()
        self._next_step_by_id: dict[str, int] = {}

    def record(
        self,
        request_ids: tuple[str, ...],
        request_seeds: tuple[int | None, ...],
        sampling_seeds: tuple[int | None, ...],
        sampling_post_steps: tuple[int | None, ...],
        hidden_snapshot: torch.Tensor | None,
        host_staging: torch.Tensor,
        valid: list[int],
        done: list[int],
        num_codebooks: int,
    ) -> None:
        import hashlib
        import json

        lines: list[str] = []
        hidden_cpu = None
        if hidden_snapshot is not None:
            hidden_cpu = hidden_snapshot.detach().to(device="cpu").contiguous()
        with self._lock:
            snapshots = zip(
                request_ids,
                request_seeds,
                sampling_seeds,
                sampling_post_steps,
                strict=True,
            )
            for row, (req_id, seed, sampling_seed, sampling_post_step) in enumerate(snapshots):
                is_valid = bool(valid[row])
                step = self._next_step_by_id.get(req_id, 0)
                sampling_decode_step = sampling_post_step
                if sampling_decode_step is not None and is_valid:
                    sampling_decode_step -= 1
                codes_hash = None
                hidden_hash = None
                if hidden_cpu is not None:
                    hidden_row = hidden_cpu[row].contiguous()
                    hidden_bytes = hidden_row.view(torch.uint8)
                    hidden_hash = hashlib.sha256(hidden_bytes.numpy().tobytes()).hexdigest()
                if is_valid:
                    codes = host_staging[row, :num_codebooks].detach().cpu().contiguous()
                    codes_hash = hashlib.sha256(codes.numpy().tobytes()).hexdigest()
                    self._next_step_by_id[req_id] = step + 1
                lines.append(
                    json.dumps(
                        {
                            "request_id": req_id,
                            "effective_seed": seed,
                            "decode_step": step,
                            "sampling_seed": sampling_seed,
                            "sampling_decode_step": sampling_decode_step,
                            "sampling_post_step": sampling_post_step,
                            "hidden_hash": hidden_hash,
                            "sampled_codes_hash": codes_hash,
                            "valid": is_valid,
                            "done": bool(done[row]),
                        },
                        sort_keys=True,
                    )
                )
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as trace_file:
                trace_file.write("\n".join(lines) + "\n")


def _create_higgs_audio_trajectory_recorder() -> _HiggsAudioTrajectoryRecorder | None:
    path = os.getenv("HIGGS_AUDIO_V3_TRAJECTORY_PATH")
    return _HiggsAudioTrajectoryRecorder(path) if path else None


class _HiggsAudioHostStagingLease:
    """Exclusive ownership of one ping-pong host staging slot."""

    def __init__(self, pool: _HiggsAudioHostStagingPool, slot: int, tensor: torch.Tensor):
        self._pool = pool
        self._slot = slot
        self.tensor = tensor
        self._released = False

    def cuda_event(self) -> torch.cuda.Event:
        return self._pool.cuda_event(self._slot)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._pool.release(self._slot)


class _HiggsAudioHostStagingPool:
    """Two pinned host buffers whose leases survive lagged async resolve."""

    def __init__(self, *, width: int, pin_memory: bool):
        self.width = int(width)
        self.pin_memory = bool(pin_memory)
        self._buffers: list[torch.Tensor | None] = [None, None]
        self._events: list[torch.cuda.Event | None] = [None, None]
        self._in_use = [False, False]
        self._cursor = 0
        self._condition = threading.Condition()

    def acquire(self, num_rows: int) -> _HiggsAudioHostStagingLease:
        num_rows = max(0, int(num_rows))
        with self._condition:
            slot = self._cursor
            self._cursor ^= 1
            while self._in_use[slot]:
                self._condition.wait()
            self._in_use[slot] = True
            buf = self._buffers[slot]
            if buf is None or int(buf.shape[0]) < num_rows:
                rows = max(num_rows, 16)
                buf = torch.empty(
                    (rows, self.width),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
                self._buffers[slot] = buf
            return _HiggsAudioHostStagingLease(self, slot, buf[:num_rows])

    def cuda_event(self, slot: int) -> torch.cuda.Event:
        with self._condition:
            event = self._events[slot]
            if event is None:
                event = torch.cuda.Event(blocking=True)
                self._events[slot] = event
            return event

    def release(self, slot: int) -> None:
        with self._condition:
            self._in_use[slot] = False
            self._condition.notify_all()


@dataclass(frozen=True)
class HiggsAudioV3PendingStep:
    """Immutable launch-time request mapping plus an owned D2H snapshot."""

    request_ids: tuple[str, ...]
    request_id_to_row: Mapping[str, int]
    host_staging: torch.Tensor
    event: Any
    num_codebooks: int
    request_seeds: tuple[int | None, ...]
    sampling_seeds: tuple[int | None, ...]
    sampling_post_steps: tuple[int | None, ...]
    hidden_snapshot: torch.Tensor | None
    _release: Callable[[], None]
    _on_resolve: Callable[[tuple[str, ...], list[int], list[int]], None] | None
    _trajectory_recorder: _HiggsAudioTrajectoryRecorder | None

    @classmethod
    def create(
        cls,
        *,
        request_ids: Sequence[str],
        host_staging: torch.Tensor,
        event: Any,
        num_codebooks: int,
        release: Callable[[], None],
        on_resolve: Callable[[tuple[str, ...], list[int], list[int]], None] | None = None,
        request_seeds: Sequence[int | None] | None = None,
        sampling_seeds: Sequence[int | None] | None = None,
        sampling_post_steps: Sequence[int | None] | None = None,
        hidden_snapshot: torch.Tensor | None = None,
        trajectory_recorder: _HiggsAudioTrajectoryRecorder | None = None,
    ) -> HiggsAudioV3PendingStep:
        ids = tuple(str(req_id) for req_id in request_ids)
        if len(ids) != int(host_staging.shape[0]):
            release()
            raise ValueError(
                "Higgs pending audio rows must match the launch-time request snapshot: "
                f"{int(host_staging.shape[0])} rows for {len(ids)} requests"
            )
        seed_snapshot = tuple(request_seeds) if request_seeds is not None else (None,) * len(ids)
        if len(seed_snapshot) != len(ids):
            release()
            raise ValueError(
                "Higgs pending audio seeds must match the launch-time request snapshot: "
                f"{len(seed_snapshot)} seeds for {len(ids)} requests"
            )
        sampling_seed_snapshot = tuple(sampling_seeds) if sampling_seeds is not None else seed_snapshot
        sampling_post_step_snapshot = (
            tuple(sampling_post_steps) if sampling_post_steps is not None else (None,) * len(ids)
        )
        if len(sampling_seed_snapshot) != len(ids) or len(sampling_post_step_snapshot) != len(ids):
            release()
            raise ValueError("Higgs pending audio sampling state must match the launch-time request snapshot")
        if hidden_snapshot is not None and int(hidden_snapshot.shape[0]) != len(ids):
            release()
            raise ValueError("Higgs pending hidden rows must match the launch-time request snapshot")
        immutable_hidden = hidden_snapshot.detach().clone() if hidden_snapshot is not None else None
        return cls(
            request_ids=ids,
            request_id_to_row=MappingProxyType({req_id: row for row, req_id in enumerate(ids)}),
            host_staging=host_staging,
            event=event,
            num_codebooks=int(num_codebooks),
            request_seeds=seed_snapshot,
            sampling_seeds=sampling_seed_snapshot,
            sampling_post_steps=sampling_post_step_snapshot,
            hidden_snapshot=immutable_hidden,
            _release=release,
            _on_resolve=on_resolve,
            _trajectory_recorder=trajectory_recorder,
        )

    def release(self) -> None:
        self._release()

    def resolve(self) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        """Wait for this step only and materialize request-keyed audio rows."""
        try:
            if self.event is not None:
                self.event.synchronize()
            valid = self.host_staging[:, self.num_codebooks].tolist()
            # Read done while this lease is held. It is intentionally retained
            # in the immutable snapshot even though downstream only consumes
            # valid audio rows; finish/cancel filtering is runner-owned.
            done = self.host_staging[:, self.num_codebooks + 1].tolist()
            if self._trajectory_recorder is not None:
                self._trajectory_recorder.record(
                    self.request_ids,
                    self.request_seeds,
                    self.sampling_seeds,
                    self.sampling_post_steps,
                    self.hidden_snapshot,
                    self.host_staging,
                    valid,
                    done,
                    self.num_codebooks,
                )
            if self._on_resolve is not None:
                self._on_resolve(self.request_ids, valid, done)
            resolved: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
            for req_id in self.request_ids:
                row = self.request_id_to_row[req_id]
                if int(valid[row]) == 0:
                    resolved[req_id] = {
                        "codes": {
                            "audio": torch.empty(0, dtype=torch.int32),
                        }
                    }
                    continue
                resolved[req_id] = {
                    "codes": {
                        # The lease is released before the caller consumes the
                        # payload, so detach this row from the reusable slot.
                        "audio": self.host_staging[row : row + 1, : self.num_codebooks].clone(),
                    }
                }
            return resolved
        finally:
            self.release()


class HiggsFusedMultiTextEmbedding(nn.Module):
    """Fused multi-codebook embedding: [N*V, D] weight + offset lookup."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.register_buffer(
            "_offsets",
            torch.arange(num_codebooks, dtype=torch.long) * vocab_size,
            persistent=False,
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        fused_ids = codes + self._offsets
        return F.embedding(fused_ids, self.weight).sum(dim=-2)


class HiggsFusedMultiTextHead(nn.Module):
    """Fused multi-codebook head: [L, D] -> [L, N, V] via one linear."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_codebooks * vocab_size, hidden_size))
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size

    def generate(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, self.weight)
        return logits.reshape(hidden.shape[0], self.num_codebooks, self.vocab_size)


class HiggsAudioV3TalkerForConditionalGeneration(nn.Module):
    """Stage-0 talker for higgs-audio v3.

    Wraps Qwen3Model backbone + fused multi-codebook modules for TTS generation
    with MusicGen-style delay pattern sampling and audio feedback embedding.
    """

    # Tell the AR runner to call model.sample() instead of the stock sampler.
    prefer_model_sampler: bool = True
    # Request-certified audio rows consume hidden states directly; the runner
    # may therefore call sample() with no text-vocabulary logits.
    supports_logits_free_model_sampler: bool = True
    supports_force_text_logits: bool = True
    # Tell the runner to call postprocess() to emit per-step audio codes.
    have_multimodal_outputs: bool = True
    has_postprocess: bool = True
    skips_model_sampler_output_token_history: bool = True
    postprocess_uses_hidden_states: bool = False
    postprocess_uses_multimodal_outputs: bool = False
    postprocess_uses_req_infos: bool = False
    supports_omni_query_start_loc: bool = True
    requires_request_ids_for_decode_state: bool = True
    # Stage 1 consumes request-local codes.audio only. Publish those tensors
    # before the async output snapshot and omit the unused talker hidden state.
    use_async_omni_output: bool = True
    eager_omni_postprocess_before_async_output: bool = False
    supports_async_omni_postprocess: bool = True
    omni_pooler_payload_include_hidden: bool = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, HiggsAudioV3Config):
            self.config = hf_config
        else:
            self.config = HiggsAudioV3Config(**hf_config.to_dict())
        model_config = getattr(vllm_config, "model_config", None)
        compilation_config = getattr(vllm_config, "compilation_config", None)
        cudagraph_mode = getattr(compilation_config, "cudagraph_mode", None)
        mode_name = getattr(cudagraph_mode, "name", str(cudagraph_mode)).upper()
        self._use_external_decode_cudagraph = (
            not bool(getattr(model_config, "enforce_eager", True))
            and cudagraph_mode is not None
            and "NONE" not in mode_name
        )
        if self._use_external_decode_cudagraph:
            self.config.enable_mlp_cudagraph = False
        assert not (self._use_external_decode_cudagraph and self.config.enable_mlp_cudagraph), (
            "Higgs Audio v3 local MLP graph and external decode CUDA graph are mutually exclusive"
        )

        self.vllm_config = vllm_config
        self.num_codebooks = int(self.config.num_codebooks)
        self.codebook_size = int(self.config.codebook_size)
        hidden_size = int(self.config.audio_hidden_size)
        self.tie_modality = self.config.tie_modality_embeddings

        # Fused multi-codebook modules
        self.multimodal_embedding = HiggsFusedMultiTextEmbedding(self.num_codebooks, self.codebook_size, hidden_size)
        self.modality_head = HiggsFusedMultiTextHead(self.num_codebooks, self.codebook_size, hidden_size)
        if self.tie_modality:
            self.modality_head.weight = self.multimodal_embedding.weight

        # Qwen3 backbone
        self._backbone_config = self.config.text_config
        backbone_vllm_config = copy.copy(vllm_config)
        backbone_model_config = copy.copy(vllm_config.model_config)
        backbone_model_config.hf_config = self._backbone_config
        backbone_vllm_config.model_config = backbone_model_config

        self.model = Qwen3Model(
            vllm_config=backbone_vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        if self.config.enable_flashinfer_api_unwrap:
            self._maybe_unwrap_flashinfer_api_wrappers()

        if self._backbone_config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self._backbone_config.vocab_size,
                self._backbone_config.hidden_size,
                prefix=f"{prefix}.lm_head" if prefix else "lm_head",
            )

        self.logits_processor = LogitsProcessor(self._backbone_config.vocab_size)
        self._text_vocab_size = int(self._backbone_config.vocab_size)

        # Audio continuation token ID — resolved lazily from tokenizer.
        # This is the <|audio|> token that serves as the LM-level continuation
        # marker during audio generation (equivalent to v2's audio_token_id).
        self._audio_continuation_id: int | None = None
        self._eos_token_id: int | None = None
        self._resolved_tokens = False

        self._last_logits_hidden: torch.Tensor | None = None
        self._last_step_input_ids: torch.Tensor | None = None
        self._last_step_query_start_loc: torch.Tensor | None = None
        self._last_step_query_start_loc_buffer: torch.Tensor | None = None
        self.supports_omni_decode_step_metadata = True
        self._decode_step_metadata_from_runner = False
        self._last_audio_codes: torch.Tensor | None = None
        self._last_audio_code_valid: list[bool] = []
        self._postprocess_cursor: int = 0
        self._last_audio_codes_buffer: torch.Tensor | None = None
        self._last_audio_host_staging: torch.Tensor | None = None
        self._last_audio_gpu_staging: torch.Tensor | None = None
        self._last_audio_staging_event: torch.cuda.Event | None = None
        self._last_audio_valid_flags: list[int] | None = None
        self._last_audio_done_flags: list[int] | None = None
        self._trajectory_hidden_snapshot: torch.Tensor | None = None
        self._audio_host_staging_pool = _HiggsAudioHostStagingPool(
            width=self.num_codebooks + 2,
            pin_memory=torch.cuda.is_available(),
        )
        self._last_audio_host_lease: _HiggsAudioHostStagingLease | None = None
        self._row_index_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._codebook_index_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._boc_frame_cache: dict[tuple[str, int], torch.Tensor] = {}
        self._fast_audio_direct_request_ids: set[str] = set()
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        self._mlp_cudagraph_max_batch = max(1, int(getattr(scheduler_config, "max_num_seqs", 16)))
        self._mlp_graphs: dict[tuple[int, int], dict[str, Any]] = {}
        self._mlp_graph_disabled: set[tuple[int, int]] = set()
        self._audio_state_step_mode = _resolve_audio_runtime_mode(
            "HIGGS_AUDIO_V3_AUDIO_STATE_STEP",
            self.config.audio_state_step_mode,
            {"legacy", "eager", "compile"},
        )
        self._audio_sampler_mode = _resolve_audio_runtime_mode(
            "HIGGS_AUDIO_V3_AUDIO_SAMPLER",
            self.config.audio_sampler_mode,
            {"torch", "flashinfer"},
        )
        if self._audio_sampler_mode == "flashinfer" and (
            _flashinfer_top_k_renorm_probs is None or _flashinfer_top_p_renorm_probs is None
        ):
            raise RuntimeError("audio_sampler_mode=flashinfer requires FlashInfer sampling kernels")
        if self._audio_sampler_mode == "flashinfer":
            _get_higgs_flashinfer_sampling_module()
        logger.info_once(
            "Higgs Audio V3 audio runtime: "
            f"state_step={self._audio_state_step_mode}, sampler={self._audio_sampler_mode}"
        )
        self._compiled_audio_state_step: Any | None = None
        self._audio_state_hidden_staging: torch.Tensor | None = None

        # Pre-allocated decode-step audio feedback buffers (CUDA-graph safe).
        # Populated by sample(), read by forward() via torch.where (no dict).
        max_bs = max(64, self._mlp_cudagraph_max_batch)
        self.register_buffer(
            "_decode_last_codes",
            torch.zeros(max_bs, self.num_codebooks, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("_decode_has_codes", torch.zeros(max_bs, dtype=torch.bool), persistent=False)
        # Keep delay/ramp counters as int64 to match arange/codebook-index
        # arithmetic in the decode hot path without per-step dtype casts.
        self.register_buffer("_decode_delay_count", torch.zeros(max_bs, dtype=torch.long), persistent=False)
        self.register_buffer("_decode_eoc_countdown", torch.full((max_bs,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("_decode_generation_done", torch.zeros(max_bs, dtype=torch.bool), persistent=False)
        self.register_buffer("_decode_seed", torch.full((max_bs,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("_decode_step_count", torch.zeros(max_bs, dtype=torch.long), persistent=False)
        self.register_buffer("_audio_state_update_mask", torch.zeros(max_bs, dtype=torch.bool), persistent=False)
        if self._audio_sampler_mode == "flashinfer":
            flashinfer_batch_rows = self._mlp_cudagraph_max_batch * self.num_codebooks
            top_p_workspace_size = _flashinfer_top_p_workspace_size(
                flashinfer_batch_rows,
                self.codebook_size,
            )
            top_k_row_states = torch.zeros(1024 * 1024, dtype=torch.uint8)
            top_p_workspace = torch.empty(top_p_workspace_size, dtype=torch.uint8)
        else:
            top_k_row_states = torch.empty(0, dtype=torch.uint8)
            top_p_workspace = torch.empty(0, dtype=torch.uint8)
        self.register_buffer(
            "_flashinfer_top_k_row_states",
            top_k_row_states,
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_top_p_workspace",
            top_p_workspace,
            persistent=False,
        )
        self._decode_active_audio_count: int = 0

        # Async scheduling may shrink or reorder the active batch while audio
        # delay state is still live. Keep request-owned state in stable pool
        # rows and expose contiguous active rows to the existing graph-safe
        # decode path. The pool is synchronized only when the request layout
        # changes; the steady-state decode loop pays no gather/scatter cost.
        self.register_buffer(
            "_request_decode_last_codes",
            torch.zeros(max_bs, self.num_codebooks, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer("_request_decode_has_codes", torch.zeros(max_bs, dtype=torch.bool), persistent=False)
        self.register_buffer("_request_decode_delay_count", torch.zeros(max_bs, dtype=torch.long), persistent=False)
        self.register_buffer(
            "_request_decode_eoc_countdown",
            torch.full((max_bs,), -1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "_request_decode_generation_done",
            torch.zeros(max_bs, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer("_request_decode_seed", torch.full((max_bs,), -1, dtype=torch.long), persistent=False)
        self.register_buffer("_request_decode_step_count", torch.zeros(max_bs, dtype=torch.long), persistent=False)
        self.register_buffer("_active_request_pool_rows", torch.empty(0, dtype=torch.long), persistent=False)
        self._request_state_row_by_id: dict[str, int] = {}
        self._request_state_free_rows: list[int] = []
        self._request_state_next_row: int = 0
        self._active_request_ids: tuple[str, ...] = ()
        self._request_sampling_seed_by_id: dict[str, int] = {}
        self._trajectory_recorder = _create_higgs_audio_trajectory_recorder()

        # PrefixCache compatibility hooks (mirror qwen3_tts pattern):
        # 1. The talker only consumes the last token's hidden state, so the
        #    runner can skip the per-step full hidden-state GPU->CPU merge
        #    that PrefixCache otherwise does.
        # 2. Per-step ``codes.audio`` rows stay GPU-resident; defer the CPU
        #    write of the prefix-cache mm-output copy to request finish so
        #    the per-step bookkeeping does not block batching.
        # Request-aware pending postprocess overlap still requires prefix
        # caching to be disabled; deploy profiles enforce that constraint.
        self.requires_full_prefix_cached_hidden_states = False
        self.deferred_prefix_cache_mm_keys = {"codes.audio"}

    def _maybe_unwrap_flashinfer_api_wrappers(self) -> None:
        """Remove FlashInfer trace wrappers only from wrappers owned by this model."""
        patched: list[str] = []
        saw_flashinfer = False
        seen: set[int] = set()
        stack: list[Any] = []
        for layer in getattr(self.model, "layers", []):
            impl = getattr(getattr(getattr(layer, "self_attn", None), "attn", None), "impl", None)
            if impl is not None:
                stack.append(impl)

        while stack and len(seen) < 512:
            obj = stack.pop()
            obj_id = id(obj)
            if obj_id in seen:
                continue
            seen.add(obj_id)

            cls = obj.__class__
            if "flashinfer" in cls.__module__.lower():
                saw_flashinfer = True
                for method_name in ("run", "paged_run", "plan"):
                    method = getattr(obj, method_name, None)
                    original = getattr(method, "__wrapped__", None)
                    if original is None:
                        continue
                    try:
                        bound = original.__get__(obj, cls) if hasattr(original, "__get__") else original
                        setattr(obj, method_name, bound)
                    except Exception:
                        continue
                    patched.append(f"{cls.__name__}.{method_name}")

            obj_dict = getattr(obj, "__dict__", None)
            if not isinstance(obj_dict, dict):
                continue
            for value in obj_dict.values():
                if isinstance(value, dict):
                    stack.extend(value.values())
                elif isinstance(value, (list, tuple, set)):
                    stack.extend(value)
                elif not isinstance(value, (bool, int, float, str, bytes, type(None), torch.Tensor)):
                    stack.append(value)

        if patched:
            logger.info("HiggsAudioV3Talker: unwrapped FlashInfer API wrappers: %s", sorted(set(patched)))
        elif saw_flashinfer:
            logger.warning(
                "HiggsAudioV3Talker: FlashInfer backend was detected but no API wrappers were unwrapped; "
                "FlashInfer internals may have changed and wrapper-unwrapping performance benefits may be unavailable."
            )

    def _resolve_token_ids(self) -> None:
        """Resolve <|audio|> and eos token IDs.

        Prefers config's pre-resolved IDs (from ``resolve_special_tokens()``),
        falls back to loading the HF tokenizer directly.
        """
        if self._resolved_tokens:
            return
        self._resolved_tokens = True

        # Try config first (populated by resolve_special_tokens or from_pretrained)
        cfg_audio = getattr(self.config, "audio_continuation_id", None)
        cfg_eos = getattr(self.config, "eos_token_id", None)
        if cfg_audio is not None:
            self._audio_continuation_id = int(cfg_audio)
        if cfg_eos is not None:
            self._eos_token_id = int(cfg_eos)

        if self._audio_continuation_id is not None:
            logger.info(
                "Resolved v3 token IDs from config: audio_continuation=%s, eos=%s",
                self._audio_continuation_id,
                self._eos_token_id,
            )
            return

        # Fallback: load tokenizer directly
        model_path = getattr(self.vllm_config.model_config, "model", None)
        if model_path is None:
            return
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            vocab = dict(tokenizer.get_added_vocab())
            if "<|audio|>" in vocab:
                self._audio_continuation_id = vocab["<|audio|>"]
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                self._eos_token_id = int(tokenizer.eos_token_id)
            logger.info(
                "Resolved v3 token IDs from tokenizer: audio_continuation=%s, eos=%s",
                self._audio_continuation_id,
                self._eos_token_id,
            )
        except Exception as exc:
            logger.warning("Failed to resolve token IDs from tokenizer: %s", exc)

    def update_decode_step_metadata(
        self,
        *,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        omni_query_start_loc: torch.Tensor | None = None,
        req_ids: list[str] | tuple[str, ...] | None = None,
        **_: Any,
    ) -> None:
        """Update per-step metadata before runner forward or CUDA graph replay."""
        _ = positions, inputs_embeds
        if input_ids is not None:
            self._last_step_input_ids = input_ids
            if self._use_external_decode_cudagraph and input_ids.is_cuda and input_ids.ndim >= 1:
                self._ensure_decode_state_capacity(int(input_ids.shape[0]), input_ids.device)
            if req_ids is not None:
                self._prepare_request_state(req_ids, input_ids.device)
        self._set_last_step_query_start_loc(omni_query_start_loc)
        self._decode_step_metadata_from_runner = True

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        metadata_from_runner = self._decode_step_metadata_from_runner
        self._decode_step_metadata_from_runner = False

        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information")

        if inputs_embeds is None:
            # Mask -100 placeholders to 0 before embedding. Use torch.where
            # (no Python data-dependent branch) so this is CUDA-graph safe.
            safe_ids = torch.where(input_ids < 0, torch.zeros_like(input_ids), input_ids)
            hidden_states = self.model.embed_tokens(safe_ids)
        else:
            hidden_states = inputs_embeds

        if input_ids is not None and not metadata_from_runner:
            self._last_step_input_ids = input_ids

        # Stash query_start_loc for audio-state row mapping and max_query_len
        # for prefill detection. Some attention backends do not expose
        # query_start_loc, so prefer backend metadata when available and
        # otherwise use the runner-supplied buffer.
        max_query_len = None
        if not metadata_from_runner:
            try:
                fallback_qsl = kwargs.get("omni_query_start_loc")
                from vllm.forward_context import get_forward_context

                attn_metadata = get_forward_context().attn_metadata
                if isinstance(attn_metadata, dict) and attn_metadata:
                    attn = next(iter(attn_metadata.values()))
                else:
                    attn = attn_metadata
                qsl = getattr(attn, "query_start_loc", None)
                self._set_last_step_query_start_loc(qsl if isinstance(qsl, torch.Tensor) else fallback_qsl)
                max_query_len = getattr(attn, "max_query_len", None)
            except Exception:
                self._last_step_query_start_loc = None

        # Prefill-only operations: ref audio substitution and audio feedback
        # require Python dict/list ops that break CUDA graph capture.
        # Prefer max_query_len where available: concurrent decode has
        # input_ids.numel() > 1 but max_query_len == 1.
        if max_query_len is not None:
            is_prefill = int(max_query_len) > 1
        else:
            is_prefill = (
                input_ids is not None
                and inputs_embeds is None
                and not self._is_single_token_decode_step(int(hidden_states.shape[0]))
            )
        if is_prefill and info_dicts:
            # Voice clone: replace -100 placeholder positions with ref audio embeddings
            hidden_states = self._apply_ref_audio_substitution(hidden_states, input_ids, info_dicts)

        # Audio feedback at decode: replace continuation token embeddings
        if input_ids is not None and inputs_embeds is None:
            hidden_states = self._apply_audio_feedback(hidden_states, input_ids)

        if self.config.enable_mlp_cudagraph:
            return self._run_qwen3_layers_mlp_graph(positions, hidden_states)

        return self._run_qwen3_layers_eager(positions, hidden_states)

    def _run_qwen3_layers_eager(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._use_external_decode_cudagraph:
            # Qwen3Model carries the vLLM support_torch_compile decorator.
            # Calling it here keeps the audio-feedback embedding outside the
            # backbone while allowing the transformer to be compiled before
            # the external FULL_DECODE CUDA graph captures it.
            return self.model(
                input_ids=None,
                positions=positions,
                inputs_embeds=hidden_states,
            )

        residual: torch.Tensor | None = None
        for layer in self.model.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        norm_out = self.model.norm(hidden_states, residual)
        if isinstance(norm_out, tuple):
            norm_out = norm_out[0]
        return norm_out

    def _run_qwen3_layers_mlp_graph(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual: torch.Tensor | None = None
        batch = int(hidden_states.shape[0]) if hidden_states.ndim >= 1 else 0
        use_graph = (
            torch.cuda.is_available()
            and hidden_states.is_cuda
            and hidden_states.ndim == 2
            and positions.ndim == 1
            and batch > 0
            and batch <= self._mlp_cudagraph_max_batch
            and int(positions.shape[0]) == batch
            and self._is_decode_only_graph_batch(batch)
        )
        for layer_idx, layer in enumerate(self.model.layers):
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            hidden_states = layer.self_attn(
                positions=positions,
                hidden_states=hidden_states,
            )
            if use_graph:
                graph_out = self._run_qwen3_mlp_graph(layer_idx, layer, hidden_states, residual)
                if graph_out is not None:
                    hidden_states, residual = graph_out
                    continue
            hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
            hidden_states = layer.mlp(hidden_states)
        norm_out = self.model.norm(hidden_states, residual)
        if isinstance(norm_out, tuple):
            norm_out = norm_out[0]
        return norm_out

    def _run_qwen3_mlp_graph(
        self,
        layer_idx: int,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        key = (int(layer_idx), int(hidden_states.shape[0]))
        if key in self._mlp_graph_disabled:
            return None
        if hidden_states.ndim != 2 or residual.ndim != 2 or hidden_states.shape != residual.shape:
            self._mlp_graph_disabled.add(key)
            return None
        if torch.cuda.is_current_stream_capturing():
            return None

        entry = self._mlp_graphs.get(key)
        if entry is not None:
            entry["static_hidden"].copy_(hidden_states)
            entry["static_residual"].copy_(residual)
            entry["graph"].replay()
            return entry["static_output"], entry["static_residual_out"]

        try:
            static_hidden = torch.empty_like(hidden_states)
            static_residual = torch.empty_like(residual)
            static_hidden.copy_(hidden_states)
            static_residual.copy_(residual)
            graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                normed, static_residual_out = layer.post_attention_layernorm(static_hidden, static_residual)
                static_output = layer.mlp(normed)
            self._mlp_graphs[key] = {
                "graph": graph,
                "static_hidden": static_hidden,
                "static_residual": static_residual,
                "static_output": static_output,
                "static_residual_out": static_residual_out,
            }
            logger.info(
                "HiggsAudioV3Talker: captured MLP CUDA graph for layer=%d decode batch=%d",
                layer_idx,
                int(hidden_states.shape[0]),
            )
            return static_output, static_residual_out
        except Exception as exc:
            self._mlp_graph_disabled.add(key)
            logger.warning(
                "HiggsAudioV3Talker: MLP CUDA graph disabled for layer=%d batch=%d: %s",
                layer_idx,
                int(hidden_states.shape[0]),
                exc,
            )
            return None

    def _is_decode_only_graph_batch(self, batch: int) -> bool:
        q_start = self._last_step_query_start_loc
        return isinstance(q_start, torch.Tensor) and int(q_start.numel()) == batch + 1

    def _can_use_logits_free_audio_sampler(self, num_rows: int, decode_only: bool) -> bool:
        request_ids = self._active_request_ids
        return (
            decode_only
            and len(request_ids) == num_rows
            and all(req_id in self._fast_audio_direct_request_ids for req_id in request_ids)
        )

    def _mark_audio_direct_requests(
        self,
        request_ids: Sequence[str],
        valid: Sequence[int],
        done: Sequence[int],
    ) -> None:
        """Commit launch-time request IDs whose audio state is authoritative."""
        for row, req_id in enumerate(request_ids):
            req_id = str(req_id)
            # A finish/cancel notification can race a lagged pending resolve.
            # Request-owned state is the authority: once its row is released,
            # an older snapshot must not resurrect logits-free certification.
            if req_id not in self._request_state_row_by_id:
                self._fast_audio_direct_request_ids.discard(req_id)
            elif int(valid[row]) or int(done[row]):
                self._fast_audio_direct_request_ids.add(req_id)
            else:
                self._fast_audio_direct_request_ids.discard(req_id)

    @staticmethod
    def _logits_processor_requires_text_logits(processor: Any) -> bool:
        """Return whether a loaded processor has active request state.

        vLLM always constructs its builtin processors, so their mere presence
        cannot disable the logits-free path. Unknown/custom processors are
        conservative: unless they expose a recognized empty request-state
        container, preserve text logits.
        """
        recognized_state = False
        for state_name in ("min_toks", "biases", "min_p_count", "req_info"):
            if hasattr(processor, state_name):
                recognized_state = True
                if bool(getattr(processor, state_name)):
                    return True
        return not recognized_state

    @staticmethod
    def _sampling_metadata_requires_text_logits(sampling_metadata: Any) -> bool:
        if sampling_metadata is None:
            return False
        if getattr(sampling_metadata, "max_num_logprobs", None) is not None:
            return True
        if getattr(sampling_metadata, "logprob_token_ids", None):
            return True
        if getattr(sampling_metadata, "allowed_token_ids_mask", None) is not None:
            return True
        if getattr(sampling_metadata, "bad_words_token_ids", None):
            return True
        if not bool(getattr(sampling_metadata, "no_penalties", True)):
            return True

        logitsprocs = getattr(sampling_metadata, "logitsprocs", None)
        if logitsprocs is not None:
            processors = getattr(logitsprocs, "all", None)
            if processors is None:
                return True
            if callable(processors):
                processors = processors()
            if any(
                HiggsAudioV3TalkerForConditionalGeneration._logits_processor_requires_text_logits(processor)
                for processor in processors
            ):
                return True

        holder = getattr(sampling_metadata, "thinking_budget_state_holder", None)
        if holder is not None:
            has_tracked_requests = getattr(holder, "has_tracked_requests", None)
            if not callable(has_tracked_requests) or has_tracked_requests():
                return True
        return False

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: Any = None,
        force_text_logits: bool = False,
    ) -> torch.Tensor | None:
        self._last_logits_hidden = hidden_states
        num_rows = int(hidden_states.shape[0])
        ids = getattr(self, "_last_step_input_ids", None)
        decode_only = isinstance(ids, torch.Tensor) and int(ids.numel()) == num_rows
        if (
            self._can_use_logits_free_audio_sampler(
                num_rows,
                decode_only,
            )
            and not force_text_logits
            and not self._sampling_metadata_requires_text_logits(sampling_metadata)
        ):
            return None
        return self.logits_processor(self.lm_head, hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        safe_ids = torch.where(input_ids < 0, torch.zeros_like(input_ids), input_ids)
        text_embed = self.model.embed_tokens(safe_ids)
        return self._apply_audio_feedback(text_embed, input_ids)

    def _set_last_step_query_start_loc(self, query_start_loc: Any) -> None:
        if not isinstance(query_start_loc, torch.Tensor):
            self._last_step_query_start_loc = None
            return
        source = query_start_loc.detach()
        numel = int(source.numel())
        buf = self._last_step_query_start_loc_buffer
        if buf is None or buf.device != source.device or buf.dtype != source.dtype or int(buf.numel()) < numel:
            buf = torch.empty(max(numel, 17), dtype=source.dtype, device=source.device)
            self._last_step_query_start_loc_buffer = buf
        buf[:numel].copy_(source.reshape(-1))
        self._last_step_query_start_loc = buf[:numel]

    # ------------------------------------------------------------------ ref audio substitution
    def _apply_ref_audio_substitution(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        info_dicts: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        """Replace -100 placeholder positions with fused multi-codebook embeddings
        of the delay-pattern-encoded reference audio codes.

        Called at prefill to inject voice clone reference. ``info_dicts`` is a
        list of per-request dicts from ``model_intermediate_buffer``, each
        containing ``audio_input_ids`` ([T, N] delayed codes) and
            ``audio_input_ids_mask`` ([T] bool mask).
        """
        if not info_dicts:
            return hidden_states

        PLACEHOLDER = -100
        flat_ids = input_ids.reshape(-1)
        placeholder_mask = flat_ids == PLACEHOLDER
        if not placeholder_mask.any():
            return hidden_states

        # Use query_start_loc to map placeholders to per-request spans
        q_start = self._last_step_query_start_loc
        if not isinstance(q_start, torch.Tensor) or q_start.numel() < 2:
            # Fallback: single-request batch
            q_start_list = [0, int(flat_ids.numel())]
        else:
            q_start_list = q_start.detach().to("cpu").tolist()

        new_hidden: torch.Tensor | None = None
        num_requests = min(len(info_dicts), len(q_start_list) - 1)

        for i in range(num_requests):
            info = info_dicts[i]
            if not isinstance(info, dict):
                continue

            codes = info.get("audio_input_ids")
            mask = info.get("audio_input_ids_mask")

            # Handle msgspec serialization (may be list-wrapped)
            if isinstance(codes, list):
                codes = codes[0] if codes else None
            if isinstance(mask, list):
                mask = mask[0] if mask else None
            if not isinstance(codes, torch.Tensor):
                continue

            # codes shape: [T, num_codebooks] delayed reference codes
            if codes.ndim == 3:
                codes = codes[0]
            if codes.ndim != 2:
                continue

            if isinstance(mask, torch.Tensor):
                if mask.ndim == 2:
                    mask = mask[0]
                codes = codes[mask.to(dtype=torch.bool)]

            if codes.numel() == 0:
                continue

            # Find placeholder positions in this request's span
            s = int(q_start_list[i])
            e = int(q_start_list[i + 1])
            if e - s <= 1:
                continue  # Decode step, skip

            span_mask = placeholder_mask[s:e]
            placeholders = span_mask.nonzero(as_tuple=True)[0]
            n_codes = int(codes.shape[0])

            if int(placeholders.numel()) < n_codes:
                continue  # Mismatch

            # Embed delayed codes via fused multi-codebook embedding
            target = placeholders[:n_codes] + s
            codes_device = codes.to(device=hidden_states.device, dtype=torch.long)
            embeds = self.multimodal_embedding(codes_device)  # [n_codes, hidden]

            if new_hidden is None:
                new_hidden = hidden_states.clone()
            flat_hidden = new_hidden.reshape(-1, new_hidden.shape[-1])
            flat_hidden[target] = embeds.to(new_hidden.dtype)

        return new_hidden if new_hidden is not None else hidden_states

    # ------------------------------------------------------------------ audio feedback
    def _apply_audio_feedback(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace decode-step embeddings with audio feedback from pre-allocated buffers.

        CUDA-graph safe: reads from pre-allocated _decode_last_codes and
        _decode_has_codes tensors using torch.where (no Python dict lookup).
        The buffers are populated by sample() after each step.

        For decode steps (1 token per request): position i maps to row i.
        For prefill: audio feedback is not needed (ref audio substitution
        handles the prefill path separately).
        """
        num_tokens = int(hidden_states.shape[0])
        external_decode_graph = bool(getattr(self, "_use_external_decode_cudagraph", False))
        if self._decode_active_audio_count == 0 and not external_decode_graph:
            return hidden_states

        dense_decode = self._is_single_token_decode_step(num_tokens)
        if dense_decode:
            req_rows = torch.arange(num_tokens, dtype=torch.long, device=hidden_states.device)
            token_positions = req_rows
        else:
            req_rows, token_positions = self._decode_request_token_positions(num_tokens, hidden_states.device)
            if int(req_rows.numel()) == 0:
                return hidden_states

        num_reqs = self._step_request_count(num_tokens)
        self._ensure_decode_state_capacity(num_reqs, hidden_states.device)

        if dense_decode:
            codes_slice = self._decode_last_codes[:num_tokens]
            has_codes = self._decode_has_codes[:num_tokens].unsqueeze(-1)
            current_hidden = hidden_states
        else:
            codes_slice = self._decode_last_codes.index_select(0, req_rows)
            has_codes = self._decode_has_codes.index_select(0, req_rows).unsqueeze(-1)
            current_hidden = hidden_states.index_select(0, token_positions)
        audio_embeds = self.multimodal_embedding(codes_slice)  # [bs, D]
        audio_embeds = audio_embeds.to(dtype=hidden_states.dtype)
        replaced = torch.where(has_codes, audio_embeds, current_hidden)

        if dense_decode:
            return replaced
        new_hidden = hidden_states.clone()
        new_hidden.index_copy_(0, token_positions, replaced)
        return new_hidden

    def _is_single_token_decode_step(self, num_rows: int) -> bool:
        q_start = self._last_step_query_start_loc
        if isinstance(q_start, torch.Tensor):
            return int(q_start.numel()) == int(num_rows) + 1
        ids = getattr(self, "_last_step_input_ids", None)
        if not isinstance(ids, torch.Tensor) or int(ids.numel()) != int(num_rows):
            return False
        if bool(getattr(self, "_use_external_decode_cudagraph", False)):
            return True
        return int(num_rows) == 1

    def _step_request_count(self, num_tokens: int) -> int:
        q_start = self._last_step_query_start_loc
        if isinstance(q_start, torch.Tensor) and int(q_start.numel()) >= 2:
            return int(q_start.numel()) - 1
        return int(num_tokens)

    def _decode_request_token_positions(
        self,
        num_tokens: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_start = self._last_step_query_start_loc
        if isinstance(q_start, torch.Tensor) and int(q_start.numel()) >= 2:
            q = q_start.to(device=device, dtype=torch.long)
            starts = q[:-1]
            spans = q[1:] - q[:-1]
            req_rows = torch.arange(int(q.numel()) - 1, dtype=torch.long, device=device)
            decode_mask = (spans == 1) & (starts >= 0) & (starts < int(num_tokens))
            return req_rows[decode_mask], starts[decode_mask]

        ids = getattr(self, "_last_step_input_ids", None)
        if isinstance(ids, torch.Tensor) and int(ids.numel()) == int(num_tokens):
            if bool(getattr(self, "_use_external_decode_cudagraph", False)):
                rows = torch.arange(int(num_tokens), dtype=torch.long, device=device)
                return rows, rows
            if int(num_tokens) == 1:
                row = torch.zeros(1, dtype=torch.long, device=device)
                return row, row
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    def _prefill_row_mask(self, num_rows: int, device: torch.device) -> torch.Tensor:
        q_start = self._last_step_query_start_loc
        if isinstance(q_start, torch.Tensor) and int(q_start.numel()) >= 2:
            q = q_start.to(device=device, dtype=torch.long)
            return (q[1:] - q[:-1]) > 1
        return torch.ones(num_rows, dtype=torch.bool, device=device)

    def _reset_decode_state_rows(self, mask: torch.Tensor, num_rows: int, device: torch.device) -> None:
        self._ensure_decode_state_capacity(num_rows, device)
        row_mask = mask.to(device=device, dtype=torch.bool)
        self._decode_has_codes[:num_rows].masked_fill_(row_mask, False)
        self._decode_generation_done[:num_rows].masked_fill_(row_mask, False)
        self._decode_delay_count[:num_rows].masked_fill_(row_mask, 0)
        self._decode_eoc_countdown[:num_rows].masked_fill_(row_mask, -1)
        self._decode_step_count[:num_rows].masked_fill_(row_mask, 0)

    def _ensure_decode_state_capacity(self, num_rows: int, device: torch.device | None = None) -> None:
        """Keep GPU-resident decode state tensors aligned and large enough."""
        if device is not None and self._decode_last_codes.device != device:
            self._decode_last_codes = self._decode_last_codes.to(device)
            self._decode_has_codes = self._decode_has_codes.to(device)
            self._decode_delay_count = self._decode_delay_count.to(device)
            self._decode_eoc_countdown = self._decode_eoc_countdown.to(device)
            self._decode_generation_done = self._decode_generation_done.to(device)
            self._decode_seed = self._decode_seed.to(device)
            self._decode_step_count = self._decode_step_count.to(device)
            self._audio_state_update_mask = self._audio_state_update_mask.to(device)

        cur_rows = int(self._decode_last_codes.shape[0])
        if num_rows <= cur_rows:
            return

        new_size = max(num_rows, cur_rows * 2)
        state_device = self._decode_last_codes.device

        new_last_codes = torch.zeros(new_size, self.num_codebooks, dtype=torch.long, device=state_device)
        new_last_codes[:cur_rows].copy_(self._decode_last_codes)
        self._decode_last_codes = new_last_codes

        new_has_codes = torch.zeros(new_size, dtype=torch.bool, device=state_device)
        new_has_codes[:cur_rows].copy_(self._decode_has_codes)
        self._decode_has_codes = new_has_codes

        new_delay_count = torch.zeros(new_size, dtype=torch.long, device=state_device)
        new_delay_count[:cur_rows].copy_(self._decode_delay_count)
        self._decode_delay_count = new_delay_count

        new_eoc_countdown = torch.full((new_size,), -1, dtype=torch.long, device=state_device)
        new_eoc_countdown[:cur_rows].copy_(self._decode_eoc_countdown)
        self._decode_eoc_countdown = new_eoc_countdown

        new_generation_done = torch.zeros(new_size, dtype=torch.bool, device=state_device)
        new_generation_done[:cur_rows].copy_(self._decode_generation_done)
        self._decode_generation_done = new_generation_done

        new_seed = torch.full((new_size,), -1, dtype=torch.long, device=state_device)
        new_seed[:cur_rows].copy_(self._decode_seed)
        self._decode_seed = new_seed

        new_step_count = torch.zeros(new_size, dtype=torch.long, device=state_device)
        new_step_count[:cur_rows].copy_(self._decode_step_count)
        self._decode_step_count = new_step_count

        new_update_mask = torch.zeros(new_size, dtype=torch.bool, device=state_device)
        new_update_mask[:cur_rows].copy_(self._audio_state_update_mask)
        self._audio_state_update_mask = new_update_mask

    def _ensure_request_state_capacity(self, num_rows: int, device: torch.device) -> None:
        """Grow the stable request-owned state pool without losing rows."""
        names_and_defaults = (
            ("_request_decode_last_codes", 0),
            ("_request_decode_has_codes", 0),
            ("_request_decode_delay_count", 0),
            ("_request_decode_eoc_countdown", -1),
            ("_request_decode_generation_done", 0),
            ("_request_decode_seed", -1),
            ("_request_decode_step_count", 0),
        )
        first = self._request_decode_last_codes
        if first.device != device:
            for name, _ in names_and_defaults:
                setattr(self, name, getattr(self, name).to(device))
            first = self._request_decode_last_codes

        cur_rows = int(first.shape[0])
        if num_rows <= cur_rows:
            return
        new_size = max(num_rows, max(1, cur_rows * 2))
        for name, default in names_and_defaults:
            old = getattr(self, name)
            shape = (new_size, *old.shape[1:])
            new = torch.full(shape, default, dtype=old.dtype, device=device)
            new[:cur_rows].copy_(old)
            setattr(self, name, new)

    def _reset_request_state_pool_rows(self, rows: torch.Tensor) -> None:
        if int(rows.numel()) == 0:
            return
        rows = rows.to(device=self._request_decode_last_codes.device, dtype=torch.long)
        self._request_decode_last_codes.index_fill_(0, rows, 0)
        self._request_decode_has_codes.index_fill_(0, rows, False)
        self._request_decode_delay_count.index_fill_(0, rows, 0)
        self._request_decode_eoc_countdown.index_fill_(0, rows, -1)
        self._request_decode_generation_done.index_fill_(0, rows, False)
        self._request_decode_seed.index_fill_(0, rows, -1)
        self._request_decode_step_count.index_fill_(0, rows, 0)

    def _commit_request_state(self, num_rows: int | None = None) -> None:
        """Commit current active rows to their immutable request-owned rows."""
        active_ids = self._active_request_ids
        if num_rows is None:
            num_rows = len(active_ids)
        num_rows = min(int(num_rows), len(active_ids))
        if num_rows <= 0:
            return

        active_indices: list[int] = []
        pool_indices: list[int] = []
        active_pool_rows = self._active_request_pool_rows
        for active_idx, req_id in enumerate(active_ids[:num_rows]):
            pool_row = self._request_state_row_by_id.get(req_id)
            if pool_row is None:
                continue
            if active_idx >= int(active_pool_rows.numel()) or int(active_pool_rows[active_idx]) != pool_row:
                continue
            active_indices.append(active_idx)
            pool_indices.append(pool_row)
        if not active_indices:
            return

        device = self._decode_last_codes.device
        active = torch.tensor(active_indices, dtype=torch.long, device=device)
        pool = torch.tensor(pool_indices, dtype=torch.long, device=device)
        self._request_decode_last_codes.index_copy_(0, pool, self._decode_last_codes.index_select(0, active))
        self._request_decode_has_codes.index_copy_(0, pool, self._decode_has_codes.index_select(0, active))
        self._request_decode_delay_count.index_copy_(0, pool, self._decode_delay_count.index_select(0, active))
        self._request_decode_eoc_countdown.index_copy_(0, pool, self._decode_eoc_countdown.index_select(0, active))
        self._request_decode_generation_done.index_copy_(0, pool, self._decode_generation_done.index_select(0, active))
        self._request_decode_seed.index_copy_(0, pool, self._decode_seed.index_select(0, active))
        self._request_decode_step_count.index_copy_(0, pool, self._decode_step_count.index_select(0, active))

    def _prepare_request_state(self, req_ids: list[str] | tuple[str, ...], device: torch.device) -> torch.Tensor:
        """Gather stable request state when the scheduler changes batch rows."""
        req_ids_tuple = tuple(req_ids)
        if req_ids_tuple == self._active_request_ids:
            return self._active_request_pool_rows[: len(req_ids_tuple)]

        self._commit_request_state()
        self._ensure_decode_state_capacity(len(req_ids_tuple), device)

        pool_rows: list[int] = []
        newly_allocated: list[int] = []
        for req_id in req_ids_tuple:
            pool_row = self._request_state_row_by_id.get(req_id)
            if pool_row is None:
                if self._request_state_free_rows:
                    pool_row = self._request_state_free_rows.pop()
                else:
                    pool_row = self._request_state_next_row
                    self._request_state_next_row += 1
                self._request_state_row_by_id[req_id] = pool_row
                newly_allocated.append(pool_row)
            pool_rows.append(pool_row)

        self._ensure_request_state_capacity(self._request_state_next_row, device)
        if newly_allocated:
            self._reset_request_state_pool_rows(torch.tensor(newly_allocated, dtype=torch.long, device=device))

        seeded_pool_rows: list[int] = []
        seeded_values: list[int] = []
        for req_id, pool_row in zip(req_ids_tuple, pool_rows, strict=True):
            seed = self._request_sampling_seed_by_id.get(req_id)
            if seed is not None:
                seeded_pool_rows.append(pool_row)
                seeded_values.append(seed)
        if seeded_pool_rows:
            self._request_decode_seed.index_copy_(
                0,
                torch.tensor(seeded_pool_rows, dtype=torch.long, device=device),
                torch.tensor(seeded_values, dtype=torch.long, device=device),
            )

        rows = torch.tensor(pool_rows, dtype=torch.long, device=device)
        num_rows = len(pool_rows)
        if num_rows:
            self._decode_last_codes[:num_rows].copy_(self._request_decode_last_codes.index_select(0, rows))
            self._decode_has_codes[:num_rows].copy_(self._request_decode_has_codes.index_select(0, rows))
            self._decode_delay_count[:num_rows].copy_(self._request_decode_delay_count.index_select(0, rows))
            self._decode_eoc_countdown[:num_rows].copy_(self._request_decode_eoc_countdown.index_select(0, rows))
            self._decode_generation_done[:num_rows].copy_(self._request_decode_generation_done.index_select(0, rows))
            self._decode_seed[:num_rows].copy_(self._request_decode_seed.index_select(0, rows))
            self._decode_step_count[:num_rows].copy_(self._request_decode_step_count.index_select(0, rows))

        self._active_request_pool_rows = rows
        self._active_request_ids = req_ids_tuple
        return rows

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        """Release request-owned state rows after finish or cancellation."""
        released: list[int] = []
        for req_id in finished_req_ids:
            self._fast_audio_direct_request_ids.discard(req_id)
            pool_row = self._request_state_row_by_id.pop(req_id, None)
            self._request_sampling_seed_by_id.pop(req_id, None)
            if pool_row is not None:
                released.append(pool_row)
                self._request_state_free_rows.append(pool_row)
        if released:
            self._reset_request_state_pool_rows(
                torch.tensor(released, dtype=torch.long, device=self._request_decode_last_codes.device)
            )

    def _audio_seed_mask_from_step_input(self, num_rows: int, device: torch.device) -> torch.Tensor | None:
        """Return rows whose previous token is <|audio|> using current step input_ids.

        This is the hot-path replacement for scanning sampling_metadata
        output_token_ids. In decode, vLLM feeds the previous sampled token as
        the current single-token input for each row.
        """
        audio_id = self._audio_continuation_id
        ids = self._last_step_input_ids
        if audio_id is None or not isinstance(ids, torch.Tensor) or int(ids.numel()) <= 0:
            return None

        flat_ids = ids.reshape(-1).to(device=device)
        q_start = self._last_step_query_start_loc
        if isinstance(q_start, torch.Tensor) and int(q_start.numel()) == num_rows + 1:
            q = q_start.to(device=device, dtype=torch.long)
            tail_idx = q[1:] - 1
            valid_tail = tail_idx >= 0
            tail_ids = flat_ids.index_select(0, tail_idx.clamp_min(0))
            return (tail_ids == int(audio_id)) & valid_tail
        if int(flat_ids.numel()) == num_rows:
            tail_ids = flat_ids
        else:
            return None
        return tail_ids == int(audio_id)

    def _fast_audio_sampler_gpu_fallback_reason(
        self,
        *,
        logits: torch.Tensor,
        sampling_metadata: Any,
        num_rows: int,
    ) -> str | None:
        if logits is None or logits.ndim != 2 or int(logits.shape[0]) != num_rows:
            return "invalid_logits"
        if num_rows <= 0:
            return "empty_batch"
        if getattr(sampling_metadata, "max_num_logprobs", None) is not None:
            return "logprobs"
        if getattr(sampling_metadata, "allowed_token_ids_mask", None) is not None:
            return "allowed_token_ids"
        if bool(getattr(sampling_metadata, "bad_words_token_ids", None)):
            return "bad_words"
        return None

    def _clear_last_audio_outputs(self) -> None:
        lease = getattr(self, "_last_audio_host_lease", None)
        if lease is not None:
            lease.release()
            self._last_audio_host_lease = None
        self._last_audio_codes = None
        self._last_audio_code_valid = []
        self._last_audio_host_staging = None
        self._last_audio_staging_event = None
        self._decode_active_audio_count = 0

    def _apply_audio_mode_bias_batched(
        self,
        logits: torch.Tensor,
        audio_mask: torch.Tensor,
        done_mask: torch.Tensor,
    ) -> None:
        """Force audio/EOS rows with GPU masks, avoiding metadata scans."""
        audio_id = self._audio_continuation_id
        if audio_id is None or logits is None or logits.ndim != 2:
            return

        vocab = int(logits.shape[-1])
        if audio_id < 0 or audio_id >= vocab:
            return

        target = torch.full(
            (int(logits.shape[0]),),
            int(audio_id),
            dtype=torch.long,
            device=logits.device,
        )
        if self._eos_token_id is not None and 0 <= int(self._eos_token_id) < vocab:
            eos_target = torch.full_like(target, int(self._eos_token_id))
            target = torch.where(done_mask.to(device=logits.device), eos_target, target)

        target = target.unsqueeze(1)
        target_values = logits.gather(1, target)
        logits.masked_fill_(audio_mask.to(device=logits.device).unsqueeze(1), float("-inf"))
        logits.scatter_(1, target, target_values)

    def _refresh_request_sampling_seeds(self, sampling_metadata: Any, num_rows: int) -> None:
        """Resolve each request's generator seed once, without advancing it."""
        generators = getattr(sampling_metadata, "generators", None)
        if not isinstance(generators, dict) or not self._active_request_ids:
            return
        for row, req_id in enumerate(self._active_request_ids[:num_rows]):
            if req_id in self._request_sampling_seed_by_id:
                continue
            generator = generators.get(row)
            if generator is None:
                continue
            seed = int(generator.initial_seed())
            self._request_sampling_seed_by_id[req_id] = seed
            self._decode_seed[row] = seed

    def register_request_sampling_seeds(self, seeds: Mapping[str, int]) -> None:
        """Register immutable serving-owned seeds before request materialization."""
        for req_id, seed_value in seeds.items():
            req_id = str(req_id)
            seed = int(seed_value)
            previous = self._request_sampling_seed_by_id.get(req_id)
            if previous is not None and previous != seed:
                raise ValueError(
                    f"Higgs request {req_id!r} already owns effective seed {previous}, cannot replace it with {seed}"
                )
            self._request_sampling_seed_by_id[req_id] = seed

    # ------------------------------------------------------------------ sampling
    def sample(self, logits: torch.Tensor | None, sampling_metadata: Any) -> Any:
        """Model-owned sampler with delay-pattern audio dispatch.

        Mirrors v2's pattern: bias LM logits to force audio continuation,
        sample multi-codebook codes via the fused head, apply delay pattern,
        and accumulate per-request state.
        """
        self._resolve_token_ids()
        self._trajectory_hidden_snapshot = None

        audio_id = self._audio_continuation_id

        def run_stock_sampler() -> Any:
            if logits is None:
                raise RuntimeError(
                    "Higgs text logits were skipped before every active request "
                    "entered the request-aware direct-audio path"
                )
            sampler = getattr(self, "_stock_sampler", None)
            if sampler is None:
                from vllm.v1.sample.sampler import Sampler

                sampler = Sampler()
                self._stock_sampler = sampler
            return sampler(logits=logits, sampling_metadata=sampling_metadata)

        hidden = self._last_logits_hidden
        self._last_logits_hidden = None
        if hidden is None or audio_id is None:
            sampler_output = run_stock_sampler()
            self._clear_last_audio_outputs()
            return sampler_output

        num_rows = int(hidden.shape[0])
        if self._trajectory_recorder is not None:
            self._trajectory_hidden_snapshot = hidden.reshape(-1, hidden.shape[-1])[:num_rows].detach()
        self._ensure_decode_state_capacity(num_rows, hidden.device)
        self._refresh_request_sampling_seeds(sampling_metadata, num_rows)
        ids = getattr(self, "_last_step_input_ids", None)
        decode_only = isinstance(ids, torch.Tensor) and int(ids.numel()) == num_rows
        # ``logits is None`` is the immutable decision made by compute_logits
        # for this exact sampling step. Async output resolution may update the
        # live request certification set before sample_tokens consumes it, so
        # do not re-decide a logits-free step from mutable model state here.
        direct_audio_batch = logits is None
        if not decode_only:
            pfmask = self._prefill_row_mask(num_rows, hidden.device)
            self._reset_decode_state_rows(pfmask, num_rows, hidden.device)
        prev_audio_mask = self._audio_seed_mask_from_step_input(num_rows, hidden.device)
        if prev_audio_mask is None:
            prev_audio_mask = torch.zeros(num_rows, dtype=torch.bool, device=hidden.device)
        active_mask = self._decode_has_codes[:num_rows].to(device=hidden.device, dtype=torch.bool)
        done_mask = self._decode_generation_done[:num_rows].to(device=hidden.device, dtype=torch.bool)
        fast_fallback_reason = (
            None
            if logits is None and direct_audio_batch
            else self._fast_audio_sampler_gpu_fallback_reason(
                logits=logits,
                sampling_metadata=sampling_metadata,
                num_rows=num_rows,
            )
        )
        seed_mask = prev_audio_mask & ~active_mask & ~done_mask
        audio_mask = prev_audio_mask | active_mask | done_mask
        code_row_mask = (seed_mask | active_mask) & ~done_mask
        gpu_stock_sampler_reasons = {"logprobs", "allowed_token_ids", "bad_words"}
        use_gpu_audio_mode = fast_fallback_reason is None or fast_fallback_reason in gpu_stock_sampler_reasons

        if fast_fallback_reason is None:
            if direct_audio_batch:
                audio_tokens = torch.full((num_rows,), int(audio_id), dtype=torch.int32, device=hidden.device)
                if self._eos_token_id is not None:
                    eos_tokens = torch.full_like(audio_tokens, int(self._eos_token_id))
                    audio_tokens = torch.where(done_mask, eos_tokens, audio_tokens)
                sampled = audio_tokens.unsqueeze(-1)
                sampler_output = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None)
            else:
                self._apply_audio_mode_bias_batched(logits, audio_mask, done_mask)
                sampler_output = run_stock_sampler()
                sampled = getattr(sampler_output, "sampled_token_ids", None)
                if sampled is None:
                    self._clear_last_audio_outputs()
                    return sampler_output
        elif use_gpu_audio_mode:
            self._apply_audio_mode_bias_batched(logits, audio_mask, done_mask)
            sampler_output = run_stock_sampler()
            sampled = getattr(sampler_output, "sampled_token_ids", None)
            if sampled is None:
                self._clear_last_audio_outputs()
                return sampler_output
        else:
            sampler_output = run_stock_sampler()
            self._clear_last_audio_outputs()
            return sampler_output

        sampled_flat = sampled.reshape(-1)
        if int(sampled_flat.numel()) != num_rows:
            self._clear_last_audio_outputs()
            return sampler_output

        if use_gpu_audio_mode:
            audio_row_tensor = self._get_row_indices(num_rows, hidden.device)
            seed_mask_1d = seed_mask.to(device=hidden.device, dtype=torch.bool)
            done_mask_1d = done_mask.to(device=hidden.device, dtype=torch.bool)
            self._decode_generation_done[:num_rows].masked_fill_(done_mask_1d, False)
            self._decode_has_codes[:num_rows].masked_fill_(done_mask_1d, False)
            self._decode_delay_count[:num_rows].masked_fill_(done_mask_1d, 0)
            self._decode_eoc_countdown[:num_rows].masked_fill_(done_mask_1d, -1)

            boc_frames = self._get_boc_frames(num_rows, hidden.device)
            seed_mask_2d = seed_mask_1d.unsqueeze(1)
            self._decode_last_codes[:num_rows] = torch.where(
                seed_mask_2d,
                boc_frames,
                self._decode_last_codes[:num_rows],
            )
            self._decode_has_codes[:num_rows].masked_fill_(seed_mask_1d, True)
            self._decode_delay_count[:num_rows].masked_fill_(seed_mask_1d, 0)
            self._decode_eoc_countdown[:num_rows].masked_fill_(seed_mask_1d, -1)
            self._decode_generation_done[:num_rows].masked_fill_(seed_mask_1d, False)
            self._decode_active_audio_count = num_rows
        else:
            audio_row_tensor = self._get_row_indices(num_rows, hidden.device)

        all_audio_rows = use_gpu_audio_mode
        if all_audio_rows and self._audio_state_step_mode != "legacy":
            bucket_rows = self._audio_state_step_bucket_size(num_rows)
            self._ensure_decode_state_capacity(bucket_rows, hidden.device)
            hidden_staging = self._get_audio_state_hidden_staging(
                bucket_rows,
                hidden.shape[-1],
                hidden.device,
                hidden.dtype,
            )
            hidden_staging.zero_()
            hidden_staging[:num_rows].copy_(hidden.reshape(-1, hidden.shape[-1])[:num_rows])
            update_staging = self._audio_state_update_mask[:bucket_rows]
            update_staging.zero_()
            update_staging[:num_rows].copy_(code_row_mask[:num_rows])
            state_step = self._get_audio_state_step_callable()
            state_result = state_step(
                hidden_staging,
                self._decode_delay_count[:bucket_rows],
                self._decode_eoc_countdown[:bucket_rows],
                self._decode_generation_done[:bucket_rows],
                self._decode_has_codes[:bucket_rows],
                self._decode_last_codes[:bucket_rows],
                self._decode_step_count[:bucket_rows],
                self._decode_seed[:bucket_rows],
                update_staging,
            )
            self._publish_audio_state_step(state_result, num_rows, hidden.device)
            return sampler_output

        # Per-codebook logits at audio positions
        if use_gpu_audio_mode:
            cb_logits = self._audio_codebook_logits_from_rows(hidden, audio_row_tensor, all_rows=True)
        else:
            cb_logits = self._audio_codebook_logits_from_rows(hidden, audio_row_tensor, all_rows=False)

        # Apply delay pattern masking BEFORE sampling
        self._apply_delay_pattern_masking_batched(cb_logits, audio_row_tensor, all_rows=all_audio_rows)

        # Sample per-codebook
        cb_logits_2d = cb_logits.reshape(-1, cb_logits.shape[-1])
        if all_audio_rows:
            row_seeds = self._decode_seed[: int(cb_logits.shape[0])]
            row_steps = self._decode_step_count[: int(cb_logits.shape[0])]
        else:
            row_seeds = self._decode_seed.index_select(0, audio_row_tensor)
            row_steps = self._decode_step_count.index_select(0, audio_row_tensor)
        codes_2d = self._sample_audio_codes(
            cb_logits_2d,
            row_seeds=row_seeds,
            row_steps=row_steps,
        )
        codes_flat = codes_2d.view(cb_logits.shape[0], cb_logits.shape[1]).to(torch.long)

        self._update_delay_state_batched(
            codes_flat,
            audio_row_tensor,
            num_rows,
            hidden.device,
            code_row_mask=code_row_mask,
            all_rows=all_audio_rows,
        )
        return sampler_output

    def _get_audio_codes_buffer(self, num_rows: int, device: torch.device) -> torch.Tensor:
        buf = self._last_audio_codes_buffer
        if buf is None or buf.device != device or int(buf.shape[0]) < num_rows:
            rows = max(num_rows, 16)
            buf = torch.empty((rows, self.num_codebooks), dtype=torch.long, device=device)
            self._last_audio_codes_buffer = buf
        return buf[:num_rows]

    def _get_audio_gpu_staging_buffer(self, num_rows: int, device: torch.device) -> torch.Tensor:
        width = self.num_codebooks + 2
        buf = self._last_audio_gpu_staging
        if buf is None or buf.device != device or int(buf.shape[0]) < num_rows:
            rows = max(num_rows, 16)
            buf = torch.empty((rows, width), dtype=torch.int32, device=device)
            self._last_audio_gpu_staging = buf
        return buf[:num_rows]

    def _get_audio_host_staging_buffer(self, num_rows: int) -> torch.Tensor:
        pool = getattr(self, "_audio_host_staging_pool", None)
        if pool is None:
            pool = _HiggsAudioHostStagingPool(
                width=self.num_codebooks + 2,
                pin_memory=torch.cuda.is_available(),
            )
            self._audio_host_staging_pool = pool
        lease = pool.acquire(num_rows)
        self._last_audio_host_lease = lease
        self._last_audio_host_staging = lease.tensor
        return lease.tensor

    @staticmethod
    def _device_cache_key(device: torch.device) -> tuple[str, int]:
        return (device.type, -1 if device.index is None else int(device.index))

    def _get_row_indices(self, num_rows: int, device: torch.device) -> torch.Tensor:
        key = self._device_cache_key(device)
        buf = self._row_index_cache.get(key)
        if buf is None or buf.device != device or int(buf.shape[0]) < num_rows:
            rows = max(num_rows, 16)
            buf = torch.arange(rows, dtype=torch.long, device=device)
            self._row_index_cache[key] = buf
        return buf[:num_rows]

    def _get_codebook_indices(self, device: torch.device) -> torch.Tensor:
        key = self._device_cache_key(device)
        buf = self._codebook_index_cache.get(key)
        if buf is None or buf.device != device or int(buf.shape[1]) != self.num_codebooks:
            buf = torch.arange(self.num_codebooks, dtype=torch.long, device=device).view(1, self.num_codebooks)
            self._codebook_index_cache[key] = buf
        return buf

    def _get_boc_frames(self, num_rows: int, device: torch.device) -> torch.Tensor:
        key = self._device_cache_key(device)
        buf = self._boc_frame_cache.get(key)
        if buf is None or buf.device != device or int(buf.shape[0]) < num_rows:
            rows = max(num_rows, 16)
            buf = torch.full((rows, self.num_codebooks), BOC_ID, dtype=torch.long, device=device)
            self._boc_frame_cache[key] = buf
        return buf[:num_rows]

    # ------------------------------------------------------------------ postprocess
    def postprocess(
        self,
        hidden_states_slice: torch.Tensor,
        multimodal_outputs: Any = None,
        **req_infos: Any,
    ) -> dict[str, Any]:
        """Publish per-request audio codes into model_intermediate_buffer.

        Called once per request in batch order. Indexes _last_audio_codes
        by a running cursor (one row per request per step).
        """
        _ = multimodal_outputs
        return self._postprocess_impl()

    def take_pending_omni_postprocess(
        self,
        request_ids: Sequence[str],
    ) -> HiggsAudioV3PendingStep | None:
        """Detach this step's D2H snapshot for background request-aware resolve."""
        host_staging = self._last_audio_host_staging
        lease = self._last_audio_host_lease
        if host_staging is None or lease is None:
            return None
        trajectory_recorder = getattr(self, "_trajectory_recorder", None)
        sampling_seeds: list[int] | None = None
        sampling_post_steps: list[int] | None = None
        if trajectory_recorder is not None:
            num_rows = len(request_ids)
            sampling_seeds = self._decode_seed[:num_rows].detach().cpu().tolist()
            sampling_post_steps = self._decode_step_count[:num_rows].detach().cpu().tolist()
        pending = HiggsAudioV3PendingStep.create(
            request_ids=request_ids,
            request_seeds=[self._request_sampling_seed_by_id.get(str(req_id)) for req_id in request_ids],
            sampling_seeds=sampling_seeds,
            sampling_post_steps=sampling_post_steps,
            hidden_snapshot=self._trajectory_hidden_snapshot,
            host_staging=host_staging,
            event=self._last_audio_staging_event,
            num_codebooks=self.num_codebooks,
            release=lease.release,
            on_resolve=self._mark_audio_direct_requests,
            trajectory_recorder=trajectory_recorder,
        )
        # Ownership moved to the immutable pending handle. The next sample may
        # acquire the other slot without touching this step's host snapshot.
        self._last_audio_host_lease = None
        self._last_audio_host_staging = None
        self._last_audio_staging_event = None
        self._last_audio_valid_flags = None
        self._last_audio_done_flags = None
        self._trajectory_hidden_snapshot = None
        self._postprocess_cursor = 0
        return pending

    def _postprocess_impl(self) -> dict[str, Any]:
        host_staging = self._last_audio_host_staging
        codes_full = self._last_audio_codes
        if host_staging is None and codes_full is None:
            return {}
        event = self._last_audio_staging_event
        if event is not None:
            event.synchronize()
            self._last_audio_staging_event = None

        cursor = int(self._postprocess_cursor)
        num_rows = int(host_staging.shape[0]) if host_staging is not None else int(codes_full.shape[0])
        if cursor >= num_rows:
            self._postprocess_cursor = 0
            return {}
        self._postprocess_cursor = cursor + 1
        last_row = cursor + 1 >= num_rows
        try:
            if host_staging is not None:
                if self._last_audio_valid_flags is None or self._last_audio_done_flags is None:
                    self._last_audio_valid_flags = host_staging[:num_rows, self.num_codebooks].tolist()
                    self._last_audio_done_flags = host_staging[:num_rows, self.num_codebooks + 1].tolist()
                valid_flag = int(self._last_audio_valid_flags[cursor])
                if last_row:
                    request_ids = self._active_request_ids
                    if len(request_ids) == num_rows:
                        self._mark_audio_direct_requests(
                            request_ids,
                            self._last_audio_valid_flags,
                            self._last_audio_done_flags,
                        )
                if valid_flag == 0:
                    return {}
                new_codes = host_staging[cursor : cursor + 1, : self.num_codebooks]
                return {"codes": {"audio": new_codes}}

            if cursor >= len(self._last_audio_code_valid) or not self._last_audio_code_valid[cursor]:
                return {}
            slice_codes = codes_full[cursor : cursor + 1]
            new_codes = slice_codes.to(torch.int32)
            return {"codes": {"audio": new_codes}}
        finally:
            if last_row:
                lease = getattr(self, "_last_audio_host_lease", None)
                if lease is not None:
                    lease.release()
                    self._last_audio_host_lease = None

    # ------------------------------------------------------------------ helpers
    def _audio_codebook_logits_from_rows(
        self, hidden_states: torch.Tensor, audio_rows: torch.Tensor, *, all_rows: bool = False
    ) -> torch.Tensor:
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if audio_rows.numel() == 0:
            return torch.empty(
                (0, self.num_codebooks, self.codebook_size),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        if all_rows:
            return self.modality_head.generate(hidden_flat[: int(audio_rows.numel())])
        return self.modality_head.generate(hidden_flat.index_select(0, audio_rows.to(hidden_flat.device)))

    def _apply_delay_pattern_masking_batched(
        self, cb_logits: torch.Tensor, audio_rows: torch.Tensor, *, all_rows: bool = False
    ) -> None:
        """Vectorized delay-pattern masking using GPU-resident sampler state."""
        if cb_logits.numel() == 0:
            return

        rows = audio_rows.to(device=cb_logits.device, dtype=torch.long)
        num_audio_rows = int(cb_logits.shape[0])
        num_codebooks = self.num_codebooks
        q = self._get_codebook_indices(cb_logits.device)
        if all_rows:
            delay = self._decode_delay_count[:num_audio_rows]
            rem = self._decode_eoc_countdown[:num_audio_rows]
        else:
            delay = self._decode_delay_count.index_select(0, rows)
            rem = self._decode_eoc_countdown.index_select(0, rows)
        if delay.dtype != torch.long:
            delay = delay.to(torch.long)
        if rem.dtype != torch.long:
            rem = rem.to(torch.long)
        ramp = rem >= 0

        bos = BOC_ID
        eos = EOC_ID

        # Ramp-down: codebooks already behind the EOC wave are locked to EOC.
        lock_until = num_codebooks - rem
        locked = ramp.unsqueeze(1) & (q < lock_until.unsqueeze(1))
        eoc_logits = cb_logits[:, :, eos].clone()
        cb_logits.masked_fill_(locked.unsqueeze(-1), float("-inf"))
        cb_logits[:, :, eos] = torch.where(locked, eoc_logits, cb_logits[:, :, eos])

        # Ramp-down tail: active codebooks cannot emit stream specials.
        ramp_tail = ramp.unsqueeze(1) & (q >= lock_until.unsqueeze(1))
        cb_logits[:, :, bos] = torch.where(
            ramp_tail,
            torch.full_like(cb_logits[:, :, bos], float("-inf")),
            cb_logits[:, :, bos],
        )
        cb_logits[:, :, eos] = torch.where(
            ramp_tail,
            torch.full_like(cb_logits[:, :, eos], float("-inf")),
            cb_logits[:, :, eos],
        )

        # Normal generation: delayed codebooks are forced to BOC.
        normal = ~ramp
        allowed_until = torch.clamp(delay + 1, max=num_codebooks)
        delayed = normal.unsqueeze(1) & (q >= allowed_until.unsqueeze(1))
        boc_logits = cb_logits[:, :, bos].clone()
        cb_logits.masked_fill_(delayed.unsqueeze(-1), float("-inf"))
        cb_logits[:, :, bos] = torch.where(delayed, boc_logits, cb_logits[:, :, bos])

        # Normal active codebooks: BOC is disallowed; EOC is only allowed on cb0.
        allowed = normal.unsqueeze(1) & (q < allowed_until.unsqueeze(1))
        cb_logits[:, :, bos] = torch.where(
            allowed,
            torch.full_like(cb_logits[:, :, bos], float("-inf")),
            cb_logits[:, :, bos],
        )
        nonzero_allowed = allowed & (q > 0)
        cb_logits[:, :, eos] = torch.where(
            nonzero_allowed,
            torch.full_like(cb_logits[:, :, eos], float("-inf")),
            cb_logits[:, :, eos],
        )

    def _update_delay_state_batched(
        self,
        codes_flat: torch.Tensor,
        audio_rows: torch.Tensor,
        num_rows: int,
        device: torch.device,
        *,
        code_row_mask: torch.Tensor | None = None,
        all_rows: bool = False,
    ) -> None:
        """Update delay/ramp-down state in batch and stage per-request outputs."""
        self._ensure_decode_state_capacity(num_rows, device)
        if codes_flat.numel() == 0:
            self._clear_last_audio_outputs()
            return

        rows = audio_rows.to(device=device, dtype=torch.long)
        num_audio_rows = int(codes_flat.shape[0])
        num_codebooks = self.num_codebooks
        q = self._get_codebook_indices(device)

        if all_rows:
            prev_delay = self._decode_delay_count[:num_audio_rows]
            prev_rem = self._decode_eoc_countdown[:num_audio_rows]
            prev_done = self._decode_generation_done[:num_audio_rows].to(torch.bool)
            prev_has_codes = self._decode_has_codes[:num_audio_rows].to(torch.bool)
            prev_codes = self._decode_last_codes[:num_audio_rows]
            prev_step_count = self._decode_step_count[:num_audio_rows]
        else:
            prev_delay = self._decode_delay_count.index_select(0, rows)
            prev_rem = self._decode_eoc_countdown.index_select(0, rows)
            prev_done = self._decode_generation_done.index_select(0, rows).to(torch.bool)
            prev_has_codes = self._decode_has_codes.index_select(0, rows).to(torch.bool)
            prev_codes = self._decode_last_codes.index_select(0, rows)
            prev_step_count = self._decode_step_count.index_select(0, rows)
        if prev_delay.dtype != torch.long:
            prev_delay = prev_delay.to(torch.long)
        if prev_rem.dtype != torch.long:
            prev_rem = prev_rem.to(torch.long)
        if code_row_mask is None:
            update_mask = torch.ones_like(prev_has_codes)
        elif all_rows:
            update_mask = code_row_mask.to(device=device, dtype=torch.bool)[:num_audio_rows]
        else:
            update_mask = code_row_mask.to(device=device, dtype=torch.bool).index_select(0, rows)
        ramp = prev_rem >= 0

        codes = codes_flat.to(device=device, dtype=torch.long).clone()

        # Leading BOC delay pad.
        delay_active = (~ramp) & ((prev_delay + 1) < num_codebooks)
        delay_mask = delay_active.unsqueeze(1) & (q > prev_delay.unsqueeze(1))
        codes = torch.where(delay_mask, torch.full_like(codes, BOC_ID), codes)
        next_delay = torch.where(delay_active, prev_delay + 1, prev_delay)

        # Trailing EOC ramp-down.
        lock_until = num_codebooks - prev_rem
        ramp_mask = ramp.unsqueeze(1) & (q < lock_until.unsqueeze(1))
        codes = torch.where(ramp_mask, torch.full_like(codes, EOC_ID), codes)
        ramp_next_rem = prev_rem - 1

        # New EOC detection in normal mode. This avoids tensor->Python control
        # flow; ties are impossible for the index because q is monotonic.
        eos_mask = (~ramp).unsqueeze(1) & (codes == EOC_ID)
        eos_idx_values = torch.where(eos_mask, q.expand_as(codes), torch.full_like(codes, -1))
        last_eos_idx = eos_idx_values.max(dim=1).values
        has_eos = last_eos_idx >= 0
        eos_prefix = has_eos.unsqueeze(1) & (q <= last_eos_idx.unsqueeze(1))
        codes = torch.where(eos_prefix, torch.full_like(codes, EOC_ID), codes)
        normal_next_rem = torch.where(has_eos, num_codebooks - last_eos_idx - 1, torch.full_like(last_eos_idx, -1))

        next_rem = torch.where(ramp, ramp_next_rem, normal_next_rem)
        done = ((next_rem >= 0) & (next_rem <= 0)) & update_mask
        valid = update_mask

        next_delay = torch.where(done, torch.zeros_like(next_delay), next_delay)
        next_rem = torch.where(done, torch.full_like(next_rem, -1), next_rem)
        output_codes = codes

        write_delay = torch.where(update_mask, next_delay, prev_delay)
        write_rem = torch.where(update_mask, next_rem, prev_rem)
        write_done = torch.where(update_mask, done, prev_done)
        write_codes = torch.where(update_mask.unsqueeze(1), codes, prev_codes)
        write_has_codes = torch.where(update_mask, ~done, prev_has_codes)
        write_step_count = prev_step_count + update_mask.to(prev_step_count.dtype)

        if all_rows:
            self._decode_delay_count[:num_audio_rows].copy_(write_delay.to(dtype=self._decode_delay_count.dtype))
            self._decode_eoc_countdown[:num_audio_rows].copy_(write_rem.to(dtype=self._decode_eoc_countdown.dtype))
            self._decode_generation_done[:num_audio_rows].copy_(write_done)
            self._decode_last_codes[:num_audio_rows].copy_(write_codes)
            self._decode_has_codes[:num_audio_rows].copy_(write_has_codes)
            self._decode_step_count[:num_audio_rows].copy_(write_step_count)
        else:
            self._decode_delay_count.index_copy_(0, rows, write_delay.to(dtype=self._decode_delay_count.dtype))
            self._decode_eoc_countdown.index_copy_(0, rows, write_rem.to(dtype=self._decode_eoc_countdown.dtype))
            self._decode_generation_done.index_copy_(0, rows, write_done)
            self._decode_last_codes.index_copy_(0, rows, write_codes)
            self._decode_has_codes.index_copy_(0, rows, write_has_codes)
            self._decode_step_count.index_copy_(0, rows, write_step_count)

        staged_codes = torch.where(update_mask.unsqueeze(1), output_codes, torch.full_like(output_codes, -1))
        if all_rows:
            valid_full = valid
            done_full = done
            staging = self._get_audio_gpu_staging_buffer(num_rows, device)
            staging[:, :num_codebooks].copy_(staged_codes.to(torch.int32))
            staging[:, num_codebooks].copy_(valid_full.to(torch.int32))
            staging[:, num_codebooks + 1].copy_(done_full.to(torch.int32))
        else:
            codes_full = self._get_audio_codes_buffer(num_rows, device)
            codes_full.fill_(-1)
            codes_full.index_copy_(0, rows, staged_codes)
            valid_full = torch.zeros(num_rows, dtype=torch.bool, device=device)
            done_full = torch.zeros(num_rows, dtype=torch.bool, device=device)
            valid_full.index_copy_(0, rows, valid)
            done_full.index_copy_(0, rows, done)
            staging = self._get_audio_gpu_staging_buffer(num_rows, device)
            staging[:, :num_codebooks].copy_(codes_full.to(torch.int32))
            staging[:, num_codebooks].copy_(valid_full.to(torch.int32))
            staging[:, num_codebooks + 1].copy_(done_full.to(torch.int32))

        host = self._get_audio_host_staging_buffer(num_rows)
        host.copy_(staging, non_blocking=True)
        if device.type == "cuda":
            lease = self._last_audio_host_lease
            assert lease is not None
            event = lease.cuda_event()
            event.record(torch.cuda.current_stream(device))
            self._last_audio_staging_event = event
        else:
            self._last_audio_staging_event = None

        self._decode_active_audio_count = num_rows if all_rows else int(rows.numel())
        # Host staging is the authoritative postprocess path. Keeping another
        # full GPU codes buffer alive adds fill/copy work on the hot path.
        self._last_audio_codes = None
        self._last_audio_host_staging = host[:num_rows]
        self._last_audio_valid_flags = None
        self._last_audio_done_flags = None
        self._last_audio_code_valid = []
        self._postprocess_cursor = 0

    def _audio_state_step_from_hidden(
        self,
        hidden: torch.Tensor,
        prev_delay: torch.Tensor,
        prev_rem: torch.Tensor,
        prev_done: torch.Tensor,
        prev_has_codes: torch.Tensor,
        prev_codes: torch.Tensor,
        prev_step_count: torch.Tensor,
        row_seeds: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        cb_logits = self.modality_head.generate(hidden)
        return HiggsAudioV3TalkerForConditionalGeneration._audio_state_step_tensor(
            cb_logits,
            prev_delay,
            prev_rem,
            prev_done,
            prev_has_codes,
            prev_codes,
            prev_step_count,
            row_seeds,
            update_mask,
            use_flashinfer=getattr(self, "_audio_sampler_mode", "torch") == "flashinfer",
            flashinfer_top_k_row_states=getattr(self, "_flashinfer_top_k_row_states", None),
            flashinfer_top_p_workspace=getattr(self, "_flashinfer_top_p_workspace", None),
        )

    @staticmethod
    def _audio_state_step_bucket_size(num_rows: int) -> int:
        """Use a bounded set of static shapes across batch shrink/reorder."""
        if num_rows <= 0:
            return 1
        return 1 << (int(num_rows) - 1).bit_length()

    def _get_audio_state_hidden_staging(
        self,
        num_rows: int,
        hidden_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = self._audio_state_hidden_staging
        if (
            buf is None
            or buf.device != device
            or buf.dtype != dtype
            or int(buf.shape[0]) < num_rows
            or int(buf.shape[1]) != hidden_size
        ):
            rows = max(num_rows, self._mlp_cudagraph_max_batch)
            buf = torch.empty((rows, hidden_size), dtype=dtype, device=device)
            self._audio_state_hidden_staging = buf
        return buf[:num_rows]

    def _get_audio_state_step_callable(self) -> Any:
        if self._audio_state_step_mode != "compile":
            return self._audio_state_step_from_hidden
        if self._compiled_audio_state_step is None:
            self._compiled_audio_state_step = torch.compile(
                self._audio_state_step_from_hidden,
                options={"triton.cudagraphs": False},
                fullgraph=True,
                dynamic=False,
            )
            logger.info_once("Higgs Audio V3: compiled modality head + sampling + delay/state transition")
        return self._compiled_audio_state_step

    def _publish_audio_state_step(
        self,
        result: tuple[torch.Tensor, ...],
        num_rows: int,
        device: torch.device,
    ) -> None:
        """Commit a functional state step and publish its static D2H snapshot."""
        (
            staged_codes,
            delay_count,
            eoc_countdown,
            generation_done,
            has_codes,
            last_codes,
            step_count,
            valid,
            done,
        ) = result
        self._decode_delay_count[:num_rows].copy_(delay_count[:num_rows])
        self._decode_eoc_countdown[:num_rows].copy_(eoc_countdown[:num_rows])
        self._decode_generation_done[:num_rows].copy_(generation_done[:num_rows])
        self._decode_has_codes[:num_rows].copy_(has_codes[:num_rows])
        self._decode_last_codes[:num_rows].copy_(last_codes[:num_rows])
        self._decode_step_count[:num_rows].copy_(step_count[:num_rows])

        staging = self._get_audio_gpu_staging_buffer(num_rows, device)
        staging[:, : self.num_codebooks].copy_(staged_codes[:num_rows].to(torch.int32))
        staging[:, self.num_codebooks].copy_(valid[:num_rows].to(torch.int32))
        staging[:, self.num_codebooks + 1].copy_(done[:num_rows].to(torch.int32))
        host = self._get_audio_host_staging_buffer(num_rows)
        host.copy_(staging, non_blocking=True)
        if device.type == "cuda":
            lease = self._last_audio_host_lease
            assert lease is not None
            event = lease.cuda_event()
            event.record(torch.cuda.current_stream(device))
            self._last_audio_staging_event = event
        else:
            self._last_audio_staging_event = None

        self._decode_active_audio_count = num_rows
        self._last_audio_codes = None
        self._last_audio_host_staging = host[:num_rows]
        self._last_audio_valid_flags = None
        self._last_audio_done_flags = None
        self._last_audio_code_valid = []
        self._postprocess_cursor = 0

    def _sample_audio_codes(
        self,
        logits_2d: torch.Tensor,
        *,
        row_seeds: torch.Tensor | None = None,
        row_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Replicate upstream sampling: temperature → top-k → top-p → multinomial."""
        return HiggsAudioV3TalkerForConditionalGeneration._sample_audio_codes_tensor(
            logits_2d,
            row_seeds=row_seeds,
            row_steps=row_steps,
            num_codebooks=self.num_codebooks,
            use_flashinfer=getattr(self, "_audio_sampler_mode", "torch") == "flashinfer",
            flashinfer_top_k_row_states=getattr(self, "_flashinfer_top_k_row_states", None),
            flashinfer_top_p_workspace=getattr(self, "_flashinfer_top_p_workspace", None),
        )

    @staticmethod
    def _sample_audio_codes_tensor(
        logits_2d: torch.Tensor,
        *,
        row_seeds: torch.Tensor | None,
        row_steps: torch.Tensor | None,
        num_codebooks: int,
        use_flashinfer: bool = False,
        flashinfer_top_k_row_states: torch.Tensor | None = None,
        flashinfer_top_p_workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tensor-only Higgs sampler shared by legacy and compiled paths."""
        x = logits_2d.float()
        top_k = 50
        top_p = 0.95
        if use_flashinfer:
            if _flashinfer_top_k_renorm_probs is None or _flashinfer_top_p_renorm_probs is None:
                raise RuntimeError("FlashInfer audio sampler requested but sampling kernels are unavailable")
            all_masked = ~torch.isfinite(x).any(dim=-1)
            fallback = x.argmax(dim=-1)
            safe_x = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(x), x)
            probs = safe_x.softmax(dim=-1).contiguous()
            if (
                flashinfer_top_k_row_states is not None
                and flashinfer_top_p_workspace is not None
                and flashinfer_top_k_row_states.numel() > 0
                and flashinfer_top_p_workspace.numel() > 0
            ):
                probs = _higgs_flashinfer_top_k_renorm_op(
                    probs,
                    flashinfer_top_k_row_states,
                )
                probs = _higgs_flashinfer_top_p_renorm_op(
                    probs,
                    flashinfer_top_p_workspace,
                )
            else:
                probs = _flashinfer_top_k_renorm_probs(probs, top_k)
                probs = _flashinfer_top_p_renorm_probs(probs, top_p, is_deterministic=True)
        else:
            if 0 < top_k < x.shape[-1]:
                kth = x.topk(top_k, dim=-1).values[..., -1:]
                x = x.masked_fill(x < kth, float("-inf"))
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = x.sort(dim=-1, descending=True)
                cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_mask = cumprobs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False
                mask = torch.zeros_like(x, dtype=torch.bool)
                mask.scatter_(-1, sorted_idx, sorted_mask)
                x = x.masked_fill(mask, float("-inf"))
            # Detect all-masked rows BEFORE softmax (softmax of all-inf yields NaN).
            all_masked = ~torch.isfinite(x).any(dim=-1)
            # Do not branch on tensor bools here; that synchronizes every audio step.
            fallback = x.argmax(dim=-1)
            safe_x = torch.where(all_masked.unsqueeze(-1), torch.zeros_like(x), x)
            probs = safe_x.softmax(dim=-1)
        if row_seeds is None:
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            num_reqs = int(row_seeds.numel())
            if int(probs.shape[0]) != num_reqs * num_codebooks:
                raise ValueError(
                    "Seeded Higgs audio sampling expects one codebook group "
                    f"per request: got {int(probs.shape[0])} rows for "
                    f"{num_reqs} requests x {num_codebooks} codebooks"
                )
            if row_steps is None or int(row_steps.numel()) != num_reqs:
                raise ValueError("row_steps must contain one entry per seeded request")

            # Stateless request RNG: each draw is keyed by
            # (request seed, AR step, codebook). This is batch-order invariant
            # and pure tensor code, unlike a Python loop over torch.Generator.
            modulus = 2_147_483_647
            seeds = torch.remainder(row_seeds.to(device=probs.device, dtype=torch.long), modulus)
            steps = row_steps.to(device=probs.device, dtype=torch.long)
            codebooks = torch.arange(num_codebooks, device=probs.device, dtype=torch.long).unsqueeze(0)
            positions = steps.unsqueeze(1) * num_codebooks + codebooks + 1
            state = torch.remainder(seeds.unsqueeze(1) + positions * 1_103_515_245, modulus)
            state = torch.remainder(torch.bitwise_xor(state, state >> 16) * 48_271, modulus)
            uniforms = (state.to(torch.float32) + 0.5) / float(modulus)

            probs_3d = probs.view(num_reqs, num_codebooks, -1)
            cdf = probs_3d.cumsum(dim=-1)
            seeded = (cdf < uniforms.unsqueeze(-1)).sum(dim=-1).clamp_max(probs_3d.shape[-1] - 1)
            sampled = seeded.reshape(-1)
        return torch.where(all_masked, fallback, sampled)

    @staticmethod
    def _audio_state_step_tensor(
        cb_logits: torch.Tensor,
        prev_delay: torch.Tensor,
        prev_rem: torch.Tensor,
        prev_done: torch.Tensor,
        prev_has_codes: torch.Tensor,
        prev_codes: torch.Tensor,
        prev_step_count: torch.Tensor,
        row_seeds: torch.Tensor,
        update_mask: torch.Tensor,
        *,
        use_flashinfer: bool = False,
        flashinfer_top_k_row_states: torch.Tensor | None = None,
        flashinfer_top_p_workspace: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Sample audio codes and advance the full delay/EOC state machine.

        This function is deliberately functional: request ownership and D2H
        publication stay outside the compiled region, while every
        quality-sensitive GPU transition is represented by tensor operations.
        """
        device = cb_logits.device
        num_rows = int(cb_logits.shape[0])
        num_codebooks = int(cb_logits.shape[1])
        q = torch.arange(num_codebooks, dtype=torch.long, device=device).view(1, num_codebooks)

        delay = prev_delay.to(device=device, dtype=torch.long)
        rem = prev_rem.to(device=device, dtype=torch.long)
        old_done = prev_done.to(device=device, dtype=torch.bool)
        old_has_codes = prev_has_codes.to(device=device, dtype=torch.bool)
        old_codes = prev_codes.to(device=device, dtype=torch.long)
        step_count = prev_step_count.to(device=device, dtype=torch.long)
        seeds = row_seeds.to(device=device, dtype=torch.long)
        updates = update_mask.to(device=device, dtype=torch.bool)
        ramp = rem >= 0

        # MusicGen delay masking. ``cb_logits`` is the private output of this
        # state step and has no downstream reader, so mutate it directly rather
        # than cloning the full [batch, codebook, vocab] tensor every AR step.
        # Keep the operation ordering identical to the legacy in-place path
        # because BOC/EOC masking is quality sensitive.
        masked_logits = cb_logits
        lock_until = num_codebooks - rem
        locked = ramp.unsqueeze(1) & (q < lock_until.unsqueeze(1))
        eoc_logits = masked_logits[:, :, EOC_ID].clone()
        masked_logits.masked_fill_(locked.unsqueeze(-1), float("-inf"))
        masked_logits[:, :, EOC_ID] = torch.where(locked, eoc_logits, masked_logits[:, :, EOC_ID])

        ramp_tail = ramp.unsqueeze(1) & (q >= lock_until.unsqueeze(1))
        masked_logits[:, :, BOC_ID] = torch.where(
            ramp_tail,
            torch.full_like(masked_logits[:, :, BOC_ID], float("-inf")),
            masked_logits[:, :, BOC_ID],
        )
        masked_logits[:, :, EOC_ID] = torch.where(
            ramp_tail,
            torch.full_like(masked_logits[:, :, EOC_ID], float("-inf")),
            masked_logits[:, :, EOC_ID],
        )

        normal = ~ramp
        allowed_until = torch.clamp(delay + 1, max=num_codebooks)
        delayed = normal.unsqueeze(1) & (q >= allowed_until.unsqueeze(1))
        boc_logits = masked_logits[:, :, BOC_ID].clone()
        masked_logits.masked_fill_(delayed.unsqueeze(-1), float("-inf"))
        masked_logits[:, :, BOC_ID] = torch.where(delayed, boc_logits, masked_logits[:, :, BOC_ID])

        allowed = normal.unsqueeze(1) & (q < allowed_until.unsqueeze(1))
        masked_logits[:, :, BOC_ID] = torch.where(
            allowed,
            torch.full_like(masked_logits[:, :, BOC_ID], float("-inf")),
            masked_logits[:, :, BOC_ID],
        )
        nonzero_allowed = allowed & (q > 0)
        masked_logits[:, :, EOC_ID] = torch.where(
            nonzero_allowed,
            torch.full_like(masked_logits[:, :, EOC_ID], float("-inf")),
            masked_logits[:, :, EOC_ID],
        )

        sampled = HiggsAudioV3TalkerForConditionalGeneration._sample_audio_codes_tensor(
            masked_logits.reshape(num_rows * num_codebooks, -1),
            row_seeds=seeds,
            row_steps=step_count,
            num_codebooks=num_codebooks,
            use_flashinfer=use_flashinfer,
            flashinfer_top_k_row_states=flashinfer_top_k_row_states,
            flashinfer_top_p_workspace=flashinfer_top_p_workspace,
        ).view(num_rows, num_codebooks)

        # Leading BOC delay pad.
        delay_active = (~ramp) & ((delay + 1) < num_codebooks)
        delay_mask = delay_active.unsqueeze(1) & (q > delay.unsqueeze(1))
        codes = torch.where(delay_mask, torch.full_like(sampled, BOC_ID), sampled)
        next_delay = torch.where(delay_active, delay + 1, delay)

        # Trailing EOC ramp-down and new EOC detection.
        ramp_mask = ramp.unsqueeze(1) & (q < lock_until.unsqueeze(1))
        codes = torch.where(ramp_mask, torch.full_like(codes, EOC_ID), codes)
        ramp_next_rem = rem - 1
        eos_mask = (~ramp).unsqueeze(1) & (codes == EOC_ID)
        eos_idx_values = torch.where(eos_mask, q.expand_as(codes), torch.full_like(codes, -1))
        last_eos_idx = eos_idx_values.max(dim=1).values
        has_eos = last_eos_idx >= 0
        eos_prefix = has_eos.unsqueeze(1) & (q <= last_eos_idx.unsqueeze(1))
        codes = torch.where(eos_prefix, torch.full_like(codes, EOC_ID), codes)
        normal_next_rem = torch.where(
            has_eos,
            num_codebooks - last_eos_idx - 1,
            torch.full_like(last_eos_idx, -1),
        )
        next_rem = torch.where(ramp, ramp_next_rem, normal_next_rem)
        done = ((next_rem >= 0) & (next_rem <= 0)) & updates
        valid = updates

        next_delay = torch.where(done, torch.zeros_like(next_delay), next_delay)
        next_rem = torch.where(done, torch.full_like(next_rem, -1), next_rem)
        write_delay = torch.where(updates, next_delay, delay)
        write_rem = torch.where(updates, next_rem, rem)
        write_done = torch.where(updates, done, old_done)
        write_codes = torch.where(updates.unsqueeze(1), codes, old_codes)
        write_has_codes = torch.where(updates, ~done, old_has_codes)
        write_step_count = step_count + updates.to(step_count.dtype)
        staged_codes = torch.where(updates.unsqueeze(1), codes, torch.full_like(codes, -1))

        return (
            staged_codes,
            write_delay,
            write_rem,
            write_done,
            write_has_codes,
            write_codes,
            write_step_count,
            valid,
            done,
        )

    # ------------------------------------------------------------------ omni output
    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs

        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information")
        if info_dicts is None:
            info_dicts = []

        audio_codes_list: list[torch.Tensor] = []
        any_nonempty = False
        for info in info_dicts:
            ac: torch.Tensor | None = None
            if isinstance(info, dict):
                codes_field = info.get("codes")
                if isinstance(codes_field, dict):
                    ac = codes_field.get("audio")
                else:
                    ac = info.get("audio_codes")
            if isinstance(ac, torch.Tensor) and ac.numel() > 0:
                audio_codes_list.append(ac)
                any_nonempty = True
            else:
                audio_codes_list.append(torch.empty(0, dtype=torch.long))

        if any_nonempty:
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={"codes": {"audio": audio_codes_list}},
            )
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs=None)

    # ------------------------------------------------------------------ weight loading

    # Per-layer suffixes from the actual V3 checkpoint (results/higgs_v3_checkpoint_analysis.txt)
    _V3_LAYER_SUFFIXES = (
        "input_layernorm.weight",
        "mlp.down_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "post_attention_layernorm.weight",
        "self_attn.k_norm.weight",
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    )

    @classmethod
    def _build_required_keys(cls, num_layers: int) -> set[str]:
        """Build the exact set of required V3 checkpoint keys."""
        keys = {
            "tied.embedding.text_embedding.weight",
            "body.norm.weight",
            f"{_MODALITY_EMBEDDING_PREFIX}weight",
        }
        for i in range(num_layers):
            for suffix in cls._V3_LAYER_SUFFIXES:
                keys.add(f"body.layers.{i}.{suffix}")
        return keys

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        loaded_params: set[str] = set()
        own_params = dict(self.named_parameters())
        seen_checkpoint_keys: set[str] = set()

        for name, tensor in weights:
            seen_checkpoint_keys.add(name)

            mapped = self._map_weight_name(name)
            if mapped is None:
                continue

            if mapped.startswith("model.") or mapped.startswith("lm_head."):
                backbone_weights.append((mapped, tensor))
            elif mapped in own_params:
                param = own_params[mapped]
                if param.shape != tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {mapped}: expected {param.shape}, "
                        f"got {tensor.shape} (checkpoint key: {name})"
                    )
                param.data.copy_(tensor.to(param.dtype))
                loaded_params.add(mapped)

        if backbone_weights:
            backbone_module = _BackboneWrapper(self.model, self.lm_head, self._backbone_config)
            loaded = backbone_module.load_weights(iter(backbone_weights))
            loaded_params.update(loaded)

        # Resolve special token IDs from the tokenizer
        model_path = getattr(self.vllm_config.model_config, "model", None)
        if model_path:
            self.config.resolve_special_tokens(model_path)
        self._resolve_token_ids()

        # Verify every required checkpoint key was seen.
        num_layers = int(self._backbone_config.num_hidden_layers)
        required = self._build_required_keys(num_layers)
        missing = required - seen_checkpoint_keys
        if missing:
            raise RuntimeError(
                f"HiggsAudioV3Talker: {len(missing)} required checkpoint keys missing: {sorted(missing)[:5]}..."
            )

        logger.info(
            "HiggsAudioV3Talker: loaded %d params, modality_embedding=%s, tied=%s",
            len(loaded_params),
            tuple(self.multimodal_embedding.weight.shape),
            self.tie_modality,
        )
        return loaded_params

    def _map_weight_name(self, name: str) -> str | None:
        if name.startswith(_CODEC_PREFIX):
            return None
        if name.startswith(_MODALITY_HEAD_PREFIX):
            if self.tie_modality:
                return None
            return name.replace(_MODALITY_HEAD_PREFIX, "modality_head.")
        if name.startswith(_MODALITY_EMBEDDING_PREFIX):
            return name.replace(_MODALITY_EMBEDDING_PREFIX, "multimodal_embedding.")
        for ckpt_prefix, model_prefix in _BACKBONE_PREFIX_MAP.items():
            if name.startswith(ckpt_prefix):
                return name.replace(ckpt_prefix, model_prefix, 1)
        # Reject unexpected non-codec Higgs checkpoint keys
        raise ValueError(
            f"Unexpected checkpoint key with no known mapping: {name!r}. "
            f"Known prefixes: {list(_BACKBONE_PREFIX_MAP.keys())}, "
            f"{_MODALITY_EMBEDDING_PREFIX!r}, {_MODALITY_HEAD_PREFIX!r}, {_CODEC_PREFIX!r}"
        )


class _BackboneWrapper(nn.Module):
    """Wrapper to use AutoWeightsLoader for Qwen3 backbone."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, model, lm_head, config):
        super().__init__()
        self.model = model
        self.lm_head = lm_head
        self.config = config

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import AutoWeightsLoader

        skip = ["lm_head."] if getattr(self.config, "tie_word_embeddings", False) else None
        loader = AutoWeightsLoader(self, skip_prefixes=skip)
        return loader.load_weights(weights)
