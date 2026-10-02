# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict batched codec-to-waveform stage for MiniCPM-o 4.5."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

import soundfile as sf
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.device_copy import to_device_nonblocking

from .batched_token2wav import (
    BatchedToken2Wav,
    BatchedToken2WavState,
    row_offset_signature,
    state_shape_signature,
)
from .cuda_graph_wrapper import ResidentAttCache, _memory_snapshot

logger = init_logger(__name__)


def _resolve_model_dir(model_ref: str, revision: str | None = None) -> str:
    """Resolve ``model_ref`` to a local directory containing the repo assets.

    ``model_config.model`` is a filesystem path in local deployments but a
    Hugging Face repo id in hub/CI deployments; the prompt-audio and
    token2wav asset lookups need a real directory either way.
    """
    if Path(model_ref).is_dir():
        return model_ref
    from vllm_omni.transformers_utils.repo_utils import hf_api

    return hf_api().snapshot_download(model_ref, revision=revision, allow_patterns=["assets/*"])


def _tf32_mode(extra: Mapping[str, Any]) -> str:
    """Stage-2 TF32 scope: ``"off"`` (default), ``"flow"`` (the CFM DiT only) or ``"all"``.

    Set by the connector extra ``code2wav_allow_tf32`` (false / "flow" / true);
    ``token2wav_allow_tf32: true`` (upstream's switch) always means ``"all"``.
    Off by default: on A800 TF32 everywhere moves the log-mel by 0.7-1.4 dB (L1)
    against fp32, 10-30x fp32's own seed-to-seed spread. ``"flow"`` keeps HiFT
    in fp32 and speeds up the DiT GEMMs, the top Stage-2 kernel.
    """
    if bool(extra.get("token2wav_allow_tf32", False)):
        return "all"
    value: Any = extra.get("code2wav_allow_tf32", False)
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "flow":
            return "flow"
        return "all" if value in ("1", "true", "all", "yes") else "off"
    return "all" if bool(value) else "off"


def _batch_error(reason: str, **details: Any) -> RuntimeError:
    payload = {"reason": reason, **details}
    return RuntimeError(f"MiniCPMO45Code2WavBatchError {json.dumps(payload, sort_keys=True)}")


def _scalar(value: Any, default: Any = None) -> Any:
    if isinstance(value, torch.Tensor):
        return value.reshape(-1)[0].item() if value.numel() else default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _scalar(value[0], default) if value else default
    return default if value is None else value


_REF_TARGET_SAMPLE_RATE = 24000
# Overridable per deployment via ``ref_audio_max_seconds``.
_REF_MAX_SECONDS = 6.0
# Channel ceiling for the optional (channels, samples) downmix guard.
_REF_MAX_CHANNELS = 8


@lru_cache(maxsize=8)
def _get_resampler(orig_freq: int, new_freq: int):
    """Cached anti-aliased resampler matching token2wav's own loader.

    ``torchaudio`` is imported lazily (it is not a hard runtime requirement of
    this model on every platform) and the transform is built once per rate pair
    instead of once per request, keeping it off the first-packet path.
    """
    import torchaudio

    return torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)


def _read_reference_wav(path: str) -> tuple[Any, int]:
    """Read a WAV file for :func:`_normalize_reference`.

    ``soundfile`` returns ``(samples, channels)`` for multi-channel files
    while the normalizer expects ``(channels, samples)``. Without the
    transpose a stereo prompt looks like a huge channel count, the downmix
    guard rejects it, and the caller silently falls back to the raw file --
    a second L0 value for the same content.
    """
    waveform, sample_rate_hz = sf.read(path, dtype="float32", always_2d=False)
    if getattr(waveform, "ndim", 1) > 1:
        waveform = waveform.T
    return waveform, int(sample_rate_hz)


def _normalize_reference(
    ref_audio: Any,
    sample_rate_hz: int,
    *,
    max_seconds: float = _REF_MAX_SECONDS,
) -> tuple[torch.Tensor, int]:
    """Normalize a per-request reference waveform onto a shared grid.

    MiniCPM-o streaming uses the reference-audio length as the CFM attention
    cache origin (L0). Unnormalized references give every request a distinct
    L0, which explodes the CFM CUDA-graph cache key space (see #6628). Fold
    every reference onto the same sample rate and a fixed length (truncate
    long ones, zero-pad short ones) so L0 becomes one constant value.

    References longer than ``max_seconds`` are truncated with a warning;
    the window is configurable via ``ref_audio_max_seconds``.
    """
    tensor = torch.as_tensor(ref_audio, dtype=torch.float32)
    if tensor.dim() > 1:
        # (channels, samples) -> mono; plain reshape(-1) would interleave.
        # Upstream flattens request references to 1-D, so anything else is a
        # non-standard caller: reject layouts we cannot downmix unambiguously
        # instead of silently averaging the waveform away.
        if tensor.shape[0] > _REF_MAX_CHANNELS:
            raise ValueError(f"reference audio must be 1-D or (channels, samples); got shape {tuple(tensor.shape)}")
        tensor = tensor.mean(dim=0)
    waveform = tensor.reshape(-1).cpu().contiguous()
    if waveform.numel() == 0:
        # Keep the caller's "empty_ref_audio" error path intact: an empty
        # waveform must not be zero-padded into a valid-length reference.
        return waveform, _REF_TARGET_SAMPLE_RATE
    if sample_rate_hz != _REF_TARGET_SAMPLE_RATE:
        # Match token2wav's own loader (torchaudio, anti-aliased): it only
        # resamples when the stored rate differs, so this output is what the
        # model consumes.
        waveform = _get_resampler(sample_rate_hz, _REF_TARGET_SAMPLE_RATE)(waveform.view(1, -1)).view(-1).contiguous()
    max_samples = max(1, int(max_seconds * _REF_TARGET_SAMPLE_RATE))
    if waveform.numel() > max_samples:
        logger.warning(
            "Reference audio is %.2fs, truncating to the %.2fs window (set ``ref_audio_max_seconds`` to keep more).",
            waveform.numel() / _REF_TARGET_SAMPLE_RATE,
            max_seconds,
        )
        waveform = waveform[:max_samples].contiguous()
    elif waveform.numel() < max_samples:
        # Zero-pad short references so every request shares one L0; trailing
        # silence has minimal style impact (verified via E3 WER/SIM).
        waveform = torch.nn.functional.pad(waveform, (0, max_samples - waveform.numel()))
    return waveform, _REF_TARGET_SAMPLE_RATE


def _codec_tensor(value: Any, fallback: torch.Tensor) -> torch.Tensor:
    # Codec ids arrive from the connector as host data. A pageable host copy
    # would block the host once per request per step; stage them pinned.
    if isinstance(value, torch.Tensor):
        return to_device_nonblocking(value.reshape(-1).to(dtype=torch.long), fallback.device)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return to_device_nonblocking(torch.as_tensor(value, dtype=torch.long).reshape(-1), fallback.device)
    return fallback.reshape(-1).to(dtype=torch.long)


# Keys the runner stamps on every step regardless of stage input (see
# OmniGPUModelRunner._preprocess and the NPU _gather_runtime_additional_information
# override). A step carrying only these has no producer payload at all.
_RUNNER_STAMPED_KEYS = frozenset({"request_id", "req_id", "generated_len", "meta"})
_PRODUCER_META_KEYS = frozenset(
    {
        "cache_epoch",
        "chunk_seq",
        "code_flat_numel",
        "last_chunk",
        "prompt_cache_id",
        "prompt_wav",
        "ref_audio_sr",
    }
)


def _carries_stage_payload(info: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    """Whether this step carries anything the Talker stage actually sent.

    Any real async-chunk payload brings producer metadata along, whether the
    transport delivers it nested under ``meta`` or as flattened ``meta.*`` keys.
    """
    codes = info.get("codes")
    if isinstance(codes, Mapping) and any(value is not None for value in codes.values()):
        return True
    if any(key not in _RUNNER_STAMPED_KEYS for key in info):
        return True
    return any(meta.get(key) is not None for key in _PRODUCER_META_KEYS)


@dataclass(frozen=True)
class _RequestState:
    cache_epoch: int
    chunk_seq: int
    prompt_cache_id: str
    prompt_wav: str
    token2wav: BatchedToken2WavState


@dataclass
class _RuntimePrompt:
    cache_id: str
    path: str
    owners: set[str]


@dataclass(frozen=True)
class _WorkItem:
    output_index: int
    state_id: str
    request_id: str
    cache_epoch: int
    chunk_seq: int
    prompt_cache_id: str
    prompt_wav: str
    last_chunk: bool
    tokens: torch.Tensor
    previous: _RequestState | None
    runtime_prompt_key: str | None
    duplex_epoch: int
    duplex_turn_id: int
    segment_text_utf8: torch.Tensor
    tts_is_last_chunk: bool
    segment_end: bool
    turn_end: bool
    has_payload: bool = True


def _resident_signature(item: _WorkItem) -> bool:
    """Whether a continuation row's estimator attention cache lives in the slot pool (not a tensor)."""
    if item.previous is None:
        return False
    cache = item.previous.token2wav.flow_cache.get("estimator_att_cache")
    return cache is not None and not isinstance(cache, torch.Tensor)


def _slot_resident(item: _WorkItem) -> bool:
    """Whether a continuation row's estimator attention cache is a Whole-Euler slot (``ResidentAttCache``)."""
    if item.previous is None:
        return False
    return isinstance(item.previous.token2wav.flow_cache.get("estimator_att_cache"), ResidentAttCache)


class _BucketStats:
    """``code2wav_bucket_stats`` (connector extra, default off): how the scheduler's rows split into decodes.

    One line per ``every`` forwards: row-count distribution, exact-shape buckets
    and decode calls per forward, rows per decode, and why rows of one forward
    did not share a bucket (each extra bucket is compared with the forward's
    first: prompt, cache epoch, a fresh stream next to a continuation, or the
    slot-pool / tensor cache signature of two continuations) or
    why a mixed onset group or a row-offset bucket fell back to its
    signature buckets.
    """

    def __init__(self, every: int = 200) -> None:
        self.every = every
        self._reset()

    def _reset(self) -> None:
        self.forwards = 0
        self.rows: dict[int, int] = {}
        self.buckets = 0
        self.decodes = 0
        self.decoded_rows = 0
        self.multi_row_forwards = 0
        self.multi_row_buckets = 0
        self.reasons: dict[str, int] = {}

    def split_reason(self, first: _WorkItem, other: _WorkItem, *, cross_turn: bool, row_offsets: bool = False) -> str:
        if (first.prompt_cache_id, first.prompt_wav) != (other.prompt_cache_id, other.prompt_wav):
            return "prompt"
        if not cross_turn and first.cache_epoch != other.cache_epoch:
            same_signature = MiniCPMO45Code2Wav._bucket_key(
                first, cross_turn=True, row_offsets=row_offsets
            ) == MiniCPMO45Code2Wav._bucket_key(other, cross_turn=True, row_offsets=row_offsets)
            if same_signature:
                return "cache_epoch"
        if (first.previous is None) != (other.previous is None):
            # A fresh stream next to a continuation: the onset split, whatever the cache layout.
            return "onset"
        if _resident_signature(first) or _resident_signature(other):
            return "signature_resident"
        return "signature_tensor"

    def reason(self, name: str) -> None:
        self.reasons[name] = self.reasons.get(name, 0) + 1

    def record_forward(
        self, rows: int, buckets: list[list[_WorkItem]], *, cross_turn: bool, row_offsets: bool = False
    ) -> None:
        self.forwards += 1
        self.rows[rows] = self.rows.get(rows, 0) + 1
        self.buckets += len(buckets)
        if rows > 1:
            self.multi_row_forwards += 1
            self.multi_row_buckets += len(buckets)
        if len(buckets) > 1:
            first = buckets[0][0]
            for bucket in buckets[1:]:
                self.reason(self.split_reason(first, bucket[0], cross_turn=cross_turn, row_offsets=row_offsets))

    def record_decode(self, rows: int) -> None:
        self.decodes += 1
        self.decoded_rows += rows

    def maybe_log(self) -> None:
        if self.forwards < self.every:
            return
        logger.info(
            "Code2Wav buckets: %d forwards rows=%s buckets/forward %.2f (multi-row %.2f over %d) "
            "decodes/forward %.2f rows/decode %.2f split=%s",
            self.forwards,
            dict(sorted(self.rows.items())),
            self.buckets / self.forwards,
            self.multi_row_buckets / self.multi_row_forwards if self.multi_row_forwards else 0.0,
            self.multi_row_forwards,
            self.decodes / self.forwards,
            self.decoded_rows / self.decodes if self.decodes else 0.0,
            dict(sorted(self.reasons.items())),
        )
        self._reset()


class MiniCPMO45Code2Wav(nn.Module):
    """LLM_GENERATION model with request-owned state and compatible batching."""

    input_modalities = "audio"
    have_multimodal_outputs = True
    enable_update_additional_information = True
    replace_runtime_additional_information = True
    requires_raw_input_tokens = True
    requires_request_ids = True
    requires_exact_input_shape = True
    has_preprocess = False
    has_postprocess = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.model_path = str(vllm_config.model_config.model)
        self._model_revision = getattr(vllm_config.model_config, "revision", None)
        self.backend: BatchedToken2Wav | None = None
        self._states: dict[str, _RequestState] = {}
        self._runtime_prompts: OrderedDict[str, _RuntimePrompt] = OrderedDict()
        self._request_prompt_keys: dict[str, str] = {}
        self._runtime_prompt_dir = tempfile.TemporaryDirectory(
            prefix="minicpmo45-runtime-prompts-",
        )
        extra = self._extra_config()
        self._runtime_prompt_cache_size = int(extra.get("token2wav_runtime_prompt_cache_size", 4))
        if self._runtime_prompt_cache_size < 0:
            raise ValueError("MiniCPM-o Code2Wav runtime prompt cache capacity must be >= 0")
        self._setup_cache_size = int(extra.get("token2wav_setup_cache_size", 1))
        if self._setup_cache_size < 0:
            raise ValueError("MiniCPM-o Code2Wav setup cache capacity must be >= 0")
        self._connector_config = {
            "codec_chunk_frames": int(extra.get("codec_chunk_frames", 25)),
            "initial_codec_chunk_frames": int(extra.get("initial_codec_chunk_frames", 0)),
            "codec_left_context_frames": int(extra.get("codec_left_context_frames", 3)),
            "hift_max_lazy_graphs": int(extra.get("hift_max_lazy_graphs", 8)),
            # Additional codec chunk sizes whose first and continuation
            # vocoder shapes get HiFT graphs, e.g. [25] for full-duplex units
            # under a 75-token turn chunk.
            "hift_graph_codec_chunk_frames": list(extra.get("hift_graph_codec_chunk_frames") or ()),
            # Exact-shape HiFT graphs (default off): inclusive [first, last]
            # codec-frame ranges of first chunks (no source cache) and of
            # continuations, captured at hift_graph_exact_batch_sizes (default
            # [1]) and replayed unpadded instead of running eager.
            "hift_graph_first_chunk_frames": list(extra.get("hift_graph_first_chunk_frames") or ()),
            "hift_graph_continuation_frames": list(extra.get("hift_graph_continuation_frames") or ()),
            "hift_graph_exact_batch_sizes": list(extra.get("hift_graph_exact_batch_sizes") or (1,)),
        }
        if self._connector_config["codec_chunk_frames"] <= 0 or self._connector_config["codec_left_context_frames"] < 0:
            raise ValueError(f"Invalid MiniCPM-o connector chunk configuration: {self._connector_config}")
        # The most requests any Stage-2 batch call ever holds; both the HiFT
        # graph batch ladder and the Whole-Euler micro-batch size are sized
        # off it below.
        max_num_seqs = getattr(getattr(vllm_config, "scheduler_config", None), "max_num_seqs", None)
        raw_capture_batch_sizes = extra.get("hift_graph_capture_batch_sizes")
        if raw_capture_batch_sizes is None:
            # Powers of two up to the scheduler's batch (capped at 32), so every
            # vocoder batch rounds up to a graph (HiFTGraphWrapper.replay).
            batch_cap = min(int(max_num_seqs), 32) if max_num_seqs else 32
            capture_batch_sizes = []
            size = 1
            while size < batch_cap:
                capture_batch_sizes.append(size)
                size *= 2
            capture_batch_sizes.append(batch_cap)
        else:
            capture_batch_sizes = raw_capture_batch_sizes
        max_serial_batch = extra.get("max_serial_batch")
        max_serial_batch = 4 if max_serial_batch is None else int(max_serial_batch)
        self._hift_graph_config = {
            "enabled": bool(extra.get("enable_hift_graph", False)),
            "capture_batch_sizes": capture_batch_sizes,
            "max_serial_batch": max_serial_batch,
        }
        enable_whole_euler = extra.get("enable_whole_euler")
        # The Whole-Euler micro batch (one arena attention cache per row) and
        # graph-path gate follow max_num_seqs alone: the old micro_batch_size /
        # max_graph_batch overrides could contradict it and are not read.
        micro_batch_size = int(max_num_seqs) if max_num_seqs else 16
        self._cfm_graph_config = {
            "enabled": bool(extra.get("enable_cfm_graph", False)),
            "max_graphs": int(extra.get("cfm_max_graphs", 32)),
            "bucket_frames": int(extra.get("cfm_graph_bucket_frames", 0)),
            "capture_frames": extra.get("cfm_graph_capture_frames"),
            # Cache lengths snap onto a grid of this many frames (0: exact),
            # so first chunks of any size share a few graphs.
            "offset_bucket_frames": int(extra.get("cfm_graph_offset_bucket_frames", 50)),
            "enable_whole_euler": enable_whole_euler is None or bool(enable_whole_euler),
            "max_serial_batch": max_serial_batch,
            "max_graph_batch": micro_batch_size,
            "micro_batch_size": micro_batch_size,
            "pad_max_rows": extra.get("whole_euler_pad_max_rows"),
            # Fused DiT block forward for the ragged / Whole-Euler solves
            # (``dit_fused.py``): same function, ~1/3 of the kernels.
            "fused_body": bool(extra.get("cfm_fused_body", False)),
            # Request estimator caches resident in a Whole-Euler slot pool
            # (``AttSlotPool``, needs fused_body): no per-replay cache copies.
            "slot_pool": bool(extra.get("cfm_slot_pool", False)),
            # Fused-body GEMM backend: cublas (default), or Triton TF32 tiles
            # sized for the streaming row counts (``triton``; ``triton_windows``
            # also runs each causal conv as one GEMM over its im2col windows).
            "dit_gemm": extra.get("cfm_dit_gemm"),
            # Its products: tf32, or tf32x3 (three TF32 products, fp32-level
            # error); unset follows code2wav_allow_tf32 (flow: tf32).
            "dit_gemm_precision": extra.get("cfm_dit_gemm_precision"),
            # Keep one immutable prompt estimator-cache suffix per voice and
            # only a request-owned prefix in each state. Disabled until a CUDA
            # probe confirms the suffix is bit-identical.
            "prompt_att_sharing": bool(extra.get("cfm_prompt_att_sharing", False)),
            # Arena storage follows the largest explicitly captured graph tier
            # instead of max_num_seqs; larger batches replay multiple tiers.
            "arena_rows_from_graph_grid": bool(extra.get("cfm_arena_rows_from_graph_grid", False)),
            # Slot-pool ragged solves keep each row's cache offset, so a
            # stream's second chunk and steady ones share one replay.
            "row_offset_merge": extra.get("cfm_row_offset_merge", False) is True,
        }
        # Exact-shape CUDA graphs of the flow encoder's continuation chunk
        # (``cfm_encoder_cuda_graph``, default off; ``FlowEncoderGraphs``):
        # one per (rows, tokens, conformer cache frames), captured at startup
        # for the default prompt, replayed only for calls of exactly that
        # shape. ``cfm_encoder_graph_rows`` is the largest row count (every
        # count up to it) or a list of counts; ``cfm_encoder_graph_token_widths``
        # defaults to the duplex unit chunk (left context + one unit).
        encoder_rows = extra.get("cfm_encoder_graph_rows", 8)
        if isinstance(encoder_rows, (list, tuple)):
            encoder_rows = sorted({int(r) for r in encoder_rows})
        else:
            encoder_rows = list(range(1, int(encoder_rows) + 1))
        if max_num_seqs:
            encoder_rows = [r for r in encoder_rows if r <= int(max_num_seqs)]
        if any(r < 1 for r in encoder_rows):
            raise ValueError("MiniCPM-o cfm_encoder_graph_rows must be positive")
        encoder_widths = extra.get("cfm_encoder_graph_token_widths")
        self._encoder_graph_config = {
            "enabled": extra.get("cfm_encoder_cuda_graph", False) is True,
            "rows": encoder_rows,
            "token_widths": [int(w) for w in encoder_widths] if encoder_widths else None,
        }
        self._ref_max_seconds = float(extra.get("ref_audio_max_seconds", _REF_MAX_SECONDS))
        if self._ref_max_seconds <= 0:
            raise ValueError("MiniCPM-o Code2Wav ref_audio_max_seconds must be > 0")
        self._min_batch_size = int(extra.get("code2wav_min_batch_size", 1))
        if self._min_batch_size < 1:
            raise ValueError("MiniCPM-o Code2Wav code2wav_min_batch_size must be >= 1")
        self._initial_batch_size = int(extra.get("code2wav_initial_batch_size", 0))
        if self._initial_batch_size < 0:
            raise ValueError("MiniCPM-o Code2Wav code2wav_initial_batch_size must be >= 0")
        if self._initial_batch_size and self._initial_batch_size < self._min_batch_size:
            raise ValueError("MiniCPM-o Code2Wav code2wav_initial_batch_size must be 0 or >= code2wav_min_batch_size")
        self._default_prompt_id = str(extra.get("prompt_cache_id", "HT_ref_audio"))
        self._prompt_wav_override = extra.get("prompt_wav")
        self._default_prompt_normalized: tuple[str, str] | None = None
        # A fresh stream normally has a different cache-shape signature from
        # a continuation.  Ragged Whole-Euler can keep estimator offsets per
        # request, so an explicit opt-in may join those rows when supported.
        self._onset_merge = bool(extra.get("cfm_onset_merge", False))
        # Decode rows of different turns of one stream family together:
        # cache_epoch only decides whether a row starts a fresh state
        # (_parse_item); the decode itself never reads it.
        self._cross_turn_buckets = extra.get("cfm_cross_turn_buckets", False) is True
        # Decode a stream's second chunk with steady ones: slot-pool rows whose
        # caches differ only in length share a bucket and one Whole-Euler
        # replay (``BatchedToken2Wav.can_merge_row_offsets``).
        self._row_offset_merge = self._cfm_graph_config["row_offset_merge"]
        self._bucket_stats = _BucketStats() if extra.get("code2wav_bucket_stats", False) is True else None
        # Return the caching allocator's unused blocks to the driver once the
        # startup precapture is done (``cfm_precapture_empty_cache``, default
        # off). Startup leaves the eager warmup/prompt solves' blocks cached;
        # releasing them changes no tensor, only what the process holds.
        self._precapture_empty_cache = extra.get("cfm_precapture_empty_cache", False) is True

    @property
    def _default_prompt_wav(self) -> str:
        if self._prompt_wav_override is not None:
            return str(self._prompt_wav_override)
        return str(Path(self.model_path) / "assets" / "HT_ref_audio.wav")

    def _normalized_default_prompt(self) -> tuple[str, str]:
        """Fold the shipped default prompt onto the request-reference grid.

        The shipped asset is 6.016 s and is loaded directly by token2wav, so a
        request without a reference would otherwise keep a second L0 value a
        couple of frames away from the normalized request references -- two
        graph-key families instead of one (both use ``ref_audio_max_seconds``).
        Returns ``(prompt_wav,
        prompt_cache_id)``; if the asset cannot be read, the shipped path is
        returned unchanged.
        """
        if self._default_prompt_normalized is not None:
            return self._default_prompt_normalized
        source = self._default_prompt_wav
        fallback = (source, self._default_prompt_id)
        try:
            waveform, sample_rate_hz = _read_reference_wav(source)
            normalized, target_sr = _normalize_reference(
                waveform,
                sample_rate_hz,
                max_seconds=self._ref_max_seconds,
            )
        except Exception:
            logger.warning("Could not normalize the default prompt %s; using it as-is", source)
            self._default_prompt_normalized = fallback
            return self._default_prompt_normalized
        if normalized.numel() == 0:
            self._default_prompt_normalized = fallback
            return self._default_prompt_normalized
        digest = sha256()
        digest.update(normalized.numpy().tobytes())
        digest.update(str(target_sr).encode())
        cache_id = f"default-ref-{digest.hexdigest()[:24]}-{target_sr}"
        path = Path(self._runtime_prompt_dir.name) / f"{cache_id}.wav"
        if not path.is_file():
            sf.write(path, normalized.numpy(), target_sr, format="WAV")
        logger.info("Default prompt normalized onto the %.2fs window: %s", self._ref_max_seconds, path)
        self._default_prompt_normalized = (str(path), cache_id)
        return self._default_prompt_normalized

    def _extra_config(self) -> dict[str, Any]:
        model_config = getattr(self.vllm_config, "model_config", None)
        connector = getattr(model_config, "stage_connector_config", None)
        if isinstance(connector, Mapping):
            extra = connector.get("extra", connector)
        else:
            extra = getattr(connector, "extra", None)
        return dict(extra) if isinstance(extra, Mapping) else {}

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.numel(), 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def _materialize_runtime_prompt(
        self,
        ref_audio: Any,
        sample_rate: Any,
    ) -> tuple[str, _RuntimePrompt]:
        sample_rate_hz = int(_scalar(sample_rate, 0))
        if sample_rate_hz <= 0:
            raise _batch_error("invalid_ref_audio_sample_rate", sample_rate=sample_rate_hz)
        waveform, sample_rate_hz = _normalize_reference(
            ref_audio,
            sample_rate_hz,
            max_seconds=self._ref_max_seconds,
        )
        if waveform.numel() == 0:
            raise _batch_error("empty_ref_audio")
        if not bool(torch.isfinite(waveform).all().item()):
            raise _batch_error("non_finite_ref_audio")

        digest = sha256()
        digest.update(waveform.numpy().tobytes())
        digest.update(str(sample_rate_hz).encode())
        cache_key = digest.hexdigest()
        cache_id = f"runtime-ref-{cache_key[:24]}-{sample_rate_hz}"
        path = str(Path(self._runtime_prompt_dir.name) / f"minicpmo45_ref_{cache_key[:24]}_{sample_rate_hz}.wav")
        entry = self._runtime_prompts.get(cache_key)
        if entry is None:
            entry = _RuntimePrompt(cache_id=cache_id, path=path, owners=set())
            self._runtime_prompts[cache_key] = entry
        else:
            self._runtime_prompts.move_to_end(cache_key)
        prompt_path = Path(entry.path)
        if not prompt_path.is_file():
            with tempfile.NamedTemporaryFile(
                dir=prompt_path.parent,
                prefix=f".{prompt_path.stem}-",
                suffix=".wav",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
            try:
                sf.write(
                    temporary_path,
                    waveform.numpy(),
                    sample_rate_hz,
                    format="WAV",
                )
                os.replace(temporary_path, prompt_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        return cache_key, entry

    def _resolve_prompt(
        self,
        state_id: str,
        info: Mapping[str, Any],
        meta: Mapping[str, Any],
        previous: _RequestState | None,
    ) -> tuple[str, str, str | None]:
        codes = info.get("codes")
        ref_audio = codes.get("ref") if isinstance(codes, Mapping) else None
        if ref_audio is not None:
            cache_key, entry = self._materialize_runtime_prompt(
                ref_audio,
                meta.get("ref_audio_sr"),
            )
            logger.debug(
                "MiniCPM-o Code2Wav selected runtime reference prompt_cache_id=%s prompt_wav=%s",
                entry.cache_id,
                entry.path,
            )
            return entry.cache_id, entry.path, cache_key

        if previous is not None:
            return previous.prompt_cache_id, previous.prompt_wav, self._request_prompt_keys.get(state_id)

        cache_key = self._request_prompt_keys.get(state_id)
        entry = self._runtime_prompts.get(cache_key) if cache_key is not None else None
        if entry is not None:
            return entry.cache_id, entry.path, cache_key

        default_wav, default_cache_id = self._normalized_default_prompt()
        return (
            str(_scalar(meta.get("prompt_cache_id"), default_cache_id)),
            str(_scalar(meta.get("prompt_wav"), default_wav)),
            None,
        )

    def _release_request_prompt(self, state_id: str) -> None:
        cache_key = self._request_prompt_keys.pop(state_id, None)
        entry = self._runtime_prompts.get(cache_key) if cache_key is not None else None
        if entry is None:
            return
        entry.owners.discard(state_id)
        self._trim_runtime_prompts()

    def _evict_runtime_prompt(self, cache_key: str, entry: _RuntimePrompt) -> None:
        if self.backend is not None:
            self.backend.evict_prompt(entry.cache_id, entry.path)
        Path(entry.path).unlink(missing_ok=True)
        self._runtime_prompts.pop(cache_key, None)

    def _commit_runtime_prompt_owners(self, items: list[_WorkItem]) -> None:
        for item in items:
            cache_key = item.runtime_prompt_key
            if cache_key is None:
                continue
            previous_key = self._request_prompt_keys.get(item.state_id)
            entry = self._runtime_prompts.get(cache_key)
            if entry is None:
                continue
            if previous_key != cache_key:
                previous = self._runtime_prompts.get(previous_key) if previous_key is not None else None
                if previous is not None:
                    previous.owners.discard(item.state_id)
            entry.owners.add(item.state_id)
            self._request_prompt_keys[item.state_id] = cache_key
            self._runtime_prompts.move_to_end(cache_key)
        self._trim_runtime_prompts()

    def _trim_runtime_prompts(self) -> None:
        """Evict least-recent unowned references, never request-owned state."""
        while len(self._runtime_prompts) > self._runtime_prompt_cache_size:
            victim = next(((key, entry) for key, entry in self._runtime_prompts.items() if not entry.owners), None)
            if victim is None:
                return
            self._evict_runtime_prompt(*victim)

    @staticmethod
    def _split_segments(input_ids: torch.Tensor, counts: Any) -> list[torch.Tensor]:
        flat = input_ids.reshape(-1)
        if counts is None:
            return [flat]
        if not isinstance(counts, Sequence) or isinstance(counts, (str, bytes, bytearray)):
            raise _batch_error("invalid_seq_token_counts", value_type=type(counts).__name__)
        normalized = [int(value) for value in counts]
        if any(value < 0 for value in normalized):
            raise _batch_error("negative_seq_token_count", counts=normalized)
        if sum(normalized) != int(flat.numel()):
            raise _batch_error(
                "seq_token_count_mismatch",
                counts=normalized,
                total=int(flat.numel()),
            )
        return list(torch.split(flat, normalized))

    def _parse_item(
        self,
        index: int,
        state_id: str,
        segment: torch.Tensor,
        info: Mapping[str, Any],
    ) -> _WorkItem:
        meta = info.get("meta")
        if not isinstance(meta, Mapping):
            meta = info
        request_id = str(_scalar(meta.get("request_id"), _scalar(info.get("request_id"), "")))
        if not _carries_stage_payload(info, meta):
            # The producer attached nothing to this step, only the bookkeeping
            # the runner stamps on every request. The stage was scheduled on
            # the placeholder prompt that async-chunk pre-warm submits before
            # the first codec window arrives: those tokens are reserved slots,
            # not codec data, and one bogus frame is shorter than the vocoder's
            # lookahead window. Such a step carries no producer metadata at
            # all, so it cannot be held to the payload contract below either.
            return _WorkItem(
                output_index=index,
                state_id=state_id,
                request_id=request_id or state_id,
                cache_epoch=0,
                chunk_seq=0,
                prompt_cache_id=self._default_prompt_id,
                prompt_wav=self._default_prompt_wav,
                last_chunk=False,
                tokens=segment.new_empty(0, dtype=torch.long),
                previous=None,
                runtime_prompt_key=None,
                duplex_epoch=-1,
                duplex_turn_id=-1,
                segment_text_utf8=torch.empty(0, dtype=torch.uint8),
                tts_is_last_chunk=False,
                segment_end=False,
                turn_end=False,
                has_payload=False,
            )
        if not request_id:
            raise _batch_error("missing_request_id", output_index=index)
        cache_epoch = int(_scalar(meta.get("cache_epoch"), 0))
        chunk_seq = int(_scalar(meta.get("chunk_seq"), 0))
        if cache_epoch < 0 or chunk_seq < 0:
            raise _batch_error(
                "negative_stream_position",
                request_id=request_id,
                cache_epoch=cache_epoch,
                chunk_seq=chunk_seq,
            )
        last_chunk = bool(_scalar(meta.get("last_chunk"), False))
        tts_is_last_chunk = bool(_scalar(meta.get("tts_is_last_chunk"), False))
        codes = info.get("codes")
        audio = codes.get("audio") if isinstance(codes, Mapping) else None
        tokens = _codec_tensor(audio, segment)
        if int(_scalar(meta.get("code_flat_numel"), tokens.numel())) == 0:
            # The generation scheduler reserves one placeholder token for an
            # empty terminal or segment-boundary chunk. The producer's
            # explicit length is the authority, so do not decode that
            # placeholder as codec data.
            tokens = segment.new_empty(0, dtype=torch.long)
        previous = self._states.get(state_id)
        if previous is None:
            if chunk_seq != 0:
                raise _batch_error(
                    "missing_state_for_chunk",
                    request_id=request_id,
                    cache_epoch=cache_epoch,
                    chunk_seq=chunk_seq,
                )
        elif cache_epoch < previous.cache_epoch:
            raise _batch_error(
                "stale_cache_epoch",
                request_id=request_id,
                expected=previous.cache_epoch,
                actual=cache_epoch,
            )
        elif cache_epoch > previous.cache_epoch:
            if chunk_seq != 0:
                raise _batch_error(
                    "new_epoch_requires_first_chunk",
                    request_id=request_id,
                    cache_epoch=cache_epoch,
                    chunk_seq=chunk_seq,
                )
            previous = None
        elif chunk_seq != previous.chunk_seq + 1:
            raise _batch_error(
                "stale_or_reordered_chunk",
                request_id=request_id,
                expected=previous.chunk_seq + 1,
                actual=chunk_seq,
            )
        prompt_cache_id, prompt_wav, runtime_prompt_key = self._resolve_prompt(
            state_id,
            info,
            meta,
            previous,
        )
        if previous is not None and prompt_cache_id != previous.prompt_cache_id:
            raise _batch_error(
                "prompt_changed_midstream",
                request_id=request_id,
                expected=previous.prompt_cache_id,
                actual=prompt_cache_id,
            )
        if previous is not None and prompt_wav != previous.prompt_wav:
            raise _batch_error(
                "prompt_changed_midstream",
                request_id=request_id,
                expected=previous.prompt_wav,
                actual=prompt_wav,
            )
        segment_text_utf8 = meta.get("llm_output_text_utf8")
        if not isinstance(segment_text_utf8, torch.Tensor):
            segment_text_utf8 = torch.empty(0, dtype=torch.uint8)
        return _WorkItem(
            output_index=index,
            state_id=state_id,
            request_id=request_id,
            cache_epoch=cache_epoch,
            chunk_seq=chunk_seq,
            prompt_cache_id=prompt_cache_id,
            prompt_wav=prompt_wav,
            last_chunk=last_chunk,
            tokens=tokens,
            previous=previous,
            runtime_prompt_key=runtime_prompt_key,
            duplex_epoch=int(_scalar(meta.get("duplex_epoch"), -1)),
            duplex_turn_id=int(_scalar(meta.get("duplex_turn_id"), -1)),
            segment_text_utf8=segment_text_utf8,
            tts_is_last_chunk=tts_is_last_chunk,
            segment_end=bool(_scalar(meta.get("segment_end"), False)),
            turn_end=bool(_scalar(meta.get("turn_end"), False)),
        )

    @staticmethod
    def _bucket_key(item: _WorkItem, *, cross_turn: bool = False, row_offsets: bool = False) -> tuple[Any, ...]:
        cache_signature: Any
        if item.previous is None:
            cache_signature = ("uninitialized",)
        elif row_offsets and _slot_resident(item):
            # A slot-pool row keeps its cache lengths per row (``cfm_row_offset_merge``).
            cache_signature = ("row_offsets", row_offset_signature(item.previous.token2wav))
        else:
            cache_signature = state_shape_signature(item.previous.token2wav)
        if cross_turn:
            return (item.prompt_cache_id, item.prompt_wav, cache_signature)
        return (
            item.prompt_cache_id,
            item.prompt_wav,
            cache_signature,
            item.cache_epoch,
        )

    def _signature_groups(
        self, bucket: list[_WorkItem], states: list[BatchedToken2WavState], *, row_offsets: bool
    ) -> list[list[int]]:
        """A bucket's rows grouped by cache signature, for a bucket that cannot decode as one.

        With ``row_offsets`` (``cfm_row_offset_merge``) the slot-pool groups
        that differ only in their cache lengths stay one group when the
        backend can solve them together.
        """
        groups: dict[tuple[Any, ...], list[int]] = {}
        for row, state in enumerate(states):
            groups.setdefault(state_shape_signature(state), []).append(row)
        if not row_offsets:
            return list(groups.values())
        clusters: dict[tuple[Any, ...], list[list[int]]] = {}
        for signature, rows in groups.items():
            if _slot_resident(bucket[rows[0]]):
                signature = ("row_offsets", row_offset_signature(states[rows[0]]))
            clusters.setdefault(signature, []).append(rows)
        result: list[list[int]] = []
        for parts in clusters.values():
            rows = sorted(row for part in parts for row in part)
            if len(parts) > 1 and self.backend.can_merge_row_offsets(
                [states[row] for row in rows], [int(bucket[row].tokens.numel()) for row in rows]
            ):
                result.append(rows)
            else:
                result.extend(parts)
        return result

    @staticmethod
    def _onset_group_key(key: tuple[Any, ...], *, cross_turn: bool = False) -> tuple[Any, ...]:
        """The onset-merge group of a bucket key: its prompt, plus its cache epoch unless rows merge across turns."""
        if cross_turn:
            return key[:2]
        prompt_cache_id, prompt_wav, _, cache_epoch = key
        return (prompt_cache_id, prompt_wav, cache_epoch)

    def _iter_limited_batches(
        self,
        buckets: Iterable[list[_WorkItem]],
        *,
        maximum: int,
    ) -> Iterable[list[_WorkItem]]:
        for bucket in buckets:
            if not bucket:
                continue
            if not maximum:
                yield bucket
                continue
            batch_count = (len(bucket) + maximum - 1) // maximum
            if len(bucket) < batch_count * self._min_batch_size:
                raise _batch_error(
                    "initial_batch_partition_below_minimum",
                    size=len(bucket),
                    minimum=self._min_batch_size,
                    maximum=maximum,
                )
            batch_size, larger_batches = divmod(len(bucket), batch_count)
            start = 0
            for batch_index in range(batch_count):
                stop = start + batch_size + (batch_index < larger_batches)
                yield bucket[start:stop]
                start = stop

    def _iter_decode_batches(
        self,
        buckets: Iterable[list[_WorkItem]],
    ) -> Iterable[list[_WorkItem]]:
        for bucket in buckets:
            if not self._initial_batch_size:
                yield bucket
                continue
            initial = [item for item in bucket if item.chunk_seq <= 1]
            steady = [item for item in bucket if item.chunk_seq > 1]

            undersized = [
                {"wave": wave, "size": len(items)}
                for wave, items in (("initial", initial), ("steady", steady))
                if items and len(items) < self._min_batch_size
            ]
            if undersized:
                raise _batch_error(
                    "decode_wave_below_minimum",
                    minimum=self._min_batch_size,
                    waves=undersized,
                )
            yield from self._iter_limited_batches([initial], maximum=self._initial_batch_size)
            if steady:
                yield steady

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        # This stage owns the vocoder process. Restore its previous matmul
        # policy after eager execution/capture; cuDNN's policy is independent.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            if self._extra_config().get("token2wav_allow_tf32", False):
                torch.backends.cuda.matmul.allow_tf32 = True
            return self._forward_impl(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                runtime_additional_information,
                **kwargs,
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32

    @torch.inference_mode()
    def _forward_impl(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        if getattr(self, "_precapture_pending", False) and self.backend is not None:
            try:
                self._precapture_default_prompt()
            finally:
                if getattr(self, "_precapture_empty_cache", False):
                    self._release_precapture_cache()
        ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.empty(0, dtype=torch.long)
        segments = self._split_segments(ids, kwargs.get("seq_token_counts"))
        empty = torch.empty(0, dtype=torch.float32, device=ids.device)
        sample_rate = torch.tensor(24000, dtype=torch.int32)
        if not runtime_additional_information:
            count = len(segments)
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty for _ in range(count)],
                    "sr": [sample_rate for _ in range(count)],
                },
            )
        if len(runtime_additional_information) != len(segments):
            raise _batch_error(
                "runtime_info_count_mismatch",
                segments=len(segments),
                runtime_infos=len(runtime_additional_information),
            )
        if self.backend is None:
            # load_format=dummy (CI core_model runs) skips model.load_weights()
            # entirely, but Token2wav's assets live beside the checkpoint rather
            # than in its weight iterator, so they still have to be loaded for
            # this stage to produce anything. Build them on first use, outside
            # inference mode so the parameters are ordinary tensors.
            logger.warning_once(
                "MiniCPM-o Code2Wav backend was not built during weight loading "
                "(load_format=%s); loading Token2wav assets now.",
                getattr(getattr(self.vllm_config, "load_config", None), "load_format", "unknown"),
            )
            with torch.inference_mode(False), torch.no_grad():
                self._build_backend()

        state_ids = kwargs.get("request_ids")
        if state_ids is None:
            state_ids = []
            for index, info in enumerate(runtime_additional_information):
                if not isinstance(info, Mapping):
                    state_ids.append(str(index))
                    continue
                meta = info.get("meta")
                source = meta if isinstance(meta, Mapping) else info
                state_ids.append(str(_scalar(source.get("request_id"), index)))
        if len(state_ids) != len(segments):
            raise _batch_error(
                "request_id_count_mismatch",
                segments=len(segments),
                request_ids=len(state_ids),
            )
        items: list[_WorkItem] = []
        try:
            for index, (state_id, segment, info) in enumerate(
                zip(state_ids, segments, runtime_additional_information, strict=True)
            ):
                if not isinstance(info, Mapping):
                    raise _batch_error(
                        "invalid_runtime_info",
                        output_index=index,
                        value_type=type(info).__name__,
                    )
                items.append(self._parse_item(index, str(state_id), segment, info))
        except Exception:
            self._trim_runtime_prompts()
            raise
        state_ids = [item.state_id for item in items]
        if len(state_ids) != len(set(state_ids)):
            self._trim_runtime_prompts()
            raise _batch_error("duplicate_request_in_forward", request_ids=state_ids)
        outputs = [empty for _ in segments]
        sentinels = [item for item in items if item.last_chunk and item.tokens.numel() == 0]
        segment_markers = [
            item for item in items if not item.last_chunk and item.tts_is_last_chunk and item.tokens.numel() == 0
        ]
        compute_items = [item for item in items if item.tokens.numel() > 0]
        invalid_empty = [
            item.request_id
            for item in items
            if item.has_payload and not item.last_chunk and not item.tts_is_last_chunk and item.tokens.numel() == 0
        ]
        if invalid_empty:
            self._trim_runtime_prompts()
            raise _batch_error("empty_nonfinal_chunk", request_ids=invalid_empty)

        cross_turn = getattr(self, "_cross_turn_buckets", False) is True
        row_offsets = getattr(self, "_row_offset_merge", False) is True
        bucket_stats = getattr(self, "_bucket_stats", None)
        buckets: dict[tuple[Any, ...], list[_WorkItem]] = {}
        for item in compute_items:
            buckets.setdefault(self._bucket_key(item, cross_turn=cross_turn, row_offsets=row_offsets), []).append(item)
        if bucket_stats is not None:
            bucket_stats.record_forward(
                len(compute_items), list(buckets.values()), cross_turn=cross_turn, row_offsets=row_offsets
            )
        undersized = [
            {
                "size": len(bucket),
                "request_ids": [item.request_id for item in bucket],
                "codec_len": int(bucket[0].tokens.numel()),
            }
            for bucket in buckets.values()
            if len(bucket) < self._min_batch_size
        ]
        if undersized:
            self._trim_runtime_prompts()
            raise _batch_error(
                "exact_shape_bucket_below_minimum",
                minimum=self._min_batch_size,
                buckets=undersized,
            )

        pending: dict[str, _RequestState | None] = {item.state_id: None for item in sentinels}
        pending.update(
            {
                item.state_id: _RequestState(
                    cache_epoch=item.cache_epoch,
                    chunk_seq=item.chunk_seq,
                    prompt_cache_id=item.prompt_cache_id,
                    prompt_wav=item.prompt_wav,
                    token2wav=item.previous.token2wav,
                )
                for item in segment_markers
                if item.previous is not None
            }
        )
        initial_marker_buckets: dict[tuple[str, str], list[_WorkItem]] = {}
        for item in segment_markers:
            if item.previous is None:
                initial_marker_buckets.setdefault(
                    (item.prompt_cache_id, item.prompt_wav),
                    [],
                ).append(item)
        for bucket in self._iter_limited_batches(
            initial_marker_buckets.values(),
            maximum=self._initial_batch_size,
        ):
            try:
                features = self.backend.prepare_prompt(
                    bucket[0].prompt_cache_id,
                    bucket[0].prompt_wav,
                )
                states = self.backend.setup_batch(features, len(bucket))
            except Exception as exc:
                self._trim_runtime_prompts()
                if isinstance(exc, RuntimeError) and str(exc).startswith("MiniCPMO45Code2WavBatchError "):
                    raise
                raise _batch_error(
                    "backend_unsupported_or_failed",
                    request_ids=[item.request_id for item in bucket],
                    error_type=type(exc).__name__,
                    error=str(exc),
                ) from exc
            if len(states) != len(bucket):
                self._trim_runtime_prompts()
                raise _batch_error(
                    "backend_result_size_mismatch",
                    expected=len(bucket),
                    states=len(states),
                )
            for item, state in zip(bucket, states, strict=True):
                pending[item.state_id] = _RequestState(
                    cache_epoch=item.cache_epoch,
                    chunk_seq=item.chunk_seq,
                    prompt_cache_id=item.prompt_cache_id,
                    prompt_wav=item.prompt_wav,
                    token2wav=state,
                )
        decode_buckets: Iterable[list[_WorkItem]]
        if self._onset_merge and self.backend is not None and self.backend._whole_euler_ragged_active():
            # Keep ordinary exact-shape buckets unchanged, but offer rows from
            # the same prompt/epoch to the ragged Whole-Euler path when a
            # fresh onset and a continuation are ready together.
            grouped: dict[tuple[Any, ...], list[list[_WorkItem]]] = {}
            for key, bucket in buckets.items():
                grouped.setdefault(self._onset_group_key(key, cross_turn=cross_turn), []).append(bucket)
            merged: list[list[_WorkItem]] = []
            for entries in grouped.values():
                combined = [item for bucket in entries for item in bucket]
                mixed = any(item.previous is None for item in combined) and any(
                    item.previous is not None for item in combined
                )
                merged.extend([combined] if mixed else entries)
            decode_buckets = merged
        else:
            decode_buckets = buckets.values()
        for bucket in self._iter_decode_batches(decode_buckets):
            batch_size = len(bucket)
            try:
                features = self.backend.prepare_prompt(
                    bucket[0].prompt_cache_id,
                    bucket[0].prompt_wav,
                )
                fresh = [item for item in bucket if item.previous is None]
                setup = iter(self.backend.setup_batch(features, len(fresh)) if fresh else [])
                states = [next(setup) if item.previous is None else item.previous.token2wav for item in bucket]
                token_lengths = {int(item.tokens.numel()) for item in bucket}
                last_chunk_values = {item.last_chunk for item in bucket}
                onset_mix = (
                    self._onset_merge
                    and any(item.previous is None for item in bucket)
                    and any(item.previous is not None for item in bucket)
                )
                onset_ready = onset_mix and self.backend.can_merge_state_shapes(
                    states, [int(item.tokens.numel()) for item in bucket]
                )
                row_groups: list[list[int]] | None = None
                offset_mix = False
                if onset_mix and not onset_ready:
                    # The opt-in grouping is deliberately conservative. If
                    # cache layouts do not satisfy ragged Whole-Euler, split
                    # back into the historical signature buckets.
                    if bucket_stats is not None:
                        bucket_stats.reason("onset_merge_rejected")
                    row_groups = self._signature_groups(bucket, states, row_offsets=row_offsets)
                elif row_offsets and not onset_ready and batch_size > 1:
                    groups = self._signature_groups(bucket, states, row_offsets=True)
                    if len(groups) > 1:
                        # Slot-pool rows the backend cannot solve at once go
                        # back to their signature buckets.
                        if bucket_stats is not None:
                            bucket_stats.reason("row_offset_merge_rejected")
                        row_groups = groups
                    else:
                        # One decode, ragged when it holds rows of different cache lengths.
                        offset_mix = len({state_shape_signature(state) for state in states}) > 1
                if row_groups is not None:
                    audios_rows: list[torch.Tensor | None] = [None] * batch_size
                    next_states_rows: list[Any] = [None] * batch_size
                    for rows in row_groups:
                        if bucket_stats is not None:
                            bucket_stats.record_decode(len(rows))
                        row_tokens = [bucket[row].tokens for row in rows]
                        row_states = [states[row] for row in rows]
                        row_last = [bucket[row].last_chunk for row in rows]
                        if (
                            len({int(tokens.numel()) for tokens in row_tokens}) > 1
                            or len(set(row_last)) > 1
                            or (row_offsets and len({state_shape_signature(state) for state in row_states}) > 1)
                        ):
                            row_audios, row_next = self.backend.decode_ragged_batch(
                                row_tokens,
                                features,
                                row_states,
                                last_chunks=row_last,
                            )
                        else:
                            row_audios, row_next = self.backend.decode_batch(
                                torch.stack(row_tokens, dim=0),
                                features,
                                row_states,
                                last_chunk=row_last[0],
                            )
                        for row, audio, next_state in zip(rows, row_audios, row_next, strict=True):
                            audios_rows[row] = audio
                            next_states_rows[row] = next_state
                    audios = cast(list[torch.Tensor], audios_rows)
                    next_states = next_states_rows
                elif onset_ready or offset_mix or len(token_lengths) > 1 or len(last_chunk_values) > 1:
                    if bucket_stats is not None:
                        bucket_stats.record_decode(batch_size)
                    audios, next_states = self.backend.decode_ragged_batch(
                        [item.tokens for item in bucket],
                        features,
                        states,
                        last_chunks=[item.last_chunk for item in bucket],
                    )
                else:
                    if bucket_stats is not None:
                        bucket_stats.record_decode(batch_size)
                    tokens = torch.stack([item.tokens for item in bucket], dim=0)
                    audios, next_states = self.backend.decode_batch(
                        tokens,
                        features,
                        states,
                        last_chunk=bucket[0].last_chunk,
                    )
            except Exception as exc:
                self._trim_runtime_prompts()
                if isinstance(exc, RuntimeError) and str(exc).startswith("MiniCPMO45Code2WavBatchError "):
                    raise
                raise _batch_error(
                    "backend_unsupported_or_failed",
                    request_ids=[item.request_id for item in bucket],
                    error_type=type(exc).__name__,
                    error=str(exc),
                ) from exc
            if len(audios) != batch_size or len(next_states) != batch_size:
                self._trim_runtime_prompts()
                raise _batch_error(
                    "backend_result_size_mismatch",
                    expected=batch_size,
                    audios=len(audios),
                    states=len(next_states),
                )
            for item, audio, next_state in zip(bucket, audios, next_states, strict=True):
                outputs[item.output_index] = audio.reshape(-1).to(dtype=torch.float32)
                pending[item.state_id] = (
                    None
                    if item.last_chunk
                    else _RequestState(
                        cache_epoch=item.cache_epoch,
                        chunk_seq=item.chunk_seq,
                        prompt_cache_id=item.prompt_cache_id,
                        prompt_wav=item.prompt_wav,
                        token2wav=next_state,
                    )
                )

        if bucket_stats is not None:
            bucket_stats.maybe_log()
        self._commit_runtime_prompt_owners(items)
        for request_id, state in pending.items():
            if state is None:
                self._states.pop(request_id, None)
            else:
                self._states[request_id] = state
        sample_rate_tensor = torch.as_tensor(sample_rate, dtype=torch.int32)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": outputs,
                "sr": [sample_rate_tensor.clone() for _ in outputs],
                # Generation runner wire payloads are flat and tensor-only.
                # Dotted metadata keys are unflattened again by the output
                # processor before the full-duplex data plane consumes them.
                "meta.duplex_epoch": [torch.tensor(item.duplex_epoch, dtype=torch.int32) for item in items],
                "meta.duplex_turn_id": [torch.tensor(item.duplex_turn_id, dtype=torch.int32) for item in items],
                "meta.llm_output_text_utf8": [item.segment_text_utf8 for item in items],
                "meta.tts_is_last_chunk": [torch.tensor(item.tts_is_last_chunk, dtype=torch.bool) for item in items],
                "meta.segment_end": [torch.tensor(item.segment_end, dtype=torch.bool) for item in items],
                "meta.turn_end": [torch.tensor(item.turn_end, dtype=torch.bool) for item in items],
            },
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for request_id in finished_req_ids:
            state_id = str(request_id)
            self._states.pop(state_id, None)
            self._release_request_prompt(state_id)

    def make_omni_output(self, model_outputs: Any, **_: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"MiniCPMO45Code2Wav expected OmniOutput, got {type(model_outputs).__name__}")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        for _ in weights:
            pass
        self._build_backend()
        # Token2wav loads flow.pt and hift.pt inside its constructor instead of
        # from the parent MiniCPM checkpoint iterator. Report those registered
        # parameters as initialized so vLLM's strict loader audit does not
        # misclassify the independently loaded Stage-2 weights as missing.
        return {name for name, _ in self.named_parameters()}

    def _build_backend(self) -> None:
        """Load the Token2wav assets that back this stage."""
        if self.backend is not None:
            return

        # In-tree adapter over StepAudio2Token2WavCore on every platform. It
        # matches the external `stepaudio2-minicpmo` Token2wav bit-for-bit on
        # CUDA (same cosyvoice2 flow/DiT modules and weights) while dropping
        # that package's hard-coded `.cuda()` calls, and auto-applies the
        # Ascend fixes (HiFT linear downsample, DiT mask expand, MATH SDPA)
        # on NPU.
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_token2wav import (
            MiniCPMO45Token2wav as Token2wav,
        )
        from vllm_omni.platforms import current_omni_platform

        extra = self._extra_config()
        # Hub repo ids only need to become local directories once the vocoder
        # assets are actually read; unit tests construct this model with fake
        # paths and must not trigger a hub download (#5442).
        self.model_path = _resolve_model_dir(self.model_path, self._model_revision)
        prompt_path = Path(self._default_prompt_wav)
        if not prompt_path.is_file():
            raise FileNotFoundError(f"MiniCPM-o Code2Wav prompt audio not found: {prompt_path}")
        token2wav_path = Path(self.model_path) / "assets" / "token2wav"
        if not token2wav_path.is_dir():
            raise FileNotFoundError(f"MiniCPM-o Code2Wav assets not found: {token2wav_path}")
        # Token2wav runs in fp32, so without TF32 every flow-DiT GEMM runs on
        # SIMT cores (the top Stage-2 kernel on A800); cuDNN convolutions
        # already default to TF32. Stage 2 owns its process, so the global flag
        # stays within it; it is set before any CUDA graph is captured.
        tf32_mode = _tf32_mode(extra) if current_omni_platform.is_cuda() else "off"
        if tf32_mode == "all":
            torch.backends.cuda.matmul.allow_tf32 = True
        if tf32_mode != "off":
            logger.info("MiniCPM-o Code2Wav: TF32 matmul enabled (%s)", tf32_mode)
        use_float16 = bool(extra.get("token2wav_float16", False))
        previous_dtype = torch.get_default_dtype()
        try:
            # vLLM constructs bf16 models under a bf16 default-dtype context.
            # Token2wav contains fp32-only S3Tokenizer/HiFT modules, so build
            # its independent assets in their native precision.
            torch.set_default_dtype(torch.float32)
            # Connector extra ``token2wav_s3tokenizer_device`` (default unset:
            # same device as the vocoder). "cpu" frees ~472 MiB of fp32
            # S3Tokenizer weights; it only runs when a new reference voice is
            # prepared (startup default prompt, runtime ``ref`` audio misses).
            tokenizer_device = extra.get("token2wav_s3tokenizer_device")
            tokenizer_kwargs = {"audio_tokenizer_device": str(tokenizer_device)} if tokenizer_device else {}
            token2wav = Token2wav(
                str(token2wav_path),
                float16=use_float16,
                n_timesteps=int(extra.get("token2wav_n_timesteps", 10)),
                # The batched backend builds its estimator caches itself and
                # never runs the upstream chunk path that owns these buffers.
                drop_upstream_chunk_att_buffers=extra.get("cfm_drop_upstream_att_buffer", False) is True,
                **tokenizer_kwargs,
            )
        finally:
            torch.set_default_dtype(previous_dtype)

        trt_stepper = None
        use_trt = bool(extra.get("token2wav_trt", False)) or os.environ.get("MINICPMO_TOKEN2WAV_TRT", "") == "1"
        # TensorRT is CUDA-only; other platforms ignore the toggle.
        if use_trt and current_omni_platform.is_cuda():
            from vllm_omni.model_executor.models.step_audio2.step_audio2_dit_trt import build_dit_trt_stepper

            dtype_name = str(
                extra.get("token2wav_trt_dtype", os.environ.get("MINICPMO_TOKEN2WAV_TRT_DTYPE", "fp16"))
            ).lower()
            trt_dtype = torch.float32 if dtype_name in ("fp32", "float32") else torch.float16
            max_batch = int(extra.get("token2wav_trt_max_batch", 16))
            device = next(token2wav.flow.parameters()).device
            trt_stepper = build_dit_trt_stepper(
                token2wav.flow.decoder.estimator,
                device=device,
                dtype=trt_dtype,
                max_batch=max_batch,
            )
            logger.info("MiniCPM-o Code2Wav: DiT estimator running on TensorRT (%s)", dtype_name)
            token2wav.enable_trt_spk_embedding()
            logger.info("MiniCPM-o Code2Wav: campplus speaker embedding running on TensorRT")
        self.backend = BatchedToken2Wav(
            token2wav,
            trt_stepper=trt_stepper,
            connector_config=self._connector_config,
            hift_graph_config=self._hift_graph_config,
            cfm_graph_config=self._cfm_graph_config,
            bfloat16_attention_cache=bool(extra.get("code2wav_bfloat16_attention_cache", False)),
            setup_cache_size=self._setup_cache_size,
            cfm_tf32=tf32_mode == "flow",
            encoder_graph_config=getattr(self, "_encoder_graph_config", None),
        )
        # Captured by the first forward (the engine's warmup run): graphs captured
        # while vLLM loads the weights held GiBs each, a few MiB at forward time.
        self._precapture_pending = bool(extra.get("cfm_graph_precapture", True))

    def _release_precapture_cache(self) -> None:
        """Release the allocator blocks startup left cached (``cfm_precapture_empty_cache``).

        Only unused cached segments go back to the driver: live tensors, the
        slot pool and the CUDA-graph pool's segments (held while their graphs
        live) stay where they are, so every later solve reads the same values.
        """
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
        device = torch.device("cuda", torch.accelerator.current_device_index())
        torch.accelerator.synchronize(device)
        before = _memory_snapshot(device)
        torch.accelerator.empty_cache()
        after = _memory_snapshot(device)
        if before is not None and after is not None:
            logger.info(
                "MiniCPM-o Code2Wav: released %.0f MiB of cached memory after precapture "
                "(allocated %.0f MiB, reserved %.0f -> %.0f MiB)",
                (before[1] - after[1]) / 2**20,
                after[0] / 2**20,
                before[1] / 2**20,
                after[1] / 2**20,
            )

    def _precapture_default_prompt(self) -> None:
        """Capture the HiFT and default-voice Whole-Euler graphs before serving.

        Otherwise each graph is captured by the first chunk that needs it,
        stalling every stream for 1-2 s per graph.
        """
        self._precapture_pending = False
        assert self.backend is not None
        started = time.perf_counter()
        try:
            hift_captured = self.backend.precapture_hift()
        except Exception:
            logger.warning("MiniCPM-o Code2Wav: HiFT precapture failed; graphs capture lazily", exc_info=True)
            hift_captured = 0
        if hift_captured:
            logger.info(
                "MiniCPM-o Code2Wav: precaptured %d HiFT CUDA Graph shapes in %.1f s",
                hift_captured,
                time.perf_counter() - started,
            )
        prompt_wav, prompt_cache_id = self._normalized_default_prompt()
        try:
            features = self.backend.prepare_prompt(prompt_cache_id, prompt_wav)
            captured = self.backend.precapture_whole_euler(features)
        except Exception:
            logger.warning("MiniCPM-o Code2Wav: Whole-Euler precapture failed; graphs capture lazily", exc_info=True)
            return
        if captured:
            logger.info(
                "MiniCPM-o Code2Wav: precaptured %d Whole-Euler graphs for the default prompt in %.1f s",
                captured,
                time.perf_counter() - started,
            )
        if getattr(self.backend, "_encoder_graphs", None) is not None:
            try:
                self.backend.precapture_flow_encoder(features)
            except Exception:
                logger.warning(
                    "MiniCPM-o Code2Wav: flow encoder precapture failed; the encoder stays eager", exc_info=True
                )
