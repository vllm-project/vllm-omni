# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

import functools
import math
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.utils.device_copy import index_to_device

from .cuda_graph_wrapper import (
    CFMGraphWrapper,
    HiFTGraphWrapper,
    ResidentAttCache,
    WholeEulerCFMGraphWrapper,
    _att_keep_ranges,
    _copy_frame_segments,
    _euler_step,
    _euler_timeline,
    _format_memory_delta,
    _materialize_att_rows,
    _memory_snapshot,
    _zero_padded_cnn_cache,
    empty_hift_outputs,
)
from .dit_fused import (
    blocks_forward_chunk_fused,
    causal_cache_index,
    dit_modulation,
    gather_causal_cache,
    supports_fused_body,
    tiled_attention_supported,
)
from .flow_encoder_graph import FlowEncoderGraphs

logger = init_logger(__name__)

_SILENCE_TOKEN = 4218
# CosyVoice2 RelPos PE is built for max_len=5000 and `forward_chunk` never
# calls ``extend_pe``. After the 2x upsample a single Code2Wav prefill longer
# than this many codec tokens overflows the PE slice (matrix_ac vs matrix_bd).
_DEFAULT_RELPOS_MAX_POS = 5000
_MAX_ENCODE_TOKEN_CAP = 1024
# Streaming cache trim (stepaudio2 ``Token2wav.stream``): past ``prompt_len +
# 100`` frames, keep the first ``prompt_len`` frames and the last 100.
_CACHE_TRIM_SUFFIX = 100
_TIMESTEP_MAX_PERIOD = 10000


def _timestep_frequencies(dim: int) -> torch.Tensor:
    """Host frequency table of ``TimestepEmbedder.timestep_embedding``."""
    half = dim // 2
    return torch.exp(-math.log(_TIMESTEP_MAX_PERIOD) * torch.arange(start=0, end=half) / half)


def _trim_streaming_cache(cache: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """The streaming trim of a cache whose frame axis is ``-2``."""
    if isinstance(cache, ResidentAttCache):
        # Its slot-pool solve already trimmed it (``WholeEulerCFMGraphWrapper._replay_slots``).
        return cache
    ranges = _att_keep_ranges(int(cache.shape[-2]), (prompt_len, _CACHE_TRIM_SUFFIX))
    if len(ranges) == 1:
        return cache
    return torch.cat([cache[..., start : start + length, :] for start, length in ranges], dim=-2)


def relpos_encode_token_budget(
    *,
    max_pos: int,
    stride: int,
    cache_offset: int,
    lookahead: int,
    cap: int = _MAX_ENCODE_TOKEN_CAP,
) -> int:
    """How many codec tokens ``forward_chunk`` can take before RelPos PE wraps.

    ``position_encoding(size)`` needs ``size <= max_pos``. After upsample,
    ``size = cache_offset * stride + ~stride * token_frames`` (plus last-chunk
    lookahead pad).
    """
    stride = max(1, int(stride))
    lookahead = max(0, int(lookahead))
    room = int(max_pos) // stride - max(0, int(cache_offset)) - lookahead - 1
    return max(lookahead + 1, min(int(cap), room))


def _supports_ragged_kernel(estimator: nn.Module) -> bool:
    """Whether ``_blocks_forward_chunk_ragged`` can run this estimator's DiT blocks."""
    blocks = getattr(estimator, "blocks", None)
    if not blocks or not hasattr(estimator, "in_proj") or not hasattr(estimator, "final_layer"):
        return False
    block = blocks[0]
    attn = getattr(block, "attn", None)
    conv_blocks = getattr(getattr(block, "conv", None), "block", None)
    return (
        hasattr(block, "adaLN_modulation")
        and all(
            hasattr(attn, name)
            for name in ("forward_chunk", "to_q", "to_k", "to_v", "to_heads", "q_norm", "k_norm", "proj")
        )
        and conv_blocks is not None
        and len(conv_blocks) >= 8
        and all(isinstance(conv_blocks[index], nn.Conv1d) for index in (1, 6))
    )


def _cfm_pad_frames(
    *,
    mel_frames: int,
    offset: int,
    noise_capacity: int,
    bucket_frames: int,
    disabled: bool,
) -> int:
    """Frame padding that aligns one CFM call onto the capture-shape grid.

    Chunk lengths vary per request: the steady-state chunk is 50 mel frames,
    but the first/last chunk and the ``plan_token2wav_encode_slices`` splits
    land on arbitrary lengths, and every distinct length is its own CUDA-graph
    capture shape. Padding the frame axis up to ``bucket_frames`` collapses
    them onto one grid (the caller trims the output back). Measured on the
    shipped config, 128 requests / concurrency 8: 61 captures and 1 cache
    flush without bucketing versus 15 captures and 0 flushes with it, audio
    RTF 1.317 -> 1.191 under otherwise identical conditions.

    The attention and CNN caches also advance by the padded width so their
    shapes stay on the same grid; in practice that is bounded by the trim in
    ``_decode_batch_once`` (the cache width saturates at ``prompt_len +
    100``) rather than by the decoder's ``rand_noise`` capacity.

    Returns 0 when bucketing does not apply: the ragged valid-lengths path,
    graphs unavailable or already disabled by ``CFMGraphWrapper._disable``,
    or padding that would overflow the decoder's noise buffer.
    """
    if disabled or bucket_frames <= 1:
        return 0
    pad = (bucket_frames - mel_frames % bucket_frames) % bucket_frames
    if offset + mel_frames + pad > noise_capacity:
        return 0
    return pad


def _zero_padded_frames(tensor: torch.Tensor, valid_frames: int | None) -> None:
    """Keep the padded columns of ``tensor`` at zero, in place.

    Bucketing pads the frame axis so the capture shape stays fixed. The padded
    columns carry no valid content: they are excluded from the attention by the
    ``attn_mask`` built in ``_decode_batch_once``, and zeroing keeps them from
    leaking back in. The integration step ``x = x + dt * velocity`` would
    otherwise make the padded region non-zero again after the initial zeroing,
    and the cross-chunk caches would read those values back.
    """
    if valid_frames is None or valid_frames >= int(tensor.shape[-1]):
        return
    tensor[..., valid_frames:] = 0.0


def plan_token2wav_encode_slices(
    num_frames: int,
    *,
    max_frames: int,
    min_nonfinal: int,
    last_chunk: bool,
) -> list[tuple[int, int]]:
    """Split a Code2Wav token span so every non-final piece has a lookahead window."""
    if num_frames <= 0:
        return []
    min_nonfinal = max(1, int(min_nonfinal))
    max_frames = max(int(max_frames), min_nonfinal)
    last_min = 1 if last_chunk else min_nonfinal
    if num_frames <= max_frames:
        return [(0, num_frames)]

    slices: list[tuple[int, int]] = []
    start = 0
    while start < num_frames:
        remaining = num_frames - start
        if remaining <= max_frames:
            slices.append((start, num_frames))
            break
        take = max_frames
        tail = remaining - take
        if tail <= max_frames and tail < last_min:
            take = remaining - last_min
        if take < min_nonfinal:
            take = remaining
        slices.append((start, start + take))
        start += take
    return slices


def _autocast_disabled(device: torch.device):
    """Disable any enclosing autocast region on ``device``.

    ``torch.amp.autocast`` resolves the autocast dtype for ``device_type``
    while constructing the context, which raises on accelerators (e.g. Ascend
    NPU) that never registered autocast support. Degrade to a no-op there: an
    enclosing region can only exist on a device type torch already knows.
    """
    try:
        return torch.amp.autocast(device.type, enabled=False)
    except (RuntimeError, TypeError, ValueError):
        return nullcontext()


def _token2wav_sdpa_context(device: torch.device):
    if device.type != "npu":
        return nullcontext()

    from vllm_omni.platforms.npu.models.step_audio2_token2wav import (
        npu_token2wav_sdpa_context,
    )

    return npu_token2wav_sdpa_context()


def tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), value.device.type


def state_shape_signature(state: BatchedToken2WavState) -> tuple[Any, ...]:
    flow = tuple((name, tensor_signature(state.flow_cache[name])) for name in sorted(state.flow_cache))
    hift = tuple((name, tensor_signature(state.hift_cache[name])) for name in sorted(state.hift_cache))
    return flow, hift


# Cache frame axes a row-offset merge keeps per row: the encoder runs once per conformer cache
# length (``decode_ragged_batch``), and a slot-pool solve attends each estimator cache in its slot.
_ROW_OFFSET_FRAME_AXES = {"conformer_att_cache": 3, "estimator_att_cache": 4}


def row_offset_signature(state: BatchedToken2WavState) -> tuple[Any, ...]:
    """``state_shape_signature`` without the cache lengths a row-offset merge keeps per row."""
    flow, hift = state_shape_signature(state)
    masked = []
    for name, (shape, dtype, device) in flow:
        axis = _ROW_OFFSET_FRAME_AXES.get(name)
        masked.append((name, (shape if axis is None else (*shape[:axis], -1, *shape[axis + 1 :]), dtype, device)))
    return tuple(masked), hift


@dataclass(frozen=True)
class PromptFeatures:
    cache_key: tuple[str, str]
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class BatchedToken2WavState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]


def _undecorate_dynamo(module: nn.Module, method: str) -> None:
    """Restore ``method`` on ``module`` if TorchDynamo wrapped it.

    ``cosyvoice2`` decorates ``UpsampleConformerEncoderV2.forward_chunk`` with
    ``torch.compile(backend="eager")``. That backend performs no Inductor
    optimisation, so the wrapper only adds tracing and guard construction, and
    duplex pays it again on every unseen chunk shape -- seconds inside a live
    response. Dropping the wrapper leaves the original implementation, which is
    what the eager backend was executing anyway.
    """
    bound = getattr(module, method, None)
    original = getattr(bound, "_torchdynamo_orig_callable", None) or getattr(bound, "__wrapped__", None)
    if original is None:
        return
    function = getattr(original, "__func__", original)
    module.__dict__[method] = function.__get__(module, type(module))
    logger.info("Bypassed TorchDynamo wrapper on %s.%s", type(module).__name__, method)


def _cfm_matmul_tf32(fn):
    """Run the flow CFM (and capture its graphs) with TF32 matmuls when ``cfm_tf32``; the rest stays fp32."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "_cfm_tf32", False):
            return fn(self, *args, **kwargs)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            return fn(self, *args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    return wrapper


class BatchedToken2Wav(nn.Module):
    """Drive Token2wav's modules with dynamically-sized, request-owned caches.

    This class intentionally never calls ``Token2wav.stream`` or
    ``Token2wav.__call__``. The upstream object is used only as a one-time
    asset loader and prompt feature extractor.
    """

    def __init__(
        self,
        token2wav: Any,
        trt_stepper: Any | None = None,
        *,
        connector_config: Mapping[str, int] | None = None,
        hift_graph_config: Mapping[str, Any] | None = None,
        cfm_graph_config: Mapping[str, Any] | None = None,
        bfloat16_attention_cache: bool = False,
        setup_cache_size: int = 1,
        cfm_tf32: bool = False,
        encoder_graph_config: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if setup_cache_size < 0:
            raise ValueError("setup_cache_size must be >= 0")
        self._token2wav = token2wav
        self._cfm_tf32 = bool(cfm_tf32)
        cfm_graph_cfg = dict(cfm_graph_config or {})
        # Ragged solves of rows whose caches differ in length (a stream's second chunk next to steady
        # ones): each row keeps its own cache offset, in one slot-pool Whole-Euler replay when it can.
        self._row_offset_merge = bool(cfm_graph_cfg.get("row_offset_merge", False))
        # Optional TrtDiTStepper (step_audio2_dit_trt): replaces only the
        # per-timestep DiT estimator call; encoder and HiFT stay on torch.
        self._trt_stepper = trt_stepper
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        enable_cached_istft = getattr(self.hift, "enable_cached_istft", None)
        if callable(enable_cached_istft):
            enable_cached_istft()
        encoder = getattr(self.flow, "encoder", None)
        if encoder is not None:
            _undecorate_dynamo(encoder, "forward_chunk")
        self._encoder_graphs = self._build_encoder_graphs(encoder_graph_config or {}, connector_config or {})
        hift_parameter = next(self.hift.parameters(), None)
        if hift_parameter is not None and hift_parameter.device.type == "cuda":
            # Prime the CUDA state used by HiFT during backend construction.
            # Otherwise, the first live audio chunk can fail when async stages
            # share one GPU.
            device = hift_parameter.device
            dtype = hift_parameter.dtype
            mel_channels = int(self.hift.conv_pre.in_channels)
            with (
                torch.inference_mode(),
                torch.random.fork_rng(devices=[device]),
                _autocast_disabled(device),
            ):
                # 50 mel frames match the default first streamed vocoder chunk.
                speech, source = self.hift.inference(
                    torch.zeros((1, mel_channels, 50), device=device, dtype=dtype),
                    torch.zeros((1, 1, 0), device=device, dtype=dtype),
                )
            torch.accelerator.synchronize(device)
            del speech, source
            torch.accelerator.empty_cache()
        self.float16 = bool(token2wav.float16)
        self._estimator_att_compute_dtype = torch.float16 if self.float16 else torch.float32
        self._estimator_att_cache_dtype = (
            torch.bfloat16 if bfloat16_attention_cache else self._estimator_att_compute_dtype
        )
        self.n_timesteps = int(token2wav.n_timesteps)
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.register_buffer(
            "speech_window",
            token2wav.speech_window.detach().clone(),
            persistent=False,
        )
        self._timeline_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, list[float]]] = {}
        self.hift_graph_wrapper: HiFTGraphWrapper | None = None
        graph_config = dict(hift_graph_config or {})
        if bool(graph_config.get("enabled", False)):
            if hift_parameter is None:
                raise ValueError("MiniCPM-o HiFT Graph requires a parameterized HiFT module")
            if hift_parameter.device.type != "cuda":
                logger.info("HiFT CUDA Graph is disabled on device type %s", hift_parameter.device.type)
            else:
                if connector_config is None:
                    raise ValueError("MiniCPM-o HiFT CUDA Graph requires connector chunk configuration")
                if self.mel_cache_len <= 0 or self.source_cache_len % self.mel_cache_len != 0:
                    raise ValueError(
                        "MiniCPM-o HiFT CUDA Graph requires source_cache_len to be divisible by mel_cache_len"
                    )
                capture_batch_sizes = graph_config.get("capture_batch_sizes", [1])
                max_serial_batch = int(graph_config.get("max_serial_batch", 4))
                logger.info(
                    "Enabling HiFT CUDA Graph with batch sizes %s (max_serial_batch=%d)",
                    capture_batch_sizes,
                    max_serial_batch,
                )
                self.hift_graph_wrapper = HiFTGraphWrapper(
                    token2wav=token2wav,
                    connector_config=dict(connector_config),
                    capture_batch_sizes=capture_batch_sizes,
                    max_serial_batch=max_serial_batch,
                )
                # Captured by the first forward (``precapture_hift``): graphs
                # captured while vLLM loads the weights hold far more memory.
        self._cfm_graph_wrapper: CFMGraphWrapper | None = None
        self._whole_euler_graph_wrapper: WholeEulerCFMGraphWrapper | None = None
        # Whether the ragged DiT body is the fused one (``dit_fused.py``).
        self._ragged_fused_body = False
        # On NPU the platform graph runner captures instead of the CUDA wrappers, so bucketing and
        # padding key off the requested flag.
        self._cfm_graph_enabled = bool(cfm_graph_cfg.get("enabled", False))
        if self._cfm_graph_enabled:
            flow_parameter = next(self.flow.parameters(), None)
            if flow_parameter is not None and flow_parameter.device.type == "cuda":
                estimator = self.flow.decoder.estimator
                max_graphs = int(cfm_graph_cfg.get("max_graphs", 32))
                max_serial_batch = int(cfm_graph_cfg.get("max_serial_batch", 4))
                max_graph_batch_cfg = cfm_graph_cfg.get("max_graph_batch")
                max_graph_batch = int(max_graph_batch_cfg) if max_graph_batch_cfg is not None else None
                micro_batch_size_cfg = cfm_graph_cfg.get("micro_batch_size")
                micro_batch_size = int(micro_batch_size_cfg) if micro_batch_size_cfg is not None else 4
                enable_whole_euler = bool(cfm_graph_cfg.get("enable_whole_euler", True))
                if bool(cfm_graph_cfg.get("fused_body", False)):
                    # A reduced-precision attention cache keeps the ragged body,
                    # which attends in the activation dtype through upstream.
                    self._ragged_fused_body = (
                        supports_fused_body(estimator) and self._estimator_att_cache_dtype == torch.float32
                    )
                    if not self._ragged_fused_body:
                        logger.warning(
                            "cfm_fused_body requested but unsupported here (DiT blocks or a non-fp32 attention cache)"
                        )
                # Resident request caches need the fused body and its tiled
                # attention, which reads keys/values out of the pool in place.
                slot_pool = bool(cfm_graph_cfg.get("slot_pool", False))
                if slot_pool and not (
                    self._ragged_fused_body and tiled_attention_supported(flow_parameter.get_device())
                ):
                    logger.warning("cfm_slot_pool requested but needs cfm_fused_body on NVIDIA SM80+; using the arena")
                    slot_pool = False
                if enable_whole_euler and self._trt_stepper is None:
                    capture_frames = cfm_graph_cfg.get("capture_frames")
                    if capture_frames is None and connector_config is not None:
                        # The steady chunk's mel width, derived like HiFTGraphWrapper's
                        # capture bucket: steady chunks run unpadded and shorter
                        # chunks share their graph under a mask.
                        capture_frames = int(connector_config["codec_chunk_frames"]) * int(
                            getattr(self.flow, "token_mel_ratio", 2)
                        )
                    if isinstance(capture_frames, (list, tuple)):
                        # Several widths (e.g. a 25-token duplex unit and a
                        # 75-token turn chunk) each get their own graphs.
                        query_bucket_frames: int | tuple[int, ...] = tuple(int(w) for w in capture_frames)
                    else:
                        query_bucket_frames = int(capture_frames if capture_frames is not None else 64)
                    self._whole_euler_graph_wrapper = WholeEulerCFMGraphWrapper(
                        estimator=estimator,
                        n_timesteps=self.n_timesteps,
                        inference_cfg_rate=getattr(self.flow.decoder, "inference_cfg_rate", 0.7),
                        att_cache_dtype=self._estimator_att_cache_dtype,
                        max_graphs=max_graphs,
                        max_serial_batch=max_serial_batch,
                        max_graph_batch=max_graph_batch,
                        micro_batch_size=micro_batch_size,
                        query_bucket_frames=query_bucket_frames,
                        pad_max_rows=cfm_graph_cfg.get("pad_max_rows"),
                        offset_bucket_frames=int(cfm_graph_cfg.get("offset_bucket_frames", 0)),
                        # Resident request caches, a slot per stream plus a few for streams
                        # that end while new ones start (``AttSlotPool``).
                        att_slots=(micro_batch_size + max(2, micro_batch_size // 8)) if slot_pool else 0,
                        # ``_ragged_body`` picks the configured body on every call, so a
                        # replaced body is the one a capture records.
                        ragged_body=self._ragged_body if _supports_ragged_kernel(estimator) else None,
                        # The fused body takes each timestep's modulation, computed once outside the graphs.
                        modulation_fn=functools.partial(dit_modulation, estimator) if self._ragged_fused_body else None,
                        row_offsets=self._row_offset_merge,
                        fused_euler_step=bool(cfm_graph_cfg.get("fused_euler_step", False)),
                    )
                    logger.info(
                        "Whole-Euler CFM CUDA Graph enabled "
                        "(max_graphs=%d, max_serial_batch=%d, max_graph_batch=%s, "
                        "micro_batch_size=%d, query_bucket_frames=%s, offset_bucket_frames=%d, fused_body=%s)",
                        max_graphs,
                        max_serial_batch,
                        str(max_graph_batch),
                        micro_batch_size,
                        query_bucket_frames,
                        self._whole_euler_graph_wrapper.offset_bucket_frames,
                        self._ragged_fused_body,
                    )
                elif enable_whole_euler:
                    logger.info("Whole-Euler CFM CUDA Graph disabled because TensorRT stepper is configured")
                self._cfm_graph_wrapper = CFMGraphWrapper(
                    graph_fn=estimator.blocks_forward_chunk,
                    max_graphs=max_graphs,
                )
                logger.info("CFM CUDA Graph enabled (max_graphs=%d)", max_graphs)
            else:
                logger.info(
                    "CFM CUDA Graph is disabled on device type %s",
                    flow_parameter.device.type if flow_parameter is not None else "unknown",
                )
        # mel-frame bucket size for the CFM CUDA Graph path. Pad each decode
        # chunk up to a multiple of this many frames so the graph cache key
        # space stays small (0 disables bucketing, e.g. when graphs are off).
        self._cfm_graph_bucket_frames = (
            int(cfm_graph_cfg.get("bucket_frames", 0))
            if (
                self._cfm_graph_wrapper is not None
                or self._whole_euler_graph_wrapper is not None
                # NPU has no wrappers; the platform runner is keyed by this flag.
                or self._cfm_graph_enabled
            )
            else 0
        )
        if self._cfm_graph_bucket_frames > 1:
            logger.info(
                "CFM CUDA Graph bucketing enabled (bucket_frames=%d)",
                self._cfm_graph_bucket_frames,
            )
        # Device copies of the DiT timestep-embedding frequency table.
        self._timestep_freqs: dict[tuple[Any, ...], torch.Tensor] = {}
        self._prompt_features: dict[tuple[str, str], PromptFeatures] = {}
        self._setup_cache_size = setup_cache_size
        self._setup_cache: OrderedDict[tuple[tuple[str, str], int], BatchedToken2WavState] = OrderedDict()

    def _hift_inference(
        self,
        mel: torch.Tensor,
        source_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(mel.shape[2]) == 0:
            # No frames, no samples: HiFT itself cannot run on an empty mel.
            return empty_hift_outputs(mel)
        if self.hift_graph_wrapper is None:
            return self.hift.inference(mel, source_cache)
        return self.hift_graph_wrapper.replay(mel, source_cache)

    def prepare_prompt(self, prompt_cache_id: str, prompt_wav: str) -> PromptFeatures:
        cache_key = (prompt_cache_id, prompt_wav)
        cached = self._prompt_features.get(cache_key)
        if cached is None:
            # The generation runner may wrap model.forward in bf16 autocast,
            # and vLLM constructs the model under a bf16 default dtype, while
            # S3Tokenizer prompt extraction uses fp32 convolution weights.
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                with _autocast_disabled(self.speech_window.device):
                    values = self._token2wav._prepare_prompt(prompt_wav)
            finally:
                torch.set_default_dtype(previous_dtype)
            cached = PromptFeatures(
                cache_key=cache_key,
                speech_tokens=values[0],
                speaker_embedding=values[2],
                mels=values[3],
            )
            self._prompt_features[cache_key] = cached
        return cached

    def evict_prompt(self, prompt_cache_id: str, prompt_wav: str) -> None:
        """Release all cached artifacts associated with one prompt."""
        prompt_key = (prompt_cache_id, prompt_wav)
        self._prompt_features.pop(prompt_key, None)
        for setup_key in list(self._setup_cache):
            if setup_key[0] == prompt_key:
                self._setup_cache.pop(setup_key)

    @staticmethod
    def _repeat_prompt(features: PromptFeatures, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (
            features.speech_tokens.expand(batch_size, -1),
            features.speaker_embedding.expand(batch_size, -1),
            features.mels.expand(batch_size, -1, -1),
        )

    def _project_speakers(self, speakers: torch.Tensor) -> torch.Tensor:
        """``spk_embed_affine_layer(normalize(speakers))`` with M=1, independent of the batch size.

        Rows expand one prompt speaker, so one row is projected: M=1 and M>=2
        GEMMs round differently (~6e-8), moving a row's CFM mel by ~1e-3.
        """
        if speakers.shape[0] > 1 and speakers.stride(0) == 0:
            projected = self.flow.spk_embed_affine_layer(F.normalize(speakers[:1], dim=1))
            return projected.expand(speakers.shape[0], -1).contiguous()
        return self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))

    def _autocast(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        if not self.float16:
            return torch.amp.autocast("cuda", enabled=False)
        return torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
        )

    def _pre_lookahead_len(self) -> int | None:
        """Right-context width of the encoder's pre-lookahead convolution.

        ``None`` when the encoder does not expose one, so callers keep working
        against encoder implementations without that layer.
        """
        layer = getattr(self.flow.encoder, "pre_lookahead_layer", None)
        width = getattr(layer, "pre_lookahead_len", None)
        return int(width) if width is not None else None

    def _relpos_max_pos(self) -> int:
        embed = getattr(self.flow.encoder, "embed", None)
        pe = getattr(embed, "pe", None)
        if isinstance(pe, torch.Tensor) and pe.numel() > 0:
            return max(1, int(pe.size(1) // 2))
        return _DEFAULT_RELPOS_MAX_POS

    def _upsample_stride(self) -> int:
        stride = getattr(getattr(self.flow.encoder, "up_layer", None), "stride", None)
        return max(1, int(stride)) if stride is not None else 2

    def _max_encode_token_frames(self, states: list[BatchedToken2WavState]) -> int:
        att = states[0].flow_cache.get("conformer_att_cache") if states else None
        offset1 = int(att.shape[3] // 2) if att is not None else 0
        lookahead = self._pre_lookahead_len() or 0
        return relpos_encode_token_budget(
            max_pos=self._relpos_max_pos(),
            stride=self._upsample_stride(),
            cache_offset=offset1,
            lookahead=lookahead,
        )

    def _ensure_relpos_pe(self, tokens: torch.Tensor, att_cache: torch.Tensor | None) -> None:
        """Grow CosyVoice RelPos PE before ``forward_chunk``, which never calls extend_pe."""
        embed = getattr(self.flow.encoder, "embed", None)
        extend_pe = getattr(embed, "extend_pe", None)
        if not callable(extend_pe):
            return
        offset1 = int(att_cache.shape[3] // 2) if att_cache is not None else 0
        lookahead = self._pre_lookahead_len() or 0
        needed = offset1 * self._upsample_stride() + self._upsample_stride() * (int(tokens.shape[1]) + lookahead + 1)
        extend_pe(tokens.new_zeros((1, max(needed, 1))))

    def _encode_chunk(
        self,
        tokens: torch.Tensor,
        *,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_relpos_pe(tokens, att_cache)
        embedded = self.flow.input_embedding(tokens)
        hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
            xs=embedded,
            last_chunk=last_chunk,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        return self.flow.encoder_proj(hidden), new_cnn, new_att

    def _flow_on_cuda(self) -> bool:
        flow_parameter = next(self.flow.parameters(), None)
        return flow_parameter is not None and flow_parameter.device.type == "cuda"

    def _build_encoder_graphs(self, config: Mapping[str, Any], connector: Mapping) -> FlowEncoderGraphs | None:
        """``cfm_encoder_cuda_graph`` (default off): exact-shape graphs of the flow encoder's continuation chunk."""
        if not config.get("enabled", False):
            return None
        if not self._flow_on_cuda():
            logger.info("Flow encoder CUDA Graph is disabled off CUDA")
            return None
        encoder = self.flow.encoder
        embed = getattr(encoder, "embed", None)
        lookahead = self._pre_lookahead_len()
        if (
            lookahead is None
            or getattr(encoder, "up_layer", None) is None
            or not isinstance(getattr(getattr(embed, "pos_enc", None), "pe", None), torch.Tensor)
            # ``_ensure_relpos_pe`` would grow such a table between calls.
            or callable(getattr(embed, "extend_pe", None))
        ):
            logger.warning("Flow encoder CUDA Graph needs the CosyVoice2 upsample-conformer encoder; staying eager")
            return None
        # Every ``position_encoding`` slice a graph reads lives in these tables.
        tables = [
            module
            for module in (embed, getattr(encoder, "up_embed", None))
            if isinstance(getattr(getattr(module, "pos_enc", None), "pe", None), torch.Tensor)
        ]
        # Default width: the steady duplex chunk, the left context plus one unit of new frames.
        unit = int(connector.get("initial_codec_chunk_frames", 0) or 0) or int(connector.get("codec_chunk_frames", 25))
        graphs = FlowEncoderGraphs(
            # With no ``extend_pe`` (checked above) this runs exactly the eager continuation's ops.
            functools.partial(self._encode_chunk, last_chunk=False),
            rows=config.get("rows") or [1],
            token_widths=config.get("token_widths") or [int(connector.get("codec_left_context_frames", 3)) + unit],
            lookahead=lookahead,
            upsample=self._upsample_stride(),
            held_tensors=lambda: tuple(module.pos_enc.pe for module in tables),
        )
        logger.info(
            "Flow encoder CUDA Graph enabled for continuation chunks (rows %s, tokens %s)",
            list(graphs.rows),
            list(graphs.token_widths),
        )
        return graphs

    def _graph_encode(
        self, tokens: torch.Tensor, states: list[BatchedToken2WavState]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """A continuation chunk's encoder outputs from its graph (shared views, rewritten by the next replay)."""
        if self._encoder_graphs is None:
            return None
        flows = [state.flow_cache for state in states]
        return self._encoder_graphs.run(
            tokens, [flow["conformer_cnn_cache"] for flow in flows], [flow["conformer_att_cache"] for flow in flows]
        )

    @torch.inference_mode()
    def precapture_flow_encoder(self, features: PromptFeatures) -> int:
        """Capture the encoder graphs of every continuation chunk this prompt's streams run; returns the count."""
        graphs = self._encoder_graphs
        if graphs is None or graphs.graphs:
            return 0
        (state,) = self.setup_batch(features, 1)
        cnn, att = state.flow_cache["conformer_cnn_cache"], state.flow_cache["conformer_att_cache"]
        prompt_len = int(features.mels.shape[1])
        keys = graphs.keys_for(start=int(att.shape[3]), prompt_len=prompt_len, suffix=_CACHE_TRIM_SUFFIX)
        torch.accelerator.synchronize(att.device)
        memory_before = _memory_snapshot(att.device)
        with self._autocast(att.device):
            captured = graphs.capture(
                keys,
                cnn_shape=(int(cnn.shape[1]), int(cnn.shape[2])),
                att_layout=(int(att.shape[0]), int(att.shape[2]), int(att.shape[4])),
                hidden_dim=int(self.flow.encoder_proj.out_features),
                dtype=att.dtype,
                device=att.device,
            )
        torch.accelerator.synchronize(att.device)
        logger.info(
            "Captured %d flow encoder CUDA Graphs (cache frames %s): shared buffers %.1f MiB%s",
            captured,
            sorted({key[2] for key in keys}),
            graphs.storage_bytes() / 2**20,
            _format_memory_delta(memory_before, _memory_snapshot(att.device)),
        )
        return captured

    @staticmethod
    def _estimator_buffers(
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = estimator.blocks
        depth = len(blocks)
        batch_size = int(x.shape[0])
        chunk_size = int(x.shape[2])
        old_att_len = int(old_att.shape[3]) if old_att is not None else 0
        block0 = blocks[0]
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        heads = int(block0.attn.num_heads)
        att_width = int(block0.attn.head_dim * 2)
        cnn = x.new_empty((depth, batch_size, cnn_channels, cnn_width))
        att = x.new_empty((depth, batch_size, heads, old_att_len + chunk_size, att_width))
        return cnn, att

    def _time_embedding(self, estimator: nn.Module, time: torch.Tensor) -> torch.Tensor:
        """``estimator.t_embedder(time)`` without a host sync per call (bitwise identical).

        Upstream rebuilds its frequency table on the host and moves it with a
        pageable ``.to(t)`` once per CFM step, blocking the host until the GPU
        drains; the table is built the same way once and kept on the device.
        """
        embedder = estimator.t_embedder
        dim, scale, mlp = (getattr(embedder, name, None) for name in ("frequency_embedding_size", "scale", "mlp"))
        if not isinstance(dim, int) or scale is None or mlp is None:
            return embedder(time)
        # The host table takes the default dtype before ``.to(t)``, as upstream.
        key = (time.device, time.dtype, torch.get_default_dtype(), dim)
        freqs = self._timestep_freqs.get(key)
        if freqs is None:
            freqs = self._timestep_freqs[key] = _timestep_frequencies(dim).to(time)
        args = (time * scale)[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return mlp(embedding)

    def _estimator_step(
        self,
        estimator: nn.Module,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        time: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
        valid_lengths: list[int] | None = None,
        valid_frames: int | None = None,
        time_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._trt_stepper is not None and valid_lengths is None:
            out, new_cnn, new_att = self._trt_stepper.step(
                x=x,
                mu=mu,
                t=time,
                spks=speakers,
                cond=cond,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
            )
            return out.to(mu.dtype), new_cnn, new_att
        time_embedding = self._time_embedding(estimator, time).unsqueeze(1)
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        if valid_frames is not None and valid_frames < width:
            # Every frame gets the speaker vector, so the padded columns would
            # stay non-zero even with x/mu/cond zeroed -- and the estimator
            # would then emit non-zero features there, leaving the
            # chunk-boundary attention/CNN caches padding-derived. Zero them.
            speaker_features = speaker_features.clone()
            _zero_padded_frames(speaker_features, valid_frames)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cnn_out, att_out = self._estimator_buffers(estimator, estimator_input, att_cache)
        old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        if isinstance(old_att, torch.Tensor) and old_att.dtype != estimator_input.dtype:
            old_att = old_att.to(dtype=estimator_input.dtype)
        if self._cfm_graph_wrapper is not None and valid_lengths is None:
            graph_cnn = torch.zeros_like(cnn_out) if cnn_cache is None else cnn_cache
            graph_att = (
                estimator_input.new_zeros(att_out.shape[:3] + (0,) + att_out.shape[4:])
                if att_cache is None
                else old_att
            )
            return self._cfm_graph_wrapper.replay(
                estimator_input,
                time_embedding,
                graph_cnn,
                graph_att,
                cnn_out,
                att_out,
                attn_mask,
            )
        if valid_lengths is not None:
            if not hasattr(estimator, "in_proj"):
                raise RuntimeError('MiniCPMO45Code2WavBatchError {"reason":"ragged_kernel_unavailable"}')
            result = self._ragged_body(
                estimator,
                estimator_input,
                time_embedding,
                attn_mask,
                old_cnn,
                old_att,
                cnn_out,
                att_out,
                valid_lengths,
            )
        else:
            result = estimator.blocks_forward_chunk(
                estimator_input,
                time_embedding,
                attn_mask,
                old_cnn,
                old_att,
                cnn_out,
                att_out,
            )
        return result, cnn_out, att_out

    def _ragged_body(
        self,
        *args: Any,
        modulation: torch.Tensor | None = None,
        slots: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The configured ragged DiT body, looked up per call so a capture records a replaced one.

        ``modulation`` (fused body only) comes from the Whole-Euler arena.
        """
        estimator_input = args[1]
        if slots is not None or (self._ragged_fused_body and estimator_input.dtype == torch.float32):
            # Slot-pool graphs are only built over the fused body (``slot_pool`` above).
            return blocks_forward_chunk_fused(*args, modulation=modulation, slots=slots)
        return self._blocks_forward_chunk_ragged(*args)

    @staticmethod
    def _causal_conv_frames_major(conv: nn.Conv1d, history: torch.Tensor) -> torch.Tensor:
        """``conv`` over ``history`` (b, width + t, c_in) as (b, t, c_out), without a layout change.

        The DiT keeps activations as (b, t, c), which already is NHWC for a
        height-1 image. Upstream transposes them to (b, c, t) for ``Conv1d``,
        and cuDNN converts that to NHWC and back around every convolution. As
        a channels-last conv2d it reads the activations where they are. The
        channels-last weight is converted once and reused. Other devices run
        upstream's ``Conv1d`` layout (Ascend has no channels-last format).
        """
        if not history.is_cuda:
            output = F.conv1d(
                history.transpose(1, 2),
                conv.weight,
                conv.bias,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
            )
            return output.transpose(1, 2)
        weight = conv.weight
        version = (weight.data_ptr(), weight._version, weight.dtype)
        cached = conv.__dict__.get("_channels_last_weight")
        if cached is None or cached[0] != version:
            cached = (version, weight.detach().unsqueeze(2).contiguous(memory_format=torch.channels_last))
            conv.__dict__["_channels_last_weight"] = cached
        output = F.conv2d(
            history.transpose(1, 2).unsqueeze(2),
            cached[1],
            conv.bias,
            (1, conv.stride[0]),
            (0, conv.padding[0]),
            (1, conv.dilation[0]),
            conv.groups,
        )
        return output.squeeze(2).transpose(1, 2)

    @staticmethod
    def _attend_in_place(
        attn: nn.Module,
        x: torch.Tensor,
        att_cache: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        kv: torch.Tensor,
    ) -> torch.Tensor:
        """``attn.forward_chunk`` that keeps the new keys and values in ``kv``.

        ``kv`` (b, heads, >= t + cache frames, 2 * head_dim) is the block's new
        cache, ``[current | cache]`` like upstream's. Upstream concatenates the
        whole cache behind the current keys and values, then both into the new
        cache, which the caller copied into ``kv``: four passes over Stage 2's
        largest tensor. Here only the current chunk is written and attention reads ``kv`` in
        place. A cache that already sits behind the current chunk in ``kv``
        (Whole-Euler) is not moved; any other is copied there once. Other
        devices keep upstream's attention and copy its cache into ``kv``.
        """
        if kv.dtype != x.dtype or not x.is_cuda:
            # A reduced-precision cache attends in the activation dtype.
            x_att, new_att = attn.forward_chunk(x, att_cache, attn_mask)
            kv[:, :, : int(new_att.shape[2])].copy_(new_att)
            return x_att
        batch, frames, _ = x.shape
        q = attn.q_norm(attn.to_heads(attn.to_q(x)))
        k = attn.k_norm(attn.to_heads(attn.to_k(x)))
        v = attn.to_heads(attn.to_v(x))
        cached = 0 if att_cache is None else int(att_cache.shape[2])
        kv = kv[:, :, : frames + cached]
        behind = kv[:, :, frames:]
        if cached and (att_cache.data_ptr() != behind.data_ptr() or att_cache.stride() != behind.stride()):
            behind.copy_(att_cache)
        head_dim = int(k.shape[3])
        kv[..., :frames, :head_dim].copy_(k)
        kv[..., :frames, head_dim:].copy_(v)
        out = F.scaled_dot_product_attention(
            q,
            kv[..., :head_dim],
            kv[..., head_dim:],
            attn_mask=None if attn_mask is None else attn_mask.unsqueeze(1),
        )
        return attn.proj(out.transpose(1, 2).reshape(batch, frames, -1))

    @staticmethod
    def _blocks_forward_chunk_ragged(
        estimator: nn.Module,
        estimator_input: torch.Tensor,
        time_embedding: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cnn_cache: Any,
        att_cache: Any,
        cnn_cache_buffer: torch.Tensor,
        att_cache_buffer: torch.Tensor,
        valid_lengths: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Run one padded DiT batch while capturing exact per-row CNN state.

        ``valid_lengths`` is one length per request, or the ``(2B,)`` CFG
        lengths tensor a CUDA graph captures as a static input. Upstream's
        ``DiTBlock.forward_chunk``, with attention over the output cache in
        place (``_attend_in_place``) and the causal convolutions on the
        frames-major activations (``_causal_conv_frames_major``).
        """
        if isinstance(valid_lengths, torch.Tensor):
            lengths = valid_lengths
        else:
            lengths = torch.tensor(
                (*valid_lengths, *valid_lengths),
                device=estimator_input.device,
                dtype=torch.long,
            )
        # Causal cache gather index per conv width, shared by the blocks.
        cache_index: dict[int, torch.Tensor] = {}
        x = estimator.in_proj(estimator_input.transpose(1, 2))
        for block_index, block in enumerate(estimator.blocks):
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                shift_conv,
                scale_conv,
                gate_conv,
            ) = block.adaLN_modulation(time_embedding).chunk(9, dim=-1)

            normalized = block.norm1(x) * (1 + scale_msa) + shift_msa
            x_att = BatchedToken2Wav._attend_in_place(
                block.attn,
                normalized,
                att_cache[block_index],
                attn_mask,
                att_cache_buffer[block_index],
            )
            x = x + gate_msa * x_att

            # CausalConvBlock: [T, conv1, T, LayerNorm, Mish, T, conv2, T]
            # with the transposes (T) dropped.
            conv = block.conv
            conv_input = block.norm3(x) * (1 + scale_conv) + shift_conv
            old_cnn = cnn_cache[block_index]
            if old_cnn is None:
                old_cnn = conv_input.new_zeros(
                    (
                        int(conv_input.shape[0]),
                        int(conv.in_channels + conv.out_channels),
                        int(conv.block[1].causal_padding[0]),
                    )
                )
            width = int(old_cnn.shape[2])
            if width not in cache_index:
                cache_index[width] = causal_cache_index(lengths, width)
            history = torch.cat((old_cnn[:, : conv.in_channels].transpose(1, 2), conv_input), dim=1)
            new_cnn1 = gather_causal_cache(history, cache_index[width])
            hidden = conv.block[4](conv.block[3](BatchedToken2Wav._causal_conv_frames_major(conv.block[1], history)))
            history = torch.cat((old_cnn[:, conv.in_channels :].transpose(1, 2), hidden), dim=1)
            new_cnn2 = gather_causal_cache(history, cache_index[width])
            x = x + gate_conv * BatchedToken2Wav._causal_conv_frames_major(conv.block[6], history)
            x = x + gate_mlp * block.mlp(block.norm2(x) * (1 + scale_mlp) + shift_mlp)

            cnn_cache_buffer[block_index][:, : conv.in_channels].copy_(new_cnn1)
            cnn_cache_buffer[block_index][:, conv.in_channels :].copy_(new_cnn2)

        return estimator.final_layer(x, time_embedding).transpose(1, 2)

    def _get_timeline(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, list[float]]:
        """The Euler timeline and its step sizes, read to the host once.

        ``_euler_step`` takes ``dt`` as a Python float, so reading it
        from the device timeline on every decode would sync once per step.
        """
        key = (device, dtype)
        cached = self._timeline_cache.get(key)
        if cached is None:
            cached = _euler_timeline(self.n_timesteps, device, dtype)
            self._timeline_cache[key] = cached
        return cached

    @_cfm_matmul_tf32
    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | list[torch.Tensor] | None,
        valid_lengths: list[int] | None = None,
        att_keep: tuple[int, int] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | list[torch.Tensor],
    ]:
        """Run the CFM Euler solve for one chunk.

        ``att_cache`` may be the stacked estimator cache or one tensor per
        request (the Whole-Euler path). A Whole-Euler solve returns the new
        cache in the same form, already trimmed by ``att_keep``; the eager
        path returns it stacked and untrimmed (ragged: per request, untrimmed).
        """
        decoder = self.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        att_rows = att_cache if isinstance(att_cache, list) else None
        if att_rows is not None:
            offset = int(att_rows[0].shape[4])
        else:
            offset = int(att_cache.shape[4]) if att_cache is not None else 0
        # ``row_offset_merge``: ragged rows whose caches differ in length each
        # start from the noise at their own offset and attend their own cache.
        row_offsets = None
        if self._row_offset_merge and att_rows is not None and valid_lengths is not None:
            row_offsets = [int(row.shape[4]) for row in att_rows]
            if len(set(row_offsets)) == 1:
                row_offsets = None
        mel_frames = int(mu.shape[2])
        # Padding only pays off while replay is active: Whole-Euler, the CFM wrapper (still enabled,
        # no TRT stepper), or on NPU the platform runner keyed by `_cfm_graph_enabled`.
        graphs_active = valid_lengths is None and (
            self._whole_euler_active()
            or (
                self._cfm_graph_wrapper is not None
                and getattr(self._cfm_graph_wrapper, "enabled", True)
                and self._trt_stepper is None
            )
            or (self._cfm_graph_wrapper is None and self._cfm_graph_enabled)
        )
        pad_frames = _cfm_pad_frames(
            mel_frames=mel_frames,
            offset=offset,
            noise_capacity=int(decoder.rand_noise.shape[2]),
            bucket_frames=self._cfm_graph_bucket_frames,
            disabled=(valid_lengths is not None or not graphs_active),
        )
        if pad_frames:
            # Replicate rather than zero: a repeated last frame is a closer
            # continuation of the chunk than silence, so the padded positions
            # carry less of a step change into the attention and CNN caches.
            mu = torch.nn.functional.pad(mu, (0, pad_frames), mode="replicate")
            cond = torch.nn.functional.pad(cond, (0, pad_frames), mode="replicate")
        end = (offset if row_offsets is None else max(row_offsets)) + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"noise_capacity","required":{end},'
                f'"available":{int(decoder.rand_noise.shape[2])}}}'
            )
        if row_offsets is None:
            x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        else:
            width = int(mu.shape[2])
            x = torch.cat([decoder.rand_noise[:, :, start : start + width] for start in row_offsets], dim=0)
        if pad_frames:
            # The padded columns would otherwise carry real noise values. They
            # are excluded from this chunk's attention by ``attn_mask`` below
            # and held at zero here, so the padded region stays inert. Masking
            # is what removes them; zeroing alone would not, since a zero row
            # still occupies part of the softmax denominator.
            x[:, :, mel_frames:] = 0.0
        timeline, dt_steps = self._get_timeline(mu.device, mu.dtype)
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        attn_mask = None
        if valid_lengths is not None:
            if len(valid_lengths) != batch_size:
                raise ValueError(f"valid length count {len(valid_lengths)} != batch {batch_size}")
            cfg_lengths = index_to_device((*valid_lengths, *valid_lengths), mu.device)
            positions = torch.arange(int(mu.shape[2]), device=mu.device)
            valid_queries = positions.unsqueeze(0) < cfg_lengths.unsqueeze(1)
            current_keys = valid_queries.unsqueeze(1).expand(-1, int(mu.shape[2]), -1)
            if row_offsets is None:
                old_keys = torch.ones(
                    (2 * batch_size, int(mu.shape[2]), offset),
                    dtype=torch.bool,
                    device=mu.device,
                )
            else:
                # Cache columns span the longest cache; a shorter one's tail stays masked.
                cfg_offsets = index_to_device((*row_offsets, *row_offsets), mu.device)
                cached = torch.arange(max(row_offsets), device=mu.device).unsqueeze(0) < cfg_offsets.unsqueeze(1)
                old_keys = cached.unsqueeze(1).expand(-1, int(mu.shape[2]), -1)
            attn_mask = valid_queries.unsqueeze(2) & torch.cat((current_keys, old_keys), dim=2)
        elif pad_frames:
            # Mask the padded keys instead of only zeroing their content: a
            # zero-valued key/value pair still takes probability mass out of the
            # softmax denominator, so the real frames keep attending to the
            # padding unless it is explicitly excluded.
            kv_len = int(mu.shape[2]) + offset
            attn_mask = torch.ones(
                2 * batch_size,
                int(mu.shape[2]),
                kv_len,
                dtype=torch.bool,
                device=mu.device,
            )
            attn_mask[:, :, mel_frames : mel_frames + pad_frames] = False

        if self._whole_euler_active() and (valid_lengths is None or self._whole_euler_ragged_active()):
            whole_euler_result = self._whole_euler_graph_wrapper.replay(
                x=x,
                mu_cfg=mu_cfg,
                speakers_cfg=speakers_cfg,
                cond_cfg=cond_cfg,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
                attn_mask=attn_mask,
                mel_frames=mel_frames,
                pad_frames=pad_frames,
                valid_lengths=valid_lengths,
                att_keep=att_keep,
            )
            if whole_euler_result is not None:
                return whole_euler_result
        if row_offsets is not None:
            # No single solve took them (graphs off, or the slot pool refused).
            return self._decode_cfm_per_offset(
                mu,
                speakers,
                cond,
                cnn_cache=cnn_cache,
                att_rows=att_rows,
                valid_lengths=valid_lengths,
                att_keep=att_keep,
            )
        if att_rows is not None:
            att_cache = self._stack_estimator_att(att_rows)

        next_cnn: list[torch.Tensor] = []
        next_att_cache: torch.Tensor | None = None
        ragged_att_cache: list[torch.Tensor] | None = None
        with _token2wav_sdpa_context(mu.device):
            for step in range(self.n_timesteps):
                old_cnn = cnn_cache[step] if cnn_cache is not None else None
                old_att = att_cache[step] if att_cache is not None else None
                step_time = timeline[step].expand(2 * batch_size)
                estimate, step_cnn, step_att = self._estimator_step(
                    estimator,
                    x=torch.cat((x, x), dim=0),
                    mu=mu_cfg,
                    time=step_time,
                    speakers=speakers_cfg,
                    cond=cond_cfg,
                    cnn_cache=old_cnn,
                    att_cache=old_att,
                    attn_mask=attn_mask,
                    valid_lengths=valid_lengths,
                    valid_frames=mel_frames if pad_frames else None,
                )
                if pad_frames:
                    _zero_padded_cnn_cache(step_cnn, estimator, pad_frames)
                x = _euler_step(x, estimate, dt_steps[step], decoder.inference_cfg_rate, batch_size)
                if pad_frames:
                    _zero_padded_frames(x, mel_frames)
                next_cnn.append(step_cnn)
                if valid_lengths is not None:
                    if ragged_att_cache is None:
                        ragged_att_cache = [
                            torch.empty(
                                (
                                    self.n_timesteps,
                                    int(step_att.shape[0]),
                                    2,
                                    int(step_att.shape[2]),
                                    valid_length + offset,
                                    int(step_att.shape[4]),
                                ),
                                device=step_att.device,
                                dtype=self._estimator_att_cache_dtype,
                            )
                            for valid_length in valid_lengths
                        ]
                    current_width = int(mu.shape[2])
                    for row, (valid_length, row_cache) in enumerate(zip(valid_lengths, ragged_att_cache, strict=True)):
                        for cfg_row, source_row in enumerate((row, batch_size + row)):
                            row_cache[step, :, cfg_row, :, :valid_length].copy_(
                                step_att[:, source_row, :, :valid_length]
                            )
                            if offset:
                                row_cache[step, :, cfg_row, :, valid_length:].copy_(
                                    step_att[:, source_row, :, current_width:]
                                )
                    del step_att
                    continue
                if next_att_cache is None:
                    next_att_cache = torch.empty(
                        (self.n_timesteps, *step_att.shape),
                        device=step_att.device,
                        dtype=self._estimator_att_cache_dtype,
                    )
                next_att_cache[step].copy_(step_att)
                if pad_frames:
                    # The padded steps ran but hold no valid content, and each
                    # cache entry becomes the next chunk's keys. Clear the
                    # padded columns so the boundary the next chunk reads is
                    # the valid one, in step with the mask built above.
                    next_att_cache[step][..., mel_frames : mel_frames + pad_frames, :] = 0.0
                del step_att
        if pad_frames:
            x = x[:, :, :mel_frames]
        if ragged_att_cache is not None:
            return x, torch.stack(next_cnn), ragged_att_cache
        assert next_att_cache is not None
        return x, torch.stack(next_cnn), next_att_cache

    def _decode_cfm_per_offset(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_rows: list,
        valid_lengths: list[int],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, list]:
        """``_decode_cfm`` once per cache length, in row order: a row-offset merge no single solve took."""
        batch_size = len(att_rows)
        groups: dict[int, list[int]] = {}
        for row, cache in enumerate(att_rows):
            groups.setdefault(int(cache.shape[4]), []).append(row)
        x: torch.Tensor | None = None
        cnn: torch.Tensor | None = None
        att: list = [None] * batch_size
        for rows in groups.values():
            index = index_to_device(rows, mu.device)
            # CFG rows are ``[cond x B | uncond x B]``.
            cfg_index = index_to_device([*rows, *(batch_size + row for row in rows)], mu.device)
            group_x, group_cnn, group_att = self._decode_cfm(
                mu.index_select(0, index),
                speakers.index_select(0, index),
                cond.index_select(0, index),
                cnn_cache=None if cnn_cache is None else cnn_cache.index_select(2, cfg_index),
                att_cache=[att_rows[row] for row in rows],
                valid_lengths=[valid_lengths[row] for row in rows],
                **kwargs,
            )
            if x is None or cnn is None:
                x = group_x.new_empty((batch_size, *group_x.shape[1:]))
                cnn = group_cnn.new_empty((*group_cnn.shape[:2], 2 * batch_size, *group_cnn.shape[3:]))
            x.index_copy_(0, index, group_x)
            cnn.index_copy_(2, cfg_index, group_cnn)
            for row, cache in zip(rows, group_att, strict=True):
                att[row] = cache
        assert x is not None and cnn is not None
        return x, cnn, att

    def _split_flow_cache(
        self,
        cache: dict[str, Any],
        batch_size: int,
        *,
        estimator_att_keep: tuple[int, int] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Per-request flow caches from a batch's stacked caches.

        ``estimator_att_cache`` is stacked ``(n_t, depth, 2B, heads, L, d)``
        with CFG rows ``[cond x B | uncond x B]``, or already one tensor per
        request (Whole-Euler). The streaming trim ``estimator_att_keep`` is
        applied while each request is copied out, so no trimmed copy of the
        whole batch is built first: at 16 requests that copy alone was ~7 GiB.
        """
        estimator_att = cache["estimator_att_cache"]
        stacked_att = isinstance(estimator_att, torch.Tensor)
        if stacked_att:
            keep_ranges = _att_keep_ranges(int(estimator_att.shape[4]), estimator_att_keep)
            request_shape = list(estimator_att.shape)
            request_shape[2] = 2
            request_shape[4] = sum(length for _, length in keep_ranges)
            att_rows = estimator_att.unflatten(2, (2, batch_size))
        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            if stacked_att:
                request_att = torch.empty(
                    request_shape,
                    device=estimator_att.device,
                    dtype=self._estimator_att_cache_dtype,
                )
                _copy_frame_segments(request_att, att_rows[:, :, :, row], keep_ranges)
            else:
                request_att = estimator_att[row]
            result.append(
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"][row : row + 1].detach().clone(),
                    "conformer_att_cache": cache["conformer_att_cache"][:, row : row + 1].detach().clone(),
                    "estimator_cnn_cache": torch.cat(
                        (
                            cache["estimator_cnn_cache"][:, :, row : row + 1],
                            cache["estimator_cnn_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                    "estimator_att_cache": request_att,
                }
            )
        return result

    @staticmethod
    def _stack_estimator_att(rows: list[torch.Tensor]) -> torch.Tensor:
        """Stack per-request ``(n_t, depth, 2, ...)`` caches into CFG order ``[cond x B | uncond x B]``."""
        rows = _materialize_att_rows(rows)
        return torch.cat((*[row[:, :, 0:1] for row in rows], *[row[:, :, 1:2] for row in rows]), dim=2)

    def _stack_flow_cache(
        self,
        states: list[BatchedToken2WavState],
        *,
        include_estimator_att: bool = True,
        include_conformer: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        """Stack per-request flow caches for one batched call.

        The estimator attention cache is by far the largest (~0.5 GiB per
        request in fp32). The Whole-Euler path reads it per request, so it
        skips stacking it (``include_estimator_att=False``). A caller that
        already ran the encoder skips the conformer caches
        (``include_conformer=False``).
        """
        flows = [state.flow_cache for state in states]
        conditional_cnn = [flow["estimator_cnn_cache"][:, :, 0:1] for flow in flows]
        unconditional_cnn = [flow["estimator_cnn_cache"][:, :, 1:2] for flow in flows]
        return {
            "conformer_cnn_cache": (
                torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0) if include_conformer else None
            ),
            "conformer_att_cache": (
                torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1) if include_conformer else None
            ),
            "estimator_cnn_cache": torch.cat((*conditional_cnn, *unconditional_cnn), dim=2),
            "estimator_att_cache": (
                self._stack_estimator_att([flow["estimator_att_cache"] for flow in flows])
                if include_estimator_att
                else None
            ),
        }

    def _whole_euler_active(self) -> bool:
        wrapper = self._whole_euler_graph_wrapper
        return wrapper is not None and wrapper.enabled and self._trt_stepper is None

    def _whole_euler_ragged_active(self) -> bool:
        """Whether Whole-Euler graphs also solve ragged (per-row length) batches."""
        return self._whole_euler_active() and self._whole_euler_graph_wrapper.ragged_body is not None

    def _ragged_merge_fits(self, states: list[BatchedToken2WavState], token_counts: list[int]) -> bool:
        """Ragged Whole-Euler is on, the rows share a CNN cache shape, and each row's tokens encode at once."""
        return (
            2 <= len(states) == len(token_counts)
            and self._whole_euler_ragged_active()
            and len({tuple(state.flow_cache["estimator_cnn_cache"].shape) for state in states}) == 1
            and all(count <= self._max_encode_token_frames([s]) for count, s in zip(token_counts, states, strict=True))
        )

    def can_merge_state_shapes(self, states: list[BatchedToken2WavState], token_counts: list[int]) -> bool:
        """Whether mixed onset/continuation rows fit ragged Whole-Euler."""
        caches = [state.flow_cache.get("estimator_att_cache") for state in states]
        return (
            all(isinstance(cache, torch.Tensor) for cache in caches)
            and len({(tuple(cache.shape[:4]), cache.dtype, cache.device) for cache in caches}) == 1
            and self._ragged_merge_fits(states, token_counts)
        )

    def can_merge_row_offsets(self, states: list[BatchedToken2WavState], token_counts: list[int]) -> bool:
        """Whether continuation rows whose caches differ only in length fit one slot-pool ragged solve.

        ``row_offset_merge``: every estimator cache must be resident in the
        Whole-Euler slot pool, where each row attends its own frames;
        ``decode_ragged_batch`` runs the encoder once per conformer cache length.
        """
        if not (self._row_offset_merge and self._ragged_merge_fits(states, token_counts)):
            return False
        pool = self._whole_euler_graph_wrapper.slot_pool
        caches = [state.flow_cache.get("estimator_att_cache") for state in states]
        return pool is not None and all(isinstance(cache, ResidentAttCache) and cache.pool is pool for cache in caches)

    @torch.inference_mode()
    def precapture_hift(self) -> int:
        """Capture every configured HiFT graph (prompt-independent) once, before serving; returns the count."""
        wrapper = self.hift_graph_wrapper
        if wrapper is None or wrapper.graph:
            return 0
        with _autocast_disabled(wrapper.device):
            wrapper.capture()
        return len(wrapper.graph)

    @torch.inference_mode()
    @_cfm_matmul_tf32
    def precapture_whole_euler(self, features: PromptFeatures) -> int:
        """Capture the Whole-Euler graphs of every cache length this prompt's streams pass through.

        Runs in ``_decode_cfm``'s precision (``_cfm_matmul_tf32``, ``_autocast``):
        a graph replays the kernels it was captured with. Returns the count.
        """
        if not self._whole_euler_active():
            return 0
        wrapper = self._whole_euler_graph_wrapper
        prompt_len = int(features.mels.shape[1])
        steady = prompt_len + _CACHE_TRIM_SUFFIX
        with self._autocast(wrapper.device):
            return wrapper.precapture(
                offsets=range(prompt_len, steady + 1),
                steady=steady,
                channels=int(features.mels.shape[2]),
                spk_dim=int(self.flow.spk_embed_affine_layer.out_features),
                keep=(prompt_len, _CACHE_TRIM_SUFFIX),
            )

    def _create_initial_states(
        self,
        features: PromptFeatures,
        batch_size: int,
    ) -> list[BatchedToken2WavState]:
        prompt_tokens, speakers, prompt_mels = self._repeat_prompt(features, batch_size)
        lookahead_width = self._pre_lookahead_len()
        lookahead = prompt_tokens.new_full(
            (batch_size, 3 if lookahead_width is None else lookahead_width),
            _SILENCE_TOKEN,
        )
        with self._autocast(prompt_tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                torch.cat((prompt_tokens, lookahead), dim=1),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
            )
            projected_speakers = self._project_speakers(speakers)
            # The prompt cache never reaches the trim length; ``att_keep`` only
            # tells the Whole-Euler arena the steady cache length to reserve.
            _, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
                att_keep=(int(prompt_mels.shape[1]), _CACHE_TRIM_SUFFIX),
            )
        flow_cache = {
            "conformer_cnn_cache": conformer_cnn,
            "conformer_att_cache": conformer_att,
            "estimator_cnn_cache": estimator_cnn,
            "estimator_att_cache": estimator_att,
        }
        split = self._split_flow_cache(flow_cache, batch_size)
        mel_channels = int(prompt_mels.shape[2])
        return [
            BatchedToken2WavState(
                flow_cache=row,
                hift_cache={
                    "mel": prompt_mels.new_zeros((1, mel_channels, 0)),
                    "source": prompt_mels.new_zeros((1, 1, 0)),
                    "speech": prompt_mels.new_zeros((1, 0)),
                },
            )
            for row in split
        ]

    def setup_batch(self, features: PromptFeatures, batch_size: int) -> list[BatchedToken2WavState]:
        """Initial states for ``batch_size`` requests sharing one prompt.

        Every request starts from the same prompt tokens, speaker, mels and
        noise, so the prompt is solved once and all requests share that one
        read-only state. Decoding writes a state only where Whole-Euler updates
        an estimator cache in place, which needs a cache held by one request
        and allocated with room for its new length; the prompt cache is shared
        and allocated exactly, and the first chunk always grows it. Solving it
        per batch size recomputed identical rows and cached one ~0.3 GiB
        estimator cache copy per request.
        """
        bucket_frames = (
            self._cfm_graph_bucket_frames
            if self._cfm_graph_wrapper is not None and self._cfm_graph_wrapper.enabled
            else 0
        )
        # Capture the policy before setup: graph capture may disable the wrapper
        # after padding has already been chosen for these initial states.
        cache_key = (features.cache_key, bucket_frames)
        state = self._setup_cache.get(cache_key)
        if state is not None:
            self._setup_cache.move_to_end(cache_key)
        else:
            (state,) = self._create_initial_states(features, 1)
            if self._setup_cache_size > 0:
                self._setup_cache[cache_key] = state
                while len(self._setup_cache) > self._setup_cache_size:
                    self._setup_cache.popitem(last=False)
        return [state] * batch_size

    @staticmethod
    def _fade_in_out(
        speech: torch.Tensor,
        previous: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        overlap = min(
            int(window.shape[0] // 2),
            int(speech.shape[-1]),
            int(previous.shape[-1]),
        )
        result = speech.clone()
        if overlap > 0:
            result[..., :overlap] = (
                result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
            )
        return result

    def decode_batch(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
        if batch_size != len(states):
            raise ValueError(f"tokens batch {batch_size} != state batch {len(states)}")
        # The encoder's pre-lookahead convolution consumes ``pre_lookahead_len``
        # frames of right context and keeps no left cache, so a non-final chunk
        # must carry at least one full kernel. Only the final chunk is allowed
        # to be shorter: ``forward_chunk`` zero-pads it by the lookahead width.
        lookahead = self._pre_lookahead_len()
        num_frames = int(tokens.shape[1])
        if lookahead is not None and not last_chunk:
            if num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","frames":{num_frames},'
                    f'"minimum":{lookahead + 1}}}'
                )
        # A non-async Talker dump can land thousands of codec tokens in one
        # Code2Wav prefill. CosyVoice RelPos PE (max_len=5000) plus 2x upsample
        # cannot score that in one ``forward_chunk`` (6968 vs 985 on NPU).
        max_frames = self._max_encode_token_frames(states)
        slices = plan_token2wav_encode_slices(
            num_frames,
            max_frames=max_frames,
            min_nonfinal=(lookahead + 1) if lookahead is not None else 1,
            last_chunk=last_chunk,
        )
        if len(slices) > 1:
            logger.info(
                "MiniCPM-o Code2Wav splitting %d codec tokens into %d encoder windows "
                "(max_frames=%d) to stay inside RelPos PE.",
                num_frames,
                len(slices),
                max_frames,
            )
            parts: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
            current = states
            for index, (start, end) in enumerate(slices):
                is_last_piece = index == len(slices) - 1
                audios, current = self._decode_batch_once(
                    tokens[:, start:end],
                    features,
                    current,
                    last_chunk=last_chunk and is_last_piece,
                    flush_encoder=flush_encoder and is_last_piece,
                )
                for row, audio in enumerate(audios):
                    parts[row].append(audio)
            merged = [
                torch.cat(row_parts) if row_parts else tokens.new_zeros((0,), dtype=torch.float32)
                for row_parts in parts
            ]
            return merged, current
        return self._decode_batch_once(
            tokens,
            features,
            states,
            last_chunk=last_chunk,
            flush_encoder=flush_encoder,
        )

    def _decode_batch_once(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
        # Whole-Euler reads and writes each request's estimator cache directly,
        # so the batch never holds a stacked copy of it.
        per_request_att = self._whole_euler_active()
        # A captured encoder graph stacks the conformer caches into its own inputs.
        graphed = None if last_chunk or flush_encoder else self._graph_encode(tokens, states)
        flow_cache = self._stack_flow_cache(
            states, include_estimator_att=not per_request_att, include_conformer=graphed is None
        )
        prompt_len = int(features.mels.shape[1])
        att_keep = (prompt_len, _CACHE_TRIM_SUFFIX)
        speakers = features.speaker_embedding.expand(batch_size, -1)
        with self._autocast(tokens.device):
            # A graph's shared results are read below and copied per row by _split_flow_cache.
            hidden, conformer_cnn, conformer_att = (
                graphed
                if graphed is not None
                else self._encode_chunk(
                    tokens,
                    last_chunk=last_chunk or flush_encoder,
                    cnn_cache=flow_cache["conformer_cnn_cache"],
                    att_cache=flow_cache["conformer_att_cache"],
                )
            )
            projected_speakers = self._project_speakers(speakers)
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=(
                    [state.flow_cache["estimator_att_cache"] for state in states]
                    if per_request_att
                    else flow_cache["estimator_att_cache"]
                ),
                att_keep=att_keep,
            )

        conformer_att = _trim_streaming_cache(conformer_att, prompt_len)
        new_flow = self._split_flow_cache(
            {
                "conformer_cnn_cache": conformer_cnn,
                "conformer_att_cache": conformer_att,
                "estimator_cnn_cache": estimator_cnn,
                "estimator_att_cache": estimator_att,
            },
            batch_size,
            estimator_att_keep=att_keep,
        )
        old_mel = torch.cat([state.hift_cache["mel"] for state in states], dim=0)
        old_source = torch.cat([state.hift_cache["source"] for state in states], dim=0)
        old_speech = torch.cat([state.hift_cache["speech"] for state in states], dim=0)
        mel = torch.cat((old_mel, chunk_mel), dim=2)
        speech, source = self._hift_inference(mel, old_source)
        if old_speech.shape[-1] > 0:
            window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
            speech = self._fade_in_out(speech, old_speech, window)
        next_hift = {
            "mel": mel[..., -self.mel_cache_len :].detach(),
            "source": source[..., -self.source_cache_len :].detach(),
            "speech": speech[..., -self.source_cache_len :].detach(),
        }
        emitted = speech if last_chunk else speech[..., : -self.source_cache_len]
        next_states = [
            BatchedToken2WavState(
                flow_cache=new_flow[row],
                hift_cache={name: value[row : row + 1].detach().clone() for name, value in next_hift.items()},
            )
            for row in range(batch_size)
        ]
        audios = [emitted[row].reshape(-1).to(dtype=torch.float32) for row in range(batch_size)]
        return audios, next_states

    @staticmethod
    def _require_complete_ragged_outputs(
        audios: list[torch.Tensor | None],
        next_states: list[BatchedToken2WavState | None],
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        missing_audio_rows = [row for row, audio in enumerate(audios) if audio is None]
        missing_state_rows = [row for row, state in enumerate(next_states) if state is None]
        if missing_audio_rows or missing_state_rows:
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"incomplete_ragged_output","audio_rows":{missing_audio_rows},'
                f'"state_rows":{missing_state_rows}}}'
            )
        return (
            cast(list[torch.Tensor], audios),
            cast(list[BatchedToken2WavState], next_states),
        )

    def decode_ragged_batch(
        self,
        tokens: list[torch.Tensor],
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunks: list[bool],
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = len(tokens)
        if batch_size != len(states) or batch_size != len(last_chunks):
            raise ValueError("ragged token, state, and final-flag batches must have the same size")
        if batch_size == 0:
            return [], []

        lookahead = self._pre_lookahead_len()
        for row, (row_tokens, last_chunk) in enumerate(zip(tokens, last_chunks, strict=True)):
            num_frames = int(row_tokens.numel())
            if lookahead is not None and not last_chunk and num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","row":{row},'
                    f'"frames":{num_frames},"minimum":{lookahead + 1}}}'
                )

        max_frames = self._max_encode_token_frames(states)
        per_request_att = self._whole_euler_ragged_active()
        if (self._whole_euler_active() and not per_request_att) or any(
            int(row_tokens.numel()) > max_frames for row_tokens in tokens
        ):
            # decode_batch owns the RelPos-safe encoder slicing used by
            # non-async Talker dumps. Preserve that path for overlong rows.
            # Whole-Euler graphs that cannot run the ragged kernel also take
            # this path: under concurrency most batches hold some request's
            # short final chunk, and one padded eager DiT would run the whole
            # batch ungraphed on a stacked copy of every cache. Otherwise short
            # duplex chunks share one padded DiT solve, graphed when possible.
            exact_groups: dict[tuple[int, bool], list[int]] = {}
            for row, (row_tokens, last_chunk) in enumerate(zip(tokens, last_chunks, strict=True)):
                exact_groups.setdefault((int(row_tokens.numel()), last_chunk), []).append(row)
            audios: list[torch.Tensor | None] = [None] * batch_size
            next_states: list[BatchedToken2WavState | None] = [None] * batch_size
            for (_, last_chunk), rows in exact_groups.items():
                group_audio, group_states = self.decode_batch(
                    torch.stack([tokens[row] for row in rows], dim=0),
                    features,
                    [states[row] for row in rows],
                    last_chunk=last_chunk,
                )
                for group_row, row in enumerate(rows):
                    audios[row] = group_audio[group_row]
                    next_states[row] = group_states[group_row]
            return self._require_complete_ragged_outputs(audios, next_states)
        encoder_groups: dict[tuple[Any, ...], list[int]] = {}
        for row, (row_tokens, state, last_chunk) in enumerate(zip(tokens, states, last_chunks, strict=True)):
            # Onset and continuation rows may differ in conformer cache length: encode them apart;
            # the estimator caches stay per-request rows in the ragged Whole-Euler solve.
            key = (
                tuple(state.flow_cache["conformer_cnn_cache"].shape),
                tuple(state.flow_cache["conformer_att_cache"].shape),
                int(row_tokens.numel()),
                last_chunk,
            )
            encoder_groups.setdefault(key, []).append(row)

        hidden_rows: list[torch.Tensor | None] = [None] * batch_size
        conformer_cnn_rows: list[torch.Tensor | None] = [None] * batch_size
        conformer_att_rows: list[torch.Tensor | None] = [None] * batch_size
        for (*_, last_chunk), rows in encoder_groups.items():
            group_states = [states[row] for row in rows]
            group_tokens = torch.stack([tokens[row] for row in rows], dim=0)
            graphed = None if last_chunk else self._graph_encode(group_tokens, group_states)
            if graphed is not None and len(encoder_groups) > 1:
                # The next group's replay rewrites the shared graph results.
                graphed = tuple(value.clone() for value in graphed)
            if graphed is None:
                # The encoder reads only the conformer caches.
                group_cache = self._stack_flow_cache(group_states, include_estimator_att=False)
                with self._autocast(group_tokens.device):
                    graphed = self._encode_chunk(
                        group_tokens,
                        last_chunk=last_chunk,
                        cnn_cache=group_cache["conformer_cnn_cache"],
                        att_cache=group_cache["conformer_att_cache"],
                    )
            hidden, conformer_cnn, conformer_att = graphed
            for group_row, row in enumerate(rows):
                hidden_rows[row] = hidden[group_row : group_row + 1]
                conformer_cnn_rows[row] = conformer_cnn[group_row : group_row + 1]
                conformer_att_rows[row] = conformer_att[:, group_row : group_row + 1]

        missing_hidden_rows = [row for row, value in enumerate(hidden_rows) if value is None]
        if missing_hidden_rows:
            raise RuntimeError(
                f'MiniCPMO45Code2WavBatchError {{"reason":"incomplete_encoder_output","rows":{missing_hidden_rows}}}'
            )
        resolved_hidden = cast(list[torch.Tensor], hidden_rows)
        hidden_lengths = [int(value.shape[1]) for value in resolved_hidden]
        max_hidden_length = max(hidden_lengths)
        padded_hidden = resolved_hidden[0].new_zeros((batch_size, max_hidden_length, int(resolved_hidden[0].shape[2])))
        for row, hidden in enumerate(resolved_hidden):
            padded_hidden[row, : int(hidden.shape[1])].copy_(hidden[0])

        # A graphed ragged solve reads and writes each request's estimator
        # cache directly (trimmed on the way out), like _decode_batch_once.
        # Row-offset merges hold caches of different lengths, which never stack: the solve takes them
        # per row, and the encoder groups above already read the conformer caches.
        per_row_att = per_request_att or (
            self._row_offset_merge
            and len({int(state.flow_cache["estimator_att_cache"].shape[4]) for state in states}) > 1
        )
        flow_cache = self._stack_flow_cache(
            states, include_estimator_att=not per_row_att, include_conformer=not self._row_offset_merge
        )
        prompt_len = int(features.mels.shape[1])
        speakers = features.speaker_embedding.expand(batch_size, -1)
        with self._autocast(padded_hidden.device):
            projected_speakers = self._project_speakers(speakers)
            cond = torch.zeros_like(padded_hidden).transpose(1, 2).contiguous()
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                padded_hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=(
                    [state.flow_cache["estimator_att_cache"] for state in states]
                    if per_row_att
                    else flow_cache["estimator_att_cache"]
                ),
                valid_lengths=hidden_lengths,
                att_keep=(prompt_len, _CACHE_TRIM_SUFFIX),
            )

        assert isinstance(estimator_att, list)
        new_flow: list[dict[str, torch.Tensor]] = []
        for row, row_estimator_att in enumerate(estimator_att):
            conformer_cnn = conformer_cnn_rows[row]
            conformer_att = conformer_att_rows[row]
            assert conformer_cnn is not None and conformer_att is not None
            # A graphed solve already trimmed the estimator cache; trimming is idempotent.
            row_estimator_att = _trim_streaming_cache(row_estimator_att, prompt_len)
            conformer_att = _trim_streaming_cache(conformer_att, prompt_len)
            new_flow.append(
                {
                    "conformer_cnn_cache": conformer_cnn.detach().clone(),
                    "conformer_att_cache": conformer_att.detach().clone(),
                    "estimator_cnn_cache": torch.cat(
                        (
                            estimator_cnn[:, :, row : row + 1],
                            estimator_cnn[:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                    "estimator_att_cache": row_estimator_att,
                }
            )

        audios: list[torch.Tensor | None] = [None] * batch_size
        next_states: list[BatchedToken2WavState | None] = [None] * batch_size
        vocoder_groups: dict[tuple[int, bool], list[int]] = {}
        for row, last_chunk in enumerate(last_chunks):
            vocoder_groups.setdefault((hidden_lengths[row], last_chunk), []).append(row)
        for (hidden_length, last_chunk), rows in vocoder_groups.items():
            old_mel = torch.cat([states[row].hift_cache["mel"] for row in rows], dim=0)
            old_source = torch.cat([states[row].hift_cache["source"] for row in rows], dim=0)
            old_speech = torch.cat([states[row].hift_cache["speech"] for row in rows], dim=0)
            group_mel = torch.cat(
                [chunk_mel[row : row + 1, :, :hidden_length] for row in rows],
                dim=0,
            )
            mel = torch.cat((old_mel, group_mel), dim=2)
            speech, source = self._hift_inference(mel, old_source)
            if old_speech.shape[-1] > 0:
                window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
                speech = self._fade_in_out(speech, old_speech, window)
            next_hift = {
                "mel": mel[..., -self.mel_cache_len :].detach(),
                "source": source[..., -self.source_cache_len :].detach(),
                "speech": speech[..., -self.source_cache_len :].detach(),
            }
            emitted = speech if last_chunk else speech[..., : -self.source_cache_len]
            for group_row, row in enumerate(rows):
                audios[row] = emitted[group_row].reshape(-1).to(dtype=torch.float32)
                next_states[row] = BatchedToken2WavState(
                    flow_cache=new_flow[row],
                    hift_cache={
                        name: value[group_row : group_row + 1].detach().clone() for name, value in next_hift.items()
                    },
                )

        return self._require_complete_ragged_outputs(audios, next_states)
