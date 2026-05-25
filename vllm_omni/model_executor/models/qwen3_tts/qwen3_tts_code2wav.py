from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

logger = init_logger(__name__)


class Qwen3TTSCode2Wav(nn.Module):
    """Stage-1 code2wav model for Qwen3-TTS (GenerationModelRunner).
    Consumes frame-aligned codec tokens from input_ids and decodes waveform
    via the SpeechTokenizer decoder directly (bypassing HF wrapper overhead)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._decode_chunk_frames = 300
        self._decode_left_context_frames = 25
        self._decode_batch_bucket_frames: list[int] = []
        self._decode_batch_max_size = 0
        self._decode_variable_chunk_batch_min_frames = self._decode_chunk_frames + self._decode_left_context_frames + 1
        self._logged_codec_stats = False
        self._logged_malformed_codec_lengths: set[tuple[int, int]] = set()
        self._batch_stats_enabled = os.environ.get("VLLM_OMNI_QWEN3_CODE2WAV_BATCH_STATS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._batch_stats_log_every = int(os.environ.get("VLLM_OMNI_QWEN3_CODE2WAV_BATCH_STATS_LOG_EVERY", "0") or 0)
        self._batch_stats_forwards = 0
        self._batch_stats_groups = 0
        self._batch_stats_requests = 0
        self._batch_stats_padded_frames = 0
        self._batch_stats_decoded_frames = 0
        self._batch_stats_actual_frames: Counter[int] = Counter()
        self._batch_stats_bucket_groups: Counter[tuple[int, int]] = Counter()

        # Construct decoder from config so it is visible to vLLM's
        # memory profiler at startup.  Weights are loaded later in
        # load_weights().
        tok_config = Qwen3TTSTokenizerV2Config.from_pretrained(
            self.model_path,
            subfolder="speech_tokenizer",
        )
        dec_config = tok_config.decoder_config
        self.decoder = Qwen3TTSTokenizerV2Decoder._from_config(dec_config)
        self.decoder.eval()
        self._num_quantizers = int(dec_config.num_quantizers)
        self._output_sample_rate = int(tok_config.output_sample_rate)
        self._total_upsample = int(self.decoder.total_upsample)
        self._decoder_sliding_window = int(getattr(dec_config, "sliding_window", 0) or 0)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings. Keep a stable dummy embedding for vLLM runner.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments.

        Uses seq_token_counts (injected by the runner via model_kwargs) when
        available, falling back to forward-context ubatch_slices when
        micro-batching is active. Returns [ids] for single-request batches.
        """
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    def _maybe_enable_decoder_cudagraph(
        self,
        *,
        device: torch.device,
        codec_chunk_frames: int,
        codec_left_context_frames: int,
        decode_cudagraph_capture_sizes: list[int] | None,
        decode_cudagraph_batch_sizes: list[int] | None,
        decode_cudagraph_extra_capture_shapes: list[tuple[int, int]] | None,
        decode_compile_shapes: list[tuple[int, int]] | None,
    ) -> None:
        """Enable inner Code2Wav CUDA graph unless stage is enforce_eager."""
        if not hasattr(self.decoder, "enable_cudagraph") or device.type != "cuda":
            return

        model_cfg = getattr(self.vllm_config, "model_config", None)
        if getattr(model_cfg, "enforce_eager", False):
            logger.info("Qwen3-TTS Code2Wav CUDA Graph disabled because enforce_eager is set")
            return

        if (
            codec_chunk_frames > 0
            and codec_left_context_frames > 0
            and self._decoder_sliding_window
            and codec_left_context_frames < self._decoder_sliding_window
        ):
            logger.warning(
                "Qwen3-TTS streaming codec_left_context_frames=%d "
                "is smaller than decoder sliding_window=%d; "
                "chunk-boundary distortion may occur. "
                "Increase codec_left_context_frames to at least "
                "%d for streaming.",
                codec_left_context_frames,
                self._decoder_sliding_window,
                self._decoder_sliding_window,
            )

        self.decoder.enable_cudagraph(
            capture_sizes=decode_cudagraph_capture_sizes,
            capture_batch_sizes=decode_cudagraph_batch_sizes,
            extra_capture_shapes=decode_cudagraph_extra_capture_shapes,
            compile_shapes=decode_compile_shapes,
            device=device,
            codec_chunk_frames=codec_chunk_frames,
            codec_left_context_frames=codec_left_context_frames,
            decode_chunk_size=self._decode_chunk_frames,
            decode_left_context=self._decode_left_context_frames,
        )
        logger.info("Code2Wav decoder CUDA Graph enabled")

    def _get_decode_batch_bucket_frames(self, actual_frames: int) -> int:
        for bucket_frames in self._decode_batch_bucket_frames:
            if actual_frames <= bucket_frames:
                return bucket_frames
        return actual_frames

    def _record_decode_batch_stats(
        self,
        *,
        group_size: int,
        bucket_frames: int,
        actual_frames: list[int],
    ) -> None:
        if not self._batch_stats_enabled:
            return

        self._batch_stats_groups += 1
        self._batch_stats_requests += group_size
        self._batch_stats_decoded_frames += group_size * bucket_frames
        self._batch_stats_padded_frames += sum(bucket_frames - frames for frames in actual_frames)
        self._batch_stats_actual_frames.update(actual_frames)
        self._batch_stats_bucket_groups[(group_size, bucket_frames)] += 1

    def log_decode_batch_stats(self) -> None:
        if not self._batch_stats_enabled or self._batch_stats_requests == 0:
            return

        avg_group_size = self._batch_stats_requests / max(1, self._batch_stats_groups)
        pad_ratio = self._batch_stats_padded_frames / max(1, self._batch_stats_decoded_frames)
        logger.info(
            "Code2Wav batch stats: forwards=%d groups=%d requests=%d "
            "avg_group_size=%.2f padded_frames=%d decoded_frames=%d pad_ratio=%.2f%% "
            "top_actual_frames=%s top_bucket_groups=%s",
            self._batch_stats_forwards,
            self._batch_stats_groups,
            self._batch_stats_requests,
            avg_group_size,
            self._batch_stats_padded_frames,
            self._batch_stats_decoded_frames,
            100.0 * pad_ratio,
            self._batch_stats_actual_frames.most_common(12),
            self._batch_stats_bucket_groups.most_common(12),
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode codec codes into audio waveform.

        input_ids layout per request: [codec_context_frames, *flat_codes]
        where flat_codes is codebook-major [q*F].

        Bypasses the HF Qwen3TTSTokenizer.decode() wrapper and calls the
        decoder.chunked_decode() directly to avoid GPU->CPU->GPU round-trips.
        Length management is done here instead of relying on HF's padding=-1
        sentinel logic.
        """
        self._batch_stats_forwards += 1
        decoder = self.decoder
        q = int(self._num_quantizers)
        upsample = int(self._total_upsample)
        sr_val = int(self._output_sample_rate)
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        parsed: list[tuple[int, int]] = []
        valid_codes_qf: list[torch.Tensor] = []
        valid_indices: list[int] = []
        left_context_size = [0] * len(request_ids_list)
        if runtime_additional_information is not None:
            for i, info in enumerate(runtime_additional_information):
                if i >= len(left_context_size):
                    break
                meta = info.get("meta", {})
                if "left_context_size" in meta:
                    left_context_size[i] = meta["left_context_size"]
        for i, req_ids in enumerate(request_ids_list):
            if req_ids.numel() < 1:
                parsed.append((0, 0))
                continue
            ctx_frames = left_context_size[i]
            flat = req_ids
            n = flat.numel()
            if n == 0 or n % q != 0:
                if n > 0:
                    key = (int(n), q)
                    if key not in self._logged_malformed_codec_lengths:
                        self._logged_malformed_codec_lengths.add(key)
                        logger.warning(
                            "Code2Wav input_ids length %d not divisible by num_quantizers %d; "
                            "skipping malformed request and suppressing repeats for this length.",
                            n,
                            q,
                        )
                parsed.append((0, 0))
                continue
            frames = n // q
            # [q*F] -> [Q, F] for direct decoder call (decoder expects [B, Q, F])
            codes_qf = flat.reshape(q, frames)
            parsed.append((ctx_frames, frames))
            valid_codes_qf.append(codes_qf)
            valid_indices.append(i)

        num_req = len(request_ids_list)
        if not valid_codes_qf:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={
                    "model_outputs": [empty] * num_req,
                    "sr": [sr_tensor] * num_req,
                },
            )

        if not self._logged_codec_stats:
            self._logged_codec_stats = True
            try:
                c = valid_codes_qf[0]
                logger.info(
                    "Code2Wav codec: frames=%d q=%d uniq=%d range=[%d,%d] batch=%d",
                    c.shape[1],
                    q,
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                    len(valid_codes_qf),
                )
            except Exception:
                pass

        wav_tensors: list[torch.Tensor | None] = [None] * len(valid_codes_qf)

        def _decode_group_chunks(group_chunks: list[list[tuple[int, torch.Tensor]]]) -> None:
            for group_chunk in group_chunks:
                actual_frames = [int(codes_qf.shape[1]) for _, codes_qf in group_chunk]
                target_frames = max(actual_frames)
                is_equal_length_batch = all(frames == target_frames for frames in actual_frames)
                use_variable_length_batch = (
                    len(group_chunk) > 1
                    and not is_equal_length_batch
                    and target_frames >= self._decode_variable_chunk_batch_min_frames
                    and hasattr(decoder, "batched_chunked_decode")
                )
                if len(group_chunk) == 1:
                    codes_bqf = group_chunk[0][1].unsqueeze(0)
                elif is_equal_length_batch:
                    codes_bqf = torch.stack([codes_qf for _, codes_qf in group_chunk], dim=0)
                else:
                    first = group_chunk[0][1]
                    codes_bqf = first.new_zeros((len(group_chunk), q, target_frames))
                    for row, (_, codes_qf) in enumerate(group_chunk):
                        codes_bqf[row, :, : codes_qf.shape[1]] = codes_qf
                self._record_decode_batch_stats(
                    group_size=len(group_chunk),
                    bucket_frames=target_frames,
                    actual_frames=actual_frames,
                )
                try:
                    if use_variable_length_batch:
                        wav_batch = decoder.batched_chunked_decode(
                            codes_bqf,
                            actual_frames,
                            chunk_size=self._decode_chunk_frames,
                            left_context_size=self._decode_left_context_frames,
                            max_batch_size=self._decode_batch_max_size,
                        )  # [B, 1, wav_len]
                    else:
                        wav_batch = decoder.chunked_decode(
                            codes_bqf,
                            chunk_size=self._decode_chunk_frames,
                            left_context_size=self._decode_left_context_frames,
                        )  # [B, 1, wav_len]
                except TypeError:
                    # Unit-test fakes and older decoder shims may not accept the
                    # explicit chunk kwargs; production Qwen3-TTS decoders do.
                    wav_batch = decoder.chunked_decode(codes_bqf)  # [B, 1, wav_len]

                if wav_batch.dim() == 3 and wav_batch.shape[1] == 1:
                    wav_rows = wav_batch[:, 0, :]
                elif wav_batch.dim() == 2:
                    wav_rows = wav_batch
                else:
                    raise ValueError(
                        "Code2Wav decoder returned unexpected shape "
                        f"{tuple(wav_batch.shape)} for batch size {len(group_chunk)}"
                    )
                if wav_rows.shape[0] != len(group_chunk):
                    raise ValueError(
                        f"Code2Wav decoder returned batch size {wav_rows.shape[0]} "
                        f"for input batch size {len(group_chunk)}"
                    )
                for row, (j, _) in enumerate(group_chunk):
                    wav_tensors[j] = wav_rows[row]

        # Group by configured frame buckets instead of only exact lengths.
        # For ordinary async streaming windows this is the real batching
        # opportunity; decoder-internal variable chunk batching is gated to
        # longer inputs where repeated full chunks can amortize its overhead.
        grouped_codes: dict[int, list[tuple[int, torch.Tensor]]] = {}
        for j, codes_qf in enumerate(valid_codes_qf):
            frames = int(codes_qf.shape[1])
            grouped_codes.setdefault(self._get_decode_batch_bucket_frames(frames), []).append((j, codes_qf))

        for _bucket_frames, group in grouped_codes.items():
            if self._decode_batch_max_size > 0 and len(group) > self._decode_batch_max_size:
                # Keep each decoder call inside the configured CUDA graph batch
                # envelope. Sorting by length lowers right-padding within each
                # split while outputs are restored by original request index.
                group = sorted(group, key=lambda item: int(item[1].shape[1]))
                group_chunks = [
                    group[start : start + self._decode_batch_max_size]
                    for start in range(0, len(group), self._decode_batch_max_size)
                ]
            else:
                group_chunks = [group]
            _decode_group_chunks(group_chunks)

        if self._batch_stats_log_every > 0 and self._batch_stats_forwards % self._batch_stats_log_every == 0:
            self.log_decode_batch_stats()

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            ctx_frames, actual_frames = parsed[idx]
            wav = wav_tensors[j]
            assert wav is not None
            # Slice on exact codec-frame boundaries instead of proportionally.
            start = max(0, ctx_frames * upsample)
            end = max(start, actual_frames * upsample)
            if start >= wav.shape[0]:
                logger.warning(
                    "Context trim start %d >= decoded length %d; returning empty audio.",
                    start,
                    wav.shape[0],
                )
                continue
            wav = wav[start : min(end, wav.shape[0])]
            if wav.shape[0] > 0:
                audios[idx] = wav.to(dtype=torch.float32).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(
                "Qwen3TTSCode2Wav expected OmniOutput, OmniOutput tuple, "
                f"or (audio_tensor, sr) outputs, got {type(model_outputs)}"
            )

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": sr,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The primary weights iterator contains no Code2Wav parameters.
        # Drain it so callers don't hang on an unconsumed generator.
        for _ in weights:
            pass

        # Load decoder weights from the speech_tokenizer/ subfolder
        # via vLLM's weight loader (handles sharded safetensors, index
        # files, and all load formats).  AutoWeightsLoader matches
        # "decoder.*" weights to self.decoder and skips encoder weights.
        model_loader = DefaultModelLoader(self.vllm_config.load_config)
        source = DefaultModelLoader.Source(
            model_or_path=self.model_path,
            revision=self.vllm_config.model_config.revision,
            subfolder="speech_tokenizer",
        )
        subfolder_weights = model_loader._get_weights_iterator(source)
        loaded = AutoWeightsLoader(
            self,
            skip_prefixes=["encoder."],
        ).load_weights(subfolder_weights)

        device = self.vllm_config.device_config.device
        self.decoder.to(device=device, dtype=torch.float32)

        # Precompute SnakeBeta exp caches (benefits both Triton and eager paths)
        if hasattr(self.decoder, "precompute_snake_caches"):
            self.decoder.precompute_snake_caches()

        # The connector codec chunk settings control inter-stage streaming
        # windows. Keep decoder-internal chunking separate; using the small
        # streaming window here causes repeated overlap decode in Code2Wav.
        codec_chunk_frames = 0
        codec_left_context_frames = 0
        model_cfg = getattr(self.vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        extra_cfg = (
            connector_cfg.get("extra", connector_cfg)
            if isinstance(connector_cfg, dict)
            else getattr(connector_cfg, "extra", None)
        )

        def _get_int_config(name: str, default: int) -> int:
            value = extra_cfg.get(name, default)
            if value is None:
                return default
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}") from exc

        def _get_bool_config(name: str, default: bool) -> bool:
            value = extra_cfg.get(name, default)
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("1", "true", "yes", "on"):
                    return True
                if lowered in ("0", "false", "no", "off"):
                    return False
            if isinstance(value, int):
                return bool(value)
            raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}")

        def _get_int_list_config(name: str) -> list[int] | None:
            value = extra_cfg.get(name)
            if value is None:
                return None
            if isinstance(value, str):
                raw_values = [item.strip() for item in value.split(",") if item.strip()]
            elif isinstance(value, int):
                raw_values = [value]
            else:
                try:
                    raw_values = list(value)
                except TypeError as exc:
                    raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}") from exc
            values: set[int] = set()
            for item in raw_values:
                try:
                    parsed = int(item)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}") from exc
                if parsed > 0:
                    values.add(parsed)
            return sorted(values)

        def _get_int_pair_list_config(name: str) -> list[tuple[int, int]] | None:
            value = extra_cfg.get(name)
            if value is None:
                return None
            if isinstance(value, str):
                raw_values = [item.strip() for item in value.split(",") if item.strip()]
            else:
                try:
                    raw_values = list(value)
                except TypeError as exc:
                    raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}") from exc

            pairs: set[tuple[int, int]] = set()
            for item in raw_values:
                if isinstance(item, str):
                    if ":" not in item:
                        raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}")
                    left, right = item.split(":", 1)
                    raw_pair = (left.strip(), right.strip())
                else:
                    try:
                        raw_pair = tuple(item)
                    except TypeError as exc:
                        raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}") from exc
                    if len(raw_pair) != 2:
                        raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}")
                try:
                    batch_size = int(raw_pair[0])
                    seq_len = int(raw_pair[1])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid Qwen3-TTS Code2Wav config {name}={value!r}") from exc
                if batch_size > 0 and seq_len > 0:
                    pairs.add((batch_size, seq_len))
            return sorted(pairs)

        if isinstance(extra_cfg, dict):
            codec_chunk_frames = int(extra_cfg.get("codec_chunk_frames") or 0)
            codec_left_context_frames = int(extra_cfg.get("codec_left_context_frames") or 0)
            decode_chunk_frames = _get_int_config("decode_chunk_frames", self._decode_chunk_frames)
            decode_left_context_frames = _get_int_config(
                "decode_left_context_frames",
                self._decode_left_context_frames,
            )
            if decode_chunk_frames <= 0 or decode_left_context_frames < 0:
                raise ValueError(
                    "Invalid Qwen3-TTS Code2Wav decode chunk config: "
                    f"decode_chunk_frames={decode_chunk_frames}, "
                    f"decode_left_context_frames={decode_left_context_frames}"
                )
            self._decode_chunk_frames = decode_chunk_frames
            self._decode_left_context_frames = decode_left_context_frames
            decode_cudagraph_capture_sizes = _get_int_list_config("decode_cudagraph_capture_sizes")
            decode_cudagraph_batch_sizes = _get_int_list_config("decode_cudagraph_batch_sizes")
            decode_cudagraph_extra_capture_shapes = _get_int_pair_list_config("decode_cudagraph_extra_capture_shapes")
            decode_compile_shapes = _get_int_pair_list_config("decode_compile_shapes")
            decode_batch_bucket_frames = _get_int_list_config("decode_batch_bucket_frames")
            if decode_batch_bucket_frames is not None:
                self._decode_batch_bucket_frames = decode_batch_bucket_frames
            decode_batch_max_size = _get_int_config("decode_batch_max_size", self._decode_batch_max_size)
            if decode_batch_max_size < 0:
                raise ValueError(f"Invalid Qwen3-TTS Code2Wav config decode_batch_max_size={decode_batch_max_size}")
            self._decode_batch_max_size = decode_batch_max_size
            decode_variable_chunk_batch_min_frames = _get_int_config(
                "decode_variable_chunk_batch_min_frames",
                self._decode_variable_chunk_batch_min_frames,
            )
            if decode_variable_chunk_batch_min_frames < 0:
                raise ValueError(
                    "Invalid Qwen3-TTS Code2Wav config "
                    f"decode_variable_chunk_batch_min_frames={decode_variable_chunk_batch_min_frames}"
                )
            self._decode_variable_chunk_batch_min_frames = decode_variable_chunk_batch_min_frames
            decode_enable_tf32 = _get_bool_config("decode_enable_tf32", False)
        else:
            decode_cudagraph_capture_sizes = None
            decode_cudagraph_batch_sizes = None
            decode_cudagraph_extra_capture_shapes = None
            decode_compile_shapes = None
            decode_enable_tf32 = False

        if decode_enable_tf32 and device.type == "cuda":
            # PyTorch exposes TF32 controls as process-wide CUDA backend
            # switches. This opt-in is intended for deployments where
            # Code2Wav runs in its own Stage1 worker process.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            logger.info(
                "Qwen3-TTS Code2Wav TF32 enabled process-wide: "
                "matmul.allow_tf32=%s cudnn.allow_tf32=%s float32_matmul_precision=%s",
                torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32,
                torch.get_float32_matmul_precision(),
            )

        if hasattr(self.decoder, "enable_cudagraph") and device.type == "cuda":
            try:
                self._maybe_enable_decoder_cudagraph(
                    device=device,
                    codec_chunk_frames=codec_chunk_frames,
                    codec_left_context_frames=codec_left_context_frames,
                    decode_cudagraph_capture_sizes=decode_cudagraph_capture_sizes,
                    decode_cudagraph_batch_sizes=decode_cudagraph_batch_sizes,
                    decode_cudagraph_extra_capture_shapes=decode_cudagraph_extra_capture_shapes,
                    decode_compile_shapes=decode_compile_shapes,
                )
            except Exception:
                logger.warning(
                    "Failed to enable CUDA Graph for Code2Wav decoder",
                    exc_info=True,
                )

        return loaded
