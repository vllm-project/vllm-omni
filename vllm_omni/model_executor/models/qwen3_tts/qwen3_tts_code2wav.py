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
from vllm_omni.model_executor.stage_input_processors.chunk_size_utils import parse_chunk_ramp

from .tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

logger = init_logger(__name__)

_DUMMY_REQUEST_ID = "__qwen3_tts_dummy_run__"


def _codec_ids_from_payload_or_input(
    input_ids: torch.Tensor,
    runtime_info: dict[str, Any] | None,
) -> torch.Tensor:
    """Prefer connector-delivered codec ids over token placeholders.

    In non-async full-payload mode, the scheduler only needs placeholder
    token ids for allocation.  The real codec sequence is delivered through
    model_intermediate_buffer as ``codes.audio``.
    """
    if isinstance(runtime_info, dict):
        codes = runtime_info.get("codes")
        if isinstance(codes, dict):
            audio = codes.get("audio")
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                return audio.reshape(-1).to(device=input_ids.device, dtype=torch.long)
            if isinstance(audio, (list, tuple)) and audio:
                return torch.as_tensor(audio, device=input_ids.device, dtype=torch.long).reshape(-1)
    return input_ids.reshape(-1).to(dtype=torch.long)


class Qwen3TTSCode2Wav(nn.Module):
    """Stage-1 code2wav model for Qwen3-TTS (GenerationModelRunner).
    Consumes frame-aligned codec tokens from input_ids and decodes waveform
    via the SpeechTokenizer decoder directly (bypassing HF wrapper overhead)."""

    input_modalities = "audio"

    # Ask the model runner for the scheduler-side request IDs. Stateful
    # decoder caches must use the same IDs delivered by on_requests_finished;
    # payload metadata carries an external ID which may differ.
    requires_request_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self._async_chunk = bool(getattr(vllm_config.model_config, "async_chunk", False))

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        self._decode_chunk_frames = 300
        self._decode_left_context_frames = 25
        self._decode_batch_max_size = 0
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
        self._decoder_state_cache: dict[str, dict[str, Any]] = {}
        self._decoder_state_cache_warn_entries = 512

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings. Keep a stable dummy embedding for vLLM runner.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        """Provide request state metadata for vLLM's external graph dummy run."""
        return [
            {"meta": {"request_id": f"{_DUMMY_REQUEST_ID}{index}", "finished": True}}
            for index in range(max(0, int(num_reqs)))
        ]

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
        initial_codec_chunk_frames: int,
        codec_chunk_ramp: list[int] | None,
        decode_cudagraph_batch_sizes: list[int] | None,
        decode_cudagraph_capture_sizes: list[int] | None,
    ) -> None:
        """Enable inner Code2Wav CUDA graph unless stage is enforce_eager."""
        if not hasattr(self.decoder, "enable_cudagraph") or device.type != "cuda":
            return

        model_cfg = getattr(self.vllm_config, "model_config", None)
        if getattr(model_cfg, "enforce_eager", False):
            logger.info("Qwen3-TTS Code2Wav CUDA Graph disabled because enforce_eager is set")
            return

        if (
            self._async_chunk
            and codec_chunk_frames > 0
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
            capture_batch_sizes=decode_cudagraph_batch_sizes,
            stateless_capture_sizes=decode_cudagraph_capture_sizes,
            device=device,
            codec_chunk_frames=codec_chunk_frames,
            codec_left_context_frames=codec_left_context_frames,
            initial_codec_chunk_frames=initial_codec_chunk_frames,
            codec_chunk_ramp=codec_chunk_ramp,
            async_chunk=self._async_chunk,
            decode_chunk_size=self._decode_chunk_frames,
            decode_left_context=self._decode_left_context_frames,
        )
        logger.info("Code2Wav decoder CUDA Graph enabled")

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

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        for req_id in finished_req_ids:
            self._decoder_state_cache.pop(req_id, None)

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
        sr_val = int(self._output_sample_rate)
        sr_tensor = torch.tensor(sr_val, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        runtime_infos = runtime_additional_information or []
        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))

        internal_request_ids = kwargs.get("request_ids")
        valid_codes_qf: list[tuple[str | None, torch.Tensor]] = []
        valid_indices: list[int] = []
        left_context_size = [0] * len(request_ids_list)
        ref_context_size = [0] * len(request_ids_list)
        segment_finished_flags = [False] * len(request_ids_list)
        request_state_ids: list[str | None] = [None] * len(request_ids_list)
        ref_context_request_ids: list[str | None] = [None] * len(request_ids_list)
        ref_context_included = [False] * len(request_ids_list)
        finished_flags = [False] * len(request_ids_list)

        def _meta_int(value: Any) -> int:
            if isinstance(value, list):
                value = value[0] if value else 0
            if isinstance(value, torch.Tensor):
                value = value.reshape(-1)[0].item() if value.numel() > 0 else 0
            return int(value or 0)

        def _meta_str(value: Any) -> str | None:
            if isinstance(value, list):
                value = value[0] if value else None
            if value is None:
                return None
            return str(value)

        def _meta_bool(value: Any) -> bool:
            if isinstance(value, list):
                value = value[0] if value else False
            if isinstance(value, torch.Tensor):
                return bool(value.reshape(-1)[0].item()) if value.numel() > 0 else False
            return bool(value)

        if runtime_infos:
            for i, info in enumerate(runtime_infos):
                if i >= len(ref_context_size):
                    break
                if not isinstance(info, dict):
                    continue
                meta = info.get("meta", {})
                if "is_segment_finished" in meta:
                    segment_finished_flags[i] = _meta_bool(meta["is_segment_finished"])
                if "left_context_size" in meta:
                    left_context_size[i] = _meta_int(meta["left_context_size"])
                if "ref_context_size" in meta:
                    ref_context_size[i] = _meta_int(meta["ref_context_size"])
                if "ref_context_request_id" in meta:
                    ref_context_request_ids[i] = _meta_str(meta["ref_context_request_id"])
                if "request_id" in meta:
                    request_state_ids[i] = _meta_str(meta["request_id"])
                if "ref_context_included" in meta:
                    ref_context_included[i] = _meta_bool(meta["ref_context_included"])
                if "finished" in meta:
                    finished_flags[i] = _meta_bool(meta["finished"])

        # Normal runner calls provide scheduler-side IDs, which are also used
        # by scheduler_output.finished_req_ids. Direct forward calls and CUDA
        # Graph dummy runs have no runner IDs, so retain the payload ID as a
        # fallback for those paths.
        cache_request_ids = request_state_ids.copy()
        if internal_request_ids is not None:
            for i, request_id in enumerate(internal_request_ids):
                if i >= len(cache_request_ids):
                    break
                cache_request_ids[i] = str(request_id)

        for i, req_ids in enumerate(request_ids_list):
            runtime_info = runtime_infos[i] if i < len(runtime_infos) else None
            req_ids = _codec_ids_from_payload_or_input(req_ids, runtime_info)
            if req_ids.numel() < 1:
                continue
            ref_ctx_frames = ref_context_size[i]
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
                continue
            frames = n // q
            # [q*F] -> [Q, F] for direct decoder call (decoder expects [B, Q, F])
            codes_qf = flat.reshape(q, frames)
            ref_req_id = ref_context_request_ids[i]
            state_req_id = cache_request_ids[i] if self._async_chunk else None
            is_new_state = state_req_id is not None and state_req_id not in self._decoder_state_cache
            if is_new_state and ref_req_id is not None and ref_ctx_frames > 0:
                if not ref_context_included[i] or frames < ref_ctx_frames:
                    raise ValueError("Qwen3-TTS async_chunk first ICL chunk must include its declared reference prefix")
            valid_codes_qf.append((state_req_id, codes_qf))
            if state_req_id is not None:
                state = self._decoder_state_cache.get(state_req_id)
                if state is None:
                    if len(self._decoder_state_cache) >= self._decoder_state_cache_warn_entries:
                        logger.warning_once(
                            "Qwen3-TTS decoder state cache exceeded the expected active-request envelope: "
                            "entries=%d threshold=%d. Keeping all active states; check request cleanup paths.",
                            len(self._decoder_state_cache),
                            self._decoder_state_cache_warn_entries,
                        )
                    state = {}
                    self._decoder_state_cache[state_req_id] = state
                state.setdefault("prefix_frames", 0)
                if state_req_id.startswith(_DUMMY_REQUEST_ID):
                    state["_is_dummy_run"] = True
                if ref_ctx_frames > 0:
                    cached_prefix_frames = state.setdefault("prefix_frames", ref_ctx_frames)
                    if cached_prefix_frames == 0:
                        state["prefix_frames"] = ref_ctx_frames
                        cached_prefix_frames = ref_ctx_frames
                    if cached_prefix_frames != ref_ctx_frames:
                        raise ValueError(
                            "Qwen3-TTS ref context size changed within request "
                            f"{ref_req_id!r}: cached={cached_prefix_frames}, current={ref_ctx_frames}"
                        )
            valid_indices.append(i)

        num_req = len(request_ids_list)
        if not valid_codes_qf:
            for req_id, finished, segment_finished in zip(
                cache_request_ids,
                finished_flags,
                segment_finished_flags,
                strict=False,
            ):
                if req_id is not None and (finished or segment_finished):
                    self._decoder_state_cache.pop(req_id, None)
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
                _, c = valid_codes_qf[0]
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

        request_states: list[dict[str, Any]] | None = None
        if self._async_chunk:
            missing_state = [index for index, (state_req_id, _) in enumerate(valid_codes_qf) if state_req_id is None]
            if missing_state:
                raise ValueError(
                    "Qwen3-TTS async_chunk Code2Wav inputs require a request_id; "
                    f"missing state for batch indices {missing_state}"
                )
            request_states = [
                self._decoder_state_cache[state_req_id]
                for state_req_id, _ in valid_codes_qf
                if state_req_id is not None
            ]

        request_lengths = [int(codes_qf.shape[-1]) for _, codes_qf in valid_codes_qf]
        max_request_length = max(request_lengths)
        request_codes = valid_codes_qf[0][1].new_zeros((len(valid_codes_qf), q, max_request_length))
        for row, (_, codes_qf) in enumerate(valid_codes_qf):
            request_codes[row, :, : codes_qf.shape[-1]].copy_(codes_qf)

        self._record_decode_batch_stats(
            group_size=len(valid_codes_qf),
            bucket_frames=max_request_length,
            actual_frames=request_lengths,
        )
        request_wavs = decoder.batched_chunked_decode(
            request_codes,
            request_lengths,
            caches=request_states,
            chunk_size=self._decode_chunk_frames,
            left_context_size=self._decode_left_context_frames,
            max_batch_size=self._decode_batch_max_size,
        )
        if len(request_wavs) != len(valid_codes_qf):
            raise ValueError(
                f"Qwen3-TTS batched decoder returned {len(request_wavs)} outputs for {len(valid_codes_qf)} requests"
            )
        wav_tensors: list[torch.Tensor] = []
        for row in range(len(valid_codes_qf)):
            wav = request_wavs[row]
            if wav.dim() == 2 and wav.shape[0] == 1:
                wav = wav[0]
            elif wav.dim() != 1:
                raise ValueError(f"Qwen3-TTS batched decoder returned unexpected row shape {tuple(wav.shape)}")
            if request_states is None:
                start = left_context_size[valid_indices[row]] * self._total_upsample
                wav = wav[start:]
            wav_tensors.append(wav)

        if self._batch_stats_log_every > 0 and self._batch_stats_forwards % self._batch_stats_log_every == 0:
            self.log_decode_batch_stats()

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            wav = wav_tensors[j]
            assert wav is not None
            if wav.numel() == 0:
                continue
            if wav.shape[0] > 0:
                # Decoder already runs in fp32, so the .to(float32) is a redundant dispatch.
                audios[idx] = (wav if wav.dtype == torch.float32 else wav.to(torch.float32)).reshape(-1)

        for req_id, finished, segment_finished in zip(
            cache_request_ids,
            finished_flags,
            segment_finished_flags,
            strict=False,
        ):
            if req_id is not None and (finished or segment_finished):
                self._decoder_state_cache.pop(req_id, None)

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
            initial_codec_chunk_frames = int(extra_cfg.get("initial_codec_chunk_frames") or 1)
            codec_chunk_ramp = parse_chunk_ramp(extra_cfg, steady=codec_chunk_frames) if self._async_chunk else None
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
            decode_cudagraph_batch_sizes = _get_int_list_config("decode_cudagraph_batch_sizes")
            decode_cudagraph_capture_sizes = (
                None if self._async_chunk else _get_int_list_config("decode_cudagraph_capture_sizes")
            )
            decode_batch_max_size = _get_int_config("decode_batch_max_size", self._decode_batch_max_size)
            if decode_batch_max_size < 0:
                raise ValueError(f"Invalid Qwen3-TTS Code2Wav config decode_batch_max_size={decode_batch_max_size}")
            self._decode_batch_max_size = decode_batch_max_size
            decode_enable_tf32 = _get_bool_config("decode_enable_tf32", False)
        else:
            codec_chunk_frames = 0
            codec_left_context_frames = 0
            initial_codec_chunk_frames = 1
            codec_chunk_ramp = None
            decode_cudagraph_batch_sizes = None
            decode_cudagraph_capture_sizes = None
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

        self.decoder._initial_codec_chunk_frames = initial_codec_chunk_frames
        self.decoder._incremental_chunk_frames = codec_chunk_frames or 25
        self.decoder._incremental_chunk_ramp = list(codec_chunk_ramp or ())

        if hasattr(self.decoder, "enable_cudagraph") and device.type == "cuda":
            try:
                self._maybe_enable_decoder_cudagraph(
                    device=device,
                    codec_chunk_frames=codec_chunk_frames,
                    codec_left_context_frames=codec_left_context_frames,
                    initial_codec_chunk_frames=initial_codec_chunk_frames,
                    codec_chunk_ramp=codec_chunk_ramp,
                    decode_cudagraph_batch_sizes=decode_cudagraph_batch_sizes,
                    decode_cudagraph_capture_sizes=decode_cudagraph_capture_sizes,
                )
            except Exception:
                logger.warning(
                    "Failed to enable CUDA Graph for Code2Wav decoder",
                    exc_info=True,
                )

        return loaded
