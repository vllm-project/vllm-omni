from __future__ import annotations

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
        self._logged_codec_stats = False

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
                    logger.warning(
                        "Code2Wav input_ids length %d not divisible by num_quantizers %d; skipping malformed request.",
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

        # Decode directly via decoder.chunked_decode(), staying entirely on GPU.
        # Each request decoded individually with CUDA graph replay at bs=1.
        wav_tensors: list[torch.Tensor] = []
        for codes_qf in valid_codes_qf:
            codes_bqf = codes_qf.unsqueeze(0)  # [1, Q, F]
            try:
                wav = decoder.chunked_decode(
                    codes_bqf,
                    chunk_size=self._decode_chunk_frames,
                    left_context_size=self._decode_left_context_frames,
                )  # [1, 1, wav_len]
            except TypeError:
                # Unit-test fakes and older decoder shims may not accept the
                # explicit chunk kwargs; production Qwen3-TTS decoders do.
                wav = decoder.chunked_decode(codes_bqf)  # [1, 1, wav_len]
            wav_tensors.append(wav.squeeze(0).squeeze(0))  # [wav_len]

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            ctx_frames, actual_frames = parsed[idx]
            wav = wav_tensors[j]
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

        if hasattr(self.decoder, "enable_cudagraph") and device.type == "cuda":
            try:
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
                    device=device,
                    codec_chunk_frames=codec_chunk_frames,
                    codec_left_context_frames=codec_left_context_frames,
                    decode_chunk_size=self._decode_chunk_frames,
                    decode_left_context=self._decode_left_context_frames,
                )
                logger.info("Code2Wav decoder CUDA Graph enabled")
            except Exception:
                logger.warning(
                    "Failed to enable CUDA Graph for Code2Wav decoder",
                    exc_info=True,
                )

        return loaded
