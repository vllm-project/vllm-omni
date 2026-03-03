from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .qwen3_tts_tokenizer import Qwen3TTSTokenizer

logger = init_logger(__name__)


class Qwen3TTSCode2Wav(nn.Module):
    """Stage-1 code2wav model for Qwen3-TTS (GenerationModelRunner).
    Consumes frame-aligned codec tokens from input_ids and decodes waveform via SpeechTokenizer."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        # Generation-only stage (no logits / sampling).
        self.requires_raw_input_tokens = True

        self._speech_tokenizer: Qwen3TTSTokenizer | None = None
        self._num_quantizers: int | None = None
        self._decode_upsample_rate: int | None = None
        self._output_sample_rate: int | None = None
        self._logged_codec_stats = False

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def _ensure_speech_tokenizer_loaded(self) -> Qwen3TTSTokenizer:
        if self._speech_tokenizer is not None:
            return self._speech_tokenizer

        # Locate speech_tokenizer dir from HF cache (or local path).
        cfg_path = cached_file(self.model_path, "speech_tokenizer/config.json")
        if cfg_path is None:
            raise ValueError(f"{self.model_path}/speech_tokenizer/config.json not found")
        speech_tokenizer_dir = os.path.dirname(cfg_path)

        # Stage-1 only needs decode; skip HF feature extractor to avoid heavy optional deps.
        # Still require preprocessor_config.json (use cached_file so online runs can fetch it).
        prep_cfg = cached_file(self.model_path, "speech_tokenizer/preprocessor_config.json")
        if prep_cfg is None:
            raise ValueError(
                f"{self.model_path}/speech_tokenizer/preprocessor_config.json not found. "
                "Please make sure the checkpoint contains the required HF preprocessing files."
            )

        tok = Qwen3TTSTokenizer.from_pretrained(
            speech_tokenizer_dir,
            torch_dtype=torch.bfloat16,
            load_feature_extractor=False,
        )

        # Align device with vLLM worker, then read back from module.
        if tok.model is not None:
            tok.model.to(device=self.vllm_config.device_config.device)
            tok.device = self._module_device(tok.model)

        # Derive codec group count and rates from tokenizer config.
        dec_cfg = getattr(tok.model.config, "decoder_config", None)
        num_q = getattr(dec_cfg, "num_quantizers", None) if dec_cfg is not None else None
        if num_q is None:
            raise ValueError("speech_tokenizer decoder_config.num_quantizers not found")
        num_q = int(num_q)
        if num_q <= 0:
            raise ValueError(f"Invalid speech_tokenizer num_quantizers={num_q}")

        try:
            upsample = int(tok.get_decode_upsample_rate())
        except Exception as e:
            raise ValueError(f"Failed to get decode upsample rate: {e}") from e
        if upsample <= 0:
            raise ValueError(f"Invalid decode upsample rate: {upsample}")

        try:
            out_sr = int(tok.get_output_sample_rate())
        except Exception as e:
            raise ValueError(f"Failed to get output sample rate: {e}") from e

        self._speech_tokenizer = tok
        self._num_quantizers = num_q
        self._decode_upsample_rate = upsample
        self._output_sample_rate = out_sr
        return tok

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
        **kwargs: Any,
    ) -> OmniOutput:
        """Decode codec codes into audio waveform.

        input_ids layout per request: [codec_context_frames, *flat_codes]
        where flat_codes is codebook-major [q*F].

        When batched, uses forward context ubatch_slices to split the
        concatenated input_ids and decode via a single batched forward pass.
        """
        self._ensure_speech_tokenizer_loaded()
        assert self._num_quantizers is not None
        assert self._decode_upsample_rate is not None
        assert self._output_sample_rate is not None

        tok = self._speech_tokenizer
        q = int(self._num_quantizers)
        upsample = int(self._decode_upsample_rate)
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

        # Parse each request: extract ctx_frames, validate, reshape codes.
        # input_ids layout per request: [codec_context_frames, *flat_codes]
        # where flat_codes is codebook-major [q*F].
        parsed = []  # (ctx_frames, actual_frames)
        valid_codes = []
        valid_indices = []
        for i, req_ids in enumerate(request_ids_list):
            if req_ids.numel() < 2:
                parsed.append((0, 0))
                continue
            ctx_frames = int(req_ids[0].item())
            flat = req_ids[1:]
            n = flat.numel()
            # Warmup / dummy_run: not divisible by num_quantizers.
            if n == 0 or n % q != 0:
                if n > 0:
                    logger.warning(
                        "Code2Wav input_ids length %d not divisible by num_quantizers %d, "
                        "likely a warmup run; returning empty audio.",
                        n,
                        q,
                    )
                parsed.append((0, 0))
                continue
            frames = n // q
            # Reshape codebook-major flat [q*F] -> [q, F] -> [F, q] for SpeechTokenizer.
            codes_fq = flat.reshape(q, frames).transpose(0, 1).contiguous()
            parsed.append((ctx_frames, frames))
            valid_codes.append({"audio_codes": codes_fq})
            valid_indices.append(i)

        num_req = len(request_ids_list)
        if not valid_codes:
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
                c = valid_codes[0]["audio_codes"]
                logger.info(
                    "Code2Wav codec: frames=%d q=%d uniq=%d range=[%d,%d] head=%s batch=%d",
                    c.shape[0],
                    q,
                    int(torch.unique(c).numel()),
                    int(c.min().item()),
                    int(c.max().item()),
                    c[: min(2, c.shape[0]), : min(8, q)].cpu().tolist(),
                    len(valid_codes),
                )
            except Exception:
                pass

        # Batched decode: single forward pass through SpeechTokenizer.
        wavs, _ = tok.decode(valid_codes)
        if len(wavs) != len(valid_codes):
            raise RuntimeError(f"Code2Wav returned {len(wavs)} waveforms for {len(valid_codes)} requests")

        # Build per-request outputs, trimming padding and left-context.
        audios = [empty] * num_req
        srs = [sr_tensor] * num_req

        for j, idx in enumerate(valid_indices):
            ctx_frames, actual_frames = parsed[idx]
            audio_np = wavs[j].astype(np.float32, copy=False)
            # Trim decoder padding (output may be longer due to batch padding).
            expected_len = actual_frames * upsample
            if audio_np.shape[0] > expected_len:
                audio_np = audio_np[:expected_len]
            # Trim left-context waveform samples (streaming sliding window).
            if ctx_frames > 0:
                cut = ctx_frames * upsample
                if cut < audio_np.shape[0]:
                    audio_np = audio_np[cut:]
                else:
                    logger.warning(
                        "Context trim %d >= decoded length %d; returning empty audio.",
                        cut,
                        audio_np.shape[0],
                    )
                    continue
            if audio_np.shape[0] > 0:
                audios[idx] = torch.from_numpy(audio_np).to(dtype=torch.float32).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs

        if not (isinstance(model_outputs, tuple) and len(model_outputs) == 2):
            raise TypeError(f"Qwen3TTSCode2Wav expected (audio_tensor, sr) outputs, got {type(model_outputs)}")

        audio_tensor, sr = model_outputs
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": audio_tensor,
                "sr": sr,
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # SpeechTokenizer weights live under `speech_tokenizer/` and are loaded
        # lazily from that directory. Ignore main checkpoint weights.
        return set()
