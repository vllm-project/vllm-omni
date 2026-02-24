from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers.utils.hub import cached_file
from vllm.config import VllmConfig
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

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode codec codes into audio waveform.

        input_ids layout: [codec_context_frames, *flat_codes]
        where flat_codes is codebook-major [q*F].
        """
        tok = self._ensure_speech_tokenizer_loaded()
        assert self._num_quantizers is not None
        assert self._output_sample_rate is not None

        sr_val = self._output_sample_rate
        empty_ret = (
            torch.zeros((0,), dtype=torch.float32),
            torch.tensor(sr_val, dtype=torch.int32),
        )

        if input_ids is None:
            return empty_ret

        q = int(self._num_quantizers)
        ids = input_ids.reshape(-1).to(dtype=torch.long)
        n_tokens = ids.numel()

        if n_tokens == 0:
            return empty_ret

        # input_ids[0] = codec_context_frames (prepended by stage_input_processor).
        ctx_frames = int(ids[0].item())
        ids = ids[1:]
        n_tokens = ids.numel()

        if n_tokens == 0:
            return empty_ret

        # Warmup / dummy_run: not divisible by num_quantizers.
        if n_tokens % q != 0:
            logger.warning(
                "Code2Wav input_ids length %d not divisible by num_quantizers %d, "
                "likely a warmup run; returning empty audio.",
                n_tokens,
                q,
            )
            return empty_ret

        total_frames = n_tokens // q

        # Reshape codebook-major flat [q*F] -> [q, F] -> [F, q] for SpeechTokenizer.
        codes_fq = ids.reshape(q, total_frames).transpose(0, 1).contiguous()

        if not self._logged_codec_stats and total_frames > 1:
            self._logged_codec_stats = True
            try:
                uniq = int(torch.unique(codes_fq).numel())
                cmin = int(codes_fq.min().item())
                cmax = int(codes_fq.max().item())
                head = codes_fq[: min(2, total_frames), : min(8, q)].cpu().tolist()
                logger.info(
                    "Code2Wav codec: frames=%d q=%d uniq=%d range=[%d,%d] head=%s",
                    total_frames,
                    q,
                    uniq,
                    cmin,
                    cmax,
                    head,
                )
            except Exception:
                pass

        wavs, sr = tok.decode({"audio_codes": codes_fq})
        if not wavs:
            raise ValueError("SpeechTokenizer code2wav produced empty waveform list.")
        audio_np = wavs[0].astype(np.float32, copy=False)

        # Trim left-context waveform samples (streaming sliding window).
        if ctx_frames > 0:
            upsample = self._decode_upsample_rate
            if upsample is None or upsample <= 0:
                raise ValueError(f"Invalid decode upsample rate: {upsample}")
            cut = ctx_frames * upsample
            if cut < audio_np.shape[0]:
                audio_np = audio_np[cut:]
            else:
                logger.warning(
                    "Context trim %d >= decoded length %d; returning empty audio.",
                    cut,
                    audio_np.shape[0],
                )
                return empty_ret

        audio_tensor = torch.from_numpy(audio_np).to(dtype=torch.float32).reshape(-1)
        sr_tensor = torch.tensor(int(sr), dtype=torch.int32)
        return audio_tensor, sr_tensor

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
