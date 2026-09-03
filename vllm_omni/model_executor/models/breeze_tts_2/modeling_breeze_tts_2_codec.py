"""Breeze-TTS-2 stage-1 codec (codec codes -> waveform).

The talker emits a frame-major ``(T, 16)`` matrix.  The stage input
processor flattens it codebook-major; this module restores the ``(T, 16)``
layout and delegates decoding to Breeze's bundled Qwen3-TTS tokenizer.  A
Mimi fallback remains available for checkpoints that omit ``audio_tokenizer``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput


def _split_flat_ids(ids: torch.Tensor, counts: list[int] | None) -> list[torch.Tensor]:
    if counts is None or len(counts) <= 1:
        return [ids]
    boundaries = [0]
    for count in counts:
        boundaries.append(boundaries[-1] + int(count))
    return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(counts))]


class BreezeTTS2MimiCodec(nn.Module):
    """Synchronous Breeze codec stage used by ``GenerationModelRunner``."""

    input_modalities = "audio"
    requires_raw_input_tokens = True
    have_multimodal_outputs = True
    has_preprocess = False
    has_postprocess = False
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.model_path = vllm_config.model_config.model

        self._audio_tokenizer = None
        codec_config = getattr(self.config, "codec_config", None)
        if isinstance(codec_config, Mapping):
            codec_kwargs = dict(codec_config)
        elif codec_config is not None:
            codec_kwargs = codec_config.to_dict()
        else:
            codec_kwargs = {}
        codec_kwargs.pop("model_type", None)
        codec_kwargs.pop("architectures", None)

        # Breeze checkpoints normally carry a Qwen3-TTS decoder in this
        # subdirectory.  Avoid constructing a second, unused Mimi network in
        # that case; this saves both startup time and worker memory.
        self._codec = None
        if not (Path(self.model_path) / "audio_tokenizer").is_dir():
            # Import lazily: stage 0 does not need the Mimi implementation.
            from transformers import MimiConfig, MimiModel

            self._codec = MimiModel(MimiConfig(**codec_kwargs)).eval()
            for parameter in self._codec.parameters():
                parameter.requires_grad_(False)

        self._num_codebooks = int(getattr(self.config, "num_codebooks", 16))
        self._codebook_size = int(codec_kwargs.get("codebook_size", 2048))
        self._sample_rate = int(
            codec_kwargs.get("sampling_rate", getattr(self.config, "sampling_rate", 24000))
        )
        self._loaded_local_weights = False

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Return stable dummy embeddings; this stage never samples."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> None:
        del hidden_states, sampling_metadata
        return None

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        empty = torch.empty(0, dtype=torch.float32)
        sr = torch.tensor(self._sample_rate, dtype=torch.int32)
        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        counts = kwargs.get("seq_token_counts")
        if counts is not None:
            counts = [int(item) for item in counts]
        requests = _split_flat_ids(ids, counts)
        waveforms: list[torch.Tensor] = []
        sample_rates: list[torch.Tensor] = []
        for index, request_ids in enumerate(requests):
            runtime_info = (
                runtime_additional_information[index]
                if runtime_additional_information is not None and index < len(runtime_additional_information)
                else None
            )
            flat = self._payload_codes(request_ids, runtime_info)
            if flat.numel() == 0:
                waveforms.append(empty.to(device=ids.device))
                sample_rates.append(sr)
                continue
            if flat.numel() % self._num_codebooks != 0:
                raise ValueError(
                    "Breeze codec input length must be divisible by num_codebooks: "
                    f"length={flat.numel()}, num_codebooks={self._num_codebooks}"
                )
            if flat.numel() and (
                int(flat.min()) < 0 or int(flat.max()) >= self._codebook_size
            ):
                raise ValueError(
                    f"Breeze codec id outside [0, {self._codebook_size}): "
                    f"min={int(flat.min())}, max={int(flat.max())}"
                )
            frames = flat.numel() // self._num_codebooks
            codes = flat.reshape(self._num_codebooks, frames).unsqueeze(0)
            waveform = self._decode_codes(codes)
            if waveform.ndim == 3:
                waveform = waveform[0, 0]
            elif waveform.ndim == 2:
                waveform = waveform[0]
            if waveform.ndim != 1:
                raise ValueError(f"Breeze Mimi waveform must be 1D, got {tuple(waveform.shape)}")
            # Serving/output serialization consumes CPU tensors.  Move only
            # the final waveform off the worker device; codec inference itself
            # remains on the accelerator.
            waveforms.append(waveform.detach().to(device="cpu", dtype=torch.float32).contiguous())
            sample_rates.append(sr)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": waveforms, "sr": sample_rates},
        )

    def _decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode one ``(1, Q, T)`` sequence using Breeze's bundled codec."""
        if self._audio_tokenizer is not None:
            wavs, sample_rate = self._audio_tokenizer.decode(
                {"audio_codes": [codes[0].transpose(0, 1).contiguous()]}
            )
            self._sample_rate = int(sample_rate)
            if not wavs:
                raise RuntimeError("Breeze Qwen3-TTS tokenizer returned no waveform")
            return torch.as_tensor(wavs[0], device=codes.device, dtype=torch.float32).reshape(-1)

        if self._codec is None:
            raise RuntimeError("Breeze codec is not initialized; load_weights() was not completed")
        decoded = self._codec.decode(codes)
        waveform = getattr(decoded, "audio_values", decoded)
        if isinstance(waveform, Mapping):
            waveform = waveform.get("audio_values")
        if waveform is None:
            raise RuntimeError("MimiModel.decode returned no audio_values")
        return torch.as_tensor(waveform, device=codes.device, dtype=torch.float32)

    @staticmethod
    def _payload_codes(input_ids: torch.Tensor, runtime_info: Mapping[str, Any] | None) -> torch.Tensor:
        if isinstance(runtime_info, Mapping):
            codes = runtime_info.get("codes")
            if isinstance(codes, Mapping):
                audio = codes.get("audio")
                if audio is not None:
                    return torch.as_tensor(audio, device=input_ids.device, dtype=torch.long).reshape(-1)
            # Full-payload transport may flatten nested OmniPayload keys at
            # the wire boundary and restore them only after this hook runs.
            audio = runtime_info.get("codes.audio")
            if audio is not None:
                return torch.as_tensor(audio, device=input_ids.device, dtype=torch.long).reshape(-1)
        return input_ids

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load Breeze's bundled Qwen3-TTS codec, or fallback Mimi weights."""
        params = dict(self._codec.named_parameters()) if self._codec is not None else {}
        loaded: set[str] = set()
        for name, tensor in weights:
            if name.startswith("codec_model."):
                target_name = name[len("codec_model.") :]
            elif name.startswith("model.codec_model."):
                target_name = name[len("model.codec_model.") :]
            else:
                continue
            parameter = params.get(target_name)
            if parameter is None:
                raise ValueError(f"Unknown Breeze Mimi weight: {target_name}")
            loader = getattr(parameter, "weight_loader", None)
            if loader is None:
                parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            else:
                loader(parameter, tensor)
            loaded.add(target_name)

        if loaded:
            self._loaded_local_weights = True
            return {f"_codec.{name}" for name in loaded}

        # Breeze inference uses the bundled Qwen3-TTS tokenizer for both
        # reference encoding and generated-code decoding.  Load it once per
        # Stage-1 worker; unlike the request-side wrapper this instance is
        # placed on the worker device and reused for the entire batch.
        tokenizer_path = Path(self.model_path) / "audio_tokenizer"
        if tokenizer_path.is_dir():
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_tokenizer import (
                Qwen3TTSTokenizer,
            )

            device = self.vllm_config.device_config.device
            self._audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
                str(tokenizer_path),
                device_map=str(device),
            )
            self._codec = None
            self._sample_rate = int(self._audio_tokenizer.model.get_output_sample_rate())
            return set()

        # A few Breeze distributions omit codec tensors and point to the
        # standalone kyutai/mimi repository instead.  Load it once, never per
        # request, and fail loudly if it cannot be resolved.
        codec_name = getattr(self.config, "codec_model_name_or_path", None) or "kyutai/mimi"
        from transformers import AutoModel

        self._codec = AutoModel.from_pretrained(codec_name).eval()
        for parameter in self._codec.parameters():
            parameter.requires_grad_(False)
        self._sample_rate = int(getattr(self._codec.config, "sampling_rate", self._sample_rate))
        self._codebook_size = int(
            getattr(getattr(self._codec.config, "quantizer", None), "cardinality", self._codebook_size)
        )
        return {f"_codec.{name}" for name, _ in self._codec.named_parameters()}


__all__ = ["BreezeTTS2MimiCodec"]
