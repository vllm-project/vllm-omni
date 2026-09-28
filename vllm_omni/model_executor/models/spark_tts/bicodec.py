# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full-utterance Spark BiCodec stage managed by Omni's generation runner.

The Apache-2.0 Spark-TTS source supplies the BiCodec module definitions. It is
imported in-process; no Spark server or Hugging Face AR generator is used.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


def _build_codec(model_path: Path) -> nn.Module:
    try:
        from sparktts.models.bicodec import BiCodec
        from sparktts.modules.encoder_decoder.feat_decoder import Decoder
        from sparktts.modules.encoder_decoder.feat_encoder import Encoder
        from sparktts.modules.encoder_decoder.wave_generator import WaveGenerator
        from sparktts.modules.speaker.speaker_encoder import SpeakerEncoder
        from sparktts.modules.vq.factorized_vector_quantize import FactorizedVectorQuantize
        from sparktts.utils.file import load_config
    except ImportError as exc:
        raise ImportError("Spark-TTS's Apache-2.0 sparktts package must be importable in the Omni environment") from exc

    cfg = load_config(str(model_path / "config.yaml"))["audio_tokenizer"]
    return BiCodec(
        mel_params=cfg["mel_params"],
        encoder=Encoder(**cfg["encoder"]),
        decoder=WaveGenerator(**cfg["decoder"]),
        quantizer=FactorizedVectorQuantize(**cfg["quantizer"]),
        speaker_encoder=SpeakerEncoder(**cfg["speaker_encoder"]),
        prenet=Decoder(**cfg["prenet"]),
        postnet=Decoder(**cfg["postnet"]),
    ).eval()


def _codes(info: dict[str, Any], key: str) -> torch.Tensor:
    nested = info.get("codes")
    value = nested.get(key) if isinstance(nested, dict) else None
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).to(dtype=torch.long)
    if isinstance(value, list | tuple):
        return torch.as_tensor(value, dtype=torch.long).reshape(-1)
    return torch.empty(0, dtype=torch.long)


class SparkBiCodecDecoder(nn.Module):
    """Decode one complete Spark semantic/global code pair per request."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.codec = _build_codec(Path(vllm_config.model_config.model))
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self._sample_rate = 16000

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state = dict(weights)
        missing, unexpected = self.codec.load_state_dict(state, strict=False)
        expected_missing = {"mel_transformer.spectrogram.window", "mel_transformer.mel_scale.fb"}
        if unexpected or set(missing) - expected_missing:
            raise RuntimeError(f"Spark BiCodec weights mismatch: missing={missing}, unexpected={unexpected}")
        self.codec.remove_weight_norm()
        self.codec.to(device=self.vllm_config.device_config.device, dtype=torch.float32)
        # remove_weight_norm replaces weight_g/weight_v with plain weight.
        # Report the post-conversion parameter names expected by vLLM's loader.
        return {name for name, _ in self.named_parameters()}

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"SparkBiCodecDecoder expected OmniOutput, got {type(model_outputs)}")

    def _decode(self, info: dict[str, Any]) -> torch.Tensor:
        semantic = _codes(info, "audio")
        global_codes = _codes(info, "ref")
        if semantic.numel() == 0 or global_codes.numel() == 0:
            logger.error(
                "Spark BiCodec received empty semantic/global codes (semantic=%d global=%d keys=%s)",
                semantic.numel(),
                global_codes.numel(),
                list(info),
            )
            return torch.empty(0, dtype=torch.float32)
        if (
            torch.any(semantic < 0)
            or torch.any(semantic >= 8192)
            or torch.any(global_codes < 0)
            or torch.any(global_codes >= 4096)
        ):
            logger.error("Spark BiCodec received out-of-range codes")
            return torch.empty(0, dtype=torch.float32)
        device = self.vllm_config.device_config.device
        with torch.inference_mode():
            wav = self.codec.detokenize(
                semantic.to(device).unsqueeze(0), global_codes.to(device).unsqueeze(0).unsqueeze(0)
            )
            audio = wav.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(audio).all():
            logger.error("Spark BiCodec generated non-finite audio")
            return torch.empty(0, dtype=torch.float32)
        return audio

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        infos = runtime_additional_information or model_intermediate_buffer or []
        counts = kwargs.get("seq_token_counts")
        count = len(counts) if counts is not None else max(1, len(infos))
        audios = [
            self._decode(infos[i])
            if i < len(infos) and isinstance(infos[i], dict)
            else torch.empty(0, dtype=torch.float32)
            for i in range(count)
        ]
        rate = torch.tensor(self._sample_rate, dtype=torch.int32)
        return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": audios, "sr": [rate] * count})
