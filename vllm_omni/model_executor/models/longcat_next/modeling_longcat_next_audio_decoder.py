"""LongCat-Next audio decoder stage (audio codes -> 24 kHz waveform).

Instantiates the checkpoint's remote-code ``LongcatNextAudioTokenizer`` and
runs its decode path: VQ bridger dequantisation -> audio decoder ->
conditional flow matching -> Cosy24k HiFT vocoder. Only the
``model.audio_tokenizer.*`` subtree of the sharded checkpoint is loaded
(shards 7/8/15) plus ``cosy24k_vocoder/hift.pt``.

Chunking mirrors the checkpoint's ``lazy_decode_and_save``: level-0 code
16384 marks a chunk boundary; chunks are decoded independently and
cross-faded with ``wave_concat_overlap`` samples.
"""

import json
import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.output_templates import OmniOutput

from .longcat_next_utils import (
    NUM_CODEBOOKS,
    get_remote_attr,
    load_remote_hf_config,
    load_weight_subtree,
    resolve_checkpoint_relative_path,
)

logger = init_logger(__name__)

_DEFAULT_SAMPLE_RATE = 24000
_DEFAULT_WAVE_CONCAT_OVERLAP = 1200  # generation_config.json audio custom_params


class LongcatNextAudioDecoder(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.prefix = prefix

        self.model_path: str = vllm_config.model_config.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.hf_config = load_remote_hf_config(self.model_path)
        vocoder_cfg = self.hf_config.audio_config.cosy24kvocoder_config
        self.vocoder_weight_path = resolve_checkpoint_relative_path(
            vocoder_cfg.weight_path, self.model_path
        )
        if not os.path.isfile(self.vocoder_weight_path):
            raise FileNotFoundError(
                f"Cosy24k vocoder weights not found at {self.vocoder_weight_path}; "
                "the checkpoint download may be incomplete."
            )

        tokenizer_cls = get_remote_attr(
            self.model_path, "modular_longcat_next_audio", "LongcatNextAudioTokenizer"
        )
        self.audio_tokenizer = tokenizer_cls(self.hf_config)
        self._vocoder = None
        self._weights_loaded = False

        self.sample_rate = _DEFAULT_SAMPLE_RATE
        self.wave_concat_overlap = _DEFAULT_WAVE_CONCAT_OVERLAP
        gen_cfg_path = os.path.join(self.model_path, "generation_config.json")
        if os.path.isfile(gen_cfg_path):
            with open(gen_cfg_path) as f:
                custom = json.load(f).get("audio_generation_config", {}).get("custom_params", {})
            self.sample_rate = int(custom.get("sampling_rate", self.sample_rate))
            self.wave_concat_overlap = int(
                custom.get("wave_concat_overlap", self.wave_concat_overlap)
            )

        self.chunk_end_code = int(self.hf_config.audio_config.vq_config.codebook_sizes[0])

    def _ensure_weights(self) -> None:
        if self._weights_loaded:
            return
        logger.info("Loading model.audio_tokenizer.* weights from %s", self.model_path)
        load_weight_subtree(
            self.audio_tokenizer,
            self.model_path,
            "model.audio_tokenizer",
            dtype=self.dtype,
        )
        self.audio_tokenizer.to(device=self.device, dtype=self.dtype)
        self.audio_tokenizer.eval()

        vocoder_cls = get_remote_attr(self.model_path, "cosy24k_vocoder", "Cosy24kVocoder")
        self._vocoder = vocoder_cls.from_pretrained(self.vocoder_weight_path).to(self.device)
        self._vocoder.eval()
        self._weights_loaded = True

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(
        self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None
    ) -> None:
        return None

    def _split_chunks(self, codes: torch.Tensor) -> list[torch.Tensor]:
        """Split [n, 8] codes into per-chunk tensors at level-0 end markers."""
        if codes.shape[0] == 0:
            return []
        if codes[-1, 0] != self.chunk_end_code:
            codes = F.pad(codes, (0, 0, 0, 1), value=self.chunk_end_code)
        end_positions = [-1] + (codes[:, 0] == self.chunk_end_code).nonzero().view(-1).tolist()
        chunks = []
        for i in range(len(end_positions) - 1):
            start = end_positions[i] + 1
            end = end_positions[i + 1]  # exclude the end-marker row
            if end > start:
                chunks.append(codes[start:end])
        return chunks

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        del input_ids, positions, intermediate_tensors, inputs_embeds

        model_intermediate_buffer = (
            kwargs.get("model_intermediate_buffer")
            or kwargs.get("runtime_additional_information")
            or {}
        )
        if isinstance(model_intermediate_buffer, dict):
            additional_info = next(
                (info for info in model_intermediate_buffer.values() if isinstance(info, dict)),
                {},
            )
        else:
            additional_info = next(
                (info for info in model_intermediate_buffer if isinstance(info, dict)),
                {},
            )
        audio_codes = additional_info.get("audio_token_ids")
        if not audio_codes:
            logger.warning("No audio token IDs provided for audio decoder")
            return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

        codes = torch.as_tensor(audio_codes, dtype=torch.long, device=self.device)
        if codes.ndim == 1:
            codes = codes.reshape(-1, NUM_CODEBOOKS)

        self._ensure_weights()

        waves: list[torch.Tensor] = []
        with torch.inference_mode():
            for chunk in self._split_chunks(codes):
                ret = self.audio_tokenizer.decode(
                    chunk,
                    bridge_length=torch.tensor([chunk.shape[0]], device=self.device),
                )
                mel = ret.flow_matching_mel[0][: ret.flow_matching_mel_lengths[0], :]
                wave = self._vocoder.decode(mel.transpose(0, 1).to(torch.float32).unsqueeze(0))
                waves.append(wave.reshape(1, -1).cpu())

        if not waves:
            logger.warning("Audio decoder produced no valid chunks")
            return OmniOutput(text_hidden_states=None, multimodal_outputs=None)

        # Cross-fade consecutive chunks, mirroring decode_save_concat2.
        overlap = self.wave_concat_overlap
        merged = [waves[0]]
        for wave in waves[1:]:
            prev = merged[-1]
            if prev.shape[1] > overlap and wave.shape[1] > overlap:
                fade_out = torch.linspace(1.0, 0.0, overlap)[None, :]
                fade_in = torch.linspace(0.0, 1.0, overlap)[None, :]
                merged.append(prev[:, -overlap:] * fade_out + wave[:, :overlap] * fade_in)
            merged.append(wave)
        waveform = torch.cat(merged, dim=1) if len(merged) > 1 else merged[0]

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": waveform,
                "sr": torch.tensor([self.sample_rate], dtype=torch.int32),
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # This stage loads its own weight subtree (model.audio_tokenizer.* +
        # cosy24k_vocoder/hift.pt) lazily on first decode; the engine-side
        # loader has nothing to place here.
        consumed = {name for name, _ in weights}
        return consumed | {name for name, _ in self.named_parameters()}
