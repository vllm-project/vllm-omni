"""LongCat-Next audio decoder stage (audio codes -> 24 kHz waveform).

Runs the checkpoint's remote-code LongcatNextAudioTokenizer decode path (VQ
dequant -> audio decoder -> flow matching -> Cosy24k HiFT vocoder), loading
only the model.audio_tokenizer.* subtree plus cosy24k_vocoder/hift.pt.
Chunking mirrors lazy_decode_and_save: level-0 code 16384 marks a chunk
boundary; chunks decode independently and cross-fade at the seams.
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
    resolve_single_request_additional_info,
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
        self.vocoder_weight_path = resolve_checkpoint_relative_path(vocoder_cfg.weight_path, self.model_path)
        if not os.path.isfile(self.vocoder_weight_path):
            raise FileNotFoundError(
                f"Cosy24k vocoder weights not found at {self.vocoder_weight_path}; "
                "the checkpoint download may be incomplete."
            )

        tokenizer_cls = get_remote_attr(self.model_path, "modular_longcat_next_audio", "LongcatNextAudioTokenizer")
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
            self.wave_concat_overlap = int(custom.get("wave_concat_overlap", self.wave_concat_overlap))

        self.codebook_sizes = [int(s) for s in self.hf_config.audio_config.vq_config.codebook_sizes]
        self.chunk_end_code = self.codebook_sizes[0]

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

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def _split_chunks(self, codes: torch.Tensor) -> list[torch.Tensor]:
        """Split [n, 8] codes into per-chunk tensors at level-0 end markers.

        Drops rows with a sentinel code at level >= 1 (the depth head emits
        vq_size+1 classes per level, so non-zero-level sentinels can be
        sampled but would OOB that codebook's embed lookup at decode) --
        only level-0's sentinel is a real chunk boundary.
        """
        if codes.shape[0] == 0:
            return []
        limit = torch.tensor(self.codebook_sizes, device=codes.device, dtype=codes.dtype)
        # A level-0 chunk-end marker row is a split boundary, not a real
        # frame, so its non-zero-level values are never decoded and must
        # survive the sentinel guard below.
        is_boundary = codes[:, 0] == self.chunk_end_code
        oob = codes >= limit
        oob[:, 0] = False  # level-0 chunk-end marker is valid here
        bad = oob.any(dim=1) & ~is_boundary
        if bad.any():
            logger.warning(
                "LongcatNextAudioDecoder dropping %d code row(s) with a "
                "sentinel code at a non-zero level (out-of-range codebook index)",
                int(bad.sum()),
            )
            codes = codes[~bad]
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

        additional_info = resolve_single_request_additional_info(kwargs, "LongcatNextAudioDecoder")
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

        # Cross-fade consecutive chunks: the blended seam replaces both the
        # previous chunk's tail and the next chunk's head (they cover the
        # same audio), trimming each chunk's overlap margin instead of
        # appending full waves, which would replay the seam and stutter.
        overlap = self.wave_concat_overlap
        parts: list[torch.Tensor] = []
        prev = waves[0]
        for wave in waves[1:]:
            if prev.shape[1] > overlap and wave.shape[1] > overlap:
                fade_out = torch.linspace(1.0, 0.0, overlap)[None, :]
                fade_in = torch.linspace(0.0, 1.0, overlap)[None, :]
                parts.append(prev[:, :-overlap])
                parts.append(prev[:, -overlap:] * fade_out + wave[:, :overlap] * fade_in)
                prev = wave[:, overlap:]
            else:
                parts.append(prev)
                prev = wave
        parts.append(prev)
        waveform = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": waveform,
                "sr": torch.tensor([self.sample_rate], dtype=torch.int32),
            },
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Weights are loaded lazily on first decode (_ensure_weights); the
        # engine-side loader has nothing to place here.
        consumed = {name for name, _ in weights}
        return consumed | {name for name, _ in self.named_parameters()}
