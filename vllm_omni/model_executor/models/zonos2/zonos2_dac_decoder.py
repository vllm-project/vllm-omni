# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 Stage 1: real float32 DAC 44.1kHz with request-local OLA."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.zonos2.zonos2_codec import DACStreamDecoder, LocalDAC, eos_boundary
from vllm_omni.model_executor.models.zonos2.zonos2_keys import TARGET_FRAMES


class Zonos2Code2WavForConditionalGeneration(nn.Module):
    input_modalities = "audio"
    requires_request_ids = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.sample_rate, self.hop_length, self.num_codebooks = 44100, 512, 9
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self._dac = LocalDAC(vllm_config.model_config.model, vllm_config.device_config.device)
        self._stream = DACStreamDecoder(self._dac.decode)

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def load_weights(self, weights: Any) -> set[str]:
        # Stage 0 safetensors are unrelated to DAC; local codec loads on first
        # real decode. Do not consume the large talker weights again.
        return set()

    def get_dummy_runtime_additional_information(self, num_reqs):
        return [
            {
                "codes": {"audio": torch.zeros((9, 24), dtype=torch.long)},
                TARGET_FRAMES: 16,
                "meta": {"finished": True},
            }
            for _ in range(num_reqs)
        ]

    def on_requests_finished(self, request_ids):
        self._stream.cleanup(request_ids)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        ids = input_ids.reshape(-1) if input_ids is not None else torch.empty(0, dtype=torch.long)
        counts = kwargs.get("seq_token_counts", [len(ids)])
        request_ids = kwargs.get("request_ids")
        wavs = []
        offset = 0
        for index, count in enumerate(counts):
            chunk = ids[offset : offset + int(count)]
            offset += int(count)
            runtime_info = runtime_additional_information or []
            info = runtime_info[index] if index < len(runtime_info) else {}
            meta = info.get("meta", {})
            codes = info.get("codes", {}).get("audio")
            if isinstance(codes, torch.Tensor):
                if codes.ndim != 2 or codes.shape[0] != 9 or codes.dtype not in (torch.int32, torch.int64):
                    raise ValueError("ZONOS2 stage payload requires [9,T] raw codes")
                frames = codes.T.to(device="cpu", dtype=torch.long)
                final = bool(meta.get("last_chunk", meta.get("finished", True)))
                target = info.get(TARGET_FRAMES, meta.get("num_processed_tokens"))
                if target is None:
                    eos = eos_boundary(frames)
                    target = len(frames) if final else max(0, len(frames) - 8)
                    if eos is not None:
                        target = min(target, eos)
                if not final and not request_ids:
                    raise ValueError("ZONOS2 streaming DAC requires runner request IDs")
                key = str(request_ids[index]) if request_ids else f"sync-{index}"
                wav = self._stream.push(
                    key, frames, final=final, target=int(target), sequence=int(meta.get("chunk_seq", 0))
                )
                if not request_ids:
                    self._stream.cleanup([key])
            else:
                # Runtime profiling and direct DAC component inputs are
                # codebook-major, already aligned codes.
                if len(chunk) % 9:
                    raise ValueError("ZONOS2 DAC flat input length must be divisible by 9")
                else:
                    wav = self._dac.decode(chunk.reshape(9, -1))
            wavs.append(wav)
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": wavs,
                "sr": [torch.tensor(44100, dtype=torch.int32)] * len(wavs),
            },
        )
