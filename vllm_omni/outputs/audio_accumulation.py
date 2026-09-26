# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local audio assembly for sentence-final alignment."""

from copy import copy
from dataclasses import dataclass, field
from typing import cast

import torch
from vllm.outputs import RequestOutput

from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput, MultimodalPayload

DEFAULT_AUDIO_SAMPLE_RATE = 24000


@dataclass
class AudioChunkBuffer:
    """Keep downstream full audio separate from client-facing DELTA outputs."""

    chunks: list[torch.Tensor] = field(default_factory=list)
    sample_rate: int | None = None

    def append(self, output: RequestOutput, *, finished: bool) -> RequestOutput:
        completions = getattr(output, "outputs", [])
        if not completions:
            return output
        if len(completions) != 1:
            raise ValueError("Sentence-final audio alignment requires one completion")
        mm = getattr(completions[0], "multimodal_output", None) or {}
        audio = mm.get("audio")
        if audio is None:
            audio = mm.get("model_outputs")
        if audio is not None:
            parts = audio if isinstance(audio, list) else [audio]
            for part in parts:
                tensor = torch.as_tensor(part).detach().cpu().flatten()
                if tensor.numel():
                    # Own the waveform: the frontend retains the original output
                    # while this buffer lives until the sentence-final handoff.
                    self.chunks.append(tensor.clone())
        rate = mm.get("sr")
        if isinstance(rate, list):
            rate = rate[-1] if rate else None
        if rate is not None:
            rate_tensor = torch.as_tensor(rate).flatten()
            rate = int(rate_tensor[-1]) if rate_tensor.numel() else None
        if rate is not None:
            if rate <= 0:
                raise ValueError("Audio sample rate must be positive")
            if self.sample_rate is not None and self.sample_rate != rate:
                raise ValueError("Audio sample rate changed within one alignment request")
            self.sample_rate = rate
        if not finished or not self.chunks:
            return output

        # The original output may already be queued for the frontend. Never
        # replace its delta with cumulative audio, or the client hears repeats.
        result = copy(output)
        result.outputs = [copy(completions[0])]
        cast(MultimodalCompletionOutput, result.outputs[0]).multimodal_output = MultimodalPayload.from_dict(
            {
                **mm,
                "audio": torch.cat(self.chunks),
                "sr": self.sample_rate or DEFAULT_AUDIO_SAMPLE_RATE,
            }
        )
        self.chunks.clear()
        self.sample_rate = None
        return result
