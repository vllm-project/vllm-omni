# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sesame CSM stage 1: decodes audio codes to a 24 kHz waveform with the Mimi codec."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
from transformers import MimiModel
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.model_executor.models.csm.csm_mimi import (
    CODEC_PREFIX,
    CsmMimiStreamingDecoder,
    MimiDecoderState,
    rename_codec_weights,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

_DUMMY_PROFILE_KEY = "csm_dummy_profile"


def _split_by_request(ids: torch.Tensor, seq_token_counts: Sequence[int] | None) -> list[torch.Tensor]:
    if seq_token_counts is None or len(seq_token_counts) <= 1:
        return [ids]
    segments, start = [], 0
    for count in seq_token_counts:
        segments.append(ids[start : start + int(count)])
        start += int(count)
    return segments


def _payload_codes(info: Any) -> torch.Tensor | None:
    """Codes delivered through the connector, which take precedence over placeholder token ids."""
    codes = info.get("codes") if isinstance(info, Mapping) else None
    audio = codes.get("audio") if isinstance(codes, Mapping) else None
    if isinstance(audio, torch.Tensor) and audio.numel() > 0:
        return audio
    if isinstance(audio, (list, tuple)) and audio:
        return torch.as_tensor(audio)
    return None


class CsmCode2Wav(nn.Module):
    """Stage-1 model for ``sesame/csm-1b``: incremental Mimi decoding under the generation runner."""

    input_modalities = "audio"
    requires_exact_input_shape = True
    # Decoder state is kept per scheduler request id.
    requires_request_ids = True
    # The repository also ships the same weights in the original format (model.safetensors, ckpt.pt).
    allow_patterns_overrides = ["transformers-*.safetensors"]
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        codec_config = config.codec_config
        self.num_codebooks = int(config.num_codebooks)
        self.sample_rate = int(codec_config.sampling_rate)
        self.samples_per_frame = round(codec_config.sampling_rate / codec_config.frame_rate)
        self.codec_vocab_size = int(codec_config.codebook_size)
        # The codec runs in float32 whatever dtype the stage is configured with.
        self.codec_model = MimiModel(codec_config).to(torch.float32)
        self._decoder = CsmMimiStreamingDecoder(self.codec_model)
        self._streams: dict[str, MimiDecoderState] = {}
        self._failed: set[str] = set()

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Return placeholder embeddings; this stage does not embed tokens."""
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        """Return None; this stage does not sample."""
        return None

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        """Mark profiling rows, whose token ids are not audio codes."""
        return [{"meta": {_DUMMY_PROFILE_KEY: True}} for _ in range(num_reqs)]

    def _validate_codes(self, flat: torch.Tensor) -> None:
        """Reject codes Mimi has no embedding for; CSM's codebook vocabulary is larger than Mimi's."""
        if flat.numel() % self.num_codebooks != 0:
            raise ValueError(f"{flat.numel()} codes is not a multiple of {self.num_codebooks} codebooks")
        low, high = int(flat.min()), int(flat.max())
        if low < 0 or high >= self.codec_vocab_size:
            raise ValueError(
                f"audio codes out of range: min={low}, max={high}; expected [0, {self.codec_vocab_size - 1}]"
            )

    @torch.no_grad()
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
        """Decode each request's new frames and return only their audio (delta).

        A request's ``input_ids`` are codebook-major flat codes,
        ``[num_codebooks * frames]``, continuing where its previous call
        stopped; the decoder state is kept per request id, so every frame
        must arrive exactly once. A request whose codes are invalid produces
        no audio from then on and carries ``meta.error`` set, which the serving
        adapter turns into a failed response; the other requests of the batch
        are unaffected.
        """
        sample_rate = torch.tensor(self.sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)
        if input_ids is None or input_ids.numel() == 0:
            return self._output([empty], [False], sample_rate)

        infos = runtime_additional_information or model_intermediate_buffer or []
        request_ids = kwargs.get("request_ids") or []
        segments = _split_by_request(input_ids.reshape(-1), kwargs.get("seq_token_counts"))
        audios = [empty] * len(segments)
        failed = [False] * len(segments)
        for index, segment in enumerate(segments):
            info = infos[index] if index < len(infos) else None
            meta = info.get("meta") if isinstance(info, Mapping) else None
            meta = meta if isinstance(meta, Mapping) else {}
            request_id = request_ids[index] if index < len(request_ids) else None
            if meta.get(_DUMMY_PROFILE_KEY):
                continue
            if request_id in self._failed:
                failed[index] = True
                continue
            payload = _payload_codes(info)
            flat = (segment if payload is None else payload).reshape(-1)
            if flat.numel() == 0:
                continue
            try:
                # Checked where the codes arrived, before they are moved to the codec's device.
                self._validate_codes(flat)
            except ValueError as error:
                logger.warning("CSM code2wav: request %s produces no more audio (%s)", request_id, error)
                failed[index] = True
                if request_id is not None:
                    self._failed.add(request_id)
                    self._streams.pop(request_id, None)
                continue

            codes = flat.to(device=segment.device, dtype=torch.long).reshape(1, self.num_codebooks, -1)
            state = self._streams.get(request_id) if request_id is not None else None
            if state is None:
                state = MimiDecoderState()
                if request_id is not None:
                    self._streams[request_id] = state
            audios[index] = self._decoder.decode(codes, state).to(torch.float32)

        return self._output(audios, failed, sample_rate)

    @staticmethod
    def _output(audios: list[torch.Tensor], failed: list[bool], sample_rate: torch.Tensor) -> OmniOutput:
        """One audio, sample rate and failure flag per request, in batch order."""
        errors = [torch.tensor([int(flag)], dtype=torch.int32) for flag in failed]
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": [sample_rate] * len(audios), "meta": {"error": errors}},
        )

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Release the decoder state of finished or aborted requests."""
        for request_id in finished_req_ids:
            self._streams.pop(request_id, None)
            self._failed.discard(request_id)

    def make_omni_output(self, model_outputs: Any, **_: Any) -> OmniOutput:
        """Return ``forward``'s output unchanged."""
        if not isinstance(model_outputs, OmniOutput):
            raise TypeError(f"CsmCode2Wav.forward returns OmniOutput, got {type(model_outputs).__name__}")
        return model_outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the codec from a CSM checkpoint, whichever Mimi layout it and transformers use."""
        renamed = rename_codec_weights(weights, self.codec_model)
        return AutoWeightsLoader(self).load_weights(item for item in renamed if item[0].startswith(CODEC_PREFIX))
