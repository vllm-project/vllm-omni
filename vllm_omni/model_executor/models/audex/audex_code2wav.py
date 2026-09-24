# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audex code2wav: streaming speech-codec-token → waveform stage.

Wraps the vendored Audex causal speech decoder (see ``speech_decoder/``).
Unlike the flow/vocoder stages of other TTS models, the decoder is natively
stateful and streaming: one ``AudexCausalSpeechDecoderSession`` per request
holds the decode cache, each engine step pushes only the newly received codec
frames, and the session's 4-frame lookahead tail is drained by ``flush()``
when the request finishes. The stage therefore never re-decodes left context
and always returns delta audio.
"""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.audex.speech_decoder import (
    AudexCausalSpeechDecoderConfig,
    AudexCausalSpeechDecoderModel,
    AudexCausalSpeechDecoderSession,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)

# Frames per decoder emit. 5 frames = 100 ms at 50 fps — the granularity the
# official streaming demo uses; small enough for a fast first audible chunk.
_DEFAULT_EMIT_CHUNK_FRAMES = 5


def _meta_int(value: Any, default: int = 0) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item()) if value.numel() else default
    if isinstance(value, list | tuple):
        return _meta_int(value[0], default) if value else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _meta_bool(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.reshape(-1)[0].item()) if value.numel() else False
    if isinstance(value, list | tuple):
        return _meta_bool(value[0]) if value else False
    return bool(value)


def _meta_str(value: Any) -> str | None:
    if isinstance(value, list | tuple):
        return _meta_str(value[0]) if value else None
    if isinstance(value, str):
        return value
    return None


def _codes_from_runtime_info(runtime_info: dict[str, Any] | None) -> list[int]:
    """Extract this step's new codec frame ids from the connector payload."""
    if not isinstance(runtime_info, dict):
        return []
    codes = runtime_info.get("codes")
    if not isinstance(codes, dict):
        return []
    audio = codes.get("audio")
    if isinstance(audio, torch.Tensor):
        return audio.reshape(-1).tolist()
    if isinstance(audio, list | tuple):
        return [int(c) for c in audio]
    return []


class AudexCode2Wav(nn.Module):
    """Stage-1 model for Audex TTS (GenerationModelRunner)."""

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

        # Build the decoder skeleton from config so vLLM's memory profiler
        # sees it at startup; weights load later in load_weights().
        config = AudexCausalSpeechDecoderConfig.from_pretrained(self.model_path)
        self.decoder = AudexCausalSpeechDecoderModel(config)
        self.decoder.eval()
        self._sample_rate = int(config.sample_rate)

        extra_cfg = self._connector_extra_config()
        self._emit_chunk_frames = _meta_int(extra_cfg.get("decoder_emit_chunk_frames"), _DEFAULT_EMIT_CHUNK_FRAMES)
        if self._emit_chunk_frames <= 0:
            raise ValueError(f"decoder_emit_chunk_frames must be positive, got {self._emit_chunk_frames}")

        # One streaming session per in-flight request, keyed by the request id
        # carried in each chunk payload's meta (equal to the stage-1 engine
        # request id, so on_requests_finished can free aborted requests too).
        self._sessions: dict[str, AudexCausalSpeechDecoderSession] = {}

    def _connector_extra_config(self) -> dict[str, Any]:
        connector_cfg = getattr(self.vllm_config.model_config, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra = connector_cfg.get("extra", connector_cfg)
            return extra if isinstance(extra, dict) else {}
        extra = getattr(connector_cfg, "extra", None)
        return extra if isinstance(extra, dict) else {}

    # -------------------- vLLM model interface --------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        # This stage ignores token embeddings; keep a stable dummy for the runner.
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: torch.Tensor | OmniOutput, sampling_metadata: Any = None) -> None:
        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        state_dict = dict(weights)
        self.decoder.load_state_dict(state_dict, strict=True)
        device = self.vllm_config.device_config.device
        self.decoder.to(device=device, dtype=torch.float32)
        return {f"decoder.{name}" for name in state_dict}

    def on_requests_finished(self, finished_req_ids: Iterable[str]) -> None:
        """Free decoder sessions for finished/aborted requests."""
        for req_id in finished_req_ids:
            self._sessions.pop(req_id, None)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        # The runner may unpack the NamedTuple in transit; rebuild it.
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(f"AudexCode2Wav expected OmniOutput, got {type(model_outputs)}")

    # -------------------- decoding --------------------

    def _session_for(self, req_id: str) -> AudexCausalSpeechDecoderSession:
        session = self._sessions.get(req_id)
        if session is None:
            session = self.decoder.create_session(
                chunk_frames=self._emit_chunk_frames,
                return_numpy=False,
            )
            self._sessions[req_id] = session
        return session

    def _decode_request(self, req_id: str, codes: list[int], finished: bool) -> torch.Tensor:
        """Push new frames into the request's session; return this step's delta audio."""
        if finished and not codes and req_id not in self._sessions:
            # Zero codec tokens for the whole request. Do NOT raise: an
            # exception in forward() kills the engine core for every request.
            # Returning empty audio lets the serving layer fail this request.
            logger.error("Audex code2wav: request %s finished with no codec tokens; no audio generated", req_id)
            return torch.empty(0, dtype=torch.float32)
        chunks: list[torch.Tensor] = []
        session = self._session_for(req_id)
        with torch.inference_mode():
            if codes:
                for _sr, chunk in session.push([[int(c)] for c in codes]):
                    chunks.append(chunk.reshape(-1))
            if finished:
                for _sr, chunk in session.flush():
                    chunks.append(chunk.reshape(-1))
        if finished:
            self._sessions.pop(req_id, None)
        if not chunks:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(chunks).to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        runtime_infos = runtime_additional_information or []
        # One output entry per scheduled request, even on steps where a
        # request's chunk payload has not arrived yet.
        seq_token_counts = kwargs.get("seq_token_counts")
        if seq_token_counts is not None:
            num_reqs = len(seq_token_counts)
        elif input_ids is not None and input_ids.numel() > 0 and not runtime_infos:
            num_reqs = 1
        else:
            num_reqs = len(runtime_infos)

        audios: list[torch.Tensor] = []
        sr_tensor = torch.tensor(self._sample_rate, dtype=torch.int32)
        for i in range(num_reqs):
            info = runtime_infos[i] if i < len(runtime_infos) else {}
            info = info if isinstance(info, dict) else {}
            meta = info.get("meta", {})
            meta = meta if isinstance(meta, dict) else {}
            req_id = _meta_str(meta.get("req_id"))
            # async_chunk delivers ``stream_finished`` (the adapter strips
            # ``finished``); the sync full-payload path delivers ``finished``.
            finished = _meta_bool(meta.get("stream_finished")) or _meta_bool(meta.get("finished"))
            codes = _codes_from_runtime_info(info)
            if req_id is None:
                if codes:
                    raise ValueError("Audex code2wav chunk payload is missing meta.req_id")
                audios.append(torch.empty(0, dtype=torch.float32))
                continue
            audios.append(self._decode_request(req_id, codes, finished))

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": [sr_tensor] * num_reqs},
        )
