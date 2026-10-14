# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex output projection: cumulative Code2Wav audio and inner-monologue text into deltas."""

from __future__ import annotations

from pathlib import Path

from vllm_omni.engine.duplex.contracts import duplex_resource_request_belongs_to_session
from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext, EncodeAudio
from vllm_omni.model_executor.common.duplex.data_plane import CumulativeAudioTextDataPlane
from vllm_omni.model_executor.common.request_outputs import unwrap_request_output
from vllm_omni.model_executor.models.personaplex.duplex.config import SAMPLE_RATE
from vllm_omni.model_executor.models.personaplex.duplex.policy import NON_TEXT_TOKEN_IDS


class PersonaPlexDataPlaneSession(CumulativeAudioTextDataPlane):
    """Project cumulative staged PersonaPlex output into Realtime deltas (24 kHz by default)."""

    default_sample_rate_hz = SAMPLE_RATE

    def __init__(self, encode_audio: EncodeAudio) -> None:
        super().__init__(encode_audio)
        self._text_pieces: tuple[str, ...] | None = None
        self._text_offsets: dict[str, int] = {}

    def load_tokenizer(self, model_path: str) -> None:
        if self._text_pieces is not None:
            return
        import sentencepiece

        tokenizer = sentencepiece.SentencePieceProcessor(str(Path(model_path) / "tokenizer_spm_32k_3.model"))
        # Match the native PersonaPlex text stream, preserving word boundaries
        # between individual tokens and omitting EPAD/PAD frames.
        self._text_pieces = tuple(
            "" if token_id in NON_TEXT_TOKEN_IDS else tokenizer.id_to_piece(token_id).replace("▁", " ")
            for token_id in range(tokenizer.get_piece_size())
        )

    def _project_output(self, output: object, *, context: DuplexDataPlaneContext) -> dict[str, object] | None:
        if getattr(output, "stage_id", None) != 0:
            return super()._project_output(output, context=context)
        output, completion = unwrap_request_output(output)
        request_id = getattr(output, "request_id", None)
        token_ids = getattr(completion, "cumulative_token_ids", ())
        if self._text_pieces is None or not isinstance(request_id, str) or not token_ids:
            return None
        offset = self._text_offsets.get(request_id, 0)
        text = "".join(self._text_pieces[token_id] for token_id in token_ids[offset:])
        self._text_offsets[request_id] = len(token_ids)
        if not text:
            return None
        return {
            "supported": True,
            "stage_role": "talker",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": text,
            "audio_data": "",
            "end_of_turn": False,
        }

    def close_stream(self, request_id: str) -> None:
        super().close_stream(request_id)
        self._text_offsets.pop(request_id, None)

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        super().close_session(session_id, active_request_id=active_request_id)
        for request_id in list(self._text_offsets):
            if request_id == active_request_id or duplex_resource_request_belongs_to_session(request_id, session_id):
                self._text_offsets.pop(request_id, None)


__all__ = ["PersonaPlexDataPlaneSession"]
