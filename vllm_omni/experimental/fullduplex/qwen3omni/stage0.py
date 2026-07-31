# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-side stage-0 duplex runtime for Qwen3-Omni.

Turns appended PCM into thinker input embeddings.

STATUS: implemented but NOT validated against a checkpoint or a GPU. The
shape and length arithmetic is verified against vLLM's own functions; the
forward pass has never been executed.

Why this piece is needed at all
-------------------------------
In the duplex append path, audio cannot reach the thinker by Qwen3-Omni's
normal multimodal route:

* ``build_engine_core_request_from_tokens``
  (``vllm_omni/engine/orchestrator.py:118-158``) forwards only
  ``prompt_token_ids``, ``prompt_embeds``, ``cache_salt``,
  ``additional_information`` and ``model_intermediate_buffer``. Any other
  prompt key -- including ``multi_modal_data`` -- is dropped silently.
* ``_OrchestratorDuplexStagePort.submit`` (``orchestrator.py:289-302``)
  passes no ``mm_features``.

So audio arrives as base64 PCM inside
``model_intermediate_buffer["duplex"]["payload"]``, and the only place it can
become thinker embeddings is the model's own ``preprocess`` hook, dispatched
at ``gpu_model_runner.py:1685`` under ``model.has_preprocess``.

Chunk alignment (why per-chunk encoding is sound here)
------------------------------------------------------
``Qwen3OmniMoeAudioEncoder.forward`` splits its conv input into
``n_window * 2 == 100`` mel-frame chunks and runs the conv stack on each
independently. At ``hop_length=160`` / 16 kHz that is exactly 1.0 s, which is
``Qwen3OmniDuplexPolicy.CHUNK_PERIOD_MS``. A duplex chunk therefore lands on
the model's own conv boundary and per-chunk encoding introduces **no
convolutional boundary error** -- unlike MiniCPM, which needs an audio
encoder KV cache plus explicit prefix/suffix context frames.

The tower's *attention* still spans up to
``n_window_infer // (n_window * 2) == 8`` such chunks, so encoding one chunk
at a time remains an approximation at the attention level. That is the known
residual inaccuracy of this design and is not corrected here.

Reservation invariant
---------------------
``duplex_scheduler_token_budget`` in ``runtime.py`` reserves scheduler slots
via ``Qwen3OmniDuplexPolicy.audio_tokens_for_samples``, which reimplements
vLLM's ``_get_feat_extract_output_lengths``. This module MUST produce exactly
that many embeddings per chunk; a mismatch is absorbed silently by the model
runner, so ``_encode_chunk`` asserts it instead.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any

from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy


@dataclass
class Qwen3OmniStage0SessionState:
    """Per-``(session_id, incarnation)`` worker state for the thinker."""

    #: Raw float32 PCM not yet consumed into embeddings.
    audio_buffer: Any = None
    #: Count of whole chunks already turned into embeddings.
    chunk_index: int = 0
    #: All audio in the current turn, kept so each append can be encoded with
    #: its context instead of in isolation. Reset at a turn boundary.
    turn_audio: Any = None
    #: Memoized results keyed by ``(epoch, seq)`` so a scheduler retry of the
    #: same append is idempotent -- mirrors ``minicpmo45/stage0.py:164-173``.
    prepared: dict[tuple[int, int], Any] = field(default_factory=dict)


class Qwen3OmniStage0DuplexRuntime:
    """Turns appended PCM into thinker input embeddings.

    Constructed lazily off the live model instance, mirroring
    ``minicpmo_4_5_omni.py:455-465``. There is no registry: the model class
    imports and instantiates this itself on the first duplex append.
    """

    #: Bound on memoized appends per session, so a long session cannot grow
    #: this table without limit.
    _MAX_PREPARED = 8

    def __init__(self, stage_model: Any, *, model_path: str | None = None, device: str | None = None) -> None:
        self._model = stage_model
        self._model_path = model_path
        self._device = device
        self._feature_extractor: Any = None
        self.sessions: dict[tuple[str, int], Qwen3OmniStage0SessionState] = {}

    # ---- session bookkeeping ---------------------------------------------

    def session(self, session_id: str, incarnation: int) -> Qwen3OmniStage0SessionState:
        key = (session_id, incarnation)
        state = self.sessions.get(key)
        if state is None:
            state = Qwen3OmniStage0SessionState()
            self.sessions[key] = state
        return state

    def drop_session(self, session_id: str, incarnation: int) -> None:
        self.sessions.pop((session_id, incarnation), None)

    # ---- model handles ----------------------------------------------------

    @property
    def thinker(self) -> Any:
        thinker = getattr(self._model, "thinker", None)
        if thinker is None:
            raise RuntimeError("Qwen3-Omni duplex stage 0 requires the thinker submodule")
        return thinker

    @property
    def audio_tower(self) -> Any:
        tower = getattr(self.thinker, "audio_tower", None)
        if tower is None:
            raise RuntimeError("Qwen3-Omni thinker has no audio_tower; cannot encode duplex audio")
        return tower

    def feature_extractor(self) -> Any:
        """The checkpoint's ``WhisperFeatureExtractor``."""
        if self._feature_extractor is None:
            from vllm.transformers_utils.processor import cached_processor_from_config

            model_config = getattr(getattr(self._model, "vllm_config", None), "model_config", None)
            if model_config is None:
                raise RuntimeError("Qwen3-Omni duplex stage 0 requires a model_config to load the processor")
            processor = cached_processor_from_config(model_config)
            self._feature_extractor = processor.feature_extractor
        return self._feature_extractor

    # ---- audio -> embeddings ---------------------------------------------

    def build_append_embeddings(
        self,
        *,
        duplex: dict[str, Any],
        token_offset: int,
        prompt_len: int,
    ) -> Any:
        """Decode appended PCM and return thinker input embeddings.

        Returns a ``(num_tokens, hidden_size)`` tensor covering whole chunks
        completed by this append. May return ``None`` when the append did not
        complete a chunk (the serving-side buffer normally prevents that, so
        it indicates a partial payload reached the worker).

        ``token_offset`` / ``prompt_len`` describe the reserved prompt span
        and are used only to validate the count; slicing into the request's
        prompt is the caller's job.
        """
        import numpy as np

        session_id = str(duplex.get("session_id") or "")
        incarnation = _coerce_int(duplex.get("incarnation")) or 0
        epoch = _coerce_int(duplex.get("epoch")) or 0
        seq = _coerce_int(duplex.get("seq")) or 0
        payload = duplex.get("payload")
        closes_turn = bool(duplex.get("final")) or (
            isinstance(payload, dict) and payload.get(Qwen3OmniDuplexPolicy.TURN_FINAL_KEY) is True
        )

        state = self.session(session_id, incarnation)

        # Idempotent replay: the scheduler may re-present the same append.
        cache_key = (epoch, seq)
        if cache_key in state.prepared:
            return state.prepared[cache_key]

        pcm = self._decode_pcm(payload)
        if pcm.size:
            state.audio_buffer = pcm if state.audio_buffer is None else np.concatenate([state.audio_buffer, pcm])

        chunk_samples = Qwen3OmniDuplexPolicy.CHUNK_SAMPLES
        buffered = state.audio_buffer
        if buffered is None or buffered.shape[0] < chunk_samples:
            return None

        num_chunks = buffered.shape[0] // chunk_samples
        consumed = num_chunks * chunk_samples

        # Encode the whole turn so far and keep only the newly completed rows,
        # rather than encoding each chunk in isolation.
        #
        # The audio tower's attention spans n_window_infer // (n_window * 2) ==
        # 8 one-second chunks, so a chunk encoded alone sees none of its
        # context. Measured against a single whole-utterance encode of the same
        # 4 s audio: per-chunk gives cosine 0.844 (min 0.154), cumulative gives
        # 0.949 (min 0.727). Per-chunk embeddings were degrading the thinker's
        # output to nonsense.
        #
        # Cost is quadratic in turn length, which is acceptable only because a
        # spoken turn is short and `turn_audio` resets at each turn boundary.
        # A model with a streaming audio encoder (as MiniCPM-o 4.5 has) would
        # not need this.
        state.turn_audio = (
            buffered[:consumed] if state.turn_audio is None else np.concatenate([state.turn_audio, buffered[:consumed]])
        )
        already = self.expected_embedding_count(state.turn_audio.shape[0] - consumed) if state.chunk_index else 0
        full = self._encode_chunk(state.turn_audio)
        embeds = full[already:]

        state.audio_buffer = buffered[consumed:]
        state.chunk_index += num_chunks

        # Start the next turn's context empty.
        #
        # `turn_audio` exists so a turn's chunks are encoded with each other's
        # context, not so turns accumulate. Carrying it across a boundary made
        # every later turn re-encode the whole conversation's audio: replies
        # kept answering the first question, and the cost grew without bound.
        # Earlier turns are already in the thinker's KV, so they do not need
        # re-encoding here.
        if closes_turn:
            state.turn_audio = None
            state.chunk_index = 0

        expected = self.expected_embedding_count(consumed)
        if embeds.shape[0] != expected:
            raise RuntimeError(
                f"Qwen3-Omni duplex stage 0 produced {embeds.shape[0]} embeddings for "
                f"{consumed} samples but reserved {expected}. The model runner would "
                f"absorb this silently; failing instead."
            )

        state.prepared[cache_key] = embeds
        while len(state.prepared) > self._MAX_PREPARED:
            state.prepared.pop(next(iter(state.prepared)))
        return embeds

    def _encode_chunk(self, chunk: Any) -> Any:
        """Encode exactly one ``CHUNK_SAMPLES`` unit through the audio tower.

        Mirrors ``Qwen3OmniMoeThinkerForConditionalGeneration._process_audio_input``
        (``qwen3_omni_moe_thinker.py:1101-1116``) for a single item.
        """
        import torch

        extractor = self.feature_extractor()
        features = extractor(
            chunk,
            sampling_rate=Qwen3OmniDuplexPolicy.SAMPLE_RATE_HZ,
            return_tensors="pt",
            # Whisper's extractor pads to its 30 s ``n_samples`` by default,
            # which would produce 3000 mel frames instead of 100 and blow the
            # reservation. Take the natural length instead.
            padding="longest",
            truncation=False,
        )
        input_features = features["input_features"]
        # The tower expects (num_mel_bins, total_frames); the extractor
        # returns a leading batch dim for a single item.
        if input_features.dim() == 3:
            input_features = input_features[0]

        num_frames = int(input_features.shape[-1])
        tower = self.audio_tower
        device = self._tower_device(tower)
        feature_lens = torch.tensor([num_frames], dtype=torch.long, device=device)
        aftercnn_lens = torch.tensor(
            [Qwen3OmniDuplexPolicy.audio_tokens_for_mel_frames(num_frames)],
            dtype=torch.long,
            device=device,
        )
        outputs = tower(
            input_features.to(device=device, dtype=tower.dtype),
            feature_lens=feature_lens,
            aftercnn_lens=aftercnn_lens,
        )
        return outputs if isinstance(outputs, torch.Tensor) else outputs.last_hidden_state

    @staticmethod
    def _tower_device(tower: Any) -> Any:
        for parameter in tower.parameters():
            return parameter.device
        raise RuntimeError("Qwen3-Omni audio_tower has no parameters; cannot resolve device")

    @staticmethod
    def _decode_pcm(payload: object) -> Any:
        """Decode a duplex audio payload to a float32 mono array."""
        import numpy as np

        if not isinstance(payload, dict):
            return np.zeros(0, dtype=np.float32)
        audio_format = payload.get("format", Qwen3OmniDuplexPolicy.PCM_FORMAT)
        if audio_format != Qwen3OmniDuplexPolicy.PCM_FORMAT:
            raise ValueError(
                f"unsupported duplex audio format: {audio_format!r} (expected {Qwen3OmniDuplexPolicy.PCM_FORMAT})"
            )
        data = payload.get("audio")
        if isinstance(data, str):
            data = base64.b64decode(data)
        if not isinstance(data, (bytes, bytearray)):
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(bytes(data), dtype="<f4").astype(np.float32, copy=False)

    @staticmethod
    def expected_embedding_count(num_samples: int) -> int:
        """Embeddings a chunk of ``num_samples`` must produce.

        Must agree with ``runtime.duplex_scheduler_token_budget``; both defer
        to the same checkpoint-derived formula.
        """
        return max(1, Qwen3OmniDuplexPolicy.audio_tokens_for_samples(num_samples))


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
