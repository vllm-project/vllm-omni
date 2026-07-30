# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Worker-side stage-0 duplex runtime for Qwen3-Omni.

STATUS: NOT IMPLEMENTED. This module defines the required structure and
documents precisely what is missing. ``build_append_embeddings`` raises
``NotImplementedError``; it has never been executed against a checkpoint.

Why this piece is unavoidable
-----------------------------
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

Two blockers, both real
-----------------------
1. **The thinker has no preprocess hook.**
   ``Qwen3OmniMoeForConditionalGeneration`` sets ``has_preprocess = False``
   at ``qwen3_omni.py:107`` and only enables it for the talker stage
   (``qwen3_omni.py:156``). MiniCPM enables it for both its LM and TTS stages
   (``minicpmo_4_5_omni.py:140``). Enabling it for the Qwen3-Omni thinker and
   implementing ``preprocess`` is core model-code work, outside this package.

2. **Qwen3-Omni's audio tower has no incremental/streaming encode.**
   MiniCPM carries an audio-encoder KV cache across chunks
   (``minicpmo45/stage0.py:443-489``: ``get_audio_embedding_streaming`` plus
   ``audio_past_key_values``, with explicit prefix/suffix context frames).
   Qwen3-Omni exposes no equivalent -- the only streaming affordance in the
   model is ``code2wav.chunked_decode_streaming`` (``qwen3_omni.py:593``),
   which is on the OUTPUT side. Without incremental encode there are three
   options, none free:

   a. Re-encode the whole accumulated buffer each append: correct, but
      quadratic in session length, which defeats the purpose of a persistent
      session.
   b. Encode each chunk in isolation: cheap, but wrong at chunk boundaries,
      because the convolutional front end loses its left context.
   c. Encode a bounded sliding window (new chunk + N frames of left context)
      and keep only the new chunk's embeddings: bounded cost and
      approximately correct. This is the shape MiniCPM achieves via
      ``cnn_redundancy_ms`` / ``prefix_extra_frames``, and it is the
      recommended approach -- but the exact frame arithmetic (mel hop ->
      conv stride -> pooling ratio -> embeddings per chunk) must be derived
      from the checkpoint, not guessed.

Reservation invariant
---------------------
``duplex_scheduler_token_budget`` in ``runtime.py`` reserves scheduler slots
via ``Qwen3OmniDuplexPolicy.audio_tokens_for_samples``, which reimplements
vLLM's ``_get_feat_extract_output_lengths`` (13 tokens per whole second plus
a sub-second remainder, derived from the checkpoint's Whisper feature
extractor at ``hop_length=160``). This module MUST produce exactly that many
embeddings per chunk. If the counts disagree the model runner truncates or
pads without raising, so both sides call the same helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm_omni.experimental.fullduplex.qwen3omni.policy import Qwen3OmniDuplexPolicy


@dataclass
class Qwen3OmniStage0SessionState:
    """Per-``(session_id, incarnation)`` worker state for the thinker."""

    #: Raw float32 PCM not yet consumed into embeddings.
    audio_buffer: Any = None
    #: Count of chunks already turned into embeddings.
    chunk_index: int = 0
    #: Tail of the previous chunk retained as encoder left context (option
    #: (c) above). Length is a function of the audio tower's receptive field.
    left_context: Any = None
    #: Memoized result keyed by (epoch, seq) so a scheduler retry of the same
    #: append is idempotent -- mirrors minicpmo45/stage0.py:164-173.
    prepared: dict[tuple[int, int], Any] = field(default_factory=dict)


class Qwen3OmniStage0DuplexRuntime:
    """Turns appended PCM into thinker input embeddings.

    Constructed lazily off the live model instance, mirroring
    ``minicpmo_4_5_omni.py:455-465``. There is no registry: the model class
    imports and instantiates this itself on the first duplex append.
    """

    def __init__(self, stage_model: Any, *, model_path: str | None = None, device: str | None = None) -> None:
        self._model = stage_model
        self._model_path = model_path
        self._device = device
        self.sessions: dict[tuple[str, int], Qwen3OmniStage0SessionState] = {}

    def session(self, session_id: str, incarnation: int) -> Qwen3OmniStage0SessionState:
        key = (session_id, incarnation)
        state = self.sessions.get(key)
        if state is None:
            state = Qwen3OmniStage0SessionState()
            self.sessions[key] = state
        return state

    def drop_session(self, session_id: str, incarnation: int) -> None:
        self.sessions.pop((session_id, incarnation), None)

    def build_append_embeddings(
        self,
        *,
        duplex: dict[str, Any],
        token_offset: int,
        prompt_len: int,
    ) -> Any:
        """Decode appended PCM and return thinker input embeddings.

        Not implemented. See the module docstring for the two blockers.

        A correct implementation must:

        1. Decode ``duplex["payload"]`` (base64 ``pcm_f32le``) and append it
           to ``state.audio_buffer``.
        2. While a whole ``Qwen3OmniDuplexPolicy.CHUNK_SAMPLES`` unit is
           available, encode ``left_context + chunk`` through the thinker's
           audio tower and keep only the embeddings corresponding to
           ``chunk`` -- discarding the left-context prefix outputs.
        3. Retain a new ``left_context`` tail sized to the tower's receptive
           field.
        4. Assert the produced embedding count equals ``prompt_len`` (the
           reserved slot count). A mismatch must raise here rather than be
           silently absorbed downstream.
        5. Return embeddings positioned at ``token_offset`` within the
           request's prompt.

        The embedding count for step 4 is already settled:
        ``expected_embedding_count`` implements the checkpoint's own conv
        length arithmetic. What remains unknown is the receptive-field width
        needed for ``left_context`` in step 3, which must come from the
        audio tower's conv stack.
        """
        raise NotImplementedError(
            "Qwen3-Omni stage-0 duplex audio embedding is not implemented. "
            "Two prerequisites are unmet: (1) the thinker stage sets "
            "has_preprocess=False (qwen3_omni.py:107) and has no preprocess hook; "
            "(2) Qwen3-Omni exposes no incremental audio encode equivalent to "
            "MiniCPM's get_audio_embedding_streaming/audio_past_key_values. "
            "See this module's docstring and "
            "docs/design/qwen3_omni_duplex_assessment.md."
        )

    @staticmethod
    def expected_embedding_count(num_samples: int) -> int:
        """Embeddings a chunk of ``num_samples`` must produce.

        Must agree with ``runtime.duplex_scheduler_token_budget``; both defer
        to the same checkpoint-derived formula.
        """
        return max(1, Qwen3OmniDuplexPolicy.audio_tokens_for_samples(num_samples))
