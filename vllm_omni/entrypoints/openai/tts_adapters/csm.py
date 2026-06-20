# SPDX-License-Identifier: Apache-2.0
"""CSM-1B (Sesame) serving adapter (#4330 TTS adapter framework).

CSM-1B is a plain text -> speech model with speaker-id conditioning, served on
the AR engine_client path. The OpenAI ``voice`` field maps to a CSM speaker id
(a non-negative integer string, default ``"0"``); there is no reference-audio
voice cloning on this path. The adapter owns request validation and Stage-0
backbone prompt construction for the two-stage CSM pipeline (backbone AR +
inline 31-step depth -> Mimi code2wav; see ``model_executor/models/csm``).

Drop-in for vllm_omni/entrypoints/openai/tts_adapters/csm.py and add ``csm`` to
the import list in that package's __init__.py.
"""

from typing import TYPE_CHECKING, Any

from vllm.inputs import tokens_input
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
    conditioning_cache_salt,
)

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

logger = init_logger(__name__)

# Request bounds (mirror serving_speech._TTS_MAX_NEW_TOKENS_{MIN,MAX}).
_TTS_MAX_NEW_TOKENS_MIN = 1
_TTS_MAX_NEW_TOKENS_MAX = 4096
# CSM-1B's inline-depth AR loop has no reliable natural EOS (greedy hits a
# non-terminating saturating attractor; do_sample EOS is bi-modal). Bound served
# audio with the cap the offline 2-stage harness validated its WAVs at
# (64 frames == 5.12 s at 80 ms/frame). The backbone forces the frame EOS at the
# cap (csm_backbone _DEFAULT_MAX_FRAMES / GATE-B).
_DEFAULT_CSM_MAX_FRAMES = 64


@register_tts_adapter
class CsmTTSAdapter(ARTTSAdapter):
    """Adapter for Sesame CSM-1B (AR ``engine_client`` backend)."""

    stage_keys = frozenset({"csm"})
    name = "csm"

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        self._tokenizer = None

    def _get_tokenizer(self):
        """Lazily load + cache the CSM Llama text tokenizer.

        CSM-1B ships a standard Llama tokenizer plus ``<|begin_of_text|>`` /
        ``<|end_of_text|>``. We tokenize the speaker-tagged text here so the
        prompt carries the exact ``prompt_token_ids`` the Stage-0 backbone embeds
        in ``_embed_text_prompt``.
        """
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            model_name = self.ctx.engine_client.model_config.model
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return self._tokenizer

    def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.voice is not None and not str(request.voice).strip().isdigit():
            return "CSM 'voice' must be a non-negative integer speaker id (e.g. '0')"
        if request.max_new_tokens is not None:
            if request.max_new_tokens < _TTS_MAX_NEW_TOKENS_MIN:
                return f"max_new_tokens must be at least {_TTS_MAX_NEW_TOKENS_MIN}"
            if request.max_new_tokens > _TTS_MAX_NEW_TOKENS_MAX:
                return f"max_new_tokens cannot exceed {_TTS_MAX_NEW_TOKENS_MAX}"
        return None

    async def build(
        self,
        request: "OpenAICreateSpeechRequest",
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        """Build the Stage-0 backbone prompt for a CSM text -> speech request.

        CSM's text conditioning is ``<|begin_of_text|>[<spk>]<text><|end_of_text|>``.
        The backbone reads the ids back out of ``additional_information`` via
        ``_pick`` (csm_backbone), which unwraps the batch-of-1 convention by
        returning ``val[0]`` for list values, so every field is LIST-WRAPPED (a
        bare list would make ``_pick`` return only the first id, embedding one BOS
        token and zero-padding the rest: the 1-2-frame truncated-audio bug). Stage
        0 runs greedy (temperature 0, top_k 0) for deterministic validated output,
        and an explicit ``max_new_frames`` cap bounds the non-terminating AR loop.
        """
        tokenizer = self._get_tokenizer()
        speaker = str(request.voice).strip() if request.voice is not None else "0"
        text = f"<|begin_of_text|>[{speaker}]{request.input}<|end_of_text|>"
        prompt_token_ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])

        max_frames = request.max_new_tokens if request.max_new_tokens is not None else _DEFAULT_CSM_MAX_FRAMES
        additional_information: dict[str, Any] = {
            "prompt_token_ids": [prompt_token_ids],
            "temperature": [0.0],
            "top_k": [0],
            "max_new_frames": [max_frames],
        }

        prompt = tokens_input(prompt_token_ids=prompt_token_ids)
        prompt["additional_information"] = additional_information
        prompt["cache_salt"] = conditioning_cache_salt(request, additional_information)
        return PreparedRequest(prompt=prompt, tts_params={}, model_type="csm")
