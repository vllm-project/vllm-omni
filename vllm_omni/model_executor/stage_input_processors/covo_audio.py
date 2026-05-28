# Copyright 2026 Tencent.
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.covo_audio.config_covo_audio import COVO_AUDIO_TOKEN_INDEX

logger = init_logger(__name__)

# Per-model REPLACE-keys for the full-payload accumulator (none for covo_audio:
# the producer side does not emit per-step hidden_states / model_outputs;
# llm2code2wav_full_payload reads token_ids directly from `request`).
_FULL_PAYLOAD_REPLACE_KEYS: frozenset[str] = frozenset()


def _filter_audio_codes(token_ids: list[int]) -> list[int]:
    """Filter codec-range token ids and rebase by COVO_AUDIO_TOKEN_INDEX."""
    audio_codes = [t - COVO_AUDIO_TOKEN_INDEX for t in token_ids if t >= COVO_AUDIO_TOKEN_INDEX]
    if not audio_codes:
        audio_codes = [-1]
    return audio_codes


def llm2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Legacy orchestrator-path builder (retained for async_chunk + back-compat).

    The non-async-chunk path now goes through ``llm2code2wav_token_only`` +
    worker connector + ``llm2code2wav_full_payload``.
    """
    talker_outputs = source_outputs
    code2wav_inputs = []

    for talker_output in talker_outputs:
        output = talker_output.outputs[0]
        audio_codes = _filter_audio_codes(list(output.token_ids))
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=audio_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )

    return code2wav_inputs


def llm2code2wav_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
) -> list[OmniTokensPrompt]:
    """Sync-side placeholder for the non-async-chunk Stage-1 input.

    Returns an OmniTokensPrompt sized to the code2wav stage's expected
    prefill length (one slot per audio code).  The actual codec ids are
    delivered via the worker connector payload built by
    ``llm2code2wav_full_payload``.
    """
    code2wav_inputs: list[OmniTokensPrompt] = []
    for output_wrapper in source_outputs:
        output = output_wrapper.outputs[0]
        audio_codes = _filter_audio_codes(list(output.token_ids))
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * len(audio_codes),
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def llm2code2wav_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Producer-side payload builder for the worker connector data plane.

    covo_audio's fused_thinker_talker stage emits codec ids via
    ``request.output_token_ids`` (token-ids only -- no
    hidden_states or embed tensors), so the connector payload is
    just the filtered audio codes plus a finished marker.
    """
    output_token_ids = list(getattr(request, "output_token_ids", None) or [])
    if not output_token_ids:
        logger.warning(
            "covo_audio.llm2code2wav_full_payload: empty output_token_ids for req=%s; consumer wait gate may hang.",
            getattr(request, "request_id", "?"),
        )
        return None
    audio_codes = _filter_audio_codes(output_token_ids)
    return {
        "codes": {"audio": audio_codes},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }
