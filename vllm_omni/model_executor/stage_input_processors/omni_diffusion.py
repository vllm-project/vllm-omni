from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.model_executor.models.omni_diffusion.utils import (
    OMNI_DIFFUSION_TEXT_ADAPTER_TOKEN_IDS_KEY,
    normalize_token_id_sequence,
)

_TEXT_ADAPTER_PROMPT_TOKEN_ID = 0


def _get_multimodal_output(source_output: Any) -> Mapping[str, Any]:
    outputs = getattr(source_output, "outputs", None)
    if not outputs:
        raise ValueError("Omni-Diffusion stage output did not contain a completion output.")

    multimodal_output = getattr(outputs[0], "multimodal_output", None)
    if not isinstance(multimodal_output, Mapping):
        raise TypeError("Omni-Diffusion completion output did not contain a multimodal_output mapping.")
    return multimodal_output


def text_tokens_to_ar_text_adapter(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
    **kwargs: Any,
) -> list[OmniTokensPrompt]:
    """Build AR text-adapter inputs from Omni-Diffusion stage-0 output.

    Stage 0 keeps the official one-shot ``DreamModel.generate`` output shape
    and exposes generated text as ``multimodal_output['text_token_ids']``.
    Stage 1 is a tiny AR adapter: it receives these target text token IDs as
    per-request runtime information, then emits them through vLLM's normal
    sampler path one token at a time. This keeps text serving model-specific
    and avoids special-casing diffusion outputs in the OpenAI/public layer.
    """

    del prompt, requires_multimodal_data, streaming_context, kwargs
    if not source_outputs:
        raise ValueError("Omni-Diffusion text adapter requires one upstream output.")

    mm_output = _get_multimodal_output(source_outputs[0])
    token_ids = normalize_token_id_sequence(
        mm_output.get("text_token_ids"),
        source="Omni-Diffusion upstream text",
    )
    if not token_ids:
        raise ValueError("Omni-Diffusion stage output did not contain text_token_ids.")
    return [
        OmniTokensPrompt(
            prompt_token_ids=[_TEXT_ADAPTER_PROMPT_TOKEN_ID],
            multi_modal_data=None,
            mm_processor_kwargs=None,
            additional_information={OMNI_DIFFUSION_TEXT_ADAPTER_TOKEN_IDS_KEY: token_ids},
        )
    ]
