# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Multimodal data plumbing for Omni's custom, non-HF processors."""

from collections.abc import Mapping
from typing import TypeVar

from transformers import BatchFeature
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseMultiModalProcessor, BaseProcessingInfo

_I = TypeVar("_I", bound=BaseProcessingInfo)


class OmniMultiModalProcessor(BaseMultiModalProcessor[_I]):
    """Share data/passthrough handling, leaving model processing in its owner.

    These processors implement ``_call_hf_processor`` locally, sometimes
    without any HF processor object. Prompt tokens are prepared separately
    in ``apply``; the upstream base still owns caching and prompt updates.
    """

    _OMNI_PROMPT_TEXT_KEY = "_vllm_omni_original_prompt_text"

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        kwargs = dict(hf_processor_mm_kwargs)
        prompt = kwargs.pop(self._OMNI_PROMPT_TEXT_KEY, None)
        if prompt is None:
            prompt = self.dummy_inputs.get_dummy_text(mm_items.get_all_counts())
        valid_items = mm_items.select({key for key, count in mm_items.get_all_counts().items() if count > 0})
        mm_data, passthrough = self._get_hf_mm_data(valid_items)
        result = self._call_hf_processor(str(prompt), dict(mm_data), kwargs, {})
        result.update(passthrough)
        return result
