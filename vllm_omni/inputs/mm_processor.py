# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Multimodal data plumbing for Omni's custom, non-HF processors."""

from collections.abc import Mapping
from typing import TypeVar

from transformers import BatchFeature
from vllm.config.multimodal import BaseDummyOptions, MultiModalDummyOptions
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    cached_encode,
)

_I = TypeVar("_I", bound=BaseProcessingInfo)


class OmniDummyInputsBuilder(BaseDummyInputsBuilder[_I]):
    """Restore Omni's processor-input hook on the vLLM 0.31 API."""

    def get_dummy_processor_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> ProcessorInputs:
        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, MultiModalDummyOptions(mm_options or {}))
        dummy_mm_items = self.info.parse_mm_data(dummy_mm_data, validate=False)

        tokenizer = self.info.ctx.tokenizer
        dummy_prompt = [] if tokenizer is None else cached_encode(tokenizer, dummy_text, truncation=False)
        return ProcessorInputs(prompt=dummy_prompt, mm_data_items=dummy_mm_items)


class OmniMultiModalProcessor(BaseMultiModalProcessor[_I]):
    """Share data/passthrough handling, leaving model processing in its owner.

    These processors implement ``_call_hf_processor`` locally, sometimes
    without any HF processor object. Prompt tokens are prepared separately
    in ``apply``; the upstream base still owns caching and prompt updates.
    """

    _OMNI_PROMPT_TEXT_KEY = "_vllm_omni_original_prompt_text"

    def get_dummy_inputs(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: MultiModalDummyOptions,
    ) -> ProcessorInputs:
        if isinstance(self.dummy_inputs, OmniDummyInputsBuilder):
            return self.dummy_inputs.get_dummy_processor_inputs(seq_len, mm_counts, mm_options)
        return super().get_dummy_inputs(seq_len, mm_counts, mm_options)

    def _apply_hf_processor_main(
        self,
        mm_items: MultiModalDataItems,
        hf_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        kwargs = dict(hf_kwargs)
        prompt = kwargs.pop(self._OMNI_PROMPT_TEXT_KEY, None)
        if prompt is None:
            prompt = self.dummy_inputs.get_dummy_text(mm_items.get_all_counts())
        valid_items = mm_items.select({key for key, count in mm_items.get_all_counts().items() if count > 0})
        # Raw processor/passthrough data (upstream removed `_get_hf_mm_data` in
        # favour of `_get_hf_mm_inputs`, which also renames `audios` -> `audio`
        # and injects a dummy text key). Omni's custom `_call_hf_processor`
        # subclasses still expect the un-renamed keys, so extract them here.
        mm_data: dict[str, object] = {}
        passthrough: dict[str, object] = {}
        for items in valid_items.values():
            if not items:
                continue
            mm_data.update(items.get_processor_data())
            passthrough.update(items.get_passthrough_data())
        result = self._call_hf_processor(str(prompt), dict(mm_data), kwargs, {})
        result.update(passthrough)
        return result
