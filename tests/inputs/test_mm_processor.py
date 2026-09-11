# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import BatchFeature

from vllm_omni.inputs.mm_processor import OmniMultiModalProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("counts", [{}, {"audio": 1, "image": 0}])
@pytest.mark.parametrize("original_text", [None, "", "voice cloning text"])
def test_custom_processor_bridge_preserves_prompt_kwargs_and_passthrough(counts, original_text):
    class Processor(OmniMultiModalProcessor):
        def _get_mm_fields_config(self, *args):
            return {}

        def _get_prompt_updates(self, *args):
            return []

    processor = object.__new__(Processor)
    processor.dummy_inputs = SimpleNamespace(get_dummy_text=Mock(return_value="dummy"))
    processor._get_hf_mm_data = Mock(return_value=({"audios": ["wave"]}, {"embedding": "preserve"}))
    processor._call_hf_processor = Mock(return_value=BatchFeature({"features": "processed"}))
    items = Mock()
    items.get_all_counts.return_value = counts
    kwargs = {"sampling_rate": 16000}
    if original_text is not None:
        kwargs[processor._OMNI_PROMPT_TEXT_KEY] = original_text
    before = dict(kwargs)
    result = processor._apply_hf_processor_main(items, kwargs)
    items.select.assert_called_once_with({k for k, v in counts.items() if v > 0})
    processor._call_hf_processor.assert_called_once_with(
        "dummy" if original_text is None else original_text,
        {"audios": ["wave"]},
        {"sampling_rate": 16000},
        {},
    )
    assert dict(result) == {"features": "processed", "embedding": "preserve"}
    assert kwargs == before
