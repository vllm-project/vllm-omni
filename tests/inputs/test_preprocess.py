# SPDX-License-Identifier: Apache-2.0
"""Unit tests for OmniInputPreprocessor._process_text routing."""

import pytest
from pytest_mock import MockerFixture

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestProcessTextMmProcessorKwargsRouting:
    """Presence-based routing: an explicit empty ``mm_processor_kwargs`` dict
    on the prompt still routes through ``_process_multimodal``. Required for
    AR-based image generation (e.g. GLM-Image t2i) where the HF processor
    supplies the image-generation scaffold from its own defaults and no extra
    kwargs are needed from the caller.
    """

    @pytest.fixture
    def preprocessor(self, mocker: MockerFixture):
        from vllm_omni.inputs.preprocess import OmniInputPreprocessor

        instance = object.__new__(OmniInputPreprocessor)
        instance._process_multimodal = mocker.MagicMock(return_value={})
        instance._tokenize_prompt = mocker.MagicMock(return_value=[1, 2, 3])
        return instance

    def test_empty_mm_processor_kwargs_routes_to_multimodal(self, preprocessor):
        preprocessor._process_text({"prompt": "hello", "mm_processor_kwargs": {}})
        assert preprocessor._process_multimodal.called
        assert not preprocessor._tokenize_prompt.called

    def test_missing_mm_processor_kwargs_routes_to_tokenize(self, preprocessor):
        preprocessor._process_text({"prompt": "hello"})
        assert preprocessor._tokenize_prompt.called
        assert not preprocessor._process_multimodal.called
