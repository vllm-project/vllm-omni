# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Check prompt dispatch without importing model or accelerator dependencies."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def preprocessor(monkeypatch):
    dependencies = {
        "vllm.inputs": dict.fromkeys(["EmbedsInput", "MultiModalInput", "SingletonInput"], dict),
        "vllm.inputs.preprocess": {"InputPreprocessor": object},
        "vllm.logger": {"init_logger": lambda name: None},
        "vllm.renderers.inputs": {"SingletonDictPrompt": dict},
        "vllm_omni.inputs.data": dict.fromkeys(
            ["OmniEmbedsPrompt", "OmniTextPrompt", "OmniTokenInputs", "OmniTokensPrompt", "token_inputs_omni"],
            dict,
        ),
    }
    for name, attrs in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    path = Path(__file__).resolve().parents[2] / "vllm_omni/inputs/preprocess.py"
    spec = importlib.util.spec_from_file_location("_test_omni_input_preprocess", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.OmniInputPreprocessor()


@pytest.mark.parametrize("options", [None, {}, {"truncate_prompt_tokens": 2}])
@pytest.mark.parametrize("multimodal", [False, True])
def test_token_dispatch_preserves_tokenization_kwargs(preprocessor, options, multimodal):
    prompt = {"prompt_token_ids": [1, 2, 3, 4]}
    if multimodal:
        prompt["multi_modal_data"] = {"image": object()}
    expected = object()
    preprocessor._process_tokens = Mock(return_value=expected)

    result = preprocessor._prompt_to_llm_inputs(prompt, tokenization_kwargs=options)

    preprocessor._process_tokens.assert_called_once_with(prompt, tokenization_kwargs=options)
    assert result is expected


def test_token_options_reach_truncation_and_multimodal_processing(preprocessor):
    options = {"truncate_prompt_tokens": 2}
    prompt = {"prompt_token_ids": [1, 2, 3, 4], "multi_modal_data": {"image": object()}}
    # The truncation policy belongs to upstream vLLM. Verify that the real Omni
    # dispatch and token-processing methods pass the options to that boundary.
    preprocessor._truncate_inputs = Mock(return_value=[3, 4])
    preprocessor._process_multimodal = Mock(return_value={"prompt_token_ids": [3, 4]})

    result = preprocessor._prompt_to_llm_inputs(prompt, tokenization_kwargs=options)

    preprocessor._truncate_inputs.assert_called_once_with([1, 2, 3, 4], options)
    preprocessor._process_multimodal.assert_called_once_with(
        [3, 4], prompt["multi_modal_data"], None, tokenization_kwargs=options, mm_uuids=None
    )
    assert result["prompt_token_ids"] == [3, 4]
