# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.stage_input_processors.tts_utils import (
    per_request_initial_chunk_size_override,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _req(entries=None):
    if entries is None:
        return SimpleNamespace(additional_information=None)
    return SimpleNamespace(additional_information=SimpleNamespace(entries=entries))


def test_no_additional_information_keeps_configured_value():
    value, overridden = per_request_initial_chunk_size_override(_req(), 10)
    assert (value, overridden) == (10, False)


def test_missing_additional_information_attribute_keeps_configured_value():
    value, overridden = per_request_initial_chunk_size_override(SimpleNamespace(), 10)
    assert (value, overridden) == (10, False)


def test_entries_missing_key_keeps_configured_value():
    value, overridden = per_request_initial_chunk_size_override(_req(entries={}), 10)
    assert (value, overridden) == (10, False)


def test_list_data_none_keeps_configured_value():
    entry = SimpleNamespace(list_data=None)
    value, overridden = per_request_initial_chunk_size_override(_req(entries={"initial_codec_chunk_frames": entry}), 10)
    assert (value, overridden) == (10, False)


def test_list_data_empty_keeps_configured_value():
    entry = SimpleNamespace(list_data=[])
    value, overridden = per_request_initial_chunk_size_override(_req(entries={"initial_codec_chunk_frames": entry}), 10)
    assert (value, overridden) == (10, False)


def test_list_data_multiple_values_keeps_configured_value():
    entry = SimpleNamespace(list_data=[5, 6])
    value, overridden = per_request_initial_chunk_size_override(_req(entries={"initial_codec_chunk_frames": entry}), 10)
    assert (value, overridden) == (10, False)


def test_single_value_override_takes_precedence():
    entry = SimpleNamespace(list_data=[7])
    value, overridden = per_request_initial_chunk_size_override(_req(entries={"initial_codec_chunk_frames": entry}), 10)
    assert (value, overridden) == (7, True)


def test_override_value_is_coerced_to_int():
    entry = SimpleNamespace(list_data=["7"])
    value, overridden = per_request_initial_chunk_size_override(_req(entries={"initial_codec_chunk_frames": entry}), 10)
    assert (value, overridden) == (7, True)
