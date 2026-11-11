# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for loading a model-level chat_template.json."""

import json

import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai.chat_template import _load_model_chat_template_json
from vllm_omni.transformers_utils.repo_utils import hf_api

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def _write_template_file(model_dir, content: str) -> None:
    (model_dir / "chat_template.json").write_text(content, encoding="utf-8")


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"chat_template": TEMPLATE}, id="dict"),
        pytest.param(TEMPLATE, id="string"),
    ],
)
def test_valid_payload_returns_template(tmp_path, payload):
    _write_template_file(tmp_path, json.dumps(payload))

    assert _load_model_chat_template_json(str(tmp_path)) == TEMPLATE


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"template": TEMPLATE}, id="dict-without-key"),
        pytest.param({"chat_template": None}, id="dict-null-template"),
        pytest.param({"chat_template": ["not", "a", "string"]}, id="dict-list-template"),
        pytest.param({"chat_template": "   "}, id="dict-blank-template"),
        pytest.param("", id="empty-string"),
        pytest.param([TEMPLATE], id="list"),
        pytest.param(42, id="number"),
    ],
)
def test_malformed_payload_returns_none(tmp_path, payload):
    _write_template_file(tmp_path, json.dumps(payload))

    assert _load_model_chat_template_json(str(tmp_path)) is None


def test_invalid_json_returns_none(tmp_path):
    _write_template_file(tmp_path, '{"chat_template": ')

    assert _load_model_chat_template_json(str(tmp_path)) is None


def test_missing_local_file_and_cache_miss_returns_none(tmp_path, mocker: MockerFixture):
    download = mocker.patch.object(hf_api(), "hf_hub_download", side_effect=FileNotFoundError)

    assert _load_model_chat_template_json(str(tmp_path)) is None
    download.assert_called_once()


def test_missing_local_file_falls_back_to_local_hf_cache(tmp_path, mocker: MockerFixture):
    cached_file = tmp_path / "cache" / "chat_template.json"
    cached_file.parent.mkdir()
    cached_file.write_text(json.dumps({"chat_template": TEMPLATE}), encoding="utf-8")
    download = mocker.patch.object(hf_api(), "hf_hub_download", return_value=str(cached_file))

    assert _load_model_chat_template_json("org/model") == TEMPLATE
    download.assert_called_once_with(repo_id="org/model", filename="chat_template.json", local_files_only=True)


def test_local_file_is_preferred_over_hf_cache(tmp_path, mocker: MockerFixture):
    _write_template_file(tmp_path, json.dumps({"chat_template": TEMPLATE}))
    download = mocker.patch.object(hf_api(), "hf_hub_download")

    assert _load_model_chat_template_json(str(tmp_path)) == TEMPLATE
    download.assert_not_called()
