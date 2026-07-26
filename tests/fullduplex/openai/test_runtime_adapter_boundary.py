# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_omni.experimental.fullduplex.openai.runtime_adapter import (
    validate_serving_runtime_adapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "module_name",
    [
        "vllm_omni.experimental.fullduplex.openai.session_runner",
        "vllm_omni.experimental.fullduplex.openai.runtime_bridge",
        "vllm_omni.experimental.fullduplex.openai.serving",
    ],
)
def test_generic_openai_runtime_import_does_not_load_minicpmo45(module_name: str) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = f"""
import sys

import {module_name}

loaded = sorted(
    name
    for name in sys.modules
    if name == "vllm_omni.experimental.fullduplex.minicpmo45"
    or name.startswith("vllm_omni.experimental.fullduplex.minicpmo45.")
)
if loaded:
    raise SystemExit("generic OpenAI runtime loaded MiniCPM-o: " + ", ".join(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_generic_handler_requires_explicit_serving_runtime_adapter() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    script = """
from types import SimpleNamespace

from vllm_omni.experimental.fullduplex.openai.serving import OmniDuplexSessionHandler

chat_service = SimpleNamespace(engine_client=SimpleNamespace())
try:
    OmniDuplexSessionHandler(chat_service=chat_service)
except ValueError as exc:
    if "serving runtime adapter" not in str(exc).lower():
        raise
else:
    raise SystemExit("generic handler silently selected a model serving adapter")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _valid_runtime_adapter() -> SimpleNamespace:
    data_plane = SimpleNamespace(
        begin_request=lambda request_id: None,
        is_terminal=lambda request_id: False,
        mark_terminal=lambda request_id: None,
        close_stream=lambda request_id: None,
        close_session=lambda session_id, **kwargs: None,
        project=lambda result, **kwargs: (),
    )
    return SimpleNamespace(
        adapter_id="test-adapter",
        session_states={},
        data_plane=data_plane,
        clean_response_done_prefix="clean",
        interrupted_tts_prefix="interrupted",
        private_runtime_config_keys=frozenset({"private_key"}),
        create_session_state=lambda: object(),
        session_state=lambda session_id: object(),
        remove_session_state=lambda session_id: None,
        is_enabled=lambda config: True,
        capabilities=lambda **kwargs: object(),
        validate_client_extra_body=lambda extra_body: None,
        prepare_runtime_config=lambda config, **kwargs: {},
        runtime_config_for_update=lambda config, current: {},
        data_plane_context=lambda **kwargs: object(),
    )


@pytest.mark.parametrize(
    ("attribute", "invalid_value"),
    [
        ("session_states", None),
        ("clean_response_done_prefix", None),
        ("interrupted_tts_prefix", None),
        ("private_runtime_config_keys", {"private_key"}),
    ],
)
def test_runtime_adapter_validator_rejects_invalid_protocol_attributes(
    attribute: str,
    invalid_value: object,
) -> None:
    adapter = _valid_runtime_adapter()
    setattr(adapter, attribute, invalid_value)

    with pytest.raises(TypeError, match=attribute):
        validate_serving_runtime_adapter(adapter)


def test_runtime_adapter_validator_rejects_incomplete_data_plane() -> None:
    adapter = _valid_runtime_adapter()
    adapter.data_plane.close_session = None

    with pytest.raises(TypeError, match="data_plane.*close_session"):
        validate_serving_runtime_adapter(adapter)
