# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o 4.5 reliability scenarios. SIGTERM serve-root only, plus partial isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.dfx.reliability.helpers import (
    PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC,
    FaultModel,
    SignalSpec,
    scenario_params,
)
from tests.dfx.reliability.scenarios import (
    PartialStageIsolation,
    ProcessKillServeRootNoLoad,
    ProcessKillServeRootWithLoad,
)
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

_MODEL = "openbmb/MiniCPM-o-4_5"
_SERVE_ROOT_SIGTERM: tuple[SignalSpec, ...] = (("SIGTERM", None),)
_NATIVE_CHAT_TEMPLATE = (
    Path(__file__).resolve().parents[3] / "vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja"
)


def _system_prompt() -> dict[str, Any]:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are MiniCPM-o 4.5, a virtual human capable of perceiving "
                    "auditory and visual inputs, as well as generating text and speech."
                ),
            }
        ],
    }


def _text_request(model: str) -> dict[str, Any]:
    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of China? Answer in one short sentence.",
    )
    return {"model": model, "messages": messages, "stream": False, "modalities": ["text"]}


def _audio_request(model: str) -> dict[str, Any]:
    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of China? Answer in one short sentence.",
    )
    return {
        "model": model,
        "messages": messages,
        "stream": False,
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": True, "enable_thinking": False},
    }


def _assert_text_200(client: Any, model: str) -> None:
    client.send_chat_completions_http_request({"json": _text_request(model), "timeout": 120})


class MiniCPMO45(FaultModel):
    model_id = "minicpmo_4_5"
    res = {"cuda": "H100"}
    num_cards = 1
    extra_marks = (pytest.mark.omni,)
    scenarios = [
        {
            "test_name": "minicpmo_4_5_reliability_default",
            "server_params": {
                "model": _MODEL,
                "stage_config_name": "minicpmo_4_5.yaml",
                "server_args": [
                    "--trust-remote-code",
                    "--async-chunk",
                    "--chat-template",
                    str(_NATIVE_CHAT_TEMPLATE),
                    "--chat-template-content-format",
                    "openai",
                ],
            },
        }
    ]
    deploy_dir = Path(get_deploy_config_path("minicpmo_4_5.yaml")).parent
    worker_pattern = "StageEngineCoreProc_stage0"
    isolation_pattern = "StageEngineCoreProc_stage1"
    with_load_timeout_sec = 300.0
    serve_root_signals = _SERVE_ROOT_SIGTERM
    serve_root_with_load_signals = _SERVE_ROOT_SIGTERM
    hard_oom_config: dict[str, Any] = {}
    fault_error_keywords = ()
    process_kill_error_keywords = ()

    def warmup(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)

    def isolation_warmup(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)

    def isolation_after(self, client: Any, server: Any) -> None:
        _assert_text_200(client, server.model)
        client.send_chat_completions_http_request(
            {"json": _audio_request(server.model), "timeout": PARTIAL_ISOLATION_HTTP_TIMEOUT_SEC},
            err_code=500,
        )


MODEL = MiniCPMO45()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, ProcessKillServeRootNoLoad)
def test_reliability_fault_process_kill_serve_root_no_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootNoLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, ProcessKillServeRootWithLoad)
def test_reliability_fault_process_kill_serve_root_with_load_fast_fail_and_cleanup(
    omni_server_function,
    online_client_function,
    signal_name: str,
) -> None:
    ProcessKillServeRootWithLoad().run(MODEL, omni_server_function, online_client_function, signal_name)


@scenario_params(MODEL, PartialStageIsolation)
def test_reliability_fault_partial_stage_isolation(omni_server_function, online_client_function) -> None:
    """Kill one stage or one worker. Requests bound to it fail, and one SIGTERM still exits the API."""
    PartialStageIsolation().run(MODEL, omni_server_function, online_client_function)
