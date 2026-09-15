# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Live /dev/shm leak check for Qwen3-Omni ``async_chunk`` (stage-0-final).

The default L2 Qwen3-Omni job uses ``ci/qwen3_omni_moe.yaml`` with
``async_chunk: false`` plus ``--no-async-chunk``, so it never exercises the
thinker→talker SharedMemoryConnector path.

A text-only request is tagged ``omni_final_stage_id=0``, so nothing downstream
ever ``get()``s what stage-0 ``put()``s for it. Before #7245 the scheduler still
``put()`` a finished marker on finish and nothing unlinked it, so every such
request left a ``SharedMemoryConnector`` segment plus lockfile in ``/dev/shm``.
Stage-0-final finishes now skip the ``put()`` and the orchestrator reclaims any
undrained segment on request cleanup.

This module starts the production async-chunk deploy and asserts the three
text-only request ids leave no connector lockfile or POSIX segment in
``/dev/shm``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_USE_PD = os.environ.get("VLLM_TEST_PD_MODE", "0") == "1"
_MODEL = os.environ.get("VLLM_OMNI_TEST_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
_SHM_DIR = Path("/dev/shm")
_NUM_TEXT_REQUESTS = 3
_SETTLE_S = 15.0

# CI overlay keeps seq/token limits small; flip async_chunk back on so stage-0
# still uses SharedMemoryConnector. Production yaml is already async_chunk:true.
_ASYNC_CHUNK_DEPLOY = modify_stage_config(
    get_deploy_config_path("ci/qwen3_omni_moe.yaml"),
    updates={"async_chunk": True},
)

test_params = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_ASYNC_CHUNK_DEPLOY,
            use_stage_cli=True,
        ),
        id="async_chunk",
    )
]


def _system_prompt() -> dict:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def _shm_used_mib() -> str:
    try:
        st = os.statvfs(_SHM_DIR)
        used = (st.f_blocks - st.f_bavail) * st.f_frsize
        return f"{used / (1024 * 1024):.1f}MiB"
    except OSError:
        return "unknown"


def _is_entry_for_request(name: str, request_id: str) -> bool:
    """Match a POSIX segment or ``shm_<put_key>_lockfile.lock`` for *request_id*."""
    if name == request_id or name.startswith(f"{request_id}_"):
        return True
    shm_prefix = f"shm_{request_id}"
    return name == shm_prefix or name.startswith(f"{shm_prefix}_")


def _entries_for_request_ids(request_ids: list[str]) -> list[str]:
    if not _SHM_DIR.is_dir():
        pytest.skip("/dev/shm is not available")
    found: list[str] = []
    with os.scandir(_SHM_DIR) as entries:
        for entry in entries:
            if any(_is_entry_for_request(entry.name, request_id) for request_id in request_ids):
                found.append(entry.name)
    return found


def _wait_no_entries_for_request_ids(request_ids: list[str], timeout_s: float = _SETTLE_S) -> list[str]:
    """Reclaim is fire-and-forget; poll until these ids are gone or *timeout_s*."""
    deadline = time.monotonic() + timeout_s
    time.sleep(1.0)
    leftover: list[str] = _entries_for_request_ids(request_ids)
    while leftover:
        if time.monotonic() >= deadline:
            return leftover
        time.sleep(2.0)
        leftover = _entries_for_request_ids(request_ids)
    return leftover


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.skipif(_USE_PD, reason="Temporarily skip PD mode in this test module.")
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_only_async_chunk_does_not_leak_shm(omni_server, online_client) -> None:
    """Stage-0-final (text-only) requests must not leave connector SHM behind."""
    print(f"[shm-leak] before requests: /dev/shm_used={_shm_used_mib()}", flush=True)

    messages = dummy_messages_from_mix_data(
        system_prompt=_system_prompt(),
        content_text="What is the capital of China? Answer in 20 words.",
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }
    responses = online_client.send_omni_request(request_config, request_num=_NUM_TEXT_REQUESTS)
    produced = [resp.text_content for resp in responses if getattr(resp, "text_content", None)]
    assert len(produced) == _NUM_TEXT_REQUESTS, (
        f"Need {_NUM_TEXT_REQUESTS} text completions to exercise stage-0 finish; "
        f"got {len(produced)} non-empty texts. A preprocess/engine miss would "
        f"leave /dev/shm unchanged and hide a leak."
    )
    request_ids = [resp.request_id for resp in responses]
    assert all(request_ids) and len(request_ids) == _NUM_TEXT_REQUESTS, (
        f"Need chat completion ids to assert per-request SHM cleanup; got {request_ids}"
    )

    leftover = _wait_no_entries_for_request_ids(request_ids)
    print(
        f"[shm-leak] after {_NUM_TEXT_REQUESTS} text-only requests: "
        f"/dev/shm_used={_shm_used_mib()} leftover={leftover} request_ids={request_ids}",
        flush=True,
    )
    assert not leftover, (
        f"SharedMemoryConnector leaked /dev/shm entries for {leftover} after stage-0-final request(s) {request_ids}"
    )
