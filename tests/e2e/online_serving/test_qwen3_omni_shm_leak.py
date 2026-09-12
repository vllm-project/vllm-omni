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

This module starts the production async-chunk deploy and asserts neither the
connector lockfiles (``shm_<key>_lockfile.lock``) nor the POSIX segments
(``<request_id>_<stage>_<chunk>``) grow after those requests.
"""

from __future__ import annotations

import os
import re
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


# ``SharedMemoryConnector.put()`` writes two entries per ``put_key``
# (``<external_req_id>_<stage_id>_<chunk_id>``, see chunk_transfer_adapter):
# the flock file ``shm_<put_key>_lockfile.lock`` and the POSIX segment named
# ``<put_key>`` itself. ``get()`` removes the lockfile and unlinks the segment on
# separate paths, so count both: a regression that drops one but not the other
# is still a leak. Online chat request ids are ``chatcmpl-...``; anchoring on
# that keeps ``torch_<pid>_<n>`` and other IPC objects out of the count.
_SEGMENT_RE = re.compile(r"chatcmpl-[^/]+_\d+_\d+")


def _is_connector_lockfile(name: str) -> bool:
    return name.startswith("shm_") and name.endswith("_lockfile.lock")


def _is_connector_segment(name: str) -> bool:
    return _SEGMENT_RE.fullmatch(name) is not None


def _is_connector_entry(name: str) -> bool:
    """SharedMemoryConnector artifacts, not CUDA/NCCL/torch IPC objects."""
    return _is_connector_lockfile(name) or _is_connector_segment(name)


def _shm_used_mib() -> str:
    try:
        st = os.statvfs(_SHM_DIR)
        used = (st.f_blocks - st.f_bavail) * st.f_frsize
        return f"{used / (1024 * 1024):.1f}MiB"
    except OSError:
        return "unknown"


def _connector_entry_count() -> int:
    if not _SHM_DIR.is_dir():
        pytest.skip("/dev/shm is not available")
    with os.scandir(_SHM_DIR) as entries:
        return sum(1 for entry in entries if _is_connector_entry(entry.name))


def _connector_entries_since(since_s: float, limit: int = 32) -> list[str]:
    """Return a sample of connector lockfiles/segments with mtime >= ``since_s``."""
    found: list[str] = []
    with os.scandir(_SHM_DIR) as entries:
        for entry in entries:
            if not _is_connector_entry(entry.name):
                continue
            try:
                if entry.stat().st_mtime >= since_s:
                    found.append(entry.name)
                    if len(found) >= limit:
                        break
            except FileNotFoundError:
                continue
    return found


def _wait_no_new_entries(since_s: float, baseline: int, timeout_s: float = _SETTLE_S) -> tuple[int, list[str]]:
    deadline = time.monotonic() + timeout_s
    time.sleep(1.0)
    while True:
        count = _connector_entry_count()
        extra = _connector_entries_since(since_s)
        if count <= baseline and not extra:
            return count, extra
        if time.monotonic() >= deadline:
            return count, extra
        time.sleep(2.0)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.skipif(_USE_PD, reason="Temporarily skip PD mode in this test module.")
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_only_async_chunk_does_not_leak_shm(omni_server, online_client) -> None:
    """Stage-0-final (text-only) requests must not leave connector SHM behind."""
    # Timestamp first so the scan below does not treat its own runtime as "new".
    since_s = time.time()
    baseline = _connector_entry_count()
    print(
        f"[shm-leak] before requests: connector_entries={baseline} /dev/shm_used={_shm_used_mib()}",
        flush=True,
    )

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

    after, extra = _wait_no_new_entries(since_s, baseline)
    print(
        f"[shm-leak] after {_NUM_TEXT_REQUESTS} text-only requests: "
        f"connector_entries={after} /dev/shm_used={_shm_used_mib()} new_sample={extra}",
        flush=True,
    )
    assert after <= baseline and not extra, (
        f"SharedMemoryConnector leaked /dev/shm entries after {_NUM_TEXT_REQUESTS} "
        f"stage-0-final request(s): before={baseline} after={after} new_sample={extra}"
    )
