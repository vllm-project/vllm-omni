# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import uuid

import pytest

from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.utils.shm_readiness import ShmReadiness

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cohort_prevents_partial_reads_and_commits_all_keys(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_OMNI_SHM_COHORT_NAMESPACE", "test_cohort")
    sender, receiver = SharedMemoryConnector({}), SharedMemoryConnector({})
    sender._cohort_directory = receiver._cohort_directory = str(tmp_path)
    watch = ShmReadiness(str(tmp_path))
    keys = [uuid.uuid4().hex for _ in range(2)]
    try:
        with sender.publication_cohort("0", "1"):
            assert sender.put("0", "1", keys[0], {"value": 1})[0]
            assert receiver.get("0", "1", keys[0]) is None
            assert sender.put("0", "1", keys[1], {"value": 2})[0]
            assert receiver.get("0", "1", keys[1]) is None
        assert watch.wait(1)[1] is True
        assert receiver.get("0", "1", keys[0])[0] == {"value": 1}
        assert receiver.get("0", "1", keys[1])[0] == {"value": 2}
        # Readers cannot recursively trigger CLOSE_WRITE notifications.
        assert watch.wait(0)[0] == set()
    finally:
        watch.close()
        sender.close()
        receiver.close()


def test_release_without_successful_put_still_wakes_deferred_readers(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_OMNI_SHM_COHORT_NAMESPACE", "empty")
    sender = SharedMemoryConnector({})
    sender._cohort_directory = str(tmp_path)
    watch = ShmReadiness(str(tmp_path))
    try:
        with sender.publication_cohort("0", "1"):
            pass
        watch.wake()
        assert watch.wait(0) == (set(), True)
    finally:
        watch.close()
        sender.close()
