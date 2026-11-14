# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, field

import pytest

from vllm_omni.distributed.omni_connectors.connectors import mooncake_store_connector as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class Clock:
    now: float = 10.0
    sleeps: list[float] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    return clock


@pytest.fixture
def connector(monkeypatch, mocker):
    store = mocker.Mock()
    store.setup.return_value = 0
    store.get.return_value = b""
    monkeypatch.setattr(module, "MooncakeDistributedStore", lambda: store)
    monkeypatch.setattr(module, "ReplicateConfig", mocker.Mock)
    connector = module.MooncakeStoreConnector({})
    monkeypatch.setattr(connector, "deserialize_obj", lambda raw: raw.decode())
    yield connector
    connector.close()


def test_expired_deadline_does_not_access_store(connector, clock):
    assert connector.get_with_deadline("0", "1", "request", deadline=clock.now) is None
    connector.store.get.assert_not_called()
    assert not clock.sleeps
    assert connector._metrics["timeouts"] == 1


@pytest.mark.parametrize("store_error", [False, True])
def test_retries_share_deadline_and_clamp_sleep(connector, clock, store_error):
    if store_error:
        connector.store.get.side_effect = RuntimeError("temporarily unavailable")
    deadline = clock.now + 0.12

    assert connector.get_with_deadline("0", "1", "request", deadline=deadline) is None

    assert connector.store.get.call_count == 3
    assert clock.sleeps == pytest.approx([0.05, 0.05, 0.02])
    assert clock.now == pytest.approx(deadline)
    assert connector._metrics["timeouts"] == 1


def test_payload_becomes_visible_before_deadline(connector, clock):
    connector.store.get.side_effect = [b"", b"payload"]

    assert connector.get_with_deadline("2", "5", "request", deadline=clock.now + 1) == ("payload", 7)

    connector.store.get.assert_called_with("request@2_5")
    assert clock.sleeps == [0.05]
    assert connector._metrics["gets"] == 1
    assert connector._metrics["timeouts"] == 0


@pytest.mark.parametrize("payload", [b"", b"payload"])
def test_native_call_is_not_preempted_or_retried_after_deadline(connector, clock, payload):
    def slow_get(key):
        clock.now += 1
        return payload

    connector.store.get.side_effect = slow_get
    result = connector.get_with_deadline("0", "1", "request", deadline=clock.now + 0.01)

    assert result == (("payload", 7) if payload else None)
    assert connector.store.get.call_count == 1
    assert not clock.sleeps


def test_legacy_get_retains_retry_limit(connector, clock):
    assert connector.get("0", "1", "request") is None
    assert connector.store.get.call_count == 20
    assert clock.sleeps == [0.05] * 19
    assert connector._metrics["timeouts"] == 1


def test_deadline_get_retains_per_call_retry_limit(connector, clock):
    assert connector.get_with_deadline("0", "1", "request", deadline=clock.now + 5) is None
    assert connector.store.get.call_count == 20
    assert clock.sleeps == [0.05] * 19


def test_uninitialized_store_returns_none(connector, clock):
    connector.close()
    assert connector.get_with_deadline("0", "1", "request", deadline=clock.now + 1) is None
    assert not clock.sleeps
