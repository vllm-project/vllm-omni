# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.metrics import MooncakeStoreConnectorStats
from vllm.v1.engine import EngineCoreOutputs

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _records(tokens):
    stats = MooncakeStoreConnectorStats()
    stats.record_operation("prefix_tokens", 0.0, tokens, status="local")
    return stats


def test_throttled_connector_records_are_emitted_once():
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.log_stats = True
    scheduler.make_stats = lambda *args: None
    outputs: dict[int, EngineCoreOutputs] = {}
    attach = OmniSchedulerMixin._attach_scheduler_stats
    attach(scheduler, outputs, None, _records(16), None, None)
    attach(scheduler, outputs, None, _records(32), None, None)
    attach(scheduler, outputs, None, None, None, None)
    assert not outputs

    observed = []

    def emit(spec, connector, graph, perf):
        observed.append(connector)
        return object()

    scheduler.make_stats = emit
    attach(scheduler, outputs, None, _records(64), None, None)
    assert observed[0].reduce()["prefix_tokens_total_keys"] == 112
    assert observed[0].reduce()["prefix_tokens_count"] == 3
    assert 0 in outputs
    attach(scheduler, outputs, None, None, None, None)
    assert observed[1] is None


def test_disabled_statistics_do_not_retain_connector_records():
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.log_stats = False
    scheduler.make_stats = lambda *args: None
    OmniSchedulerMixin._attach_scheduler_stats(scheduler, {}, None, _records(16), None, None)
    assert scheduler._pending_kv_connector_stats is None
