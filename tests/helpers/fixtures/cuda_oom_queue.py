# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in xdist queue for concurrent model tests that share one CUDA device."""

from collections import OrderedDict

import pytest
from xdist import is_xdist_worker
from xdist.scheduler.loadgroup import LoadGroupScheduling


class _WorkerIdle(pytest.Item):
    """Lookahead for xdist so a worker can finish its last test and accept more."""

    def runtest(self):
        raise AssertionError("the queue lookahead must not be run as a test")


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_collection_finish(session):
    yield
    if is_xdist_worker(session):
        # xdist has already sent the real collection to the controller. Its
        # worker needs a nextitem (or shutdown) before it runs a queued test.
        session.items.append(_WorkerIdle.from_parent(session, name="cuda-oom-queue-idle"))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    if isinstance(item, _WorkerIdle):
        return True
    return None


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    if report.failed and call.excinfo is not None and call.when in ("setup", "call") and is_xdist_worker(item.session):
        error = call.excinfo.value
        message = str(error).lower()
        if isinstance(error, RuntimeError) and (
            "cuda out of memory" in message or "cuda error: out of memory" in message
        ):
            # The controller owns the retry count, as either attempt can run
            # on any worker. Keep the original traceback on the wire.
            report.outcome = "cuda_oom"
    return report


def pytest_report_teststatus(report):
    if report.outcome == "cuda_oom":
        return "rerun", "R", "CUDA OOM (requeued)"
    return None


def pytest_xdist_make_scheduler(config, log):
    scheduler = _CudaOomQueue(config, log)
    config.pluginmanager.register(scheduler)
    return scheduler


class _CudaOomQueue(LoadGroupScheduling):
    def __init__(self, config, log):
        super().__init__(config, log)
        self.first_oom = set()
        self.requeued = set()
        self.failed = set()
        self.retries = {}

    @property
    def tests_finished(self):
        return self.collection_is_completed and not self.has_pending and not self.retries

    def schedule(self):
        if self.collection is None:
            if not self._check_nodes_have_same_collection():
                return
            self.collection = list(next(iter(self.registered_collections.values())))
            for nodeid in self.collection:
                self.workqueue.setdefault(self._split_scope(nodeid), {})[nodeid] = False
            if self.config.option.loadscopereorder:
                self.workqueue = OrderedDict(sorted(self.workqueue.items(), key=lambda entry: -len(entry[1])))
        for node in self.nodes:
            self._reschedule(node)

    def _reschedule(self, node):
        if node.shutting_down or node not in self.registered_collections or self.assigned_work[node]:
            return
        if not self.workqueue:
            if not self.retries:
                return
            nodeid, (peers, owner) = next(iter(self.retries.items()))
            if owner in self.assigned_work and node is not owner:
                return
            if peers and peers <= self._running():
                return
            del self.retries[nodeid]
            self.workqueue[self._split_scope(nodeid)] = {nodeid: False}
        self._assign_work_unit(node)
        node.send_runtest_some([len(self.collection)])

    def _running(self):
        return {nodeid for assigned in self.assigned_work.values() for group in assigned.values() for nodeid in group}

    def mark_test_complete(self, node, item_index, duration=0):
        if item_index == len(self.collection):
            return
        nodeid = self.collection[item_index]
        scope = self._split_scope(nodeid)
        group = self.assigned_work[node][scope]
        del group[nodeid]
        if not group:
            del self.assigned_work[node][scope]
        if nodeid in self.first_oom and nodeid not in self.failed:
            self.first_oom.remove(nodeid)
            self.requeued.add(nodeid)
            # An explicitly grouped test must still use its group's worker.
            owner = node if scope != nodeid else None
            self.retries[nodeid] = (self._running(), owner)
        self.schedule()

    def remove_node(self, node):
        # xdist reports a crashed process as a test failure. Only its
        # unstarted group members go back to the queue.
        assigned = self.assigned_work.pop(node)
        if not assigned:
            return None
        scope, group = assigned.popitem()
        crashitem = next(iter(group))
        del group[crashitem]
        if group:
            self.workqueue[scope] = group
        self.schedule()
        return crashitem

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report):
        if report.outcome == "cuda_oom":
            if report.nodeid in self.requeued:
                report.outcome = "failed"
            else:
                self.first_oom.add(report.nodeid)
                terminal = self.config.pluginmanager.getplugin("terminalreporter")
                if terminal is not None:
                    terminal.write_line(f"CUDA OOM on first attempt: {report.nodeid}")
                    terminal.write_line(report.longreprtext)
                    for title, content in report.sections:
                        terminal.write_sep("-", title)
                        terminal.write_line(content)
        if report.failed:
            self.failed.add(report.nodeid)
