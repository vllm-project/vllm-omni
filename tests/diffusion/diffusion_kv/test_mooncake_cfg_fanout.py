# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise upstream Mooncake completion accounting without device/network I/O."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import msgspec
import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import mooncake_connector as mc
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import KVConnectorOutput

from vllm_omni.diffusion.diffusion_kv.kv_connector import install_mooncake_cfg_fanout
from vllm_omni.worker.mixins import OmniWorkerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def producer(monkeypatch):
    worker = object.__new__(mc.MooncakeConnectorWorker)
    worker.shutdown = Mock()  # No hardware constructor, sockets or listener thread.
    worker.is_kv_consumer = False
    worker.tp_rank, worker.tp_size = 0, 1
    worker.transfer_topo = SimpleNamespace(handshake_target_ranks=lambda size: list(range(size)))
    worker.reqs_need_send, worker.finished_sending_reqs = {}, set()
    for name in (
        "kv_caches_base_addr",
        "block_len_per_layer",
        "kv_block_len_per_layer",
        "registered_layer_names",
        "registered_layer_indices",
        "registered_group_indices",
    ):
        setattr(worker, name, [])
    worker._get_transfer_regions = Mock(return_value=[])
    worker._producer_cache_is_replicated = Mock(return_value=False)
    worker._encoder = msgspec.msgpack.Encoder()
    worker._sender_executor = None
    worker._send_blocks = Mock(return_value=0)
    worker.xfer_stats = Mock()
    worker._build_transfer_params = AsyncMock(return_value=([1], [2], [16], [], None))
    monkeypatch.setattr(mc, "_align_transfer_regions", lambda *_: ([], [], None))
    monkeypatch.setattr(mc, "_validate_asymmetric_region_lengths", lambda **_: None)
    monkeypatch.setattr(mc.envs, "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", 0.05)
    connector = object.__new__(mc.MooncakeConnector)
    connector.connector_worker = worker
    return connector


async def register(worker, ticket):
    worker.sender_loop = asyncio.get_running_loop()
    for blocks in ([], [[10]]):
        meta = mc.MooncakeConnectorMetadata()
        meta.add_new_req(ticket, blocks, {"transfer_id": ticket}, load_remote_cache=False)
        await worker.record_send_reqs(meta)
    return worker.reqs_need_send[ticket]


def packet(rank, ranks, tickets, empty=False):
    return mc.MooncakeXferMetadata(
        remote_hostname="localhost",
        remote_port=1234 + rank,
        remote_tp_size=ranks,
        remote_tp_rank=rank,
        req_blocks={
            f"{ticket}/{row}": (ticket, [] if empty else [[100 + rank * 10 + row]])
            for ticket, rows in tickets.items()
            for row in range(rows)
        },
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
    )


async def send(worker, rank, ranks, tickets, empty=False):
    sock = SimpleNamespace(send_multipart=AsyncMock())
    await worker.send_kv_to_decode(str(rank).encode(), sock, packet(rank, ranks, tickets, empty))
    return [
        msgspec.msgpack.decode(c.args[0][1], type=mc.MooncakeXferResponse) for c in sock.send_multipart.call_args_list
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("ranks", [2, 4])
@pytest.mark.parametrize("in_flight", [False, True])
async def test_source_pages_wait_for_every_rank_and_row(producer, rows, ranks, in_flight):
    worker = producer.connector_worker
    entered, release = asyncio.Event(), asyncio.Event()
    original = worker._build_transfer_params

    async def plan(ready, meta, *regions):
        if in_flight and meta.remote_tp_rank == ranks - 1:
            entered.set()
            await release.wait()
        return await original(ready, meta, *regions)

    worker._build_transfer_params = plan
    install_mooncake_cfg_fanout(producer)
    adapted = worker._build_transfer_params
    install_mooncake_cfg_fanout(producer)
    assert worker._build_transfer_params is adapted  # Do not multiply twice.
    ticket = await register(worker, "request")
    pending = asyncio.create_task(send(worker, ranks - 1, ranks, {"request": rows})) if in_flight else None
    try:
        if pending is not None:
            await asyncio.wait_for(entered.wait(), 2)
        for rank in range(ranks - 1):
            await send(worker, rank, ranks, {"request": rows})
            assert ticket.need_send == ranks * rows
            assert ticket.sent == (rank + 1) * rows
            assert worker.reqs_need_send["request"] is ticket
            finished = await worker.fetch_finished_sending_reqs()
            scheduler = SimpleNamespace(connector=None, requests={"request": object()}, _free_blocks=Mock())
            Scheduler._update_from_kv_xfer_finished(scheduler, KVConnectorOutput(finished_sending=finished))
            scheduler._free_blocks.assert_not_called()
        release.set()
        responses = (
            await asyncio.wait_for(pending, 2) if pending else await send(worker, ranks - 1, ranks, {"request": rows})
        )
        assert sum(len(response.ok_reqs or []) for response in responses) == rows
        assert all(not response.err_reqs for response in responses)
        assert "request" not in worker.reqs_need_send
        finished = await worker.fetch_finished_sending_reqs()
        assert finished == {"request"}
        Scheduler._update_from_kv_xfer_finished(scheduler, KVConnectorOutput(finished_sending=finished))
        scheduler._free_blocks.assert_called_once()
        assert not await worker.fetch_finished_sending_reqs()
    finally:
        release.set()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("ranks", [1, 2])
async def test_mixed_tickets_and_empty_block_notifications(producer, empty, ranks):
    worker = producer.connector_worker
    if empty:
        worker._build_transfer_params.return_value = ([], [], [], [], None)
    install_mooncake_cfg_fanout(producer)
    single, cfg = await register(worker, "single"), await register(worker, "cfg")
    await asyncio.gather(*(send(worker, rank, ranks, {"single": 1, "cfg": 2}, empty) for rank in range(ranks)))
    assert (single.need_send, single.sent) == (ranks, ranks)
    assert (cfg.need_send, cfg.sent) == (2 * ranks, 2 * ranks)
    assert await worker.fetch_finished_sending_reqs() == {"single", "cfg"}
    assert not worker.reqs_need_send


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["row_count", "transport"])
async def test_failed_rank_does_not_report_source_complete(producer, failure):
    worker = producer.connector_worker
    install_mooncake_cfg_fanout(producer)
    ticket = await register(worker, "request")
    await send(worker, 0, 2, {"request": 2})
    if failure == "transport":
        worker._send_blocks.return_value = -1
    responses = await send(worker, 1, 2, {"request": 1 if failure == "row_count" else 2})
    assert any(response.err_reqs for response in responses)
    assert ticket.sent == 2 and ticket.need_send == 4 and ticket.sending == 0
    assert worker.reqs_need_send["request"] is ticket
    assert not await worker.fetch_finished_sending_reqs()
    # Preserve upstream timeout cleanup after all in-flight writes have stopped.
    ticket.expire_time = 0
    assert await worker.fetch_finished_sending_reqs() == {"request"}
    assert not worker.reqs_need_send


def test_worker_installs_adapter_after_native_initialization(producer, monkeypatch):
    import vllm.distributed.kv_transfer as kv_transfer

    worker = producer.connector_worker
    original = worker._build_transfer_params

    class UpstreamWorker:
        def initialize_from_config(self, config):
            assert config == "config"
            assert worker._build_transfer_params is original

    class OmniWorker(OmniWorkerMixin, UpstreamWorker):
        pass

    monkeypatch.setattr(kv_transfer, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(kv_transfer, "get_kv_transfer_group", lambda: producer)
    object.__new__(OmniWorker).initialize_from_config("config")
    assert worker._build_transfer_params is not original


@pytest.mark.parametrize("role", ["other_connector", "scheduler", "consumer"])
def test_adapter_leaves_other_roles_untouched(producer, role):
    worker = producer.connector_worker
    original = worker._build_transfer_params
    if role == "scheduler":
        producer.connector_worker = None
    elif role == "consumer":
        worker.is_kv_consumer = True
    install_mooncake_cfg_fanout(object() if role == "other_connector" else producer)
    assert worker._build_transfer_params is original
