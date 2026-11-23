# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import queue
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine import OmniEngineCoreOutputs
from vllm_omni.worker_v2.first_audio_sender import engine_output_queue_sink

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _scheduler(**client_by_request):
    return SimpleNamespace(
        requests={request_id: SimpleNamespace(client_index=client) for request_id, client in client_by_request.items()}
    )


def test_sink_groups_outputs_by_client_and_carries_audio():
    output_queue: queue.Queue = queue.Queue()
    sink = engine_output_queue_sink(output_queue, _scheduler(a=0, b=1, c=0))
    sample_rate = torch.tensor(24000, dtype=torch.int32)
    rows = [torch.full((4,), float(i)) for i in range(3)]

    sink.prepare(["a", "b", "c"])(["a", "b", "c"], rows, sample_rate)

    delivered = {}
    while not output_queue.empty():
        client_index, outputs = output_queue.get_nowait()
        assert isinstance(outputs, OmniEngineCoreOutputs)
        delivered[client_index] = outputs.outputs
    assert sorted(delivered) == [0, 1]
    assert [o.request_id for o in delivered[0]] == ["a", "c"]
    assert [o.request_id for o in delivered[1]] == ["b"]
    for output in delivered[0] + delivered[1]:
        # First audio carries no tokens and does not finish the request.
        assert output.new_token_ids == []
        assert output.finish_reason is None
        assert output.multimodal_output["sr"] is sample_rate
    assert torch.equal(delivered[0][1].multimodal_output["model_outputs"], rows[2])


def test_sink_skips_requests_that_already_left_the_scheduler():
    output_queue: queue.Queue = queue.Queue()
    sink = engine_output_queue_sink(output_queue, _scheduler(kept=0))

    sink.prepare(["gone", "kept"])(["gone", "kept"], [torch.zeros(2), torch.ones(2)], torch.tensor(24000))
    sink.prepare(["gone"])(["gone"], [torch.zeros(2)], torch.tensor(24000))

    client_index, outputs = output_queue.get_nowait()
    assert client_index == 0
    assert [o.request_id for o in outputs.outputs] == ["kept"]
    assert output_queue.empty()


class _DoneEvent:
    def record(self):
        pass

    def synchronize(self):
        pass


def _run_sender(items):
    from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender

    output_queue: queue.Queue = queue.Queue()
    sink = engine_output_queue_sink(output_queue, _scheduler(**{rid: 0 for item in items for rid in item[2]}))
    sender = FirstAudioSender(sink)
    for item in items:
        sender._queue.put((*item, sink.prepare(item[2])))
    sender.close()
    delivered = []
    while not output_queue.empty():
        _, outputs = output_queue.get_nowait()
        delivered.append(
            ([o.request_id for o in outputs.outputs], [o.multimodal_output["model_outputs"] for o in outputs.outputs])
        )
    return delivered


def test_sender_drops_rows_whose_frame_is_not_audio():
    pcm = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    sample_rate = torch.tensor(24000, dtype=torch.int32)
    valid = torch.tensor([True, False, True])

    delivered = _run_sender(
        [
            (_DoneEvent(), pcm, ["a", "b", "c"], sample_rate, valid),
            # Every row invalid: nothing reaches the sink.
            (_DoneEvent(), pcm[:1], ["d"], sample_rate, torch.tensor([False])),
            # No validity: every row is delivered (previous behaviour).
            (_DoneEvent(), pcm[:1], ["e"], sample_rate, None),
        ]
    )

    assert [ids for ids, _rows in delivered] == [["a", "c"], ["e"]]
    assert torch.equal(delivered[0][1][1], pcm[2])


def test_prepared_delivery_survives_scheduler_completion():
    output_queue: queue.Queue = queue.Queue()
    scheduler = _scheduler(r=3)
    sink = engine_output_queue_sink(output_queue, scheduler)
    deliver = sink.prepare(["r"])
    scheduler.requests.clear()
    deliver(["r"], [torch.ones(2)], torch.tensor(24000))
    client, outputs = output_queue.get_nowait()
    assert client == 3
    assert outputs.outputs[0].multimodal_output["_omni_first_audio"]


def test_delivery_failure_emits_request_error():
    from vllm.v1.engine import FinishReason

    output_queue: queue.Queue = queue.Queue()
    delivery = engine_output_queue_sink(output_queue, _scheduler(r=0)).prepare(["r"])
    delivery.fail(["r"])
    _, outputs = output_queue.get_nowait()
    assert outputs.outputs[0].finish_reason == FinishReason.ERROR


@pytest.mark.parametrize("client_index", [0, 3])
def test_submit_reports_only_requests_with_a_prepared_route(monkeypatch, client_index):
    from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender

    # Exercise the routing contract without allocating CUDA or pinned memory.
    original_empty = torch.empty
    monkeypatch.setattr(torch, "empty", lambda *args, pin_memory=False, **kwargs: original_empty(*args, **kwargs))
    monkeypatch.setattr(torch.cuda, "Event", _DoneEvent)
    output_queue: queue.Queue = queue.Queue()
    sender = FirstAudioSender(engine_output_queue_sink(output_queue, _scheduler(kept=client_index)))
    pcm = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    try:
        accepted = sender.submit(["gone", "kept", "also-gone"], pcm, torch.tensor(24000))
    finally:
        sender.close()

    assert accepted == ["kept"]
    client, outputs = output_queue.get_nowait()
    assert client == client_index
    assert [output.request_id for output in outputs.outputs] == ["kept"]
    assert torch.equal(outputs.outputs[0].multimodal_output["model_outputs"], pcm[1])
    assert output_queue.empty()


def test_submit_without_any_route_leaves_audio_to_the_codec(monkeypatch):
    from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender

    def unexpected_copy(*args, **kwargs):
        pytest.fail("No delivery is possible; do not start a device copy")

    monkeypatch.setattr(torch, "empty", unexpected_copy)
    monkeypatch.setattr(torch.cuda, "Event", unexpected_copy)
    output_queue: queue.Queue = queue.Queue()
    sender = FirstAudioSender(engine_output_queue_sink(output_queue, _scheduler()))
    try:
        accepted = sender.submit(["gone"], torch.ones(1, 2), torch.tensor(24000))
    finally:
        sender.close()

    assert accepted == []
    assert output_queue.empty()


@pytest.mark.parametrize("sink", [lambda ids, rows, sr: None, SimpleNamespace(prepare=None)])
def test_sender_rejects_sink_without_route_preparation(sink):
    from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender

    with pytest.raises(TypeError, match="prepare"):
        FirstAudioSender(sink)


def test_sender_failure_only_aborts_valid_undelivered_rows():
    from vllm.v1.engine import FinishReason

    from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender

    class FailSecondClientOnce(queue.Queue):
        def put_nowait(self, item):
            client, outputs = item
            if client == 1 and outputs.outputs[0].finish_reason is None:
                raise RuntimeError("client enqueue failed")
            super().put_nowait(item)

    output_queue = FailSecondClientOnce()
    sink = engine_output_queue_sink(output_queue, _scheduler(eos=1, sent=0, failed=1, pending=2, next=3))
    sender = FirstAudioSender(sink)
    ids = ["eos", "sent", "failed", "pending", "gone"]
    sender._queue.put(
        (
            _DoneEvent(),
            torch.ones(5, 2),
            ids,
            torch.tensor(24000),
            torch.tensor([False, True, True, True, True]),
            sink.prepare(ids),
        )
    )
    sender._queue.put((_DoneEvent(), torch.ones(1, 2), ["next"], torch.tensor(24000), None, sink.prepare(["next"])))
    sender.close()

    actual: list[tuple[int, str, FinishReason | None]] = []
    while not output_queue.empty():
        client, outputs = output_queue.get_nowait()
        actual.extend((client, o.request_id, o.finish_reason) for o in outputs.outputs)
    assert actual == [
        (0, "sent", None),
        (1, "failed", FinishReason.ERROR),
        (2, "pending", FinishReason.ERROR),
        (3, "next", None),
    ]


def test_sender_copy_failure_reports_all_prepared_routes():
    from vllm.v1.engine import FinishReason

    from vllm_omni.worker_v2.first_audio_sender import FirstAudioSender

    class FailedEvent:
        def synchronize(self):
            raise RuntimeError("copy failed before validity became readable")

    output_queue: queue.Queue = queue.Queue()
    sink = engine_output_queue_sink(output_queue, _scheduler(r=3))
    sender = FirstAudioSender(sink)
    sender._queue.put(
        (FailedEvent(), torch.ones(2, 2), ["r", "gone"], torch.tensor(24000), None, sink.prepare(["r", "gone"]))
    )
    sender.close()
    client, outputs = output_queue.get_nowait()
    assert client == 3
    assert [(o.request_id, o.finish_reason) for o in outputs.outputs] == [("r", FinishReason.ERROR)]
    assert output_queue.empty()
