# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-stream boundary tests; no model weights are required."""

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.cosyvoice3.async_codec import AsyncCodec, SampleEntry

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA stream test requires a CUDA device"),
]


class Decoder:
    def forward_streaming(self, token, finalize, **kwargs):
        # Distinguishable PCM and finalization, without loading a token_mask model.
        return token.float().reshape(1, 1, -1) + (100 if finalize else 0), None


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_codec_progresses_and_flushes_eos_without_another_ar_step(monkeypatch):
    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW", "0")
    worker = AsyncCodec(Decoder(), torch.device("cuda", 0), 32, 4, chunk_size=2, lookahead=0)
    conditioning = {
        "speech_token": torch.empty(1, 0, dtype=torch.long, device="cuda"),
        "speech_feat": torch.empty(1, 0, 80, device="cuda"),
        "embedding": torch.ones(1, 192, device="cuda"),
    }
    try:
        for position, token in enumerate((7, 8, 100)):
            entry = SampleEntry("r", position, False, conditioning)
            worker.submit([entry], torch.tensor([token], device="cuda"), torch.tensor([token < 100], device="cuda"))
        # No engine/runner query drives either decode or D2H publication.
        first = worker.outputs.get(timeout=10)
        last = worker.outputs.get(timeout=10)
        assert first.pcm.device.type == "cpu"
        assert first.pcm.tolist() == [7.0, 8.0]
        assert not first.final
        assert last.final
        assert last.pcm.tolist() == [107.0, 108.0]
    finally:
        worker.close()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_ar_can_append_on_another_stream_while_codec_is_busy(monkeypatch):
    import threading

    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setenv("COSYVOICE3_BATCH_FLOW", "0")
    entered, resume, produced = threading.Event(), threading.Event(), threading.Event()
    errors = []

    class SlowDecoder(Decoder):
        def forward_streaming(self, token, finalize, **kwargs):
            if not entered.is_set():
                entered.set()
                if not resume.wait(10):
                    raise TimeoutError("Test did not release decoder")
            return super().forward_streaming(token, finalize, **kwargs)

    worker = AsyncCodec(SlowDecoder(), torch.device("cuda", 0), 64, 1, chunk_size=2, lookahead=0)
    conditioning = {
        "speech_token": torch.empty(1, 0, dtype=torch.long, device="cuda"),
        "speech_feat": torch.empty(1, 0, 80, device="cuda"),
        "embedding": torch.ones(1, 192, device="cuda"),
    }
    # Producer stream must see conditioning initialization before the worker
    # consumes it. Production creates conditioning before the sample event.
    init = torch.cuda.Event()
    init.record()
    producer_stream = torch.cuda.Stream()
    producer_stream.wait_event(init)

    def push(position, token):
        worker.submit(
            [SampleEntry("r", position, False, conditioning)],
            torch.tensor([token], device="cuda"),
            torch.tensor([token < 100], device="cuda"),
        )

    def append_rest():
        try:
            current_omni_platform.set_device(torch.device("cuda", 0))
            with torch.cuda.stream(producer_stream):
                for position in range(2, 32):
                    push(position, position + 1)
                push(32, 100)
                push(33, 77)  # Already-scheduled sample after EOS must be ignored.
        except BaseException as error:
            errors.append(error)
        finally:
            produced.set()

    producer = threading.Thread(target=append_rest)
    try:
        with torch.cuda.stream(producer_stream):
            push(0, 1)
            push(1, 2)
        assert entered.wait(5)
        producer.start()
        assert produced.wait(5), "AR waited for codec to drain step notifications"
        assert not errors
        with worker.lock:
            assert len(worker.mailbox) == 1
        resume.set()
        chunks = []
        while not chunks or not chunks[-1].final:
            chunks.append(worker.outputs.get(timeout=10))
        assert [x.pcm.numel() for x in chunks] == [2, 6, 14, 22, 30, 32]
        assert chunks[0].pcm.tolist() == [1.0, 2.0]
        assert chunks[-1].pcm.tolist() == [float(x + 100) for x in range(1, 33)]
    finally:
        resume.set()
        if producer.ident is not None:
            producer.join(timeout=10)
        worker.close()


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_source_row_selection_does_not_upload_indices_or_synchronize():
    """Check CUDA API events inside producer submission, not profiler teardown."""
    worker = AsyncCodec(Decoder(), torch.device("cuda", 0), 32, 1, chunk_size=64, lookahead=0)
    conditioning = {
        "speech_token": torch.empty(1, 0, dtype=torch.long, device="cuda"),
        "speech_feat": torch.empty(1, 0, 80, device="cuda"),
        "embedding": torch.ones(1, 192, device="cuda"),
    }
    tokens = torch.tensor([13, 77, 29], device="cuda")
    token_mask = torch.tensor([True, False, True], device="cuda")

    def entries(position):
        return [SampleEntry(req_id, position, False, conditioning) for req_id in ("a", "b")]

    try:
        # Allocate request buffers before profiling so allocator cold start is
        # separate from the two per-step index uploads that caused the trace stall.
        worker.submit(entries(0), tokens, token_mask, rows=[2, 0])
        next_entries = entries(1)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as p:
            with torch.profiler.record_function("cosyvoice3_submit_rows"):
                worker.submit(next_entries, tokens, token_mask, rows=[2, 0])
        names = []
        for event in p.events():
            parent = event.cpu_parent
            while parent is not None:
                if parent.name == "cosyvoice3_submit_rows":
                    names.append(event.name)
                    break
                parent = parent.cpu_parent
        assert "aten::copy_" in names  # Confirm the intended producer region was recorded.
        assert "aten::index" not in names
        assert "cudaStreamSynchronize" not in names
        assert "cudaDeviceSynchronize" not in names
        tokens.fill_(999)  # Reusing graph output storage must not overwrite owned tokens.
        assert worker.buffers["a"].tokens[:, :2].cpu().tolist() == [[29, 29]]
        assert worker.buffers["b"].tokens[:, :2].cpu().tolist() == [[13, 13]]
    finally:
        worker.close()
