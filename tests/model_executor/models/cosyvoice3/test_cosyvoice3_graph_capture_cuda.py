# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Overlap actual codec capture with an independent producer allocation."""

import threading
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("kind", ["flow", "hift"])
def test_ar_allocation_during_codec_capture(kind):
    entered, resume = threading.Event(), threading.Event()
    results, errors = [], []
    device = torch.device("cuda", 0)
    value = torch.ones((1, 4, 3), device=device)
    initialized = torch.cuda.Event()
    initialized.record()

    def body(x):
        if torch.cuda.is_current_stream_capturing():
            entered.set()
            if not resume.wait(10):
                raise TimeoutError("producer did not release codec capture")
        return x * 2

    if kind == "flow":
        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import FlowGraphRunner

        # Exercise the production runner's warmup/capture/replay boundary,
        # replacing only the heavyweight DiT solve with a deterministic op.
        runner = object.__new__(FlowGraphRunner)
        runner.graphs, runner.pool = {}, None
        runner.capture_stream = None
        runner._layout = lambda *a: (("layout",), (), (), (), ())
        runner._allocate = lambda *a: SimpleNamespace(graph=torch.cuda.CUDAGraph())
        runner._load = lambda *a: None
        runner._solve = lambda *a: body(value)

        def run():
            return runner.run(
                value,
                value,
                value,
                value,
                value,
                SimpleNamespace(lengths=(4,)),
                cfg_rate=0.7,
                streaming=False,
                key=(),
            )
    else:
        from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.hift_graph import HiFTDecodeGraphs

        runner = HiFTDecodeGraphs(lambda mel, source, finalize: body(mel))

        def run():
            return runner.run(value, value, False)

        # HiFT records recurring shapes on their third invocation.
        run()
        run()
        initialized.record()

    @torch.inference_mode()
    def codec():
        try:
            current_omni_platform.set_device(device)
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                stream.wait_event(initialized)
                results.append(run())
                stream.synchronize()
        except BaseException as error:
            errors.append(error)
            entered.set()

    worker = threading.Thread(target=codec)
    worker.start()
    try:
        assert entered.wait(10), "codec did not reach graph capture"
        assert not errors
        # Reproduce the failing AR torch.zeros while codec capture is active.
        # Its allocation and memset belong to the producer, never to the graph.
        independent = torch.zeros((1024, 1024), device=device)
    finally:
        resume.set()
        worker.join(timeout=15)
    assert not worker.is_alive()
    assert not errors
    torch.testing.assert_close(results[0], value * 2)
    assert independent.count_nonzero().item() == 0
    with torch.inference_mode():
        torch.testing.assert_close(run(), value * 2)
    assert len(runner.graphs) == 1
