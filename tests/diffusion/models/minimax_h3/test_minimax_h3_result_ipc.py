# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 output contracts across a real spawned worker result queue."""

import multiprocessing
import traceback
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _send_h3_output(result_queue, num_outputs):
    try:
        from vllm_omni.diffusion.data import DiffusionOutput
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
            _MiniMaxH3EncodedOutput,
            get_minimax_h3_post_process_func,
        )

        encoded = [b"encoded-video"] * num_outputs
        output = _MiniMaxH3EncodedOutput(video=encoded, frame_counts=[7] * num_outputs) if num_outputs else (b"", None)
        result_queue.put(
            DiffusionOutput(output=output, post_process_func=get_minimax_h3_post_process_func(SimpleNamespace()))
        )
    except Exception:
        result_queue.put({"error": traceback.format_exc()})


@pytest.mark.parametrize("num_outputs", [0, 1, 2], ids=["peer", "single-output", "two-outputs"])
def test_h3_result_crosses_spawned_worker_queue(num_outputs):
    from vllm_omni.diffusion.data import DiffusionOutput
    from vllm_omni.diffusion.output_formatter import normalize_diffusion_postprocess_output

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    worker = context.Process(target=_send_h3_output, args=(result_queue, num_outputs))
    try:
        worker.start()
        result = result_queue.get(timeout=90)
        worker.join(timeout=30)
        assert not worker.is_alive(), "H3 result worker did not exit after sending its result"
        assert worker.exitcode == 0
        assert isinstance(result, DiffusionOutput), result
        normalized = normalize_diffusion_postprocess_output(result.post_process_func(result.output))
        assert normalized.outputs["video"] == ([b"encoded-video"] * num_outputs if num_outputs else [b""])
        assert normalized.outputs["audio"] == [None] * max(1, num_outputs)
        assert normalized.metadata["video"]["fps"] == 24
        assert normalized.metadata["audio"]["sample_rate"] == 32000
        if num_outputs:
            assert normalized.metadata["video"]["frame_counts"] == [7] * num_outputs
        else:
            assert "frame_counts" not in normalized.metadata["video"]
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()
