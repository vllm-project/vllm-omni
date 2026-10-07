# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-bucket alignment for the MiniCPM-o 4.5 NPU code2wav graph path.

The NPU capture key is the full tensor shape
(``NPUExactGraphRunner._tensor_signature``), so every distinct batch size
multiplies the captured shape space. Mel frames are already aligned by
``cfm_graph_bucket_frames``; these tests pin the complementary batch-side
alignment: pad up to a bucket before capture, then drop the padded rows from
the outputs so callers still see exactly the live requests.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm_omni.platforms.npu.models import minicpmo_4_5_code2wav as npu_patch


class TestBatchBucket:
    def test_rounds_up_to_next_bucket(self):
        buckets = (1, 2, 4, 8)
        assert [npu_patch._batch_bucket(n, buckets) for n in range(1, 9)] == [1, 2, 4, 4, 8, 8, 8, 8]

    def test_exact_bucket_is_unchanged(self):
        assert npu_patch._batch_bucket(8, (1, 2, 4, 8)) == 8

    def test_batch_beyond_largest_bucket_stays_exact(self):
        # Rounding 12 up to a hypothetical 16 would be correct but 12 is not a
        # configured bucket, so the caller keeps the exact shape instead.
        assert npu_patch._batch_bucket(12, (1, 2, 4, 8)) == 12


class TestParseBatchBuckets:
    @pytest.mark.parametrize("raw", [None, "", [], ()])
    def test_absent_keeps_exact_batch_capture(self, raw):
        assert npu_patch._parse_batch_buckets(raw) == ()

    def test_sorts_and_deduplicates(self):
        assert npu_patch._parse_batch_buckets([4, 1, 2, 2]) == (1, 2, 4)

    def test_accepts_string_form(self):
        assert npu_patch._parse_batch_buckets("1, 4 8") == (1, 4, 8)

    @pytest.mark.parametrize("raw", [[0], [-2], [1.5], {"a": 1}])
    def test_rejects_invalid_entries(self, raw):
        with pytest.raises(ValueError):
            npu_patch._parse_batch_buckets(raw)


class TestPadding:
    def test_pad_rows_appends_zero_rows(self):
        value = torch.ones((3, 2, 4))
        padded = npu_patch._pad_rows(value, 4)
        assert padded.shape == (4, 2, 4)
        torch.testing.assert_close(padded[:3], value)
        assert padded[3].abs().sum().item() == 0.0

    def test_pad_rows_is_a_noop_when_already_sized(self):
        value = torch.ones((4, 2))
        assert npu_patch._pad_rows(value, 4) is value

    def test_pad_cache_rows_pads_the_batch_axis(self):
        # Packed caches are (num_blocks, batch, channels, width).
        value = torch.ones((2, 3, 5, 7))
        padded = npu_patch._pad_cache_rows(value, 4)
        assert padded.shape == (2, 4, 5, 7)
        torch.testing.assert_close(padded[:, :3], value)
        assert padded[:, 3].abs().sum().item() == 0.0

    def test_pad_cache_rows_is_a_noop_when_already_sized(self):
        value = torch.ones((2, 4, 5, 7))
        assert npu_patch._pad_cache_rows(value, 4) is value


def _run_batch_bucket_script(body: str) -> subprocess.CompletedProcess:
    # The interpreter-teardown heap corruption in this image (unrelated to the
    # code under test) would turn a passing script into SIGABRT, so the script
    # exits hard once its assertions and prints are done.
    script = f"""
import os
import sys

import torch

from tests.model_executor.models.minicpmo_4_5.test_code2wav_batching import (
    _FakeToken2Wav,
)
from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
import vllm_omni.platforms.npu.models.minicpmo_4_5_code2wav as npu_patch


class _RecordingGraphRunner:
    \"\"\"Stands in for NPUExactGraphRunner: records capture inputs, runs eager.\"\"\"

    def __init__(self):
        self.batch_sizes = []

    def run(self, operation, inputs, constants, compute):
        self.batch_sizes.append(tuple(value.shape[0] for value in inputs))
        return compute(*inputs)


class _ShapeRecordingGraphRunner:
    \"\"\"Records the full shape of every capture input, so cache axes show up.\"\"\"

    def __init__(self):
        self.inputs: list[tuple[tuple[int, ...], ...]] = []

    def run(self, operation, inputs, constants, compute):
        self.inputs.append(tuple(tuple(value.shape) for value in inputs))
        return compute(*inputs)


npu_patch.apply_minicpmo_4_5_code2wav_patch()
{body}
sys.stdout.flush()
os._exit(0)
"""
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[4],
        capture_output=True,
        text=True,
    )


def _step_body(buckets: str, batch: int) -> str:
    """Drive one estimator step with an exact batch, no decode_batch involved.

    ``decode_batch`` splits work into micro-batches and runs the estimator under
    CFG, so its batch numbers do not equal the request count. Calling the
    patched step directly keeps this test about the alignment itself.
    """
    return f"""
adapter = BatchedToken2Wav(_FakeToken2Wav())
adapter._cfm_graph_batch_buckets = {buckets}
runner = _RecordingGraphRunner()
npu_patch._backend_graph_runners[adapter] = runner
estimator = adapter.flow.decoder.estimator

batch, chunk, dim = {batch}, 4, 2
result, new_cnn, new_att = npu_patch._patched_estimator_step(
    adapter,
    estimator,
    x=torch.arange(batch * chunk * dim, dtype=torch.float32).reshape(batch, chunk, dim),
    mu=torch.ones(batch, chunk, dim),
    time=torch.zeros(batch),
    speakers=torch.ones(batch, chunk),
    cond=torch.ones(batch, chunk, dim),
    cnn_cache=None,
    att_cache=None,
)
print("SEEN", sorted({{s[0] for s in runner.batch_sizes}}), "OUT", tuple(result.shape))
"""


def _seen_and_out(completed: subprocess.CompletedProcess) -> tuple[set[int], tuple[int, ...]]:
    assert completed.returncode == 0, completed.stderr
    line = next(line for line in completed.stdout.splitlines() if line.startswith("SEEN"))
    seen_text, out_text = line[len("SEEN ") :].split(" OUT ")
    return set(ast.literal_eval(seen_text)), tuple(ast.literal_eval(out_text))


def test_batch_three_is_captured_as_four_and_trimmed_back():
    seen, out = _seen_and_out(_run_batch_bucket_script(_step_body("(4,)", 3)))
    assert seen == {4}, seen
    # The batch axis of the result is trimmed back to the live rows.
    assert out[0] == 3, out


def test_bucket_of_one_never_pads():
    seen, out = _seen_and_out(_run_batch_bucket_script(_step_body("(1,)", 3)))
    assert seen == {3}, seen
    assert out[0] == 3, out


def test_unset_buckets_keep_exact_batch_shapes():
    seen, out = _seen_and_out(_run_batch_bucket_script(_step_body("()", 3)))
    assert seen == {3}, seen
    assert out[0] == 3, out


def test_every_batch_rounds_into_the_bucket_set():
    """7 live rows share the 8-row capture, and outputs stay at 7."""
    seen, out = _seen_and_out(_run_batch_bucket_script(_step_body("(1, 2, 4, 8)", 7)))
    assert seen == {8}, seen
    assert out[0] == 7, out


def test_padded_cache_slots_are_dropped_from_outputs():
    """Packed caches are padded on dim 1 and trimmed back on the way out."""
    body = """
adapter = BatchedToken2Wav(_FakeToken2Wav())
adapter._cfm_graph_batch_buckets = (4,)
runner = _ShapeRecordingGraphRunner()
npu_patch._backend_graph_runners[adapter] = runner
estimator = adapter.flow.decoder.estimator

batch, chunk, dim = 3, 4, 2
depth = len(estimator.blocks)
cnn_cache = torch.zeros(depth, batch, 3, 6)
att_cache = torch.zeros(depth, batch, 2, 0, 4)
result, new_cnn, new_att = npu_patch._patched_estimator_step(
    adapter,
    estimator,
    x=torch.ones(batch, chunk, dim),
    mu=torch.ones(batch, chunk, dim),
    time=torch.zeros(batch),
    speakers=torch.ones(batch, chunk),
    cond=torch.ones(batch, chunk, dim),
    cnn_cache=cnn_cache,
    att_cache=att_cache,
)
# x, mu, time_embedding, speakers, cond, cnn_cache, att_cache
assert len(runner.inputs) == 1, runner.inputs
shapes = runner.inputs[0]
assert len(shapes) == 7, shapes
# Padded to 4 on the way in ...
for shape in shapes[:5]:
    assert shape[0] == 4, shape
# ... including the packed caches, whose batch axis is dim 1.
for shape in shapes[5:]:
    assert shape[1] == 4, shape
# ... and trimmed back to the live rows on the way out.
assert result.shape[0] == batch, result.shape
assert new_cnn.shape[1] == batch, new_cnn.shape
assert new_att.shape[1] == batch, new_att.shape
print("SEEN", {4}, "OUT", (batch,))
"""
    seen, out = _seen_and_out(_run_batch_bucket_script(body))
    assert out[0] == 3, out
