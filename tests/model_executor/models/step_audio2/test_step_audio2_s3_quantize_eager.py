# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``_quantize_s3_graph`` must stay bit-identical to s3tokenizer's short-audio branch (eager fallback off-NPU).

The eager path inlines the else-branch of ``S3TokenizerV2.quantize`` so short
audio skips the implicit D2H sync that ``long_audio_mask.any()`` forces. This
test pins the inlined copy to the library on CPU with random weights, across
varied lengths, batch shapes, and the long-audio boundary.
"""

import contextlib
import threading

import pytest
import torch

from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
    _S3_MAX_FRAMES,
    StepAudio2Token2WavCore,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

s3tokenizer = pytest.importorskip("s3tokenizer")


def _core_with(model) -> StepAudio2Token2WavCore:
    # The eager path only touches audio_tokenizer and device; skip the full
    # model bootstrap (yaml, campplus, vocoder weights). Core subclasses
    # nn.Module, so its base bookkeeping must be initialized before a child
    # module can be assigned. Seed the private fields directly so reads go
    # through the real `audio_tokenizer` property instead of relying on the
    # nn.Module.__getattr__ fallback after a failed `_ensure_models_loaded`.
    core = StepAudio2Token2WavCore.__new__(StepAudio2Token2WavCore)
    torch.nn.Module.__init__(core)
    core._models_loaded = True
    core._audio_tokenizer = model
    core.device = torch.device("cpu")
    # Graph bookkeeping normally declared in __init__ (skipped above).
    core._s3_graphs = {}
    core._s3_failed = set()
    core._s3_graph_lock = threading.Lock()
    core._s3_graph_done = None
    core._s3_replay_calls = 0
    core._s3_eager_calls = 0
    return core


def _batch(lengths, seed=0):
    g = torch.Generator().manual_seed(seed)
    max_len = max(lengths)
    mels = torch.randn(len(lengths), 128, max_len, generator=g)
    mels_lens = torch.tensor(lengths, dtype=torch.long)
    return mels, mels_lens


@pytest.mark.parametrize(
    "lengths",
    [[50], [50, 120, 200], [100, 100], [1, 2999], [3000]],
)
def test_short_audio_eager_matches_library(lengths) -> None:
    torch.manual_seed(0)
    model = s3tokenizer.model_v2.S3TokenizerV2("speech_tokenizer_v2_25hz").eval()
    core = _core_with(model)

    mels, mels_lens = _batch(lengths)
    with torch.inference_mode():
        lib_code, lib_len = model.quantize(mels, mels_lens)
        eager_code, eager_len = core._quantize_s3_graph(mels, mels_lens)

    assert torch.equal(lib_code, eager_code)
    assert torch.equal(lib_len, eager_len)


def test_max_frames_boundary_matches_library() -> None:
    # Both the library and the host-side check use strict ">", so exactly
    # _S3_MAX_FRAMES frames must take the short path on both sides.
    torch.manual_seed(1)
    model = s3tokenizer.model_v2.S3TokenizerV2("speech_tokenizer_v2_25hz").eval()
    core = _core_with(model)

    mels, mels_lens = _batch([_S3_MAX_FRAMES], seed=1)
    assert int(mels_lens.max()) <= _S3_MAX_FRAMES

    with torch.inference_mode():
        lib_code, lib_len = model.quantize(mels, mels_lens)
        eager_code, eager_len = core._quantize_s3_graph(mels, mels_lens)

    assert torch.equal(lib_code, eager_code)
    assert torch.equal(lib_len, eager_len)


class _FakeNPUDevice:
    """Stands in for ``torch.device("npu")`` on a CPU-only box: ``type``
    reads "npu" so the capture branch is entered, while equality matches
    cpu tensors (and the fake itself, plugged into the freqs_cis device
    check) so nothing is moved to a device that does not exist."""

    type = "npu"

    def __eq__(self, other):
        if other is self:
            return True
        return isinstance(other, torch.device) and other.type == "cpu"

    def __hash__(self):
        return hash(("fake-npu", id(self)))


class _FakeStream:
    def wait_event(self, event):
        pass

    def record_event(self):
        return object()


class _BrokenGraph:
    """NPUGraph whose first ``replay()`` raises: the failure mode the
    registration-order fix must survive without publishing a bucket."""

    instances = []

    def __init__(self):
        self.replays = 0
        _BrokenGraph.instances.append(self)

    def replay(self):
        self.replays += 1
        raise RuntimeError("simulated first-replay failure")


class _FakeNPUModule:
    """Minimal ``torch.npu`` stand-in: the real encoder/quantizer kernels
    run eagerly on CPU, only the graph bookkeeping is faked."""

    class config:
        """Mimic ``torch.npu.config`` (the prepare step flips this)."""

        allow_internal_format = True

    @staticmethod
    def set_compile_mode(jit_compile=False):
        pass

    graph = staticmethod(contextlib.nullcontext)

    @staticmethod
    def current_stream(device):
        return _FakeStream()

    @staticmethod
    def synchronize():
        pass

    NPUGraph = _BrokenGraph


def test_first_replay_failure_falls_back_to_eager(monkeypatch) -> None:
    # Review r7: the bucket entry is published only after the first
    # replay+clone succeed. Fake a graph whose first replay() raises and
    # assert no bucket is left behind, the frame count is marked failed,
    # and the same-length retry returns the eager result.
    torch.manual_seed(2)
    model = s3tokenizer.model_v2.S3TokenizerV2("speech_tokenizer_v2_25hz").eval()
    core = _core_with(model)
    core.device = _FakeNPUDevice()
    # No freqs_cis stand-in needed: the real CPU-resident buffer compares
    # equal to the fake device (its __eq__ matches cpu), so the hoist is
    # skipped, and AudioEncoderV2.forward's internal
    # ``self.freqs_cis.to(x.device)`` is a no-op on cpu tensors.
    monkeypatch.setattr(torch, "npu", _FakeNPUModule, raising=False)
    _BrokenGraph.instances.clear()

    mels, mels_lens = _batch([50], seed=2)
    frames = int(mels.shape[2])

    with torch.inference_mode():
        fallback_code, fallback_len = core._quantize_s3_graph(mels, mels_lens)
        lib_code, lib_len = model.quantize(mels, mels_lens)

    assert frames not in core._s3_graphs
    assert frames in core._s3_failed
    assert _BrokenGraph.instances[0].replays == 1
    # The except branch already served this request the eager result.
    assert torch.equal(fallback_code, lib_code)
    assert torch.equal(fallback_len, lib_len)

    with torch.inference_mode():
        retry_code, retry_len = core._quantize_s3_graph(mels, mels_lens)

    assert torch.equal(retry_code, lib_code)
    assert torch.equal(retry_len, lib_len)
    assert core._s3_eager_calls == 1  # only the retry; the fallback does not count
    assert core._s3_replay_calls == 0
