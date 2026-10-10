# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU fault injection for benchmark state ownership; no checkpoint required."""

import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.diffusion.models.seedvr2 import benchmark_dit_cache as b
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

CACHE_ENV = "VLLM_OMNI_SEEDVR2_DIT_CACHE"


@pytest.fixture
def harness(monkeypatch, tmp_path):
    state = SimpleNamespace(calls=0, closes=0, wrappers=[], failure=None, close_failure=False, report_failure=False)
    methods = ((b.vae.SeedVR2VAE, "encode"), (b.nadit.SeedVR2NaDiT, "forward"), (b.vae.SeedVR2VAE, "decode"))

    def encode(self):
        return None

    def forward(self, runtime=None):
        return b.nadit.NaDiTOutput(vid_sample=torch.zeros(1))

    def decode(self):
        return None

    for (owner, name), original in zip(methods, (encode, forward, decode)):
        monkeypatch.setattr(owner, name, original)
    originals = [getattr(owner, name) for owner, name in methods]
    output_path = tmp_path / "results" / "benchmark.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "unused",
            "--output",
            str(output_path),
            "--frames",
            "5",
            "--height",
            "16",
            "--width",
            "16",
            "--rounds",
            "2",
        ],
    )
    monkeypatch.setattr(b.current_omni_platform, "synchronize", lambda: None)
    monkeypatch.setattr(
        b.torch,
        "get_device_module",
        lambda _: SimpleNamespace(
            get_device_properties=lambda _: "CPU fault injection",
            reset_peak_memory_stats=lambda: None,
            max_memory_allocated=lambda: 0,
            max_memory_reserved=lambda: 0,
        ),
    )

    def version(name):
        state.wrappers = [getattr(owner, method) for owner, method in methods]
        if state.failure == "metadata":
            raise RuntimeError("metadata failure")
        return "test"

    monkeypatch.setattr(b.importlib.metadata, "version", version)
    original_mkdir, original_write = Path.mkdir, Path.write_text

    def mkdir(path, *args, **kwargs):
        if path == output_path.parent and state.failure == "directory":
            raise RuntimeError("directory failure")
        return original_mkdir(path, *args, **kwargs)

    def write(path, *args, **kwargs):
        if path == output_path and (state.failure == "report" or state.report_failure):
            raise RuntimeError("report failure")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(Path, "write_text", write)

    class Engine:
        def __init__(self, **kwargs):
            if state.failure == "constructor":
                raise RuntimeError("constructor failure")

        def generate(self, prompt, params, **kwargs):
            state.calls += 1
            if state.failure == "warmup":
                raise RuntimeError("warmup failure")
            b.vae.SeedVR2VAE.encode(self)
            runtime = b.nadit.SeedVR2WindowRuntime((1, 1, 1), text_len=0)
            context = runtime.context(runtime.layout_for_layer(0), "cpu")
            context.sdpa_groups = ()
            context.rotary_cache.builds, context.rotary_cache.hits = 1, 2
            b.nadit.SeedVR2NaDiT.forward(self, runtime=runtime)
            b.vae.SeedVR2VAE.decode(self)
            if state.failure == "inference" and state.calls == 3:
                raise RuntimeError("inference failure")
            return [
                OmniRequestOutput.from_diffusion(
                    request_id="cleanup-test", images=[np.zeros((1, 5, 16, 16, 3), dtype=np.uint8)]
                )
            ]

        def close(self):
            state.closes += 1
            if state.failure == "close" or state.close_failure:
                raise RuntimeError("close failure")

    monkeypatch.setattr(b, "Omni", Engine)
    # Exercise the real timed loop and output checks while excluding model/kernel work.
    state.methods, state.originals, state.output_path = methods, originals, output_path
    return state


def assert_restored(state, original_env, original_cudnn):
    assert [getattr(owner, method) for owner, method in state.methods] == state.originals
    assert os.environ.get(CACHE_ENV) == original_env
    assert torch.backends.cudnn.benchmark is original_cudnn
    # Also check a separately retained wrapper cannot keep its last output alive.
    for wrapper in state.wrappers:
        assert not inspect.getclosurevars(wrapper).nonlocals["captures"]


@pytest.mark.parametrize("failure", ["metadata", "directory", "constructor", "warmup", "inference", "close", "report"])
@pytest.mark.parametrize("original_env", [None, "original"])
def test_cleanup_after_failure(harness, monkeypatch, failure, original_env):
    if original_env is None:
        monkeypatch.delenv(CACHE_ENV, raising=False)
    else:
        monkeypatch.setenv(CACHE_ENV, original_env)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    harness.failure = failure
    # Reference mode also verifies this helper can still run against unmodified main.
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--reference-only"])
    with pytest.raises(RuntimeError, match=f"{failure} failure"):
        b.main()
    assert_restored(harness, original_env, True)
    assert harness.closes == (0 if failure in {"metadata", "directory", "constructor"} else 1)


@pytest.mark.parametrize("original_cudnn", [False, True])
@pytest.mark.parametrize("reference_only", [False, True])
def test_repeated_success_restores_state(harness, monkeypatch, original_cudnn, reference_only):
    monkeypatch.setenv(CACHE_ENV, "original")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", original_cudnn)
    if reference_only:
        monkeypatch.setattr(sys, "argv", [*sys.argv, "--reference-only"])
    for invocation in range(2):
        b.main()
        assert_restored(harness, "original", original_cudnn)
        assert harness.closes == invocation + 1
        report = json.loads(harness.output_path.read_text())
        assert report["status"] == "PASS"
        assert len(report["runs"]) == (2 if reference_only else 4)
        assert all(row["dit_bit_exact"] and row["rgb_bit_exact"] for row in report["runs"])


@pytest.mark.parametrize("report_failure", [False, True])
def test_cleanup_continues_after_multiple_failures(harness, monkeypatch, report_failure):
    monkeypatch.setenv(CACHE_ENV, "original")
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    harness.failure = "inference"
    harness.close_failure = True
    harness.report_failure = report_failure
    expected = "report" if report_failure else "close"
    with pytest.raises(RuntimeError, match=f"{expected} failure") as error:
        b.main()
    assert_restored(harness, "original", True)
    assert harness.closes == 1
    messages = []
    exception = error.value
    while exception is not None:
        messages.append(str(exception))
        exception = exception.__context__
    assert "inference failure" in messages and "close failure" in messages
