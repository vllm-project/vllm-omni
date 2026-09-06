# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.offline_inference.dreamzero import benchmark_prediction_video


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_parse_args_defaults_to_two_generate_requests() -> None:
    args = benchmark_prediction_video.parse_args(
        [
            "--deploy-config",
            "vllm_omni/deploy/dreamzero_tp1_cfg2.yaml",
        ]
    )

    assert args.model == "GEAR-Dreams/DreamZero-DROID"
    assert args.deploy_config == Path("vllm_omni/deploy/dreamzero_tp1_cfg2.yaml")
    assert args.num_requests == 2
    assert args.output_dir == Path("outputs/dreamzero/benchmark")
    assert args.output_stem == "dreamzero_benchmark"
    assert args.fps == 5
    assert args.accuracy_atol == pytest.approx(1e-3)
    assert args.profiler_config is None
    assert args.profile_request_index == 1


def test_parse_args_rejects_non_positive_request_count() -> None:
    with pytest.raises(SystemExit):
        benchmark_prediction_video.parse_args(
            [
                "--deploy-config",
                "vllm_omni/deploy/dreamzero.yaml",
                "--num-requests",
                "0",
            ]
        )


def test_parse_args_accepts_profiler_config_json() -> None:
    args = benchmark_prediction_video.parse_args(
        [
            "--deploy-config",
            "vllm_omni/deploy/dreamzero_tp1_cfg2.yaml",
            "--profiler-config",
            '{"profiler":"torch","torch_profiler_dir":"/tmp/dreamzero-profile"}',
            "--profile-request-index",
            "2",
        ]
    )

    assert args.profiler_config == {
        "profiler": "torch",
        "torch_profiler_dir": "/tmp/dreamzero-profile",
    }
    assert args.profile_request_index == 2


def test_parse_args_rejects_non_object_profiler_config() -> None:
    with pytest.raises(SystemExit):
        benchmark_prediction_video.parse_args(
            [
                "--deploy-config",
                "vllm_omni/deploy/dreamzero.yaml",
                "--profiler-config",
                '["torch"]',
            ]
        )


def test_run_generation_profiles_only_requested_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    instances: list[object] = []

    class FakeOmni:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            instances.append(self)

        def start_profile(self):
            calls.append("start_profile")

        def stop_profile(self):
            calls.append("stop_profile")
            return {"traces": ["trace_rank0.json"]}

        def generate(self, prompt, sampling_params_list):
            calls.append(f"generate:{prompt}")
            return [
                SimpleNamespace(
                    images=[torch.zeros((1, 16, 1, 2, 2), dtype=torch.float32)],
                    multimodal_output={
                        "actions": np.zeros(
                            (
                                benchmark_prediction_video.ACTION_HORIZON,
                                8,
                            ),
                            dtype=np.float32,
                        )
                    },
                )
            ]

    class FakeSamplingParams:
        def __init__(self, *, extra_args):
            self.extra_args = extra_args

    monkeypatch.setitem(sys.modules, "vllm_omni", SimpleNamespace(Omni=FakeOmni))
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.inputs.data",
        SimpleNamespace(OmniDiffusionSamplingParams=FakeSamplingParams),
    )

    observations = [
        {"prompt": "initial", "session_id": "session-1"},
        {"prompt": "chunk", "session_id": "session-1"},
    ]

    _, _, _, request_timings = benchmark_prediction_video._run_generation(
        model="model",
        deploy_config_path=Path("deploy.yaml"),
        observations=observations,
        profiler_config={"profiler": "torch"},
        profile_request_index=1,
    )

    assert calls == [
        "generate:initial",
        "start_profile",
        "generate:chunk",
        "stop_profile",
    ]
    assert instances[0].kwargs["profiler_config"] == {"profiler": "torch"}
    assert request_timings[1].label == "chunk_0"
