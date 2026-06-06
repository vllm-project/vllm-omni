# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.quantization.tools.compare_diffusion_trajectory_similarity import (
    VariantRun,
    _build_variant_config,
    _request_peak_memory_mb,
    _run_summary,
    compute_tensor_metrics,
    compute_uint8_image_metrics,
    metric_guidance,
    parse_args,
    summarize_output_image_metrics,
)


def test_compute_tensor_metrics_identical_tensors():
    metrics = compute_tensor_metrics(torch.ones(2, 3), torch.ones(2, 3))

    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["max_abs"] == 0.0
    assert metrics["l2"] == 0.0
    assert metrics["relative_l2"] == 0.0


def test_compute_uint8_image_metrics_adds_psnr():
    lhs = np.zeros((2, 2, 3), dtype=np.uint8)
    rhs = np.zeros((2, 2, 3), dtype=np.uint8)

    metrics = compute_uint8_image_metrics(lhs, rhs)

    assert math.isinf(metrics["psnr_db"])


def test_summarize_output_image_metrics_stacks_pil_images():
    reference = [Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))]
    candidate = [Image.fromarray(np.ones((2, 2, 3), dtype=np.uint8))]

    summary = summarize_output_image_metrics(reference, candidate)

    assert summary["num_images"] == 1
    assert summary["image0_metrics"]["mae"] == 1.0
    assert summary["all_images_metrics"]["mse"] == 1.0


def test_run_summary_reports_worker_peak_memory():
    summary = _run_summary(
        VariantRun(
            label="candidate",
            result=object(),
            generation_times_s=[1.0, 3.0],
            peak_memory_mb=[100.0, 150.0],
        )
    )

    assert summary["peak_memory_mb"] == 150.0
    assert summary["avg_peak_memory_mb"] == 125.0
    assert summary["max_peak_memory_mb"] == 150.0
    assert summary["per_run_peak_memory_mb"] == [100.0, 150.0]


def test_request_peak_memory_prefers_inner_diffusion_output():
    inner = type("InnerOutput", (), {"peak_memory_mb": 321.0})()
    outer = type("OuterOutput", (), {"request_output": inner, "peak_memory_mb": 123.0})()

    assert _request_peak_memory_mb(outer) == 321.0


def test_metric_guidance_describes_thresholds():
    guidance = metric_guidance()

    assert "cosine_similarity" in guidance["descriptions"]
    assert guidance["recommended_thresholds"]["output_images_or_frames_uint8"]["psnr_db"]["recommended_min"] == 20.0
    assert (
        guidance["recommended_thresholds"]["performance"]["max_peak_memory_ratio_candidate_over_reference"][
            "recommended_max"
        ]
        == 1.00
    )


def test_candidate_model_can_point_to_offline_checkpoint_without_online_quantization():
    args = SimpleNamespace(
        model="Qwen/Qwen-Image",
        candidate_model="Qwen/Qwen-Image-FP8",
        candidate_quantization=None,
        candidate_quantization_config_json=None,
        candidate_ignored_layers=None,
        ignored_layers=None,
        candidate_gguf_model=None,
        candidate_load_format="default",
    )

    config = _build_variant_config(args, "candidate")

    assert config.model == "Qwen/Qwen-Image-FP8"
    assert config.quantization is None
    assert config.quantization_config is None


def test_step_execution_defaults_to_false(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_diffusion_trajectory_similarity.py",
            "--output-json",
            "result.json",
        ],
    )

    args = parse_args()

    assert args.step_execution is False
