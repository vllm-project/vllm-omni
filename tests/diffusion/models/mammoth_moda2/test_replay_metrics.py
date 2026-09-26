# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from benchmarks.mammoth_moda2.compare_replays import image_metrics, metrics

pytestmark = [pytest.mark.advanced_model, pytest.mark.cpu]


def test_replay_metrics_identical_bf16_tensors() -> None:
    reference = torch.arange(10000, dtype=torch.bfloat16).reshape(1, 10000)
    result = metrics(reference, reference)
    assert result["exact"]
    assert result["max_abs"] == result["mean_abs"] == result["rms_relative"] == 0
    assert result["cosine"] == pytest.approx(1.0, abs=1e-12)


def test_replay_metrics_zero_and_known_delta() -> None:
    reference = torch.zeros(4)
    candidate = torch.ones(4)
    assert metrics(reference, reference)["cosine"] is None
    result = metrics(reference, candidate)
    assert not result["exact"]
    assert result["mean_abs"] == result["max_abs"] == 1.0


def test_replay_image_metrics_identity() -> None:
    image = torch.linspace(-1, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    result = image_metrics(image, image)
    assert result["normalized_rgb_mae"] == 0.0
    assert result["psnr_db"] == "infinite"
    assert result["ssim_11x11_sigma1_5"] == pytest.approx(1.0, abs=1e-6)
