# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import subprocess
from types import SimpleNamespace

import pytest
import torch

from benchmarks.diffusion.bench_mammoth_vae_patch_parallel import (
    center_stripe_error_profile,
    comparison_modes,
    error_metrics,
    json_safe,
    prepare_memory_measurement,
    source_revision,
    tile_boundary_error_profile,
    tile_boundary_evidence,
    tile_rank_layout,
    validate_decode_output,
)
from vllm_omni.diffusion.distributed.autoencoders.distributed_vae_executor import (
    DistributedVaeExecutor,
    GridSpec,
    TileTask,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_source_revision_without_git_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(subprocess, "check_output", missing_git)
    monkeypatch.setenv("VLLM_OMNI_SOURCE_COMMIT", "test-commit")
    assert source_revision() == ("test-commit", ["Git metadata unavailable"])


def test_error_metrics_for_equal_outputs() -> None:
    result = error_metrics(torch.ones(1, 3, 2, 2), torch.ones(1, 3, 2, 2))
    assert result["max_abs"] == 0
    assert result["mean_abs"] == 0
    assert result["relative_l2"] == 0
    assert result["psnr_db"] == float("inf")


def test_validate_decode_output_checks_original_dtype_shape_and_finiteness() -> None:
    valid = torch.zeros((1, 3, 16, 16), dtype=torch.float16)
    validate_decode_output(valid, expected_shape=(1, 3, 16, 16), expected_dtype=torch.float16)
    with pytest.raises(ValueError, match="shape"):
        validate_decode_output(valid, expected_shape=(1, 3, 8, 8), expected_dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        validate_decode_output(valid, expected_shape=(1, 3, 16, 16), expected_dtype=torch.bfloat16)
    invalid = valid.clone()
    invalid[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_decode_output(invalid, expected_shape=(1, 3, 16, 16), expected_dtype=torch.float16)


def test_error_metrics_for_known_difference() -> None:
    result = error_metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 3.0]))
    assert result["max_abs"] == 1
    assert result["mean_abs"] == 0.5
    assert result["relative_l2"] == pytest.approx(1 / 5**0.5)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (torch.ones(2), torch.ones(3)),
        (torch.tensor([float("nan")]), torch.ones(1)),
        (torch.ones(1), torch.tensor([float("inf")])),
    ],
)
def test_error_metrics_reject_invalid_inputs(left: torch.Tensor, right: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        error_metrics(left, right)


def test_center_stripe_error_profile_separates_center_from_off_center() -> None:
    reference = torch.zeros(1, 1, 2, 4)
    actual = reference.clone()
    actual[..., 1:3] = 1

    result = center_stripe_error_profile(reference, actual, half_width=1)

    assert result["center_mean_abs"] == 1
    assert result["off_center_mean_abs"] == 0


def test_tile_boundary_profile_uses_actual_grid_and_rank_assignments() -> None:
    reference = torch.zeros(1, 1, 7, 11)
    actual = reference.clone()
    actual[..., :, 4:6] = 2
    actual[..., 3:5, :] = 1

    result = tile_boundary_error_profile(
        reference,
        actual,
        grid_shape=(2, 3),
        row_limit=4,
        rank_grid=((0, 1, 1), (1, 0, 0)),
        half_width=1,
    )

    assert result["vertical_boundaries"] == [4, 8]
    assert result["horizontal_boundaries"] == [4]
    assert result["cross_rank"]["pixels"] == 32
    assert result["same_rank"]["pixels"] == 10
    assert result["interior"]["pixels"] == 35
    assert result["cross_rank"]["mean_abs"] > result["interior"]["mean_abs"]


def test_tile_boundary_profile_odd_output_matches_recorded_geometry() -> None:
    output = torch.zeros(1, 1, 2056, 2056)
    result = tile_boundary_error_profile(
        output,
        output,
        grid_shape=(3, 3),
        row_limit=768,
        rank_grid=((0, 1, 0), (0, 1, 1), (0, 1, 0)),
        half_width=16,
    )

    assert result["vertical_boundaries"] == [768, 1536]
    assert result["horizontal_boundaries"] == [768, 1536]
    assert result["cross_rank"]["pixels"] == 139776
    assert result["same_rank"]["pixels"] == 119296
    assert result["interior"]["pixels"] == 3968064


def test_tile_boundary_profile_rejects_wrong_rank_grid() -> None:
    output = torch.zeros(1, 1, 7, 11)
    with pytest.raises(ValueError, match="rank_grid"):
        tile_boundary_error_profile(output, output, grid_shape=(2, 3), row_limit=4, rank_grid=((0, 1),))


def test_tile_rank_layout_follows_actual_balancer_assignment() -> None:
    latent = torch.zeros(1, 1, 7, 11)
    tasks = [TileTask(idx, (idx // 3, idx % 3), latent, workload=1) for idx in range(6)]
    spec = GridSpec(split_dims=(2, 3), grid_shape=(2, 3), tile_spec={"row_limit": 4})
    vae = SimpleNamespace(
        tile_split=lambda _: (tasks, spec),
        distributed_executor=DistributedVaeExecutor.__new__(DistributedVaeExecutor),
    )

    assert tile_rank_layout(vae, latent, 2) == ((2, 3), 4, ((0, 1, 0), (1, 0, 1)))


def test_tile_boundary_evidence_records_actual_layout_and_error() -> None:
    latent = torch.zeros(1, 1, 7, 11)
    tasks = [TileTask(idx, (idx // 3, idx % 3), latent, workload=1) for idx in range(6)]
    spec = GridSpec(split_dims=(2, 3), grid_shape=(2, 3), tile_spec={"row_limit": 4})
    vae = SimpleNamespace(
        tile_split=lambda _: (tasks, spec),
        distributed_executor=DistributedVaeExecutor.__new__(DistributedVaeExecutor),
    )
    reference = torch.zeros(1, 1, 7, 11)
    actual = reference.clone()
    actual[..., :, 4:6] = 1

    result = tile_boundary_evidence(vae, latent, reference, actual, pp_size=2, half_width=1)

    assert result["grid_shape"] == [2, 3]
    assert result["row_limit"] == 4
    assert result["rank_grid"] == [[0, 1, 0], [1, 0, 1]]
    assert result["error_vs_tiled"]["vertical_boundaries"] == [4, 8]
    assert result["error_vs_tiled"]["cross_rank"]["mean_abs"] > 0


def test_comparison_modes_include_tiled_single_gpu_control() -> None:
    assert comparison_modes() == [(1, False), (1, True), (2, True)]


def test_json_safe_replaces_nonfinite_metrics() -> None:
    assert json_safe({"metric": [float("inf"), float("nan"), 0.5]}) == {"metric": [None, None, 0.5]}


def test_memory_measurement_retains_allocator_cache_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda: calls.append("reset"))

    prepare_memory_measurement()

    assert calls == ["reset"]


def test_memory_measurement_can_clear_cache_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: calls.append("empty"))
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", lambda: calls.append("reset"))

    prepare_memory_measurement(clear_cache=True)

    assert calls == ["empty", "reset"]
