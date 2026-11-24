# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.utils import flow_matching

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_safe_linalg_solve_uses_torch_native_path(monkeypatch: pytest.MonkeyPatch):
    matrix = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    rhs = torch.tensor([9.0, 8.0], dtype=torch.float32)
    expected = torch.tensor([2.0, 3.0], dtype=torch.float32)

    def fake_torch_solve(solve_matrix: torch.Tensor, solve_rhs: torch.Tensor) -> torch.Tensor:
        assert torch.equal(solve_matrix, matrix)
        assert torch.equal(solve_rhs, rhs)
        return expected

    def fail_numpy(*_args, **_kwargs):
        raise AssertionError("NumPy fallback should not run")

    monkeypatch.setattr(flow_matching.torch.linalg, "solve", fake_torch_solve)
    monkeypatch.setattr(flow_matching.np.linalg, "solve", fail_numpy)

    result = flow_matching.safe_linalg_solve(matrix, rhs)

    assert result is expected


def test_safe_linalg_solve_falls_back_on_missing_cpu_lapack(monkeypatch: pytest.MonkeyPatch):
    matrix = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    rhs = torch.tensor([9.0, 8.0], dtype=torch.float32)
    expected = np.array([2.0, 3.0], dtype=np.float32)

    def fail_with_missing_lapack(*_args, **_kwargs):
        raise RuntimeError(
            "Calling torch.linalg.lu_factor on a CPU tensor requires compiling PyTorch with LAPACK. "
            "Please use PyTorch built with LAPACK support."
        )

    monkeypatch.setattr(flow_matching.torch.linalg, "solve", fail_with_missing_lapack)
    monkeypatch.setattr(flow_matching.np.linalg, "solve", lambda *_args, **_kwargs: expected)

    result = flow_matching.safe_linalg_solve(matrix, rhs)

    assert torch.equal(result, torch.from_numpy(expected))
    assert result.dtype == matrix.dtype
    assert result.device == matrix.device


def test_safe_linalg_solve_reraises_unrelated_runtime_error(monkeypatch: pytest.MonkeyPatch):
    matrix = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
    rhs = torch.tensor([9.0, 8.0], dtype=torch.float32)

    def fail_with_unrelated_error(*_args, **_kwargs):
        raise RuntimeError("solver exploded for another reason")

    def fail_numpy(*_args, **_kwargs):
        raise AssertionError("NumPy fallback should not run")

    monkeypatch.setattr(flow_matching.torch.linalg, "solve", fail_with_unrelated_error)
    monkeypatch.setattr(flow_matching.np.linalg, "solve", fail_numpy)

    with pytest.raises(RuntimeError, match="solver exploded for another reason"):
        flow_matching.safe_linalg_solve(matrix, rhs)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "shift must be a number, not a boolean, got True."),
        ("abc", "shift must be a number, got 'abc'."),
        (None, "shift must be a number, got None."),
        pytest.param(10**400, "shift is too large to represent as a float.", id="huge-int"),
        ("nan", "shift must be a finite float, got 'nan'."),
        (float("-inf"), "shift must be a finite float, got -inf."),
        ("1e400", "shift must be a finite float, got '1e400'."),
        (0.0, "shift must be greater than 0, got 0.0."),
        (-2, "shift must be greater than 0, got -2."),
    ],
)
def test_normalize_shift_raises_value_error_with_specific_message(value: object, message: str):
    with pytest.raises(ValueError) as exc_info:
        flow_matching.normalize_shift(value, name="shift")

    assert type(exc_info.value) is ValueError
    assert str(exc_info.value) == message


@pytest.mark.parametrize(("value", "expected"), [(0.25, 0.25), ("3.0", 3.0), (12, 12.0)])
def test_normalize_shift_returns_positive_finite_float(value: object, expected: float):
    assert flow_matching.normalize_shift(value, name="shift") == expected
