# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DiffusionParallelConfig warnings."""

import logging

import pytest

from vllm_omni.diffusion.data import DiffusionParallelConfig

pytestmark = [pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu, pytest.mark.core_model]


def _capture_data_warnings(caplog, fn):
    """Run fn() with caplog capturing vllm_omni.diffusion.data logger output."""
    target_logger = logging.getLogger("vllm_omni.diffusion.data")
    target_logger.addHandler(caplog.handler)
    prev_level = target_logger.level
    target_logger.setLevel(logging.WARNING)
    try:
        fn()
    finally:
        target_logger.removeHandler(caplog.handler)
        target_logger.setLevel(prev_level)


class TestTPWithoutUSPWarning:
    """DiffusionParallelConfig should warn when TP > 1 and USP is not used."""

    def test_tp2_without_usp_warns(self, caplog):
        _capture_data_warnings(
            caplog,
            lambda: DiffusionParallelConfig(tensor_parallel_size=2),
        )
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "tensor_parallel_size=2" in m and "usp" in m.lower() for m in msgs
        ), f"Expected TP-without-USP warning; got: {msgs}"

    def test_tp4_without_usp_warns(self, caplog):
        _capture_data_warnings(
            caplog,
            lambda: DiffusionParallelConfig(tensor_parallel_size=4),
        )
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "tensor_parallel_size=4" in m and "usp" in m.lower() for m in msgs
        ), f"Expected TP-without-USP warning; got: {msgs}"

    def test_tp1_no_warning(self, caplog):
        _capture_data_warnings(
            caplog,
            lambda: DiffusionParallelConfig(tensor_parallel_size=1),
        )
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("tensor_parallel_size" in m for m in msgs), (
            f"Unexpected TP warning for TP=1; got: {msgs}"
        )

    def test_tp2_with_usp2_no_warning(self, caplog):
        _capture_data_warnings(
            caplog,
            lambda: DiffusionParallelConfig(tensor_parallel_size=2, ulysses_degree=2),
        )
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("tensor_parallel_size" in m and "usp" in m.lower() for m in msgs), (
            f"Unexpected TP warning when USP is also set; got: {msgs}"
        )
