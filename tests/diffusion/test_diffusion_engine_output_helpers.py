# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DiffusionEngine output helper functions."""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest
import torch
from pytest_mock import MockerFixture

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _load_engine_module(mocker: MockerFixture):
    """Load diffusion_engine.py with mocked heavy dependencies."""
    engine_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "vllm_omni",
            "diffusion",
            "diffusion_engine.py",
        )
    )

    mocks = {
        "vllm_omni": mocker.MagicMock(),
        "vllm_omni.diffusion": types.ModuleType("vllm_omni.diffusion"),
        "vllm_omni.diffusion.data": mocker.MagicMock(),
        "vllm_omni.diffusion.executor": mocker.MagicMock(),
        "vllm_omni.diffusion.executor.abstract": mocker.MagicMock(),
        "vllm_omni.diffusion.registry": mocker.MagicMock(),
        "vllm_omni.diffusion.request": mocker.MagicMock(),
        "vllm_omni.diffusion.sched": mocker.MagicMock(),
        "vllm_omni.diffusion.sched.interface": mocker.MagicMock(),
        "vllm_omni.diffusion.worker": mocker.MagicMock(),
        "vllm_omni.diffusion.worker.utils": mocker.MagicMock(),
        "vllm_omni.inputs": mocker.MagicMock(),
        "vllm_omni.inputs.data": mocker.MagicMock(),
        "vllm_omni.outputs": mocker.MagicMock(),
        "vllm": types.ModuleType("vllm"),
        "vllm.v1": types.ModuleType("vllm.v1"),
        "vllm.v1.engine": types.ModuleType("vllm.v1.engine"),
        "vllm.v1.engine.exceptions": mocker.MagicMock(),
        "vllm.logger": mocker.MagicMock(init_logger=lambda name: mocker.MagicMock()),
        "PIL": mocker.MagicMock(),
        "PIL.Image": mocker.MagicMock(),
    }
    mocker.patch.dict(sys.modules, mocks)

    spec = importlib.util.spec_from_file_location("diffusion_engine", engine_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def engine_mod(mocker: MockerFixture):
    return _load_engine_module(mocker)


@pytest.fixture
def move_tensors_to_cpu(engine_mod):
    return engine_mod._move_tensors_to_cpu


@pytest.fixture
def normalize_outputs(engine_mod):
    return engine_mod._normalize_outputs


class TestMoveTensorsToCpu:
    """Test recursive CPU offload helper."""

    def test_single_tensor(self, move_tensors_to_cpu) -> None:
        t = torch.randn(2, 3)
        result = move_tensors_to_cpu(t)
        assert result.device.type == "cpu"
        assert torch.equal(result, t)

    def test_already_on_cpu(self, move_tensors_to_cpu) -> None:
        t = torch.randn(2, 3)
        result = move_tensors_to_cpu(t)
        assert result is t

    def test_dict_with_tensors(self, move_tensors_to_cpu) -> None:
        data = {"video": torch.randn(2, 3), "audio": torch.randn(4), "label": "test"}
        result = move_tensors_to_cpu(data)
        assert isinstance(result, dict)
        assert result["video"].device.type == "cpu"
        assert result["audio"].device.type == "cpu"
        assert result["label"] == "test"

    def test_tuple_with_tensors(self, move_tensors_to_cpu) -> None:
        data = (torch.randn(2, 3), torch.randn(4))
        result = move_tensors_to_cpu(data)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].device.type == "cpu"
        assert result[1].device.type == "cpu"

    def test_list_with_tensors(self, move_tensors_to_cpu) -> None:
        data = [torch.randn(2, 3), torch.randn(4)]
        result = move_tensors_to_cpu(data)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_nested_structure(self, move_tensors_to_cpu) -> None:
        data = {"outputs": (torch.randn(2, 3), {"inner": torch.randn(4)})}
        result = move_tensors_to_cpu(data)
        assert result["outputs"][1]["inner"].device.type == "cpu"

    def test_non_tensor_passthrough(self, move_tensors_to_cpu) -> None:
        assert move_tensors_to_cpu(42) == 42
        assert move_tensors_to_cpu("hello") == "hello"
        assert move_tensors_to_cpu(None) is None


class TestBatchedOutputSlicing:
    """Test that batched tensor outputs are properly split per-prompt."""

    def test_batched_tensor_is_split(self, normalize_outputs) -> None:
        outputs = torch.randn(3, 16000)
        normalized = normalize_outputs(outputs, expected_items=3)
        assert len(normalized) == 3
        assert normalized[0].shape == (16000,)

    def test_single_sample_not_split(self, normalize_outputs) -> None:
        outputs = torch.randn(1, 16000)
        normalized = normalize_outputs(outputs, expected_items=1)
        assert len(normalized) == 1
        assert normalized[0].shape == (1, 16000)

    def test_numpy_batched_is_split(self, normalize_outputs) -> None:
        outputs = np.random.randn(3, 16000)
        normalized = normalize_outputs(outputs, expected_items=3)
        assert len(normalized) == 3

    def test_list_passthrough(self, normalize_outputs) -> None:
        outputs = [torch.randn(16000), torch.randn(16000)]
        normalized = normalize_outputs(outputs, expected_items=2)
        assert normalized is outputs

    def test_shape_mismatch_is_not_split(self, normalize_outputs) -> None:
        outputs = torch.randn(16000)
        normalized = normalize_outputs(outputs, expected_items=2)
        assert len(normalized) == 1
        assert normalized[0].shape == (16000,)
