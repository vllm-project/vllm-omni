# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Guard explicit graph configuration and leave default predictor behavior alone."""

import unittest
from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.common.qwen3_code_predictor import CodePredictorWrapperConfig
from vllm_omni.model_executor.models.drvexo_speech_8b.code_predictor import configure_body_cuda_graphs

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def model(device="cuda"):
    return SimpleNamespace(
        _compiled_model_fwd=None,
        _device_graphs={},
        _prefix_graphs_enabled=False,
        _wrapper_config=CodePredictorWrapperConfig(use_projection=True, sampling_mode="per_call"),
        parameters=lambda: iter([SimpleNamespace(device=SimpleNamespace(type=device))]),
    )


class GraphConfigurationTests(unittest.TestCase):
    def test_default_does_not_change_configuration(self):
        instance = model("cpu")
        original = instance._wrapper_config
        configure_body_cuda_graphs(instance, False)
        self.assertIs(instance._wrapper_config, original)
        self.assertFalse(original.use_cuda_graphs)

    def test_enabling_preserves_sampling_and_projection(self):
        instance = model()
        original = instance._wrapper_config
        configure_body_cuda_graphs(instance, True)
        self.assertFalse(original.use_cuda_graphs)
        self.assertTrue(instance._wrapper_config.use_cuda_graphs)
        self.assertTrue(instance._wrapper_config.use_projection)
        self.assertEqual(instance._wrapper_config.sampling_mode, "per_call")
        self.assertFalse(instance._prefix_graphs_enabled)
        self.assertEqual(instance._device_graphs, {})  # capture is still lazy

    def test_rejects_string_or_numeric_boolean(self):
        for value in ("true", "false", 0, 1, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "boolean"):
                configure_body_cuda_graphs(model(), value)

    def test_rejects_reconfiguration_after_initialization(self):
        for field, value in [("_compiled_model_fwd", object()), ("_device_graphs", {1: object()})]:
            instance = model()
            setattr(instance, field, value)
            with self.assertRaisesRegex(ValueError, "before inference"):
                configure_body_cuda_graphs(instance, True)

    def test_rejects_cpu_and_prefix_graphs(self):
        with self.assertRaisesRegex(ValueError, "CUDA weights"):
            configure_body_cuda_graphs(model("cpu"), True)
        instance = model()
        instance._prefix_graphs_enabled = True
        with self.assertRaisesRegex(ValueError, "prefix-length"):
            configure_body_cuda_graphs(instance, True)


if __name__ == "__main__":
    unittest.main()
