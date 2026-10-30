# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Reject Codec assets that do not match the native loading reference."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.model_executor.models.drvexo_speech_8b import codec_quantization as codec

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CodecSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = torch.nn.Sequential(torch.nn.Linear(4, 4, dtype=torch.float32))
        save_file(self.model.state_dict(), str(self.root / "source.safetensors"))
        save_file({"unused": torch.ones(1)}, str(self.root / "packed.safetensors"))

    def test_source_drift_does_not_mutate_decoder(self):
        original = self.model[0]
        with torch.no_grad():
            original.weight.add_(0.003)
        before = {n: v.clone() for n, v in self.model.state_dict().items()}
        with (
            patch.object(codec, "get_tensor_model_parallel_world_size", return_value=1),
            self.assertRaisesRegex(ValueError, "differs from calibration source"),
        ):
            codec.install_packed_codec_linears(
                self.model, "nvfp4", self.root / "packed.safetensors", self.root / "source.safetensors"
            )
        self.assertIs(self.model[0], original)
        self.assertTrue(all(torch.equal(v, self.model.state_dict()[n]) for n, v in before.items()))

    def test_wrong_native_dtype_rejected(self):
        self.model.to(torch.bfloat16)
        with (
            patch.object(codec, "get_tensor_model_parallel_world_size", return_value=1),
            self.assertRaisesRegex(ValueError, "differs from calibration source"),
        ):
            codec.install_packed_codec_linears(
                self.model, "fp8", self.root / "packed.safetensors", self.root / "source.safetensors"
            )

    def test_unsupported_precision_fails_before_io(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Codec precision"):
            codec.install_packed_codec_linears(self.model, "int4", "missing", "missing")

    def test_non_tp1_fails_before_io(self):
        with (
            patch.object(codec, "get_tensor_model_parallel_world_size", return_value=2),
            self.assertRaisesRegex(ValueError, "TP1"),
        ):
            codec.install_packed_codec_linears(self.model, "fp8", "missing", "missing")

    def test_convolution_transpose_is_contiguous_at_quant_kernel_boundary(self):
        class ContiguousKernel(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, value):
                assert value.is_contiguous() and value.dtype == torch.bfloat16
                return value

        original = torch.nn.Linear(4, 4, dtype=torch.float32)
        with patch.object(codec, "ReplicatedLinear", ContiguousKernel):
            wrapper = codec.CodecLinear(original, None, "test")
        # Same B,T,C layout emitted by Codec pre_conv(...).transpose(1,2).
        strided = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).transpose(1, 2)
        self.assertFalse(strided.is_contiguous())
        actual = wrapper(strided)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, strided.bfloat16().float() + original.bias))

    def test_bf16_boundary_returns_bf16_without_casting_packed_weights(self):
        class PackedKernel(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float8_e4m3fn), requires_grad=False)
                self.scale = torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)

            def forward(self, value):
                assert value.is_contiguous() and value.dtype == torch.bfloat16
                return value

        original = torch.nn.Linear(4, 4, dtype=torch.float32)
        with patch.object(codec, "ReplicatedLinear", PackedKernel):
            wrapper = codec.CodecLinear(original, None, "test", torch.bfloat16)
        value = torch.arange(24).reshape(2, 4, 3).transpose(1, 2).float()
        self.assertEqual(wrapper(value).dtype, torch.bfloat16)
        self.assertEqual(wrapper.linear.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(wrapper.linear.scale.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
