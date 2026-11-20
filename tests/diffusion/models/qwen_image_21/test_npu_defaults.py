# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image_21 import npu_attention
from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import QwenImage21Transformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "case,expected",
    [
        ("default", "fusion"),
        ("masked", "sdpa"),
        ("fp32", "sdpa"),
        ("batch", "sdpa"),
        ("graph", "sdpa"),
        ("grad", "sdpa"),
        ("compile", "sdpa"),
        ("gqa", "sdpa"),
        ("tp", "sdpa"),
        ("sp", "sdpa"),
    ],
)
def test_npu_default_dispatch_and_fallback(monkeypatch, case, expected):
    monkeypatch.delenv("QWEN21_FUSION_PRECISE", raising=False)
    config = OmniDiffusionConfig(enforce_eager=case != "graph")
    if case == "tp":
        config.parallel_config.tensor_parallel_size = 2
    if case == "sp":
        config.parallel_config.ulysses_degree = 2
        config.parallel_config.sequence_parallel_size = 2
    with set_current_diffusion_config(config):
        impl = npu_attention.QwenImage21SDPAImpl(2, 8, 8**-0.5)
    calls = []
    output = torch.empty(0)

    def fusion(query, key, value, metadata, scale, precise):
        calls.append("fusion")
        assert precise == 0
        return output

    def fallback(self, query, key, value, metadata=None):
        calls.append("sdpa")
        return output

    monkeypatch.setattr(npu_attention, "_npu_fusion_attention", fusion)
    monkeypatch.setattr(SDPAImpl, "forward_npu", fallback)
    if case == "compile":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    q = torch.randn(2 if case == "batch" else 1, 4, 2, 8, dtype=torch.float32 if case == "fp32" else torch.bfloat16)
    k = q[:, :, :1] if case == "gqa" else q
    metadata = AttentionMetadata(attn_mask=torch.ones(4, 4, dtype=torch.bool)) if case == "masked" else None
    # CPU tensors test dispatch only; NPU kernels are covered by the hardware A/B.
    with torch.set_grad_enabled(case == "grad"):
        assert impl.forward_npu(q, k, k, metadata) is output
    assert calls == [expected]


@pytest.mark.parametrize("supported", [False, True])
def test_nz_defaults_skip_unsupported_devices(supported):
    model = QwenImage21Transformer2DModel.__new__(QwenImage21Transformer2DModel)
    nn.Module.__init__(model)
    model._nz_weights_supported = supported
    model._enforce_eager = False
    model._nz_runs_eager = False
    model.enable_cuda_graph_decode = False
    model.parallel_config = OmniDiffusionConfig().parallel_config
    assert model.load_weights([]) == set()


@pytest.mark.parametrize("case", ["default", "quant", "tp", "hsdp", "lora"])
def test_nz_support_excludes_post_load_weight_rewrites(case):
    config = OmniDiffusionConfig()
    if case == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif case == "hsdp":
        config.parallel_config.use_hsdp = True
    elif case == "lora":
        config.lora_path = "distilled-lora"
    quant_config = object() if case == "quant" else None
    supported = QwenImage21Transformer2DModel._nz_weights_supported_for(config, quant_config)
    assert supported is (case == "default")


@pytest.mark.parametrize(
    ("npu_eager", "enforce_eager", "graph_decode", "sequence_parallel_size", "expected"),
    [
        # Default NPU manager cannot capture; decode stays eager and converts.
        (True, False, True, 1, True),
        # Graph off without enforce_eager is still eager on NPU.
        (True, False, False, 1, True),
        (True, True, True, 1, True),
        (True, False, True, 2, False),
        # Non-NPU: disabling the graph flag is not an eager request.
        (False, False, False, 1, False),
        (False, True, False, 1, True),
        (False, False, True, 1, False),
    ],
)
def test_automatic_nz_follows_eager_execution(npu_eager, enforce_eager, graph_decode, sequence_parallel_size, expected):
    model = QwenImage21Transformer2DModel.__new__(QwenImage21Transformer2DModel)
    model._nz_runs_eager = npu_eager
    model._enforce_eager = enforce_eager
    model.enable_cuda_graph_decode = graph_decode
    model.parallel_config = OmniDiffusionConfig().parallel_config
    model.parallel_config.sequence_parallel_size = sequence_parallel_size
    assert model._automatic_nz_eligible() is expected
