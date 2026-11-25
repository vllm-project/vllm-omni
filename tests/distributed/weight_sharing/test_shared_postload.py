# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for mixed private/shared finalization and router precision."""

import sys
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from tests.distributed.weight_sharing.helpers import load_class_methods

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TRANSFORMER = "vllm_omni/diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py"


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("cached_meta", [False, True])
def test_meta_attention_initializes_only_private_rope_on_execution_device(shared, cached_meta):
    cached_rope = None

    def get_rope(**kwargs):
        nonlocal cached_rope
        if cached_rope is None:
            cached_rope = nn.Module()
            cached_rope.register_buffer("cos_sin_cache", torch.arange(8).float().cos(), persistent=False)
        return cached_rope

    if cached_meta:
        with torch.device("meta"):
            get_rope()

    attention_cls = load_class_methods(
        TRANSFORMER,
        "HunYuanAttention",
        ["__init__"],
        {
            "torch": torch,
            "nullcontext": nullcontext,
            "get_tensor_model_parallel_world_size": lambda: 1,
            "get_rope": get_rope,
            "QKVParallelLinear": lambda **_: nn.Linear(4, 4, bias=False),
            "RowParallelLinear": lambda **_: nn.Linear(4, 4, bias=False),
            **{
                name: lambda **_: nn.Identity()
                for name in ("Attention", "ImageKVCacheManager", "HunYuanRotary2DEmbedder")
            },
        },
        base_class=nn.Module,
    )
    config = SimpleNamespace(num_key_value_heads=1, use_qk_norm=False)
    if shared and cached_meta:
        with torch.device("meta"), pytest.raises(ValueError, match="cached meta RoPE"):
            attention_cls(config, 4, 1, 1, rope_device=torch.device("cpu"))
        return
    with torch.device("meta"):
        layers = [attention_cls(config, 4, 1, 1, rope_device=torch.device("cpu") if shared else None) for _ in range(2)]
    for layer in layers:
        assert layer.qkv_proj.weight.is_meta and layer.o_proj.weight.is_meta
        cache = layer.rotary_emb.cos_sin_cache
        assert cache.device.type == ("cpu" if shared else "meta")
        if shared:
            torch.testing.assert_close(cache, torch.arange(8).float().cos())
    assert layers[0].rotary_emb is layers[1].rotary_emb


@pytest.mark.parametrize("fail", [False, True])
def test_shared_kernel_init_skips_repack_and_restores_context(monkeypatch, fail):
    processed = ContextVar("processed", default=False)

    @contextmanager
    def weights_already_processed():
        token = processed.set(True)
        try:
            yield
        finally:
            processed.reset(token)

    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.utils", SimpleNamespace(weights_already_processed=weights_already_processed)
    )

    class Method:
        def __init__(self):
            self.kernel = None
            self.preprocessed = None

        def process_weights_after_loading(self, module):
            self.preprocessed = processed.get()
            if self.preprocessed and fail:
                raise RuntimeError("kernel init failed")
            if not self.preprocessed:
                with torch.no_grad():
                    module.weight.add_(10)
            self.kernel = object()

    model = nn.Module()
    for name in ("shared", "private"):
        layer = nn.Linear(4, 4, bias=False)
        layer.quant_method = Method()
        model.add_module(name, layer)
    shared_before = model.shared.weight.detach().clone()
    private_before = model.private.weight.detach().clone()
    pointer = model.shared.weight.data_ptr()
    # Even a same-device .to() must not be part of shared finalization.
    model.shared.to = Mock(side_effect=AssertionError("shared module was moved"))
    loader_cls = load_class_methods(
        "vllm_omni/diffusion/model_loader/diffusers_loader.py",
        "DiffusersPipelineLoader",
        ["_process_weights_after_loading"],
        {"QuantizeMethodBase": Method},
    )
    loader = loader_cls()
    loader._has_online_quant = lambda _: False

    def finalize():
        loader._process_weights_after_loading(
            model, torch.device("cpu"), preprocessed_module_names=frozenset({"shared"})
        )

    if fail:
        with pytest.raises(RuntimeError, match="kernel init failed"):
            finalize()
        assert model.private.quant_method.kernel is None
    else:
        finalize()
        assert model.shared.quant_method.kernel is not None
        assert model.shared.quant_method.preprocessed is True
        assert model.private.quant_method.preprocessed is False
        torch.testing.assert_close(model.private.weight, private_before + 10)
    assert not processed.get()
    assert model.shared.weight.data_ptr() == pointer
    torch.testing.assert_close(model.shared.weight, shared_before, rtol=0, atol=0)


class Router(nn.Linear):
    def __init__(self, inp, out, *, bias, params_dtype, **kwargs):
        super().__init__(inp, out, bias=bias, dtype=params_dtype)

    def forward(self, x):
        self.input_dtype = x.dtype
        self.autocast_enabled = torch.is_autocast_enabled("cpu")
        result = super().forward(x)
        self.output_dtype = result.dtype
        return result, None


class Experts(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, *, hidden_states, router_logits):
        self.input_dtype = hidden_states.dtype
        return hidden_states


@pytest.fixture
def moe_block_cls():
    return load_class_methods(
        TRANSFORMER,
        "HunYuanSparseMoeBlock",
        ["__init__", "forward"],
        {
            "torch": torch,
            "ReplicatedLinear": Router,
            "FusedMoE": Experts,
            "get_tensor_model_parallel_world_size": lambda: 1,
            "get_current_vllm_config": lambda: SimpleNamespace(
                parallel_config=SimpleNamespace(enable_expert_parallel=False)
            ),
        },
        base_class=nn.Module,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_router_always_runs_in_fp32(moe_block_cls, dtype):
    config = SimpleNamespace(
        num_experts=2, moe_topk=1, hidden_size=4, intermediate_size=8, moe_intermediate_size=None, use_mixed_mlp_moe=0
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        block = moe_block_cls(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    assert block.gate.weight.dtype == torch.float32
    x = torch.randn(2, 4, dtype=dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = block(x)
        assert torch.is_autocast_enabled("cpu")
    assert block.gate.input_dtype == torch.float32
    assert block.gate.autocast_enabled is False
    assert block.gate.output_dtype == torch.float32
    assert block.experts.input_dtype == dtype
    assert output.dtype == dtype


def test_backbone_propagates_fp32_router_through_all_layers(moe_block_cls):
    namespace = moe_block_cls.__init__.__globals__
    namespace.update(
        logger=Mock(),
        get_local_device=lambda: torch.device("cpu"),
        get_pp_group=lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
        VocabParallelEmbedding=lambda n, d, **_: nn.Embedding(n, d),
        RMSNorm=lambda *args, **kwargs: nn.Identity(),
        HunYuanAttention=Mock(side_effect=lambda **_: nn.Identity()),
        HunyuanImagePreprocessor=nn.Identity,
        UnifiledCat=nn.Identity,
        HunyuanImagePostprocessor=nn.Identity,
        _get_cla_factor=lambda _: 1,
        AttentionType=SimpleNamespace(DECODER="decoder", ENCODER_DECODER="cross"),
        make_layers=lambda n, factory, prefix: (0, n, nn.ModuleList(factory(f"{prefix}.{i}") for i in range(n))),
    )
    load_class_methods(TRANSFORMER, "HunyuanImage3DecoderLayer", ["__init__"], namespace, base_class=nn.Module)
    model_cls = load_class_methods(TRANSFORMER, "HunyuanImage3Model", ["__init__"], namespace, base_class=nn.Module)
    config = SimpleNamespace(
        pad_token_id=0,
        vocab_size=16,
        num_hidden_layers=2,
        hidden_size=4,
        num_attention_heads=1,
        intermediate_size=8,
        num_experts=2,
        moe_topk=1,
        moe_intermediate_size=None,
        use_mixed_mlp_moe=0,
        moe_layer_num_skipped=0,
        rms_norm_eps=1e-6,
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = model_cls(config, rope_device=torch.device("cpu"))
    finally:
        torch.set_default_dtype(previous_dtype)
    for layer in model.layers:
        assert layer.mlp.gate.weight.dtype == torch.float32
        assert layer.mlp.gate.weight.device.type == "meta"
    for call in namespace["HunYuanAttention"].call_args_list:
        assert call.kwargs["rope_device"] == torch.device("cpu")


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("meta", [False, True])
def test_pipeline_router_dtype_is_fp32_regardless_of_sharing_or_meta(monkeypatch, shared, meta):
    class PipelineBase(nn.Module):
        def __init__(self, config):
            super().__init__()

    # Stop at the backbone constructor; unrelated VAE/tokenizer loading is outside this contract.
    backbone = Mock(side_effect=RuntimeError("backbone constructor reached"))
    config = SimpleNamespace()
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.distributed.weight_sharing.adapters.hunyuan_image3",
        SimpleNamespace(validate_supported_layout=lambda _: None),
    )
    cls = load_class_methods(
        "vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py",
        "HunyuanImage3Pipeline",
        ["__init__"],
        {
            "torch": torch,
            "get_config": lambda *args, **kwargs: config,
            "_config_flag": lambda *args: False,
            "GenerationConfig": Mock(),
            "DiffusersPipelineLoader": Mock(),
            "HunyuanImage3Model": backbone,
            "get_local_device": lambda: torch.device("cpu"),
            # The attention schedule startup note is outside this contract.
            "_log_step0_profile_reuse": lambda _: None,
        },
        base_class=PipelineBase,
    )
    od_config = SimpleNamespace(
        model="unused",
        revision="revision",
        quantization_config=None,
        weight_sharing=SimpleNamespace(role="consumer") if shared else None,
    )
    with pytest.raises(RuntimeError, match="backbone constructor reached"):
        cls(od_config, backbone_meta=meta)
    assert backbone.call_args.kwargs["shared_router_dtype"] == torch.float32
    assert backbone.call_args.kwargs.get("rope_device") == (torch.device("cpu") if meta else None)


def test_shared_load_rejects_merged_lora_before_ipc_import():
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

    loader = object.__new__(DiffusersPipelineLoader)
    loader.od_config = SimpleNamespace(lora_application="merged")
    loader.quant_config = None
    loader.parallel_config = SimpleNamespace(use_hsdp=False)
    loader._init_from_load_format = Mock()

    provider_weight = torch.randn(8)
    provider_snapshot = provider_weight.clone()
    provider_ptr = provider_weight.data_ptr()
    backend = Mock()

    with pytest.raises(ValueError, match="merged LoRA"):
        loader._load_model_with_shared_backbone(
            shared_group=SimpleNamespace(role="consumer"),
            custom_pipeline_name=None,
            target_device=torch.device("cpu"),
            backend=backend,
        )

    # The merged-adapter config must be rejected before any pipeline build or
    # IPC import, so a provider-backed parameter is never aliased or written.
    loader._init_from_load_format.assert_not_called()
    backend.open_import.assert_not_called()
    assert provider_weight.data_ptr() == provider_ptr
    assert torch.equal(provider_weight, provider_snapshot)
