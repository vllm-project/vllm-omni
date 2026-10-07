# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for Klein's packed TP layout and local-head execution."""

from contextlib import contextmanager

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.utils import default_unquantized_gemm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import ForwardContext
from vllm_omni.diffusion.models.flux2_klein import flux2_klein_transformer as klein

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@contextmanager
def _default_dtype(dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class _CPUAttention(torch.nn.Module):
    """Use SDPA while retaining the model's actual Q/K norm and RoPE."""

    def __init__(self, num_heads, head_size, softmax_scale, causal=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.scale = softmax_scale

    def forward(self, query, key, value, metadata=None):
        assert query.shape[2] == self.num_heads
        return F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=None if metadata is None else metadata.attn_mask,
            scale=self.scale,
        ).transpose(1, 2)


@pytest.fixture(autouse=True)
def _cpu_dispatch(monkeypatch):
    monkeypatch.setattr(linear, "dispatch_unquantized_gemm", lambda *a, **kw: default_unquantized_gemm)
    monkeypatch.setattr(klein, "Attention", _CPUAttention)
    monkeypatch.setattr(klein, "get_forward_context", ForwardContext)


def _set_tp(monkeypatch, tp, rank):
    # Keep real vLLM parameters, constructors and checkpoint loaders. Only the
    # distributed process metadata is substituted for this CPU unit test.
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(klein, "get_tensor_model_parallel_world_size", lambda: tp)


def _attention(monkeypatch, tp, rank, *, local_tp=True, heads=8, dim_head=8, mlp_ratio=3, sp=1):
    _set_tp(monkeypatch, tp, rank)
    with _default_dtype(torch.bfloat16):
        attention = klein.Flux2ParallelSelfAttention(
            parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp, ulysses_degree=sp),
            query_dim=heads * dim_head,
            heads=heads,
            dim_head=dim_head,
            mlp_ratio=mlp_ratio,
            bias=True,
            local_tp=local_tp,
        )
    attention.norm_q.forward = attention.norm_q.forward_native
    attention.norm_k.forward = attention.norm_k.forward_native
    attention.rope.forward = attention.rope.forward_native
    return attention


def _rank_rows(tensor, sizes, tp, rank):
    return torch.cat([part.chunk(tp, dim=0)[rank] for part in tensor.split(sizes, dim=0)])


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("name", ["weight", "bias"])
def test_packed_checkpoint_reconstruction_and_reload(monkeypatch, tp, name):
    sizes = [64, 64, 64, 192, 192]
    shape = (sum(sizes), 64) if name == "weight" else (sum(sizes),)
    canonical = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    shards = []
    for rank in range(tp):
        attention = _attention(monkeypatch, tp, rank)
        assert attention.local_tp == (tp > 1)
        assert attention.query_num_heads == 8 // tp
        projection = attention.to_qkv_mlp_proj
        param = getattr(projection, name)
        # Exact row markers verify the installed loader, without BF16 rounding
        # hiding a channel-order error. The constructor still uses BF16.
        param.data = param.data.float()
        for weights in (canonical + 1, canonical):
            param.weight_loader(param, weights)
            torch.testing.assert_close(param, _rank_rows(weights, sizes, tp, rank), rtol=0, atol=0)
        shards.append(param.detach().clone())
    local_sizes = [size // tp for size in sizes]
    reconstructed = torch.cat(
        [torch.cat([shard.split(local_sizes, dim=0)[i] for shard in shards]) for i in range(len(sizes))]
    )
    torch.testing.assert_close(reconstructed, canonical, rtol=0, atol=0)


@pytest.mark.parametrize("tp,heads", [(2, 8), (4, 24), (8, 24)])
@pytest.mark.parametrize("mask_kind", ["none", "3d", "broadcast", "per_head"])
@torch.inference_mode()
def test_local_heads_and_gather_order_match_dense_attention(monkeypatch, tp, heads, mask_kind):
    torch.manual_seed(17)
    dim = heads * 8
    baseline = _attention(monkeypatch, 1, 0, local_tp=False, heads=heads)
    for name, param in baseline.named_parameters():
        if name.startswith("norm_"):
            param.fill_(1)
        else:
            param.normal_(std=0.03)
    # A non-contiguous input and batch > 1 exercise the reshape boundaries.
    hidden = torch.randn(2, dim, 7, dtype=torch.bfloat16).transpose(1, 2)
    angles = torch.randn(7, 4)
    rotary = angles.cos(), angles.sin()
    mask = None
    if mask_kind != "none":
        mask_heads = heads if mask_kind == "per_head" else 1
        mask = torch.rand(2, mask_heads, 7, 7) > 0.25
        mask[..., 0] = True
        if mask_kind == "3d":
            mask = mask[:, 0]

    captured = []
    hook = baseline.to_out.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    expected = baseline(hidden, attention_mask=mask, image_rotary_emb=rotary)
    hook.remove()
    full_attention, full_mlp = captured[0].split([dim, 3 * dim], dim=-1)
    weights = {name: param.detach().clone() for name, param in baseline.named_parameters()}

    for rank in range(tp):
        local = _attention(monkeypatch, tp, rank, heads=heads)
        for name, param in local.named_parameters():
            getattr(param, "weight_loader", default_weight_loader)(param, weights[name])
        feature_calls: list[int] = []
        output_calls: list[int] = []

        def gather_features(value, dim=-1):
            reference = (full_attention, full_mlp)[len(feature_calls)]
            torch.testing.assert_close(value, reference.chunk(tp, dim=-1)[rank], rtol=0.02, atol=0.002)
            feature_calls.append(value.shape[-1])
            return reference

        def gather_output(value, dim=-1):
            torch.testing.assert_close(value, expected.chunk(tp, dim=-1)[rank], rtol=0.02, atol=0.002)
            output_calls.append(value.shape[-1])
            return expected

        # Each outgoing tensor must match the independently computed dense
        # result before the collective supplies the other ranks' slices.
        monkeypatch.setattr(klein, "tensor_model_parallel_all_gather", gather_features)
        monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", gather_output)
        actual = local(hidden, attention_mask=mask, image_rotary_emb=rotary)
        torch.testing.assert_close(actual, expected)
        assert feature_calls == [dim // tp, 3 * dim // tp]
        assert output_calls == [dim // tp]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tp": 1},
        {"tp": 4, "local_tp": False},
        {"tp": 4, "sp": 2},
        {"tp": 4, "heads": 6},
        {"tp": 4, "mlp_ratio": 2.1},
    ],
)
def test_ineligible_shapes_keep_original_projection(monkeypatch, kwargs):
    attention = _attention(monkeypatch, rank=0, **kwargs)
    assert not attention.local_tp
    assert type(attention.to_qkv_mlp_proj) is linear.ColumnParallelLinear
    assert attention.to_qkv_mlp_proj.gather_output
    assert attention.query_num_heads == attention.heads


@pytest.mark.parametrize(
    "overrides,parallel_overrides,eligible",
    [
        ({}, {}, True),
        ({"dtype": torch.float32}, {}, False),
        ({"dtype": torch.float16}, {}, False),
        ({}, {"tensor_parallel_size": 1}, False),
        ({}, {"ulysses_degree": 2}, False),
        ({"quantization_config": "fp8"}, {}, False),
        ({"lora_path": "local-adapter"}, {}, False),
        ({"enable_cpu_offload": True}, {}, False),
        ({"enable_layerwise_offload": True}, {}, False),
        ({"diffusion_offload_config": {"mode": "module", "components": ["dit"]}}, {}, False),
        ({}, {"tensor_parallel_size": 1, "use_hsdp": True, "hsdp_shard_size": 4}, False),
    ],
)
def test_pipeline_eligibility_uses_resolved_config(tmp_path, overrides, parallel_overrides, eligible):
    parallel = DiffusionParallelConfig(**({"tensor_parallel_size": 4} | parallel_overrides))
    config = OmniDiffusionConfig(
        **({"model": str(tmp_path), "dtype": torch.bfloat16} | overrides),
        parallel_config=parallel,
    )
    assert klein._use_local_single_stream_tp(config) is eligible


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_host_restore_contract_and_model_weight_reload(monkeypatch, tp):
    _set_tp(monkeypatch, tp, 0)
    kwargs = dict(
        in_channels=8,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=8,
        joint_attention_dim=8,
        timestep_guidance_channels=8,
        axes_dims_rope=(2, 2, 2, 2),
        guidance_embeds=False,
    )
    with _default_dtype(torch.bfloat16):
        baseline = klein.Flux2Transformer2DModel(**kwargs)
        model = klein.Flux2Transformer2DModel(**kwargs, local_tp=True)
    assert baseline.host_weight_restore_contract.version == "1"
    assert model.host_weight_restore_contract.version == ("2-local-single-stream-tp" if tp > 1 else "1")
    model.validate_restored_host_weights()
    name = "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"
    weights = torch.linspace(-1, 1, 576 * 64).reshape(576, 64).to(torch.bfloat16)
    for canonical in (weights, weights.neg()):
        assert name in model.load_weights([(name, canonical)])
        param = dict(model.named_parameters())[name]
        torch.testing.assert_close(param, _rank_rows(canonical, [64, 64, 64, 192, 192], tp, 0), rtol=0, atol=0)
    with _default_dtype(torch.bfloat16):
        restored = klein.Flux2Transformer2DModel(**kwargs, local_tp=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.validate_restored_host_weights()
    torch.testing.assert_close(dict(restored.named_parameters())[name], param, rtol=0, atol=0)


@pytest.mark.parametrize("sizes", [(8, 24), (8, 8, 8, 24, 24)])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@torch.inference_mode()
def test_fused_lora_global_rows_dynamic_activation(monkeypatch, sizes, tp):
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.lora_weights import LoRALayerWeights
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.lora.request import LoRARequest

    from vllm_omni.diffusion.lora.layers.column_parallel_linear import DiffusionMergedColumnParallelLinearWithLoRA
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

    torch.manual_seed(29)
    weights = torch.randn(sum(sizes), 8)
    lora_a = torch.randn(2, 8)
    lora_b = torch.randn(sum(sizes), 2)
    hidden = torch.randn(2, 3, 8)
    name = "transformer.to_qkv_mlp_proj"
    peft = PEFTHelper(r=2, lora_alpha=2, target_modules=["to_qkv_mlp_proj"])
    request = LoRARequest(lora_name="test", lora_int_id=7, lora_path="unused-by-memory-loader")
    for rank in range(tp):
        _set_tp(monkeypatch, tp, rank)
        projection = linear.MergedColumnParallelLinear(8, list(sizes), bias=False, gather_output=False)
        projection.weight.weight_loader(projection.weight, weights)
        pipeline = torch.nn.Module()
        pipeline.transformer = torch.nn.Module()
        pipeline.transformer.to_qkv_mlp_proj = projection
        manager = DiffusionLoRAManager(pipeline, device=torch.device("cpu"), dtype=torch.float32)
        adapter = LoRAModel(7, 2, {name: LoRALayerWeights(name, 2, 2, lora_a, lora_b)})
        # Adapter disk I/O is outside this regression. Actual manager injection,
        # five-/two-slice wrapper, rank slicing and activation all run below.
        monkeypatch.setattr(manager, "_load_adapter", lambda request: (adapter, peft))
        base_output = F.linear(hidden, _rank_rows(weights, sizes, tp, rank))
        delta = F.linear(F.linear(hidden, lora_a), _rank_rows(lora_b, sizes, tp, rank))
        for scale in (0.5, 1.0):
            manager.set_active_adapter(request, lora_scale=scale)
            wrapped = pipeline.transformer.to_qkv_mlp_proj
            assert isinstance(wrapped, DiffusionMergedColumnParallelLinearWithLoRA)
            assert wrapped.n_slices == len(sizes)
            actual, _ = wrapped(hidden)
            torch.testing.assert_close(actual, base_output + scale * delta, rtol=1e-5, atol=1e-5)
            manager.set_active_adapter(None)
            actual, _ = wrapped(hidden)
            torch.testing.assert_close(actual, base_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("image_count", [1, 2])
def test_native_editing_reaches_normal_input_validation(tmp_path, monkeypatch, image_count):
    from PIL import Image

    from vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein import Flux2KleinPipeline
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    class InputsReachedError(Exception):
        pass

    pipeline = object.__new__(Flux2KleinPipeline)
    pipeline.od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        dtype=torch.bfloat16,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=4),
    )
    assert klein._use_local_single_stream_tp(pipeline.od_config)
    images = [Image.new("RGB", (16, 16)) for _ in range(image_count)]
    image_input = images[0] if image_count == 1 else images
    request = OmniDiffusionRequest(
        prompt={"prompt": "paint the boat red", "multi_modal_data": {"image": image_input}},
        request_id="native-editing",
        sampling_params=OmniDiffusionSamplingParams(height=512, width=512, num_inference_steps=4),
    )

    def check_inputs(**kwargs):
        assert kwargs["prompt"] == "paint the boat red"
        raise InputsReachedError

    monkeypatch.setattr(pipeline, "check_inputs", check_inputs)
    with pytest.raises(InputsReachedError):
        pipeline.forward(DiffusionRequestBatch(requests=[request]))
    assert request.prompt["multi_modal_data"]["image"] is image_input
