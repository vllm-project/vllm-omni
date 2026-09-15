# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.parameter import (
    ChannelQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
    RowvLLMParameter,
)

from vllm_omni.diffusion.data import (
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    TransformerConfig,
)
from vllm_omni.diffusion.models.flux.flux_transformer import (
    ColumnParallelApproxGELU,
    FluxSingleBlockOutput,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    _should_use_flux_optimizations,
    _use_sharded_single_block_path,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _init_distributed():
    """Initialize minimal distributed state for TP-aware linear layers.

    ``world_size=1``: these tests cover branch selection and weight-loading
    bookkeeping only. ``tensor_parallel_size=2`` below selects the sharded
    branch but no collective, per-rank shard or all-reduce actually runs.
    Real two-rank forward/load parity lives in
    ``tests/diffusion/distributed/test_flux_sharded_proj_tp2.py``.
    """
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method="tcp://127.0.0.1:29513",
    )
    initialize_model_parallel()
    yield
    cleanup_dist_env_and_memory()


@pytest.mark.parametrize(
    "env_value,expected",
    [
        (None, False),
        ("", False),
        ("1", True),
        ("true", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("off", False),
        ("disabled", False),
        ("unexpected", False),
    ],
)
def test_should_use_flux_optimizations_env_values(monkeypatch, env_value, expected):
    env_key = "VLLM_OMNI_FLUX1_SHARDED_PROJ"
    if env_value is None:
        monkeypatch.delenv(env_key, raising=False)
    else:
        monkeypatch.setenv(env_key, env_value)

    assert _should_use_flux_optimizations() is expected


def test_use_sharded_single_block_path_respects_env_and_tp(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")

    tp2 = DiffusionParallelConfig(tensor_parallel_size=2)
    tp1 = DiffusionParallelConfig(tensor_parallel_size=1)

    assert _use_sharded_single_block_path(tp2) is True
    assert _use_sharded_single_block_path(tp1) is False

    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "0")
    assert _use_sharded_single_block_path(tp2) is False


def test_use_sharded_single_block_path_uses_runtime_tp_when_config_missing(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux.flux_transformer.get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    assert _use_sharded_single_block_path(None) is True

    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux.flux_transformer.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    assert _use_sharded_single_block_path(None) is False


def test_single_transformer_block_uses_sharded_modules_when_enabled(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")
    block = FluxSingleTransformerBlock(
        dim=64,
        num_attention_heads=2,
        attention_head_dim=32,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
        prefix="single_transformer_blocks.0",
    )

    assert block.use_sharded_single_block is True
    assert isinstance(block.proj_mlp, ColumnParallelApproxGELU)
    assert isinstance(block.proj_out, FluxSingleBlockOutput)
    assert block.attn.output_is_parallel is True


def test_single_transformer_block_uses_replicated_modules_when_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "0")
    block = FluxSingleTransformerBlock(
        dim=64,
        num_attention_heads=2,
        attention_head_dim=32,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
        prefix="single_transformer_blocks.0",
    )

    assert block.use_sharded_single_block is False
    assert isinstance(block.proj_out, ReplicatedLinear)
    assert hasattr(block, "act_mlp")
    assert block.attn.output_is_parallel is False


def _od_config(*, num_layers: int, tensor_parallel_size: int) -> OmniDiffusionConfig:
    """Build a real ``OmniDiffusionConfig`` rather than a duck-typed stand-in.

    ``TransformerConfig`` keeps its values in a ``params`` dict behind
    ``__getattr__``, so the real class exercises the same attribute path
    ``FluxTransformer2DModel.__init__`` uses in production.
    """
    return OmniDiffusionConfig(
        tf_model_config=TransformerConfig(params={"num_layers": num_layers}),
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=tensor_parallel_size),
    )


def _sharded_proj_out() -> FluxSingleBlockOutput:
    block = FluxSingleTransformerBlock(
        dim=64,
        num_attention_heads=2,
        attention_head_dim=32,
        parallel_config=DiffusionParallelConfig(tensor_parallel_size=2),
        prefix="single_transformer_blocks.0",
    )
    assert block.use_sharded_single_block is True
    assert isinstance(block.proj_out, FluxSingleBlockOutput)
    return block.proj_out


def test_load_weight_narrows_real_model_weight_parameter(monkeypatch):
    """The fused checkpoint weight is split by logical width, honoring ``input_dim``."""
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")
    proj_out = _sharded_proj_out()
    attn_dim, mlp_dim, out_dim = proj_out.attn_dim, proj_out.mlp_dim, proj_out.out_dim

    # Mirror how quantized linear methods register weights: real ModelWeightParameter
    # dispatched through the layer's own weight_loader_v2, not a hand-rolled callable.
    for proj, width in ((proj_out.attn_proj, attn_dim), (proj_out.mlp_proj, mlp_dim)):
        proj.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.zeros(out_dim, width, dtype=torch.float32),
                input_dim=1,
                output_dim=0,
                weight_loader=proj.weight_loader_v2,
            ),
        )

    loaded_weight = torch.randn(out_dim, attn_dim + mlp_dim)
    proj_out.load_weight("weight", loaded_weight)

    torch.testing.assert_close(proj_out.attn_proj.weight.data, loaded_weight[:, :attn_dim])
    torch.testing.assert_close(proj_out.mlp_proj.weight.data, loaded_weight[:, attn_dim:])


@pytest.mark.parametrize("scale_kind", ["per_tensor", "per_channel"])
def test_load_weight_replicates_real_quantized_scales(monkeypatch, scale_kind):
    """FP8 scales describe output rows, which both halves share, so they replicate.

    Uses vLLM's real ``PerTensorScaleParameter`` / ``ChannelQuantScaleParameter``
    with the layer's real ``weight_loader_v2`` dispatcher: neither exposes
    ``input_dim``, and narrowing them would trip the loader's shape assert.
    """
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")
    proj_out = _sharded_proj_out()
    out_dim = proj_out.out_dim

    def _make_scale(proj):
        if scale_kind == "per_tensor":
            # One scale per logical output partition, as ModelOpt/AutoFP8 register it.
            return PerTensorScaleParameter(
                data=torch.zeros(1, dtype=torch.float32),
                weight_loader=proj.weight_loader_v2,
            )
        return ChannelQuantScaleParameter(
            data=torch.zeros(out_dim, 1, dtype=torch.float32),
            output_dim=0,
            weight_loader=proj.weight_loader_v2,
        )

    for proj in (proj_out.attn_proj, proj_out.mlp_proj):
        proj.register_parameter("weight_scale", _make_scale(proj))

    expected = torch.rand(1) if scale_kind == "per_tensor" else torch.rand(out_dim, 1)
    proj_out.load_weight("weight_scale", expected)

    torch.testing.assert_close(proj_out.attn_proj.weight_scale.data, expected)
    torch.testing.assert_close(proj_out.mlp_proj.weight_scale.data, expected)
    assert proj_out.loaded_parameter_names("weight_scale") == [
        "attn_proj.weight_scale",
        "mlp_proj.weight_scale",
    ]


@pytest.mark.parametrize("weight_name", ["g_idx", "weight_g_idx"])
def test_load_weight_rejects_act_order_group_indices(monkeypatch, weight_name):
    """Act-order indices must be rejected, not narrowed.

    ``g_idx`` holds positions along the input dim, so narrowing it by logical width
    rebases the positions but not the values: the mlp half would receive indices
    offset past the end of its own scale table. That is silent corruption, so the
    split loader raises and points at the env var instead.
    """
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")
    proj_out = _sharded_proj_out()
    total_dim = proj_out.attn_dim + proj_out.mlp_dim

    # Shaped like a real act-order tensor: one group index per input column.
    g_idx = torch.arange(total_dim, dtype=torch.int32) // 8
    for proj in (proj_out.attn_proj, proj_out.mlp_proj):
        proj.register_parameter(
            weight_name,
            RowvLLMParameter(
                data=torch.zeros(proj.input_size_per_partition, dtype=torch.int32),
                input_dim=0,
                weight_loader=proj.weight_loader_v2,
            ),
        )

    with pytest.raises(ValueError, match="VLLM_OMNI_FLUX1_SHARDED_PROJ"):
        proj_out.load_weight(weight_name, g_idx)

    # Nothing was written before the raise.
    assert torch.all(proj_out.attn_proj.get_parameter(weight_name) == 0)


def test_load_weight_raises_for_parameter_absent_on_split_projections(monkeypatch):
    """A checkpoint key with no matching parameter must name the env var to unset."""
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")
    proj_out = _sharded_proj_out()

    # "qweight" exists on quantized layers but not on these unquantized projections.
    with pytest.raises(ValueError, match="VLLM_OMNI_FLUX1_SHARDED_PROJ"):
        proj_out.load_weight("qweight", torch.ones(proj_out.out_dim, proj_out.attn_dim + proj_out.mlp_dim))


def test_tp2_load_weights_splits_proj_out_weight_in_sharded_path(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_FLUX1_SHARDED_PROJ", "1")

    model = FluxTransformer2DModel(
        od_config=_od_config(num_layers=0, tensor_parallel_size=2),
        in_channels=4,
        num_layers=0,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        axes_dims_rope=(4, 6, 6),
        guidance_embeds=False,
    )

    block = model.single_transformer_blocks[0]
    assert block.use_sharded_single_block is True
    assert isinstance(block.proj_out, FluxSingleBlockOutput)

    proj_out = block.proj_out
    attn_dim = proj_out.attn_dim
    mlp_dim = proj_out.mlp_dim
    total_dim = attn_dim + mlp_dim

    loaded_weight = torch.randn(proj_out.out_dim, total_dim)
    loaded = model.load_weights(
        [
            ("single_transformer_blocks.0.proj_out.weight", loaded_weight),
        ]
    )

    assert "single_transformer_blocks.0.proj_out.weight" in loaded
    assert "single_transformer_blocks.0.proj_out.attn_proj.weight" in loaded
    assert "single_transformer_blocks.0.proj_out.mlp_proj.weight" in loaded

    attn_param = proj_out.attn_proj.weight
    mlp_param = proj_out.mlp_proj.weight
    split_dim = attn_param.input_dim
    dim_size = loaded_weight.shape[split_dim]

    attn_start = 0
    attn_size = dim_size * attn_dim // total_dim
    mlp_start = dim_size * attn_dim // total_dim
    mlp_size = dim_size * mlp_dim // total_dim

    expected_attn = loaded_weight.narrow(split_dim, attn_start, attn_size)
    expected_mlp = loaded_weight.narrow(split_dim, mlp_start, mlp_size)

    torch.testing.assert_close(attn_param.detach(), expected_attn)
    torch.testing.assert_close(mlp_param.detach(), expected_mlp)
