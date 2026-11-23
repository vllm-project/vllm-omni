# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FLUX's single-block MLP split over the TP group, as vLLM splits an MLP.

``proj_mlp`` is column-parallel without a gather, the pre-only attention keeps
its rank's heads, and ``proj_out`` is row-parallel over each rank's
``[own heads | own MLP channels]`` columns, with one all-reduce. Gloo ranks build
the single block both ways from the same checkpoint tensors, loaded through
``FluxTransformer2DModel.load_weights``. At one rank the split block is the
replicated block; at two ranks it is vLLM's row-parallel arithmetic over the
whole concatenation: each rank's partial over its K-slice, rank 0's with the
bias, summed.
"""

import os

import pytest
import torch

from tests.helpers.distributed import configure_cpu_collective_worker, start_collective_workers
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.models.flux import flux_transformer
from vllm_omni.platforms.interface import OmniPlatformEnum, UnspecifiedOmniPlatform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DIM, HEADS, HEAD_DIM, IMG, TXT = 64, 4, 16, 6, 3
MLP = 4 * DIM


def _checkpoint() -> dict[str, torch.Tensor]:
    """A single block's checkpoint in diffusers' names, seeded alike on every rank."""
    shapes = {
        "norm.linear.weight": (3 * DIM, DIM),
        "norm.linear.bias": (3 * DIM,),
        "proj_mlp.weight": (MLP, DIM),
        "proj_mlp.bias": (MLP,),
        "proj_out.weight": (DIM, DIM + MLP),
        "proj_out.bias": (DIM,),
        "attn.norm_q.weight": (HEAD_DIM,),
        "attn.norm_k.weight": (HEAD_DIM,),
    }
    for name in ("to_q", "to_k", "to_v"):
        shapes[f"attn.{name}.weight"] = (DIM, DIM)
        shapes[f"attn.{name}.bias"] = (DIM,)
    generator = torch.Generator().manual_seed(7)
    return {key: 0.2 * torch.randn(shape, generator=generator) for key, shape in shapes.items()}


def _rank_columns(rank: int, world: int) -> torch.Tensor:
    """Rank ``rank``'s columns of the ``[attention | MLP]`` concatenation: its heads, then its MLP channels."""
    a, m = DIM // world, MLP // world
    return torch.cat([torch.arange(rank * a, (rank + 1) * a), DIM + torch.arange(rank * m, (rank + 1) * m)])


def test_rank_major_columns():
    weight = torch.arange(12).repeat(3, 1)  # columns [attention 0..3 | MLP 4..11]

    reordered = flux_transformer.rank_major_columns(weight, 4, 8, 2)

    assert torch.equal(reordered[0], torch.tensor([0, 1, 4, 5, 6, 7, 2, 3, 8, 9, 10, 11]))
    assert flux_transformer.rank_major_columns(weight, 4, 8, 1) is weight
    transposed = flux_transformer.rank_major_columns(weight.t(), 4, 8, 2, dim=0)
    assert torch.equal(transposed[:, 0], reordered[0])


@pytest.mark.parametrize(
    ("tp", "overrides", "offload", "expected"),
    [
        (2, {}, False, True),
        (1, {}, False, False),
        (2, {"quantization_config": "fp8"}, False, False),
        (2, {"lora_path": "adapter"}, False, False),
        (2, {"lora_application": "merged", "weight_transfer_config": {"backend": "omni_lora_ipc"}}, False, False),
        (2, {}, True, False),
    ],
    ids=["tp2", "tp1", "quantized", "lora", "merged_lora", "offload"],
)
def test_single_block_mlp_split_gate(monkeypatch, tp, overrides, offload, expected):
    from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

    monkeypatch.setattr(flux_transformer, "offload_enabled", lambda config: offload)
    od_config = OmniDiffusionConfig(
        model="test", parallel_config=DiffusionParallelConfig(tensor_parallel_size=tp), **overrides
    )

    assert flux_transformer._shard_single_block_mlp(od_config) is expected
    assert flux_transformer._shard_single_block_mlp(None) is False


def _load(block, checkpoint) -> None:
    # The transformer's own loader: to_q/to_k/to_v into the packed to_qkv, every other
    # tensor through its parameter's weight_loader. It reads only the module's parameters.
    loaded = flux_transformer.FluxTransformer2DModel.load_weights(block, iter(checkpoint.items()))
    assert not set(dict(block.named_parameters())) - loaded


class _OutOfTreePlatform(UnspecifiedOmniPlatform):
    """An out-of-tree platform: the attention dispatches to torch SDPA on CPU."""

    _omni_enum = OmniPlatformEnum.OOT


def _split_block_worker(rank: int, world: int, dtype: torch.dtype, master_port: int) -> None:
    os.environ.update(
        {
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )
    configure_cpu_collective_worker()
    import torch.nn.functional as F
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.utils import default_unquantized_gemm
    from vllm.utils.torch_utils import set_default_torch_dtype

    from vllm_omni.diffusion.attention import layer as attention_layer
    from vllm_omni.diffusion.attention.backends import abstract
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(backend="gloo")
    initialize_model_parallel(tensor_parallel_size=world, backend="gloo")
    linear.dispatch_unquantized_gemm = lambda *a, **kw: default_unquantized_gemm
    attention_layer.get_attn_backend_for_role = lambda *a, **kw: (SDPABackend, None)
    abstract.current_omni_platform = _OutOfTreePlatform()
    try:
        block_cls = flux_transformer.FluxSingleTransformerBlock
        checkpoint = _checkpoint()
        kwargs = dict(dim=DIM, num_attention_heads=HEADS, attention_head_dim=HEAD_DIM)
        with (
            set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))),
            set_default_torch_dtype(dtype),
            torch.no_grad(),
        ):
            replicated = block_cls(**kwargs, prefix="r")
            split = block_cls(**kwargs, prefix="s", shard_mlp=True)
            assert split.shard_mlp is (world > 1)
            if world > 1:
                assert isinstance(split.proj_mlp, linear.ColumnParallelLinear) and not split.proj_mlp.gather_output
                assert isinstance(split.proj_out, linear.RowParallelLinear) and split.proj_out.reduce_results
                assert split.attn.gather_output is False
            _load(replicated, checkpoint)
            _load(split, checkpoint)
            whole = {key: value.to(dtype) for key, value in checkpoint.items()}
            if world > 1:
                rows = MLP // world
                assert torch.equal(split.proj_mlp.weight, whole["proj_mlp.weight"][rank * rows : (rank + 1) * rows])
                assert torch.equal(split.proj_out.weight, whole["proj_out.weight"][:, _rank_columns(rank, world)])
                assert torch.equal(split.proj_out.bias, whole["proj_out.bias"])

            generator = torch.Generator().manual_seed(11)
            image = torch.randn(1, IMG, DIM, generator=generator).to(dtype)
            text = torch.randn(1, TXT, DIM, generator=generator).to(dtype)
            temb = torch.randn(1, DIM, generator=generator).to(dtype)
            ids = torch.stack(
                [torch.zeros(TXT + IMG), torch.arange(TXT + IMG) // 3, torch.arange(TXT + IMG) % 3], dim=-1
            )
            rotary = flux_transformer.FluxPosEmbed(theta=10000, axes_dim=[4, 6, 6])(ids)
            inputs = dict(hidden_states=image, encoder_hidden_states=text, temb=temb, image_rotary_emb=rotary)
            text_replicated, image_replicated = replicated(**inputs)
            text_split, image_split = split(**inputs)

            # vLLM's row-parallel arithmetic over the whole concatenation, by hand: the
            # replicated block's MLP and gathered attention, then each rank's partial over
            # its K-slice in the dtype, rank 0's with the bias, summed in rank order.
            joint = torch.cat([text, image], dim=1)
            norm_hidden_states, gate = replicated.norm(joint, emb=temb)
            mlp = replicated.act_mlp(replicated.proj_mlp(norm_hidden_states))
            attn = replicated.attn(hidden_states=norm_hidden_states, image_rotary_emb=rotary)
            concat = torch.cat([attn, mlp], dim=2)
            total = None
            for r in range(world):
                k = _rank_columns(r, world)
                partial = F.linear(
                    concat[..., k], whole["proj_out.weight"][:, k], whole["proj_out.bias"] if r == 0 else None
                )
                total = partial if total is None else total + partial
            expected = joint + gate.unsqueeze(1) * total

            actual = torch.cat([text_split, image_split], dim=1)
            assert actual.dtype == dtype
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if world == 1:
                torch.testing.assert_close(image_split, image_replicated, rtol=0, atol=0)
                torch.testing.assert_close(text_split, text_replicated, rtol=0, atol=0)
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("world", [1, 2])
def test_split_block_is_vllm_row_parallel_arithmetic(world, dtype):
    start_collective_workers(_split_block_worker, args=(world, dtype, get_open_port()), nprocs=world, device_kind="cpu")


def test_sharded_proj_out_has_no_lora_wrapper(monkeypatch):
    """The documented limitation: a sharded single block's proj_out is not LoRA-wrapped.

    vLLM's row-parallel LoRA wrapper slices ``lora_a`` contiguously, which would pair
    the wrong columns with the rank-major input, so the layer is left unwrapped and an
    adapter targeting it fails at activation instead of computing a wrong delta.
    """
    from vllm.config.lora import LoRAConfig
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    from vllm_omni.lora.binding import from_layer

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    proj_out = flux_transformer._HeadsAndMlpRowParallelLinear(
        DIM + MLP, DIM, attention_size=DIM, bias=True, return_bias=False, prefix="proj_out"
    )

    wrapped = from_layer(
        layer=proj_out,
        max_loras=1,
        lora_config=LoRAConfig(max_lora_rank=8, max_loras=1, lora_dtype=torch.float32),
        packed_modules_list=[],
    )

    assert wrapped is proj_out
