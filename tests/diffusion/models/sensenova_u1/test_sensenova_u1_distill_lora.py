# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for SenseNova-U1 distilled-LoRA fusion.

The checkpoints ship unfused ``q_proj_mot_gen`` / ``gate_proj`` keys while the
pipeline runs fused layers, so each delta lands in its own row slice.
"""

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

HIDDEN, KV, INTER, RANK, ALPHA = 8, 4, 16, 2, 4.0


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj_mot_gen = nn.Linear(HIDDEN, HIDDEN + 2 * KV, bias=False)
        self.self_attn.o_proj_mot_gen = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp_mot_gen = nn.Module()
        self.mlp_mot_gen.gate_up_proj = nn.Linear(HIDDEN, 2 * INTER, bias=False)
        self.mlp_mot_gen.down_proj = nn.Linear(INTER, HIDDEN, bias=False)


class _Stub(SenseNovaU1Pipeline):
    """Only the parameter tree matters here, so skip the real __init__."""

    def __init__(self):
        nn.Module.__init__(self)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([_Layer()])


def _triple(out: int, in_: int, fill: float) -> dict[str, torch.Tensor]:
    return {
        "lora_down.weight": torch.full((RANK, in_), fill, dtype=torch.float32),
        "lora_up.weight": torch.full((out, RANK), 1.0, dtype=torch.float32),
        "alpha": torch.tensor(ALPHA),
    }


def _write(tmp_path, entries: dict[str, dict[str, torch.Tensor]]):
    flat = {f"{mod}.{leaf}": t for mod, parts in entries.items() for leaf, t in parts.items()}
    path = tmp_path / "lora.safetensors"
    save_file(flat, str(path))
    return str(path)


def _prefix(i: int = 0) -> str:
    return f"language_model.model.layers.{i}"


def test_qkv_shards_land_in_their_own_row_slices(tmp_path):
    pipe = _Stub()
    qkv = pipe.language_model.model.layers[0].self_attn.qkv_proj_mot_gen.weight
    with torch.no_grad():
        qkv.zero_()
    entries = {
        f"{_prefix()}.self_attn.q_proj_mot_gen": _triple(HIDDEN, HIDDEN, 1.0),
        f"{_prefix()}.self_attn.k_proj_mot_gen": _triple(KV, HIDDEN, 2.0),
        f"{_prefix()}.self_attn.v_proj_mot_gen": _triple(KV, HIDDEN, 3.0),
    }
    pipe.load_lora_weights(_write(tmp_path, entries))

    scale = ALPHA / RANK
    assert torch.allclose(qkv[:HIDDEN], torch.full((HIDDEN, HIDDEN), scale * RANK * 1.0))
    assert torch.allclose(qkv[HIDDEN : HIDDEN + KV], torch.full((KV, HIDDEN), scale * RANK * 2.0))
    assert torch.allclose(qkv[HIDDEN + KV :], torch.full((KV, HIDDEN), scale * RANK * 3.0))


def test_gate_and_up_land_in_their_own_halves(tmp_path):
    pipe = _Stub()
    gu = pipe.language_model.model.layers[0].mlp_mot_gen.gate_up_proj.weight
    with torch.no_grad():
        gu.zero_()
    entries = {
        f"{_prefix()}.mlp_mot_gen.gate_proj": _triple(INTER, HIDDEN, 1.0),
        f"{_prefix()}.mlp_mot_gen.up_proj": _triple(INTER, HIDDEN, 5.0),
    }
    pipe.load_lora_weights(_write(tmp_path, entries))

    scale = ALPHA / RANK
    assert torch.allclose(gu[:INTER], torch.full((INTER, HIDDEN), scale * RANK * 1.0))
    assert torch.allclose(gu[INTER:], torch.full((INTER, HIDDEN), scale * RANK * 5.0))


def test_unfused_targets_are_added_whole(tmp_path):
    pipe = _Stub()
    o = pipe.language_model.model.layers[0].self_attn.o_proj_mot_gen.weight
    before = o.detach().clone()
    entries = {f"{_prefix()}.self_attn.o_proj_mot_gen": _triple(HIDDEN, HIDDEN, 1.0)}
    pipe.load_lora_weights(_write(tmp_path, entries))
    assert torch.allclose(o - before, torch.full((HIDDEN, HIDDEN), (ALPHA / RANK) * RANK))


def test_lora_that_matches_nothing_raises_instead_of_no_op(tmp_path):
    """Without a loader the worker only warned and generated without the
    adapter, so a wrong path produced a plausible image and no error."""
    pipe = _Stub()
    entries = {f"{_prefix()}.self_attn.does_not_exist": _triple(HIDDEN, HIDDEN, 1.0)}
    with pytest.raises(ValueError, match="matched no parameter"):
        pipe.load_lora_weights(_write(tmp_path, entries))


def test_shape_mismatch_raises(tmp_path):
    pipe = _Stub()
    entries = {f"{_prefix()}.self_attn.o_proj_mot_gen": _triple(HIDDEN + 1, HIDDEN, 1.0)}
    with pytest.raises(RuntimeError, match="must match the size"):
        pipe.load_lora_weights(_write(tmp_path, entries))


def test_multiple_files_rejected(tmp_path):
    pipe = _Stub()
    path = _write(tmp_path, {f"{_prefix()}.self_attn.o_proj_mot_gen": _triple(HIDDEN, HIDDEN, 1.0)})
    with pytest.raises(ValueError, match="exactly one"):
        pipe.load_lora_weights([path, path])


class _ShardedLayer(nn.Module):
    """One rank of a two-way sharded fused QKV, with a vLLM-style weight loader.

    The parameter holds half of each of q, k and v. The loader is handed the whole
    layer's tensor and keeps only this rank's slice, which is what vLLM does.
    """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(HIDDEN // 2 + KV, HIDDEN))
        self.seen: list[tuple[str, tuple[int, ...]]] = []
        self.weight.weight_loader = self._weight_loader

    def _weight_loader(self, param, loaded_weight, shard_id):
        self.seen.append((shard_id, tuple(loaded_weight.shape)))
        sizes = {"q": HIDDEN // 2, "k": KV // 2, "v": KV // 2}
        offsets = {"q": 0, "k": HIDDEN // 2, "v": HIDDEN // 2 + KV // 2}
        rows = sizes[shard_id]
        param.data[offsets[shard_id] : offsets[shard_id] + rows] = loaded_weight[:rows]


class _ShardedStub(SenseNovaU1Pipeline):
    def __init__(self):
        nn.Module.__init__(self)
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.qkv_proj_mot_gen = _ShardedLayer()
        self.language_model.model.layers = nn.ModuleList([layer])


def test_fused_delta_is_sharded_through_the_weight_loader(tmp_path):
    """Under tensor parallelism the parameter is a rank-local shard while the
    checkpoint delta spans the whole layer, so the delta has to go through the
    layer's weight loader. Adding it directly raises a shape error."""
    pipe = _ShardedStub()
    layer = pipe.language_model.model.layers[0].self_attn.qkv_proj_mot_gen
    entries = {
        f"{_prefix()}.self_attn.q_proj_mot_gen": _triple(HIDDEN, HIDDEN, 1.0),
        f"{_prefix()}.self_attn.k_proj_mot_gen": _triple(KV, HIDDEN, 2.0),
        f"{_prefix()}.self_attn.v_proj_mot_gen": _triple(KV, HIDDEN, 3.0),
    }
    pipe.load_lora_weights(_write(tmp_path, entries))

    # the loader saw the whole layer's tensors, not a pre-sharded slice
    assert layer.seen == [("q", (HIDDEN, HIDDEN)), ("k", (KV, HIDDEN)), ("v", (KV, HIDDEN))]

    scale = ALPHA / RANK * RANK
    got = layer.weight
    assert torch.allclose(got[: HIDDEN // 2], torch.full((HIDDEN // 2, HIDDEN), scale * 1.0))
    assert torch.allclose(got[HIDDEN // 2 : HIDDEN // 2 + KV // 2], torch.full((KV // 2, HIDDEN), scale * 2.0))
    assert torch.allclose(got[HIDDEN // 2 + KV // 2 :], torch.full((KV // 2, HIDDEN), scale * 3.0))


def test_fusion_does_not_retain_the_adapter_tensors(tmp_path):
    """Fusion is one-way and this pipeline has no unload path, so holding the
    fp32 state dict would keep about 1.5 GiB alive for the shipped checkpoint."""
    pipe = _Stub()
    path = _write(tmp_path, {f"{_prefix()}.self_attn.o_proj_mot_gen": _triple(HIDDEN, HIDDEN, 1.0)})
    pipe.load_lora_weights(path, adapter_name="distill")

    kept = pipe.lora_loaded["distill"]
    assert not isinstance(kept, dict)
    assert not isinstance(kept, torch.Tensor)

    # the sentinel still makes a second load a no-op
    before = pipe.language_model.model.layers[0].self_attn.o_proj_mot_gen.weight.detach().clone()
    pipe.load_lora_weights(path, adapter_name="distill")
    torch.testing.assert_close(pipe.language_model.model.layers[0].self_attn.o_proj_mot_gen.weight, before)


@pytest.mark.parametrize("rank,alpha", [(3, 5.0), (128, 8.0)])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tp_fusion_matches_independent_full_weight_reference(
    tmp_path, monkeypatch, tp_size, kv_heads, dtype, rank, alpha
):
    """Original U1.5 names -> real vLLM loaders -> rank-local fused weights.

    Only rank discovery and accelerator GEMM selection are replaced. The LoRA
    file, alpha scaling, products, and vLLM parameter sharding are real. The
    oracle fuses full matrices using an independent sum of outer products,
    then explicitly selects each rank's rows/columns without a weight loader.
    One KV head additionally exercises replicated KV under TP2.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    generator = torch.Generator().manual_seed(216)
    hidden, head_dim, heads, intermediate = 8, 2, 4, 12
    kv = kv_heads * head_dim
    shapes = {
        "self_attn.q_proj_mot_gen": (hidden, hidden),
        "self_attn.k_proj_mot_gen": (kv, hidden),
        "self_attn.v_proj_mot_gen": (kv, hidden),
        "self_attn.o_proj_mot_gen": (hidden, hidden),
        "mlp_mot_gen.gate_proj": (intermediate, hidden),
        "mlp_mot_gen.up_proj": (intermediate, hidden),
        "mlp_mot_gen.down_proj": (hidden, intermediate),
    }
    entries, original, fused = {}, {}, {}
    for name, (out_size, in_size) in shapes.items():
        # Model the shipped BF16 adapter, even with FP32 base parameters.
        down = (torch.randn(rank, in_size, generator=generator) / 4).bfloat16()
        up = (torch.randn(out_size, rank, generator=generator) / 4).bfloat16()
        base = torch.randn(out_size, in_size, generator=generator).to(dtype)
        entries[f"{_prefix()}.{name}"] = {
            "lora_down.weight": down,
            "lora_up.weight": up,
            "alpha": torch.tensor(alpha, dtype=torch.bfloat16),
        }
        # Independent dense reference: no loader/scaling helper or matmul.
        delta = sum(torch.outer(up[:, r].double(), down[r].double()) for r in range(rank)) * (alpha / rank)
        original[name] = base
        # The existing distill contract casts the FP32 delta to the base dtype
        # before its in-place addition. Preserve this BF16 rounding behavior.
        fused[name] = base + delta.to(dtype)

    path = _write(tmp_path, entries)
    monkeypatch.setattr(
        linear,
        "dispatch_unquantized_gemm",
        lambda *_args, **_kwargs: lambda _layer, x, weight, bias=None: torch.nn.functional.linear(x, weight, bias),
    )
    for tp_rank in range(tp_size):
        for module in (linear, parameter):
            monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp_size)
            monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: tp_rank)
        with set_current_vllm_config(VllmConfig()):
            pipe = _Stub()
            block = pipe.language_model.model.layers[0]
            block.self_attn.qkv_proj_mot_gen = linear.QKVParallelLinear(
                hidden, head_dim, heads, kv_heads, bias=False, params_dtype=dtype
            )
            block.self_attn.o_proj_mot_gen = linear.RowParallelLinear(hidden, hidden, bias=False, params_dtype=dtype)
            block.mlp_mot_gen.gate_up_proj = linear.MergedColumnParallelLinear(
                hidden, [intermediate, intermediate], bias=False, params_dtype=dtype
            )
            block.mlp_mot_gen.down_proj = linear.RowParallelLinear(intermediate, hidden, bias=False, params_dtype=dtype)

        def local_weights(full):
            q_rows = hidden // tp_size
            kv_rows = kv // min(tp_size, kv_heads)
            kv_rank = tp_rank // max(tp_size // kv_heads, 1)
            mlp_rows = intermediate // tp_size
            return {
                "self_attn.qkv_proj_mot_gen.weight": torch.cat(
                    [
                        full["self_attn.q_proj_mot_gen"][tp_rank * q_rows : (tp_rank + 1) * q_rows],
                        full["self_attn.k_proj_mot_gen"][kv_rank * kv_rows : (kv_rank + 1) * kv_rows],
                        full["self_attn.v_proj_mot_gen"][kv_rank * kv_rows : (kv_rank + 1) * kv_rows],
                    ]
                ),
                "self_attn.o_proj_mot_gen.weight": full["self_attn.o_proj_mot_gen"][
                    :, tp_rank * q_rows : (tp_rank + 1) * q_rows
                ],
                "mlp_mot_gen.gate_up_proj.weight": torch.cat(
                    [
                        full[f"mlp_mot_gen.{part}_proj"][tp_rank * mlp_rows : (tp_rank + 1) * mlp_rows]
                        for part in ("gate", "up")
                    ]
                ),
                "mlp_mot_gen.down_proj.weight": full["mlp_mot_gen.down_proj"][
                    :, tp_rank * mlp_rows : (tp_rank + 1) * mlp_rows
                ],
            }

        # The original adapter targets only generation projections.
        block.self_attn.o_proj = nn.Linear(hidden, hidden, bias=False, dtype=dtype)
        expected = local_weights(fused)
        parameters = dict(block.named_parameters())
        for name, value in local_weights(original).items():
            parameters[name].data.copy_(value)
        before = {name: param.detach().clone() for name, param in parameters.items()}
        pipe.load_lora_weights(path)
        for name, value in expected.items():
            assert parameters[name].dtype == dtype
            torch.testing.assert_close(parameters[name], value, rtol=1e-6, atol=1e-6)
            assert not torch.equal(parameters[name], before[name])
        torch.testing.assert_close(
            parameters["self_attn.o_proj.weight"], before["self_attn.o_proj.weight"], rtol=0, atol=0
        )
        once = {name: param.detach().clone() for name, param in parameters.items()}
        pipe.load_lora_weights(path)
        for name, value in once.items():
            torch.testing.assert_close(parameters[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_replicated_projection_fusion_matches_outer_product_reference(tmp_path, dtype):
    """Unsharded FM projections keep the same complete delta on every rank."""
    pipe = _Stub()
    pipe.fm_modules = nn.ModuleDict({"fm_head": nn.Sequential(nn.Linear(8, 6, bias=False, dtype=dtype))})
    weight = pipe.fm_modules["fm_head"][0].weight
    generator = torch.Generator().manual_seed(88)
    down = torch.randn(3, 8, generator=generator).bfloat16()
    up = torch.randn(6, 3, generator=generator).bfloat16()
    delta = sum(torch.outer(up[:, r].double(), down[r].double()) for r in range(3)) * (5 / 3)
    expected = weight.detach().clone() + delta.to(dtype)
    path = _write(
        tmp_path,
        {"fm_modules.fm_head.0": {"lora_down.weight": down, "lora_up.weight": up, "alpha": torch.tensor(5.0)}},
    )
    pipe.load_lora_weights(path)
    torch.testing.assert_close(weight, expected, rtol=1e-6, atol=1e-6)
