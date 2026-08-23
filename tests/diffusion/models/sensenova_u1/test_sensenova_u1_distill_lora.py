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
