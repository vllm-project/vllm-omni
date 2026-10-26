# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU bytewise gates against the interpolation frozen at 32e1a2dba."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def load_module():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_parallel.py"
    spec = importlib.util.spec_from_file_location("vision_position_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def assert_bytes(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    assert left.stride() == right.stride()
    assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def reference_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    device = self.pos_embed.weight.device
    idx_list: list[list[int]] = [[] for _ in range(4)]
    weight_list: list[list[float]] = [[] for _ in range(4)]
    for _, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor
        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side
        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]
        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]
        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
    patch_pos_embeds_permute = []
    merge_size = self.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    return torch.cat(patch_pos_embeds_permute)


def current_method():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/encoder.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "fast_pos_embed_interpolate"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method.name]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "merge,grid",
    [(1, [[2, 3, 5], [1, 2, 7]]), (2, [[2, 6, 10], [1, 4, 2]]), (4, [[1, 8, 12], [3, 4, 4]]), (2, [[1, 2, 2]])],
)
def test_local_interpolation_matches_frozen_reference(mocker, dtype, merge, grid):
    model = mocker.Mock(
        pos_embed=torch.nn.Embedding(49, 19, dtype=dtype), num_grid_per_side=7, spatial_merge_size=merge
    )
    with torch.no_grad():
        values = torch.arange(49 * 19).reshape(49, 19)
        model.pos_embed.weight.copy_(((values % 31 - 15) * 0.125).to(dtype))
        model.pos_embed.weight[0, 0] = -0.0
    grid = torch.tensor(grid)
    reference = reference_interpolate(model, grid)
    method = current_method()
    assert_bytes(method(model, grid), reference)
    plan = load_module().build_vision_plan(
        tuple(h * w for t, h, w in grid.tolist() for _ in range(t)), 8, merge_size=merge
    )
    selections = list(plan.rank_token_indices) + [(reference.shape[0] - 1, 0, 0, reference.shape[0] // 2)]
    for ids in selections:
        seen = []
        hook = model.pos_embed.register_forward_pre_hook(lambda module, args: seen.append(tuple(args[0].shape)))
        try:
            result = method(model, grid, token_indices=ids)
        finally:
            hook.remove()
        assert_bytes(result, reference.index_select(0, torch.tensor(ids, dtype=torch.long)))
        assert seen == [(4, len(ids))]
        assert torch.isfinite(result).all()


@pytest.mark.parametrize("ids", [(-1,), (4,)])
def test_invalid_local_token_ids(mocker, ids):
    model = mocker.Mock(pos_embed=torch.nn.Embedding(49, 3), num_grid_per_side=7, spatial_merge_size=2)
    with pytest.raises(ValueError, match="token"):
        current_method()(model, torch.tensor([[1, 2, 2]]), token_indices=ids)


def test_large_bf16_representative_grid(mocker):
    # Representative factorization of the historical segment lengths, not a
    # claim that these h/w values are the original frozen six-image geometry.
    grid = torch.tensor([[1, 136, 60], [1, 136, 60], [1, 44, 20], [1, 30, 22], [1, 30, 22], [1, 66, 116]])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = mocker.Mock(
            pos_embed=torch.nn.Embedding(48**2, 1152, dtype=torch.bfloat16), num_grid_per_side=48, spatial_merge_size=2
        )
        generator = torch.Generator().manual_seed(9398)
        with torch.no_grad():
            model.pos_embed.weight.copy_(torch.randn(model.pos_embed.weight.shape, generator=generator))
            reference = reference_interpolate(model, grid)
            plan = load_module().build_vision_plan((8160, 8160, 880, 660, 660, 7656), 8)
            for ids in plan.rank_token_indices:
                result = current_method()(model, grid, token_indices=ids)
                assert_bytes(result, reference[list(ids)])
    finally:
        torch.set_num_threads(threads)


def sp_namespace(mocker, monkeypatch, rank):
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_sp.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forward_sp8")
    parallel = load_module()
    namespace = {
        "torch": torch,
        "dist": parallel.dist,
        "build_vision_plan": parallel.build_vision_plan,
        "vision_permutation_enabled": parallel.vision_permutation_enabled,
        "VisionAllToAll": lambda *args, **kwargs: mocker.Mock(rank=rank),
        "gather_vision_outputs": lambda result, *args, **kwargs: result,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    monkeypatch.setattr(parallel.dist, "all_reduce", lambda status, **kwargs: None)
    return namespace


@pytest.mark.parametrize("enabled,verify", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("rank", [0, 7])
def test_sp8_position_dispatch(mocker, monkeypatch, enabled, verify, rank):
    monkeypatch.setenv("VLLM_OMNI_H3_VISION_LOCAL_POS", str(int(enabled)))
    monkeypatch.setenv("VLLM_OMNI_H3_VISION_LOCAL_POS_VERIFY", str(int(verify)))
    namespace = sp_namespace(mocker, monkeypatch, rank)
    model = mocker.Mock(
        pos_embed=torch.nn.Embedding(49, 4),
        num_grid_per_side=7,
        spatial_merge_size=2,
        config=mocker.Mock(num_heads=16),
        patch_embed=lambda pixels: pixels,
        rot_pos_emb=lambda grid: torch.zeros(24, 2),
        blocks=[],
        deepstack_visual_indexes=[],
        merger=lambda hidden: hidden,
    )
    calls = []

    def interpolate(grid, **kwargs):
        calls.append(kwargs)
        return current_method()(model, grid, **kwargs)

    model.fast_pos_embed_interpolate = interpolate
    pixels, grid = torch.zeros(24, 4), torch.tensor([[2, 2, 6]])
    result, deep = namespace["forward_sp8"](model, pixels, grid, group=None)
    ids = namespace["build_vision_plan"]((12, 12), 8).rank_token_indices[rank]
    assert_bytes(result, reference_interpolate(model, grid)[list(ids)])
    assert deep == []
    assert calls[0] == ({"token_indices": ids} if enabled else {})
    assert len(calls) == 1 + int(verify)


def test_sp8_remote_rank_failure_is_propagated(mocker, monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_H3_VISION_LOCAL_POS", "1")
    monkeypatch.setenv("VLLM_OMNI_H3_VISION_LOCAL_POS_VERIFY", "1")
    namespace = sp_namespace(mocker, monkeypatch, 0)
    monkeypatch.setattr(namespace["dist"], "all_reduce", lambda status, **kwargs: status.fill_(1))
    model = mocker.Mock(
        config=mocker.Mock(num_heads=16),
        spatial_merge_size=2,
        patch_embed=lambda pixels: pixels,
        fast_pos_embed_interpolate=lambda grid, **kwargs: torch.zeros(4, 2),
    )
    with pytest.raises(RuntimeError, match="changed bytes"):
        namespace["forward_sp8"](model, torch.zeros(32, 2), torch.tensor([[1, 4, 8]]), group=None)


@pytest.mark.parametrize("name", ["VLLM_OMNI_H3_VISION_LOCAL_POS", "VLLM_OMNI_H3_VISION_LOCAL_POS_VERIFY"])
def test_local_position_flags_default_off_and_strict(monkeypatch, name):
    parallel = load_module()
    monkeypatch.delenv(name, raising=False)
    assert not parallel.vision_permutation_enabled(name)
    monkeypatch.setenv(name, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parallel.vision_permutation_enabled(name)
