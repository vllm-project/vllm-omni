# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.models.magi2 import layers as magi2_layers
from vllm_omni.diffusion.models.magi2.configuration_magi2 import Magi2RefinerConfig
from vllm_omni.diffusion.models.magi2.layers import ModalityDispatcher
from vllm_omni.diffusion.models.magi2.modeling_magi2_refiner import Magi2RefinerTransformer
from vllm_omni.diffusion.models.magi2.window_attention import (
    build_reference_window_ranges,
    build_window_attention_plan,
    torch_window_attention_reference,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

_MANIFEST = Path(__file__).parent / "fixtures" / "magi2_refiner_checkpoint_manifest.txt"
_CHECKPOINT_PARAMETERS = 6_720_075_280
_SAFETENSORS_DTYPES = {torch.bfloat16: "BF16", torch.float32: "F32"}
_BF16 = torch.bfloat16
_VIDEO, _AUDIO, _TEXT = 0, 1, 2


def _tiny_config(**overrides) -> Magi2RefinerConfig:
    fields = dict(
        num_layers=4,
        hidden_size=256,
        head_dim=32,
        num_query_groups=2,
        video_in_channels=6,
        audio_in_channels=5,
        text_in_channels=40,
        multimodal_layers=(0, 3),
    )
    fields.update(overrides)
    return Magi2RefinerConfig(**fields)


def _randomize(model: Magi2RefinerTransformer, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            values = torch.randn(parameter.shape, generator=generator)
            if name.endswith("rope.bands"):
                continue
            # Norm weights are scale offsets: (1 + w).  Keep every tensor distinct and nonzero.
            scale = 0.2 if parameter.ndim == 1 else parameter.shape[-1] ** -0.5
            parameter.copy_((values * scale).to(parameter.dtype))


def _tiny_model(seed: int = 0, **overrides) -> Magi2RefinerTransformer:
    model = Magi2RefinerTransformer(_tiny_config(**overrides))
    _randomize(model, seed)
    return model


# Straight-line transcription of the released refiner forward (SandAI
# magi2_refiner.py: MultiModalityRMSNorm, _bf16_compute_linear, swiglu7,
# apply_rotary_emb_torch, Attention, MLP, TransformerLayer, Adapter,
# PostAdapter, Transformer).  Only the window-attention kernel is shared.


class _RefDispatcher:
    def __init__(self, modality: torch.Tensor) -> None:
        # The released dispatcher uses an unstable argsort; any modality-grouping order is
        # valid, and the port's stable order is used so that row order matches bit for bit.
        self.permute_mapping = torch.argsort(modality, stable=True)
        self.inv_permute_mapping = torch.argsort(self.permute_mapping)
        permuted = modality[self.permute_mapping]
        self.group_size = [int(n) for n in torch.bincount(permuted, minlength=3).tolist()]

    def dispatch(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(torch.split(x, self.group_size, dim=0))


def _ref_rms_norm(x, weight, num_modality, dispatcher, eps=1e-6):
    original_dtype = x.dtype
    t = x.float()
    t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + eps)
    if num_modality == 1:
        return (t * (weight + 1)).to(original_dtype)
    weight_chunked = weight.chunk(num_modality, dim=0)
    t_list = dispatcher.dispatch(t)
    for i in range(num_modality):
        t_list[i] = t_list[i] * (weight_chunked[i] + 1)
    return torch.cat(t_list, dim=0).to(original_dtype)


def _ref_linear(x, weight, num_experts, dispatcher, output_dtype=None):
    output_dtype = x.dtype if output_dtype is None else output_dtype

    def bf16_linear(inp, w):
        return torch.matmul(inp.to(_BF16), w.to(_BF16).t()).to(output_dtype)

    if num_experts == 1:
        return bf16_linear(x, weight)
    inputs = dispatcher.dispatch(x)
    chunks = weight.chunk(num_experts, dim=0)
    return torch.cat([bf16_linear(inputs[i], chunks[i]) for i in range(num_experts)], dim=0)


def _ref_swiglu7(x, alpha=1.702, limit=7.0):
    out_dtype = x.dtype
    x = x.to(torch.float32)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    return (out_glu * (x_linear + 1)).to(out_dtype)


def _ref_rotary(x, cos, sin):
    ro_dim = cos.shape[-1] * 2
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(-2)  # "... d -> ... 1 (2 d)"
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(-2)
    x1, x2 = x[..., :ro_dim].chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat([x[..., :ro_dim] * cos + rotated * sin, x[..., ro_dim:]], dim=-1)


def _ref_layer(params, prefix, cfg, num_modality, hidden_states, rope, dispatcher, plan):
    p = {name[len(prefix) :]: value for name, value in params.items() if name.startswith(prefix)}
    hq, hkv, d = cfg.num_heads_q, cfg.num_heads_kv, cfg.head_dim
    # Attention
    h = _ref_rms_norm(hidden_states, p["attention.pre_norm.weight"], num_modality, dispatcher).to(_BF16)
    qkv = _ref_linear(h, p["attention.linear_qkv.weight"], num_modality, dispatcher).to(torch.float32)
    q, k, v = torch.split(qkv, [hq * d, hkv * d, hkv * d], dim=1)
    q = q.view(-1, hq, d)
    k = k.view(-1, hkv, d)
    v = v.view(-1, hkv, d)
    g = _ref_linear(h, p["attention.linear_g.weight"], num_modality, dispatcher).to(torch.float32)
    g = g.unsqueeze(-1)
    q = _ref_rms_norm(q, p["attention.q_norm.weight"], num_modality, dispatcher)
    k = _ref_rms_norm(k, p["attention.k_norm.weight"], num_modality, dispatcher)
    q = q[dispatcher.inv_permute_mapping].unsqueeze(0)
    k = k[dispatcher.inv_permute_mapping].unsqueeze(0)
    v = v[dispatcher.inv_permute_mapping].unsqueeze(0)
    sin_emb, cos_emb = rope.tensor_split(2, -1)
    q = _ref_rotary(q, cos_emb, sin_emb)
    k = _ref_rotary(k, cos_emb, sin_emb)
    q, k, v = q.to(_BF16).squeeze(0), k.to(_BF16).squeeze(0), v.to(_BF16).squeeze(0)
    out = torch_window_attention_reference(q, k, v, plan, softmax_scale=d**-0.5)
    out = out[dispatcher.permute_mapping]
    out = out * torch.sigmoid(g)
    out = out.view(-1, hq * d).to(_BF16)
    attn_out = _ref_linear(out, p["attention.linear_proj.weight"], num_modality, dispatcher)
    hidden_states = hidden_states + attn_out
    # MLP
    x = _ref_rms_norm(hidden_states, p["mlp.pre_norm.weight"], num_modality, dispatcher).to(_BF16)
    x = _ref_linear(x, p["mlp.up_gate_proj.weight"], num_modality, dispatcher).to(torch.float32)
    x = _ref_swiglu7(x).to(_BF16)
    x = _ref_linear(x, p["mlp.down_proj.weight"], num_modality, dispatcher).to(torch.float32)
    return hidden_states + x


def _ref_rope(coords, bands):
    coords_xyz = coords[:, :3]
    sizes = coords[:, 3:6]
    refs = coords[:, 6:9]
    scales = (refs - 1) / (sizes - 1)
    scales[(refs == 1) & (sizes == 1)] = 1
    centers = (sizes - 1) / 2
    centers[:, 0] = 0
    coords_xyz = coords_xyz - centers
    proj = coords_xyz.unsqueeze(-1) * scales.unsqueeze(-1) * bands
    return torch.cat((proj.sin(), proj.cos()), dim=1).flatten(1)


def _ref_transformer(model, video, text, coords, modality, plan):
    cfg = model.config
    params = dict(model.named_parameters())
    x = torch.cat((F.pad(video, (0, text.shape[1] - video.shape[1])), text))
    dispatcher = _RefDispatcher(modality)
    video_mask, text_mask = modality == _VIDEO, modality == _TEXT
    rope = _ref_rope(coords, params["pre_adapter.rope.bands"])
    out = torch.zeros(x.shape[0], cfg.hidden_size, dtype=x.dtype)
    out[text_mask] = F.linear(
        x[text_mask, : cfg.text_in_channels],
        params["pre_adapter.text_embedder.weight"],
        params["pre_adapter.text_embedder.bias"],
    )
    out[video_mask] = F.linear(
        x[video_mask, : cfg.video_in_channels],
        params["pre_adapter.video_embedder.weight"],
        params["pre_adapter.video_embedder.bias"],
    )
    hidden_states = out.to(_BF16)[dispatcher.permute_mapping]
    for i in range(cfg.num_layers):
        hidden_states = _ref_layer(
            params, f"block.layers.{i}.", cfg, cfg.num_modalities(i), hidden_states, rope, dispatcher, plan
        )
    hidden_states = hidden_states[dispatcher.inv_permute_mapping]
    x_video = hidden_states[video_mask].to(torch.float32)
    x_video = _ref_rms_norm(x_video, params["post_adapter.final_norm_video.weight"], 1, None)
    return F.linear(x_video, params["post_adapter.final_linear_video.weight"])


def _video_coords(grid):
    t, h, w = grid
    mesh = torch.meshgrid(*(torch.arange(n, dtype=torch.float32) for n in grid), indexing="ij")
    flat = torch.stack(mesh, dim=-1).reshape(-1, 3)
    meta = torch.tensor([t, h, w, t, h, w], dtype=torch.float32).expand(flat.shape[0], -1)
    return torch.cat((flat, meta), dim=-1)


def _text_coords(length):
    t = torch.arange(length, dtype=torch.float32) - length
    zeros = torch.zeros(length)
    meta = torch.tensor([length, 1, 1, 1, 1, 1], dtype=torch.float32).expand(length, -1)
    return torch.cat((torch.stack((t, zeros, zeros), dim=-1), meta), dim=-1)


def _forward_inputs(cfg, grid, text_len, seed=1):
    generator = torch.Generator().manual_seed(seed)
    num_video = math.prod(grid)
    video = torch.randn(num_video, cfg.video_in_channels, generator=generator)
    text = torch.randn(text_len, cfg.text_in_channels, generator=generator)
    block_order = build_reference_window_ranges(grid, text_len).token_order[:num_video].long()
    coords = torch.cat((_video_coords(grid)[block_order], _text_coords(text_len)))
    modality = torch.cat((torch.full((num_video,), _VIDEO), torch.full((text_len,), _TEXT)))
    return video, text, coords, modality, build_window_attention_plan(grid, text_len)


def _read_manifest() -> tuple[dict[str, tuple[str, tuple[int, ...]]], int]:
    """Return ``{name: (dtype, shape)}`` and the total byte size from the manifest."""

    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    total_size = -1
    for line in _MANIFEST.read_text().splitlines():
        if line.startswith("# total_size "):
            total_size = int(line.split()[-1])
        if not line or line.startswith("#"):
            continue
        name, dtype, shape = line.split()
        assert name not in tensors
        tensors[name] = (dtype, tuple(int(n) for n in shape.split(",")))
    return tensors, total_size


def test_meta_model_state_dict_matches_checkpoint_manifest() -> None:
    tensors, total_size = _read_manifest()
    assert len(tensors) == 281
    nbytes = {"BF16": 2, "F32": 4}
    assert sum(math.prod(shape) * nbytes[dtype] for dtype, shape in tensors.values()) == total_size == 13_484_611_648

    with torch.device("meta"):
        model = Magi2RefinerTransformer()
    state = {name: (_SAFETENSORS_DTYPES[t.dtype], tuple(t.shape)) for name, t in model.state_dict().items()}

    assert state == tensors
    assert {name for name, _ in model.named_parameters()} == set(state)
    assert sum(p.numel() for p in model.parameters()) == _CHECKPOINT_PARAMETERS
    assert sum(math.prod(shape) for _, shape in tensors.values()) == _CHECKPOINT_PARAMETERS


def test_multimodal_layers_carry_video_audio_text_experts() -> None:
    with torch.device("meta"):
        model = Magi2RefinerTransformer()
    for index, layer in enumerate(model.layers):
        expected = 3 if index in (0, 1, 28, 29) else 1
        norms = (layer.attention.pre_norm, layer.attention.q_norm, layer.attention.k_norm, layer.mlp.pre_norm)
        assert [norm.num_modality for norm in norms] == [expected] * 4
        assert layer.attention.q_norm.weight.shape == (expected * 128,)
        assert layer.attention.q_norm.out_dtype == torch.float32
        assert layer.attention.linear_qkv.num_experts == expected
        assert layer.mlp.down_proj.weight.shape == (expected * 4096, 10920)


def _write_checkpoint(directory: Path, tensors: dict[str, torch.Tensor], weight_map: dict[str, str] | None = None):
    directory.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    shards = {name: f"model-0000{1 + i % 2}-of-00002.safetensors" for i, name in enumerate(names)}
    for shard in sorted(set(shards.values())):
        save_file({n: tensors[n].contiguous() for n in names if shards[n] == shard}, str(directory / shard))
    index = {"metadata": {"total_size": 0}, "weight_map": weight_map or shards}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))


def _checkpoint_tensors(seed: int = 3) -> dict[str, torch.Tensor]:
    source = _tiny_model(seed)
    tensors = {name: tensor.detach().clone() for name, tensor in source.state_dict().items()}
    # A non-formula value proves the loader covers the bands key.
    tensors["pre_adapter.rope.bands"] = torch.linspace(0.5, 0.25, tensors["pre_adapter.rope.bands"].numel())
    return tensors


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_strict_loader_loads_every_tensor(tmp_path: Path) -> None:
    tensors = _checkpoint_tensors()
    _write_checkpoint(tmp_path / "refiner", tensors)
    model = Magi2RefinerTransformer(_tiny_config())

    loaded = model.load_checkpoint(tmp_path / "refiner")

    assert loaded == set(tensors) == set(model.state_dict())
    for name, tensor in model.state_dict().items():
        assert tensor.dtype == tensors[name].dtype
        assert torch.equal(tensor, tensors[name]), name
    assert torch.equal(model.pre_adapter.rope.bands, tensors["pre_adapter.rope.bands"])


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda t: t.pop("block.layers.1.mlp.down_proj.weight"), r"1 missing \['block.layers.1.mlp.down_proj"),
        (lambda t: t.update({"block.layers.1.mlp.extra.weight": torch.zeros(2)}), r"1 unexpected"),
        (
            lambda t: t.update({"block.layers.0.attention.q_norm.weight": torch.zeros(32)}),
            r"block.layers.0.attention.q_norm.weight.*F32\[32\].*expected torch.float32\[96\]",
        ),
        (
            lambda t: t.update({"block.layers.2.attention.linear_g.weight": torch.zeros(8, 256)}),
            r"block.layers.2.attention.linear_g.weight.*is F32\[8, 256\], expected torch.bfloat16",
        ),
    ],
    ids=["missing", "unexpected", "wrong-shape", "wrong-dtype"],
)
def test_strict_loader_rejects_mismatches_before_writing(tmp_path: Path, mutate, match: str) -> None:
    tensors = _checkpoint_tensors()
    mutate(tensors)
    _write_checkpoint(tmp_path, tensors)
    model = _tiny_model(seed=11)
    before = _snapshot(model)

    with pytest.raises(ValueError, match=match):
        model.load_checkpoint(tmp_path)

    after = _snapshot(model)
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_strict_loader_checks_shards_against_the_index(tmp_path: Path) -> None:
    tensors = _checkpoint_tensors()
    _write_checkpoint(tmp_path, tensors)
    index_path = tmp_path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    name = "block.layers.3.mlp.pre_norm.weight"
    shards = sorted(set(index["weight_map"].values()))
    index["weight_map"][name] = next(s for s in shards if s != index["weight_map"][name])
    index_path.write_text(json.dumps(index))

    with pytest.raises(ValueError, match="does not match the index"):
        _tiny_model().load_checkpoint(tmp_path)


def test_strict_loader_rejects_shard_paths_and_missing_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model.safetensors.index.json"):
        _tiny_model().load_checkpoint(tmp_path)

    tensors = _checkpoint_tensors()
    weight_map = {name: "../model.safetensors" for name in tensors}
    _write_checkpoint(tmp_path, tensors, weight_map)
    with pytest.raises(ValueError, match="is not a file name"):
        _tiny_model().load_checkpoint(tmp_path)


@pytest.mark.parametrize("text_len", [21, 0])
def test_transformer_forward_matches_reference_transcription_bitwise(text_len: int) -> None:
    model = _tiny_model()
    # (9, 4, 8) ends with a one-frame temporal block; L=21 leaves a partial text page.
    video, text, coords, modality, plan = _forward_inputs(model.config, (9, 4, 8), text_len)

    with torch.no_grad():
        actual = model(video, text, coords, modality, plan)
        expected = _ref_transformer(model, video, text, coords, modality, plan)

    assert actual.dtype == torch.float32
    assert actual.shape == (video.shape[0], model.config.video_in_channels)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("layer_index", "input_dtype"),
    [(0, torch.bfloat16), (1, torch.float32), (3, torch.float32)],
    ids=["mm-bf16-residual", "dense-fp32-residual", "mm-fp32-residual"],
)
def test_layer_matches_reference_transcription_bitwise(layer_index: int, input_dtype: torch.dtype) -> None:
    model = _tiny_model()
    cfg = model.config
    grid, text_len = (9, 4, 8), 21
    plan = build_window_attention_plan(grid, text_len)
    generator = torch.Generator().manual_seed(5)
    num_tokens = plan.num_tokens
    # Interleaved video, audio and text tokens exercise the modality permutation of every expert.
    modality = torch.randint(0, 3, (num_tokens,), generator=generator)
    assert torch.bincount(modality, minlength=3).min() > 0
    hidden = torch.randn(num_tokens, cfg.hidden_size, generator=generator).to(input_dtype)
    coords = torch.cat((_video_coords(grid), _text_coords(text_len)))
    rope = model.pre_adapter.rope(coords)
    dispatcher = ModalityDispatcher(modality, 3)
    layer = model.layers[layer_index]
    metadata_extra = {"magi2_window_plan": plan, "magi2_split_sizes": [num_tokens]}

    with torch.no_grad():
        permuted = dispatcher.permute(hidden)
        actual = layer(permuted, rope, dispatcher, AttentionMetadata(extra=metadata_extra))
        expected = _ref_layer(
            dict(model.named_parameters()),
            f"block.layers.{layer_index}.",
            cfg,
            cfg.num_modalities(layer_index),
            permuted,
            rope,
            _RefDispatcher(modality),
            plan,
        )

    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


def test_layer_regions_bracket_the_window_attention() -> None:
    assert all(layer.region_methods == ("_attention_input", "_attention_output") for layer in _tiny_model().layers)


@pytest.mark.parametrize("text_len", [21, 0])
def test_compiled_regions_match_eager_forward_bitwise(text_len: int) -> None:
    eager = _tiny_model()
    compiled = _tiny_model()
    compiled.compile_regions(fullgraph=True, backend="eager", dynamic=False)
    video, text, coords, modality, plan = _forward_inputs(eager.config, (9, 4, 8), text_len)

    with torch.no_grad():
        expected = eager(video, text, coords, modality, plan)
        actual = compiled(video, text, coords, modality, plan)

    assert all(
        hasattr(getattr(layer, name), "_torchdynamo_orig_callable")
        for layer in compiled.layers
        for name in layer.region_methods
    )
    assert torch.equal(actual, expected)


def test_static_regions_compile_every_prompt_length_under_the_pipeline_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.magi2 import pipeline_magi2

    class _Transformer(torch.nn.Module):
        def compile_regions(self, **compile_kwargs: Any) -> None:
            pass

    monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 8)
    monkeypatch.setattr(torch._dynamo.config, "accumulated_recompile_limit", 256)
    pipeline = object.__new__(pipeline_magi2.Magi2Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(model="sand-ai/MAGI-2-preview", model_class_name="Magi2Pipeline")
    pipeline.transformer = _Transformer()
    with monkeypatch.context() as musa:
        musa.setattr(pipeline_magi2.current_omni_platform, "is_musa", lambda: True)
        pipeline.setup_compile()
    compile_kwargs = {name: value for name, value in pipeline._region_compile_kwargs.items() if name != "options"}

    model = _tiny_model()
    model.compile_regions(backend="eager", **compile_kwargs)
    torch._dynamo.reset()
    try:
        with torch.no_grad():
            # Each length compiles three variants of both regions: the bf16 first
            # layer, then fp32 inputs with one and with three modality experts.
            for text_len in (5, 6, 7, 8):
                model(*_forward_inputs(model.config, (9, 4, 8), text_len))
    finally:
        torch._dynamo.reset()


def test_cat_rows_copy_path_matches_torch_cat(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = torch.Generator().manual_seed(9)
    parts = [torch.randn(rows, 6, generator=generator).to(_BF16) for rows in (3, 0, 5)]
    monkeypatch.setattr(magi2_layers, "_CAT_ROWS_MAX_NUMEL", 1)
    copied = magi2_layers.cat_rows(parts)
    assert torch.equal(copied, torch.cat(parts, dim=0))
    assert copied.dtype == _BF16


def test_layer_is_bitwise_unchanged_when_concatenations_take_the_copy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _tiny_model()
    grid, text_len = (9, 4, 8), 21
    plan = build_window_attention_plan(grid, text_len)
    generator = torch.Generator().manual_seed(5)
    modality = torch.randint(0, 3, (plan.num_tokens,), generator=generator)
    hidden = torch.randn(plan.num_tokens, model.config.hidden_size, generator=generator)
    rope = model.pre_adapter.rope(torch.cat((_video_coords(grid), _text_coords(text_len))))
    dispatcher = ModalityDispatcher(modality, 3)
    metadata = AttentionMetadata(extra={"magi2_window_plan": plan, "magi2_split_sizes": [plan.num_tokens]})
    permuted = dispatcher.permute(hidden)

    with torch.no_grad():
        expected = model.layers[0](permuted, rope, dispatcher, metadata)
        monkeypatch.setattr(magi2_layers, "_CAT_ROWS_MAX_NUMEL", 1)
        actual = model.layers[0](permuted, rope, dispatcher, metadata)

    assert torch.equal(actual, expected)


def test_dtype_chain_gate_and_swiglu_layout() -> None:
    model = _tiny_model()
    video, text, coords, modality, plan = _forward_inputs(model.config, (9, 4, 8), 21)
    layer = model.layers[0]
    captured: dict[str, tuple[tuple[Any, ...], torch.Tensor]] = {}

    def keep(name):
        return lambda _module, args, output: captured.update({name: (args, output)})

    hooks = [
        layer.register_forward_hook(keep("layer")),
        layer.mlp.register_forward_hook(keep("mlp")),
        layer.attention.linear_g.register_forward_hook(keep("gate")),
        layer.attention.window_attention.register_forward_hook(keep("kernel")),
        layer.attention.linear_proj.register_forward_hook(keep("proj")),
        layer.mlp.up_gate_proj.register_forward_hook(keep("up_gate")),
        layer.mlp.down_proj.register_forward_hook(keep("down")),
        model.layers[1].register_forward_hook(keep("layer1")),
    ]
    with torch.no_grad():
        model(video, text, coords, modality, plan)
    for hook in hooks:
        hook.remove()

    # The residual enters layer 0 in bf16, stays bf16 through the attention add, and
    # becomes fp32 at the MLP add.
    (layer_input, *_), layer_output = captured["layer"]
    (mlp_input, *_), mlp_output = captured["mlp"]
    assert layer_input.dtype == torch.bfloat16
    assert captured["proj"][1].dtype == torch.bfloat16
    assert mlp_input.dtype == torch.bfloat16
    assert mlp_output.dtype == torch.float32
    assert layer_output.dtype == torch.float32
    assert captured["layer1"][0][0].dtype == torch.float32

    # The bf16 attention output is gated in fp32, not with a bf16-rounded gate.
    dispatcher = captured["layer"][0][2]
    gate = captured["gate"][1]
    attention = dispatcher.permute(captured["kernel"][1])
    assert attention.dtype == torch.bfloat16
    projected = captured["proj"][0][0]
    fp32_gate = (attention * torch.sigmoid(gate.float()).unsqueeze(-1)).reshape(projected.shape).to(_BF16)
    bf16_gate = (attention * torch.sigmoid(gate.float()).to(_BF16).unsqueeze(-1)).reshape(projected.shape)
    assert torch.equal(projected, fp32_gate)
    assert not torch.equal(projected, bf16_gate)

    # SwiGLU7 takes the gate from the even features of up_gate_proj.
    up_gate = captured["up_gate"][1]
    activated = captured["down"][0][0]
    assert up_gate.dtype == activated.dtype == torch.bfloat16
    swapped = torch.stack((up_gate[..., 1::2], up_gate[..., ::2]), dim=-1).flatten(-2)
    assert torch.equal(activated, _ref_swiglu7(up_gate.float()).to(_BF16))
    assert not torch.equal(activated, _ref_swiglu7(swapped.float()).to(_BF16))


def test_rope_bands_are_the_released_frequencies() -> None:
    model = Magi2RefinerTransformer(_tiny_config(head_dim=128, num_query_groups=1))
    bands = model.state_dict()["pre_adapter.rope.bands"]
    expected = 1.0 / 10000.0 ** (torch.arange(16, dtype=torch.float32) / 16)
    assert bands.dtype == torch.float32
    assert torch.equal(bands, expected)


def test_rope_embedding_matches_reference_layout() -> None:
    model = _tiny_model(head_dim=128, num_query_groups=1)
    grid = (9, 4, 8)
    coords = torch.cat((_video_coords(grid), _text_coords(7), _text_coords(1)))
    rope = model.pre_adapter.rope(coords)
    bands = model.pre_adapter.rope.bands

    assert rope.dtype == torch.float32
    assert rope.shape == (coords.shape[0], model.config.rotary_dim)
    assert torch.equal(rope, _ref_rope(coords.clone(), bands))
    num_video = math.prod(grid)
    # [sin_t, sin_h, sin_w, cos_t, cos_h, cos_w], one column per band.
    t, h, w = 5.0, 3.0, 6.0
    row = rope[int((t * grid[1] + h) * grid[2] + w)]
    offsets = torch.tensor([t, h - (grid[1] - 1) / 2, w - (grid[2] - 1) / 2])
    angles = offsets[:, None] * bands
    assert torch.equal(row, torch.cat((angles.sin(), angles.cos())).flatten())
    # Text of length L > 1 has zero angles; a single text token does not.
    text = rope[num_video : num_video + 7]
    assert torch.equal(text[:, :48].abs(), torch.zeros(7, 48))
    assert torch.equal(text[:, 48:], torch.ones(7, 48))
    single = rope[-1]
    assert torch.equal(single[:16], (-bands).sin())
    assert torch.equal(single[16:48].abs(), torch.zeros(32))


def test_rotary_covers_the_first_96_dims_in_neox_halves() -> None:
    model = _tiny_model(head_dim=128, num_query_groups=1, multimodal_layers=(0,), num_layers=1)
    video, text, coords, modality, plan = _forward_inputs(model.config, (9, 4, 8), 5)
    attention = model.layers[0].attention
    captured: dict[str, torch.Tensor] = {}
    hooks = [
        attention.q_norm.register_forward_hook(lambda _m, _a, out: captured.update(q=out)),
        attention.window_attention.register_forward_pre_hook(lambda _m, args: captured.update(kernel_q=args[0])),
    ]
    with torch.no_grad():
        model(video, text, coords, modality, plan)
    for hook in hooks:
        hook.remove()

    # Video then text is already modality order, so the q-norm output is in sequence order.
    q, kernel_q = captured["q"], captured["kernel_q"]
    assert q.dtype == torch.float32 and kernel_q.dtype == torch.bfloat16
    num_video = video.shape[0]
    assert torch.equal(kernel_q[..., 96:], q[..., 96:].to(_BF16))
    assert not torch.equal(kernel_q[:num_video, :, :96], q[:num_video, :, :96].to(_BF16))
    # Text tokens have zero angles: the rotation is the identity.
    assert torch.equal(kernel_q[num_video:], q[num_video:].to(_BF16))
    # NeoX halves: dim i rotates with dim i + 48 by the angle of column i.
    sin, cos = model.pre_adapter.rope(coords)[:, None, :].tensor_split(2, dim=-1)
    first, second = q[..., :48], q[..., 48:96]
    expected = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    assert torch.equal(kernel_q[..., :96], expected.to(_BF16))


def test_forward_rejects_audio_tokens_and_mismatched_features() -> None:
    model = _tiny_model(num_layers=1, multimodal_layers=(0,))
    video, text, coords, modality, plan = _forward_inputs(model.config, (9, 4, 8), 3)

    audio = modality.clone()
    audio[-1] = _AUDIO
    with pytest.raises(ValueError, match="video or text"):
        model(video, text[:-1], coords, audio, plan)
    with pytest.raises(ValueError, match="2 text features|3 text tokens"):
        model(video, text[:2], coords, modality, plan)
    with pytest.raises(ValueError, match=r"coords \[n, 9\]"):
        model(video, text, coords[:, :8], modality, plan)
