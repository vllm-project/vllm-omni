# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of RBS configuration and model-owned token geometry.

Load the production definitions without importing vLLM or NPU libraries. These
checks do not execute or simulate the native attention kernels.
"""

import ast
import dataclasses
import functools
import inspect
import math
import sys
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import ModuleType

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]
ROOT = Path(__file__).resolve().parents[4]


def definitions(name: str, relative: str, names: set[str], dependencies: dict) -> ModuleType:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ModuleType(name)
    module.__dict__.update(dependencies)
    sys.modules[name] = module
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), relative, "exec"),
        module.__dict__,
    )
    return module


@pytest.fixture
def config():
    return definitions(
        "_rbs_attention_config",
        "vllm_omni/diffusion/data.py",
        {
            "parse_kv_cache_skip_selector",
            "_in_range",
            "RainFusionPrecision",
            "BlockSparseSpec",
            "RainFusionSpec",
            "SkipSoftmaxSpec",
            "AttnQuantSpec",
            "SpargeAttnSpec",
            "AttentionSpec",
            "AttentionConfig",
        },
        {
            "dataclass": dataclasses.dataclass,
            "field": dataclasses.field,
            "Enum": Enum,
            "Mapping": Mapping,
            "math": math,
            "BLOCK_SPARSE_BACKENDS": frozenset({"RAINFUSION_ATTN"}),
        },
    )


@pytest.fixture
def geometry():
    return definitions(
        "_rbs_layout",
        "vllm_omni/diffusion/attention/backends/abstract.py",
        {"ImageTokenSpan", "VideoTokenSpan", "VideoTokenLayout", "ReferenceImageLayout", "AttentionMetadata"},
        {"dataclass": dataclasses.dataclass, "field": dataclasses.field},
    )


@pytest.fixture
def progress():
    @dataclasses.dataclass
    class Progress:
        denoise_step_idx: int | None = 0
        total_denoise_steps: int | None = 8

    return Progress()


@pytest.fixture
def backend(progress, mocker):
    return definitions(
        "_rbs_backend",
        "vllm_omni/diffusion/attention/backends/rainfusion_attn.py",
        {
            "RainFusionConfig",
            "RainFusionPlan",
            "RainFusionAttentionImpl",
            "_supports_video_spans",
            "_supports_image_sparsity",
        },
        {
            "dataclass": dataclasses.dataclass,
            "replace": dataclasses.replace,
            "AttentionImpl": object,
            "math": math,
            "torch": torch,
            "cast": lambda annotation, value: value,
            "AttentionMetadata": object,
            "functools": functools,
            "inspect": inspect,
            "get_forward_context": lambda: progress,
            "is_forward_context_available": lambda: True,
            "_mindiesd_supports_precision": lambda: True,
            "_INPUT_LAYOUT": "BSND",
            "_INNER_PRECISE": 0,
            "logger": mocker.Mock(),
            "_MIN_VIDEO_BLOCKS": 32,
            "_BLOCK_SIZE": 128,
        },
    )


@pytest.mark.parametrize("ratio", [0.0, 0.2, 0.8, 1.0])
def test_image_sparsity_reaches_backend_from_standard_config(config, backend, ratio):
    parsed = config.AttentionConfig(
        default={
            "backend": "RAINFUSION_ATTN",
            "block_sparse": {"sparsity": 0.0, "precision": "mix"},
            "rainfusion": {"image_sparsity": ratio},
        }
    )
    kwargs = parsed.default.backend_kwargs()
    resolved = backend.RainFusionConfig.from_backend_kwargs(kwargs)
    assert resolved.image_sparsity == ratio
    assert resolved.enabled is (ratio > 0)
    if ratio == 0.0:
        assert "image_sparsity" not in kwargs


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf"), True, "0.9"])
def test_invalid_image_sparsity_is_rejected(config, backend, ratio):
    for cls in (config.RainFusionSpec, backend.RainFusionConfig):
        with pytest.raises(ValueError, match="image_sparsity"):
            cls(image_sparsity=ratio)


def test_rainfusion_options_reject_unrelated_backend(config):
    with pytest.raises(ValueError, match="only supported"):
        config.AttentionSpec(backend="FLASH_ATTN", rainfusion={"image_sparsity": 0.5})


def test_full_plan_carries_image_geometry_and_checks_alignment(backend, geometry):
    impl = backend.RainFusionAttentionImpl.__new__(backend.RainFusionAttentionImpl)
    layout = geometry.VideoTokenLayout(
        used_len=8192,
        image_spans=(geometry.ImageTokenSpan(16, (8, 20)), geometry.ImageTokenSpan(300, (16, 16))),
        video_spans=(geometry.VideoTokenSpan(start=1024, latent_grid=(4, 32, 56), role="target"),),
    )
    impl.rainfusion = backend.RainFusionConfig(sparsity=0, image_sparsity=0.75)
    plan = impl._resolve_multi_span_plan(layout, 8192)
    assert plan.video_spans == [{"start": 1024, "latent_shape": [4, 32, 56]}]
    assert plan.image_spans == [{"start": 16, "latent_shape": [8, 20]}, {"start": 300, "latent_shape": [16, 16]}]
    # The 160-token image needs 96 context tokens before the next span.
    tight = dataclasses.replace(
        layout,
        used_len=7328,
        image_spans=(geometry.ImageTokenSpan(0, (8, 20)),),
        video_spans=(geometry.VideoTokenSpan(start=160, latent_grid=(4, 32, 56), role="target"),),
    )
    assert impl._resolve_multi_span_plan(tight, tight.used_len) is None
    assert impl._resolve_multi_span_plan(dataclasses.replace(layout, image_spans=()), 8192) is None
    impl.rainfusion = backend.RainFusionConfig(sparsity=0.8)
    assert impl._resolve_multi_span_plan(layout, 8192).image_spans is None


@pytest.mark.parametrize("precision", ["bf16", "mix"])
def test_default_compact_keeps_existing_path(backend, geometry, monkeypatch, mocker, precision):
    calls = []

    def sparse_attention(q, k, v, *, video_spans, **kwargs):
        calls.append((q.shape[1], k.shape[1]))
        return torch.zeros_like(q)

    native = ModuleType("mindiesd")
    native.__dict__["sparse_attention"] = sparse_attention
    platform = ModuleType("mindiesd.utils.get_platform")
    platform.__dict__["is_a5_device"] = lambda: True
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setitem(sys.modules, platform.__name__, platform)
    instance = backend.RainFusionAttentionImpl.__new__(backend.RainFusionAttentionImpl)
    instance.rainfusion = backend.RainFusionConfig(precision=precision)
    instance.qkv_layout = "BSND"
    instance.softmax_scale = 0.25
    instance._forward_rectangular_npu = mocker.Mock(side_effect=AssertionError("default route changed"))
    instance._resolve_plan = lambda metadata: backend.RainFusionPlan(
        used_len=12, video_spans=[{"start": 4, "latent_shape": [1, 1, 8]}]
    )
    layout = geometry.VideoTokenLayout(
        used_len=8, video_spans=(geometry.VideoTokenSpan(start=0, latent_grid=(1, 1, 8), role="target"),)
    )
    metadata = geometry.AttentionMetadata(
        extra={"max_seqlen_q": 8, "max_seqlen_k": 12, "minimax_h3_compact_reference_rows": 4}, video_layout=layout
    )
    q = torch.ones(1, 8, 2, 16)
    kv = torch.ones(1, 12, 2, 16)
    assert instance.forward_npu(q, kv, kv, metadata).shape == q.shape
    assert calls == [(12, 12)]
    instance._forward_rectangular_npu.assert_not_called()


@pytest.mark.parametrize("override", [None, "float"])
@pytest.mark.parametrize("image_sparsity", [0.0, 0.9])
def test_eight_steps_dispatch_full_then_rectangular(
    backend, config, geometry, progress, monkeypatch, mocker, override, image_sparsity
):
    calls = []
    full_len = 4480
    ref_rows = 256
    active_len = full_len - ref_rows
    images = [{"start": 128, "latent_shape": [16, 16]}]

    def sparse_attention(q, k, v, *, precision, video_spans, sparsity, image_spans=None, image_sparsity=None, **kwargs):
        assert q.shape[1] == k.shape[1] == full_len
        calls.append(("full", sparsity, image_spans, image_sparsity, precision))
        return torch.zeros_like(q)

    def run(q, k, v, plan, sparsity, scale, *, image_sparsity, precision):
        assert q.shape[1] == active_len < k.shape[1] == full_len
        calls.append(("rectangular", sparsity, None, image_sparsity, precision))
        return torch.zeros_like(q)

    native = ModuleType("mindiesd")
    native.__dict__["sparse_attention"] = sparse_attention
    platform = ModuleType("mindiesd.utils.get_platform")
    platform.__dict__["is_a5_device"] = lambda: True
    rectangular = ModuleType("vllm_omni.diffusion.attention.ops.rainfusion_rectangular")
    get_plan = mocker.Mock()
    rectangular.__dict__.update(get_plan=get_plan, run=run)
    for module in (native, platform, rectangular):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    instance = backend.RainFusionAttentionImpl.__new__(backend.RainFusionAttentionImpl)
    instance.rainfusion = backend.RainFusionConfig.from_backend_kwargs(
        config.AttentionSpec(
            backend="RAINFUSION_ATTN",
            block_sparse={"sparsity": 0.8, "precision": "mix"},
            rainfusion={"image_sparsity": image_sparsity},
        ).backend_kwargs()
    )
    instance.layer_idx = None
    instance.qkv_layout = "BSND"
    instance.softmax_scale = 0.25
    span = geometry.VideoTokenSpan(start=384, latent_grid=(8, 16, 32), role="target")
    original = geometry.VideoTokenLayout(
        used_len=full_len, video_spans=(span,), image_spans=(geometry.ImageTokenSpan(128, (16, 16)),)
    )
    compact = geometry.VideoTokenLayout(used_len=active_len, video_spans=(dataclasses.replace(span, start=128),))
    physical = geometry.VideoTokenLayout(used_len=full_len, video_spans=(span,))
    image_layout = geometry.ReferenceImageLayout(original, tuple(range(128, 384)))
    full = torch.ones(1, full_len, 1, 4)
    compact_q = torch.ones(1, active_len, 1, 4)
    schedule = definitions(
        "_rbs_schedule",
        "vllm_omni/diffusion/attention/schedule.py",
        {"AttentionScheduleRange", "_validate_step", "validate_attention_profile_name", "select_attention_profile"},
        {"dataclass": dataclasses.dataclass},
    )
    ranges = (schedule.AttentionScheduleRange(2, None, "late"),)
    for step in range(8):
        profile = schedule.select_attention_profile(ranges, step, total_steps=8)
        instance.rainfusion = dataclasses.replace(instance.rainfusion, sparsity=0.85 if profile == "late" else 0.8)
        progress.denoise_step_idx = step
        q = compact_q if step else full
        extra = {"max_seqlen_q": full_len}
        metadata = geometry.AttentionMetadata(extra=extra, video_layout=original)
        if step:
            extra.update(
                max_seqlen_q=active_len,
                max_seqlen_k=full_len,
                minimax_h3_compact_reference_rows=ref_rows,
                minimax_h3_resident_reference_kv=True,
                minimax_h3_reference_kv_video_layout=physical,
                minimax_h3_reference_image_layout=image_layout,
            )
            metadata.video_layout = compact
        if override is not None:
            extra["kv_cache_dtype"] = override
        assert instance.forward_npu(q, full, full, metadata).shape == q.shape
    precision = "bf16" if override == "float" else "mix"
    assert (
        calls
        == [
            ("full", 0.8, images if image_sparsity else None, image_sparsity or None, precision),
            ("rectangular", 0.8, None, image_sparsity, precision),
        ]
        + [("rectangular", 0.85, None, image_sparsity, precision)] * 6
    )
    assert get_plan.call_args.kwargs == {
        "image_spans": ((128, (16, 16)),) if image_sparsity else (),
        "image_layout": image_layout if image_sparsity else None,
    }


@pytest.mark.parametrize("precision", ["bf16", "mix"])
@pytest.mark.parametrize("partial", [False, True])
def test_rectangular_image_mask_and_cached_row_mapping(geometry, monkeypatch, mocker, precision, partial):
    # RF order for a layout with text between two images and a reference video.
    # Block 5 contains the second image's 32-token tail and 96 context tokens.
    perm = torch.tensor(
        [
            *range(24, 280),
            *range(312, 568),
            *range(568, 728),
            *range(24),
            *range(280, 312),
            *range(728, 768),
            *range(768, 1152),
        ]
    )
    first = {2, 6}
    protected = {0, 1, 4, 5}
    video_rows = (2, 3, 6, 7, 8)
    packed_mask_input = mocker.Mock(return_value=(perm, None, protected, first, (((0, 1), (4,)), video_rows)))
    rf_v2 = ModuleType("mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2")
    rf_v2.__dict__["_normalize_image_spans"] = lambda spans, length: spans
    monkeypatch.setitem(sys.modules, rf_v2.__name__, rf_v2)
    rectangular = definitions(
        "_rbs_rectangular",
        "vllm_omni/diffusion/attention/ops/rainfusion_rectangular.py",
        {"Plan", "get_plan", "run"},
        {
            "dataclass": dataclasses.dataclass,
            "lru_cache": functools.lru_cache,
            "math": math,
            "torch": torch,
            "_normalize_video_spans": lambda spans, length: spans,
            "_multi_span_permutation": packed_mask_input,
            "avgpool": lambda values, block_size, layout: values.reshape(1, -1, block_size, 1, 1).mean(2),
        },
    )
    calls = []

    def native(q, k, v, mask, scale):
        calls.append((q.clone(), k.clone(), mask.clone()))
        return q

    rectangular.__dict__["native_bf16"] = native if precision == "bf16" else mocker.Mock(side_effect=AssertionError)
    rectangular.__dict__["native_mix"] = native if precision == "mix" else mocker.Mock(side_effect=AssertionError)
    ref_positions = tuple(range(24, 280)) + tuple(range(312, 728))
    if partial:
        ref_positions += (896,)
    active_positions = tuple(sorted(set(range(1152)) - set(ref_positions)))
    physical = ref_positions + active_positions
    layout = geometry.ReferenceImageLayout(geometry.VideoTokenLayout(used_len=1152), ref_positions)
    videos = ((312, (2, 8, 16)), (768, (3, 8, 16)))
    images = ((24, (16, 16)), (568, (10, 16)))
    plan = rectangular.get_plan(1152, len(ref_positions), videos, "cpu", image_spans=images, image_layout=layout)
    assert (
        rectangular.get_plan(1152, len(ref_positions), videos, "cpu", image_spans=images, image_layout=layout) is plan
    )
    assert packed_mask_input.call_count == 1
    assert packed_mask_input.call_args.kwargs["return_reference_plan"]
    assert plan.image_query_blocks.tolist() == ([0] if partial else [0, 1])

    # Head dim 1 suffices for recording kernels. The second block of image 1
    # wins, while image 2's only full block and its protected tail both survive.
    logical = torch.arange(1152, dtype=torch.float32).view(1, -1, 1, 1).div(1152).to(torch.bfloat16)
    kv = logical[:, list(physical)]
    q = torch.ones_like(logical[:, list(active_positions)])
    result = rectangular.run(q, kv, kv, plan, 0, 1, precision=precision, image_sparsity=0.75)
    torch.testing.assert_close(result, q)
    assert len(calls) == 2
    sparse_q, reordered_k, mask = calls[0]
    assert sparse_q.shape[1] == (128 if partial else 256)
    torch.testing.assert_close(reordered_k, logical[:, perm])
    assert mask[0, 0].tolist() == [[0, 1, 1, 1, 1, 1, 1, 1, 1]] * (1 if partial else 2)
    assert calls[1][2].all()
