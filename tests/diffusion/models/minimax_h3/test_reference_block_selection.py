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
from bisect import bisect_left
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
            "SkipSoftmaxSpec",
            "AttnQuantSpec",
            "SpargeAttnSpec",
            "RainFusionSpec",
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
    layouts = definitions(
        "_rbs_layout",
        "vllm_omni/diffusion/attention/backends/abstract.py",
        {"ImageTokenSpan", "VideoTokenSpan", "VideoTokenLayout", "AttentionMetadata"},
        {"dataclass": dataclasses.dataclass, "field": dataclasses.field},
    )
    return definitions(
        "_rbs_geometry",
        "vllm_omni/diffusion/models/minimax_h3/minimax_h3_transformer.py",
        {"_reference_kv_video_layout", "_compact_reference_video_layout"},
        {
            "ImageTokenSpan": layouts.ImageTokenSpan,
            "VideoTokenSpan": layouts.VideoTokenSpan,
            "VideoTokenLayout": layouts.VideoTokenLayout,
            "AttentionMetadata": layouts.AttentionMetadata,
            "math": math,
            "bisect_left": bisect_left,
        },
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
        {"RainFusionConfig", "RainFusionPlan", "RainFusionAttentionImpl", "_supports_video_spans"},
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


@pytest.mark.parametrize("ratio", [0.0, 0.2, 0.5, 0.9, 1.0])
def test_ratio_reaches_backend_from_standard_config(config, backend, ratio):
    parsed = config.AttentionConfig(
        default={
            "backend": "RAINFUSION_ATTN",
            "block_sparse": {"sparsity": 0.8, "precision": "mix"},
            "rainfusion": {"image_sparsity": ratio},
        }
    )
    kwargs = parsed.default.backend_kwargs()
    assert backend.RainFusionConfig.from_backend_kwargs(kwargs).image_sparsity == ratio
    if ratio == 0.0:
        assert "image_sparsity" not in kwargs


def test_new_image_api_roundtrip(config, backend):
    spec = config.AttentionSpec(
        backend="RAINFUSION_ATTN",
        block_sparse={"sparsity": 0.8, "precision": "mix"},
        rainfusion={"image_sparsity": 0.9},
    )
    kwargs = spec.backend_kwargs()
    assert kwargs["image_sparsity"] == 0.9
    assert "ref_keep_ratio" not in kwargs
    assert backend.RainFusionConfig.from_backend_kwargs(kwargs).image_sparsity == 0.9


@pytest.mark.parametrize("ratio", [-0.1, 1.1, float("nan"), float("inf"), True, "0.9"])
def test_invalid_ratio_is_rejected(config, backend, ratio):
    for cls in (config.RainFusionSpec, backend.RainFusionConfig):
        with pytest.raises(ValueError, match="image_sparsity"):
            cls(image_sparsity=ratio)


def test_image_selection_can_run_without_global_sparsity(config, backend):
    assert config.RainFusionSpec(image_sparsity=0.9).image_sparsity == 0.9
    assert backend.RainFusionConfig(sparsity=0, image_sparsity=0.9).enabled
    assert not backend.RainFusionConfig(sparsity=0).enabled


def test_reference_images_follow_cached_prefix_positions(geometry):
    layout = geometry.VideoTokenLayout(
        used_len=36,
        image_spans=(geometry.ImageTokenSpan(2, (1, 2)), geometry.ImageTokenSpan(8, (1, 2))),
        video_spans=(
            geometry.VideoTokenSpan(start=16, latent_grid=(1, 2, 2), role="reference"),
            geometry.VideoTokenSpan(start=20, latent_grid=(4, 2, 2), role="target"),
        ),
    )
    # Other cached context makes the second image non-contiguous in the prefix.
    positions = torch.tensor([0, 2, 3, 6, 8, 9, 16, 17, 18, 19])
    compact = geometry.VideoTokenLayout(
        used_len=26, video_spans=(geometry.VideoTokenSpan(start=10, latent_grid=(4, 2, 2), role="target"),)
    )
    remapped = geometry._reference_kv_video_layout(layout, positions, compact)
    assert [(x.start, x.latent_grid) for x in remapped.image_spans] == [(1, (1, 2)), (4, (1, 2))]
    assert [(span.start, span.role) for span in remapped.video_spans] == [(6, "reference"), (20, "target")]


def test_partially_cached_image_is_rejected(geometry):
    original = geometry.VideoTokenLayout(image_spans=(geometry.ImageTokenSpan(2, (1, 3)),))
    compact = geometry.VideoTokenLayout(used_len=10)
    with pytest.raises(ValueError, match="not fully represented"):
        geometry._reference_kv_video_layout(original, torch.tensor([2, 4]), compact)


@pytest.mark.parametrize("cached", [False, True])
def test_image_grid_survives_cache_compaction_and_restoration(geometry, cached):
    original = geometry.VideoTokenLayout(
        used_len=16,
        image_spans=(geometry.ImageTokenSpan(0, (2, 2)),),
        video_spans=(geometry.VideoTokenSpan(8, (2, 2, 2), "target"),),
    )
    positions = torch.arange(4) if cached else torch.tensor([], dtype=torch.long)
    compact = geometry._compact_reference_video_layout(original, positions, 16 - len(positions))
    assert compact is not None
    assert len(compact.image_spans) == (0 if cached else 1)
    restored = geometry._reference_kv_video_layout(original, positions, compact)
    assert restored.image_spans == original.image_spans
    assert restored.video_spans == original.video_spans


def test_full_query_plan_preserves_image_grids_and_all_video_spans(backend, geometry):
    impl = backend.RainFusionAttentionImpl.__new__(backend.RainFusionAttentionImpl)
    impl.rainfusion = backend.RainFusionConfig(sparsity=0.8, image_sparsity=0.9)
    layout = geometry.VideoTokenLayout(
        used_len=8192,
        image_spans=(geometry.ImageTokenSpan(0, (16, 16)), geometry.ImageTokenSpan(384, (16, 16))),
        video_spans=(
            geometry.VideoTokenSpan(start=768, latent_grid=(2, 8, 16), role="reference"),
            geometry.VideoTokenSpan(start=1024, latent_grid=(4, 32, 56), role="target"),
        ),
    )
    plan = impl._resolve_multi_span_plan(layout, 8192)
    assert plan.image_spans == [
        {"start": 0, "latent_shape": [16, 16]},
        {"start": 384, "latent_shape": [16, 16]},
    ]
    assert len(plan.video_spans) == 2
    assert all(set(span) == {"start", "latent_shape"} for span in plan.video_spans)
    impl.rainfusion = backend.RainFusionConfig(sparsity=0.8)
    assert impl._resolve_multi_span_plan(layout, 8192).image_spans is None


@pytest.mark.parametrize("precision", ["bf16", "mix"])
def test_old_mindie_image_api_is_rejected(backend, precision, monkeypatch):
    module = ModuleType("mindiesd")

    def sparse_attention(q, k, v, *, video_spans, precision, **kwargs):
        pytest.fail("unsupported image API must fail before execution")

    module.__dict__["sparse_attention"] = sparse_attention
    monkeypatch.setitem(sys.modules, "mindiesd", module)
    impl = backend.RainFusionAttentionImpl.__new__(backend.RainFusionAttentionImpl)
    impl.rainfusion = backend.RainFusionConfig(sparsity=0.8, image_sparsity=0.9, precision=precision)
    impl.softmax_scale = 1.0
    q = torch.zeros(1, 256, 1, 128)
    plan = backend.RainFusionPlan(used_len=256, video_spans=[], image_spans=[{"start": 0, "latent_shape": [8, 16]}])
    with pytest.raises(ValueError, match="image_spans/image_sparsity"):
        impl._forward_sparse_npu(q, q, q, plan)


@pytest.mark.parametrize("override", [None, "float"])
@pytest.mark.parametrize("image_drop", [0.0, 0.9])
@pytest.mark.parametrize("padding", [0, 2])
def test_eight_steps_dispatch_image_api_with_padded_reference_queries(
    backend, config, geometry, progress, monkeypatch, image_drop, padding, override
):
    calls = []

    @dataclasses.dataclass
    class RectangularPlan:
        dense_rows: int = 0

    def sparse_attention(q, k, v, *, precision, video_spans, sparsity, image_spans=None, image_sparsity=0.0, **kwargs):
        assert q.shape == k.shape == v.shape
        assert image_spans == ([{"start": 0, "latent_shape": [2, 2]}] if image_drop else None)
        assert precision == ("bf16" if override else "mix")
        assert kwargs["sparse_type"] == ("rf_v2" if override else "rf_v3")
        if progress.denoise_step_idx:
            assert q.shape[1] == 12
            assert not q[:, :4].any()
        calls.append(("full", sparsity, image_sparsity))
        return q + 100

    def run(q, k, v, plan, sparsity, scale, **kwargs):
        assert kwargs["precision"] == ("bf16" if override else "mix")
        calls.append(("rectangular", sparsity, 0.0))
        return q + 100

    native = ModuleType("mindiesd")
    native.__dict__["sparse_attention"] = sparse_attention
    platform = ModuleType("mindiesd.utils.get_platform")
    platform.__dict__["is_a5_device"] = lambda: True
    rectangular = ModuleType("vllm_omni.diffusion.attention.ops.rainfusion_rectangular")
    rectangular.__dict__["get_plan"] = lambda *args: RectangularPlan()
    rectangular.__dict__["run"] = run
    for module in (native, platform, rectangular):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    instance = backend.RainFusionAttentionImpl.__new__(backend.RainFusionAttentionImpl)
    instance.rainfusion = backend.RainFusionConfig.from_backend_kwargs(
        config.AttentionSpec(
            backend="RAINFUSION_ATTN",
            block_sparse={"sparsity": 0.8, "precision": "mix"},
            rainfusion={"image_sparsity": image_drop},
        ).backend_kwargs()
    )
    instance.qkv_layout = "BSND"
    instance.softmax_scale = 0.25
    instance._resolve_plan = lambda metadata: backend.RainFusionPlan(
        used_len=int(metadata.extra["max_seqlen_q"]),
        video_spans=[{"start": 4, "latent_shape": [1, 1, 8]}],
        image_spans=[{"start": 0, "latent_shape": [2, 2]}] if image_drop else None,
    )
    q = torch.ones(1, 8 + padding, 2, 16)
    span = geometry.VideoTokenSpan(start=0, latent_grid=(1, 1, 8), role="target")
    layout = geometry.VideoTokenLayout(used_len=8, video_spans=(span,))
    kv_layout = dataclasses.replace(layout, used_len=12, image_spans=(geometry.ImageTokenSpan(0, (2, 2)),))
    schedule = definitions(
        "_lq_main_schedule",
        "vllm_omni/diffusion/attention/schedule.py",
        {"AttentionScheduleRange", "_validate_step", "validate_attention_profile_name", "select_attention_profile"},
        {"dataclass": dataclasses.dataclass},
    )
    ranges = (schedule.AttentionScheduleRange(2, None, "late"),)
    for step in range(8):
        profile = schedule.select_attention_profile(ranges, step, total_steps=8)
        instance.rainfusion = dataclasses.replace(instance.rainfusion, sparsity=0.85 if profile == "late" else 0.8)
        progress.denoise_step_idx = step
        metadata = geometry.AttentionMetadata(extra={"max_seqlen_q": 8})
        kv = q
        if step:
            metadata = geometry.AttentionMetadata(
                extra={
                    "max_seqlen_q": 8,
                    "max_seqlen_k": 12,
                    "minimax_h3_compact_reference_rows": 4,
                    "minimax_h3_resident_reference_kv": True,
                    "minimax_h3_reference_kv_video_layout": kv_layout,
                },
                video_layout=layout,
            )
            kv = torch.ones(1, 12 + padding, 2, 16)
        if override is not None:
            metadata.extra["kv_cache_dtype"] = override
        result = instance.forward_npu(q, kv, kv, metadata)
        assert result.shape == q.shape
        torch.testing.assert_close(result[:, :8], q[:, :8] + 100)
        torch.testing.assert_close(result[:, 8:], q[:, 8:])
    if image_drop:
        assert calls == [("full", 0.8, 0.9)] * 2 + [("full", 0.85, 0.9)] * 6
    else:
        assert calls == [("full", 0.8, 0.0), ("rectangular", 0.8, 0.0)] + [("rectangular", 0.85, 0.0)] * 6
    # Token-refiner calls without sparse geometry must not need a step index.
    progress.denoise_step_idx = progress.total_denoise_steps = None
    instance._resolve_plan = lambda metadata: None
    instance.dense_fallback = torch.nn.Module()
    instance.dense_fallback.forward_npu = lambda q, k, v, metadata: q
    assert instance.forward_npu(q, q, q) is q


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


def test_rainfusion_options_reject_unrelated_backend(config):
    with pytest.raises(ValueError, match="rainfusion is only supported"):
        config.AttentionSpec(backend="FLASH_ATTN", rainfusion={"image_sparsity": 0.9})
