# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contracts for per-layer schedule candidate preparation and startup validation.

Calibration stamping is tested in test_trtllm_calibration.py.

No backend touches a GPU and no model weights load; backends and impls are fakes
resolved from the supplied AttentionConfig, mirroring test_attention_config.py.
"""

from types import SimpleNamespace

import pytest

import vllm_omni.diffusion.attention.layer as layer_mod
from vllm_omni.diffusion.attention.backends.abstract import AttentionImpl, AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
from vllm_omni.diffusion.attention.layer import Attention, _PreparedCandidate
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionScheduleConfig,
    AttentionSpec,
    DiffusionParallelConfig,
    OmniDiffusionConfig,
    SkipSoftmaxSpec,
)
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeImpl:
    """Records construction kwargs; optionally accepts layer calibration."""

    _kv_cache_dtypes_ok: tuple[str, ...] = ()

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calibration = None

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype, platform_key) -> bool:
        del platform_key
        return kv_cache_dtype is None or kv_cache_dtype in cls._kv_cache_dtypes_ok

    def forward(self, query, key, value, attn_metadata=None):
        return query

    def set_layer_calibration(self, a, b):
        self.calibration = (a, b)


class _MetadataOnlyImpl(_FakeImpl, AttentionImpl[AttentionMetadata]):
    def forward(self, *args, **kwargs):
        raise AssertionError("metadata-only fixture must not execute attention")


def _make_metadata_candidate(backend_cls: type, spec: AttentionSpec | None = None) -> _PreparedCandidate:
    impl = _MetadataOnlyImpl()
    return _PreparedCandidate(
        backend_cls=backend_cls,
        spec=spec,
        impl_cls=_MetadataOnlyImpl,
        impl=impl,
        backend_explicit=spec is not None,
        backend_pref=spec.backend if spec is not None else None,
    )


_BACKEND_CACHE: dict[str, type] = {}
_BACKEND_IMPLS: dict[str, type[_FakeImpl]] = {}


def _fake_backend(name: str) -> type:
    """A distinct fake backend (and impl subclass) per backend name."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    impl_cls = type(f"Fake{name}Impl", (_FakeImpl,), {})
    _BACKEND_IMPLS[name] = impl_cls

    class _Backend:
        supports_paged_kv = False
        supports_piecewise_spans = False

        @staticmethod
        def get_name() -> str:
            return name

        @staticmethod
        def get_impl_cls():
            return impl_cls

        @staticmethod
        def supports_attention_mask(spec) -> bool:
            return True

    _Backend.__name__ = f"Fake{name}Backend"
    _BACKEND_CACHE[name] = _Backend
    return _Backend


def _fake_resolve(*, role, head_size, attention_config=None, role_category=None, allow_trtllm_default=True):
    """Resolve a fake backend from the supplied AttentionConfig, like the real selector."""
    spec = None
    if attention_config is not None:
        spec, _source = attention_config.resolve_with_source(role=role, role_category=role_category)
    if spec is None:
        return _fake_backend("PLATFORM_DEFAULT"), None
    return _fake_backend(spec.backend.upper()), spec


def _make_config(
    *,
    baseline: AttentionConfig | None = None,
    schedule: AttentionScheduleConfig | None = None,
    kv_mode: layer_mod.DiffusionKVCacheMode | None = None,
    allgather_degree: int = 1,
    kv_dtype: str | None = None,
    sp_size: int = 1,
    ring_degree: int = 1,
) -> OmniDiffusionConfig:
    return OmniDiffusionConfig(
        diffusion_attention_config=baseline if baseline is not None else AttentionConfig(),
        diffusion_attention_schedule=schedule,
        diffusion_kv_mode=kv_mode if kv_mode is not None else layer_mod.DiffusionKVCacheMode.DENSE_LEGACY,
        parallel_config=DiffusionParallelConfig(
            ring_degree=ring_degree,
            allgather_degree=allgather_degree,
            ulysses_degree=sp_size,
        ),
        diffusion_kv_cache_dtype=kv_dtype,
        diffusion_kv_max_rows_per_request=64 if kv_mode is layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER else None,
    )


@pytest.fixture
def attention_env(monkeypatch):
    """Patch the layer module so Attention constructs against fake backends."""
    monkeypatch.setattr(layer_mod.SDPABackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kwargs: NoParallelAttention())
    monkeypatch.setattr(layer_mod, "is_forward_context_available", lambda: False)

    calls: list[dict] = []

    def _counting(*, role, head_size, attention_config=None, role_category=None, allow_trtllm_default=True):
        calls.append({"role": role, "attention_config": attention_config})
        return _fake_resolve(
            role=role,
            head_size=head_size,
            attention_config=attention_config,
            role_category=role_category,
            allow_trtllm_default=allow_trtllm_default,
        )

    monkeypatch.setattr(layer_mod, "get_attn_backend_for_role", _counting)

    def build(config, **kwargs):
        with set_current_diffusion_config(config):
            return Attention(num_heads=4, head_size=64, causal=False, softmax_scale=1.0, **kwargs)

    return SimpleNamespace(build=build, calls=calls, monkeypatch=monkeypatch)


def test_identical_effective_profiles_dedup_to_shared_record(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={
            "a": AttentionConfig(default=AttentionSpec(backend="SDPA")),
            "b": AttentionConfig(default=AttentionSpec(backend="SDPA")),
        },
        default=[{"start": 0, "end": None, "profile": "a"}],
    )
    attention = attention_env.build(_make_config(schedule=schedule))

    assert attention._schedule_profiles == ("a", "b")
    assert attention._schedule_candidates["a"] is attention._schedule_candidates["b"]
    assert attention._schedule_candidates["a"].impl is attention._schedule_candidates["b"].impl


def test_allgather_trtllm_candidate_is_rejected(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={"trt": AttentionConfig(default=AttentionSpec(backend="TRTLLM_ATTN"))},
        default=[{"start": 0, "end": None, "profile": "trt"}],
    )
    config = _make_config(schedule=schedule, allgather_degree=2)
    with pytest.raises(ValueError, match="AllGather-KV sequence parallelism"):
        attention_env.build(config)


def _pipeline(layer, *, sp_plan=None, name="attn1"):
    """A minimal loaded-pipeline stand-in exposing named_modules() and an optional _sp_plan."""
    import torch.nn as nn

    transformer = nn.Module()
    setattr(transformer, name, layer)
    if sp_plan is not None:
        transformer._sp_plan = sp_plan
    root = nn.Module()
    root.transformer = transformer
    return root


def _flat_auto_pad_plan():
    """auto_pad on the plan entry itself, one level below the plan root."""
    return {"hidden_states": SequenceParallelInput(split_dim=1, expected_dims=3, auto_pad=True)}


def test_startup_validation_rejects_candidate_without_paged_support(attention_env):
    attention_env.monkeypatch.setattr(FlashAttentionBackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    schedule = AttentionScheduleConfig(
        profiles={"nopaged": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "nopaged"}],
    )
    config = _make_config(schedule=schedule, kv_mode=layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER)
    layer = attention_env.build(config, paged_kv_cache_role="primary")

    with pytest.raises(ValueError, match=r"profile 'nopaged'.*does not support Scheduler-managed paged KV"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_rejects_paged_capable_candidate_that_differs_from_baseline(attention_env):
    attention_env.monkeypatch.setattr(FlashAttentionBackend, "get_impl_cls", staticmethod(lambda: _FakeImpl))
    attention_env.monkeypatch.setattr(_fake_backend("BLOCK_SPARSE"), "supports_paged_kv", True)
    schedule = AttentionScheduleConfig(
        profiles={"paged": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "paged"}],
    )
    config = _make_config(schedule=schedule, kv_mode=layer_mod.DiffusionKVCacheMode.PAGED_SCHEDULER)
    layer = attention_env.build(config, paged_kv_cache_role="primary")

    with pytest.raises(ValueError, match=r"profile 'paged'.*native implementation bound to the baseline"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_rejects_auto_pad_incompatible_candidate(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={"nomask": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "nomask"}],
    )
    attention_env.monkeypatch.setattr(
        _fake_backend("BLOCK_SPARSE"), "supports_attention_mask", staticmethod(lambda spec=None: False)
    )
    config = _make_config(schedule=schedule, sp_size=2)
    layer = attention_env.build(config)

    with pytest.raises(ValueError, match=r"profile 'nomask'.*attention_mask"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer, sp_plan=_flat_auto_pad_plan()), config)


def _ring_config(schedule, **kwargs):
    """An od_config for the traversal only."""
    return _make_config(schedule=schedule, ring_degree=2, **kwargs)


def test_startup_validation_rejects_skip_softmax_candidate_under_ring(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={
            "sparse": AttentionConfig(
                default=AttentionSpec(backend="TRTLLM_ATTN", skip_softmax=SkipSoftmaxSpec(threshold=0.2))
            )
        },
        default=[{"start": 0, "end": None, "profile": "sparse"}],
    )
    layer = attention_env.build(_make_config(schedule=schedule))

    with pytest.raises(ValueError, match=r"profile 'sparse'.*[Rr]ing"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), _ring_config(schedule))


def test_startup_validation_rejects_candidate_backend_divergence_under_ring(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={"other": AttentionConfig(default=AttentionSpec(backend="BLOCK_SPARSE"))},
        default=[{"start": 0, "end": None, "profile": "other"}],
    )
    layer = attention_env.build(_make_config(schedule=schedule))

    assert layer.backend_pref != layer._schedule_candidates["other"].backend_pref
    with pytest.raises(ValueError, match=r"profile 'other'.*ring"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), _ring_config(schedule))


@pytest.mark.parametrize(("candidate_topk", "accepted"), [(8, False), (4, True)], ids=["different", "same"])
def test_startup_validation_compares_backend_kwargs_under_ring(attention_env, candidate_topk, accepted):
    baseline = AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=4))
    schedule = AttentionScheduleConfig(
        profiles={
            "vsa": AttentionConfig(default=AttentionSpec(backend="FASTVIDEO_VSA", fastvideo_vsa_topk=candidate_topk))
        },
        default=[{"start": 0, "end": None, "profile": "vsa"}],
    )
    layer = attention_env.build(_make_config(baseline=baseline, schedule=schedule))
    config = _ring_config(schedule, baseline=baseline)

    assert layer._schedule_candidates["vsa"].backend_cls is layer.attn_backend
    if accepted:
        assert layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config) == 1
    else:
        with pytest.raises(ValueError, match=r"profile 'vsa'.*backend kwargs"):
            layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def _fake_impl(name: str) -> type[_FakeImpl]:
    """Typed accessor for the impl class a fake backend resolves to.

    ``_fake_backend`` is annotated as a bare ``type`` because the backend class is built
    per name, so assertions that need the impl class go through here rather than calling
    ``.get_impl_cls()`` on an untyped class object.
    """
    _fake_backend(name)
    return _BACKEND_IMPLS[name]


def test_startup_validation_rejects_candidate_without_kv_cache_dtype_support(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={"quant": AttentionConfig(default=AttentionSpec(backend="SDPA"))},
        default=[{"start": 0, "end": None, "profile": "quant"}],
    )
    config = _make_config(schedule=schedule, kv_dtype="fp8")
    attention_env.monkeypatch.setattr(_fake_impl("PLATFORM_DEFAULT"), "_kv_cache_dtypes_ok", ("fp8",))
    layer = attention_env.build(config)

    with pytest.raises(ValueError, match=r"profile 'quant'.*kv_cache_dtype='fp8'"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)


def test_startup_validation_rejects_candidate_without_resolvable_calibration(attention_env):
    schedule = AttentionScheduleConfig(
        profiles={
            "sparse": AttentionConfig(
                default=AttentionSpec(
                    backend="TRTLLM_ATTN",
                    skip_softmax=SkipSoftmaxSpec(target_sparsity=0.5),
                    skip_calibration={"by_expert": {"other_expert": {"a": 1.0, "b": 2.0}}},
                )
            )
        },
        default=[{"start": 0, "end": None, "profile": "sparse"}],
    )
    config = _make_config(schedule=schedule)
    layer = attention_env.build(config)

    with pytest.raises(ValueError, match=r"profile 'sparse'.*calibration"):
        layer_mod.validate_attention_schedule_candidates(_pipeline(layer), config)
