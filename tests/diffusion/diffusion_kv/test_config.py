# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import logging

import pytest

from vllm_omni.config.omni_config import _DiffusionConfigProjection
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv import config as diffusion_kv_config
from vllm_omni.diffusion.diffusion_kv.config import (
    DiffusionKVCacheMode,
    resolve_diffusion_kv_cache_mode,
)
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

PAGED_MODEL = "HunyuanImage3Pipeline"
MOONCAKE_TRANSFER_CONFIG = {
    "kv_connector": "MooncakeConnector",
    "kv_role": "kv_consumer",
    "engine_id": "test-diffusion",
}

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _fixed_master_port(monkeypatch) -> None:
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda _self: 29500)


def test_dense_legacy_is_default() -> None:
    config = OmniDiffusionConfig.from_kwargs()

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.DENSE_LEGACY


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
@pytest.mark.parametrize("mode", [None, "dense_legacy", DiffusionKVCacheMode.DENSE_LEGACY])
def test_prefix_caching_rejects_non_paged_mode(config_cls, mode) -> None:
    kwargs = {} if mode is None else {"diffusion_kv_mode": mode}
    with pytest.raises(
        ValueError,
        match="set diffusion_kv_mode='paged_scheduler' or disable enable_prefix_caching",
    ):
        config_cls.from_kwargs(enable_prefix_caching=True, **kwargs)


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_prefix_caching_accepts_paged_scheduler(config_cls) -> None:
    config = config_cls.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=True,
    )

    assert config.enable_prefix_caching is True
    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER


@pytest.mark.parametrize("build_config", [OmniDiffusionConfig, OmniDiffusionConfig.from_kwargs])
def test_prefix_caching_rejects_sleep_mode(build_config) -> None:
    with pytest.raises(ValueError, match="disable enable_prefix_caching or enable_sleep_mode"):
        build_config(
            diffusion_kv_mode="paged_scheduler",
            diffusion_kv_max_rows_per_request=2,
            enable_prefix_caching=True,
            enable_sleep_mode=True,
        )


@pytest.mark.parametrize("mode", ["dense_legacy", "paged_scheduler"])
def test_sleep_mode_accepts_disabled_prefix_caching(mode) -> None:
    config = OmniDiffusionConfig.from_kwargs(
        diffusion_kv_mode=mode,
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=False,
        enable_sleep_mode=True,
    )

    assert config.enable_sleep_mode is True
    assert config.enable_prefix_caching is False


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_native_kv_transfer_requires_prefix_caching_disabled(enable_prefix_caching) -> None:
    kwargs = dict(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=enable_prefix_caching,
        kv_transfer_config={
            "kv_connector": "MooncakeConnector",
            "kv_role": "kv_consumer",
            "engine_id": "test-diffusion",
        },
    )
    if enable_prefix_caching:
        with pytest.raises(ValueError, match="disable enable_prefix_caching for AR KV import"):
            OmniDiffusionConfig.from_kwargs(**kwargs)
    else:
        config = OmniDiffusionConfig.from_kwargs(**kwargs)
        assert config.kv_transfer_config.kv_role == "kv_consumer"


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
@pytest.mark.parametrize("mode", ["dense_legacy", "paged_scheduler"])
def test_disabled_prefix_caching_accepts_both_modes(config_cls, mode) -> None:
    config = config_cls.from_kwargs(
        diffusion_kv_mode=mode,
        diffusion_kv_max_rows_per_request=2,
        enable_prefix_caching=False,
    )

    assert config.enable_prefix_caching is False
    assert config.diffusion_kv_mode.value == mode


def test_paged_worker_local_is_rejected_until_implemented() -> None:
    with pytest.raises(ValueError, match="reserved but not implemented"):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_mode="paged_worker_local")


def test_unknown_cache_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported Diffusion KV diffusion_kv_mode"):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_mode="unknown")


def test_non_mapping_omni_kv_config_is_rejected() -> None:
    with pytest.raises(TypeError, match="omni_kv_config must be a mapping"):
        OmniDiffusionConfig.from_kwargs(omni_kv_config="paged_scheduler")


def test_paged_scheduler_rejects_dense_connector_kv_receive() -> None:
    with pytest.raises(ValueError, match="does not support imported AR KV"):
        OmniDiffusionConfig.from_kwargs(
            diffusion_kv_mode="paged_scheduler",
            diffusion_kv_max_rows_per_request=1,
            max_num_batched_tokens=1,
            omni_kv_config={"need_recv_cache": True},
        )


def test_paged_scheduler_does_not_depend_on_model_registry() -> None:
    config = OmniDiffusionConfig.from_kwargs(
        model_class_name="FutureDiffusionModel",
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
        max_num_batched_tokens=1,
    )

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_paged_scheduler_mode_is_platform_agnostic(config_cls) -> None:
    config = config_cls.from_kwargs(
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=1,
    )

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER


def test_paged_scheduler_requires_a_worker_row_limit() -> None:
    with pytest.raises(ValueError, match="requires diffusion_kv_max_rows_per_request"):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_mode="paged_scheduler")
    with pytest.raises(ValueError, match="requires diffusion_kv_max_rows_per_request"):
        _DiffusionConfigProjection.from_kwargs(diffusion_kv_mode="paged_scheduler")


@pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.5])
def test_worker_row_limit_must_be_a_positive_integer(invalid_limit: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        OmniDiffusionConfig.from_kwargs(diffusion_kv_max_rows_per_request=invalid_limit)
    with pytest.raises((TypeError, ValueError)):
        _DiffusionConfigProjection.from_kwargs(diffusion_kv_max_rows_per_request=invalid_limit)


@pytest.mark.parametrize("dtype", ["fp8", "fp8_e4m3"])
def test_dense_legacy_allows_attention_quantization(dtype) -> None:
    config = OmniDiffusionConfig.from_kwargs(diffusion_kv_cache_dtype=dtype)
    assert config.diffusion_kv_mode is DiffusionKVCacheMode.DENSE_LEGACY


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_unified_knob_conflicts_with_explicit_quant_spec(config_cls) -> None:
    with pytest.raises(ValueError, match="conflicts with a diffusion_attention_config quant spec"):
        config_cls.from_kwargs(
            diffusion_kv_cache_dtype="fp8",
            diffusion_attention_config={"default": {"backend": "TRTLLM_ATTN", "quant": {"dtype_qk": "int8"}}},
        )


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_unified_knob_allows_backend_pin_without_dtypes(config_cls) -> None:
    config = config_cls.from_kwargs(
        diffusion_kv_cache_dtype="fp8",
        diffusion_attention_config={"default": {"backend": "TRTLLM_ATTN", "quant": {"flashinfer_backend": "fa2"}}},
    )
    assert config.diffusion_kv_cache_dtype == "fp8"


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_explicit_quant_spec_without_knob_is_allowed(config_cls) -> None:
    config = config_cls.from_kwargs(
        diffusion_attention_config={"default": {"backend": "TRTLLM_ATTN", "quant": {"dtype_qk": "int8"}}},
    )
    assert config.diffusion_kv_cache_dtype is None


def test_paged_model_declares_default_eligibility_and_row_capacity() -> None:
    metadata = get_diffusion_model_metadata(PAGED_MODEL)

    assert metadata.paged_kv_default_eligible is True
    # One BlockTable row per CFG branch.
    assert metadata.paged_kv_max_rows_per_request == 2


def test_non_paged_model_does_not_declare_default_eligibility() -> None:
    metadata = get_diffusion_model_metadata("FutureDiffusionModel")

    assert metadata.paged_kv_default_eligible is False
    assert metadata.paged_kv_max_rows_per_request is None


def test_unset_mode_stays_dense_until_the_default_is_enabled() -> None:
    config = OmniDiffusionConfig.from_kwargs(model_class_name=PAGED_MODEL)

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.DENSE_LEGACY


def test_projection_preserves_unset_mode_for_serialization() -> None:
    # The projection is serialized into the stage config; it must not bake in a
    # resolved default, otherwise a later default change cannot reach stages.
    config = _DiffusionConfigProjection.from_kwargs(model_class_name=PAGED_MODEL)

    assert config.diffusion_kv_mode is None


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_explicit_dense_opt_out_is_preserved_for_paged_model(config_cls) -> None:
    config = config_cls.from_kwargs(model_class_name=PAGED_MODEL, diffusion_kv_mode="dense_legacy")

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.DENSE_LEGACY


def test_paged_model_fills_row_capacity_from_metadata() -> None:
    config = OmniDiffusionConfig.from_kwargs(model_class_name=PAGED_MODEL, diffusion_kv_mode="paged_scheduler")

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER
    assert config.diffusion_kv_max_rows_per_request == 2


def test_projection_does_not_freeze_an_inherited_row_capacity() -> None:
    # The projection is serialized into the stage config, so an inherited row
    # limit must stay unset there: the runtime resolves it from the model
    # integration again instead of pinning a value the metadata could revise.
    config = _DiffusionConfigProjection.from_kwargs(model_class_name=PAGED_MODEL, diffusion_kv_mode="paged_scheduler")

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER
    assert config.diffusion_kv_max_rows_per_request is None


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_explicit_row_limit_overrides_the_model_default(config_cls) -> None:
    config = config_cls.from_kwargs(
        model_class_name=PAGED_MODEL,
        diffusion_kv_mode="paged_scheduler",
        diffusion_kv_max_rows_per_request=4,
    )

    assert config.diffusion_kv_max_rows_per_request == 4


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_explicit_paged_rejects_step_execution(config_cls) -> None:
    with pytest.raises(ValueError, match="request-level execution only"):
        config_cls.from_kwargs(model_class_name=PAGED_MODEL, diffusion_kv_mode="paged_scheduler", step_execution=True)


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_explicit_paged_rejects_unsupported_attention_backend(config_cls) -> None:
    with pytest.raises(ValueError, match="advertises paged KV support"):
        config_cls.from_kwargs(
            model_class_name=PAGED_MODEL,
            diffusion_kv_mode="paged_scheduler",
            diffusion_attention_config={"default": {"backend": "SDPA"}},
        )


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_explicit_paged_rejects_unsupported_per_role_backend(config_cls) -> None:
    # A per_role override wins over ``default`` in the attention layer, so it
    # must win in the compatibility check too.
    with pytest.raises(ValueError, match="advertises paged KV support"):
        config_cls.from_kwargs(
            model_class_name=PAGED_MODEL,
            diffusion_kv_mode="paged_scheduler",
            diffusion_attention_config={"per_role": {"self": {"backend": "SDPA"}}},
        )


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_paged_role_override_beats_an_unsupported_default(config_cls) -> None:
    config = config_cls.from_kwargs(
        model_class_name=PAGED_MODEL,
        diffusion_kv_mode="paged_scheduler",
        diffusion_attention_config={"default": {"backend": "SDPA"}, "per_role": {"self": {"backend": "FLASH_ATTN"}}},
    )

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER


@pytest.mark.parametrize(
    "parallel_config, message",
    [
        ({"ring_degree": 2}, "does not support Ring sequence parallelism"),
        ({"allgather_degree": 2}, "does not support AllGather-KV sequence parallelism"),
        ({"ulysses_degree": 2, "ulysses_mode": "advanced_uaa"}, "supports only strict Ulysses"),
    ],
)
def test_explicit_paged_rejects_unsupported_sequence_parallelism(parallel_config, message) -> None:
    # The paged attention adapter rejects these layouts when the first cache
    # layer is registered; config time must not accept them first.
    with pytest.raises(ValueError, match=message):
        OmniDiffusionConfig.from_kwargs(
            model_class_name=PAGED_MODEL,
            diffusion_kv_mode="paged_scheduler",
            parallel_config=parallel_config,
        )


@pytest.mark.parametrize(
    "parallel_config",
    [
        {"ring_degree": 2},
        {"allgather_degree": 2},
        {"ulysses_degree": 2, "ulysses_mode": "advanced_uaa"},
    ],
)
def test_implicit_mode_falls_back_to_dense_for_unsupported_sequence_parallelism(monkeypatch, parallel_config) -> None:
    monkeypatch.setattr(diffusion_kv_config, "PAGED_KV_IMPLICIT_DEFAULT_ENABLED", True)

    config = OmniDiffusionConfig.from_kwargs(model_class_name=PAGED_MODEL, parallel_config=parallel_config)

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.DENSE_LEGACY


def test_strict_ulysses_keeps_the_implicit_paged_default(monkeypatch) -> None:
    monkeypatch.setattr(diffusion_kv_config, "PAGED_KV_IMPLICIT_DEFAULT_ENABLED", True)

    config = OmniDiffusionConfig.from_kwargs(
        model_class_name=PAGED_MODEL,
        parallel_config={"ulysses_degree": 2, "ulysses_mode": "strict"},
    )

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_explicit_paged_rejects_offload(config_cls) -> None:
    with pytest.raises(ValueError, match="not been validated with CPU, layerwise"):
        config_cls.from_kwargs(
            model_class_name=PAGED_MODEL,
            diffusion_kv_mode="paged_scheduler",
            enable_cpu_offload=True,
        )


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_explicit_paged_accepts_supported_attention_backend(config_cls) -> None:
    config = config_cls.from_kwargs(
        model_class_name=PAGED_MODEL,
        diffusion_kv_mode="paged_scheduler",
        diffusion_attention_config={"default": {"backend": "FLASH_ATTN"}},
    )

    assert config.diffusion_kv_mode is DiffusionKVCacheMode.PAGED_SCHEDULER


def test_explicit_paged_supported_backend_inherits_row_capacity_at_runtime() -> None:
    config = OmniDiffusionConfig.from_kwargs(
        model_class_name=PAGED_MODEL,
        diffusion_kv_mode="paged_scheduler",
        diffusion_attention_config={"default": {"backend": "FLASH_ATTN"}},
    )

    assert config.diffusion_kv_max_rows_per_request == 2


def test_explicit_paged_rejects_env_selected_unsupported_backend(monkeypatch) -> None:
    # The env fallback is applied while normalizing the attention config, so the
    # paged compatibility check must consult the effective backend, not the raw
    # ``diffusion_attention_config``.
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "SDPA")

    with pytest.raises(ValueError, match="advertises paged KV support"):
        OmniDiffusionConfig.from_kwargs(model_class_name=PAGED_MODEL, diffusion_kv_mode="paged_scheduler")


def test_implicit_paged_incompatibility_logs_warning(monkeypatch, caplog) -> None:
    # The gate is on but step execution still forces the dense path: that
    # fallback must be visible in the logs, not silent.
    monkeypatch.setattr(diffusion_kv_config, "PAGED_KV_IMPLICIT_DEFAULT_ENABLED", True)

    with caplog.at_level(logging.WARNING):
        OmniDiffusionConfig.from_kwargs(model_class_name=PAGED_MODEL, step_execution=True)

    assert "supports paged diffusion KV" in caplog.text
    assert "step_execution" in caplog.text


def test_gate_flip_reaches_both_config_classes(monkeypatch) -> None:
    # The constant is bound when the resolver runs, not when the module loads,
    # so an operator (or a test) can flip it without rebuilding the callers.
    for config_cls in (OmniDiffusionConfig, _DiffusionConfigProjection):
        assert config_cls.from_kwargs(model_class_name=PAGED_MODEL).diffusion_kv_mode in (
            DiffusionKVCacheMode.DENSE_LEGACY,
            None,
        )

    monkeypatch.setattr(diffusion_kv_config, "PAGED_KV_IMPLICIT_DEFAULT_ENABLED", True)

    assert (
        OmniDiffusionConfig.from_kwargs(model_class_name=PAGED_MODEL).diffusion_kv_mode
        is DiffusionKVCacheMode.PAGED_SCHEDULER
    )
    # The projection intentionally keeps an unset mode unset for serialization.
    assert _DiffusionConfigProjection.from_kwargs(model_class_name=PAGED_MODEL).diffusion_kv_mode is None


@pytest.mark.parametrize("config_cls", [OmniDiffusionConfig, _DiffusionConfigProjection])
def test_prefix_caching_surfaces_the_real_paged_incompatibility(monkeypatch, config_cls) -> None:
    # Prefix caching only runs on the paged path, so an incompatible stage must
    # report the incompatibility instead of the generic "set diffusion_kv_mode"
    # message.
    monkeypatch.setattr(diffusion_kv_config, "PAGED_KV_IMPLICIT_DEFAULT_ENABLED", True)

    with pytest.raises(
        ValueError,
        match="enable_prefix_caching requires paged Diffusion KV, but .*advertises paged KV support",
    ):
        config_cls.from_kwargs(
            model_class_name=PAGED_MODEL,
            enable_prefix_caching=True,
            diffusion_attention_config={"default": {"backend": "SDPA"}},
        )


def test_native_kv_transfer_surfaces_the_real_paged_incompatibility(monkeypatch) -> None:
    monkeypatch.setattr(diffusion_kv_config, "PAGED_KV_IMPLICIT_DEFAULT_ENABLED", True)

    with pytest.raises(
        ValueError,
        match="kv_transfer_config requires paged Diffusion KV, but .*not been validated with CPU, layerwise",
    ):
        OmniDiffusionConfig.from_kwargs(
            model_class_name=PAGED_MODEL,
            kv_transfer_config=MOONCAKE_TRANSFER_CONFIG,
            enable_cpu_offload=True,
        )


def test_implicit_resolution_selects_paged_when_enabled() -> None:
    resolution = resolve_diffusion_kv_cache_mode(
        None,
        model_paged_kv_default_eligible=True,
        model_default_max_rows=2,
        allow_implicit_paged=True,
    )

    assert resolution.mode is DiffusionKVCacheMode.PAGED_SCHEDULER
    assert resolution.max_rows_per_request == 2


@pytest.mark.parametrize(
    "incompatibility",
    [
        {"step_execution": True},
        {"offload_enabled": True},
        {"legacy_imported_kv": True},
        {"attention_backend": "SDPA"},
        {"parallel_config": {"ring_degree": 2}},
        {"parallel_config": {"allgather_degree": 2}},
        {"parallel_config": {"ulysses_degree": 2, "ulysses_mode": "advanced_uaa"}},
    ],
)
def test_implicit_resolution_falls_back_to_dense(incompatibility) -> None:
    resolution = resolve_diffusion_kv_cache_mode(
        None,
        model_paged_kv_default_eligible=True,
        model_default_max_rows=2,
        allow_implicit_paged=True,
        **incompatibility,
    )

    assert resolution.mode is DiffusionKVCacheMode.DENSE_LEGACY


def test_implicit_resolution_accepts_strict_ulysses_parallel_config_object() -> None:
    from vllm_omni.diffusion.data import DiffusionParallelConfig

    resolution = resolve_diffusion_kv_cache_mode(
        None,
        model_paged_kv_default_eligible=True,
        model_default_max_rows=2,
        allow_implicit_paged=True,
        parallel_config=DiffusionParallelConfig(ulysses_degree=2, ulysses_mode="strict"),
    )

    assert resolution.mode is DiffusionKVCacheMode.PAGED_SCHEDULER


def test_implicit_resolution_requires_row_capacity() -> None:
    resolution = resolve_diffusion_kv_cache_mode(
        None,
        model_paged_kv_default_eligible=True,
        allow_implicit_paged=True,
    )

    assert resolution.mode is DiffusionKVCacheMode.DENSE_LEGACY


def test_explicit_paged_is_never_downgraded_to_dense() -> None:
    with pytest.raises(ValueError, match="request-level execution only"):
        resolve_diffusion_kv_cache_mode(
            "paged_scheduler",
            model_paged_kv_default_eligible=False,
            step_execution=True,
            max_rows_per_request=2,
        )


def test_resolver_gate_default_is_read_at_call_time(monkeypatch) -> None:
    kwargs = dict(model_paged_kv_default_eligible=True, model_default_max_rows=2)

    assert resolve_diffusion_kv_cache_mode(None, **kwargs).mode is DiffusionKVCacheMode.DENSE_LEGACY

    monkeypatch.setattr(diffusion_kv_config, "PAGED_KV_IMPLICIT_DEFAULT_ENABLED", True)

    assert resolve_diffusion_kv_cache_mode(None, **kwargs).mode is DiffusionKVCacheMode.PAGED_SCHEDULER


_SHARED_INCOMPATIBILITIES = [
    pytest.param({}, id="compatible"),
    pytest.param({"step_execution": True}, id="step-execution"),
    pytest.param({"enable_cpu_offload": True}, id="offload"),
    pytest.param({"diffusion_attention_config": {"default": {"backend": "SDPA"}}}, id="sdpa-default"),
    pytest.param({"diffusion_attention_config": {"per_role": {"self": {"backend": "SDPA"}}}}, id="sdpa-per-role"),
]


def _outcome(config_cls, **kwargs):
    try:
        return config_cls.from_kwargs(**kwargs)
    except ValueError as exc:
        return str(exc)


@pytest.mark.parametrize("gate", [False, True])
@pytest.mark.parametrize("mode", [None, "paged_scheduler", "dense_legacy"])
@pytest.mark.parametrize("incompatibility", _SHARED_INCOMPATIBILITIES)
def test_mode_matrix_agrees_across_config_classes(monkeypatch, gate, mode, incompatibility) -> None:
    # One table, run through both config classes: the projection validates the
    # same resolution the runtime does, so a stage can never serialize a mode
    # the runtime would reject (or vice versa).
    monkeypatch.setattr(diffusion_kv_config, "PAGED_KV_IMPLICIT_DEFAULT_ENABLED", gate)
    kwargs = {"model_class_name": PAGED_MODEL, **incompatibility}
    if mode is not None:
        kwargs["diffusion_kv_mode"] = mode

    runtime = _outcome(OmniDiffusionConfig, **kwargs)
    projection = _outcome(_DiffusionConfigProjection, **kwargs)

    assert isinstance(runtime, str) == isinstance(projection, str), (runtime, projection)
    if not isinstance(runtime, str):
        # The projection keeps an unset mode unset for serialization; otherwise
        # it must resolve to the same mode as the runtime config.
        assert projection.diffusion_kv_mode is None or projection.diffusion_kv_mode is runtime.diffusion_kv_mode
