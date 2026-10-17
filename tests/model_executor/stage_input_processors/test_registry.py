# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for the stage-input processor registry (table-driven).

Pure-logic, CPU-only: exercises kind inference, structural validation,
dead-processor hints and ``resolve_processor`` without executing any processor
logic or loading model weights.

Subject under test: ``vllm_omni.model_executor.stage_input_processors._registry``
"""

import importlib
import warnings

import pytest

from vllm_omni.model_executor.stage_input_processors import (
    ProcessorKind,
    ProcessorSpec,
    ProcessorValidationError,
    _registry,
    dead_processor_hint,
    infer_kind,
    register_processor,
    resolve_processor,
    validate_processor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ---------------------------------------------------------------------------
# Dummy processors (simple callables used as validation subjects).
# ---------------------------------------------------------------------------


def placeholder_a_token_only(source_outputs, _prompt=None, _requires_multimodal_data=False):
    return []


def placeholder_b_token_only(prompt, ctx=None):
    return []


def diffusion_a_ar2diffusion(source_outputs, _prompt=None, _requires_multimodal_data=False):
    return {}


def producer_a_full_payload(transfer_manager, pooling_output, request):
    return None


def producer_b_full_payload(transfer_manager, pooling_output, request, is_finished=False):
    return None


def producer_c_full_payload(transfer_manager, multimodal_output, request, is_finished=False):
    return None


def producer_d_full_payload(transfer_manager, pooling_output, request, *, is_finished):
    return None


def producer_a_async_chunk(transfer_manager, multimodal_output, request, is_finished=False):
    return None


def producer_b_async_chunk(transfer_manager, multimodal_output, request):
    return None


def producer_c_async_chunk(transfer_manager, multimodal_output, request, is_finished, /):
    return None


def producer_kwargs_full_payload(**kwargs):
    return None


def producer_kwargs_async_chunk(**kwargs):
    return None


def producer_badname_full_payload(transfer_manager, payload, request, is_finished=False):
    return None


def producer_badname_async_chunk(transfer_manager, payload, request, is_finished=False):
    return None


def producer_positional_full_payload(transfer_manager, pooling_output, request, /):
    return None


def talker2codec_shape(stage_list, engine_input_source, prompt=None, requires_multimodal_data=False):
    return []


def _mod_path(fn_name: str) -> str:
    """Dotted path to a callable defined in this test module."""
    return f"{__name__}.{fn_name}"


# ---------------------------------------------------------------------------
# Kind inference (suffix rules).
# ---------------------------------------------------------------------------


class TestInferKind:
    @pytest.mark.parametrize(
        "path,expected",
        [
            ("pkg.mod.foo_token_only", "placeholder_prompt_builder"),
            ("pkg.mod.foo_full_payload", "producer_full_payload"),
            ("pkg.mod.foo_batch", "producer_full_payload"),
            ("pkg.mod.foo_async_chunk", "producer_async_chunk"),
            ("pkg.mod.ar2diffusion", "diffusion_input_builder"),
            ("pkg.mod.ar2dit", "diffusion_input_builder"),
            ("pkg.mod.thinker2imagegen", "diffusion_input_builder"),
            ("pkg.mod.legacy_builder", "legacy_orchestrator_builder"),
            ("pkg.mod.talker2code2wav", "legacy_orchestrator_builder"),
            ("vllm_omni.model_executor.stage_input_processors.moss_tts.talker2codec", "legacy_multi_source"),
            ("pkg.moss_tts.talker2codec", "legacy_multi_source"),
            (
                "vllm_omni.model_executor.stage_input_processors.moss_tts.talker2codec_delay_async_chunk",
                "producer_async_chunk",
            ),
        ],
    )
    def test_suffix_rules(self, path: str, expected: ProcessorKind):
        assert infer_kind(None, path=path) == expected

    def test_invalid_path_raises(self):
        with pytest.raises(ValueError):
            infer_kind(None, path="")
        with pytest.raises(ValueError):
            infer_kind(None, path="nodots")

    def test_register_processor_override(self):
        register_processor("pkg.mod.custom_path", "diffusion_input_builder")
        try:
            assert infer_kind(None, path="pkg.mod.custom_path") == "diffusion_input_builder"
        finally:
            _registry._KIND_OVERRIDES.clear()

    def test_register_processor_invalid_kind(self):
        with pytest.raises(ValueError):
            register_processor("pkg.mod.custom_path", "bogus_kind")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Structural validation (table-driven: ok / warn / reject).
# ---------------------------------------------------------------------------


def _ok(fn, kind, path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_processor(fn, kind=kind, path=path)


def _warns(fn, kind, path, match):
    with pytest.warns(RuntimeWarning, match=match):
        validate_processor(fn, kind=kind, path=path)


def _rejects(fn, kind, path, rule, match=None):
    with pytest.raises(ProcessorValidationError) as excinfo:
        validate_processor(fn, kind=kind, path=path)
    assert excinfo.value.rule == rule
    if match is not None:
        assert match in str(excinfo.value)


_VALIDATION_CASES = [
    # (fn, kind, path, expected-behaviour, extra)
    # full-payload producers
    pytest.param(
        producer_b_full_payload,
        "producer_full_payload",
        "pkg.mod.producer_b_full_payload",
        "ok",
        None,
        id="full-payload-ok",
    ),
    pytest.param(
        producer_a_full_payload,
        "producer_full_payload",
        "pkg.mod.producer_a_full_payload",
        "warn",
        "is_finished",
        id="full-payload-missing-is_finished-warns",
    ),
    pytest.param(
        producer_c_full_payload,
        "producer_full_payload",
        "pkg.mod.producer_c_full_payload",
        "reject",
        ("producer_kwargs", None),
        id="full-payload-cross-kind-second-param-rejected",
    ),
    pytest.param(
        producer_badname_full_payload,
        "producer_full_payload",
        "pkg.mod.producer_badname_full_payload",
        "reject",
        ("producer_kwargs", "pooling_output"),
        id="full-payload-bad-param-names-rejected",
    ),
    pytest.param(
        producer_positional_full_payload,
        "producer_full_payload",
        "pkg.mod.producer_positional_full_payload",
        "reject",
        ("producer_kwargs", None),
        id="full-payload-positional-only-rejected",
    ),
    pytest.param(
        producer_kwargs_full_payload,
        "producer_full_payload",
        "pkg.mod.producer_kwargs_full_payload",
        "ok",
        None,
        id="full-payload-var-keyword-accepted",
    ),
    pytest.param(
        producer_d_full_payload,
        "producer_full_payload",
        "pkg.mod.producer_d_full_payload",
        "ok",
        None,
        id="full-payload-required-kw-only-is_finished-accepted",
    ),
    pytest.param(
        producer_b_full_payload,
        "producer_full_payload",
        "pkg.mod.producer_b_full_payload",
        "ok",
        None,
        id="full-payload-default-is_finished-accepted",
    ),
    # async-chunk producers
    pytest.param(
        producer_badname_async_chunk,
        "producer_async_chunk",
        "pkg.mod.producer_badname_async_chunk",
        "reject",
        ("producer_kwargs", "multimodal_output"),
        id="async-chunk-bad-param-names-rejected",
    ),
    pytest.param(
        producer_kwargs_async_chunk,
        "producer_async_chunk",
        "pkg.mod.producer_kwargs_async_chunk",
        "ok",
        None,
        id="async-chunk-var-keyword-accepted",
    ),
    pytest.param(
        producer_a_async_chunk,
        "producer_async_chunk",
        "pkg.mod.producer_a_async_chunk",
        "ok",
        None,
        id="async-chunk-ok",
    ),
    pytest.param(
        producer_b_async_chunk,
        "producer_async_chunk",
        "pkg.mod.producer_b_async_chunk",
        "reject",
        ("is_finished_required", "producer_b_async_chunk"),
        id="async-chunk-missing-is_finished-hard-fail",
    ),
    pytest.param(
        producer_c_async_chunk,
        "producer_async_chunk",
        "pkg.mod.producer_c_async_chunk",
        "reject",
        ("is_finished_positional_only", None),
        id="async-chunk-positional-only-is_finished-hard-fail",
    ),
    # orchestrator-side
    pytest.param(
        placeholder_a_token_only,
        "placeholder_prompt_builder",
        "pkg.mod.placeholder_a_token_only",
        "ok",
        None,
        id="placeholder-ok",
    ),
    pytest.param(
        placeholder_b_token_only,
        "placeholder_prompt_builder",
        "pkg.mod.placeholder_b_token_only",
        "ok",
        None,
        id="placeholder-ctx-contract-ok",
    ),
    pytest.param(
        diffusion_a_ar2diffusion,
        "diffusion_input_builder",
        "pkg.mod.diffusion_a_ar2diffusion",
        "ok",
        None,
        id="diffusion-ok",
    ),
    pytest.param(
        talker2codec_shape,
        "legacy_multi_source",
        "vllm_omni.model_executor.stage_input_processors.moss_tts.talker2codec",
        "ok",
        None,
        id="legacy-multi-source-allowlist-ok",
    ),
    pytest.param(
        talker2codec_shape,
        "legacy_multi_source",
        "pkg.mod.talker2codec",
        "reject",
        ("legacy_multi_source_allowlist", None),
        id="legacy-multi-source-rejects-non-moss",
    ),
    pytest.param(
        object(),
        "producer_full_payload",
        "pkg.mod.producer_full_payload",
        "reject",
        ("inspectable_signature", None),
        id="uninspectable-callable-fails",
    ),
]


@pytest.mark.parametrize("fn,kind,path,behaviour,extra", _VALIDATION_CASES)
def test_validate_processor_table(fn, kind, path, behaviour, extra):
    if behaviour == "ok":
        _ok(fn, kind, path)
    elif behaviour == "warn":
        _warns(fn, kind, path, extra)
    else:
        rule, match = extra
        _rejects(fn, kind, path, rule, match)


# ---------------------------------------------------------------------------
# Dead-processor hint (gate three-state).
# ---------------------------------------------------------------------------


# One entry per legacy test method; each entry carries the scenarios the
# original method asserted (keeps the collected-test count unchanged).
_DEAD_CASES = [
    pytest.param(
        [
            ("placeholder_prompt_builder", True, True, False, False, True),
        ],
        id="async-chunk-receiving-downstream-dead",
    ),
    pytest.param(
        [
            ("diffusion_input_builder", True, False, False, False, False),
        ],
        id="async-chunk-non-receiving-downstream-alive",
    ),
    pytest.param(
        [
            ("diffusion_input_builder", False, False, False, False, False),
            ("diffusion_input_builder", False, False, True, False, True),
        ],
        id="sync-mode-alive-unless-overridden",
    ),
    pytest.param(
        [
            ("placeholder_prompt_builder", False, False, True, True, False),
            ("placeholder_prompt_builder", False, False, True, False, True),
            ("placeholder_prompt_builder", True, True, True, True, True),
        ],
        id="sync-mode-selected-sync-hook-is-not-dead",
    ),
    pytest.param(
        [
            ("producer_full_payload", True, True, False, False, False),
            ("producer_async_chunk", True, True, False, False, False),
        ],
        id="producer-never-dead",
    ),
]


@pytest.mark.parametrize("scenarios", _DEAD_CASES)
def test_dead_processor_hint_table(scenarios):
    for kind, async_chunk, downstream, has_sync, custom_is_sync, expected in scenarios:
        assert (
            dead_processor_hint(
                kind,
                async_chunk=async_chunk,
                downstream_receives_async_chunks=downstream,
                has_sync=has_sync,
                custom_is_sync=custom_is_sync,
            )
            is expected
        )


# ---------------------------------------------------------------------------
# Resolution entry point.
# ---------------------------------------------------------------------------


class TestResolveProcessor:
    @pytest.mark.parametrize(
        "fn_name,expected_kind",
        [
            ("placeholder_a_token_only", "placeholder_prompt_builder"),
            ("producer_b_full_payload", None),
            ("producer_d_full_payload", "producer_full_payload"),
        ],
    )
    def test_resolve_normal(self, fn_name, expected_kind):
        spec = resolve_processor(_mod_path(fn_name), expected_kind=expected_kind)
        assert isinstance(spec, ProcessorSpec)
        assert spec.kind == (expected_kind or infer_kind(None, path=_mod_path(fn_name)))
        assert spec.path == _mod_path(fn_name)
        assert spec.fn is getattr(importlib.import_module(__name__), fn_name)

    def test_resolve_drop_in_equivalence(self):
        """resolve_processor().fn must be the same object as the legacy lookup."""
        spec = resolve_processor(
            _mod_path("producer_b_full_payload"),
            expected_kind="producer_full_payload",
        )
        legacy = getattr(importlib.import_module(__name__), "producer_b_full_payload")
        assert spec.fn is legacy

    def test_expected_kind_mismatch_raises(self):
        with pytest.raises(ProcessorValidationError) as excinfo:
            resolve_processor(
                _mod_path("placeholder_a_token_only"),
                expected_kind="producer_full_payload",
            )
        assert excinfo.value.rule == "expected_kind"
        assert "placeholder_prompt_builder" in str(excinfo.value)

    def test_missing_symbol_raises(self):
        with pytest.raises(AttributeError):
            resolve_processor(_mod_path("does_not_exist"))

    def test_package_exports(self):
        import vllm_omni.model_executor.stage_input_processors as sip

        assert sip.resolve_processor is resolve_processor
        assert sip.ProcessorSpec is ProcessorSpec
        assert sip.ProcessorValidationError is ProcessorValidationError
        assert sip.infer_kind is infer_kind
        assert sip.dead_processor_hint is dead_processor_hint
        assert sip.register_processor is register_processor
