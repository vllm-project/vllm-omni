# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability probe that gates ``/v1/audio/transcriptions``.

``AsyncOmniEngine`` advertises ``"transcription"`` only when a comprehension
stage's model class implements vLLM's ``SupportsTranscription``. These tests
drive ``_probe_transcription_support`` against synthetic stage pools so the
whole matrix runs on CPU without loading weights.
"""

import types

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Transcribing:
    """Stand-in for a model class that implements ``SupportsTranscription``."""

    supports_transcription = True


class _NotTranscribing:
    supports_transcription = False


def _make_pool(
    *,
    is_comprehension: bool = True,
    final_output: bool = True,
    final_output_type: str | None = "text",
    architectures: list[str] | None = ("StubArch",),
    has_client: bool = True,
    model_config: object | None = ...,
):
    """Build a stage pool stub shaped like ``StageRuntime.stage_pools`` entries."""
    if model_config is ...:
        model_config = types.SimpleNamespace(architectures=list(architectures) if architectures else architectures)
    client = types.SimpleNamespace(
        is_comprehension=is_comprehension,
        final_output=final_output,
        final_output_type=final_output_type,
    )
    return types.SimpleNamespace(
        stage_client=client if has_client else None,
        stage_vllm_config=types.SimpleNamespace(model_config=model_config),
    )


def _engine(pools) -> AsyncOmniEngine:
    """An uninitialized engine carrying only what the probe reads."""
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
    engine.stage_pools = pools
    return engine


@pytest.fixture
def resolve_to(monkeypatch):
    """Patch ``ModelRegistry.resolve_model_cls`` with a per-architecture map.

    Values may be a class (resolved) or an exception instance (raised), letting
    a single fixture cover both the happy path and registry failures.
    """
    from vllm.model_executor.models.registry import ModelRegistry

    def _apply(mapping: dict):
        calls: list = []

        def fake_resolve(architectures, model_config):
            calls.append(architectures)
            key = architectures[0] if isinstance(architectures, list | tuple) else architectures
            result = mapping[key]
            if isinstance(result, Exception):
                raise result
            return result, key

        monkeypatch.setattr(ModelRegistry, "resolve_model_cls", staticmethod(fake_resolve))
        return calls

    return _apply


def test_probe_true_when_comprehension_model_supports_transcription(resolve_to):
    resolve_to({"StubArch": _Transcribing})
    assert _engine([_make_pool()])._probe_transcription_support() is True


def test_probe_false_when_model_does_not_support_transcription(resolve_to):
    resolve_to({"StubArch": _NotTranscribing})
    assert _engine([_make_pool()])._probe_transcription_support() is False


def test_probe_ignores_non_comprehension_stages(resolve_to):
    """A talker/vocoder stage must not enable the endpoint even if it resolves."""
    calls = resolve_to({"StubArch": _Transcribing})
    engine = _engine([_make_pool(is_comprehension=False)])
    assert engine._probe_transcription_support() is False
    assert calls == [], "non-comprehension stages should never hit the registry"


def test_probe_scans_past_a_non_transcribing_comprehension_stage(resolve_to):
    """Pipelines may hold several comprehension stages; any one suffices."""
    resolve_to({"OtherArch": _NotTranscribing, "AsrArch": _Transcribing})
    engine = _engine(
        [
            _make_pool(architectures=["OtherArch"]),
            _make_pool(architectures=["AsrArch"]),
        ]
    )
    assert engine._probe_transcription_support() is True


@pytest.mark.parametrize(
    ("final_output", "final_output_type", "reason"),
    [
        pytest.param(False, None, "intermediate stage", id="not-a-final-output"),
        pytest.param(False, "text", "intermediate stage typed text", id="typed-but-not-final"),
        pytest.param(True, "audio", "audio terminal", id="final-but-not-text"),
        pytest.param(True, None, "untyped terminal", id="final-but-untyped"),
    ],
)
def test_probe_requires_a_text_terminal_stage(resolve_to, final_output, final_output_type, reason):
    """A transcription-capable stage that is not the text terminal is unaddressable.

    ``output_modalities`` resolves to the *last* stage emitting a modality, so
    advertising the endpoint for a deeper stage would route the request to the
    wrong stage and return that stage's output instead of a transcript.
    """
    resolve_to({"StubArch": _Transcribing})
    engine = _engine([_make_pool(final_output=final_output, final_output_type=final_output_type)])
    assert engine._probe_transcription_support() is False, reason


def test_probe_rejects_aura_shaped_pipeline(resolve_to):
    """Regression guard for the motivating case.

    ``aura_omni`` is ASR (stage 0, intermediate) -> LLM (stage 1, text terminal)
    -> TTS. Stage 0 is Qwen3-ASR and does support transcription, but a request
    lands on stage 1, so the endpoint must stay disabled.
    """
    resolve_to({"Qwen3ASRForConditionalGeneration": _Transcribing, "AuraQwen3VL": _NotTranscribing})
    engine = _engine(
        [
            _make_pool(
                architectures=["Qwen3ASRForConditionalGeneration"],
                final_output=False,
                final_output_type=None,
            ),
            _make_pool(architectures=["AuraQwen3VL"], final_output=True, final_output_type="text"),
        ]
    )
    assert engine._probe_transcription_support() is False


def test_probe_accepts_single_stage_asr_pipeline(resolve_to):
    """The shape a dedicated ASR pipeline has: one comprehension text terminal."""
    resolve_to({"Qwen3ASRForConditionalGeneration": _Transcribing})
    engine = _engine([_make_pool(architectures=["Qwen3ASRForConditionalGeneration"])])
    assert engine._probe_transcription_support() is True


def test_probe_short_circuits_on_first_match(resolve_to):
    calls = resolve_to({"AsrArch": _Transcribing, "OtherArch": _Transcribing})
    engine = _engine(
        [
            _make_pool(architectures=["AsrArch"]),
            _make_pool(architectures=["OtherArch"]),
        ]
    )
    assert engine._probe_transcription_support() is True
    assert calls == [["AsrArch"]], "probe should stop at the first supporting stage"


def test_probe_survives_registry_resolution_failure(resolve_to, caplog):
    """An unresolvable architecture disables the endpoint instead of crashing."""
    resolve_to({"StubArch": ValueError("no registered model")})
    engine = _engine([_make_pool()])
    assert engine._probe_transcription_support() is False
    assert "transcription capability probe could not resolve" in caplog.text


def test_probe_keeps_scanning_after_a_resolution_failure(resolve_to):
    """One bad stage must not mask a later stage that does support ASR."""
    resolve_to({"BadArch": ValueError("boom"), "AsrArch": _Transcribing})
    engine = _engine(
        [
            _make_pool(architectures=["BadArch"]),
            _make_pool(architectures=["AsrArch"]),
        ]
    )
    assert engine._probe_transcription_support() is True


@pytest.mark.parametrize(
    "pool_kwargs",
    [
        pytest.param({"has_client": False}, id="stage-without-client"),
        pytest.param({"architectures": []}, id="empty-architectures"),
        pytest.param({"architectures": None}, id="architectures-none"),
        pytest.param({"model_config": None}, id="model-config-none"),
    ],
)
def test_probe_false_on_incomplete_stage_metadata(resolve_to, pool_kwargs):
    """Stages missing a client or architectures are skipped, not fatal."""
    calls = resolve_to({"StubArch": _Transcribing})
    engine = _engine([_make_pool(**pool_kwargs)])
    assert engine._probe_transcription_support() is False
    assert calls == []


def test_probe_false_with_no_stages(resolve_to):
    resolve_to({})
    assert _engine([])._probe_transcription_support() is False


def test_probe_uses_real_vllm_interface_semantics():
    """Guard the contract against upstream drift in ``supports_transcription``.

    The probe delegates to vLLM's own predicate rather than duck-typing an
    attribute, so a plain class carrying ``supports_transcription = True`` must
    still be rejected unless it actually satisfies the interface.
    """
    from vllm.model_executor.models.interfaces import supports_transcription

    assert supports_transcription(_NotTranscribing) is False


def test_aura_omni_asr_stage_is_still_intermediate():
    """Canary on the real registry behind ``test_probe_rejects_aura_shaped_pipeline``.

    If ``aura_omni``'s ASR stage ever becomes a text final output, the probe
    would start advertising ``/v1/audio/transcriptions`` for it -- revisit the
    routing question before letting that happen.
    """
    from vllm_omni.config.pipeline_registry import OMNI_PIPELINES

    asr_stage = next(s for s in OMNI_PIPELINES["aura_omni"].stages if s.model_stage == "asr")
    assert asr_stage.model_arch == "Qwen3ASRForConditionalGeneration"
    assert asr_stage.final_output is False


def test_qwen3_asr_is_a_transcription_capable_architecture():
    """``Qwen3ASRForConditionalGeneration`` is the model behind ``aura_omni``
    stage 0; upstream already implements the interface, so the probe enables
    the endpoint for it with no vllm-omni model changes."""
    from vllm.model_executor.models.interfaces import supports_transcription
    from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration

    assert supports_transcription(Qwen3ASRForConditionalGeneration) is True
