# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""``DuplexModelPlugin`` loading/validation helpers and the ``PipelineConfig.duplex_plugin`` binding."""

from __future__ import annotations

import base64
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig, PipelineConfig
from vllm_omni.engine.duplex.config import DuplexCapabilities, DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexAppendPlan
from vllm_omni.engine.duplex.plugin import (
    DefaultDuplexModelSessionState,
    DuplexDataPlane,
    DuplexDataPlaneContext,
    DuplexModelPlugin,
    DuplexModelSessionState,
    DuplexRuntimeConfigError,
    PcmAppendBuffer,
    coerce_int,
    load_duplex_plugin,
    payload_turn_id,
    reject_changed_runtime_value,
    validate_duplex_plugin_sampling,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode_audio(audio: object, sample_rate_hz: int, fmt: str, speed: float | None) -> str | None:
    del audio, sample_rate_hz, fmt, speed
    return None


class FakeDataPlane(DuplexDataPlane):
    def begin_request(self, request_id: str) -> None:
        pass

    def is_terminal(self, request_id: str | None) -> bool:
        return False

    def mark_terminal(self, request_id: str) -> None:
        pass

    def close_stream(self, request_id: str) -> None:
        pass

    def close_session(self, session_id: str, *, active_request_id: str | None = None) -> None:
        pass

    def project(self, result, *, context=None):
        del result, context
        return []


class FakePlugin(DuplexModelPlugin):
    """Minimal plugin; referenced by dotted path from ``load_duplex_plugin`` tests."""

    plugin_id = "fake-duplex"

    def __init__(self, encode_audio) -> None:
        super().__init__(encode_audio)
        self.data_plane = FakeDataPlane()

    def configure_sampling_params(self, *, runtime_config, defaults):
        del runtime_config
        return tuple(defaults)

    def plan_append(self, **kwargs) -> DuplexAppendPlan:
        del kwargs
        return DuplexAppendPlan(prompt={"prompt_token_ids": [1]})

    def decide_output(self, **kwargs):
        del kwargs
        return None

    def create_session_state(self) -> DuplexModelSessionState:
        raise NotImplementedError

    def capabilities(self, *, max_sessions: int) -> DuplexCapabilities:
        del max_sessions
        return DuplexCapabilities()

    def validate_client_extra_body(self, extra_body: object) -> None:
        pass

    async def prepare_runtime_config(self, config: DuplexSessionConfig, *, model_config: Any) -> dict[str, object]:
        del config, model_config
        return {}

    def runtime_config_for_update(self, config, current):
        del config
        return dict(current)

    def data_plane_context(self, **kwargs) -> object:
        return dict(kwargs)


class PluginWithoutId(FakePlugin):
    plugin_id = ""


class PluginWithoutDataPlane(FakePlugin):
    def __init__(self, encode_audio) -> None:
        DuplexModelPlugin.__init__(self, encode_audio)


class PluginWithWrongDataPlane(FakePlugin):
    def __init__(self, encode_audio) -> None:
        DuplexModelPlugin.__init__(self, encode_audio)
        self.data_plane = object()  # type: ignore[assignment]


class NotAPlugin:
    plugin_id = "lookalike"
    data_plane = FakeDataPlane()

    def __init__(self, encode_audio) -> None:
        self.encode_audio = encode_audio


def _path(cls: type) -> str:
    return f"{__name__}.{cls.__name__}"


# --------------------------------------------------------------------------- #
# load_duplex_plugin                                                          #
# --------------------------------------------------------------------------- #


def test_load_duplex_plugin_instantiates_the_class_at_the_dotted_path() -> None:
    plugin = load_duplex_plugin(_path(FakePlugin), _encode_audio)

    assert type(plugin) is FakePlugin
    assert plugin.plugin_id == "fake-duplex"
    assert isinstance(plugin.data_plane, FakeDataPlane)
    # Defaults of the optional class-level policy.
    assert plugin.private_runtime_config_keys == frozenset()
    assert plugin.silence_continuation_samples == 16000
    assert plugin.runtime_config_for_function_output(DuplexSessionConfig(), {}, {"name": "tool"}) is None


def test_load_duplex_plugin_rejects_a_non_plugin_class() -> None:
    with pytest.raises(TypeError, match="not a DuplexModelPlugin"):
        load_duplex_plugin(_path(NotAPlugin), _encode_audio)


def test_load_duplex_plugin_requires_a_plugin_id() -> None:
    with pytest.raises(TypeError, match="plugin_id"):
        load_duplex_plugin(_path(PluginWithoutId), _encode_audio)


@pytest.mark.parametrize("plugin_cls", [PluginWithoutDataPlane, PluginWithWrongDataPlane])
def test_load_duplex_plugin_requires_a_data_plane(plugin_cls: type) -> None:
    with pytest.raises(TypeError, match="DuplexDataPlane as data_plane"):
        load_duplex_plugin(_path(plugin_cls), _encode_audio)


def test_load_duplex_plugin_rejects_paths_without_a_module() -> None:
    with pytest.raises(ValueError, match="Invalid duplex plugin path"):
        load_duplex_plugin("FakePlugin", _encode_audio)


def test_load_duplex_plugin_propagates_missing_attributes() -> None:
    with pytest.raises(AttributeError):
        load_duplex_plugin(f"{__name__}.NoSuchPlugin", _encode_audio)


# --------------------------------------------------------------------------- #
# validate_duplex_plugin_sampling                                             #
# --------------------------------------------------------------------------- #


def test_validate_duplex_plugin_sampling_accepts_one_parameter_per_stage() -> None:
    plugin = FakePlugin(_encode_audio)

    validate_duplex_plugin_sampling(plugin, sampling_defaults=(SamplingParams(), SamplingParams()))
    validate_duplex_plugin_sampling(plugin, sampling_defaults=())


def test_validate_duplex_plugin_sampling_rejects_stage_count_mismatch() -> None:
    class WrongStageCountPlugin(FakePlugin):
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return defaults[:1]

    with pytest.raises(ValueError, match="one sampling parameter per stage"):
        validate_duplex_plugin_sampling(
            WrongStageCountPlugin(_encode_audio),
            sampling_defaults=(SamplingParams(), SamplingParams()),
        )


def test_validate_duplex_plugin_sampling_rejects_type_mismatch() -> None:
    class WrongSamplingTypePlugin(FakePlugin):
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return tuple(object() for _ in defaults)

    with pytest.raises(TypeError, match="sampling parameter type mismatch for stage 0"):
        validate_duplex_plugin_sampling(
            WrongSamplingTypePlugin(_encode_audio),
            sampling_defaults=(SamplingParams(), SamplingParams()),
        )


def test_validate_duplex_plugin_sampling_skips_type_check_for_stages_without_defaults() -> None:
    class ObjectSamplingPlugin(FakePlugin):
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return tuple(object() for _ in defaults)

    validate_duplex_plugin_sampling(ObjectSamplingPlugin(_encode_audio), sampling_defaults=(None, None))


def test_validate_duplex_plugin_sampling_rejects_non_tuple_results() -> None:
    class ListSamplingPlugin(FakePlugin):
        def configure_sampling_params(self, *, runtime_config, defaults):
            del runtime_config
            return list(defaults)

    with pytest.raises(TypeError, match="as a tuple"):
        validate_duplex_plugin_sampling(ListSamplingPlugin(_encode_audio), sampling_defaults=(SamplingParams(),))


# --------------------------------------------------------------------------- #
# Runtime-config helpers                                                      #
# --------------------------------------------------------------------------- #


def test_reject_changed_runtime_value_only_raises_on_change() -> None:
    reject_changed_runtime_value("voice-a", "voice-a", message="voice is fixed", code="voice_changed")

    with pytest.raises(DuplexRuntimeConfigError, match="voice is fixed") as excinfo:
        reject_changed_runtime_value("voice-b", "voice-a", message="voice is fixed", code="voice_changed")

    assert excinfo.value.code == "voice_changed"
    assert isinstance(excinfo.value, ValueError)


def test_reject_changed_runtime_value_uses_the_requested_error_class() -> None:
    class CustomRuntimeConfigError(DuplexRuntimeConfigError):
        pass

    with pytest.raises(CustomRuntimeConfigError) as excinfo:
        reject_changed_runtime_value(1, 2, message="changed", code="custom", error_cls=CustomRuntimeConfigError)

    assert excinfo.value.code == "custom"


def test_duplex_runtime_config_error_defaults_its_code() -> None:
    assert DuplexRuntimeConfigError("bad").code == "invalid_duplex_runtime_config"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"duplex_turn_id": 3}, 3),
        ({"duplex_turn_id": "7"}, 7),
        ({"model_turn_id": 2}, 2),
        ({"duplex_turn_id": 5, "model_turn_id": 9}, 5),
        ({"duplex_turn_id": "not-a-number"}, None),
        ({}, None),
        ("turn", None),
        (None, None),
    ],
)
def test_payload_turn_id_prefers_duplex_turn_id_and_coerces(payload: object, expected: int | None) -> None:
    assert payload_turn_id(payload) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(4, 4), ("11", 11), (2.9, 2), (True, 1), (None, None), ("x", None), (object(), None), ([1], None)],
)
def test_coerce_int(value: object, expected: int | None) -> None:
    assert coerce_int(value) == expected


# --------------------------------------------------------------------------- #
# PipelineConfig binding                                                      #
# --------------------------------------------------------------------------- #


def test_pipeline_config_binds_one_duplex_plugin_path() -> None:
    field_defaults = {f.name: f.default for f in fields(PipelineConfig)}

    assert "duplex_plugin" in field_defaults
    assert field_defaults["duplex_plugin"] is None

    # The pre-framework fields survive only so that the pipelines of the models
    # that are not ported yet still construct (RFC vllm-omni#7181 ports them).
    # Nothing in the duplex framework may read them: a pipeline is a duplex
    # model iff it declares ``duplex_plugin``.
    legacy_fields = ("duplex_runtime_extension", "duplex_serving_adapter", "duplex_control_enabled")
    for legacy_field in legacy_fields:
        assert legacy_field in field_defaults

    framework_sources = [
        *(Path("vllm_omni/engine/duplex").glob("*.py")),
        *(Path("vllm_omni/entrypoints/duplex").glob("*.py")),
        Path("vllm_omni/engine/duplex_omni_engine.py"),
        Path("vllm_omni/engine/duplex_orchestrator.py"),
        Path("vllm_omni/entrypoints/duplex_omni.py"),
    ]
    readers = [
        f"{source}:{legacy_field}"
        for source in framework_sources
        for legacy_field in legacy_fields
        if legacy_field in source.read_text(encoding="utf-8")
    ]
    assert readers == []

    # Same rule for the runtime-config knob the framework stopped reading: the
    # append-retry table it bounded went away with the correlated append RPC.
    assert "completed_append_cache_size" in {f.name for f in fields(DuplexSessionRuntimeConfig)}
    assert [
        str(source)
        for source in framework_sources
        if "completed_append_cache_size" in source.read_text(encoding="utf-8")
    ] == []


# --------------------------------------------------------------------------- #
# Plugin defaults shared by every model                                       #
# --------------------------------------------------------------------------- #


def test_default_silence_unit_is_16k_zeros_and_follows_the_plugin_attributes() -> None:
    plugin = FakePlugin(_encode_audio)

    unit = plugin.silence_unit_payload()

    assert unit["type"] == "audio"
    assert unit["format"] == "pcm_f32le"
    assert unit["sample_rate_hz"] == 16000
    assert base64.b64decode(unit["audio"]) == bytes(16000 * 4)

    class FramePlugin(FakePlugin):
        silence_continuation_samples = 1920
        silence_continuation_sample_rate_hz = 24000

    frame_unit = FramePlugin(_encode_audio).silence_unit_payload()
    assert frame_unit["sample_rate_hz"] == 24000
    assert len(base64.b64decode(frame_unit["audio"])) == 1920 * 4


def test_default_extra_body_validation_rejects_the_private_keys_naming_the_plugin() -> None:
    class PrivateKeysPlugin(FakePlugin):
        plugin_id = "keyed"
        private_runtime_config_keys = frozenset({"secret_a", "secret_b"})

        def validate_client_extra_body(self, extra_body: object) -> None:
            DuplexModelPlugin.validate_client_extra_body(self, extra_body)

    plugin = PrivateKeysPlugin(_encode_audio)
    plugin.validate_client_extra_body(None)
    plugin.validate_client_extra_body({"auto_response": True})
    with pytest.raises(
        DuplexRuntimeConfigError, match="keyed runtime configuration is server-owned: secret_a, secret_b"
    ):
        plugin.validate_client_extra_body({"secret_b": 1, "secret_a": 2})


def test_default_data_plane_context_is_the_framework_dataclass() -> None:
    class DefaultContextPlugin(FakePlugin):
        def data_plane_context(self, **kwargs):
            return DuplexModelPlugin.data_plane_context(self, **kwargs)

    context = DefaultContextPlugin(_encode_audio).data_plane_context(
        epoch=2,
        turn_id=3,
        active_response_turn_id=1,
        active_response_id="resp",
        auto_responds=True,
        response_format="pcm16",
        speed=1.5,
        modalities=("audio",),
    )

    assert isinstance(context, DuplexDataPlaneContext)
    assert context == DuplexDataPlaneContext(
        epoch=2,
        turn_id=3,
        active_response_turn_id=1,
        active_response_id="resp",
        auto_responds=True,
        response_format="pcm16",
        speed=1.5,
        modalities=("audio",),
    )


def test_default_session_state_implements_the_shared_transitions() -> None:
    class Buffer(PcmAppendBuffer):
        pending_byte_count = 0

        def clear(self) -> None: ...

        def clear_force_listen(self) -> None: ...

        def has_pending(self) -> bool:
            return False

        def has_reserved(self) -> bool:
            return False

        def prepare_append(self, payload, *, operation_id, chunk_period_ms, allow_emit):
            return None

        def prepare_commit(self, *, operation_id, chunk_period_ms):
            raise NotImplementedError

        def flush(self, *, chunk_period_ms):
            return None

    state = DefaultDuplexModelSessionState(audio_buffer=Buffer())

    assert isinstance(state, DuplexModelSessionState)
    assert state.committed_audio_reserved_bytes == 0 and state.continuation_units == 0
    state.retain_committed_audio({"audio": "a"}, operation_id="op-1", reserved_bytes=10)
    state.retain_committed_audio({"audio": "b"}, operation_id="op-2", reserved_bytes=5)
    state.deferred_response_create = True
    assert state.committed_audio_payload == {"audio": "b"}
    assert state.committed_audio_operation_id == "op-2"
    assert state.clear_committed_audio() == 15
    assert state.committed_audio_payload is None
    assert state.committed_audio_operation_id is None
    assert state.deferred_response_create is False

    state.continuation_owner_id = "response:r"
    state.continuation_units = 3
    state.pending_silence_owner_id = "response:r"
    state.clear_continuation()
    assert state.continuation_owner_id is None
    assert state.continuation_units == 0
    assert state.pending_silence_task is None and state.pending_silence_owner_id is None
