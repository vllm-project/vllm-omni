# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Opt-in deployment settings and real Qwen extraction hooks, without a GPU."""

import json
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from tests.model_executor.models.qwen3_omni.test_memory_card_extraction import context as _context_fixture
from tests.model_executor.models.qwen3_omni.test_memory_card_request import (
    output,
    result,
    upsert,
)
from tests.model_executor.models.qwen3_omni.test_memory_card_request import qwen_processor as _processor_fixture
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig, load_deploy_config
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
context = _context_fixture
qwen_processor = _processor_fixture


def settings(**kwargs):
    from vllm_omni.config.context_memory import DuplexContextMemoryConfig

    return DuplexContextMemoryConfig(**kwargs)


def configured(processor, **kwargs):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = processor
    plugin.configure_session_runtime(DuplexSessionRuntimeConfig(context_memory=settings(**kwargs)))
    return plugin


def test_deployment_default_is_off_and_yaml_config_is_frozen(tmp_path):
    assert DuplexSessionRuntimeConfig().context_memory is None
    path = tmp_path / "deploy.yaml"
    path.write_text("duplex_session:\n  context_memory:\n    max_chars: 123\nstages: []\n", encoding="utf-8")
    config = load_deploy_config(path).duplex_session.context_memory
    assert config == settings(max_chars=123)
    with pytest.raises(FrozenInstanceError):
        config.max_chars = 1


@pytest.mark.parametrize(
    "value",
    [False, True, 1, "enabled", [], {"unexpected": 1}, {"processor": "remote"}, {"max_chars": True}],
)
def test_deployment_rejects_untyped_or_unknown_settings(value):
    with pytest.raises((ValueError, TypeError)):
        DuplexSessionRuntimeConfig(context_memory=value)


@pytest.mark.parametrize(
    "field,bad",
    [
        ("max_sources", 0),
        ("max_sources", 9),
        ("max_sources", True),
        ("max_chars", 0),
        ("max_chars", 8193),
        ("max_chars", 1.5),
        ("input_token_limit", 0),
        ("input_token_limit", 2049),
        ("output_token_limit", False),
        ("output_token_limit", 513),
        ("timeout_s", 0),
        ("timeout_s", 31),
        ("timeout_s", float("nan")),
        ("compute_input_tokens", 0),
        ("compute_output_tokens", 0),
        ("compute_output_tokens", 1),
        ("compute_window_s", float("inf")),
        ("compute_window_s", 0),
        ("cooldown_s", -1),
        ("idle_s", True),
    ],
)
def test_invalid_limits_fail_before_runtime(field, bad):
    with pytest.raises(ValueError):
        settings(**{field: bad})


@pytest.mark.parametrize("key", ["context_memory", "duplex_session", "memory_compute_budget"])
def test_client_cannot_enable_or_override_extraction(key):
    from types import MappingProxyType

    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    bodies: tuple[Mapping[str, object], ...] = ({key: {}}, MappingProxyType({key: {}}))
    for body in bodies:
        with pytest.raises(DuplexRuntimeConfigError):
            plugin.validate_client_extra_body(body)


def test_unsupported_plugin_rejects_opt_in_before_admission():
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

    plugin = MiniCPMO45DuplexPlugin(lambda *args: "")
    plugin.configure_session_runtime(DuplexSessionRuntimeConfig())
    with pytest.raises(ValueError, match="context_memory"):
        plugin.configure_session_runtime(DuplexSessionRuntimeConfig(context_memory=settings()))


@pytest.mark.asyncio
async def test_disabled_hooks_need_no_processor_capture_or_deployed_limit(context):
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.configure_session_runtime(DuplexSessionRuntimeConfig())
    assert plugin.supports_context_auxiliary_requests is False
    assert plugin.context_auxiliary_policy(runtime_config={}) is None
    assert await plugin.prepare_context_auxiliary_request(context=context, runtime_config={}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, True, 0, -1, 1.5])
async def test_enabled_hooks_require_actual_deployed_model_limit(qwen_processor, limit):
    plugin = configured(qwen_processor)
    model_config = SimpleNamespace(max_model_len=limit, hf_config=SimpleNamespace())
    with pytest.raises(DuplexRuntimeConfigError, match="max_model_len"):
        await plugin.prepare_runtime_config(DuplexSessionConfig(model="Qwen/test"), model_config=model_config)


@pytest.mark.asyncio
async def test_hook_counts_real_template_and_ignores_client_tools_voice_and_instructions(context, qwen_processor):
    from vllm_omni.engine.duplex.memory_types import DuplexContextAuxiliaryRechunk
    from vllm_omni.model_executor.models.qwen3_omni.duplex.memory_request import build_memory_request

    plugin = configured(qwen_processor, timeout_s=2.0)
    client = DuplexSessionConfig(model="Qwen/test", instructions="CLIENT SECRET", voice="Chelsie")
    runtime = await plugin.prepare_runtime_config(
        client, model_config=SimpleNamespace(max_model_len=4096, hf_config=SimpleNamespace())
    )
    policy = plugin.context_auxiliary_policy(runtime_config=runtime)
    assert plugin.supports_context_auxiliary_requests
    assert (policy.max_sources, policy.max_chars) == (8, 512)
    assert policy.compute_budget.max_input_tokens == 4096
    request = await plugin.prepare_context_auxiliary_request(context=context, runtime_config=runtime)
    expected = build_memory_request(
        context, processor=qwen_processor, input_token_limit=2048, output_token_limit=512, max_model_len=4096
    )
    assert request.prompt == expected.prompt
    assert request.stage_id == 0 and request.timeout_s == 2.0
    assert request.retry_delays_s == (0,) and request.retry_cooldown_s is None
    assert "CLIENT SECRET" not in qwen_processor.tokenizer.decode(request.prompt["prompt_token_ids"])
    bound = replace(request, context_work=context.work)
    assert (
        plugin.decode_context_auxiliary_result(request=bound, output=output(json.dumps(result(upsert())))) is not None
    )
    plugin._memory_max_model_len = len(request.prompt["prompt_token_ids"]) + 511
    assert isinstance(
        await plugin.prepare_context_auxiliary_request(context=context, runtime_config=runtime),
        DuplexContextAuxiliaryRechunk,
    )


@pytest.mark.asyncio
async def test_real_hooks_runner_rechunks_then_admits_only_exact_smaller_capture(qwen_processor):
    from tests.engine.duplex.test_session_auxiliary import auxiliary_submissions, wait_until
    from tests.engine.duplex.test_session_runner import close_harness, open_harness
    from vllm_omni.engine.duplex.commands import Heartbeat

    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = qwen_processor
    limits = settings(max_chars=8192, idle_s=0, cooldown_s=0)
    h = await open_harness(
        auto_response=False,
        stage_count=3,
        plugin=plugin,
        model="Qwen/test",
        model_config=SimpleNamespace(max_model_len=4096, hf_config=SimpleNamespace()),
        runtime_config=DuplexSessionRuntimeConfig(context_memory=limits),
    )
    try:
        text = "杭州" * 3500
        h.session.register_history_item("old", {"role": "user", "content": text})
        before = h.session.history_snapshot()
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        submission = auxiliary_submissions(h)[0]
        assert submission.context.stage_id == submission.context.final_stage_id == 0
        current = h.session.capture_memory_context(max_chars=8192)
        assert 0 < len(current.evidence[0].text) < len(text)
        assert current.evidence[0].char_start == 0
        request_ids = submission.prompt["prompt_token_ids"]
        assert len(request_ids) <= 2048
        response = output(json.dumps(result()), omni=True)
        response.request_id = submission.context.request_id
        assert h.deliver(response, stage_id=0, segment_finished=True)
        await wait_until(lambda: bool(h.port.cleanups))
        await wait_until(lambda: len(auxiliary_submissions(h)) == 2)
        next_capture = h.session.capture_memory_context(max_chars=8192)
        assert next_capture.evidence[0].char_start == current.evidence[0].char_end
        assert h.session.history_snapshot() == before
        assert not h.session.context_ledger.card_candidates()
        assert (
            h.session.context_compute_budget(plugin.context_auxiliary_policy(runtime_config={}).compute_budget).usage(
                now=h.manager._clock()
            )[1]
            == 1024
        )
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_disabled_runner_does_not_capture_history(qwen_processor, monkeypatch):
    from tests.engine.duplex.test_session_auxiliary import auxiliary_submissions
    from tests.engine.duplex.test_session_runner import close_harness, open_harness
    from vllm_omni.engine.duplex.commands import Heartbeat

    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = qwen_processor
    h = await open_harness(auto_response=False, plugin=plugin, model="Qwen/test", stage_count=3)
    try:
        h.session.register_history_item("old", {"role": "user", "content": "I live in Hangzhou."})

        def forbidden_capture(*args, **kwargs):
            pytest.fail("disabled memory captured history")

        monkeypatch.setattr(h.session, "capture_memory_context", forbidden_capture)
        await h.run(Heartbeat())
        assert not auxiliary_submissions(h)
        assert h.session._memory_context is None and h.session._memory_compute_budget is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("response_kind", ["accepted", "malformed", "stale", "timeout"])
async def test_live_qwen_hooks_preserve_private_output_and_admission_fences(qwen_processor, response_kind):
    from tests.engine.duplex.test_session_auxiliary import auxiliary_submissions, wait_until
    from tests.engine.duplex.test_session_runner import close_harness, open_harness
    from vllm_omni.engine.duplex.commands import Heartbeat
    from vllm_omni.engine.duplex.contracts import duplex_is_auxiliary_request
    from vllm_omni.engine.duplex.events import SessionCreated, SessionHeartbeatAck, SessionUpdated

    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    plugin.processor = qwen_processor
    limits = settings(
        idle_s=0, cooldown_s=0, compute_output_tokens=512, timeout_s=0.1 if response_kind == "timeout" else 2
    )
    h = await open_harness(
        auto_response=False,
        stage_count=3,
        plugin=plugin,
        model="Qwen/test",
        model_config=SimpleNamespace(max_model_len=4096, hf_config=SimpleNamespace()),
        runtime_config=DuplexSessionRuntimeConfig(context_memory=limits),
    )
    try:
        h.session.register_history_item("old", {"role": "user", "content": "I live in Hangzhou."})
        h.submit(Heartbeat())
        await wait_until(lambda: len(auxiliary_submissions(h)) == 1)
        submission = auxiliary_submissions(h)[0]
        if response_kind == "stale":
            h.session.register_history_item("old", {"role": "user", "content": "I live in Shanghai."})
        before = h.session.history_snapshot()
        if response_kind != "timeout":
            response = output("{}" if response_kind == "malformed" else json.dumps(result(upsert())), omni=True)
            response.request_id = submission.context.request_id
            assert h.deliver(response, stage_id=0, segment_finished=True)
        await wait_until(lambda: bool(h.port.cleanups))
        await h.settle()
        candidates = h.session.context_ledger.card_candidates()
        assert len(candidates) == (1 if response_kind == "accepted" else 0)
        if candidates:
            assert candidates[0].message["content"] == "The user lives in Hangzhou."
        assert h.session.history_snapshot() == before
        assert not h.manager._auxiliary_request_permits
        assert not any(duplex_is_auxiliary_request(rid) for _, rid in h.session.request_resources)
        assert len(auxiliary_submissions(h)) == 1
        assert all(isinstance(event, (SessionCreated, SessionHeartbeatAck, SessionUpdated)) for event in h.events)
        policy = plugin.context_auxiliary_policy(runtime_config={})
        assert h.session.context_compute_budget(policy.compute_budget).usage(now=h.manager._clock()) == (
            len(submission.prompt["prompt_token_ids"]),
            512,
        )
    finally:
        await close_harness(h)
