# SPDX-License-Identifier: Apache-2.0
"""Tests for Raon serving, prompts, and tokenizer helpers."""

from __future__ import annotations

import asyncio
import dataclasses as _dataclasses
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.model_executor.models.raon import serving_utils as _su
from vllm_omni.model_executor.models.raon.serving_utils import (
    _GLOBAL_TOP_K,
    RaonServingHooks,
    prepare_raon_tts_sampling_params,
)
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.tokenizers.raon_tokenizer import (
    AUDIO_END,
    AUDIO_INPUT_PLACEHOLDER,
    AUDIO_OUTPUT_END_PAD,
    AUDIO_OUTPUT_OPEN_SEQ,
    AUDIO_OUTPUT_PAD,
    AUDIO_OUTPUT_PAD_TOKEN,
    AUDIO_OUTPUT_PLACEHOLDER,
    AUDIO_PLACEHOLDER_SEQ,
    AUDIO_START,
    AUDIO_START_TOKEN,
    IM_END,
    IM_START,
    LEGACY_AUDIO_PAD_TOKEN,
    LEGACY_AUDIO_PLACEHOLDER_SEQ,
    SPEAKER_EMBEDDING_PLACEHOLDER,
    OutputMode,
    RaonChatTemplateBuilder,
    RaonResolvedIds,
    TaskType,
    count_audio_placeholders_str,
    inject_placeholders_into_str,
    inject_placeholders_into_token_ids,
    normalize_placeholders_str,
    normalize_token_ids,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def _default_raon_ids() -> RaonResolvedIds:
    return RaonResolvedIds(
        audio_start=AUDIO_START.id,
        audio_end=AUDIO_END.id,
        audio_input_placeholder=AUDIO_INPUT_PLACEHOLDER.id,
        audio_output_placeholder=AUDIO_OUTPUT_PLACEHOLDER.id,
        speaker_placeholder=SPEAKER_EMBEDDING_PLACEHOLDER.id,
        audio_output_pad=AUDIO_OUTPUT_PAD.id,
        audio_output_end_pad=AUDIO_OUTPUT_END_PAD.id,
    )


def _stub_tokenizer(mapping: dict[str, int]) -> MagicMock:
    tok = MagicMock()
    tok.convert_tokens_to_ids.side_effect = lambda text: mapping.get(text, tok.unk_token_id)
    tok.unk_token_id = 0
    return tok


# ===================================================================
# Chat Template Builder: stubs & helpers
# ===================================================================


class _StubChatTokenizer:
    """Minimal tokenizer for chat template builder tests."""

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"{IM_START.text}{role}\n{content}{IM_END.text}\n")
        if add_generation_prompt:
            parts.append(f"{IM_START.text}assistant\n")
        return "".join(parts)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        token_map = {
            IM_START.text: IM_START.id,
            IM_END.text: IM_END.id,
            AUDIO_START.text: AUDIO_START.id,
            AUDIO_END.text: AUDIO_END.id,
            AUDIO_INPUT_PLACEHOLDER.text: AUDIO_INPUT_PLACEHOLDER.id,
            AUDIO_OUTPUT_PLACEHOLDER.text: AUDIO_OUTPUT_PLACEHOLDER.id,
            SPEAKER_EMBEDDING_PLACEHOLDER.text: SPEAKER_EMBEDDING_PLACEHOLDER.id,
            LEGACY_AUDIO_PAD_TOKEN: 151673,
        }
        ids: list[int] = []
        remaining = text
        while remaining:
            matched = False
            for token_text, token_id in sorted(token_map.items(), key=lambda x: -len(x[0])):
                if remaining.startswith(token_text):
                    ids.append(token_id)
                    remaining = remaining[len(token_text) :]
                    matched = True
                    break
            if not matched:
                ids.append(ord(remaining[0]))
                remaining = remaining[1:]
        return ids


def _builder(tokenizer=None) -> RaonChatTemplateBuilder:
    return RaonChatTemplateBuilder(tokenizer=tokenizer or _StubChatTokenizer())


def _assert_chat_frame(prompt: str) -> None:
    assert f"{IM_START.text}user" in prompt
    assert IM_END.text in prompt
    assert f"{IM_START.text}assistant\n" in prompt


def test_prepare_raon_tts_sampling_params_copies_and_overrides_request_fields():
    base_params = [SimpleNamespace(max_tokens=64, temperature=0.1, seed=None)]
    request = SimpleNamespace(max_new_tokens=512, temperature=0.7, seed=1234)

    class _FakeHooks:
        def apply_task_sampling_params(self, params, *, task, request=None):
            assert task == "tts"
            params.temperature = request.temperature
            params.seed = request.seed

    serving = SimpleNamespace(_raon_hooks=_FakeHooks())

    prepared = prepare_raon_tts_sampling_params(serving, base_params, request)

    assert prepared is not base_params
    assert prepared[0] is not base_params[0]
    assert base_params[0].max_tokens == 64
    assert prepared[0].max_tokens == 512
    assert prepared[0].temperature == 0.7
    assert prepared[0].seed == 1234


PH = AUDIO_PLACEHOLDER_SEQ

_AUDIO_INPUT_TASKS = {TaskType.STT, TaskType.SPOKEN_QA, TaskType.SPEECH_QA}
_TEXT_CONTENT_TASKS = {TaskType.TEXT_QA, TaskType.TTS, TaskType.SPEECH_QA}


def _kwargs_for_task(task: TaskType) -> dict:
    kwargs: dict = {"task": task}
    if task in _AUDIO_INPUT_TASKS:
        kwargs["audio_count"] = 1
    if task in _TEXT_CONTENT_TASKS or task.value == "tts_icl":
        kwargs["user_content"] = "test"
    return kwargs


# ===================================================================
# Single-turn tasks — parametrized
# ===================================================================


@pytest.mark.parametrize(
    "task,user_content,audio_count,expected_mode,expect_audio_input,expect_ph",
    [
        (TaskType.TEXT_QA, "Capital of France?", 0, OutputMode.TEXT_ONLY, False, False),
        (TaskType.STT, None, 1, OutputMode.TEXT_ONLY, True, True),
        (TaskType.TTS, "Hello world", 0, OutputMode.AUDIO_ONLY, False, False),
        (TaskType.SPOKEN_QA, None, 1, OutputMode.TEXT_ONLY, True, True),
        (TaskType.SPEECH_QA, "What is being said?", 1, OutputMode.TEXT_ONLY, True, True),
    ],
    ids=["text_qa", "stt", "tts", "spoken_qa", "speech_qa"],
)
def test_single_turn_task_basics(task, user_content, audio_count, expected_mode, expect_audio_input, expect_ph):
    kwargs: dict = {"task": task}
    if user_content:
        kwargs["user_content"] = user_content
    if audio_count:
        kwargs["audio_count"] = audio_count
    result = _builder().build_prompt(**kwargs)

    _assert_chat_frame(result.prompt_text)
    assert result.output_mode == expected_mode
    assert result.has_audio_input is expect_audio_input
    if expect_ph:
        assert PH in result.prompt_text
    else:
        assert PH not in result.prompt_text


# ===================================================================
# Prompt builder canonical behavior
# ===================================================================


def test_tts_prompt_layout_with_speaker_and_audio_start():
    spk = SPEAKER_EMBEDDING_PLACEHOLDER.text
    result = _builder().build_prompt(
        task=TaskType.TTS,
        user_content="Hello world",
        prepend_speaker_token=True,
        append_audio_start=True,
    )
    expected_user_content = f"{spk}Speak the following text:\nHello world"
    expected = (
        f"{IM_START.text}user\n{expected_user_content}{IM_END.text}\n{IM_START.text}assistant\n{AUDIO_START.text}"
    )
    assert result.prompt_text == expected


def test_multi_turn_audio_placeholders_are_preserved_in_order():
    result = _builder().build_prompt(
        task=TaskType.SPEECH_QA,
        messages=[
            {"role": "user", "content": "My color is blue."},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "First clip", "audio_count": 1},
            {"role": "assistant", "content": "Heard it."},
            {"role": "user", "content": "Second clip", "audio_count": 1},
        ],
        system_prompt="You are a translator.",
    )

    _assert_chat_frame(result.prompt_text)
    assert "You are a translator." in result.prompt_text
    assert result.prompt_text.count(PH) == 2
    assert result.has_audio_input is True
    assert result.audio_count == 2


# ===================================================================
# Legacy token normalisation
# ===================================================================


class TestLegacyTokenNormalisation:
    def test_normalize_legacy_placeholder_str(self):
        legacy = f"prefix{LEGACY_AUDIO_PLACEHOLDER_SEQ}suffix"
        normalized = normalize_placeholders_str(legacy)
        assert PH in normalized
        assert LEGACY_AUDIO_PAD_TOKEN not in normalized

    def test_count_includes_legacy(self):
        text = f"{PH}{LEGACY_AUDIO_PLACEHOLDER_SEQ}"
        assert count_audio_placeholders_str(text) == 2

    def test_count_no_double_counting(self):
        assert count_audio_placeholders_str(f"{PH}{PH}") == 2

    def test_legacy_in_multi_turn_normalised(self):
        legacy_content = f"{LEGACY_AUDIO_PLACEHOLDER_SEQ}Listen to this."
        result = _builder().build_prompt(
            task=TaskType.SPEECH_QA,
            messages=[{"role": "user", "content": legacy_content, "audio_count": 1}],
        )
        assert LEGACY_AUDIO_PAD_TOKEN not in result.prompt_text
        assert result.prompt_text.count(PH) == 1


# ===================================================================
# Placeholder injection: str path
# ===================================================================


class TestInjectPlaceholdersStr:
    def test_inject_into_empty(self):
        result = inject_placeholders_into_str("hello", num_audios=1)
        assert result.count(PH) == 1

    def test_no_injection_when_enough(self):
        assert inject_placeholders_into_str(f"{PH}hello", num_audios=1).count(PH) == 1

    def test_inject_distributes_across_turns(self):
        prompt = (
            f"{IM_START.text}user\nTurn1{IM_END.text}\n"
            f"{IM_START.text}assistant\nOK{IM_END.text}\n"
            f"{IM_START.text}user\nTurn2{IM_END.text}\n"
        )
        assert inject_placeholders_into_str(prompt, num_audios=2).count(PH) == 2

    def test_inject_zero_is_noop(self):
        text = "hello"
        assert inject_placeholders_into_str(text, num_audios=0) == text

    def test_legacy_counted_and_normalised(self):
        text = f"{LEGACY_AUDIO_PLACEHOLDER_SEQ}hello"
        result = inject_placeholders_into_str(text, num_audios=1)
        assert result.count(PH) == 1
        assert LEGACY_AUDIO_PAD_TOKEN not in result


# ===================================================================
# Placeholder injection: token ID path
# ===================================================================


class TestInjectPlaceholdersTokenIds:
    PH_IDS = [AUDIO_START.id, AUDIO_INPUT_PLACEHOLDER.id, AUDIO_END.id]
    LEGACY_PH_IDS = [AUDIO_START.id, 151673, AUDIO_END.id]
    MARKER_IDS = [IM_START.id, ord("u"), ord("s"), ord("e"), ord("r"), ord("\n")]

    def test_inject_when_missing(self):
        result = inject_placeholders_into_token_ids(
            [1, 2, 3],
            num_audios=1,
            ph_ids=self.PH_IDS,
        )
        assert self.PH_IDS[0] in result

    def test_no_inject_when_present(self):
        ids = self.PH_IDS + [1, 2, 3]
        assert inject_placeholders_into_token_ids(ids, num_audios=1, ph_ids=self.PH_IDS) == ids

    def test_counts_legacy(self):
        ids = self.LEGACY_PH_IDS + [1, 2, 3]
        result = inject_placeholders_into_token_ids(
            ids,
            num_audios=1,
            ph_ids=self.PH_IDS,
            legacy_ph_ids=self.LEGACY_PH_IDS,
        )
        assert result == ids

    def test_inject_into_correct_turn(self):
        ids = self.MARKER_IDS + [ord("H"), ord("i")] + [IM_END.id] + self.MARKER_IDS + [ord("Q")]
        result = inject_placeholders_into_token_ids(
            ids,
            num_audios=1,
            ph_ids=self.PH_IDS,
            marker_ids=self.MARKER_IDS,
        )
        assert result.count(AUDIO_START.id) == 1

    def test_zero_is_noop(self):
        ids = [1, 2, 3]
        assert inject_placeholders_into_token_ids(ids, num_audios=0, ph_ids=self.PH_IDS) == ids


# ===================================================================
# normalize_token_ids
# ===================================================================


class TestNormalizeTokenIds:
    def test_replaces_output_with_input(self, _default_raon_ids):
        ids = [1, AUDIO_OUTPUT_PLACEHOLDER.id, 3]
        result = normalize_token_ids(ids, special=_default_raon_ids)
        assert AUDIO_OUTPUT_PLACEHOLDER.id not in result
        assert AUDIO_INPUT_PLACEHOLDER.id in result

    def test_ensures_placeholder_subsequence(self, _default_raon_ids):
        ids = [1, AUDIO_INPUT_PLACEHOLDER.id, 3]
        result = normalize_token_ids(ids, special=_default_raon_ids)
        assert AUDIO_START.id in result and AUDIO_END.id in result

    def test_no_change_when_already_correct(self, _default_raon_ids):
        ids = [1, AUDIO_START.id, AUDIO_INPUT_PLACEHOLDER.id, AUDIO_END.id, 3]
        assert normalize_token_ids(ids, special=_default_raon_ids) == ids

    def test_no_audio_tokens_passthrough(self, _default_raon_ids):
        assert normalize_token_ids([1, 2, 3], special=_default_raon_ids) == [1, 2, 3]


class TestResolveRaonSpecialIds:
    def test_default_checkpoint_matches_hardcoded(self):
        from vllm_omni.tokenizers.raon_tokenizer import resolve_raon_special_ids

        ids = resolve_raon_special_ids(
            _stub_tokenizer(
                {
                    "<|audio_start|>": 151669,
                    "<|audio_end|>": 151670,
                    "<|audio_input_placeholder|>": 151676,
                    "<|audio_output_placeholder|>": 151675,
                    "<|speaker_embedding_placeholder|>": 151671,
                    "<|audio_output_pad|>": 151677,
                    "<|audio_output_end_pad|>": 151678,
                }
            )
        )

        assert ids.audio_start == 151669
        assert ids.audio_end == 151670
        assert ids.audio_input_placeholder == 151676
        assert ids.audio_output_placeholder == 151675

    def test_drifted_ids_are_respected(self):
        from vllm_omni.tokenizers.raon_tokenizer import resolve_raon_special_ids

        ids = resolve_raon_special_ids(
            _stub_tokenizer(
                {
                    "<|audio_start|>": 200001,
                    "<|audio_end|>": 200002,
                    "<|audio_input_placeholder|>": 200003,
                    "<|audio_output_placeholder|>": 200004,
                    "<|speaker_embedding_placeholder|>": 200005,
                    "<|audio_output_pad|>": 200006,
                    "<|audio_output_end_pad|>": 200007,
                }
            )
        )

        assert ids.audio_start == 200001
        assert ids.audio_end == 200002
        assert ids.audio_output_placeholder == 200004

    def test_missing_token_warns(self):
        from vllm_omni.tokenizers import raon_tokenizer
        from vllm_omni.tokenizers.raon_tokenizer import resolve_raon_special_ids

        records: list[logging.LogRecord] = []

        class _ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _ListHandler(level=logging.WARNING)
        target_logger = raon_tokenizer.logger
        prev_level = target_logger.level
        target_logger.addHandler(handler)
        target_logger.setLevel(logging.WARNING)
        try:
            resolve_raon_special_ids(
                _stub_tokenizer(
                    {
                        "<|audio_end|>": 151670,
                        "<|audio_input_placeholder|>": 151676,
                        "<|audio_output_placeholder|>": 151675,
                        "<|speaker_embedding_placeholder|>": 151671,
                        "<|audio_output_pad|>": 151677,
                        "<|audio_output_end_pad|>": 151678,
                    }
                )
            )
        finally:
            target_logger.removeHandler(handler)
            target_logger.setLevel(prev_level)

        assert any("audio_start" in record.getMessage() for record in records)


# ===================================================================
# Tokenizer fallback
# ===================================================================


class TestTokenizerFallback:
    def test_fallback_without_apply_chat_template(self):
        class _NoTemplate:
            pass

        result = _builder(tokenizer=_NoTemplate()).build_prompt(
            task=TaskType.TEXT_QA,
            user_content="Hello",
        )
        _assert_chat_frame(result.prompt_text)

    def test_fallback_tts(self):
        class _NoTemplate:
            pass

        result = _builder(tokenizer=_NoTemplate()).build_prompt(
            task=TaskType.TTS,
            user_content="Speak",
            append_audio_start=True,
        )
        assert f"{IM_START.text}user\n" in result.prompt_text
        assert f"{IM_START.text}assistant\n" in result.prompt_text


# ===================================================================
# Token ID encoding
# ===================================================================


class TestBuildPromptTokenIds:
    def test_stt_contains_audio_ids(self):
        b = _builder()
        result = b.build_prompt(task=TaskType.STT, audio_count=1)
        ids = b.encode_prompt(result.prompt_text)
        assert AUDIO_START.id in ids
        assert AUDIO_INPUT_PLACEHOLDER.id in ids

    def test_text_qa_no_audio_ids(self):
        b = _builder()
        result = b.build_prompt(task=TaskType.TEXT_QA, user_content="Hello")
        ids = b.encode_prompt(result.prompt_text)
        assert AUDIO_INPUT_PLACEHOLDER.id not in ids

    def test_no_output_placeholder_leaked(self):
        b = _builder()
        for task in TaskType:
            result = b.build_prompt(**_kwargs_for_task(task))
            assert AUDIO_OUTPUT_PLACEHOLDER.text not in result.prompt_text


# ===================================================================
# PromptResult metadata — parametrized
# ===================================================================


class TestPromptResultMetadata:
    @pytest.mark.parametrize(
        "task,expected_mode",
        [
            (TaskType.TEXT_QA, OutputMode.TEXT_ONLY),
            (TaskType.STT, OutputMode.TEXT_ONLY),
            (TaskType.TTS, OutputMode.AUDIO_ONLY),
            (TaskType.SPOKEN_QA, OutputMode.TEXT_ONLY),
            (TaskType.SPEECH_QA, OutputMode.TEXT_ONLY),
        ],
    )
    def test_output_mode(self, task, expected_mode):
        assert _builder().build_prompt(**_kwargs_for_task(task)).output_mode == expected_mode

    def test_audio_input_flag(self):
        b = _builder()
        for task in TaskType:
            result = b.build_prompt(**_kwargs_for_task(task))
            assert result.has_audio_input is (task in _AUDIO_INPUT_TASKS)

    def test_task_preserved(self):
        b = _builder()
        for task in TaskType:
            assert b.build_prompt(**_kwargs_for_task(task)).task == task


# ===================================================================
# E2E: ICL TTS path (kept — unique ICL scenarios)
# ===================================================================


def _build_icl_prompt(
    target_text: str,
    ref_text: str,
    prepend_speaker_token: bool = True,
) -> str:
    spk = SPEAKER_EMBEDDING_PLACEHOLDER.text
    user_content = f"Speak the following text:\n{ref_text} {target_text}"
    if prepend_speaker_token and not user_content.startswith(spk):
        user_content = f"{spk}{user_content}"

    assistant_content = f"{AUDIO_START_TOKEN}{AUDIO_OUTPUT_PAD_TOKEN}"
    prompt = f"{IM_START.text}user\n{user_content}{IM_END.text}\n{IM_START.text}assistant\n{assistant_content}"
    return prompt


class TestE2EICL:
    def test_icl_prompt_layout_and_placeholder_accounting(self):
        spk = SPEAKER_EMBEDDING_PLACEHOLDER.text
        prompt = _build_icl_prompt("Target.", "Reference.")
        expected = (
            f"{IM_START.text}user\n"
            f"{spk}Speak the following text:\n"
            f"Reference. Target.{IM_END.text}\n"
            f"{IM_START.text}assistant\n"
            f"{AUDIO_START.text}{AUDIO_OUTPUT_PLACEHOLDER.text}"
        )
        assert prompt == expected
        assert PH not in prompt
        assert count_audio_placeholders_str(prompt) == 1

    def test_icl_inject_with_legacy_and_output_placeholder(self):
        prompt = (
            f"{IM_START.text}user\n{LEGACY_AUDIO_PLACEHOLDER_SEQ}Hello{IM_END.text}\n"
            f"{IM_START.text}assistant\n{AUDIO_OUTPUT_OPEN_SEQ}"
        )
        after = inject_placeholders_into_str(prompt, num_audios=2)
        assert count_audio_placeholders_str(after) >= 2


# ===================================================================
# Serving Hooks: stubs
# ===================================================================


class _StubHfConfig:
    def __init__(
        self,
        *,
        im_end_token_id: int | None = 151645,
        audio_end_token_id: int | None = 151670,
        im_start_token_id: int | None = None,
        audio_start_token_id: int | None = None,
    ) -> None:
        self.im_end_token_id = im_end_token_id
        self.audio_end_token_id = audio_end_token_id
        self.im_start_token_id = im_start_token_id
        self.audio_start_token_id = audio_start_token_id


class _StubModelConfig:
    def __init__(self, hf_config: _StubHfConfig | None = None) -> None:
        self.hf_config = hf_config
        self.model = None
        self.trust_remote_code = False


class _StubServingTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        content = messages[0]["content"]
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def _make_hooks(
    im_end_token_id: int | None = 151645,
    audio_end_token_id: int | None = 151670,
) -> RaonServingHooks:
    hf = _StubHfConfig(
        im_end_token_id=im_end_token_id,
        audio_end_token_id=audio_end_token_id,
    )
    return RaonServingHooks(_StubModelConfig(hf))


# ===================================================================
# Serving Hooks: build_tts_prompt
# ===================================================================


def test_resolve_chat_modalities_forces_text_only_for_audio_requests():
    hooks = _make_hooks()
    request = SimpleNamespace(
        modalities=["audio"],
        _engine_output_modalities=["text", "audio"],
    )
    engine_prompt = {"prompt": "hi"}

    output = hooks.resolve_chat_modalities(request, [engine_prompt], tokenizer=None)

    assert output == ["text"]
    assert request.modalities == ["text"]
    assert engine_prompt["additional_information"]["output_mode"] == ["text_only"]
    assert "force_audio_first_token" not in engine_prompt["additional_information"]


@pytest.mark.asyncio
async def test_apply_default_modalities_wraps_raon_chat_preprocess(monkeypatch):
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    async def _orig_preprocess(self, request, *args, **kwargs):
        return [], [{"prompt": "hi"}]

    monkeypatch.setattr(OmniOpenAIServingChat, "_preprocess_chat", _orig_preprocess)
    monkeypatch.delattr(
        OmniOpenAIServingChat,
        "_raon_default_modalities_applied",
        raising=False,
    )

    RaonServingHooks.apply_default_modalities()

    request = SimpleNamespace(modalities=["audio"])
    engine_client = SimpleNamespace(
        output_modalities=["text", "audio"],
        get_tokenizer=AsyncMock(return_value=None),
    )
    serving = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="raon")),
        engine_client=engine_client,
    )

    _, engine_prompts = await OmniOpenAIServingChat._preprocess_chat(serving, request)

    assert request.modalities == ["text"]
    assert engine_prompts[0]["additional_information"]["output_mode"] == ["text_only"]
    assert "force_audio_first_token" not in engine_prompts[0]["additional_information"]
    engine_client.get_tokenizer.assert_awaited_once()


def test_raon_text_only_engine_client_forces_text_and_unwraps_request_output():
    class _FakeEngine:
        default_sampling_params_list = ["stage0-default", "stage1-default"]

        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append((args, kwargs))

            async def _gen():
                yield OmniRequestOutput(
                    request_id="req-text",
                    final_output_type="text",
                    request_output="request-output",
                )

            return _gen()

    engine = _FakeEngine()
    adapter = _su._RaonTextOnlyEngineClient(engine)

    async def _collect_outputs():
        return [output async for output in adapter.generate("prompt", "sampling", "req-text")]

    outputs = asyncio.run(_collect_outputs())

    assert outputs == ["request-output"]
    assert engine.calls[0][1]["output_modalities"] == ["text"]
    assert engine.calls[0][1]["sampling_params_list"] == ["sampling", "stage1-default"]


class TestBuildTtsPrompt:
    @pytest.mark.asyncio
    async def test_raw_style_no_speaker_token(self):
        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Hello world")
        assert "Hello world" in result
        assert "<|im_start|>user" in result

    @pytest.mark.asyncio
    async def test_instruction_style_via_env(self, monkeypatch):
        monkeypatch.setenv("RAON_TTS_PROMPT_STYLE", "instruction")
        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Hello world")
        assert "Speak the following text:" in result

    @pytest.mark.asyncio
    async def test_raw_style_is_default(self, monkeypatch):
        monkeypatch.delenv("RAON_TTS_PROMPT_STYLE", raising=False)
        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Hello world")
        assert "Speak the following text:" not in result

    @pytest.mark.asyncio
    async def test_prepend_speaker_token_adds_prefix(self):
        from vllm_omni.tokenizers.raon_tokenizer import SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN

        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Hi", prepend_speaker_token=True)
        assert SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN in result

    @pytest.mark.asyncio
    async def test_prepend_speaker_token_not_duplicated(self):
        from vllm_omni.tokenizers.raon_tokenizer import SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN

        hooks = _make_hooks()
        text_with_token = f"{SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN}Hi"
        result = await hooks.build_tts_prompt(text_with_token, prepend_speaker_token=True)
        assert result.count(SPEAKER_EMBEDDING_PLACEHOLDER_TOKEN) == 1

    @pytest.mark.asyncio
    async def test_tokenizer_apply_chat_template_used_when_available(self):
        class _FakeEngineClient:
            async def get_tokenizer(self):
                return _StubServingTokenizer()

        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Test text", engine_client=_FakeEngineClient())
        assert "Test text" in result

    @pytest.mark.asyncio
    async def test_fallback_template_when_no_engine_client(self):
        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Fallback text")
        assert result == "<|im_start|>user\nFallback text<|im_end|>\n<|im_start|>assistant\n"

    @pytest.mark.asyncio
    async def test_engine_client_exception_falls_back_gracefully(self):
        class _BrokenEngineClient:
            async def get_tokenizer(self):
                raise RuntimeError("tokenizer unavailable")

        hooks = _make_hooks()
        result = await hooks.build_tts_prompt("Graceful", engine_client=_BrokenEngineClient())
        assert "Graceful" in result


# ===================================================================
# Serving Hooks: estimate_tts_min_tokens — parametrized
# ===================================================================


class TestEstimateTtsMinTokens:
    @pytest.mark.parametrize(
        "text,max_tokens,expected",
        [
            ("", None, 6),
            ("hi", None, 6),
            ("one two three four five six seven eight nine ten", None, 20),
            ("one two three four five six seven eight nine ten", 10, 9),
            ("hello world", 1, 0),
            ("", 0, 0),
            ("  hello   world  ", None, 4),
        ],
        ids=["empty", "short", "longer-text", "max-tokens-cap", "very-small-max", "zero-max", "extra-whitespace"],
    )
    def test_estimation(self, text, max_tokens, expected):
        hooks = _make_hooks()
        result = hooks.estimate_tts_min_tokens(text, max_tokens=max_tokens)
        if text == "  hello   world  ":
            clean = hooks.estimate_tts_min_tokens("hello world", max_tokens=None)
            assert result == clean
        else:
            assert result == expected

    def test_result_never_negative(self):
        assert _make_hooks().estimate_tts_min_tokens("", max_tokens=0) >= 0


# ===================================================================
# Serving Hooks: estimate_tts_max_tokens — parametrized
# ===================================================================


class TestEstimateTtsMaxTokens:
    @pytest.mark.parametrize(
        "text,hard_cap,expected",
        [
            ("", None, 48),
            ("hi", None, 48),
            ("one two three four", None, 64),
            (" ".join(["word"] * 100), None, 832),
            ("one two three four", 50, 50),
        ],
        ids=["empty", "short", "medium", "100-words", "custom-hard-cap"],
    )
    def test_estimation(self, text, hard_cap, expected):
        hooks = _make_hooks()
        kwargs = {"hard_cap": hard_cap} if hard_cap else {}
        assert hooks.estimate_tts_max_tokens(text, **kwargs) == expected

    def test_env_override_respected(self, monkeypatch):
        monkeypatch.setenv("RAON_TTS_FIXED_MAX_TOKENS", "200")
        assert _make_hooks().estimate_tts_max_tokens("some text") == 200

    def test_env_override_clamped_to_hard_cap(self, monkeypatch):
        monkeypatch.setenv("RAON_TTS_FIXED_MAX_TOKENS", "9999")
        assert _make_hooks().estimate_tts_max_tokens("some text", hard_cap=300) == 300

    @pytest.mark.parametrize("env_val", ["not_a_number", ""])
    def test_env_override_invalid_falls_back(self, monkeypatch, env_val):
        monkeypatch.setenv("RAON_TTS_FIXED_MAX_TOKENS", env_val)
        assert _make_hooks().estimate_tts_max_tokens("one two three four") == 64

    def test_result_at_least_one(self):
        assert _make_hooks().estimate_tts_max_tokens("", hard_cap=1) >= 1

    def test_env_override_at_least_one(self, monkeypatch):
        monkeypatch.setenv("RAON_TTS_FIXED_MAX_TOKENS", "0")
        assert _make_hooks().estimate_tts_max_tokens("text") >= 1


# ===================================================================
# Serving Hooks: apply_sampling_parity
# ===================================================================


class TestApplySamplingParity:
    def _make_params(self, **kwargs) -> dict:
        base = {
            "stop": ["<extra_stop>"],
            "stop_token_ids": [9999],
            "ignore_eos": True,
            "min_tokens": 0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 0,
            "repetition_penalty": 1.0,
        }
        base.update(kwargs)
        return base

    def test_empty_list_is_noop(self):
        _make_hooks().apply_sampling_parity([], comprehension_stage_index=0)

    def test_applies_to_correct_stage_index(self):
        params_0 = self._make_params()
        params_1 = self._make_params()
        _make_hooks().apply_sampling_parity([params_0, params_1], comprehension_stage_index=1)
        assert params_1["temperature"] == 1.2
        assert params_0["temperature"] == 1.0

    def test_out_of_bounds_index_clamps_to_last(self):
        params_0 = self._make_params()
        params_1 = self._make_params()
        _make_hooks().apply_sampling_parity([params_0, params_1], comprehension_stage_index=99)
        assert params_1["temperature"] == 1.2

    def test_negative_index_clamps_to_first(self):
        params_0 = self._make_params()
        _make_hooks().apply_sampling_parity([params_0], comprehension_stage_index=-5)
        assert params_0["temperature"] == 1.2

    def test_audio_stop_and_eos_set(self):
        params = self._make_params()
        _make_hooks().apply_sampling_parity([params], comprehension_stage_index=0)
        assert params["stop"] == []
        assert 151645 in params["stop_token_ids"]
        assert 151670 in params["stop_token_ids"]
        assert params["ignore_eos"] is False
        assert params["min_tokens"] >= 1

    def test_tts_defaults_applied(self):
        params = self._make_params()
        _make_hooks().apply_sampling_parity([params], comprehension_stage_index=0)
        assert params["temperature"] == 1.2
        assert params["top_k"] == _GLOBAL_TOP_K

    def test_client_explicit_temperature_respected(self):
        params = self._make_params(temperature=0.9)
        _make_hooks().apply_sampling_parity([params], comprehension_stage_index=0)
        assert params["temperature"] == 0.9


# ===================================================================
# Serving Hooks: stop token ID resolution
# ===================================================================


class TestStopTokenIdResolution:
    def test_explicit_im_end_and_audio_end_used(self):
        hooks = _make_hooks(im_end_token_id=100, audio_end_token_id=200)
        assert 100 in hooks._audio_stop_token_ids
        assert 200 in hooks._audio_stop_token_ids

    def test_im_end_derived_from_im_start_plus_one(self):
        hf = _StubHfConfig(
            im_end_token_id=None,
            audio_end_token_id=200,
            im_start_token_id=99,
        )
        hooks = RaonServingHooks(_StubModelConfig(hf))
        assert 100 in hooks._audio_stop_token_ids

    def test_audio_end_derived_from_audio_start_plus_one(self):
        hf = _StubHfConfig(
            im_end_token_id=100,
            audio_end_token_id=None,
            audio_start_token_id=169,
        )
        hooks = RaonServingHooks(_StubModelConfig(hf))
        assert 170 in hooks._audio_stop_token_ids

    def test_no_hf_config_results_in_empty_stop_ids(self):
        hooks = RaonServingHooks(_StubModelConfig(hf_config=None))
        assert hooks._audio_stop_token_ids == []


# ===================================================================
# Long TTS helpers
# ===================================================================


def _build_rolling_icl_server() -> OmniOpenAIServingSpeech:
    server = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    server._tts_model_type = "raon"
    server._diffusion_mode = False
    server.model_config = MagicMock()
    return server


def _replace_raon_env(monkeypatch: pytest.MonkeyPatch, **overrides) -> None:
    from vllm_omni.transformers_utils.configs import raon as _raon_cfg

    monkeypatch.setattr(_raon_cfg, "ENV", _dataclasses.replace(_raon_cfg.ENV, **overrides))


def _capture_chunk_reqs(server, mocker: MockerFixture, *, sr: int = 24000):
    captured: list[dict] = []
    # Keep each synthetic chunk long enough for the rolling ref-audio chain.
    pcm = np.zeros(sr, dtype=np.float32)

    async def _fake_collect(serving, chunk_req):
        captured.append(
            {
                "input": chunk_req.input,
                "task_type": chunk_req.task_type,
                "voice": chunk_req.voice,
                "ref_audio": chunk_req.ref_audio,
                "ref_text": chunk_req.ref_text,
                "speaker_embedding": getattr(chunk_req, "speaker_embedding", None),
                "x_vector_only_mode": getattr(chunk_req, "x_vector_only_mode", False),
                "_speaker_anchor_ref_audio": getattr(chunk_req, "_speaker_anchor_ref_audio", None),
            }
        )
        return pcm, sr

    mocker.patch.object(_su, "collect_request_pcm", _fake_collect)
    return captured


def _long_rolling_text(n_sentences: int = 5) -> str:
    filler = " ".join(["word"] * 20)
    return " ".join([f"Sentence number {i}: {filler}." for i in range(n_sentences)])


def test_rolling_icl_gating_short_text_skips_orchestrator():
    assert _su.should_use_rolling_icl("hello world", mode="rolling_icl", threshold=90) is False


def test_rolling_icl_gating_long_text_activates_orchestrator():
    long_text = " ".join(["word"] * 100)
    assert _su.should_use_rolling_icl(long_text, mode="rolling_icl", threshold=90) is True


def test_rolling_icl_preserves_voice_anchor(mocker: MockerFixture, monkeypatch):
    server = _build_rolling_icl_server()
    _replace_raon_env(
        monkeypatch,
        tts_long_keep_original_speaker_anchor=True,
        tts_long_anchor_reset_every_chunks=0,
        tts_long_max_sentences_per_chunk=1,
        tts_long_min_ref_audio_s=0.0,
    )
    captured = _capture_chunk_reqs(server, mocker)

    request = OpenAICreateSpeechRequest(
        input=_long_rolling_text(3),
        model="raon",
        voice="vivian",
        response_format="wav",
    )
    pcm, sr = asyncio.run(_su.generate_raon_long_tts_rolling_icl(server, request))
    assert sr == 24000
    assert pcm.size > 0
    assert len(captured) >= 2
    assert captured[0]["voice"] == "vivian"
    for idx, chunk in enumerate(captured[1:], start=1):
        assert chunk["voice"] == "vivian", f"chunk {idx}: voice drifted to {chunk['voice']!r}"
        assert chunk["_speaker_anchor_ref_audio"] is None


def test_rolling_icl_preserves_embedding_anchor(mocker: MockerFixture, monkeypatch):
    server = _build_rolling_icl_server()
    _replace_raon_env(
        monkeypatch,
        tts_long_keep_original_speaker_anchor=True,
        tts_long_anchor_reset_every_chunks=0,
        tts_long_max_sentences_per_chunk=1,
        tts_long_min_ref_audio_s=0.0,
    )
    captured = _capture_chunk_reqs(server, mocker)

    emb = [0.125] * 192
    request = OpenAICreateSpeechRequest(
        input=_long_rolling_text(3),
        model="raon",
        task_type="Base",
        speaker_embedding=emb,
        response_format="wav",
    )
    asyncio.run(_su.generate_raon_long_tts_rolling_icl(server, request))
    assert len(captured) >= 2
    for idx, chunk in enumerate(captured[1:], start=1):
        assert chunk["speaker_embedding"] == emb, f"chunk {idx}: speaker_embedding was altered or cleared"
        assert chunk["_speaker_anchor_ref_audio"] is None


def test_rolling_icl_preserves_ref_audio_anchor(mocker: MockerFixture, monkeypatch):
    server = _build_rolling_icl_server()
    _replace_raon_env(
        monkeypatch,
        tts_long_keep_original_speaker_anchor=True,
        tts_long_anchor_reset_every_chunks=0,
        tts_long_max_sentences_per_chunk=1,
        tts_long_min_ref_audio_s=0.0,
    )

    async def _fake_resolve(self, ref_audio_str):
        return np.zeros(24000, dtype=np.float32).tolist(), 24000

    mocker.patch.object(OmniOpenAIServingSpeech, "_resolve_ref_audio", _fake_resolve)
    captured = _capture_chunk_reqs(server, mocker)

    original_ref = "https://example.invalid/ref.wav"
    request = OpenAICreateSpeechRequest(
        input=_long_rolling_text(3),
        model="raon",
        task_type="Base",
        ref_audio=original_ref,
        ref_text="some reference transcript text",
        response_format="wav",
    )
    asyncio.run(_su.generate_raon_long_tts_rolling_icl(server, request))

    assert len(captured) >= 2
    assert captured[0]["ref_audio"] == original_ref
    for idx, chunk in enumerate(captured[1:], start=1):
        assert chunk["ref_audio"] != original_ref, f"chunk {idx}: ref_audio was not rewritten to prev_pcm"
        assert chunk["ref_audio"].startswith("data:"), f"chunk {idx}: expected PCM data URL"
        assert chunk["_speaker_anchor_ref_audio"] == original_ref


def test_rolling_icl_nonzero_silence_frames(mocker: MockerFixture, monkeypatch):
    server = _build_rolling_icl_server()
    _replace_raon_env(monkeypatch, continuation_silence_frames=2)
    server.engine_client = MagicMock()
    server.engine_client.get_tokenizer = AsyncMock(return_value=None)
    server.uploaded_speakers = {}
    server.model_config = MagicMock()

    async def _fake_build_icl_tts_prompt(self, **kwargs):
        return "stub-icl-prompt"

    mocker.patch.object(_su.RaonServingHooks, "_build_icl_tts_prompt", _fake_build_icl_tts_prompt)

    async def _fail_resolve(self, ref_audio_str):
        raise RuntimeError("force skip multimodal branch")

    mocker.patch.object(OmniOpenAIServingSpeech, "_resolve_ref_audio", _fail_resolve)
    mocker.patch.object(_su.RaonServingHooks, "estimate_tts_max_tokens", return_value=512)
    fake_hooks = _su.RaonServingHooks.__new__(_su.RaonServingHooks)
    fake_hooks._hf_config = MagicMock()
    server.__dict__["_raon_hooks"] = fake_hooks

    chunk_req = OpenAICreateSpeechRequest(
        input="second chunk target text",
        model="raon",
        task_type="Base",
        ref_audio="data:audio/wav;base64,UklGRgAAAABXQVZFZm10IBAA",
        ref_text="previous chunk text as reference",
        response_format="wav",
    )
    prompt = asyncio.run(_su.build_raon_speech_prompt(server, chunk_req))
    add = prompt["additional_information"]
    assert add["continuation_silence_frames"] == [2]
    assert add["continuation_silence_frames"] != [0]


def test_uploaded_audio_voice_uses_reencoded_wav_data_url():
    wav_data_url = "data:audio/wav;base64,UklGRgAAAAA="

    class _FakeServing:
        def __init__(self):
            self.engine_client = SimpleNamespace()
            self.audio_data_calls = []
            self.uploaded_speakers = {
                "custom_voice": {
                    "embedding_source": "audio",
                    "file_path": "/tmp/custom_voice.safetensors",
                    "mime_type": "audio/wav",
                }
            }
            self._raon_hooks = _FakeHooks()

        def _get_uploaded_audio_data(self, voice):
            self.audio_data_calls.append(voice)
            return wav_data_url

    class _FakeHooks:
        @staticmethod
        def is_icl_request(request):
            return False

        async def build_tts_prompt(self, *args, **kwargs):
            return "stub-tts-prompt"

        def estimate_tts_max_tokens(self, text):
            return 512

    serving = _FakeServing()
    request = SimpleNamespace(
        input="Hello from the uploaded voice.",
        voice="custom_voice",
        ref_audio=None,
        speaker_embedding=None,
        max_new_tokens=None,
    )

    prompt = asyncio.run(_su.build_raon_speech_prompt(serving, request))

    assert serving.audio_data_calls == ["custom_voice"]
    assert prompt["additional_information"]["speaker_ref_audio"] == [wav_data_url]


def test_rolling_icl_prompt_preserves_explicit_chunk_budget(mocker: MockerFixture):
    server = _build_rolling_icl_server()
    server.engine_client = MagicMock()
    server.engine_client.get_tokenizer = AsyncMock(return_value=None)
    server.uploaded_speakers = {}
    server.model_config = MagicMock()

    class _FakeHooks:
        @staticmethod
        def is_icl_request(request):
            return True

        async def _build_icl_tts_prompt(self, **kwargs):
            return "stub-icl-prompt"

        def estimate_tts_max_tokens(self, text):
            return 512

    async def _fail_resolve(self, ref_audio_str):
        raise RuntimeError("force skip multimodal branch")

    mocker.patch.object(OmniOpenAIServingSpeech, "_resolve_ref_audio", _fail_resolve)
    server.__dict__["_raon_hooks"] = _FakeHooks()

    chunk_req = OpenAICreateSpeechRequest(
        input="second chunk target text",
        model="raon",
        task_type="Base",
        ref_audio="data:audio/wav;base64,UklGRgAAAABXQVZFZm10IBAA",
        ref_text="previous chunk text as reference",
        response_format="wav",
        max_new_tokens=2048,
    )
    object.__setattr__(chunk_req, "_rolling_plan_budget_explicit", True)

    asyncio.run(_su.build_raon_speech_prompt(server, chunk_req))
    assert chunk_req.max_new_tokens == 2048


def test_rolling_icl_prompt_attaches_final_audio_min_steps(mocker: MockerFixture):
    server = _build_rolling_icl_server()
    server.engine_client = MagicMock()
    server.engine_client.get_tokenizer = AsyncMock(return_value=None)
    server.uploaded_speakers = {}
    server.model_config = MagicMock()

    class _FakeHooks:
        @staticmethod
        def is_icl_request(request):
            return True

        async def _build_icl_tts_prompt(self, **kwargs):
            return "stub-icl-prompt"

        def estimate_tts_max_tokens(self, text):
            return 512

    async def _fail_resolve(self, ref_audio_str):
        raise RuntimeError("force skip multimodal branch")

    mocker.patch.object(OmniOpenAIServingSpeech, "_resolve_ref_audio", _fail_resolve)
    server.__dict__["_raon_hooks"] = _FakeHooks()

    chunk_req = OpenAICreateSpeechRequest(
        input="final chunk target text",
        model="raon",
        task_type="Base",
        ref_audio="data:audio/wav;base64,UklGRgAAAABXQVZFZm10IBAA",
        ref_text="previous chunk text as reference",
        response_format="wav",
    )
    object.__setattr__(chunk_req, "_raon_min_audio_steps", 216)

    prompt = asyncio.run(_su.build_raon_speech_prompt(server, chunk_req))

    assert prompt["additional_information"]["audio_min_steps"] == [216]


def test_estimate_final_eos_min_steps_uses_observed_wps():
    assert (
        _su.estimate_final_eos_min_steps(
            target_words=30,
            previous_chunk_stats=[(30, 10.0), (30, 10.0)],
            fallback_wps=2.8,
            min_duration_ratio=0.9,
            frame_rate_hz=24.0,
        )
        == 216
    )
