# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving-layer regressions for the Audex zero-codec-token contract.

A zero-codec-token (or phase-invalid TTA) request reaches the serving layer
as a stream whose only audio payloads are zero-size tensors. The streaming
generator must not turn that into a header-only/empty successful stream: the
raw-bytes generator raises (before any WAV header), and the SSE wrapper
surfaces ``speech.audio.error`` instead of ``speech.audio.done``. The
contract covers both audex pipelines (``audex`` TTS and ``audex_tta``).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.serving_speech import (
    _AUDEX_NO_AUDIO_GUARD_MODEL_TYPES,
    OmniOpenAIServingSpeech,
)


def _serving(model_type: str = "audex") -> OmniOpenAIServingSpeech:
    serving = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    serving._tts_model_type = model_type
    serving._mark_ref_audio_artifact_ready_for_request = lambda request_id: None
    serving._discard_ref_audio_artifact_warmup = lambda request_id: None
    return serving


def _result(audio: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(multimodal_output={"model_outputs": audio, "sr": torch.tensor(16000, dtype=torch.int32)})


async def _gen(*results):
    for res in results:
        yield res


async def _collect(chunks):
    out = []
    async for chunk in chunks:
        out.append(chunk)
    return out


def test_no_audio_guard_covers_both_audex_pipelines():
    # The non-streaming CreateAudio guard keys off the same constant; pin
    # its membership so neither pipeline regresses to empty-200 responses.
    assert _AUDEX_NO_AUDIO_GUARD_MODEL_TYPES == {"audex", "audex_tta"}


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["audex", "audex_tta"])
async def test_raw_stream_raises_on_empty_audio_before_any_header(model_type):
    serving = _serving(model_type)
    chunks = serving._generate_audio_chunks(
        _gen(_result(torch.empty(0)), _result(torch.empty(0))),
        request_id="req-empty",
        response_format="wav",
    )
    with pytest.raises(ValueError, match="no audio"):
        await _collect(chunks)


@pytest.mark.asyncio
async def test_raw_stream_yields_audio_normally_when_non_empty():
    serving = _serving()
    chunks = serving._generate_audio_chunks(
        _gen(_result(torch.zeros(320)), _result(torch.empty(0))),
        request_id="req-ok",
        response_format="wav",
    )
    collected = await _collect(chunks)
    # WAV header + one PCM chunk; the trailing empty chunk is skipped silently.
    assert len(collected) == 2
    assert collected[0][:4] == b"RIFF"
    assert len(collected[1]) == 320 * 2  # int16 PCM


@pytest.mark.asyncio
@pytest.mark.parametrize("model_type", ["audex", "audex_tta"])
async def test_sse_stream_emits_error_not_done_on_empty_audio(model_type):
    serving = _serving(model_type)
    events = serving._generate_audio_sse_events(
        _gen(_result(torch.empty(0))),
        request_id="req-empty",
        response_format="pcm",
    )
    collected = await _collect(events)
    assert any("speech.audio.error" in event for event in collected)
    assert not any("speech.audio.done" in event for event in collected)


@pytest.mark.asyncio
async def test_non_audex_empty_chunks_keep_legacy_behavior():
    """Other models keep their existing semantics (no new rejection)."""
    serving = _serving(model_type="qwen3_tts")
    chunks = serving._generate_audio_chunks(
        _gen(_result(torch.empty(0))),
        request_id="req-other",
        response_format="pcm",
    )
    await _collect(chunks)  # must not raise


# ---------------------------------------------------------------- CFG serving surface

import re  # noqa: E402

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest  # noqa: E402
from vllm_omni.entrypoints.openai.tts_adapters.audex import AudexAdapter  # noqa: E402


def _adapter() -> AudexAdapter:
    return AudexAdapter.__new__(AudexAdapter)


def _speech_request(**extra_params) -> OpenAICreateSpeechRequest:
    payload = {"model": "audex", "input": "Hello world."}
    if extra_params:
        payload["extra_params"] = extra_params
    return OpenAICreateSpeechRequest(**payload)


class _MarkerTokenizer:
    """Special markers count one token; words and punctuation split."""

    _TOKEN_RE = re.compile(r"<unk>|<\|[^|]*\|>|<[a-z_]+>|\w+|[^\w\s]")

    def encode(self, text: str) -> list[int]:
        return [0] * len(self._TOKEN_RE.findall(text))


class TestAudexCfgValidation:
    def test_no_cfg_scale_accepted(self):
        assert _adapter().validate(_speech_request()) is None

    @pytest.mark.parametrize("scale", [1.0, 1.5, 3.0, 10.0])
    def test_in_range_scale_accepted(self, scale):
        assert _adapter().validate(_speech_request(cfg_scale=scale)) is None

    @pytest.mark.parametrize("scale", [0.5, 0.0, -1.0, 10.5, "big"])
    def test_out_of_range_or_malformed_scale_rejected(self, scale):
        error = _adapter().validate(_speech_request(cfg_scale=scale))
        assert error is not None and "cfg_scale" in error

    def test_explicit_null_scale_treated_as_absent(self):
        assert _adapter().validate(_speech_request(cfg_scale=None)) is None
        serving = TestAudexCfgInjection()._serving_with_fake_tokenizer()
        params = SimpleNamespace(extra_args={"cfg_scale": None})
        out = serving._inject_audex_cfg_pair_args("req-1", {"prompt": "p"}, params)
        assert out.extra_args == {}

    @pytest.mark.parametrize("key", ["cfg_role", "cfg_pair_id", "cfg_null_prompt"])
    def test_internal_cfg_keys_rejected(self, key):
        error = _adapter().validate(_speech_request(**{key: "x"}))
        assert error is not None and key in error


class TestAudexCfgInjection:
    def _serving_with_fake_tokenizer(self):
        serving = _serving()
        serving._get_audex_tokenizer = lambda: _MarkerTokenizer()
        return serving

    def _cond_prompt(self, text: str = "Hello world.") -> str:
        from vllm_omni.model_executor.models.audex.prompt import build_cond_prompt

        return build_cond_prompt(text)

    def test_no_extra_args_is_untouched(self):
        serving = self._serving_with_fake_tokenizer()
        params = SimpleNamespace(extra_args=None)
        out = serving._inject_audex_cfg_pair_args("req-1", {"prompt": self._cond_prompt()}, params)
        assert out.extra_args is None
        assert params.extra_args is None

    def test_scale_one_leaves_params_byte_identical(self):
        serving = self._serving_with_fake_tokenizer()
        params = SimpleNamespace(extra_args={"cfg_scale": 1.0})
        out = serving._inject_audex_cfg_pair_args("req-1", {"prompt": self._cond_prompt()}, params)
        assert out.extra_args == {}

    def test_guided_request_gains_pair_contract(self):
        serving = self._serving_with_fake_tokenizer()
        params = SimpleNamespace(extra_args={"cfg_scale": 1.5})
        cond = self._cond_prompt("Some transcription text here.")
        out = serving._inject_audex_cfg_pair_args("req-42", {"prompt": cond}, params)

        extra = out.extra_args
        assert extra["cfg_scale"] == 1.5
        assert extra["cfg_role"] == "cond"
        assert extra["cfg_pair_id"] == "req-42"
        null_prompt = extra["cfg_null_prompt"]
        assert "<unk>" in null_prompt
        assert "Some transcription" not in null_prompt
        tok = _MarkerTokenizer()
        assert len(tok.encode(null_prompt)) == len(tok.encode(cond))

    def test_missing_text_prompt_raises(self):
        serving = self._serving_with_fake_tokenizer()
        params = SimpleNamespace(extra_args={"cfg_scale": 1.5})
        with pytest.raises(ValueError, match="adapter-built text prompt"):
            serving._inject_audex_cfg_pair_args("req-1", {"prompt_token_ids": [1]}, params)

    def test_injection_never_mutates_input_params(self):
        """The input may be the engine's SHARED default params (review P2):
        pair state must land only on the returned clone."""
        serving = self._serving_with_fake_tokenizer()
        shared = SimpleNamespace(extra_args={"cfg_scale": 1.5}, temperature=0.1)
        out = serving._inject_audex_cfg_pair_args("req-7", {"prompt": self._cond_prompt()}, shared)
        assert out is not shared
        assert shared.extra_args == {"cfg_scale": 1.5}
        assert shared.temperature == 0.1
        assert out.extra_args["cfg_pair_id"] == "req-7"
        assert out.temperature == 0.05


class TestAudexStageDetection:
    """audex_omni is speech-capable only alongside the speech decoder."""

    def _serving_with_stages(self, stages: list[str]) -> OmniOpenAIServingSpeech:
        serving = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
        serving.engine_client = SimpleNamespace(
            stage_configs=[SimpleNamespace(engine_args=SimpleNamespace(model_stage=s)) for s in stages]
        )
        return serving

    def test_full_pipeline_detected_as_tts(self):
        serving = self._serving_with_stages(["audex_omni", "audex_code2wav"])
        stage = serving._find_tts_stage()
        assert stage is not None and stage.engine_args.model_stage == "audex_omni"

    def test_thinker_only_not_detected_as_tts(self):
        serving = self._serving_with_stages(["audex_omni"])
        assert serving._find_tts_stage() is None

    def test_tts_pipeline_still_detected(self):
        serving = self._serving_with_stages(["audex_thinker", "audex_code2wav"])
        stage = serving._find_tts_stage()
        assert stage is not None and stage.engine_args.model_stage == "audex_thinker"


def test_guided_injection_sets_guided_temperature():
    """Guided requests decode at the measured CFG temperature (0.05)."""
    serving = TestAudexCfgInjection()._serving_with_fake_tokenizer()
    cond = TestAudexCfgInjection()._cond_prompt()
    params = SimpleNamespace(extra_args={"cfg_scale": 1.5}, temperature=0.1)
    out = serving._inject_audex_cfg_pair_args("req-1", {"prompt": cond}, params)
    assert out.temperature == 0.05

    unguided = SimpleNamespace(extra_args={"cfg_scale": 1.0}, temperature=0.1)
    out = serving._inject_audex_cfg_pair_args("req-2", {"prompt": cond}, unguided)
    assert out.temperature == 0.1


# ---------------------------------------------------------------- TTA serving surface

from vllm_omni.entrypoints.openai.tts_adapters import resolve_adapter  # noqa: E402
from vllm_omni.entrypoints.openai.tts_adapters.audex_tta import AudexTTAAdapter  # noqa: E402


def _tta_adapter() -> AudexTTAAdapter:
    return AudexTTAAdapter.__new__(AudexTTAAdapter)


class _TTAVocabTokenizer(_MarkerTokenizer):
    """Marker tokenizer + a full audiocodec vocab for phase-id building."""

    def get_vocab(self):
        vocab = {"<audiogen_start>": 7, "<audiogen_end>": 8}
        for n in range(8192):
            vocab[f"<audiocodec_{n}>"] = 1000 + n
        return vocab


class TestAudexTTAAdapterValidation:
    def test_default_caption_accepted(self):
        assert _tta_adapter().validate(_speech_request()) is None

    def test_empty_caption_rejected(self):
        req = _speech_request()
        req.input = "  "
        assert _tta_adapter().validate(req) is not None

    @pytest.mark.parametrize("scale", [1.0, 3.0, 10.0])
    def test_in_range_scale_accepted(self, scale):
        assert _tta_adapter().validate(_speech_request(cfg_scale=scale)) is None

    @pytest.mark.parametrize("scale", [0.5, 10.5, "big"])
    def test_out_of_range_scale_rejected(self, scale):
        error = _tta_adapter().validate(_speech_request(cfg_scale=scale))
        assert error is not None and "cfg_scale" in error

    @pytest.mark.parametrize("key", ["cfg_role", "cfg_pair_id", "cfg_null_prompt", "tta_rvq"])
    def test_internal_keys_rejected(self, key):
        error = _tta_adapter().validate(_speech_request(**{key: "x"}))
        assert error is not None and key in error


class TestAdapterRouting:
    def test_tta_adapter_resolves_by_model_type(self):
        adapter_cls = resolve_adapter("audex_tta")
        assert adapter_cls is AudexTTAAdapter
        assert adapter_cls.stage_keys == frozenset({"audex_tta_thinker"})

    def test_tts_adapter_never_selects_tta_stage(self):
        assert "audex_tta_thinker" not in AudexAdapter.stage_keys
        assert not (AudexAdapter.stage_keys & AudexTTAAdapter.stage_keys)

    def test_detection_separates_tts_and_tta(self):
        tta_serving = TestAudexStageDetection()._serving_with_stages(["audex_tta_thinker", "audex_xcodec"])
        tta_serving._tts_stage = tta_serving._find_tts_stage()
        assert tta_serving._tts_stage.engine_args.model_stage == "audex_tta_thinker"
        assert tta_serving._detect_tts_model_type() == "audex_tta"

        tts_serving = TestAudexStageDetection()._serving_with_stages(["audex_thinker", "audex_code2wav"])
        tts_serving._tts_stage = tts_serving._find_tts_stage()
        assert tts_serving._detect_tts_model_type() == "audex"


class TestAudexTTAInjection:
    def _serving(self):
        serving = _serving(model_type="audex_tta")
        serving._get_audex_tokenizer = lambda: _TTAVocabTokenizer()
        serving._audex_tta_rvq = None
        return serving

    def _prompt(self):
        from vllm_omni.model_executor.models.audex.prompt import build_tta_cond_prompt

        return {"prompt": build_tta_cond_prompt("Rain on a tin roof.")}

    def test_default_injection_applies_rvq_and_cfg3(self):
        serving = self._serving()
        params = SimpleNamespace(extra_args=None)
        out = serving._inject_audex_tta_args("req-9", self._prompt(), params)

        extra = out.extra_args
        rvq = extra["tta_rvq"]
        assert len(rvq["phase_token_ids"]) == 4
        assert all(len(ids) == 1024 for ids in rvq["phase_token_ids"])
        assert (rvq["start_tid"], rvq["end_tid"]) == (7, 8)
        assert rvq["codec_cap"] == 4000 and rvq["start_in_prompt"] is True
        assert extra["cfg_scale"] == 3.0
        assert extra["cfg_role"] == "cond" and extra["cfg_pair_id"] == "req-9"
        assert "<unk>" in extra["cfg_null_prompt"]

    def test_explicit_scale_one_disables_cfg_but_keeps_rvq(self):
        serving = self._serving()
        params = SimpleNamespace(extra_args={"cfg_scale": 1.0})
        out = serving._inject_audex_tta_args("req-1", self._prompt(), params)
        assert "tta_rvq" in out.extra_args
        assert "cfg_scale" not in out.extra_args
        assert "cfg_role" not in out.extra_args

    def test_rvq_contract_cached_per_process(self):
        serving = self._serving()
        out_a = serving._inject_audex_tta_args("a", self._prompt(), SimpleNamespace(extra_args=None))
        out_b = serving._inject_audex_tta_args("b", self._prompt(), SimpleNamespace(extra_args=None))
        assert out_a.extra_args["tta_rvq"] is out_b.extra_args["tta_rvq"]

    def test_injection_never_mutates_input_params(self):
        """The default online path hands the injector an element of the
        engine's SHARED default_sampling_params_list (review P2): the
        per-request pair state must land only on the returned clone."""
        serving = self._serving()
        shared = SimpleNamespace(extra_args=None)
        out_1 = serving._inject_audex_tta_args("req-1", self._prompt(), shared)
        out_2 = serving._inject_audex_tta_args("req-2", self._prompt(), shared)
        assert shared.extra_args is None
        assert out_1 is not shared and out_2 is not shared
        assert out_1.extra_args["cfg_pair_id"] == "req-1"
        assert out_2.extra_args["cfg_pair_id"] == "req-2"
