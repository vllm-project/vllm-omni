# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the YuE2-3B speech adapter.

Covers detection, the reject matrix (cot/abc coupling, unsupported speech
contract fields, streaming, fixed sampling), max_new_tokens frame bounds,
build() parity with the offline prompt helper, the context budget, sampling
overrides, and response-metadata collection.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.yue2 import Yue2Adapter
from vllm_omni.model_executor.models.yue2.constants import (
    CONTEXT,
    KEY_MAX_AUDIO_FRAMES,
    KEY_PHASE,
    KEY_PREFIX_IDS,
    KEY_SEED,
    KEY_SKIP_SYNTHESIS,
    SEMANTIC_SAMPLING,
    STOP_TOKEN_IDS,
)
from vllm_omni.model_executor.models.yue2.prompt import semantic_prefix_ids

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_MODEL = "m-a-p/YuE2-3B"

_LYRICS = "一闪一闪亮晶晶\n满天都是小星星"
_STYLE = "Chinese heavy metal, distorted guitars, 140 BPM, key of C"
_ABC = "X:1\nT:Twinkle\nM:4/4\nL:1/4\nK:C\nC C G G|A A G2|"


def _adapter() -> Yue2Adapter:
    engine = SimpleNamespace(model_config=SimpleNamespace(model=_MODEL, revision=None))
    return Yue2Adapter(SpeechServingContext(server=SimpleNamespace(engine_client=engine), engine_client=engine))


def _request(**kwargs) -> OpenAICreateSpeechRequest:
    base = {"input": _LYRICS, "instructions": _STYLE}
    base.update(kwargs)
    return OpenAICreateSpeechRequest(**base)


def _fake_encode(text: str) -> list[int]:
    # Deterministic stand-in for the tiktoken BPE: one id per character.
    return [ord(c) % 50000 + 100 for c in text]


def _adapter_with_tokenizer() -> Yue2Adapter:
    adapter = _adapter()
    adapter._cached_tokenizer = SimpleNamespace(encode=_fake_encode)
    return adapter


def test_yue2_detection_and_registration() -> None:
    assert resolve_adapter("yue2") is Yue2Adapter
    assert Yue2Adapter.stage_keys == {"yue2"}
    assert Yue2Adapter.model_archs == {"Yue2ForCausalLM"}
    assert detect_tts_model_type("yue2", None) == "yue2"
    assert detect_tts_model_type("yue2", "Yue2ForCausalLM") == "yue2"
    assert detect_tts_model_type(None, "Yue2ForCausalLM") == "yue2"


def test_yue2_accepts_minimal_request() -> None:
    adapter = _adapter()
    assert adapter.validate(_request()) is None
    assert adapter.validate(_request(extra_params={"cot": "off"})) is None
    assert adapter.validate(_request(extra_params={"cot": "melody", "abc": _ABC})) is None
    assert adapter.validate(_request(extra_params={"cot": "full", "abc": _ABC})) is None
    assert adapter.validate(_request(voice="default", speed=1.0)) is None
    assert adapter.validate(_request(max_new_tokens=200)) is None
    assert adapter.validate(_request(max_new_tokens=9000)) is None


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_yue2_rejects_empty_lyrics(text: str) -> None:
    err = _adapter().validate(_request(input=text))
    assert err is not None
    assert "input" in err


@pytest.mark.parametrize("instructions", [None, "", "   "])
def test_yue2_rejects_missing_instructions(instructions) -> None:
    err = _adapter().validate(_request(instructions=instructions))
    assert err is not None
    assert "instructions" in err


def test_yue2_rejects_invalid_cot() -> None:
    err = _adapter().validate(_request(extra_params={"cot": "partial"}))
    assert err is not None
    assert "cot" in err


def test_yue2_rejects_abc_without_cot() -> None:
    err = _adapter().validate(_request(extra_params={"abc": _ABC}))
    assert err is not None
    assert "cot=melody|full" in err


@pytest.mark.parametrize("cot", ["melody", "full"])
@pytest.mark.parametrize("abc", [None, "", "   "])
def test_yue2_rejects_cot_without_abc(cot: str, abc) -> None:
    err = _adapter().validate(_request(extra_params={"cot": cot, "abc": abc}))
    assert err is not None
    assert "abc" in err


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"voice": "alloy"}, "voice"),
        ({"ref_audio": "https://example.com/ref.wav"}, "reference-audio"),
        ({"ref_text": "transcript"}, "reference-audio"),
        ({"language": "Chinese"}, "language"),
        ({"task_type": "Base"}, "task_type"),
        ({"speed": 1.5}, "speed"),
        ({"stream": True}, "streaming"),
        ({"stream_format": "sse"}, "streaming"),
        ({"stream_format": "audio"}, "streaming"),
        ({"extra_params": {"temperature": 0.3}}, "temperature"),
        ({"extra_params": {"top_p": 0.9}}, "top_p"),
        ({"extra_params": {"top_k": 20}}, "top_k"),
        ({"extra_params": {"repetition_penalty": 1.1}}, "repetition_penalty"),
    ],
)
def test_yue2_rejects_unsupported_fields(kwargs: dict, field: str) -> None:
    err = _adapter().validate(_request(**kwargs))
    assert err is not None
    assert field in err


@pytest.mark.parametrize("max_new_tokens", [1, 199])
def test_yue2_rejects_max_new_tokens_below_min(max_new_tokens: int) -> None:
    err = _adapter().validate(_request(max_new_tokens=max_new_tokens))
    assert err is not None
    assert "max_new_tokens" in err


def test_yue2_rejects_max_new_tokens_above_frame_budget() -> None:
    err = _adapter().validate(_request(max_new_tokens=9001))
    assert err is not None
    assert "max_new_tokens" in err
    assert "frames" in err


def test_yue2_build_matches_offline_prompt_builder() -> None:
    adapter = _adapter_with_tokenizer()
    prepared = asyncio.run(adapter.build(_request(), [], False))
    expected = semantic_prefix_ids(_fake_encode, _STYLE, _LYRICS, "off", abc_ids=None)
    assert prepared.prompt["prompt_token_ids"] == expected
    assert prepared.tts_params == {"max_audio_frames": [200]}
    assert prepared.model_type == "yue2"


def test_yue2_build_with_abc_includes_score_span() -> None:
    adapter = _adapter_with_tokenizer()
    request = _request(extra_params={"cot": "melody", "abc": _ABC}, max_new_tokens=400)
    prepared = asyncio.run(adapter.build(request, [], False))
    expected = semantic_prefix_ids(_fake_encode, _STYLE, _LYRICS, "melody", abc_ids=_fake_encode(_ABC))
    assert prepared.prompt["prompt_token_ids"] == expected
    assert prepared.tts_params == {"max_audio_frames": [400]}


def test_yue2_build_enforces_context_budget() -> None:
    adapter = _adapter_with_tokenizer()
    request = _request(max_new_tokens=9000, input="啦" * (CONTEXT - 100))
    with pytest.raises(ValueError, match="context"):
        asyncio.run(adapter.build(request, [], False))


def _stage_defaults() -> list:
    return [SimpleNamespace(max_tokens=1000, seed=None, extra_args=None, stop_token_ids=None, detokenize=True)]


def test_yue2_apply_sampling_overrides_pins_preset() -> None:
    adapter = _adapter()
    prefix = [1, 2, 3]
    request = _request(seed=831001, max_new_tokens=400)
    overridden = adapter.apply_sampling_overrides(_stage_defaults(), request, prompt={"prompt_token_ids": prefix})
    params = overridden[0]
    args = params.extra_args
    assert args[KEY_PHASE] == "semantic"
    assert args[KEY_SEED] == 831001
    assert args[KEY_MAX_AUDIO_FRAMES] == 400
    assert args[KEY_SKIP_SYNTHESIS] is False
    assert args[KEY_PREFIX_IDS] == prefix
    for key in ("temperature", "top_p", "top_k", "repetition_penalty", "penalty_window", "min_tokens"):
        assert args[f"yue2_{key}"] == SEMANTIC_SAMPLING[key]
    assert params.max_tokens == 401  # one extra step for the terminal NAR/VAE pass
    assert params.stop_token_ids == list(STOP_TOKEN_IDS)
    assert params.detokenize is False


def test_yue2_apply_sampling_overrides_draws_seed_when_absent() -> None:
    adapter = _adapter()
    prompt = {"prompt_token_ids": [1]}
    first = adapter.apply_sampling_overrides(_stage_defaults(), _request(), prompt=prompt)[0]
    second = adapter.apply_sampling_overrides(_stage_defaults(), _request(), prompt=prompt)[0]
    assert isinstance(first.extra_args[KEY_SEED], int)
    assert first.extra_args[KEY_SEED] != second.extra_args[KEY_SEED]


def test_yue2_collect_response_metadata_reads_truncated_flag() -> None:
    adapter = _adapter()
    collect: dict = {}
    adapter.collect_response_metadata({"meta": {"truncated": ["1"]}}, collect)
    assert collect["audio_truncated"] is True
    collect.clear()
    adapter.collect_response_metadata({"meta": {"truncated": ["0"]}}, collect)
    assert collect["audio_truncated"] is False
    collect.clear()
    adapter.collect_response_metadata({"model_outputs": []}, collect)
    assert "audio_truncated" not in collect


def test_yue2_collect_response_metadata_reads_flattened_runner_key() -> None:
    # The runner flattens meta.* to dotted keys and unwraps the per-request
    # list before the payload reaches serving (flatten_payload).
    adapter = _adapter()
    collect: dict = {}
    adapter.collect_response_metadata({"meta.truncated": "1"}, collect)
    assert collect["audio_truncated"] is True
    collect.clear()
    adapter.collect_response_metadata({"meta.truncated": "0"}, collect)
    assert collect["audio_truncated"] is False
    collect.clear()
    adapter.collect_response_metadata({"meta.truncated": ["1"]}, collect)
    assert collect["audio_truncated"] is True


def test_yue2_collect_response_metadata_reads_wire_tensor() -> None:
    # The wire payload is tensor-only (_ensure_tensor_values), so the flag
    # arrives at serving as a 0-d int tensor under the flattened dotted key.
    adapter = _adapter()
    collect: dict = {}
    adapter.collect_response_metadata({"meta.truncated": torch.tensor(1)}, collect)
    assert collect["audio_truncated"] is True
    collect.clear()
    adapter.collect_response_metadata({"meta.truncated": torch.tensor(0)}, collect)
    assert collect["audio_truncated"] is False


def test_yue2_collect_response_metadata_reads_error_flag() -> None:
    # The model flags a failed NAR/VAE pass with meta.error=1 (int, so the
    # tensor-only wire keeps it); serving turns it into a 500 via the generic
    # audio_synthesis_error collect key, same shapes as meta.truncated.
    adapter = _adapter()
    collect: dict = {}
    adapter.collect_response_metadata({"meta": {"error": ["1"]}}, collect)
    assert collect["audio_synthesis_error"] is True
    collect.clear()
    adapter.collect_response_metadata({"meta.error": "1"}, collect)
    assert collect["audio_synthesis_error"] is True
    collect.clear()
    adapter.collect_response_metadata({"meta.error": torch.tensor(1)}, collect)
    assert collect["audio_synthesis_error"] is True
    collect.clear()
    # No flag (or an explicit 0) means success: the key stays absent so the
    # generic serving check does not fire on healthy requests.
    adapter.collect_response_metadata({"meta.error": torch.tensor(0)}, collect)
    assert "audio_synthesis_error" not in collect
    adapter.collect_response_metadata({"meta.truncated": "1"}, collect)
    assert "audio_synthesis_error" not in collect


def test_yue2_tokenizer_resolves_local_dir_and_hf_id(tmp_path, monkeypatch) -> None:
    import vllm_omni.entrypoints.openai.tts_adapters.yue2 as yue2_mod

    recorded: list[str] = []

    class _FakeTokenizer:
        def __init__(self, merge_file) -> None:
            recorded.append(str(merge_file))

    monkeypatch.setattr("vllm_omni.model_executor.models.yue2.tokenizer.YuE2TextTokenizer", _FakeTokenizer)

    # Local checkpoint dir: the merge file is read in place, no hub call.
    local_dir = tmp_path / "YuE2-3B"
    local_dir.mkdir()
    (local_dir / "qwen.tiktoken").write_bytes(b"")
    monkeypatch.setattr(yue2_mod, "resolve_stage_model_path", lambda _engine: str(local_dir))
    local_adapter = _adapter()
    local_adapter._tokenizer()
    assert recorded == [str(local_dir / "qwen.tiktoken")]

    # HF repo id: only the merge file is pulled into the hub cache.
    cached = tmp_path / "hub" / "qwen.tiktoken"
    cached.parent.mkdir()
    cached.write_bytes(b"")
    calls: list[tuple[str, str]] = []

    class _FakeApi:
        def hf_hub_download(self, repo_id: str, filename: str) -> str:
            calls.append((repo_id, filename))
            return str(cached)

    monkeypatch.setattr(yue2_mod, "resolve_stage_model_path", lambda _engine: _MODEL)
    monkeypatch.setattr("vllm_omni.transformers_utils.repo_utils.hf_api", lambda: _FakeApi())
    hub_adapter = _adapter()
    hub_adapter._tokenizer()
    assert calls == [(_MODEL, "qwen.tiktoken")]
    assert recorded[-1] == str(cached)
