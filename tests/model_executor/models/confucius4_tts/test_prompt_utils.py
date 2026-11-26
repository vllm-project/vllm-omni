# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.confucius4_tts import prompt_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prompt_format_and_language_validation() -> None:
    assert tuple(prompt_utils.LANGUAGE_TOKEN_MAP) == (
        "zh",
        "en",
        "ja",
        "ko",
        "de",
        "fr",
        "es",
        "id",
        "it",
        "th",
        "pt",
        "ru",
        "ms",
        "vi",
    )
    assert prompt_utils.format_confucius4_prompt(
        "hello",
        "en",
    ) == ("You are a helpful assistant. 请用英文朗读接下来的文字:hello")
    with pytest.raises(ValueError, match="Unsupported"):
        prompt_utils.format_confucius4_prompt("hello", "invalid")
    with pytest.raises(ValueError, match="Unsupported"):
        prompt_utils.format_confucius4_prompt("hello", "tl")


_PROMPT = "You are a helpful assistant. 请用中文朗读接下来的文字:战斗战斗，只做第一"


class _Tokenizer:
    """Tokenizer double with the BOS/EOS ids of the Confucius4 tokenizer."""

    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, body: list[int], *, auto_specials: bool) -> None:
        self._body = body
        self._auto_specials = auto_specials
        self.calls: list[bool] = []

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == _PROMPT
        self.calls.append(add_special_tokens)
        if add_special_tokens and self._auto_specials:
            return [self.bos_token_id, *self._body, self.eos_token_id]
        return list(self._body)


@pytest.mark.parametrize("auto_specials", [True, False])
def test_prompt_always_wrapped_in_bos_and_eos(
    monkeypatch: pytest.MonkeyPatch,
    auto_specials: bool,
) -> None:
    # transformers 4.x appends <s>/</s> for this tokenizer config, 5.x does not. The prompt must be
    # ``<s> ... </s>`` either way: the checkpoint was trained that way, and a missing </s> shifts
    # every semantic position by one.
    tokenizer = _Tokenizer([10, 11], auto_specials=auto_specials)
    monkeypatch.setattr(prompt_utils, "_tokenizer", lambda _model_path: tokenizer)

    assert prompt_utils.tokenize_confucius4_prompt("model", "战斗战斗，只做第一", "zh") == [1, 10, 11, 2]


def test_prompt_specials_are_not_duplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    class _AlreadyWrapped(_Tokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [1, 10, 11, 2]

    monkeypatch.setattr(prompt_utils, "_tokenizer", lambda _m: _AlreadyWrapped([], auto_specials=False))

    assert prompt_utils.tokenize_confucius4_prompt("model", "战斗战斗，只做第一", "zh") == [1, 10, 11, 2]


def test_prompt_without_declared_special_ids_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoSpecials(_Tokenizer):
        bos_token_id = None
        eos_token_id = None

    monkeypatch.setattr(prompt_utils, "_tokenizer", lambda _m: _NoSpecials([10, 11], auto_specials=False))

    assert prompt_utils.tokenize_confucius4_prompt("model", "战斗战斗，只做第一", "zh") == [10, 11]


def test_remote_tokenizer_prefers_repository_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_utils._tokenizer.cache_clear()
    seen: dict[str, object] = {}

    def fake_from_pretrained(path, **kwargs):
        seen["path"] = path
        seen["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(
        prompt_utils.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )
    prompt_utils._tokenizer("netease-youdao/Confucius4-TTS")
    assert seen == {
        "path": "netease-youdao/Confucius4-TTS",
        "kwargs": {},
    }


def test_remote_tokenizer_falls_back_to_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_utils._tokenizer.cache_clear()
    calls: list[dict[str, object]] = []

    def fake_from_pretrained(path, **kwargs):
        calls.append({"path": path, "kwargs": kwargs})
        if "subfolder" not in kwargs:
            raise OSError("root tokenizer is unavailable")
        return object()

    monkeypatch.setattr(
        prompt_utils.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )
    prompt_utils._tokenizer("remote/confucius4-layout")
    assert calls == [
        {
            "path": "remote/confucius4-layout",
            "kwargs": {},
        },
        {
            "path": "remote/confucius4-layout",
            "kwargs": {
                "subfolder": "checkpoints",
            },
        },
    ]
