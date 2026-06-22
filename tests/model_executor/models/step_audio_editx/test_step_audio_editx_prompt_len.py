"""Regression tests for StepAudioEditX prompt length estimation.

These tests keep the default CI path codec-free. They verify the estimator's
prompt construction and duration-based audio-token formula without loading the
real StepAudio tokenizer / ONNX models.
"""

import re
from typing import Any

import pytest
import torch

from vllm_omni.model_executor.models.step_audio_editx import step_audio_tokenizer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Stand-in overhead for role/special tokens inserted by apply_chat_template.
CHAT_TEMPLATE_FIXED_TOKEN_OVERHEAD = 4


class _FakeTextTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self.last_input_ids: list[int] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> dict[str, list[int]]:
        assert tokenize is True
        assert add_generation_prompt is True
        self.calls.append(messages)

        content = "\n".join(message["content"] for message in messages)
        audio_token_count = len(re.findall(r"<audio_\d+>", content))
        text_without_audio = re.sub(r"<audio_\d+>", "", content)

        # Small deterministic stand-in for chat-template tokenization. Audio
        # placeholders count as one token each, which lets the estimator replace
        # the single dummy placeholder with the duration-based audio-token count.
        token_count = CHAT_TEMPLATE_FIXED_TOKEN_OVERHEAD + len(text_without_audio.split()) + audio_token_count
        self.last_input_ids = list(range(token_count))
        return {"input_ids": self.last_input_ids}


def _patch_estimator_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    text_tokenizer: _FakeTextTokenizer,
    ref_audio_samples: int = 16_000,
    vq02_len: int = 16,
) -> None:
    def fake_from_pretrained(path: str, **kwargs: Any) -> _FakeTextTokenizer:
        assert path == "fake-model"
        assert kwargs == {"trust_remote_code": True}
        return text_tokenizer

    monkeypatch.setattr(
        step_audio_tokenizer.AutoTokenizer,
        "from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        step_audio_tokenizer.StepAudioTokenizer,
        "preprocess_wav",
        staticmethod(lambda _audio, _sr: torch.zeros(1, ref_audio_samples)),
    )
    monkeypatch.setattr(
        step_audio_tokenizer.StepAudioTokenizer,
        "split_audio",
        staticmethod(lambda audio, chunk_duration=480000: [audio]),
    )
    monkeypatch.setattr(step_audio_tokenizer, "estimate_vq02_len", lambda _num_samples: vq02_len)


def test_estimate_prompt_len_clone_replaces_dummy_audio_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text_tokenizer = _FakeTextTokenizer()
    _patch_estimator_dependencies(monkeypatch, text_tokenizer=text_tokenizer)

    est = step_audio_tokenizer.estimate_step_audio_editx_prompt_len(
        additional_information={
            "edit_type": ["clone"],
            "text": ["Please review the document before we begin."],
            "ref_text": ["Good one."],
            "ref_audio": ["unused.wav"],
            "sr": [16000],
        },
        model_path="fake-model/tokenizer_config.json",
    )

    # ref_audio_samples=16000 gives vq06_len=25. With patched vq02_len=16:
    # min(vq02//2=8, vq06//3=8) * 5 = 40 audio placeholder tokens.
    assert est == len(text_tokenizer.last_input_ids) - 1 + 40

    rendered_prompt = "\n".join(message["content"] for message in text_tokenizer.calls[0])
    assert "Good one." in rendered_prompt
    assert "Please review the document before we begin." in rendered_prompt
    assert rendered_prompt.count("<audio_0>") == 1


@pytest.mark.parametrize(
    ("edit_type", "edit_info", "text"),
    [
        ("emotion", "angry", "Please review the document before we begin."),
        ("style", "sweet", "Please review the document before we begin."),
        ("paralinguistic", None, "[laughter] Please review the document."),
        ("denoise", None, ""),
    ],
)
def test_estimate_prompt_len_edit_tasks_accept_scalar_and_list_inputs(
    monkeypatch: pytest.MonkeyPatch,
    edit_type: str,
    edit_info: str | None,
    text: str,
) -> None:
    text_tokenizer = _FakeTextTokenizer()
    _patch_estimator_dependencies(
        monkeypatch,
        text_tokenizer=text_tokenizer,
        ref_audio_samples=32_000,
        vq02_len=32,
    )

    info: dict[str, Any] = {
        "edit_type": edit_type,
        "text": text,
        "ref_text": ["reference transcript"],
        "ref_audio": "unused.wav",
        "sr": 16000,
    }
    if edit_info is not None:
        info["edit_info"] = [edit_info]

    est = step_audio_tokenizer.estimate_step_audio_editx_prompt_len(
        additional_information=info,
        model_path="fake-model",
    )

    # ref_audio_samples=32000 gives vq06_len=50. With patched vq02_len=32:
    # min(vq02//2=16, vq06//3=16) * 5 = 80 audio placeholder tokens.
    assert est == len(text_tokenizer.last_input_ids) - 1 + 80

    rendered_prompt = "\n".join(message["content"] for message in text_tokenizer.calls[0])
    if edit_type not in {"denoise", "vad"}:
        assert "reference transcript" in rendered_prompt
    assert rendered_prompt.count("<audio_0>") == 1


@pytest.mark.parametrize(
    ("num_samples", "expected"),
    [
        (0, 0),
        (399, 0),
        (400, 0),
        (879, 0),
        (880, 1),
        (1359, 1),
        (1360, 2),
        (3839, 4),
        (3840, 4),
        (3841, 4),
        (3840 + 79, 4),
        (3840 + 80, 4),
        (3840 + 879, 4),
        (3840 + 880, 5),
        (2 * 3840, 8),
        (2 * 3840 + 880, 9),
    ],
)
def test_estimate_vq02_len_boundaries(num_samples: int, expected: int) -> None:
    assert step_audio_tokenizer.estimate_vq02_len(num_samples) == expected
