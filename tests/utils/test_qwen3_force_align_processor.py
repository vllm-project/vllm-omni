import pytest

from vllm_omni.utils import qwen3_force_align_processor as processor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_build_prompt_has_boundary_timestamp_markers():
    prompt = processor.build_prompt(["hello", "world"])

    assert prompt.count("<timestamp>") == 4
    assert "hello<timestamp><timestamp>world" in prompt
    # Official aligner format: audio placeholder + words, no chat template
    # (a leading "<|im_start|>user\n" shifts the predicted markers one bin late).
    assert prompt == f"{processor.AUDIO_PLACEHOLDER}hello<timestamp><timestamp>world<timestamp><timestamp>"
    assert "<|im_start|>" not in prompt
    assert processor.build_prompt([]) == f"{processor.AUDIO_PLACEHOLDER}<timestamp><timestamp>"


def test_build_prompt_matches_official_encode_timestamp():
    pytest.importorskip("qwen_asr")
    from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor

    text = "It's 3 o'clock, 你好 world."
    word_list, official_prompt = Qwen3ForceAlignProcessor().encode_timestamp(text, "english")

    assert processor.build_prompt(list(word_list)) == official_prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        # Punctuation inside a whitespace segment is stripped, not split on:
        # the hand-rolled splitter used to break these into many fake words.
        ("U.S.A", ["USA"]),
        ("hello, world!", ["hello", "world"]),
        ("don't stop", ["don't", "stop"]),
        # CJK characters peel off as individual tokens, Latin runs stay whole.
        ("你好world", ["你", "好", "world"]),
        ("我爱 China", ["我", "爱", "China"]),
        ("   spaced   out   ", ["spaced", "out"]),
    ],
)
def test_tokenize_space_lang_matches_official_segmentation(text, expected):
    assert processor._tokenize_space_lang(text) == expected


def test_segment_words_falls_back_to_port_without_qwen_asr(monkeypatch):
    # With qwen_asr unavailable, segmentation must use the built-in port and
    # still produce the faithful result for the common (non-JP/KO) case.
    monkeypatch.setattr(processor, "_get_official_processor", lambda: None)

    assert processor.segment_words("U.S.A is here", "auto") == ["USA", "is", "here"]


def test_fix_timestamp_repairs_dip_onto_monotonic_sequence():
    # 200 dips below the preceding 400; it snaps back up to 400.
    assert processor.fix_timestamp([0, 400, 200, 800]) == [0, 400, 400, 800]


def test_fix_timestamp_passes_through_monotonic():
    assert processor.fix_timestamp([0, 100, 100, 250]) == [0, 100, 100, 250]
