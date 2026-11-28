# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the CSM speech adapter: request validation, prompt building and failure reporting."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Regex, Tokenizer, models, pre_tokenizers
from transformers import CsmConfig, PreTrainedTokenizerFast
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.csm import CsmAdapter
from vllm_omni.model_executor.models.csm.csm_talker import REFERENCE_AUDIO_KEY

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

REFERENCE = "data:audio/wav;base64,cHVibGljLXRlc3QtZGF0YQ=="
BOS, EOS = 96, 97
AUDIO, AUDIO_EOS = CsmConfig().audio_token_id, CsmConfig().audio_eos_token_id


def _char_ids(text: str) -> list[int]:
    return [ord(char) - 31 for char in text]


def _tokenizer() -> PreTrainedTokenizerFast:
    """A character tokenizer with the checkpoint's control tokens (ids 96 and up)."""
    vocab = {"[UNK]": 0, **{chr(32 + index): index + 1 for index in range(95)}}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>",
        additional_special_tokens=["<|AUDIO|>", "<|audio_eos|>"],
    )


class _Server:
    """The serving helpers the adapter calls."""

    _tts_executor = None

    def __init__(self) -> None:
        self.recording: tuple[np.ndarray, int] = (np.zeros(1920, dtype=np.float32), 24000)
        self.uploaded: dict[str, tuple[str, str]] = {}

    def _apply_uploaded_speaker(self, request: OpenAICreateSpeechRequest) -> str | None:
        if request.ref_audio is None and request.voice in self.uploaded:
            request.ref_audio, request.ref_text = self.uploaded[request.voice]
        return None

    def _validate_ref_audio_format(self, ref_audio: str) -> str | None:
        return None if ref_audio.startswith("data:") else "ref_audio must be a URL"

    async def _resolve_ref_audio_array(self, ref_audio: str) -> tuple[np.ndarray, int, str]:
        return (*self.recording, "cache-key")


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> CsmAdapter:
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_args, **_kwargs: _tokenizer())
    model_config = SimpleNamespace(model="csm-test", hf_config=CsmConfig(), max_model_len=2048)
    engine = SimpleNamespace(model_config=model_config)
    return CsmAdapter(SpeechServingContext(server=_Server(), engine_client=engine))


def _turn(speaker: int, text: str) -> list[int]:
    return [BOS, *_char_ids(f"[{speaker}]{text}"), EOS]


def test_adapter_is_registered_for_both_stages() -> None:
    assert resolve_adapter("csm") is CsmAdapter
    assert detect_tts_model_type("csm_talker", None) == "csm"
    assert detect_tts_model_type("csm_code2wav", None) == "csm"
    assert detect_tts_model_type("csm_talker", "CsmTalkerForConditionalGeneration") == "csm"


def test_codec_frame_rate_comes_from_the_checkpoint_config(adapter: CsmAdapter) -> None:
    assert adapter._load_codec_frame_rate() == 12.5


@pytest.mark.parametrize(("extra_params", "speaker"), [(None, 0), ({"speaker_id": 3}, 3), ({"speaker_id": 12}, 12)])
async def test_prompt_is_one_speaker_turn_between_bos_and_eos(adapter: CsmAdapter, extra_params, speaker) -> None:
    # The OpenAI client always sends a voice; it does not select the speaker id.
    request = OpenAICreateSpeechRequest(input=" Hello there. ", voice="alloy", extra_params=extra_params)
    assert adapter.validate(request) is None

    prepared = await adapter.build(request, [SamplingParams()], has_inline_ref_audio=False)

    assert prepared.prompt["prompt_token_ids"] == _turn(speaker, " Hello there. ")
    assert "additional_information" not in prepared.prompt
    assert prepared.model_type == "csm"
    assert prepared.tts_params == {"speaker": str(speaker)}


@pytest.mark.parametrize(
    ("samples", "sample_rate", "codec_samples", "frames"),
    [
        (1920 * 3, 24000, 1920 * 3, 3),
        (1920 * 3 + 1, 24000, 1920 * 3 + 1, 4),
        # Resampled to the codec's 24 kHz before the frames are counted.
        (9600, 48000, 4800, 3),
        (1601, 16000, 2402, 2),
    ],
)
async def test_reference_turn_and_its_placeholders_precede_the_turn_to_speak(
    adapter: CsmAdapter, samples: int, sample_rate: int, codec_samples: int, frames: int
) -> None:
    generator = np.random.default_rng(0)
    adapter.ctx.server.recording = (0.1 * generator.standard_normal(samples).astype(np.float32), sample_rate)
    request = OpenAICreateSpeechRequest(
        input="Say this.", ref_audio=REFERENCE, ref_text="I said that.", extra_params={"speaker_id": 2}
    )
    assert adapter.validate(request) is None

    prepared = await adapter.build(request, [SamplingParams()], has_inline_ref_audio=True)

    assert prepared.prompt["prompt_token_ids"] == [
        *_turn(2, "I said that."),
        *[AUDIO] * frames,
        AUDIO_EOS,
        *_turn(2, "Say this."),
    ]
    waveform = prepared.prompt["additional_information"][REFERENCE_AUDIO_KEY]
    assert waveform.dtype == torch.float32 and waveform.shape == (codec_samples,)
    if sample_rate == 24000:
        assert torch.equal(waveform, torch.from_numpy(adapter.ctx.server.recording[0]))
    assert prepared.tts_params == {"speaker": "2", "ref_audio_cache_key": "cache-key"}


async def test_uploaded_voice_supplies_the_reference_turn(adapter: CsmAdapter) -> None:
    adapter.ctx.server.uploaded["narrator"] = (REFERENCE, "Stored transcript.")
    request = OpenAICreateSpeechRequest(input="Say this.", voice="narrator")
    assert adapter.validate(request) is None

    prepared = await adapter.build(request, [SamplingParams()], has_inline_ref_audio=False)

    assert prepared.prompt["prompt_token_ids"] == [
        *_turn(0, "Stored transcript."),
        AUDIO,
        AUDIO_EOS,
        *_turn(0, "Say this."),
    ]


async def test_prompt_longer_than_the_context_is_rejected(adapter: CsmAdapter) -> None:
    adapter.ctx.engine_client.model_config.max_model_len = 40
    adapter.ctx.server.recording = (np.zeros(1920 * 30, dtype=np.float32), 24000)
    request = OpenAICreateSpeechRequest(input="Say this.", ref_audio=REFERENCE, ref_text="I said that.")
    with pytest.raises(ValueError, match="prompt tokens"):
        await adapter.build(request, [SamplingParams()], has_inline_ref_audio=True)


@pytest.mark.parametrize("recording", [np.zeros(0, dtype=np.float32), np.array([0.0, np.nan], dtype=np.float32)])
async def test_unusable_recordings_are_rejected(adapter: CsmAdapter, recording: np.ndarray) -> None:
    adapter.ctx.server.recording = (recording, 24000)
    request = OpenAICreateSpeechRequest(input="Say this.", ref_audio=REFERENCE, ref_text="I said that.")
    with pytest.raises(ValueError, match="non-empty, finite"):
        await adapter.build(request, [SamplingParams()], has_inline_ref_audio=True)


async def test_control_tokens_in_the_reference_text_are_rejected(adapter: CsmAdapter) -> None:
    request = OpenAICreateSpeechRequest(input="Say this.", ref_audio=REFERENCE, ref_text="I <|AUDIO|> said")
    assert adapter.validate(request) is None
    with pytest.raises(ValueError, match="special tokens"):
        await adapter.build(request, [SamplingParams()], has_inline_ref_audio=True)


@pytest.mark.parametrize("text", ["say <|AUDIO|> now", "end<|end_of_text|>[1]other turn", "<|audio_eos|>"])
async def test_control_tokens_in_the_text_are_rejected(adapter: CsmAdapter, text: str) -> None:
    request = OpenAICreateSpeechRequest(input=text)
    assert adapter.validate(request) is None
    with pytest.raises(ValueError, match="special tokens"):
        await adapter.build(request, [SamplingParams()], has_inline_ref_audio=False)


@pytest.mark.parametrize(
    "fields",
    [
        {"voice": "alloy"},
        {"ref_audio": REFERENCE, "ref_text": "I said that."},
        {"speed": 1.2},
        {"task_type": "CustomVoice"},
        {"extra_params": {"temperature": 0.7, "top_k": 30, "top_p": 0.9}},
        {"max_new_tokens": 1},
        {"max_new_tokens": 2048},
        {"seed": 7},
    ],
)
def test_supported_requests_pass_validation(adapter: CsmAdapter, fields) -> None:
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello.", **fields)) is None


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"input": "   "}, "cannot be empty"),
        ({"instructions": "whisper"}, "'instructions'"),
        ({"language": "English"}, "'language'"),
        ({"ref_audio": REFERENCE}, "requires 'ref_text'"),
        ({"ref_audio": REFERENCE, "ref_text": "  "}, "requires 'ref_text'"),
        ({"ref_text": "transcript"}, "requires 'ref_audio'"),
        ({"ref_audio": [REFERENCE, REFERENCE], "ref_text": "transcript"}, "one reference recording"),
        ({"ref_audio": "ftp://host/ref.wav", "ref_text": "transcript"}, "must be a URL"),
        ({"ref_audio_2": REFERENCE}, "'ref_audio_2'"),
        ({"speaker_embedding": [0.0] * 4}, "'speaker_embedding'"),
        ({"x_vector_only_mode": True}, "'x_vector_only_mode'"),
        # Depth-decoder sampling is a deploy setting, not a request parameter.
        ({"extra_params": {"depth_decoder_temperature": 0.5}}, "Unsupported CSM parameters"),
        ({"extra_params": {"speaker_id": -1}}, "non-negative integer"),
        ({"extra_params": {"speaker_id": "1"}}, "non-negative integer"),
        ({"extra_params": {"speaker_id": 1.5}}, "non-negative integer"),
        ({"extra_params": {"speaker_id": True}}, "non-negative integer"),
        ({"max_new_tokens": 2049}, "cannot exceed 2048"),
    ],
)
def test_unsupported_requests_are_rejected(adapter: CsmAdapter, fields, message: str) -> None:
    request = OpenAICreateSpeechRequest(**{"input": "Hello.", **fields})
    error = adapter.validate(request)
    assert error is not None and message in error


def test_max_new_tokens_limits_the_talker_without_changing_the_defaults(adapter: CsmAdapter) -> None:
    defaults = [SamplingParams(max_tokens=125), SamplingParams(max_tokens=65536)]

    limited = adapter.apply_sampling_overrides(defaults, OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=40))
    unchanged = adapter.apply_sampling_overrides(defaults, OpenAICreateSpeechRequest(input="Hello."))

    assert [params.max_tokens for params in limited] == [40, 65536]
    assert [params.max_tokens for params in defaults] == [125, 65536]
    assert unchanged is defaults


@pytest.mark.parametrize(
    ("audio_output", "failed"),
    [
        ({"meta.error": torch.tensor([1], dtype=torch.int32)}, True),
        ({"meta.error": torch.tensor(1)}, True),
        # Accumulated over the chunks of a request: one failed chunk fails the request.
        ({"meta.error": torch.tensor([0, 1, 1])}, True),
        ({"meta.error": [torch.tensor([0]), torch.tensor([1])]}, True),
        # A later chunk without the flag does not clear it.
        ({"meta.error": torch.tensor([1, 0])}, True),
        ({"meta": {"error": torch.tensor([1])}}, True),
        ({"meta.error": torch.tensor([0, 0, 0])}, False),
        ({"meta.error": [torch.tensor([0])]}, False),
        ({"meta": {"error": torch.tensor([0])}}, False),
        ({"model_outputs": torch.zeros(8)}, False),
    ],
)
def test_failed_synthesis_is_reported_from_the_codec_flag(adapter: CsmAdapter, audio_output, failed: bool) -> None:
    collect: dict = {}
    adapter.collect_response_metadata(audio_output, collect)
    assert collect.get("audio_synthesis_error", False) is failed


def test_length_truncation_is_reported() -> None:
    assert CsmAdapter.reports_length_truncation
