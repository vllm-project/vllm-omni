# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for SeedTTSTextDataset, SeedTTSTextSampleRequest, SeedTTSDesignDataset,
and SeedTTSDesignSampleRequest.

vllm stubs are installed by tests/benchmarks/conftest.py before collection.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Load the data module directly (bypasses vllm_omni.__init__ heavy imports).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "vllm_omni" / "benchmarks" / "data_modules" / "seed_tts_dataset.py"
_MODULE_NAME = "vllm_omni.benchmarks.data_modules.seed_tts_dataset"

if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import (  # noqa: E402
    SeedTTSDataset,
    SeedTTSDesignDataset,
    SeedTTSDesignSampleRequest,
    SeedTTSSampleRequest,
    SeedTTSTextDataset,
    SeedTTSTextSampleRequest,
    SeedTTSTurn,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seed_tts_root(tmp_path: Path) -> Path:
    """Minimal seed-tts-style directory with 5 entries."""
    locale_dir = tmp_path / "en"
    locale_dir.mkdir()
    wav_dir = locale_dir / "prompt-wavs"
    wav_dir.mkdir()
    for i in range(5):
        (wav_dir / f"utt{i:03d}.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    meta = "\n".join(f"utt{i:03d}|ref text {i}|prompt-wavs/utt{i:03d}.wav|target text {i}" for i in range(5))
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def mock_tokenizer(mocker):
    tokenizer = mocker.MagicMock()
    tokenizer.encode = lambda text, **kw: [0] * len(text.split())
    tokenizer.get_vocab.return_value = {"<pad>": 0}
    tokenizer.all_special_ids = []
    tokenizer.all_special_tokens = []
    tokenizer.vocab_size = 1
    tokenizer.__len__.return_value = 1
    return tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_seed_tts_dataset_groups_four_turns_with_one_reference(seed_tts_root, mock_tokenizer):
    ds = SeedTTSDataset(
        dataset_path=str(seed_tts_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )

    requests = ds.sample(
        mock_tokenizer,
        num_requests=1,
        turns_per_session=4,
        no_oversample=True,
    )

    assert len(requests) == 1
    request = requests[0]
    assert [turn.utterance_id for turn in request.seed_tts_turns] == [
        "utt000",
        "utt001",
        "utt002",
        "utt003",
    ]
    assert [turn.target_text for turn in request.seed_tts_turns] == [
        "target text 0",
        "target text 1",
        "target text 2",
        "target text 3",
    ]
    assert request.seed_tts_speech_extra["ref_text"] == "ref text 0"
    assert request.seed_tts_ref_wav_path.endswith("utt000.wav")


def test_seed_tts_eval_expands_grouped_pcm_by_turn():
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import (
        _expand_seed_tts_turn_outputs,
    )

    request = SeedTTSSampleRequest(
        prompt="target 0",
        prompt_len=4,
        expected_output_len=100,
        multi_modal_data=None,
        request_id="session-0",
        seed_tts_turns=tuple(
            SeedTTSTurn(utterance_id=f"utt-{index}", target_text=f"target {index}") for index in range(4)
        ),
    )
    output = types.SimpleNamespace(
        success=True,
        tts_turn_pcm_bytes=[bytes([index]) for index in range(4)],
    )

    requests, outputs = _expand_seed_tts_turn_outputs([request], [output])

    assert [item.prompt for item in requests] == [f"target {index}" for index in range(4)]
    assert [item.seed_tts_utterance_id for item in requests] == [f"utt-{index}" for index in range(4)]
    assert [item.tts_output_pcm_bytes for item in outputs] == [bytes([index]) for index in range(4)]


def test_seed_tts_eval_keeps_session_pcm_for_single_turn_chat():
    """Chat-omni Seed-TTS sets session PCM only; expand must not wipe it."""
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import (
        _expand_seed_tts_turn_outputs,
    )

    request = SeedTTSSampleRequest(
        prompt="target 0",
        prompt_len=4,
        expected_output_len=100,
        multi_modal_data=None,
        request_id="session-0",
        seed_tts_turns=(SeedTTSTurn(utterance_id="utt-0", target_text="target 0"),),
    )
    pcm = b"\x01\x02\x03\x04"
    output = types.SimpleNamespace(
        success=True,
        tts_output_pcm_bytes=pcm,
        tts_turn_pcm_bytes=None,
    )

    requests, outputs = _expand_seed_tts_turn_outputs([request], [output])

    assert len(requests) == 1
    assert requests[0].prompt == "target 0"
    assert outputs[0].tts_output_pcm_bytes == pcm


def test_seed_tts_text_dataset_omits_ref_audio(seed_tts_root, mock_tokenizer):
    ds = SeedTTSTextDataset(
        dataset_path=str(seed_tts_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = ds.sample(mock_tokenizer, num_requests=3)
    assert len(requests) == 3
    for req in requests:
        assert isinstance(req, SeedTTSTextSampleRequest)
        assert req.seed_tts_speech_extra is None or "ref_audio" not in (req.seed_tts_speech_extra or {})
        assert req.seed_tts_ref_wav_path == ""
        assert "target text" in req.prompt


# ---------------------------------------------------------------------------
# SeedTTSDesignDataset tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def seed_tts_design_root(tmp_path: Path) -> Path:
    """seed-tts-design directory with 5-field meta.lst entries."""
    locale_dir = tmp_path / "en"
    locale_dir.mkdir()
    meta = "\n".join(
        f"des{i:03d}|||target text {i}|A warm {['female', 'male'][i % 2]} voice with neutral accent." for i in range(5)
    )
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    return tmp_path


def test_seed_tts_design_dataset_has_instructions(seed_tts_design_root, mock_tokenizer):
    ds = SeedTTSDesignDataset(
        dataset_path=str(seed_tts_design_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = ds.sample(mock_tokenizer, num_requests=3)
    assert len(requests) == 3
    for req in requests:
        assert isinstance(req, SeedTTSDesignSampleRequest)
        extra = req.seed_tts_speech_extra or {}
        assert "instructions" in extra
        assert extra["instructions"], "instructions must be non-empty"
        assert extra.get("task_type") == "VoiceDesign"
        assert "ref_audio" not in extra
        assert req.seed_tts_ref_wav_path == ""


def test_seed_tts_design_dataset_rejects_missing_description(seed_tts_design_root, mock_tokenizer):
    """Lines without a voice_description should be skipped."""
    locale_dir = seed_tts_design_root / "en"
    # The bad line has 4 fields, not 5, so will be filtered
    meta = "bad|||target text without description\n" + "\n".join(
        f"ok|||target text {i}|A clear female voice." for i in range(9)
    )
    (locale_dir / "meta.lst").write_text(meta, encoding="utf-8")
    ds = SeedTTSDesignDataset(
        dataset_path=str(seed_tts_design_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = ds.sample(mock_tokenizer, num_requests=10, no_oversample=True)
    assert len(requests) == 9  # since we filter the bad row out and don't oversample
    for req in requests:
        assert isinstance(req, SeedTTSDesignSampleRequest)
        assert req.seed_tts_utterance_id == "ok"


def test_utmos_device_uses_visible_cuda_or_npu_ordinal():
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import _resolve_utmos_device

    # Nightly accuracy keeps the server on the first visible card and scores on the second.
    assert _resolve_utmos_device("cuda:1", "cuda:1", "0,1") == "cuda:1"
    assert _resolve_utmos_device("", "cuda:1", "4,5") == "cuda:1"
    assert _resolve_utmos_device("cuda:0", "npu:1", None) == "cuda:0"
    assert _resolve_utmos_device("", "npu:1", None) == "npu:1"
    assert _resolve_utmos_device("npu:1", "", "") == "npu:1"


def test_utmos_device_rejects_cpu_empty_mask_and_out_of_range():
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import _resolve_utmos_device

    with pytest.raises(ValueError, match="cannot run on"):
        _resolve_utmos_device("cpu", "cuda:1", "0,1")
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        _resolve_utmos_device("cuda:2", "", "0,1")
    with pytest.raises(ValueError, match="set but empty"):
        _resolve_utmos_device("", "", "")
    with pytest.raises(ValueError, match="set but empty"):
        _resolve_utmos_device("cuda:0", "", " , ")


def test_utmos_npu_physical_id_uses_visible_mask():
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import _utmos_npu_physical_id

    # Unset mask: logical ordinal is the physical id. Nightly scores on npu:1.
    assert _utmos_npu_physical_id(1, None) == "1"
    assert _utmos_npu_physical_id(0, None) == "0"
    # Parent sees a subset; the child must be given the physical id, not the logical one.
    assert _utmos_npu_physical_id(1, "6,7") == "7"
    assert _utmos_npu_physical_id(0, "6,7") == "6"


def test_utmos_npu_physical_id_rejects_empty_mask_and_out_of_range():
    from vllm_omni.benchmarks.data_modules.seed_tts_eval import _utmos_npu_physical_id

    with pytest.raises(ValueError, match="set but empty"):
        _utmos_npu_physical_id(0, "")
    with pytest.raises(ValueError, match="set but empty"):
        _utmos_npu_physical_id(0, " , ")
    with pytest.raises(ValueError, match="outside ASCEND_RT_VISIBLE_DEVICES"):
        _utmos_npu_physical_id(1, "4")


def test_utmos_npu_worker_start_isolates_one_card_and_restores_parent(monkeypatch):
    """The child inherits a one-card mask; the parent keeps the eval mask for Whisper."""
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    seen: dict[str, str | None] = {}

    class FakeConn:
        def recv(self):
            return ("ready", "")

        def close(self):
            pass

        def poll(self, timeout):
            assert timeout == 300
            return True

    class FakeProc:
        def start(self):
            seen["during_start"] = os.environ.get("ASCEND_RT_VISIBLE_DEVICES")

        def is_alive(self):
            return True

        def terminate(self):
            seen["terminated"] = "yes"

    class FakeCtx:
        def Pipe(self, duplex=True):
            assert duplex is True
            return FakeConn(), FakeConn()

        def Process(self, target, args, daemon, name):
            assert daemon is True
            assert name == "utmos-npu"
            assert args[0] == "/tmp/utmos.jit"
            seen["target"] = target.__name__
            return FakeProc()

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "6,7")
    monkeypatch.setattr(seed_tts_eval.multiprocessing, "get_context", lambda method: FakeCtx())
    monkeypatch.setattr(seed_tts_eval, "_utmos_npu_conn", None)
    monkeypatch.setattr(seed_tts_eval, "_utmos_npu_proc", None)
    monkeypatch.setattr(seed_tts_eval, "_utmos_model", None)

    seed_tts_eval._start_utmos_npu_worker("/tmp/utmos.jit", "7")

    assert seen["during_start"] == "7"
    assert seen["target"] == "serve_utmos_npu"
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "6,7"
    assert seed_tts_eval._utmos_model is seed_tts_eval._UTMOS_NPU_WORKER
    assert seed_tts_eval._utmos_npu_conn is not None


@pytest.mark.parametrize("failure", ["timeout", "eof"])
def test_utmos_npu_worker_failure_retires_worker(monkeypatch, failure):
    """A reply left in flight would shift every later score, so the worker must not be reused."""
    import numpy as np

    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    sent: list[object] = []

    class FakeConn:
        def send(self, payload):
            sent.append(payload)

        def poll(self, timeout):
            return failure != "timeout"

        def recv(self):
            raise EOFError

    class FakeProc:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            pass

        def terminate(self):
            pass

    monkeypatch.setattr(seed_tts_eval, "_utmos_npu_conn", FakeConn())
    monkeypatch.setattr(seed_tts_eval, "_utmos_npu_proc", FakeProc())
    monkeypatch.setattr(seed_tts_eval, "_utmos_model", seed_tts_eval._UTMOS_NPU_WORKER)
    monkeypatch.setattr(seed_tts_eval, "_utmos_jit_load_failed", False)

    with pytest.raises((TimeoutError, EOFError)):
        seed_tts_eval._utmos_predict_f32_16k(np.ones(16000, dtype=np.float32))

    assert seed_tts_eval._utmos_npu_conn is None
    assert seed_tts_eval._utmos_npu_proc is None
    assert seed_tts_eval._utmos_jit_load_failed is True
    assert seed_tts_eval._utmos_predict_f32_16k(np.ones(16000, dtype=np.float32)) is None
    assert sent[-1] is None


@pytest.mark.parametrize("audio_seconds", [0.1, 61])
def test_seed_tts_whisper_transcribe_passes_attention_mask(monkeypatch, audio_seconds):
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    calls = {}

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        def __call__(self, wav, *, sampling_rate, return_tensors, return_attention_mask=False, **kwargs):
            calls["return_attention_mask"] = return_attention_mask
            calls["processor_kwargs"] = kwargs
            assert sampling_rate == 16000
            assert return_tensors == "pt"
            assert len(wav) > 0
            return types.SimpleNamespace(
                input_features=FakeTensor("features"),
                attention_mask=FakeTensor("mask") if return_attention_mask else None,
            )

        def get_decoder_prompt_ids(self, *, language, task, no_timestamps=True):
            assert language == "english"
            assert task == "transcribe"
            calls["no_timestamps"] = no_timestamps
            return [(1, 2)]

        def batch_decode(self, predicted_ids, *, skip_special_tokens):
            assert skip_special_tokens
            assert predicted_ids == [[42]]
            return ["hello"]

    class FakeModel:
        def generate(self, input_features, **kwargs):
            calls["input_features"] = input_features
            calls["generate_kwargs"] = kwargs
            return [[42]]

    monkeypatch.setattr(seed_tts_eval, "_ensure_en_asr", lambda: None)
    monkeypatch.setattr(seed_tts_eval, "_en_processor", FakeProcessor())
    monkeypatch.setattr(seed_tts_eval, "_en_model", FakeModel())
    monkeypatch.setattr(seed_tts_eval, "_device", "cuda:1")

    text = seed_tts_eval._transcribe_en_f32_16k(np.ones(int(audio_seconds * 16000), dtype=np.float32))

    assert text == "hello"
    assert calls["return_attention_mask"] is True
    assert calls["input_features"].device == "cuda:1"
    assert calls["generate_kwargs"]["attention_mask"].device == "cuda:1"
    assert calls["generate_kwargs"]["forced_decoder_ids"] == [(1, 2)]
    if audio_seconds > 30:
        assert calls["processor_kwargs"] == {"truncation": False, "padding": "longest"}
        assert calls["generate_kwargs"]["return_timestamps"] is True
        assert calls["no_timestamps"] is False
    else:
        assert calls["processor_kwargs"] == {}
        assert "return_timestamps" not in calls["generate_kwargs"]
        assert calls["no_timestamps"] is True


@pytest.mark.parametrize("audio_seconds", [0.1, 61])
def test_seed_tts_transcribe_retry_uses_language_task_kwargs(monkeypatch, audio_seconds):
    """English-only checkpoints: ``get_decoder_prompt_ids`` rejects the
    language/task kwargs, so the retry sends ``language``/``task`` to generate
    instead of ``forced_decoder_ids`` — keeping attention_mask (and, for >30s
    audio, return_timestamps) from the base kwargs."""
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    generate_calls = []

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        def __call__(self, wav, *, sampling_rate, return_tensors, return_attention_mask=False, **kwargs):
            return types.SimpleNamespace(
                input_features=FakeTensor("features"),
                attention_mask=FakeTensor("mask") if return_attention_mask else None,
            )

        def get_decoder_prompt_ids(self, **kwargs):
            raise ValueError("unsupported named arguments")

        def batch_decode(self, predicted_ids, *, skip_special_tokens):
            assert predicted_ids == [[42]]
            return ["hello"]

    class FakeModel:
        def generate(self, input_features, **kwargs):
            generate_calls.append(kwargs)
            return [[42]]

    monkeypatch.setattr(seed_tts_eval, "_ensure_en_asr", lambda: None)
    monkeypatch.setattr(seed_tts_eval, "_en_processor", FakeProcessor())
    monkeypatch.setattr(seed_tts_eval, "_en_model", FakeModel())
    monkeypatch.setattr(seed_tts_eval, "_device", "cuda:1")

    text = seed_tts_eval._transcribe_en_f32_16k(np.ones(int(audio_seconds * 16000), dtype=np.float32))

    assert text == "hello"
    assert len(generate_calls) == 1
    kwargs = generate_calls[0]
    assert "forced_decoder_ids" not in kwargs
    assert kwargs["language"] == "english"
    assert kwargs["task"] == "transcribe"
    assert kwargs["attention_mask"].device == "cuda:1"
    assert kwargs.get("return_timestamps") is (True if audio_seconds > 30 else None)


@pytest.mark.parametrize("audio_seconds", [0.1, 61])
def test_seed_tts_transcribe_retry_falls_back_to_bare_kwargs(monkeypatch, audio_seconds):
    """Checkpoints rejecting both ``forced_decoder_ids`` and the
    ``language``/``task`` kwargs: the second retry calls generate with only the
    base kwargs, attention_mask and return_timestamps still preserved."""
    from vllm_omni.benchmarks.data_modules import seed_tts_eval

    generate_calls = []

    class FakeTensor:
        def __init__(self, name: str):
            self.name = name
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        def __call__(self, wav, *, sampling_rate, return_tensors, return_attention_mask=False, **kwargs):
            return types.SimpleNamespace(
                input_features=FakeTensor("features"),
                attention_mask=FakeTensor("mask") if return_attention_mask else None,
            )

        def get_decoder_prompt_ids(self, **kwargs):
            raise ValueError("unsupported named arguments")

        def batch_decode(self, predicted_ids, *, skip_special_tokens):
            assert predicted_ids == [[42]]
            return ["hello"]

    class FakeModel:
        def generate(self, input_features, **kwargs):
            generate_calls.append(kwargs)
            if len(generate_calls) == 1:
                raise ValueError("language not supported")
            return [[42]]

    monkeypatch.setattr(seed_tts_eval, "_ensure_en_asr", lambda: None)
    monkeypatch.setattr(seed_tts_eval, "_en_processor", FakeProcessor())
    monkeypatch.setattr(seed_tts_eval, "_en_model", FakeModel())
    monkeypatch.setattr(seed_tts_eval, "_device", "cuda:1")

    text = seed_tts_eval._transcribe_en_f32_16k(np.ones(int(audio_seconds * 16000), dtype=np.float32))

    assert text == "hello"
    assert len(generate_calls) == 2
    assert generate_calls[0].get("language") == "english"
    assert generate_calls[0].get("task") == "transcribe"
    bare = generate_calls[1]
    assert "language" not in bare and "task" not in bare and "forced_decoder_ids" not in bare
    assert bare["attention_mask"].device == "cuda:1"
    assert bare.get("return_timestamps") is (True if audio_seconds > 30 else None)
