"""Tests for SeedTTSTextDataset, SeedTTSTextSampleRequest, SeedTTSDesignDataset,
and SeedTTSDesignSampleRequest.

vllm stubs are installed by tests/benchmarks/conftest.py before collection.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Load the data module directly (bypasses vllm_omni.__init__ heavy imports).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = _REPO_ROOT / "vllm_omni" / "benchmarks" / "data_modules" / "seed_tts_dataset.py"
_MODULE_NAME = "vllm_omni.benchmarks.data_modules.seed_tts_dataset"

if _MODULE_NAME not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MODULE_NAME] = _mod
    _spec.loader.exec_module(_mod)

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import (  # noqa: E402
    SeedTTSDesignDataset,
    SeedTTSDesignSampleRequest,
    SeedTTSTextDataset,
    SeedTTSTextSampleRequest,
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_seed_tts_text_dataset_omits_ref_audio(seed_tts_root, mocker):
    ds = SeedTTSTextDataset(
        dataset_path=str(seed_tts_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    tokenizer = mocker.MagicMock()
    tokenizer.encode = lambda text, **kw: [0] * len(text.split())
    requests = ds.sample(tokenizer, num_requests=3)
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


def test_seed_tts_design_dataset_has_instructions(seed_tts_design_root, mocker):
    ds = SeedTTSDesignDataset(
        dataset_path=str(seed_tts_design_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    tokenizer = mocker.MagicMock()
    tokenizer.encode = lambda text, **kw: [0] * len(text.split())
    requests = ds.sample(tokenizer, num_requests=3)
    assert len(requests) == 3
    for req in requests:
        assert isinstance(req, SeedTTSDesignSampleRequest)
        extra = req.seed_tts_speech_extra or {}
        assert "instructions" in extra
        assert extra["instructions"], "instructions must be non-empty"
        assert extra.get("task_type") == "VoiceDesign"
        assert "ref_audio" not in extra
        assert req.seed_tts_ref_wav_path == ""


def test_seed_tts_design_dataset_rejects_missing_description(seed_tts_design_root, mocker):
    """Lines without a voice_description should be skipped."""
    locale_dir = seed_tts_design_root / "en"
    (locale_dir / "meta.lst").write_text(
        "bad001|||target text without description\nok001|||good target|A clear female voice.\n",
        encoding="utf-8",
    )
    ds = SeedTTSDesignDataset(
        dataset_path=str(seed_tts_design_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    tokenizer = mocker.MagicMock()
    tokenizer.encode = lambda text, **kw: [0] * len(text.split())
    requests = ds.sample(tokenizer, num_requests=10)
    assert len(requests) == 1  # only the valid row


def test_attach_sets_seed_tts_row_even_without_extra_body():
    """seed_tts_row=True must be set for SeedTTSTextSampleRequest (no extra body)."""
    from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSTextSampleRequest

    req = SeedTTSTextSampleRequest(
        prompt="hello world",
        prompt_len=2,
        expected_output_len=100,
        multi_modal_data=None,
        request_id="test-0",
        seed_tts_speech_extra=None,
        seed_tts_ref_wav_path="",
    )
    assert req.seed_tts_speech_extra is None
    assert req.seed_tts_ref_wav_path == ""
    # The fix ensures that even with speech_extra=None, the function
    # sets seed_tts_row=True. We verify the source code has the fix.
    import inspect

    import vllm_omni.benchmarks.patch.patch as patch_mod

    src = inspect.getsource(patch_mod._attach_seed_tts_to_request_func_input)
    # seed_tts_row must be set BEFORE the 'if not ex: return' check
    row_pos = src.index("seed_tts_row")
    not_ex_pos = src.index("if not ex:")
    assert row_pos < not_ex_pos, "seed_tts_row must be set before 'if not ex: return'"
