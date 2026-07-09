"""Tests for the Audex TTA caption dataset and its benchmark registration.

vllm stubs are installed by tests/benchmarks/conftest.py before collection.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Load the data modules directly (bypasses vllm_omni.__init__ heavy imports).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name: str, rel_path: str):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _REPO_ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("vllm_omni.benchmarks.data_modules.seed_tts_dataset", "vllm_omni/benchmarks/data_modules/seed_tts_dataset.py")
_tta_mod = _load(
    "vllm_omni.benchmarks.data_modules.audex_tta_dataset", "vllm_omni/benchmarks/data_modules/audex_tta_dataset.py"
)
AudexTTADataset = _tta_mod.AudexTTADataset


@pytest.fixture()
def tta_root(tmp_path: Path) -> Path:
    locale_dir = tmp_path / "en"
    locale_dir.mkdir()
    captions = "\n".join(f"cap{i:03d}|ambient caption number {i}" for i in range(5))
    (locale_dir / "captions.lst").write_text(captions + "\n# comment\nmalformed-line\n", encoding="utf-8")
    return tmp_path


class _WordTokenizer:
    all_special_ids: list[int] = []
    all_special_tokens: list[str] = []
    vocab_size = 1

    def encode(self, text: str, **kw) -> list[int]:
        return [0] * len(text.split())

    def get_vocab(self):
        return {"<pad>": 0}

    def __len__(self) -> int:
        return 1


@pytest.fixture()
def mock_tokenizer():
    return _WordTokenizer()


def test_caption_rows_become_prompt_requests(tta_root, mock_tokenizer):
    dataset = AudexTTADataset(
        dataset_path=str(tta_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    requests = dataset.sample(tokenizer=mock_tokenizer, num_requests=3)

    assert len(requests) == 3
    for i, req in enumerate(requests):
        assert req.prompt == f"ambient caption number {i}"
        assert req.seed_tts_utterance_id == f"cap{i:03d}"
        # Non-speech output: WER/SIM must be skipped by the eval pipeline.
        assert req.seed_tts_ref_wav_path == ""
        assert req.expected_output_len == AudexTTADataset.DEFAULT_OUTPUT_LEN


def test_malformed_and_comment_lines_skipped(tta_root, mock_tokenizer):
    dataset = AudexTTADataset(
        dataset_path=str(tta_root),
        random_seed=0,
        locale="en",
        disable_shuffle=True,
    )
    assert len(dataset.data) == 5


def test_missing_captions_file_raises(tmp_path, mock_tokenizer):
    (tmp_path / "en").mkdir()
    with pytest.raises(FileNotFoundError, match="captions"):
        AudexTTADataset(dataset_path=str(tmp_path), random_seed=0, locale="en", disable_shuffle=True)


def test_registered_in_bench_patch_source():
    """The `vllm bench serve` patch must dispatch --dataset-name audex-tta."""
    patch_src = (_REPO_ROOT / "vllm_omni" / "benchmarks" / "patch" / "patch.py").read_text()
    assert '"audex-tta"' in patch_src
    assert '"audex-tta": AudexTTADataset' in patch_src
    # The seed-tts branch must NOT force its 2048 default onto audex-tta:
    # out_len stays None so sample() applies DEFAULT_OUTPUT_LEN (4200).
    assert "DatasetCls is not AudexTTADataset" in patch_src
