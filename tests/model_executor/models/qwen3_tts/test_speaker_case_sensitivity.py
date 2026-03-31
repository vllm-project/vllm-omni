# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for speaker name case-insensitive lookup in Qwen3 TTS talker.

Verifies the fix for https://github.com/vllm-project/vllm-omni/issues/2304:
The model config stores speaker names in mixed case (e.g. "Ryan", "Vivian"),
but the serving layer normalizes them to lowercase before passing to the
talker. The talker must normalize its own spk_id_map keys to lowercase so
that lookups succeed regardless of the original casing in the config.
"""

import importlib.util
import os

import pytest

# Load normalize_spk_id_map without pulling in the full vllm_omni import chain.
try:
    from vllm_omni.model_executor.models.qwen3_tts.speaker_utils import normalize_spk_id_map
except Exception:
    _UTILS_PATH = os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        os.pardir,
        os.pardir,
        "vllm_omni",
        "model_executor",
        "models",
        "qwen3_tts",
        "speaker_utils.py",
    )
    _spec = importlib.util.spec_from_file_location("speaker_utils", os.path.abspath(_UTILS_PATH))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    normalize_spk_id_map = _mod.normalize_spk_id_map


@pytest.mark.core_model
@pytest.mark.cpu
class TestSpeakerCaseSensitivity:
    """Test that speaker name lookup is case-insensitive."""

    @pytest.mark.parametrize(
        "config_name, query_name",
        [
            ("Ryan", "ryan"),
            ("VIVIAN", "vivian"),
            ("Aiden", "aiden"),
            ("ryan", "ryan"),
            ("RYAN", "ryan"),
            ("RyAn", "ryan"),
        ],
    )
    def test_mixed_case_speaker_lookup(self, config_name: str, query_name: str):
        """Speaker lookup succeeds regardless of config casing."""
        raw_map = {config_name: 42}
        normalized = normalize_spk_id_map(raw_map)
        assert query_name in normalized
        assert normalized[query_name] == 42

    def test_multiple_speakers_normalized(self):
        """All speakers in a multi-entry map are normalized."""
        raw_map = {"Ryan": 0, "Vivian": 1, "AIDEN": 2, "emily": 3}
        normalized = normalize_spk_id_map(raw_map)
        assert normalized == {"ryan": 0, "vivian": 1, "aiden": 2, "emily": 3}

    def test_empty_map(self):
        """Empty or None maps produce empty dict."""
        assert normalize_spk_id_map({}) == {}
        assert normalize_spk_id_map(None) == {}

    def test_unsupported_speaker_not_in_normalized_map(self):
        """Querying a non-existent speaker is not found in normalized map."""
        raw_map = {"Ryan": 0, "Vivian": 1}
        normalized = normalize_spk_id_map(raw_map)
        assert "unknown" not in normalized
