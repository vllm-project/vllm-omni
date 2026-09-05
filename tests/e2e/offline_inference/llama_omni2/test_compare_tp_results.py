# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import numpy as np
import pytest
import soundfile as sf

from tests.e2e.offline_inference.llama_omni2.compare_tp_results import (
    compare_results,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_result(tmp_path, name: str, codec_token_ids: list[int]):
    wav_path = tmp_path / f"{name}.wav"
    sf.write(wav_path, np.array([0.1, -0.1], dtype=np.float32), 24000)
    result_path = tmp_path / f"{name}.json"
    result_path.write_text(
        json.dumps(
            {
                "requests": [
                    {
                        "text_token_ids": [1, 2],
                        "codec_token_ids": codec_token_ids,
                        "consumed_units": [2],
                        "sequence_indices": [0],
                        "sample_rate": 24000,
                        "wav_path": str(wav_path),
                    }
                ]
            }
        )
    )
    return result_path


def test_compare_results_rejects_codec_token_mismatch(tmp_path):
    tp1 = _write_result(tmp_path, "tp1", [10, 20])
    tp2 = _write_result(tmp_path, "tp2", [10, 21])

    with pytest.raises(AssertionError):
        compare_results(tp1, tp2, rtol=1e-3, atol=1e-4)


def test_compare_results_rejects_missing_codec_evidence(tmp_path):
    tp1 = _write_result(tmp_path, "tp1", [])
    tp2 = _write_result(tmp_path, "tp2", [])

    with pytest.raises(AssertionError, match="codec token"):
        compare_results(tp1, tp2, rtol=1e-3, atol=1e-4)
