# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def compare_results(
    tp1_path: str | Path,
    tp2_path: str | Path,
    *,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    tp1 = json.loads(Path(tp1_path).read_text())
    tp2 = json.loads(Path(tp2_path).read_text())
    assert len(tp1["requests"]) == len(tp2["requests"]) == 1
    first = tp1["requests"][0]
    second = tp2["requests"][0]

    assert first["text_token_ids"] == second["text_token_ids"]
    assert first["codec_token_ids"], "TP1 result has no codec token evidence"
    assert second["codec_token_ids"], "TP2 result has no codec token evidence"
    assert first["codec_token_ids"] == second["codec_token_ids"]
    assert first["consumed_units"] == second["consumed_units"]
    assert first["sequence_indices"] == second["sequence_indices"]
    assert first["sample_rate"] == second["sample_rate"] == 24000

    first_audio, first_rate = sf.read(first["wav_path"], dtype="float32")
    second_audio, second_rate = sf.read(second["wav_path"], dtype="float32")
    assert first_rate == second_rate == 24000
    assert first_audio.shape == second_audio.shape
    np.testing.assert_allclose(
        first_audio,
        second_audio,
        rtol=rtol,
        atol=atol,
    )
    return {
        "tp_parity": True,
        "text_tokens": len(first["text_token_ids"]),
        "codec_tokens": len(first["codec_token_ids"]),
        "audio_samples": int(first_audio.size),
        "rtol": rtol,
        "atol": atol,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tp1")
    parser.add_argument("tp2")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()
    print(
        json.dumps(
            compare_results(
                args.tp1,
                args.tp2,
                rtol=args.rtol,
                atol=args.atol,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
