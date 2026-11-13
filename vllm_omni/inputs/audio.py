# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Preserve Omni's audio waveforms across upstream decoder default changes."""

from io import BytesIO
from pathlib import Path

import numpy as np
import numpy.typing as npt
from vllm.multimodal.media import audio as audio_io

_UPSTREAM_LOAD_AUDIO = audio_io.load_audio


def load_audio_soundfile_first(
    path: BytesIO | Path | str,
    *,
    sr: float | None = 22050,
    mono: bool = True,
    max_duration_s: float | None = None,
    max_decode_bytes: int | None = None,
    backend: str | None = None,
) -> tuple[npt.NDArray[np.float32], float]:
    """Keep the pre-0.32 auto preference without restricting input formats.

    vLLM 0.32 prefers torchcodec, whose different decoded samples change
    Qwen2.5-Omni's BF16 audio features and mixed-media answers on L4.
    Explicit backend choices retain upstream behavior. Unsupported soundfile
    formats and missing dependencies use upstream's remaining decoder chain;
    recognized corrupt streams and decode-limit errors must still fail.
    """
    if backend in (None, "auto"):
        if isinstance(path, BytesIO):
            path.seek(0)
        try:
            return audio_io.load_audio_soundfile(
                path,
                sr=sr,
                mono=mono,
                max_duration_s=max_duration_s,
                max_decode_bytes=max_decode_bytes,
            )
        except ImportError:
            pass
        except audio_io.soundfile.LibsndfileError as error:
            if error.code not in audio_io._BAD_SF_CODES:
                raise

    return _UPSTREAM_LOAD_AUDIO(
        path,
        sr=sr,
        mono=mono,
        max_duration_s=max_duration_s,
        max_decode_bytes=max_decode_bytes,
        backend=backend,
    )


def apply_soundfile_first_audio_decode_patch() -> None:
    # The original loader is captured once in this module; repeated patch
    # installation (including reloading vllm_omni.patch) never nests wrappers.
    audio_io.load_audio = load_audio_soundfile_first
