# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stop the MiniCPM-o audio front end padding every clip out to 30 seconds.

The checkpoint's ``processing_minicpmo.py`` (loaded with ``trust_remote_code``)
calls its Whisper feature extractor once per audio item with
``padding="max_length"``, no ``max_length`` and ``return_attention_mask=True``.
``WhisperFeatureExtractor`` fills the missing length in with ``n_samples`` --
30 seconds -- so a 1-second clip becomes an 80x3000 log-mel, and the processor
then slices it back to the clip's own 100 frames with the attention mask. A
request with 64 interleaved one-second audio items (the Daily-Omni layout)
computes 64x3000 frames on the frontend's CPU to keep 64x100.

The processor lives in the model repo, so the call site cannot be changed here.
Instead the processor's own extractor instance is given a ``__call__`` that
supplies the length the caller left out. Other users of the extractor class
are untouched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from transformers import WhisperFeatureExtractor

_TRIMMED_EXTRACTOR_CLASSES: dict[type, type] = {}


def trim_whisper_max_length_padding(hf_processor: Any) -> None:
    """Pad MiniCPM-o audio clips to their own length instead of to 30 seconds.

    Only calls that ask for ``padding="max_length"`` without a ``max_length``
    and request the attention mask are changed -- the pattern whose caller
    slices the padding off again. They are padded to the longest waveform in
    the call plus ``n_fft``; inputs that already reach ``n_samples`` keep the
    stock 30-second padding and its truncation.

    The kept frames are numerically identical, not merely close, and ``n_fft``
    is sufficient with margin for two separate reasons:

    * Whisper's STFT is centred, so the last frame the mask keeps reads
      ``n_fft // 2`` samples past the end of the waveform. With at least that
      many zeros after it, every kept frame and its reflect padding see the
      same zeros as in the 30-second layout.
    * The dynamic-range floor (``max - 8`` over each clip's own log-mel) is set
      by the whole computed window, and for a near-silent clip with a loud
      transient in its final samples the argmax is the frame at index
      ``mask_sum`` -- the first frame the mask *drops*. Keeping ``hop_length``
      more samples than the waveform still computes that frame.

    So the binding constraint is ``max(hop_length, n_fft // 2)``, and ``n_fft``
    clears it by roughly 2x. Padding to the bare waveform length satisfies
    neither: it reflects real samples into the last frames, moves the floor,
    and drops a frame whenever the length is not a multiple of ``hop_length``.

    Idempotent; a processor without a Whisper-style extractor is left alone.
    """
    extractor = getattr(hf_processor, "audio_processor", None)
    if extractor is None:
        extractor = getattr(hf_processor, "feature_extractor", None)
    if not isinstance(extractor, WhisperFeatureExtractor):
        return
    if getattr(type(extractor), "_omni_trims_max_length", False):
        return
    n_samples = getattr(extractor, "n_samples", None)
    n_fft = getattr(extractor, "n_fft", None)
    if not (isinstance(n_samples, int) and n_samples > 0 and isinstance(n_fft, int) and n_fft > 0):
        return
    # Swap the instance's class rather than assigning ``extractor.__call__``:
    # ``extractor(...)`` looks ``__call__`` up on the type, so an instance
    # attribute would never be called. The subclass keeps ``isinstance`` checks
    # and every other attribute of the original.
    extractor.__class__ = _trimmed_extractor_class(type(extractor))


def _trimmed_extractor_class(base: type) -> type:
    cls = _TRIMMED_EXTRACTOR_CLASSES.get(base)
    if cls is not None:
        return cls

    def call(self, raw_speech, *args, **kwargs):
        if (
            not args
            and kwargs.get("padding") == "max_length"
            and kwargs.get("max_length") is None
            and kwargs.get("return_attention_mask")
        ):
            longest = _longest_waveform(raw_speech)
            if longest is not None and longest + self.n_fft < self.n_samples:
                kwargs["max_length"] = longest + self.n_fft
        return base.__call__(self, raw_speech, *args, **kwargs)

    def reduce(self):
        # The subclass has no importable name, so the reduction names the stock
        # class; _rebuild_extractor re-applies the trim to the copy.
        return _rebuild_extractor, (base, self.__dict__.copy())

    cls = type(
        base.__name__,
        (base,),
        {
            "__call__": call,
            "__reduce__": reduce,
            "__module__": base.__module__,
            "_omni_trims_max_length": True,
        },
    )
    _TRIMMED_EXTRACTOR_CLASSES[base] = cls
    return cls


def _rebuild_extractor(cls: type, state: dict[str, Any]) -> Any:
    """Rebuild a trimmed extractor from the stock class a reduction named."""
    trimmed = _trimmed_extractor_class(cls)
    extractor = trimmed.__new__(trimmed)
    extractor.__dict__.update(state)
    return extractor


def _longest_waveform(raw_speech: Any) -> int | None:
    """Sample count of the longest mono waveform; ``None`` for unrecognised input."""
    if isinstance(raw_speech, np.ndarray):
        return int(raw_speech.shape[-1]) if raw_speech.ndim in (1, 2) else None
    if not isinstance(raw_speech, (list, tuple)) or not raw_speech:
        return None
    if not isinstance(raw_speech[0], (np.ndarray, list, tuple)):
        return len(raw_speech)
    lengths = [np.shape(item) for item in raw_speech]
    if any(len(shape) != 1 for shape in lengths):
        return None
    return max(shape[0] for shape in lengths)
