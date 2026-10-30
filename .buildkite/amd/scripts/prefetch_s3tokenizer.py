#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Download the CosyVoice3 s3tokenizer checkpoint before the engine starts.

CosyVoice3 calls ``s3tokenizer.load_model("speech_tokenizer_v3_25hz")`` while
the LLM stage initializes. On a cold cache that is a ~925 MiB download from
modelscope.cn, and on a slow mirror it alone can exceed ``--stage-init-timeout``
and fail the whole test. Fetching the checkpoint up front keeps the download
outside the engine startup deadline and makes a mirror outage show up here,
as a clear download failure, instead of as an opaque stage-init timeout.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence

DEFAULT_MODEL = "speech_tokenizer_v3_25hz"
DEFAULT_ATTEMPTS = 3
RETRY_DELAY_S = 10.0


def prefetch(
    name: str = DEFAULT_MODEL,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    load_model: Callable[[str], object] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Load ``name`` through ``s3tokenizer`` so its checkpoint lands in the cache.

    ``s3tokenizer`` verifies the checkpoint SHA-256 and re-downloads a corrupt
    or partial file on the next call, so simply retrying ``load_model`` is
    enough to recover from an interrupted transfer.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    if load_model is None:
        import s3tokenizer

        load_model = s3tokenizer.load_model

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        try:
            load_model(name)
        except Exception as exc:  # noqa: BLE001 - surface any download failure after retries
            last_error = exc
            print(
                f"s3tokenizer prefetch of {name} failed on attempt {attempt}/{attempts}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            if attempt < attempts:
                sleep(RETRY_DELAY_S)
            continue
        print(
            f"s3tokenizer checkpoint {name} ready in {time.monotonic() - started:.1f}s",
            flush=True,
        )
        return

    raise RuntimeError(
        f"Could not prefetch the s3tokenizer checkpoint {name} after {attempts} attempts"
    ) from last_error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="s3tokenizer model name to prefetch")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS, help="download attempts before failing")
    args = parser.parse_args(argv)
    prefetch(args.model, attempts=args.attempts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
