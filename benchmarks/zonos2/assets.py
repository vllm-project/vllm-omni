# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit one-time preparation of pinned evaluation assets in an owned folder."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

UTMOS_SHA = "3080923f49ee69eb81ad875491eef94e533bf329d70e0cc37fa23745c464669d"


def main():
    import whisper

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    whisper_url = whisper._MODELS["large-v3"]
    jobs = [
        ("large-v3.pt", whisper_url, whisper_url.split("/")[-2]),
        ("utmos.jit", "https://hf-mirror.com/balacoon/utmos/resolve/main/utmos.jit", UTMOS_SHA),
    ]
    rows = []
    for name, url, expected in jobs:
        path = args.out / name
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as target:
                while chunk := response.read(4 * 1024 * 1024):
                    target.write(chunk)
            partial.rename(path)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"Evaluation asset SHA mismatch: {path}")
        rows.append(
            {
                "name": name,
                "source": url,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    (args.out / "manifest.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
