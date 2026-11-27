# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Portions derived from https://github.com/step-out/Duplex-MPE
# (Copyright (c) 2026 MPeval authors, MIT License; see benchmarks/duplex_mpe/LICENSE).
"""Module entrypoint for `python -m benchmarks.duplex_mpe`."""

from __future__ import annotations

from benchmarks.duplex_mpe.evaluate import main

if __name__ == "__main__":
    raise SystemExit(main())
