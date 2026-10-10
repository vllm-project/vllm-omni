# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("trace", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
rows = []
for path in a.trace.glob("rank-*.jsonl"):
    rows.extend(json.loads(line) for line in path.read_text().splitlines())
result = {}
diagnostics = {}
for stage in sorted({r["pp_rank"] for r in rows}):
    data = [
        r
        for r in rows
        if r["pp_rank"] == stage
        and r["kind"] == "calibration"
        and r.get("request_name") not in ("warmup", "engine-warmup", None)
    ]
    x = np.array([r["input_distance"] for r in data])
    y = np.array([r["residual_distance"] for r in data])
    assert len(x) > 100 and np.isfinite(x).all() and np.isfinite(y).all()
    # Fit only calibration data. Clip polynomial predictions to nonnegative values
    # at runtime in the Wan-specific tracing hook to avoid negative accumulation.
    coefficients = np.polynomial.Polynomial.fit(x, y, 4).convert().coef[::-1]
    result[str(stage)] = coefficients.tolist()
    diagnostics[str(stage)] = {
        "pairs": len(x),
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "mae": float(np.abs(np.polyval(coefficients, x) - y).mean()),
    }
a.output.write_text(json.dumps(result, indent=2))
a.output.with_suffix(".fit.json").write_text(json.dumps(diagnostics, indent=2))
