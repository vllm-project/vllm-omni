# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Load the balacoon UTMOS TorchScript graph and score one waveform.

The checkpoint traces ``prim::Constant[value="cuda:0"]`` into the forward.
Rewriting those constants onto ``cuda:N`` works. On Ascend the same rewrite
onto ``npu:N`` only executes when that card is the process's ``npu:0``; a
non-zero ordinal returns garbage view sizes and faults the vector core.
NPU scoring therefore runs in a child whose ``ASCEND_RT_VISIBLE_DEVICES``
contains only the requested physical card, then retargets onto ``npu:0``.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np


def retarget_utmos_jit_devices(module: Any, device: str) -> int:
    """Rewrite baked ``cuda:0`` device constants onto ``device``."""

    def _walk_graph(graph: Any) -> int:
        changed = 0
        for node in graph.nodes():
            if node.kind() == "prim::Constant" and "cuda:0" in str(node):
                node.s_("value", device)
                changed += 1
            for block in node.blocks():
                changed += _walk_graph(block)
        return changed

    def _walk_module(current: Any) -> int:
        total = 0
        try:
            methods = list(current._c._method_names())
        except Exception:
            methods = []
        for method in methods:
            try:
                graph = current._c._get_method(method).graph
            except Exception:
                continue
            total += _walk_graph(graph)
        for child in current.children():
            total += _walk_module(child)
        return total

    return _walk_module(module)


def load_utmos_jit(path: str, device: str) -> Any:
    """Load UTMOS on ``device`` and retarget the traced ``cuda:0`` constants."""
    if device.startswith("npu"):
        import torch_npu  # noqa: F401
    import torch

    model = torch.jit.load(path, map_location=device)
    model.eval()
    rewritten = retarget_utmos_jit_devices(model, device)
    if rewritten == 0:
        raise RuntimeError(f"UTMOS graph has no cuda:0 device constants to retarget onto {device}.")
    warmup = torch.zeros(1, 16000, dtype=torch.float32, device=device)
    with torch.no_grad():
        model(warmup)
    return model


def serve_utmos_npu(jit_path: str, conn: Any) -> None:
    """Score waveforms on this process's only NPU, which is ``npu:0``.

    The parent sets ``ASCEND_RT_VISIBLE_DEVICES`` to one physical id before
    starting this process, so the logical ``npu:1`` eval card is ``npu:0`` here.
    """
    try:
        model = load_utmos_jit(jit_path, "npu:0")
    except Exception as exc:
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", ""))
    import torch

    while True:
        wav = conn.recv()
        if wav is None:
            return
        try:
            audio = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32).copy()).unsqueeze(0)
            audio = audio.to(device="npu:0", dtype=torch.float32)
            with torch.no_grad():
                out = model(audio)
            value = float(out.reshape(-1)[0].item())
            conn.send(value if math.isfinite(value) else None)
        except Exception as exc:
            print(f"UTMOS NPU score failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            conn.send(None)
