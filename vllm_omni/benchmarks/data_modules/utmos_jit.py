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
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch

_INT16_INPUT_SCALE = 32768.0


def _detect_utmos_input_scale(model: torch.jit.ScriptModule) -> float:
    """Recognize int16 normalization on the waveform, through optional ``to`` casts.

    Only follow a single-use path from the sole waveform argument. A division
    of a feature/output, or a graph that also consumes the unnormalized input,
    does not establish an int16 input contract and must not rescale the input.
    """
    import torch

    inputs = list(model.graph.inputs())
    if len(inputs) != 2 or inputs[1].type().kind() != "TensorType":
        return 1.0
    waveform = inputs[1]
    while True:
        uses = list(waveform.uses())
        if len(uses) != 1 or uses[0].offset != 0:
            return 1.0
        node = uses[0].user
        if node.kind() == "aten::to":
            waveform = node.output()
            continue
        operands = list(node.inputs())
        if node.kind() != "aten::div" or len(operands) != 2:
            return 1.0
        if operands[1].node().kind() != "prim::Constant":
            return 1.0
        divisor = operands[1].toIValue()
        if isinstance(divisor, torch.Tensor):
            if divisor.numel() != 1:
                return 1.0
            divisor = divisor.item()
        return _INT16_INPUT_SCALE if divisor == _INT16_INPUT_SCALE else 1.0


class _UTMOSScorer:
    """Accept float32 [-1, 1] audio while preserving the archive's input scale."""

    def __init__(self, model: torch.jit.ScriptModule) -> None:
        self._model = model
        self._input_scale = _detect_utmos_input_scale(model)

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        if self._input_scale != 1.0:
            # Out of place: a caller may reuse its waveform for WER/SIM.
            audio = audio * self._input_scale
        return self._model(audio)


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


def load_utmos_jit(path: str, device: str) -> _UTMOSScorer:
    """Load a scorer accepting float32 [-1, 1] audio on ``device``.

    Retarget baked devices as before, then adapt amplitude once per call if
    the archive normalizes its waveform input by 32768 (``balacoon/utmos``).
    Both the in-process scorer and the NPU worker use this callable.
    """
    if device.startswith("npu"):
        import torch_npu  # noqa: F401
    import torch

    model = torch.jit.load(path, map_location=device)
    model.eval()
    rewritten = retarget_utmos_jit_devices(model, device)
    if rewritten == 0:
        raise RuntimeError(f"UTMOS graph has no cuda:0 device constants to retarget onto {device}.")
    scorer = _UTMOSScorer(model)
    warmup = torch.zeros(1, 16000, dtype=torch.float32, device=device)
    with torch.no_grad():
        scorer(warmup)
    return scorer


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
