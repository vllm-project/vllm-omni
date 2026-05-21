# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
"""MiniMind-O multimodal injection helpers (aligned with HF model_omni)."""

from __future__ import annotations

import torch


def inject_audio_features(
    tokens: torch.Tensor,
    hidden_states: torch.Tensor,
    audio_feats: list[torch.Tensor | None],
    *,
    audio_marker: int,
) -> torch.Tensor:
    if audio_feats is None or not audio_marker:
        return hidden_states
    out = []
    for b in range(hidden_states.size(0)):
        hb = hidden_states[b]
        seq = tokens[b].tolist()
        af = audio_feats[b] if b < len(audio_feats) else None
        i = 0
        while i < len(seq):
            if seq[i] == audio_marker:
                start = i
                while i < len(seq) and seq[i] == audio_marker:
                    i += 1
                if af is not None:
                    inject_len = min(af.size(0), i - start)
                    hb = torch.cat((hb[:start], af[:inject_len], hb[start + inject_len :]), dim=0)
                    af = None
            else:
                i += 1
        out.append(hb)
    return torch.stack(out)


def inject_vision_features(
    tokens: torch.Tensor,
    hidden_states: torch.Tensor,
    vision_tensors: torch.Tensor | None,
    *,
    image_marker: int,
    seqlen: int | None = None,
) -> torch.Tensor:
    if vision_tensors is None or not image_marker:
        return hidden_states
    vf = vision_tensors
    if vf.dim() == 3:
        vf = vf.unsqueeze(1)
    out = []
    for b in range(hidden_states.size(0)):
        hb = hidden_states[b]
        seq = tokens[b].tolist()
        k = 0
        i = 0
        while i < len(seq):
            if seq[i] == image_marker:
                start = i
                while i < len(seq) and seq[i] == image_marker:
                    i += 1
                if k < vf.size(1):
                    chunk = vf[b][k][: i - start]
                    hb = torch.cat((hb[:start], chunk, hb[i:]), dim=0)
                    k += 1
            else:
                i += 1
        if seqlen is not None:
            hb = hb[:seqlen]
        out.append(hb)
    return torch.stack(out)
