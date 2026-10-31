# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import torch

from vllm_omni.diffusion.models.minimax_h3.packed_sequence import _axis_from_sqrt_area, _video_t_grid


def phase_aligned_frames(read_frames, query_frames, window_start, sink_frames=1):
    if read_frames >= window_start:
        return tuple(range(read_frames + query_frames))
    if not 0 <= sink_frames <= read_frames:
        raise ValueError("A detached H3 history must include its clean sink")
    first = window_start - (read_frames - sink_frames)
    shift = 5 * ((first - sink_frames) // 5)
    return tuple(range(sink_frames)) + tuple(frame - shift for frame in range(first, window_start + query_frames))


def video_positions(frames, height, width, patch_size, text_tokens, device):
    _, ph, pw = patch_size
    area = math.sqrt(height * width)
    h = _axis_from_sqrt_area(height, ph, area)
    w = _axis_from_sqrt_area(width, pw, area)
    hh, ww = torch.meshgrid(h, w, indexing="ij")
    spatial = torch.stack((hh.flatten(), ww.flatten()), dim=-1)
    selected = torch.tensor(frames, dtype=torch.long, device="cpu")
    times = _video_t_grid(max(frames) + 1, float(text_tokens)).index_select(0, selected)
    positions = torch.empty(len(frames), spatial.shape[0], 3, dtype=torch.float64, device="cpu")
    positions[..., 0] = times[:, None]
    positions[..., 1:] = spatial[None]
    return positions.flatten(0, 1).to(device)


def output_frames(chunks):
    return sum((1, 4, 4, 4, 4)[index % 5] for index in range(chunks))
