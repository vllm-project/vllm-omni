# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bounded CPU integer reductions for codec payload validity.

The parser still owns shape validation, packing, device transfer and ordering.
Unsupported payloads keep the ordinary per-request validity path. The NumPy
view and reductions are temporary; no payload, request or tensor is retained.
"""

import os

import numpy as np
import torch

ENV = "EASYMAGPIE_VLLM_CODEC_BATCHED_PAYLOAD_VALIDITY"
MIN_BATCH = 8
MAX_BATCH = 512
MAX_FRAMES = 4096


def flag():
    value = os.environ.get(ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1")
    return value


def _config(model):
    config = model.config
    values = tuple(
        getattr(config, name)
        for name in (
            "num_stacked_codebooks",
            "frame_stacking_factor",
            "codebook_size",
            "samples_per_frame",
            "samples_per_codec_frame",
        )
    )
    if not all(type(value) is int and 0 < value <= 1048576 for value in values):
        return None
    q, stacking, _, samples, subframe = values
    return values if q % stacking == 0 and samples == stacking * subframe else None


def payload_batch_reason(model, infos, device, spans):
    """Metadata only; declining never converts, reads or changes a payload."""
    if type(infos) is not list or not MIN_BATCH <= len(infos) <= MAX_BATCH:
        return "batch_size"
    if type(device) is not torch.device or device.type not in ("cpu", "cuda"):
        return "destination_device"
    config = _config(model)
    if config is None:
        return "configuration"
    q = config[0]
    if spans is not None and (type(spans) is not list or len(spans) != len(infos)):
        return "request_spans"
    count = total = 0
    for index, info in enumerate(infos):
        scheduled = None
        if spans is not None:
            span = spans[index]
            if type(span) is not tuple or len(span) != 2 or not all(type(x) is int for x in span):
                return "request_spans"
            scheduled = span[1] - span[0]
            if scheduled == 0:
                continue
        if type(info) is not dict or type(info.get("codes")) is not dict:
            return "payload_type"
        value = info["codes"].get("audio")
        if type(value) is not torch.Tensor or value.device.type != "cpu" or value.dtype != torch.long:
            return "payload_type"
        if value.layout != torch.strided or value.is_conj() or value.is_neg():
            return "payload_layout"
        if value.ndim == 2 and value.shape[1] == q:
            frames = value.shape[0]
        elif value.ndim == 1 and value.numel() % q == 0:
            frames = value.numel() // q
        else:
            return "payload_shape"
        if scheduled is not None and frames != scheduled:
            return "scheduled_frames"
        count += 1
        total += frames
    if count < MIN_BATCH or not 0 < total <= MAX_FRAMES:
        return "active_batch_size"
    return None


def batched_valid_sample_counts(model, codes, frame_counts):
    """Exact native first-padding/terminal-subframe rules on one CPU packing.

    The NumPy array is a temporary view of the same owned contiguous CPU cat.
    Only nonempty segments enter reduceat; empty payloads retain a zero result.
    """
    q, stacking, codebook_size, samples_per_frame, samples_per_subframe = _config(model)
    values = codes.numpy()
    counts = np.asarray(frame_counts, dtype=np.int64)
    ends = counts.cumsum()
    starts = ends - counts
    present = np.flatnonzero(counts)
    result = np.zeros(len(frame_counts), dtype=np.int64)
    if not len(present):
        return result.tolist()
    positions = np.arange(values.shape[0], dtype=np.int64)
    padding = (values < 0).all(axis=1)
    first_padding = np.minimum.reduceat(np.where(padding, positions, values.shape[0]), starts[present])
    valid_frames = np.minimum(first_padding, ends[present]) - starts[present]
    last_rows = starts[present] + np.maximum(valid_frames - 1, 0)
    control = (values[last_rows].reshape(len(present), q // stacking, stacking) >= codebook_size).any(axis=1)
    first_control = np.where(control, np.arange(stacking), stacking).min(axis=1)
    samples = np.where(
        first_control < stacking,
        (valid_frames - 1) * samples_per_frame + first_control * samples_per_subframe,
        valid_frames * samples_per_frame,
    )
    result[present] = np.where(valid_frames == 0, 0, samples)
    return result.tolist()
