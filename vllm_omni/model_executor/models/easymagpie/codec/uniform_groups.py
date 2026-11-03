# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Group existing codec work by frame count without changing request order.

Each request retains its vLLM-owned cache page and initial-state flag. Grouping
only changes the matrix shape used for this already-scheduled codec invocation;
no frames are padded, delayed, discarded or transferred between requests.
"""

from __future__ import annotations

import copy
import os
from collections import Counter

import torch
from vllm.forward_context import get_forward_context, set_forward_context

from vllm_omni.model_executor.models.easymagpie.buffer_leases import BufferLeasePool, leased_buffers_enabled
from vllm_omni.model_executor.models.easymagpie.codec.packed import CodecStateLayer


def uniform_groups_enabled():
    return os.environ.get("EASYMAGPIE_VLLM_CODEC_UNIFORM_GROUPS", "0").lower() in ("1", "true", "yes", "on")


class CodecUniformGroups:
    def __init__(self, codec, vllm_config, graphs=None):
        self.codec = codec
        self.vllm_config = vllm_config
        self.graphs = graphs
        self.layers = tuple(module for module in codec.modules() if isinstance(module, CodecStateLayer))
        self.invocations = 0
        self.group_calls = 0
        self.frames_histogram = Counter()
        self.fallbacks = Counter()
        self._scratch_pool = BufferLeasePool(24) if leased_buffers_enabled() else None

    def status(self):
        return {
            "enabled": True,
            "invocations": self.invocations,
            "group_calls": self.group_calls,
            "frames_histogram": dict(self.frames_histogram),
            "fallbacks_by_reason": dict(self.fallbacks),
            "preserves_request_order": True,
            "pads_frames": False,
            "scratch_buffers": self._scratch_pool.status() if self._scratch_pool is not None else None,
        }

    def _decode(self, codes):
        return self.codec(codes) if self.graphs is None else self.graphs.run(codes)

    @staticmethod
    def _compatible(metadata, frame_counts):
        if metadata is None or metadata.num_decode_tokens != metadata.num_decodes:
            return False
        if metadata.num_reqs != len(frame_counts) or metadata.num_decodes + metadata.num_prefills != len(frame_counts):
            return False
        if any(frames != 1 for frames in frame_counts[: metadata.num_decodes]):
            return False
        return sum(frame_counts[metadata.num_decodes :]) == metadata.num_prefill_tokens

    @staticmethod
    def _group_metadata(metadata, members, frames, device, pool=None, leases=None):
        def scratch(count, dtype):
            lease = pool.acquire(count, dtype=dtype, device=device)
            leases.append(lease)
            return lease.tensor

        parts, initials = [], []
        if metadata.num_decodes:
            indices = metadata.state_indices_tensor_d
            if indices.ndim == 2:
                indices = indices[:, 0]
            parts.append(indices)
            initials.append(
                torch.ones(metadata.num_decodes, dtype=torch.bool, device=device)
                if pool is None
                else scratch(metadata.num_decodes, torch.bool).fill_(True)
            )
        if metadata.num_prefills:
            parts.append(metadata.state_indices_tensor_p)
            initials.append(metadata.has_initial_states_p)
        if pool is None:
            select = torch.tensor(members, dtype=torch.long, device=device)
            routing = parts[0] if len(parts) == 1 else torch.cat(parts)
            has_initial = initials[0] if len(initials) == 1 else torch.cat(initials)
        else:
            select = scratch(len(members), torch.long)
            select.copy_(torch.tensor(members, dtype=torch.long))
            routing = (
                parts[0]
                if len(parts) == 1
                else torch.cat(parts, out=scratch(sum(x.numel() for x in parts), parts[0].dtype))
            )
            has_initial = (
                initials[0]
                if len(initials) == 1
                else torch.cat(initials, out=scratch(sum(x.numel() for x in initials), torch.bool))
            )
        grouped = copy.copy(metadata)
        grouped.num_prefills = grouped.num_reqs = len(members)
        grouped.num_prefill_tokens = len(members) * frames
        grouped.num_decodes = grouped.num_decode_tokens = 0
        if pool is None:
            grouped.state_indices_tensor_p = routing.index_select(0, select)
            grouped.has_initial_states_p = has_initial.index_select(0, select)
            grouped.query_start_loc_p = torch.arange(len(members) + 1, dtype=torch.int32, device=device) * frames
        else:
            grouped.state_indices_tensor_p = torch.index_select(
                routing, 0, select, out=scratch(len(members), routing.dtype)
            )
            grouped.has_initial_states_p = torch.index_select(
                has_initial, 0, select, out=scratch(len(members), torch.bool)
            )
            query = scratch(len(members) + 1, torch.int32)
            torch.arange(len(members) + 1, out=query)
            grouped.query_start_loc_p = query.mul_(frames)
        grouped.state_indices_tensor_d = None
        grouped.query_start_loc_d = None
        grouped.codec_uniform = grouped.codec_prefill_uniform = True
        grouped.codec_max_query_len = frames
        return grouped

    def run(self, codes, frame_counts):
        raw = get_forward_context().attn_metadata
        if not codes.is_cuda or not isinstance(raw, dict) or not frame_counts or any(n <= 0 for n in frame_counts):
            self.fallbacks["profile_cpu_or_empty"] += 1
            return self._decode(codes)
        if torch.cuda.is_current_stream_capturing():
            self.fallbacks["outer_capture"] += 1
            return self._decode(codes)
        if sum(frame_counts) != codes.shape[0] or any(
            not self._compatible(raw.get(layer.prefix), frame_counts) for layer in self.layers
        ):
            self.fallbacks["metadata_mismatch"] += 1
            return self._decode(codes)
        groups = {}
        offsets = [0]
        for index, frames in enumerate(frame_counts):
            groups.setdefault(frames, []).append(index)
            offsets.append(offsets[-1] + frames)
        self.invocations += 1
        outputs = [None] * len(frame_counts)
        for frames, members in groups.items():
            # Preserve the zero-copy fast path for an already-uniform prefill.
            existing = raw[self.layers[0].prefix]
            if len(groups) == 1 and existing.num_decodes == 0 and getattr(existing, "codec_uniform", False):
                self.group_calls += 1
                self.frames_histogram[frames] += 1
                return self._decode(codes)
            leases = []
            try:
                segments = [codes[offsets[i] : offsets[i + 1]] for i in members]
                if self._scratch_pool is None:
                    grouped_codes = torch.cat(segments, dim=0)
                else:
                    lease = self._scratch_pool.acquire(
                        len(members) * frames * codes.shape[1], dtype=codes.dtype, device=codes.device
                    )
                    leases.append(lease)
                    grouped_codes = torch.cat(
                        segments, dim=0, out=lease.tensor.view(len(members) * frames, codes.shape[1])
                    )
                replacements = {}
                grouped_raw = {}
                for layer in self.layers:
                    original = raw[layer.prefix]
                    identity = id(original)
                    if identity not in replacements:
                        replacements[identity] = self._group_metadata(
                            original, members, frames, codes.device, self._scratch_pool, leases
                        )
                    grouped_raw[layer.prefix] = replacements[identity]
                with set_forward_context(grouped_raw, self.vllm_config):
                    pcm = self._decode(grouped_codes)
            finally:
                if leases:
                    ready = torch.cuda.Event()
                    ready.record(torch.cuda.current_stream(codes.device))
                    for lease in leases:
                        lease.ready_after(ready)
                        lease.release()
            samples = frames * self.codec.config.samples_per_frame
            for row, member in enumerate(members):
                outputs[member] = pcm[row * samples : (row + 1) * samples]
            self.group_calls += 1
            self.frames_histogram[frames] += 1
        return torch.cat(outputs)
