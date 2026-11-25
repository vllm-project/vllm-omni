# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measured gates for the automatic Ulysses a2a transports.

Keyed by compute capability (``"major.minor"``) and Ulysses world size, like vLLM's ``CUSTOM_ALL_REDUCE_MAX_SIZES``.
Entries exist only where they were measured; a missing capability or world size makes that transport ineligible, and
dispatch falls through to the next one (FlashInfer -> copy-engine overlap -> copy engine -> NCCL).
"""

MiB = 1024 * 1024

# Minimum bytes of one rank's Q (B * S/N * H * D * itemsize) for which the transport replaces the next one in the chain.
# B200 (1000 W), 56 heads x 128, BF16, from the sweep: the copy engine beat NCCL by 7-37% from the smallest swept
# message at every world size, and FlashInfer beat the copy engine by 3-27% at 8 GPUs. At 2 and 4 GPUs the sweep has
# FlashInfer within 1-2% of the copy engine only at its largest message, while MiniMax-H3 end to end ran slower with
# it there, so it has no entry.
FLASHINFER_MIN_BYTES: dict[str, dict[int, int]] = {
    "10.0": {8: 14 * MiB},
}

COPY_ENGINE_MIN_BYTES: dict[str, dict[int, int]] = {
    "10.0": {2: 56 * MiB, 4: 28 * MiB, 8: 14 * MiB},
}

# Minimum bytes of one chunk's copy to one peer (S/N * heads per chunk * D * itemsize) for the head-chunked overlap;
# below it, per-copy overhead outweighs the overlap. B200: from MiniMax-H3 end to end, where 9.25 MiB chunk copies at
# 2 GPUs and 4.62 MiB at 4 beat the plain copy engine (-0.4 and -0.6 s per clip) and ~1 MiB at 8 GPUs did not. The
# isolated-layer sweep does not show this gain (the overlap tied or lost there), so the entries follow end to end.
COPY_ENGINE_OVERLAP_MIN_CHUNK_BYTES: dict[str, dict[int, int]] = {
    "10.0": {2: 8 * MiB, 4: 4 * MiB},
}
