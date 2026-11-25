# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measured gates for the automatic Ulysses a2a transports.

Keyed by compute capability (``"major.minor"``) and Ulysses world size, like vLLM's ``CUSTOM_ALL_REDUCE_MAX_SIZES``.
Entries exist only where they were measured; a missing capability or world size makes that transport ineligible, and
dispatch falls through to the next one (copy-engine overlap -> copy engine -> NCCL).
"""

MiB = 1024 * 1024

# Minimum bytes of one rank's Q (B * S/N * H * D * itemsize) for which the transport replaces the next one in the chain.
# B200 (1000 W), 56 heads x 128, BF16, from the sweep: the copy engine beat NCCL by 7-37% from the smallest swept
# message at every world size. H100 SXM (700 W, NVLink), MiniMax-H3 end to end only (Q is 259 / 129.5 / 64.75 MiB per
# rank at 2 / 4 / 8 GPUs): the copy engine beat NCCL by 3.0 / 2.5 / 1.5%, so the entries stop just below those sizes,
# since nothing smaller was measured. The overlap gained nothing there, so it has no sm9.0 entry.
COPY_ENGINE_MIN_BYTES: dict[str, dict[int, int]] = {
    "9.0": {2: 256 * MiB, 4: 128 * MiB, 8: 64 * MiB},
    "10.0": {2: 56 * MiB, 4: 28 * MiB, 8: 14 * MiB},
}

# Minimum bytes of one chunk's copy to one peer (S/N * heads per chunk * D * itemsize) for the head-chunked overlap;
# below it, per-copy overhead outweighs the overlap. B200: from MiniMax-H3 end to end, where 9.25 MiB chunk copies at
# 2 GPUs and 4.62 MiB at 4 beat the plain copy engine (-0.4 and -0.6 s per clip) and ~1 MiB at 8 GPUs did not. The
# isolated-layer sweep does not show this gain (the overlap tied or lost there), so the entries follow end to end.
COPY_ENGINE_OVERLAP_MIN_CHUNK_BYTES: dict[str, dict[int, int]] = {
    "10.0": {2: 8 * MiB, 4: 4 * MiB},
}
