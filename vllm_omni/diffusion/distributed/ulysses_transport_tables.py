# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Measured gates for the automatic Ulysses a2a transports.

Keyed by compute capability (``"major.minor"``) and Ulysses world size, like vLLM's ``CUSTOM_ALL_REDUCE_MAX_SIZES``.
Entries exist only where they were measured; a missing capability or world size makes that transport ineligible, and
dispatch falls through to the next one (FlashInfer -> copy-engine overlap -> copy engine -> NCCL).

Regenerate with ``benchmarks/diffusion/benchmark_ulysses_transports.py``, which sweeps the transports over message
sizes at each world size and prints these entries.
"""

MiB = 1024 * 1024

# Minimum bytes of one rank's Q (B * S/N * H * D * itemsize) for which the transport replaces the next one in the chain.
# B200 (1000 W), 56 heads x 128, BF16. Copy engine: from the sweep's smallest size, already >= 20% faster than NCCL.
# FlashInfer: provisional, pending the 8-GPU sweep.
FLASHINFER_MIN_BYTES: dict[str, dict[int, int]] = {}

COPY_ENGINE_MIN_BYTES: dict[str, dict[int, int]] = {
    "10.0": {2: 56 * MiB, 4: 28 * MiB},
}

# Minimum bytes of one chunk's copy to one peer (S/N * heads per chunk * D * itemsize) for the head-chunked overlap;
# below it, per-copy overhead outweighs the overlap. B200: provisional, from MiniMax-H3 end to end (9.25 MiB chunk
# copies at 2 GPUs and 4.62 MiB at 4 were faster than the plain copy engine; ~1 MiB at 8 GPUs was not).
COPY_ENGINE_OVERLAP_MIN_CHUNK_BYTES: dict[str, dict[int, int]] = {
    "10.0": {2: 8 * MiB, 4: 4 * MiB},
}
