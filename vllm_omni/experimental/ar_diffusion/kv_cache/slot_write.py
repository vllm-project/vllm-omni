# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Row-indexed KV writes without NPU's general advanced-indexing kernel."""


def write_kv_slots(pool, slots, values):
    values = values.to(pool.dtype)
    # The cache planner supplies distinct physical slots for each write. Keep
    # mutations in the existing write+attention op and on the caller's stream.
    if pool.device.type in ("npu", "privateuseone"):
        pool.index_copy_(0, slots, values)
    else:
        pool[slots] = values
