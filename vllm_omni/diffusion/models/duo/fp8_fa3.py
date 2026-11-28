# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Load an explicitly built FP8 FA3 library before graph capture."""

from functools import lru_cache


@lru_cache(maxsize=2)
def load_library(path):
    from pathlib import Path

    import torch

    path = Path(path)
    torch.ops.load_library(str(path))
    return getattr(torch.ops, path.stem)


def library_metadata(ops, rows, capacity, heads, dim, lengths, cuq):
    import torch

    return ops.get_scheduler_metadata(
        1,
        rows,
        capacity,
        heads,
        heads,
        dim,
        dim,
        torch.float8_e4m3fn,
        lengths,
        cuq,
        None,
        None,
        None,
        None,
        None,
        0,
        False,
        -1,
        -1,
        False,
        1,
        None,
        0,
    )


def library_forward(ops, q, k, v, cuq, cuk, qs, ks, vs, scale, metadata, capacity):
    import torch

    out = torch.empty_like(q, dtype=torch.bfloat16)
    return ops.fwd(
        q,
        k,
        v,
        None,
        None,
        None,
        out,
        cuq,
        cuk,
        None,
        None,
        None,
        q.shape[0],
        capacity,
        None,
        None,
        None,
        None,
        None,
        None,
        qs,
        ks,
        vs,
        scale,
        False,
        -1,
        -1,
        0.0,
        True,
        metadata,
        1,
        None,
        0,
        None,
        1,
        0,
        None,
    )[0]
