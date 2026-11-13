# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The released Video Depth Anything checkpoints and how to recognize them.

The Hub repositories ship one ``.pth`` state dict and no config; the file name
names the encoder and whether the model predicts metric depth. This module has
no torch dependency so the config resolver can import it cheaply.
"""

from __future__ import annotations

import os

ENCODERS = ("vits", "vitb", "vitl")

# Hub repository -> checkpoint file. Small is Apache-2.0; Base and Large are CC-BY-NC-4.0.
HUB_CHECKPOINTS = {
    f"depth-anything/{'Metric-' if metric else ''}Video-Depth-Anything-{size}": (
        f"{'metric_' if metric else ''}video_depth_anything_{encoder}.pth"
    )
    for metric in (False, True)
    for size, encoder in (("Small", "vits"), ("Base", "vitb"), ("Large", "vitl"))
}


def checkpoint_variant(filename: str) -> tuple[str, bool]:
    """``(encoder, metric)`` for a released checkpoint file name."""
    name = os.path.basename(filename)
    metric = name.startswith("metric_")
    encoder = name.removeprefix("metric_").removeprefix("video_depth_anything_").removesuffix(".pth")
    if encoder not in ENCODERS or name != f"{'metric_' if metric else ''}video_depth_anything_{encoder}.pth":
        raise ValueError(f"{filename!r} is not a released Video Depth Anything checkpoint")
    return encoder, metric


def resolve_checkpoint(model: str) -> str | None:
    """The checkpoint file a model id or local directory serves, or ``None`` if it is not Video Depth Anything."""
    normalized = str(model).strip().rstrip("/")
    for repo, filename in HUB_CHECKPOINTS.items():
        if normalized.lower() == repo.lower():
            return filename
    if not os.path.isdir(normalized):
        return None
    found = sorted(name for name in HUB_CHECKPOINTS.values() if os.path.isfile(os.path.join(normalized, name)))
    # A directory holding several variants is ambiguous; serve exactly one.
    return found[0] if len(found) == 1 else None
