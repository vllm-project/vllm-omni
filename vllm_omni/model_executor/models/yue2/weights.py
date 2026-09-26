# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR/side routing for YuE2's single-file MoT checkpoint.

The checkpoint interleaves AR-path, NAR-path and projection tensors under one
namespace. AR tensors go through the vLLM backbone loader unchanged; NAR and
projection tensors are hand-loaded, with ``model.layers.N.nar_*`` names
remapped onto the ``nar_layers.N.*`` module layout.
"""

from collections.abc import Iterable

import torch

_SIDE_TOP_LEVEL = {"vae2llm", "llm2vae", "time_embedder"}


def partition_checkpoint_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    """Split checkpoint tensors into (backbone AR pairs, remapped side pairs)."""
    ar: list[tuple[str, torch.Tensor]] = []
    side: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        if name == "latent_pos_embed.pe":
            continue  # deterministic sinusoid, rebuilt in __init__
        if ".nar_" in name or name.split(".", 1)[0] in _SIDE_TOP_LEVEL:
            new = name
            if name.startswith("model.layers."):
                rest = name[len("model.layers.") :]
                layer_no, _, tail = rest.partition(".")
                if tail.startswith("nar_"):
                    tail = tail[len("nar_") :]
                new = f"nar_layers.{layer_no}.{tail}"
            side.append((new, tensor))
            continue
        ar.append((name, tensor))
    return ar, side
