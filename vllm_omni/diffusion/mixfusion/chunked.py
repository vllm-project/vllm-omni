# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Framework-owned MixFusion packing for joint text/image DiTs.

Models whose blocks are token-local apart from joint attention (dual-stream
DiTs with per-row AdaLN, e.g. Qwen-Image) declare a ``_mixfusion_plan`` and get
packing from this module; they write no adapter. The root boundary builds the
:class:`PackedExecutionPlan` from the per-request latent list and the text
stream, the :class:`~vllm_omni.diffusion.mixfusion.plan.MixFusionChunkOutputs`
boundary runs the model's own embedding/RoPE module once per request and
chunk-stacks its outputs, and the output is reassembled per request.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout, MixFusionAdapter, PackedInputs, StreamSpec
from vllm_omni.diffusion.mixfusion.execution_plan import (
    PackedExecutionPlan,
    build_execution_plan,
    reassemble_chunk_rows,
)
from vllm_omni.diffusion.mixfusion.plan import MixFusionRoot

#: Joint attention streams of a packed call, in per-request token order: the
#: request-level text stream opens each request, the chunk-stacked image
#: stream follows. The model passes image Q/K/V positionally and text through
#: ``extra[MIXFUSION_STREAMS_KEY]``, keyed ``"text"``.
JOINT_TEXT_IMAGE_LAYOUT = AttentionLayout((StreamSpec("text", "request"), StreamSpec("image", "chunked")))


class ChunkedJointAdapter:
    """Packs a joint text/image DiT call from its ``MixFusionRoot`` declaration.

    Input requirements are declared on the root alongside the layout.
    """

    def __init__(self, root: MixFusionRoot) -> None:
        self.root = root

    def validate_inputs(self, module: nn.Module, input_batch: Any) -> None:
        for name in self.root.required_inputs:
            if getattr(input_batch, name, None) is None:
                raise ValueError(f"Scheduled MixFusion batch is missing {name}")
        for name in self.root.unsupported_inputs:
            if getattr(input_batch, name, None) is not None:
                raise ValueError(f"Scheduled MixFusion batch contains unsupported {name}")

    def pack_inputs(self, module: nn.Module, bound: Any, *, adapter_state: Any | None) -> PackedInputs:
        """Build (or reuse) the plan; the call itself is left unchanged.

        The per-request latent list stays in place for the chunk-outputs
        boundary, which is its first consumer in ``forward``. The plan depends
        only on the request shapes and text lengths, so a matching
        ``adapter_state`` from the previous step skips the rebuild.
        """
        latents = bound.arguments[self.root.ragged_arg]
        text = bound.arguments[self.root.text_arg]
        mask = bound.arguments.get(self.root.text_mask_arg) if self.root.text_mask_arg else None
        rows = [sample[idx : idx + 1] for sample in latents for idx in range(sample.shape[0])]
        if len(rows) != text.shape[0]:
            raise ValueError(
                f"MixFusion text rows ({text.shape[0]}) do not match the total requested images ({len(rows)})."
            )
        padded = int(text.shape[1])
        real_lens = [padded] * len(rows) if mask is None else [int(n) for n in mask.sum(dim=1).tolist()]
        key = (tuple(tuple(sample.shape) for sample in latents), tuple(text.shape), tuple(real_lens))
        if adapter_state is not None and adapter_state[0] == key:
            plan = adapter_state[1]
        else:
            # Derive the layout from this step's actual prepared batch. Adding or
            # removing requests can change the GCD, including preparation failures.
            chunk_size = math.gcd(*(int(sample.shape[1]) for sample in latents))
            plan = _build_plan(latents, rows, real_lens, padded, chunk_size=chunk_size, device=text.device)
        return PackedInputs(args=tuple(bound.args), kwargs=dict(bound.kwargs), plan=plan, adapter_state=(key, plan))

    def restore_output(self, module: nn.Module, output: Any, plan: PackedExecutionPlan) -> list[torch.Tensor]:
        sample = output[0] if isinstance(output, (tuple, list)) else getattr(output, self.root.output_attr)
        return reassemble_chunk_rows(sample, plan.request_chunk_ranges, plan.image_row_counts)


def _build_plan(
    latents: list[torch.Tensor],
    rows: list[torch.Tensor],
    text_lens: list[int],
    text_padded: int,
    *,
    chunk_size: int,
    device: torch.device,
) -> PackedExecutionPlan:
    ranges: list[tuple[int, int]] = []
    chunk_to_request: list[int] = []
    for row_idx, row in enumerate(rows):
        start = len(chunk_to_request)
        chunk_to_request.extend([row_idx] * (int(row.shape[1]) // chunk_size))
        ranges.append((start, len(chunk_to_request)))
    return build_execution_plan(
        layout=JOINT_TEXT_IMAGE_LAYOUT,
        chunk_size=chunk_size,
        request_chunk_ranges=ranges,
        image_row_counts=[int(sample.shape[0]) for sample in latents],
        request_stream_lens={"text": text_lens, "image": [int(row.shape[1]) for row in rows]},
        stream_padded_lens={"text": text_padded},
        chunk_to_request=torch.tensor(chunk_to_request, dtype=torch.long, device=device),
        device=device,
    )


def resolve_mixfusion_adapter(module: nn.Module, plan: dict | None) -> MixFusionAdapter | None:
    """The adapter that packs ``module``: an explicit ``mixfusion_adapter`` wins.

    Otherwise a plan whose root is a :class:`MixFusionRoot` gets the
    framework's :class:`ChunkedJointAdapter`. ``None`` means the model is not
    packable and requests retain dense admission.
    """
    explicit = getattr(module, "mixfusion_adapter", None)
    if explicit is not None:
        return explicit
    root = (plan or {}).get("")
    return ChunkedJointAdapter(root) if isinstance(root, MixFusionRoot) else None
