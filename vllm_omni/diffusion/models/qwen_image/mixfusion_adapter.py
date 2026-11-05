# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side MixFusion half for the Qwen image transformer.

Qwen-specific pieces: the packed-call surface, the text/image attention
layout, and the preparation that turns a per-request latent list into the
chunk-stacked layout the shared forward body consumes. The common driver and
hooks only know the protocols in
:mod:`vllm_omni.diffusion.mixfusion.adapter`; porting a new model means
writing this file, registering the adapter and ``_mixfusion_plan``, and
adding a model test.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout, PackedInputs, StreamSpec
from vllm_omni.diffusion.mixfusion.execution_plan import (
    PackedExecutionPlan,
    build_execution_plan,
    gcd_chunk_size,
    reassemble_chunk_rows,
    split_sequence_chunks,
)

# Text stays request-level and opens each request's logical sequence; the
# image chunks follow it and are the positional Q/K/V tensors.
QWEN_ATTENTION_LAYOUT = AttentionLayout((StreamSpec("text", "request"), StreamSpec("image", "chunked")))


@dataclass
class MixFusionInputs:
    """Prepared packed-MixFusion inputs for one denoise step.

    Only ``hidden_states`` changes across a generation's denoise steps; the
    rest depends solely on the request shapes recorded in ``key`` and is
    reused through the ``cached`` path.
    """

    # Chunk-stacked image stream, already projected by ``img_in``.
    hidden_states: torch.Tensor
    # Normalized: None when every text token is valid.
    encoder_hidden_states_mask: torch.Tensor | None
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor]
    plan: PackedExecutionPlan
    # Per-image RoPE tables, re-split with each step's latents.
    row_vid_freqs: list[torch.Tensor]
    key: tuple


class QwenImageMixFusionAdapter:
    """Model-side half of MixFusion for the Qwen image transformer.

    The only place that knows this model's forward signature (a per-request
    latent list plus per-image prompts) and how the packed output maps back
    to one tensor per request.
    """

    def can_pack(self, module, input_batch) -> tuple[bool, str]:
        if getattr(module, "zero_cond_t", False):
            # The editing path passes per-token modulate indices that the
            # packed layout does not build yet.
            return False, "zero_cond_t/editing path is not supported"
        if input_batch.image_latents is not None:
            # Reference-image tokens need their own RoPE and per-token
            # conditioning in the packed layout, which is not built yet.
            return False, "image_latents (reference images) are not supported"
        if input_batch.img_shapes is None or input_batch.txt_seq_lens is None:
            return False, "img_shapes and txt_seq_lens are required"
        return True, ""

    def pack_inputs(self, module, bound, *, cache: PackedInputs | None) -> PackedInputs:
        packed = self.prepare_inputs(
            module,
            hidden_states=bound.arguments["hidden_states"],
            encoder_hidden_states=bound.arguments["encoder_hidden_states"],
            encoder_hidden_states_mask=bound.arguments.get("encoder_hidden_states_mask"),
            img_shapes=bound.arguments["img_shapes"],
            txt_seq_lens=bound.arguments["txt_seq_lens"],
            cached=None if cache is None else cache.adapter_state,
        )
        bound.arguments["hidden_states"] = packed.hidden_states
        bound.arguments["encoder_hidden_states_mask"] = packed.encoder_hidden_states_mask
        return PackedInputs(
            args=tuple(bound.args),
            kwargs=dict(bound.kwargs),
            plan=packed.plan,
            # The rope boundary hands ``forward`` the root-computed
            # (hidden_states, vid_freqs, txt_freqs) triple.
            boundary_outputs={"rope": (packed.hidden_states, *packed.image_rotary_emb)},
            adapter_state=packed,
        )

    def restore_output(self, module, output, plan: PackedExecutionPlan) -> list[torch.Tensor]:
        return reassemble_chunk_rows(output.sample, plan.request_chunk_ranges, plan.image_row_counts)

    def prepare_inputs(
        self,
        module,
        hidden_states: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None,
        img_shapes: list[tuple[int, int, int]],
        txt_seq_lens: list[int],
        cached: MixFusionInputs | None = None,
    ) -> MixFusionInputs:
        """Split per-request latents into the packed chunk layout and build its plan.

        The plan and RoPE tables depend only on the request shapes, so
        ``cached`` from the previous denoise step skips their rebuild; only
        the per-step latents are re-split and re-projected.
        """
        # Normalize the all-valid text mask: the plan and the shared forward
        # body follow this convention.
        if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.all():
            encoder_hidden_states_mask = None

        # One row per requested image so every image gets its own prompt row
        # (encoder_hidden_states / temb are already per-image) and chunk mapping.
        image_rows = [sample[img_idx : img_idx + 1] for sample in hidden_states for img_idx in range(sample.shape[0])]
        image_row_states = [state_idx for state_idx, sample in enumerate(hidden_states) for _ in range(sample.shape[0])]
        if len(image_rows) != encoder_hidden_states.shape[0]:
            raise ValueError(
                f"Qwen MixFusion prompt rows ({encoder_hidden_states.shape[0]}) do not "
                f"match total requested images ({len(image_rows)})."
            )

        seq_len_txt = int(encoder_hidden_states.shape[1])
        if encoder_hidden_states_mask is not None:
            real_txt_lens = encoder_hidden_states_mask.sum(dim=1).tolist()
        else:
            real_txt_lens = [seq_len_txt] * len(image_rows)

        layout_key = (
            tuple(sample.shape for sample in hidden_states),
            tuple(encoder_hidden_states.shape),
            tuple(real_txt_lens),
            tuple(img_shapes),
            tuple(txt_seq_lens),
        )
        if cached is not None and cached.key == layout_key:
            plan = cached.plan
            row_vid_freqs = cached.row_vid_freqs
            image_chunks, _, _, _ = split_sequence_chunks(image_rows, row_vid_freqs, plan.chunk_size)
            return MixFusionInputs(
                hidden_states=module.img_in(torch.cat(image_chunks, dim=0)),
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                image_rotary_emb=cached.image_rotary_emb,
                plan=plan,
                row_vid_freqs=row_vid_freqs,
                key=layout_key,
            )

        # Per-image RoPE tables on the full sequence keep absolute positions.
        chunk_size = gcd_chunk_size(image_rows)
        row_vid_freqs: list[torch.Tensor] = []
        text_freqs: list[torch.Tensor] = []
        for sample, state_idx in zip(image_rows, image_row_states):
            vid_freqs, txt_freqs = module.pos_embed(img_shapes[state_idx], [seq_len_txt], device=sample.device)
            row_vid_freqs.append(vid_freqs)
            text_freqs.append(txt_freqs[:seq_len_txt])

        image_chunks, image_freq_chunks, request_chunk_ranges, chunk_to_request = split_sequence_chunks(
            image_rows, row_vid_freqs, chunk_size
        )
        packed_hidden_states = module.img_in(torch.cat(image_chunks, dim=0))
        device = packed_hidden_states.device
        # Built once per layout (cu_seqlens / gather indices as device tensors)
        # so every block runs pure GPU work.
        plan = build_execution_plan(
            layout=QWEN_ATTENTION_LAYOUT,
            chunk_size=chunk_size,
            request_chunk_ranges=request_chunk_ranges,
            image_row_counts=[int(sample.shape[0]) for sample in hidden_states],
            request_stream_lens={
                "text": real_txt_lens,
                "image": [(end - start) * chunk_size for start, end in request_chunk_ranges],
            },
            stream_padded_lens={"text": seq_len_txt},
            chunk_to_request=torch.tensor(chunk_to_request, dtype=torch.long, device=device),
            device=device,
        )
        return MixFusionInputs(
            hidden_states=packed_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=(torch.stack(image_freq_chunks, dim=0), torch.stack(text_freqs, dim=0)),
            plan=plan,
            row_vid_freqs=row_vid_freqs,
            key=layout_key,
        )
