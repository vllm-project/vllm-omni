# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side MixFusion half for the Qwen image transformer.

Everything here is Qwen-specific: the packed-call surface, the text/image
attention layout, and the preparation that turns a per-request latent list
into the chunk-stacked layout the shared forward body consumes. The common
driver and hooks only know the protocols in
:mod:`vllm_omni.diffusion.mixfusion.adapter`, so a new model ports MixFusion
by writing this file, registering the adapter and ``_mixfusion_plan`` on its
transformer, and adding a model test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.diffusion.hooks.mixfusion import plan_foreign_hooks
from vllm_omni.diffusion.mixfusion.adapter import AttentionLayout, PackedInputs, StreamSpec
from vllm_omni.diffusion.mixfusion.execution_plan import (
    PackedExecutionPlan,
    build_execution_plan,
    gcd_chunk_size,
    reassemble_chunk_rows,
    split_sequence_chunks,
)


@dataclass
class MixFusionInputs:
    """Prepared packed-MixFusion inputs for one denoise step.

    Built by :meth:`QwenImageMixFusionAdapter.prepare_inputs` from the
    MixFusion root hook; the hook then hands the chunk-stacked
    ``hidden_states`` and RoPE tables to ``forward``, which runs the packed
    layout through the same body as a batched tensor input.

    Only ``hidden_states`` changes across the denoise steps of one generation;
    the remaining fields depend solely on the request shapes and are reused
    via the ``cached`` fast path (``key`` records what they were built from).
    """

    hidden_states: torch.Tensor
    # Chunk-stacked image stream, already projected by ``img_in``.
    encoder_hidden_states_mask: torch.Tensor | None  # normalized: None when all-valid
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor]
    plan: PackedExecutionPlan
    image_row_counts: list[int]
    # Step-invariant layout, kept for the ``cached`` fast path:
    row_vid_freqs: list[torch.Tensor]
    text_freqs: list[torch.Tensor]
    chunk_size: int
    key: tuple


class QwenImageMixFusionAdapter:
    """Model-side half of MixFusion for the Qwen image transformer.

    The common root hook binds this model's forward call and delegates to the
    adapter (see :mod:`vllm_omni.diffusion.mixfusion.adapter`): this class is
    the only place that knows the forward signature (a per-request latent
    list plus per-image prompts), the packed call's shape and how the packed
    output maps back to one tensor per request.
    """

    def build_attention_layout(self, plan: PackedExecutionPlan | None = None) -> AttentionLayout:
        # Text stays request-level and opens each request's logical sequence;
        # the image chunks follow it and are the positional Q/K/V tensors.
        return AttentionLayout((StreamSpec("text", "request"), StreamSpec("image", "chunked")))

    def build_packed_call(self, input_batch) -> dict[str, Any]:
        # Per-request latents stay a list: the root hook packs them. The
        # per-image conditioning tensors are passed as-is; the plan's
        # request/row mapping handles the expansion.
        latents = input_batch.latents
        return {
            "hidden_states": latents,
            "timestep": input_batch.timesteps.to(device=latents[0].device, dtype=latents[0].dtype) / 1000,
            "guidance": input_batch.guidance,
            "encoder_hidden_states": input_batch.prompt_embeds,
            "encoder_hidden_states_mask": input_batch.prompt_embeds_mask,
            "img_shapes": input_batch.img_shapes,
            "txt_seq_lens": input_batch.txt_seq_lens,
            "return_dict": False,
        }

    def can_pack(self, module, bound) -> tuple[bool, str]:
        if "hidden_states" not in bound.arguments:
            # Contract: packing needs the per-request latent argument; keep
            # the failure actionable rather than silently packing nothing.
            raise TypeError(
                f"MixFusion packing requires a `hidden_states` argument in the {type(module).__name__} forward call."
            )
        if not isinstance(bound.arguments["hidden_states"], (list, tuple)):
            # A batched tensor means a non-packed forward while the step is
            # armed (e.g. memory profiling): stay serial.
            return False, "hidden_states is a batched tensor (serial forward)"
        return True, ""

    def pack_inputs(self, module, bound, *, step, cache: PackedInputs | None) -> PackedInputs:
        cached = None if cache is None or cache.adapter_state is None else cache.adapter_state.get("prepare_inputs")
        packed = self.prepare_inputs(
            module,
            hidden_states=bound.arguments["hidden_states"],
            encoder_hidden_states=bound.arguments.get("encoder_hidden_states"),
            encoder_hidden_states_mask=bound.arguments.get("encoder_hidden_states_mask"),
            img_shapes=bound.arguments.get("img_shapes"),
            txt_seq_lens=bound.arguments.get("txt_seq_lens"),
            attention_kwargs=bound.arguments.get("attention_kwargs"),
            cached=cached,
        )
        bound.arguments["hidden_states"] = packed.hidden_states
        bound.arguments["encoder_hidden_states_mask"] = packed.encoder_hidden_states_mask
        return PackedInputs(
            args=tuple(bound.args),
            kwargs=dict(bound.kwargs),
            plan=packed.plan,
            # The rope boundary hands ``forward`` the root-computed packed
            # triple (hidden_states, vid_freqs, txt_freqs) instead of
            # recomputing it on the packed batch.
            boundary_outputs={"rope": (packed.hidden_states, *packed.image_rotary_emb)},
            cache_key=packed.key,
            adapter_state={"prepare_inputs": packed},
        )

    def restore_output(self, module, output, plan: PackedExecutionPlan) -> list[torch.Tensor]:
        return reassemble_chunk_rows(output.sample, plan.request_chunk_ranges, plan.image_row_counts)

    def prepare_inputs(
        self,
        module,
        hidden_states: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        img_shapes: list[tuple[int, int, int]] | None = None,
        txt_seq_lens: list[int] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        attention_layout: AttentionLayout | None = None,
        cached: MixFusionInputs | None = None,
    ) -> MixFusionInputs:
        """Split per-request latents into the packed chunk layout and build its plan.

        This is the entire mixfusion-specific preparation; the MixFusion root
        hook calls it and hands the packed tensors to ``forward``, which runs
        the packed layout through the same body as a batched tensor input.
        The RoPE tables, packing plan and per-request image counts depend only
        on the request shapes, so the bundle from the previous denoise step
        can be handed back as ``cached`` to skip the rebuild (only the
        per-step latents are re-split and re-projected).
        """
        if attention_kwargs:
            # The packed path has no ``attention_kwargs`` channel; refusing
            # keeps a caller's per-attention arguments from being silently
            # dropped.
            raise ValueError(
                "Qwen MixFusion packed forward does not consume attention_kwargs; "
                "pass batched tensor hidden_states (the serial path threads "
                "attention_kwargs into every block) or drop them."
            )
        if module.zero_cond_t:
            raise ValueError("Qwen MixFusion does not support zero_cond_t/editing path yet.")
        if module.parallel_config is not None and module.parallel_config.sequence_parallel_size > 1:
            raise ValueError("Qwen MixFusion requires sequence parallel disabled.")
        foreign_hooks = plan_foreign_hooks(module, getattr(module, "_mixfusion_plan", None))
        if foreign_hooks:
            raise ValueError(
                "Qwen MixFusion packed forward runs modules on the chunk-stacked "
                f"layout, but they carry hooks with unvalidated semantics ({', '.join(foreign_hooks)}); "
                "fall back to the standard forward path."
            )
        if not hidden_states:
            raise ValueError("Qwen MixFusion requires at least one hidden-state tensor.")
        if img_shapes is None or txt_seq_lens is None:
            raise ValueError("Qwen MixFusion requires img_shapes and txt_seq_lens.")
        if attention_layout is None:
            attention_layout = self.build_attention_layout()

        # Normalize the all-valid text mask early: the plan and the shared
        # forward body follow this convention.
        if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.all():
            encoder_hidden_states_mask = None

        # Expand each state's latents into one row per requested image so that
        # every image gets its own prompt row (encoder_hidden_states / temb
        # are already per-image) and its own chunk mapping.
        image_rows: list[torch.Tensor] = []
        image_row_states: list[int] = []
        total_images = 0
        for state_idx, sample in enumerate(hidden_states):
            for img_idx in range(int(sample.shape[0])):
                image_rows.append(sample[img_idx : img_idx + 1])
                image_row_states.append(state_idx)
            total_images += int(sample.shape[0])

        if total_images != int(encoder_hidden_states.shape[0]):
            raise ValueError(
                f"Qwen MixFusion prompt rows ({encoder_hidden_states.shape[0]}) do not "
                f"match total requested images ({total_images})."
            )

        seq_len_txt = int(encoder_hidden_states.shape[1])
        if encoder_hidden_states_mask is not None:
            real_txt_lens = encoder_hidden_states_mask.sum(dim=1).tolist()
        else:
            real_txt_lens = [seq_len_txt] * encoder_hidden_states.shape[0]

        # Keep the per-request image counts for the final reassembly.
        image_row_counts = [int(sample.shape[0]) for sample in hidden_states]

        # RoPE frequency tables are computed per image row on the FULL sequence
        # (absolute positions preserved); img_shapes / txt_seq_lens are
        # per-state, images of the same request share the same resolution and
        # prompt. The hidden states and their freqs are then chunked under the
        # same partition. All of this is step-invariant, so it is keyed on the
        # request shapes and reused verbatim from ``cached`` within a
        # generation's denoise loop.
        layout_key = (
            tuple(sample.shape for sample in hidden_states),
            tuple(encoder_hidden_states.shape),
            tuple(real_txt_lens),
            tuple(img_shapes),
            tuple(txt_seq_lens),
        )
        reuse_layout = cached is not None and cached.key == layout_key
        if reuse_layout:
            row_vid_freqs = cached.row_vid_freqs
            text_freqs = cached.text_freqs
            chunk_size = cached.chunk_size
        else:
            chunk_size = gcd_chunk_size(image_rows)
            row_vid_freqs = []
            text_freqs = []
            for row_idx, sample in enumerate(image_rows):
                state_idx = image_row_states[row_idx]
                req_vid_freqs, req_txt_freqs = module.pos_embed(
                    img_shapes[state_idx], [seq_len_txt], device=sample.device
                )
                row_vid_freqs.append(req_vid_freqs)
                text_freqs.append(req_txt_freqs[:seq_len_txt])

        image_chunks, image_freq_chunks, request_chunk_ranges, chunk_to_request = split_sequence_chunks(
            image_rows, row_vid_freqs, chunk_size
        )

        # Rebind to the chunk-stacked image stream so the shared forward body
        # is layout-agnostic.
        packed_hidden_states = module.img_in(torch.cat(image_chunks, dim=0))

        if reuse_layout:
            image_rotary_emb = cached.image_rotary_emb
            mixfusion_plan = cached.plan
        else:
            image_rotary_emb = (torch.stack(image_freq_chunks, dim=0), torch.stack(text_freqs, dim=0))
            # Per-stream token counts: request-level streams keep their real
            # (masked) length and are padded to the caller-side width; chunked
            # streams cover their chunks' rows.
            request_stream_lens: dict[str, list[int]] = {}
            stream_padded_lens: dict[str, int] = {}
            for spec in attention_layout.streams:
                if spec.kind == "request":
                    request_stream_lens[spec.name] = list(real_txt_lens)
                    stream_padded_lens[spec.name] = seq_len_txt
                else:
                    request_stream_lens[spec.name] = [
                        (chunk_end - chunk_start) * chunk_size for chunk_start, chunk_end in request_chunk_ranges
                    ]
            # Packing layout is identical across blocks and steps: build it once
            # per layout (cu_seqlens / out-gather indices as device tensors) so
            # the 60 blocks run pure GPU work.
            mixfusion_plan = build_execution_plan(
                layout=attention_layout,
                chunk_size=chunk_size,
                request_chunk_ranges=request_chunk_ranges,
                image_row_counts=image_row_counts,
                request_stream_lens=request_stream_lens,
                stream_padded_lens=stream_padded_lens,
                chunk_to_request=torch.tensor(chunk_to_request, dtype=torch.long, device=packed_hidden_states.device),
                device=packed_hidden_states.device,
            )

        return MixFusionInputs(
            hidden_states=packed_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=image_rotary_emb,
            plan=mixfusion_plan,
            image_row_counts=image_row_counts,
            row_vid_freqs=row_vid_freqs,
            text_freqs=text_freqs,
            chunk_size=chunk_size,
            key=layout_key,
        )
