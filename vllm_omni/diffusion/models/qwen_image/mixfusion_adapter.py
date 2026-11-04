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
from typing import Any

import torch

from vllm_omni.diffusion.hooks.mixfusion import packed_call_rejection
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

    Built by :meth:`QwenImageMixFusionAdapter.prepare_inputs` under the root
    hook, which hands the chunk-stacked ``hidden_states`` and RoPE tables to
    ``forward``. Only ``hidden_states`` changes across a generation's denoise
    steps; the rest depends solely on the request shapes and is reused via the
    ``cached`` path (``key`` records what it was built from).
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


@dataclass(frozen=True)
class PositionPlan:
    """Model-private input bundle for :meth:`prepare_position_embeddings`.

    Opaque to the framework; it separates the step-invariant layout inputs
    from the per-step latents.
    """

    image_rows: list[torch.Tensor]
    image_row_states: list[int]
    img_shapes: list[tuple[int, int, int]]
    seq_len_txt: int
    key: tuple


class QwenImageMixFusionAdapter:
    """Model-side half of MixFusion for the Qwen image transformer.

    The root hook binds this model's forward call and delegates here; this
    class is the only place that knows the forward signature (a per-request
    latent list plus per-image prompts), the packed call's shape and how the
    packed output maps back to one tensor per request.
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
        # A ``**kwargs`` forward leaves them nested in the bound arguments
        # (memory-profiling stubs); explicit params stay authoritative.
        args = {**(bound.arguments.get("kwargs") or {}), **bound.arguments}
        if "hidden_states" not in args:
            # Contract: packing needs the per-request latent argument.
            raise TypeError(
                f"MixFusion packing requires a `hidden_states` argument in the {type(module).__name__} forward call."
            )
        if not isinstance(args["hidden_states"], (list, tuple)):
            # A batched tensor means a non-packed forward while the step is
            # armed (e.g. memory profiling): stay serial.
            return False, "hidden_states is a batched tensor (serial forward)"
        rejection = packed_call_rejection(
            module,
            attention_kwargs=args.get("attention_kwargs"),
            parallel_config=getattr(module, "parallel_config", None),
        )
        if rejection:
            return False, rejection
        # Model-side input contract. The driver preflights ``can_pack`` before
        # arming, so rejecting here yields the serial fallback, not an exception.
        if not args["hidden_states"]:
            return False, "no hidden-state tensors to pack"
        if args.get("img_shapes") is None or args.get("txt_seq_lens") is None:
            return False, "img_shapes and txt_seq_lens are required"
        if getattr(module, "zero_cond_t", False):
            # The editing path passes per-token modulate indices that the
            # packed layout does not build yet.
            return False, "zero_cond_t/editing path is not supported"
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
            cached=cached,
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
            cache_key=packed.key,
            adapter_state={"prepare_inputs": packed},
        )

    def prepare_position_embeddings(
        self,
        module,
        plan: PositionPlan,
        cache: MixFusionInputs | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        """Build the per-image RoPE tables and chunk size for a packed step.

        Called only by :meth:`prepare_inputs` (and therefore ``pack_inputs``).
        Tables are computed per image row on the full sequence so absolute
        positions are preserved; everything returned depends only on
        ``plan.key``, so the previous bundle can be reused via ``cache``.
        """
        if cache is not None and cache.key == plan.key:
            return cache.row_vid_freqs, cache.text_freqs, cache.chunk_size

        chunk_size = gcd_chunk_size(plan.image_rows)
        row_vid_freqs: list[torch.Tensor] = []
        text_freqs: list[torch.Tensor] = []
        for row_idx, sample in enumerate(plan.image_rows):
            state_idx = plan.image_row_states[row_idx]
            req_vid_freqs, req_txt_freqs = module.pos_embed(
                plan.img_shapes[state_idx], [plan.seq_len_txt], device=sample.device
            )
            row_vid_freqs.append(req_vid_freqs)
            text_freqs.append(req_txt_freqs[: plan.seq_len_txt])
        return row_vid_freqs, text_freqs, chunk_size

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
        attention_layout: AttentionLayout | None = None,
        cached: MixFusionInputs | None = None,
    ) -> MixFusionInputs:
        """Split per-request latents into the packed chunk layout and build its plan.

        Called by the root hook; ``forward`` then runs the packed layout
        through the same body as a batched tensor input. The plan and RoPE
        tables depend only on the request shapes, so ``cached`` from the
        previous denoise step skips the rebuild. Callers must clear
        :meth:`can_pack` first: rejections live there so they become the
        serial fallback rather than an exception here.
        """
        if attention_layout is None:
            attention_layout = self.build_attention_layout()

        # Normalize the all-valid text mask: the plan and the shared forward
        # body follow this convention.
        if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.all():
            encoder_hidden_states_mask = None

        # One row per requested image so every image gets its own prompt row
        # (encoder_hidden_states / temb are already per-image) and chunk mapping.
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

        # Images of the same request share resolution and prompt. Position
        # tables and chunk size are step-invariant, so ``cached`` is keyed on
        # the request shapes and reused across a generation's denoise loop.
        layout_key = (
            tuple(sample.shape for sample in hidden_states),
            tuple(encoder_hidden_states.shape),
            tuple(real_txt_lens),
            tuple(img_shapes),
            tuple(txt_seq_lens),
        )
        position_plan = PositionPlan(
            image_rows=image_rows,
            image_row_states=image_row_states,
            img_shapes=img_shapes,
            seq_len_txt=seq_len_txt,
            key=layout_key,
        )
        row_vid_freqs, text_freqs, chunk_size = self.prepare_position_embeddings(module, position_plan, cached)
        reuse_layout = cached is not None and cached.key == layout_key

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
            # (masked) length, padded to the caller-side width; chunked streams
            # cover their chunks' rows.
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
            # Build the layout once per step (cu_seqlens / gather indices as
            # device tensors) so all blocks run pure GPU work.
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
