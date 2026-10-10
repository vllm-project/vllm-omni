# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRV2 slot ownership for VoxCPM2's request-local state."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


@dataclasses.dataclass
class VoxCPM2AudioOutput:
    """One async D2H copy for all current request audio chunks."""

    wav: torch.Tensor
    valid_samples: torch.Tensor
    sample_rate: torch.Tensor
    batch_rows: tuple[int, ...]
    num_reqs: int

    def to_cpu(
        self, copy_stream: torch.cuda.Stream, copy_tensor: Callable[[torch.Tensor], torch.Tensor]
    ) -> VoxCPM2AudioOutput:
        return VoxCPM2AudioOutput(
            copy_tensor(self.wav) if self.wav.numel() else self.wav,
            copy_tensor(self.valid_samples) if self.valid_samples.is_cuda else self.valid_samples,
            self.sample_rate,
            self.batch_rows,
            self.num_reqs,
        )

    def get_output(self) -> list[dict[str, torch.Tensor] | None]:
        outputs: list[dict[str, torch.Tensor] | None] = [None] * self.num_reqs
        offset = 0
        for row, length in zip(self.batch_rows, self.valid_samples.tolist(), strict=True):
            outputs[row] = {"model_outputs": self.wav[offset : offset + length], "sr": self.sample_rate}
            offset += length
        return outputs


@dataclasses.dataclass(frozen=True)
class VoxCPM2BatchSlots:
    slots: torch.Tensor
    row_by_slot: dict[int, int]


class VoxCPM2ModelState(OmniModelState):
    def __init__(self, vllm_config: Any, model: Any, encoder_cache: Any, device: torch.device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        from .voxcpm2_talker import _RequestState

        count = self.scheduler_config.max_num_seqs
        samples_per_patch = model._patch_size * int(model.tts.audio_vae.decode_chunk_size)
        capacity = samples_per_patch * model._vae_decode_every * model._audio_emit_every
        self.slots = [_RequestState(request_id="") for _ in range(count)]
        self.audio_buffer = torch.empty((count, capacity), device=device, dtype=torch.float32)
        self.audio_output_lengths = np.zeros(count, dtype=np.int64)
        self.suspended_audio: dict[str, tuple[torch.Tensor, int]] = {}
        self.slot_indices = torch.empty(count, device=device, dtype=torch.long)
        embed_dim = model.config.hidden_size
        self.next_embed = torch.empty((count, embed_dim), device=device, dtype=model._side_dtype)
        self.prefix_feat = torch.empty((count, model._patch_size, model._feat_dim), device=device, dtype=torch.float32)
        self.stop_logits = torch.empty((count, 2), device=device, dtype=model._side_dtype)
        self.decode_pad = torch.empty((count, model._n_decode_pad_frames, model._feat_dim), device=device)
        self.pending_latents = torch.empty(
            (count, model._vae_decode_every, model._patch_size, model._feat_dim), device=device
        )
        max_frames = model._n_decode_pad_frames + model._patch_size * model._vae_decode_every
        self.vae_batch_input = torch.empty((count, model._feat_dim, max_frames), device=device, dtype=torch.float32)
        self.suspended: dict[str, _RequestState] = {}
        # V2 batch slot mapping used while preparing request inputs.
        self.preprocess_batch_slots: VoxCPM2BatchSlots | None = None
        model._mrv2_model_state = self

    def prepare_inputs(self, input_batch: Any, req_states: Any) -> dict[str, Any]:
        inputs = super().prepare_inputs(input_batch, req_states)
        inputs["batch_slots"] = input_batch.idx_mapping[: input_batch.num_reqs]
        inputs["batch_slot_rows"] = {
            int(slot): row for row, slot in enumerate(input_batch.idx_mapping_np[: input_batch.num_reqs])
        }
        return inputs

    def run_preprocess(
        self,
        input_batch: Any,
        model_inputs: dict[str, Any],
        req_states: Any = None,
        mtp_batch_descriptor_dispatcher: Any = None,
    ) -> None:
        self.prepare_audio_length_limits(input_batch, req_states)
        # The generic runner batches decode before invoking individual prefill
        # hooks. Preserve batch order for VoxCPM2's forward token offsets.
        slots = input_batch.idx_mapping_np[: input_batch.num_reqs]
        for row, slot in enumerate(slots):
            self.intermediate_buffer.buffers[int(slot)]["batch_row"] = row
        self.model._pending_batch_rows = []
        self.preprocess_batch_slots = VoxCPM2BatchSlots(
            input_batch.idx_mapping[: input_batch.num_reqs],
            {int(slot): row for row, slot in enumerate(slots)},
        )
        try:
            super().run_preprocess(input_batch, model_inputs, req_states, mtp_batch_descriptor_dispatcher)
            pending = self.model._pending_requests
            rows = self.model._pending_batch_rows
            if len(pending) != len(rows):
                raise RuntimeError("VoxCPM2 MRv2 pending requests lost their batch row")
            if any(a > b for a, b in zip(rows, rows[1:])):
                pending[:] = [pending[i] for i in sorted(range(len(rows)), key=rows.__getitem__)]
        finally:
            self.preprocess_batch_slots = None
            for slot in slots:
                self.intermediate_buffer.buffers[int(slot)].pop("batch_row", None)
                self.intermediate_buffer.buffers[int(slot)].pop("prefill_text_embed", None)
                self.intermediate_buffer.buffers[int(slot)].pop("prepared_prefill", None)
            self.model._pending_batch_rows = None

    def prepare_audio_length_limits(self, input_batch: Any, req_states: Any) -> None:
        """Flush latent and PCM tails on the step that exhausts the output budget."""
        if req_states is None:
            return
        count = input_batch.num_reqs
        slots = input_batch.idx_mapping_np[:count]
        ends_by_length = self.model._audio_length_limit_reached(
            input_batch.num_computed_tokens_np[:count],
            np.asarray(input_batch.num_scheduled_tokens[:count]),
            req_states.max_seq_len[slots],
        )
        for slot, reached in zip(slots, ends_by_length, strict=True):
            self.slots[int(slot)].audio_length_limit_reached = bool(reached)

    def indices_for(self, states: list[Any], batch_context: VoxCPM2BatchSlots | None = None) -> torch.Tensor:
        """Slice runner slots directly; upload row positions only for irregular cohorts."""
        if batch_context is not None and states:
            positions = [batch_context.row_by_slot.get(state.slot_index, -1) for state in states]
            start = positions[0]
            if start >= 0 and positions == list(range(start, start + len(states))):
                return batch_context.slots[start : start + len(states)].long()
            if all(position >= 0 for position in positions):
                rows = self._device_indices(positions)
                return batch_context.slots.index_select(0, rows).long()
        return self._device_indices([self._slot_for(state) for state in states])

    @staticmethod
    def _slot_for(state: Any) -> int:
        index = state.slot_index
        if index is None:
            raise RuntimeError(f"VoxCPM2 request {state.request_id} has no runner slot")
        return index

    def _device_indices(self, values: list[int]) -> torch.Tensor:
        indices = self.slot_indices[: len(values)]
        host = torch.tensor(values, dtype=torch.long)
        if indices.is_cuda:
            host = host.pin_memory()
        indices.copy_(host, non_blocking=indices.is_cuda)
        return indices

    def tensor_for(self, state: Any, name: str) -> torch.Tensor | None:
        index = self._slot_for(state)
        if name in {"curr_embed_for_next", "prev_feat_embed"}:
            return self.next_embed[index : index + 1] if state.decode_state_ready else None
        if name == "curr_prefix_feat_cond":
            return self.prefix_feat[index] if state.decode_state_ready else None
        if name == "last_audio_patch_gpu":
            return self.prefix_feat[index : index + 1] if state.audio_patch_ready else None
        if name == "precomputed_stop_logits":
            return self.stop_logits[index : index + 1] if state.stop_logits_ready else None
        if name == "decode_pad":
            return self.decode_pad[index, : state.decode_pad_len] if state.decode_pad_len else None
        raise KeyError(name)

    def gather_embeddings(self, states: list[Any], batch_context: VoxCPM2BatchSlots | None = None) -> torch.Tensor:
        indices = self.indices_for(states, batch_context)
        return self.next_embed.index_select(0, indices)

    def build_stop_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """One metadata upload and one kernel, preserving legacy sampling scores."""
        from .logits import write_stop_logits

        count = hidden_states.shape[0]
        logits = hidden_states.new_empty((count, self.model.config.vocab_size))
        metadata = np.full((count, 2), -1, dtype=np.int32)
        if not self.model._results_queue:
            metadata[:, 1] = 0
        for row, (owner, stop_logits) in enumerate(self.model._results_queue[:count]):
            state = self.model._active_states[owner] if isinstance(owner, str) else owner
            metadata[row, 0] = self._slot_for(state)
            if stop_logits is not None:
                metadata[row, 1] = 2 if state.is_stopping else 1
                if not state.is_stopping and state.precomputed_is_stopping is not None:
                    state.is_stopping = state.precomputed_is_stopping
                self.clear_tensor(state, "precomputed_stop_logits")
                state.precomputed_is_stopping = None
            else:
                metadata[row, 1] = 3 if state.prefill_completed else 0
        self.model._results_queue.clear()
        if count:
            host = torch.from_numpy(metadata).pin_memory()
            device_metadata = host.to(hidden_states.device, non_blocking=True)
            block = 256
            write_stop_logits[(count, (logits.shape[1] + block - 1) // block)](
                self.stop_logits,
                device_metadata,
                logits,
                stop_stride=self.stop_logits.stride(0),
                vocab_size=logits.shape[1],
                block=block,
            )
        return logits

    @staticmethod
    def clear_tensor(state: Any, name: str) -> None:
        if name == "last_audio_patch_gpu":
            state.audio_patch_ready = False
        elif name == "precomputed_stop_logits":
            state.stop_logits_ready = False
        elif name == "decode_pad":
            state.decode_pad_len = 0
        else:
            raise KeyError(name)

    def store_decode_state(
        self, state: Any, stop_logits: torch.Tensor, next_embed: torch.Tensor, pred_feat: torch.Tensor
    ) -> None:
        index = self._slot_for(state)
        self.next_embed[index].copy_(next_embed.reshape(-1))
        self.prefix_feat[index].copy_(pred_feat.reshape(self.model._patch_size, self.model._feat_dim))
        self.stop_logits[index].copy_(stop_logits.reshape(-1)[:2])
        state.decode_state_ready = True
        state.audio_patch_ready = True
        state.stop_logits_ready = True
        state.precomputed_is_stopping = None

    def store_decode_batch(
        self,
        states: list[Any],
        stop_logits: torch.Tensor,
        next_embed: torch.Tensor,
        pred_feat: torch.Tensor,
        batch_context: VoxCPM2BatchSlots | None = None,
    ) -> None:
        indices = self.indices_for(states, batch_context)
        self.next_embed.index_copy_(0, indices, next_embed.reshape(len(states), -1).to(dtype=self.next_embed.dtype))
        features = pred_feat.reshape(len(states), self.model._patch_size, self.model._feat_dim).to(
            dtype=self.prefix_feat.dtype
        )
        self.prefix_feat.index_copy_(0, indices, features)
        self.stop_logits.index_copy_(
            0, indices, stop_logits.reshape(len(states), -1)[:, :2].to(dtype=self.stop_logits.dtype)
        )
        for state in states:
            state.decode_state_ready = True
            state.audio_patch_ready = True
            state.stop_logits_ready = True
            state.precomputed_is_stopping = None

    def store_decode_pad(self, state: Any, latents: torch.Tensor) -> None:
        index = self._slot_for(state)
        length = min(latents.shape[0], self.decode_pad.shape[1])
        self.decode_pad[index, :length].copy_(latents[-length:])
        state.decode_pad_len = length

    def append_pending_latent(self, state: Any, latent: torch.Tensor) -> torch.Tensor | None:
        index = self._slot_for(state)
        count = state.pending_vae_count
        if count >= self.pending_latents.shape[1]:
            raise RuntimeError("VoxCPM2 pending latent slot is full")
        self.pending_latents[index, count].copy_(latent.reshape(self.model._patch_size, self.model._feat_dim))
        state.pending_vae_count = count + 1
        return self.pending_latents[index, : state.pending_vae_count].reshape(-1, self.model._feat_dim)

    def append_pending_batch(self, states: list[Any], batch_context: VoxCPM2BatchSlots | None = None) -> None:
        """Save all current patches with one device launch, preserving per-slot phase."""
        if not states:
            return
        counts = [state.pending_vae_count for state in states]
        if any(count < 0 or count >= self.pending_latents.shape[1] for count in counts):
            raise RuntimeError("VoxCPM2 pending latent slot is full")
        if self.pending_latents.is_cuda:
            from .vae_pack import append_pending_patches

            indices = self.indices_for(states, batch_context)
            host_counts = torch.tensor(counts, dtype=torch.int32, pin_memory=True)
            device_counts = host_counts.to(self.pending_latents.device, non_blocking=True)
            elements = self.model._patch_size * self.model._feat_dim
            append_pending_patches[(len(states), (elements + 255) // 256)](
                self.prefix_feat,
                self.pending_latents,
                indices,
                device_counts,
                PREFIX_STRIDE=self.prefix_feat.stride(0),
                PENDING_STRIDE=self.pending_latents.stride(0),
                PATCH_STRIDE=self.pending_latents.stride(1),
                ELEMENTS=elements,
                BLOCK=256,
            )
        else:
            for state, count in zip(states, counts, strict=True):
                slot = self._slot_for(state)
                self.pending_latents[slot, count].copy_(self.prefix_feat[slot])
        for state, count in zip(states, counts, strict=True):
            state.pending_vae_count = count + 1

    def decode_ready_slots(
        self,
        group: list[tuple[Any, int, int]],
        batch_context: VoxCPM2BatchSlots | None = None,
    ) -> None:
        """Decode equal-length slots and append PCM directly to owned output rows."""
        states = [state for state, _, _ in group]
        frames = group[0][1] + group[0][2]
        if any(pad + new != frames for _, pad, new in group):
            raise ValueError("VoxCPM2 VAE cohort must have matching total frame lengths")
        count = len(group)
        if frames > self.vae_batch_input.shape[-1]:
            raise RuntimeError("VoxCPM2 VAE input exceeds preallocated latent capacity")
        indices = self.indices_for(states, batch_context)
        inputs = self.vae_batch_input[:count, :, :frames]
        use_pending = self.model._vae_decode_every > 1
        if inputs.is_cuda:
            from .vae_pack import pack_vae_inputs

            host_pad_lengths = torch.tensor([pad for _, pad, _ in group], dtype=torch.int32, pin_memory=True)
            pad_lengths = host_pad_lengths.to(inputs.device, non_blocking=True)
            pack_vae_inputs[(count, (frames * self.model._feat_dim + 255) // 256)](
                self.decode_pad,
                self.pending_latents,
                self.prefix_feat,
                indices,
                pad_lengths,
                inputs,
                PAD_STRIDE=self.decode_pad.stride(0),
                PENDING_STRIDE=self.pending_latents.stride(0),
                PREFIX_STRIDE=self.prefix_feat.stride(0),
                OUT_STRIDE=inputs.stride(0),
                OUT_CHANNEL_STRIDE=inputs.stride(1),
                DIM=self.model._feat_dim,
                FRAMES=frames,
                USE_PENDING=use_pending,
                BLOCK=256,
            )
        else:
            for row, (state, pad_frames, n_new) in enumerate(group):
                slot = self._slot_for(state)
                inputs[row, :, :pad_frames].copy_(self.decode_pad[slot, :pad_frames].T)
                source = (
                    self.pending_latents[slot].reshape(-1, self.model._feat_dim)
                    if use_pending
                    else self.prefix_feat[slot]
                )
                inputs[row, :, pad_frames:].copy_(source[:n_new].T)
        decoded = self.model._run_vae_decode(inputs)
        dcs = int(self.model.tts.audio_vae.decode_chunk_size)
        tail = min(frames, self.decode_pad.shape[1])
        slots = [self._slot_for(state) for state in states]
        starts = [int(self.audio_output_lengths[slot]) for slot in slots]
        lengths = [n_new * dcs for _, _, n_new in group]
        if any(start + length > self.audio_buffer.shape[1] for start, length in zip(starts, lengths)):
            raise RuntimeError("VoxCPM2 audio output exceeds preallocated chunk capacity")
        if inputs.is_cuda:
            from .vae_pack import write_vae_outputs

            # Reuse the pack kernel's slots/pad lengths. Only append offsets
            # need a new upload; PCM and context are written in one launch.
            host_starts = torch.tensor(starts, dtype=torch.int32, pin_memory=True)
            device_starts = host_starts.to(inputs.device, non_blocking=True)
            pcm = decoded.reshape(count, -1)
            elements = max(max(lengths), tail * self.model._feat_dim)
            write_vae_outputs[(count, (elements + 255) // 256)](
                pcm,
                inputs,
                self.audio_buffer,
                self.decode_pad,
                indices,
                pad_lengths,
                device_starts,
                DECODED_STRIDE=pcm.stride(0),
                SAMPLE_STRIDE=pcm.stride(1),
                INPUT_STRIDE=inputs.stride(0),
                CHANNEL_STRIDE=inputs.stride(1),
                AUDIO_STRIDE=self.audio_buffer.stride(0),
                PAD_STRIDE=self.decode_pad.stride(0),
                DIM=self.model._feat_dim,
                FRAMES=frames,
                DCS=dcs,
                TAIL=tail,
                BLOCK=256,
            )
        for row, (state, pad_frames, n_new) in enumerate(group):
            slot, start, length = slots[row], starts[row], lengths[row]
            if not inputs.is_cuda:
                self.audio_buffer[slot, start : start + length].copy_(
                    decoded[row].reshape(-1)[pad_frames * dcs : frames * dcs]
                )
                if tail:
                    self.decode_pad[slot, :tail].copy_(inputs[row, :, frames - tail :].T)
            self.audio_output_lengths[slot] = start + length
            state.decode_pad_len = tail
            state.pending_vae_count = 0

    def make_audio_output(self, model_outputs: torch.Tensor, request_ids: list[str]) -> Any:
        from vllm_omni.model_executor.models.output_templates import OmniOutput

        if len(request_ids) != len(set(request_ids)):
            raise ValueError("VoxCPM2 MRV2 batch contains duplicate request IDs")
        if not {
            owner.request_id if not isinstance(owner, str) else owner for owner, _ in self.model._audio_queue
        }.issubset(request_ids):
            raise ValueError("VoxCPM2 MRV2 audio contains a request outside the current batch")
        for owner, audio in self.model._audio_queue:
            if audio is None:
                continue
            state = self.model._active_states[owner] if isinstance(owner, str) else owner
            index = self._slot_for(state)
            start = self.audio_output_lengths[index]
            length = audio.numel()
            end = start + length
            if end > self.audio_buffer.shape[1]:
                raise RuntimeError("VoxCPM2 audio output exceeds preallocated chunk capacity")
            self.audio_buffer[index, start:end].copy_(audio.reshape(-1))
            self.audio_output_lengths[index] = end
        self.model._audio_queue.clear()
        # Keep the streaming output channel active even on audio-free steps.
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs={"voxcpm2_audio_pending": True})

    def prepare_streaming_audio_output(
        self,
        input_batch: Any,
        req_states: Any,
        outputs: dict[str, Any],
    ) -> VoxCPM2AudioOutput:
        count = input_batch.num_reqs
        slots = input_batch.idx_mapping_np[:count]
        lengths = self.audio_output_lengths[slots]
        ends_by_length = self.model._audio_length_limit_reached(
            input_batch.num_computed_tokens_np[:count],
            np.asarray(input_batch.num_scheduled_tokens[:count]),
            req_states.max_seq_len[slots],
        )
        # Partial PCM exists only after a tail decode. Its stop decision was
        # already resolved on the host before audio collection; do not read
        # sampled CUDA IDs back to the host just to decide whether to copy.
        stopping = np.asarray([self.slots[int(slot)].is_stopping for slot in slots], dtype=bool)
        ready = (lengths > 0) & ((lengths == self.audio_buffer.shape[1]) | stopping | ends_by_length)
        rows = tuple(int(row) for row in np.flatnonzero(ready))
        chunks = [self.audio_buffer[int(slots[row]), : int(lengths[row])] for row in rows]
        # Own only valid samples so the copy stream never reads mutable slots.
        wav = torch.cat(chunks) if chunks else self.audio_buffer.new_empty(0)
        valid = torch.from_numpy(lengths[ready].copy())
        self.audio_output_lengths[slots[ready]] = 0
        return VoxCPM2AudioOutput(wav, valid, torch.tensor(self.model._sample_rate, dtype=torch.int32), rows, count)

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        state = self.slots[req_index]
        request_id = new_req_data.req_id
        if state.request_id:
            raise RuntimeError(f"VoxCPM2 slot {req_index} is still owned by {state.request_id}")
        state.request_id = request_id
        state.slot_index = req_index
        self.intermediate_buffer.buffers[req_index]["slot_index"] = req_index
        parked = self.suspended.pop(request_id, None)
        audio_parked = self.suspended_audio.pop(request_id, None)
        if audio_parked is not None:
            audio, length = audio_parked
            self.audio_buffer[req_index, :length].copy_(audio)
            self.audio_output_lengths[req_index] = length
        if parked is not None:
            for field in dataclasses.fields(state):
                if field.name not in {"request_id", "slot_index"}:
                    setattr(state, field.name, getattr(parked, field.name))
            if parked.curr_embed_for_next is not None:
                self.next_embed[req_index].copy_(parked.curr_embed_for_next.reshape(-1))
            if parked.curr_prefix_feat_cond is not None:
                self.prefix_feat[req_index].copy_(parked.curr_prefix_feat_cond)
            elif parked.last_audio_patch_gpu is not None:
                self.prefix_feat[req_index].copy_(parked.last_audio_patch_gpu.reshape_as(self.prefix_feat[req_index]))
            if parked.precomputed_stop_logits is not None:
                self.stop_logits[req_index].copy_(parked.precomputed_stop_logits.reshape(-1)[:2])
            if parked.decode_pad is not None:
                self.store_decode_pad(state, parked.decode_pad)
            if parked.pending_vae_count:
                self.pending_latents[req_index, : parked.pending_vae_count].copy_(parked.pending_vae_latents_gpu[0])
                state.pending_vae_latents_gpu.clear()
            for name in (
                "curr_embed_for_next",
                "prev_feat_embed",
                "curr_prefix_feat_cond",
                "last_audio_patch_gpu",
                "precomputed_stop_logits",
                "decode_pad",
            ):
                setattr(state, name, None)
        self.model._active_states[request_id] = state

    def remove_request(self, req_index_or_id: int | str) -> None:
        from .voxcpm2_talker import _RequestState

        req_index = self._resolve_req_index(req_index_or_id)
        if req_index is not None:
            state = self.slots[req_index]
            if state.request_id:
                self.model._active_states.pop(state.request_id, None)
            state.slot_index = None
            self.audio_output_lengths[req_index] = 0
            fresh = _RequestState(request_id="")
            for field in dataclasses.fields(state):
                setattr(state, field.name, getattr(fresh, field.name))
        super().remove_request(req_index_or_id)

    def on_request_preempted(self, req_id: str, req_index: int) -> None:
        super().on_request_preempted(req_id, req_index)
        state = self.slots[req_index]
        if state.request_id:
            length = self.audio_output_lengths[req_index]
            if length:
                self.suspended_audio[state.request_id] = (self.audio_buffer[req_index, :length].clone(), length)
            parked = dataclasses.replace(state)
            parked.slot_index = None
            parked.pending_audio_chunks_gpu = list(state.pending_audio_chunks_gpu)
            parked.pending_audio_copies = list(state.pending_audio_copies)
            if state.pending_vae_count:
                parked.pending_vae_latents_gpu = [self.pending_latents[req_index, : state.pending_vae_count].clone()]
            if state.decode_state_ready:
                parked.curr_embed_for_next = self.next_embed[req_index : req_index + 1].clone()
                parked.prev_feat_embed = parked.curr_embed_for_next
                parked.curr_prefix_feat_cond = self.prefix_feat[req_index].clone()
            if state.audio_patch_ready:
                # Both flags describe the same slot tensor. Share the saved
                # snapshot instead of cloning the conditioning features twice.
                parked.last_audio_patch_gpu = (
                    parked.curr_prefix_feat_cond.unsqueeze(0)
                    if state.decode_state_ready
                    else self.prefix_feat[req_index : req_index + 1].clone()
                )
            if state.stop_logits_ready:
                parked.precomputed_stop_logits = self.stop_logits[req_index : req_index + 1].clone()
            if state.decode_pad_len:
                parked.decode_pad = self.decode_pad[req_index, : state.decode_pad_len].clone()
            self.suspended[state.request_id] = parked

    def on_requests_finished(self, req_ids: set[str]) -> None:
        super().on_requests_finished(req_ids)
        for req_id in req_ids:
            self.suspended.pop(req_id, None)
            self.suspended_audio.pop(req_id, None)
