# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRV2 lifecycle, shared-slot AR feedback and incremental PCM output."""

from dataclasses import dataclass

import torch

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

from .cuda_graph_wrapper import FlowSampleRunner, MimiDecodeRunner


@dataclass
class PhononAudioOutput:
    audio: torch.Tensor | None
    valid: torch.Tensor | None
    rows: list[int]
    num_reqs: int
    event: object = None

    def to_cpu(self, copy_stream, copy_tensor):
        if self.event is not None:
            copy_stream.wait_event(self.event)
        return PhononAudioOutput(
            copy_tensor(self.audio) if self.audio is not None else None,
            copy_tensor(self.valid) if self.valid is not None else None,
            self.rows,
            self.num_reqs,
        )

    def get_output(self):
        outputs = [None] * self.num_reqs
        if self.audio is not None:
            for index, row in enumerate(self.rows):
                if bool(self.valid[index]):
                    outputs[row] = {"model_outputs": self.audio[index].flatten(), "sr": torch.tensor(24000)}
        return outputs


class PhononModelState(OmniModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        if self.scheduler_config.max_num_seqs != 1:
            raise ValueError("Phonon initial release requires max_num_seqs=1")
        count = self.scheduler_config.max_num_seqs
        self.mimi_positions = torch.zeros(count, dtype=torch.long, device=device)
        self.remaining_tail = torch.full((count,), -1, dtype=torch.long, device=device)
        self.active = torch.ones(count, dtype=torch.bool, device=device)
        self.next_embeddings = torch.zeros(
            count, model.phonon_config["flow_lm"]["d_model"], dtype=self.dtype, device=device
        )
        self.decoder = MimiDecodeRunner(model.mimi.float(), capture=not vllm_config.model_config.enforce_eager)
        self.flow = FlowSampleRunner(model.flow_lm, capture=not vllm_config.model_config.enforce_eager)
        self.audio_output = PhononAudioOutput(None, None, [], 0)
        self._rows = []
        self.condition = model.flow_lm.conditioning()

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        self.decoder.reset_slots([req_index])
        self.mimi_positions[req_index] = 0
        self.remaining_tail[req_index] = -1
        self.active[req_index] = True
        self.next_embeddings[req_index] = 0
        buffer = self.intermediate_buffer.buffers[req_index]
        params = buffer.get("sampling_params")
        seed = getattr(params, "seed", None)
        buffer["phonon_generator"] = (
            torch.Generator(device=self.next_embeddings.device).manual_seed(seed) if seed is not None else None
        )
        from vllm_omni.worker_v2.warmup import is_warmup_request

        if is_warmup_request(new_req_data.req_id):
            length = len(new_req_data.prompt_token_ids or [])
            buffer["phonon_prompt_embeds"] = self.next_embeddings.new_zeros(length, self.next_embeddings.shape[-1])

        if "phonon_text_ids" in buffer:
            from .prompt import build_prompt

            voice = torch.as_tensor(buffer["phonon_voice"])
            buffer["phonon_prompt_embeds"] = build_prompt(
                self.model.flow_lm,
                buffer["phonon_text_ids"],
                voice,
                project_voice=bool(buffer.get("phonon_project_voice", True)),
            )

    def remove_request(self, req_index_or_id):
        index = self._resolve_req_index(req_index_or_id)
        if index is not None:
            self.decoder.reset_slots([index])
            self.active[index] = False
        super().remove_request(req_index_or_id)

    def on_request_preempted(self, req_id, req_index):
        raise RuntimeError("Phonon streaming requests cannot be recompute-preempted")

    def run_preprocess(self, input_batch, model_inputs, req_states=None, mtp_batch_descriptor_dispatcher=None):
        self._rows = []
        embeds = self._static_inputs_embeds
        for row in range(input_batch.num_reqs):
            slot = int(input_batch.idx_mapping_np[row])
            start = int(input_batch.query_start_loc_np[row])
            count = int(input_batch.num_scheduled_tokens[row])
            computed = int(input_batch.num_computed_tokens_np[row])
            buffer = self.intermediate_buffer.buffers[slot]
            prompt = buffer.get("phonon_prompt_embeds")
            if prompt is None:
                raise ValueError(
                    "Phonon requires phonon_prompt_embeds built from the official text and speaker conditioning"
                )
            prompt = torch.as_tensor(prompt, device=embeds.device, dtype=embeds.dtype)
            if computed < prompt.shape[0]:
                embeds[start : start + count].copy_(prompt[computed : computed + count])
                if computed + count == prompt.shape[0]:
                    self._rows.append(row)
            else:
                embeds[start : start + count].copy_(self.next_embeddings[slot])
                self._rows.append(row)
        model_inputs["inputs_embeds"] = embeds[: input_batch.num_tokens_after_padding]

    def run_postprocess(self, hidden_states, input_batch):
        count = input_batch.num_reqs
        self.model._continue = torch.ones(count, dtype=torch.bool, device=hidden_states.device)
        if not self._rows:
            self.audio_output = PhononAudioOutput(None, None, [], count)
            return
        rows = torch.tensor(self._rows, device=hidden_states.device)
        slots = input_batch.idx_mapping[rows].long()
        last = input_batch.query_start_loc[rows + 1].long() - 1
        hidden = hidden_states[last]
        lm = self.model.flow_lm
        noise = (
            torch.randn(
                hidden.shape[0],
                lm.bos_emb.numel(),
                device=hidden.device,
                dtype=hidden.dtype,
                generator=self.intermediate_buffer.buffers[int(input_batch.idx_mapping_np[self._rows[0]])].get(
                    "phonon_generator"
                ),
            )
            * self.model.phonon_config.get("temp", 0.8) ** 0.5
        )
        latent, eos = self.flow.sample(hidden, noise)
        valid = self.active[slots]
        previous = self.remaining_tail[slots]
        tail_frames = torch.tensor(
            [
                self.intermediate_buffer.buffers[int(input_batch.idx_mapping_np[row])].get("phonon_tail_frames", 3)
                for row in self._rows
            ],
            device=hidden.device,
        )
        tail = torch.where(
            (previous < 0) & (eos.flatten() > self.model.phonon_config["eos_threshold"]), tail_frames, previous
        )
        ended = tail == 0
        self.remaining_tail[slots] = torch.where(tail >= 0, tail - 1, tail)
        self.active[slots] = valid & ~ended
        self.model._continue[rows] = self.active[slots]
        feedback = lm.input_linear(latent) + self.condition
        self.next_embeddings[slots] = feedback
        for index, slot in enumerate(input_batch.idx_mapping_np[self._rows]):
            self.intermediate_buffer.buffers[int(slot)]["next_embedding"] = feedback[index]
        denormalized = (latent * lm.emb_std + lm.emb_mean).float().unsqueeze(-1)
        audio, event = self.decoder.decode(denormalized, slots, self.mimi_positions[slots])
        self.mimi_positions[slots] += self.model.mimi.upsample_ratio
        self.audio_output = PhononAudioOutput(audio, valid, list(self._rows), count, event)

    def postprocess_model_output(self, model_output, input_batch, req_states):
        return model_output, {"phonon_audio_pending": True}

    def prepare_streaming_audio_output(self, input_batch, req_states, outputs):
        return self.audio_output
