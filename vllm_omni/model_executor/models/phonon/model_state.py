# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRV2 lifecycle, shared-slot AR feedback and incremental PCM output."""

from dataclasses import dataclass

import numpy as np
import torch
from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.sampler import Sampler

from vllm_omni.utils.device_copy import index_to_device
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
        self.tail_frames = torch.full((count,), 3, dtype=torch.long, device=device)
        self.remaining_tail = torch.full((count,), -1, dtype=torch.long, device=device)
        self.active = torch.ones(count, dtype=torch.bool, device=device)
        self.next_embeddings = torch.zeros(
            count, model.phonon_config["flow_lm"]["d_model"], dtype=self.dtype, device=device
        )
        self.decoder = MimiDecodeRunner(model.mimi.float(), capture=not vllm_config.model_config.enforce_eager)
        self.flow = FlowSampleRunner(
            model.flow_lm,
            capture=not vllm_config.model_config.enforce_eager,
            postprocess=self._flow_postprocess,
            state_buffers=(self.remaining_tail, self.active, self.next_embeddings),
        )
        self.audio_output = PhononAudioOutput(None, None, [], 0)
        self._rows = []
        self.condition = model.flow_lm.conditioning()
        from .prompt import load_default_voice

        self.voice = load_default_voice(vllm_config.model_config, model.flow_lm)

    def sample_determined_tokens(self, batch, sampler):
        """Return determined protocol tokens when standard sampling features are inactive."""
        if type(sampler) is not Sampler:
            return None
        rows = batch.idx_mapping_np[: batch.num_reqs]
        if (
            sampler.compute_nans
            or sampler.return_sampling_mask
            or sampler.trace_replay_state is not None
            or sampler.get_logprobs_dims(rows) is not None
            or (
                sampler.thinking_budget_state.enabled
                and np.any(sampler.thinking_budget_state.use_thinking_budget[rows])
            )
        ):
            return None
        for processor in sampler.logits_processors:
            if type(processor) is LogitBiasState:
                active = np.any(processor.use_logit_bias[rows])
            elif type(processor) is PenaltiesState:
                active = np.any(processor.use_penalty[rows])
            elif type(processor) is BadWordsState:
                active = np.any(processor.num_bad_words.np[rows])
            else:
                # Custom processors can alter even a single finite logit.
                return None
            if active:
                return None
        keep = self.model._continue
        if keep is None or keep.numel() != batch.num_reqs:
            return None
        # Owned storage: subsequent forwards can replace the continue mask
        # while these token IDs are being copied or delivered asynchronously.
        tokens = torch.where(keep, self.model.continue_token_id, self.model.eos_token_id)
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            batch.seq_lens.new_ones(batch.num_reqs),
            batch.seq_lens,
            batch.cu_num_logits,
            batch.idx_mapping,
            sampler.req_states.prefill_len.gpu,
        )
        return SamplerOutput(tokens.reshape(-1, 1), None, None, num_sampled, num_rejected)

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        self.mimi_positions[req_index].zero_()
        self.remaining_tail[req_index].fill_(-1)
        self.active[req_index].fill_(True)
        self.next_embeddings[req_index].zero_()
        buffer = self.intermediate_buffer.buffers[req_index]
        self.tail_frames[req_index].fill_(int(buffer.get("phonon_tail_frames", 3)))
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

            buffer["phonon_prompt_embeds"] = build_prompt(
                self.model.flow_lm,
                buffer["phonon_text_ids"],
                self.voice,
                project_voice=False,
                conditioning=self.condition,
            )

    def remove_request(self, req_index_or_id):
        index = self._resolve_req_index(req_index_or_id)
        if index is not None:
            self.active[index].fill_(False)
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
        rows = index_to_device(self._rows, hidden_states.device)
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
        valid, feedback, denormalized, keep = self.flow.sample(hidden, noise, slots)
        if len(self._rows) == count:
            self.model._continue = keep
        else:
            self.model._continue[rows] = keep
        # Sync B=1 waits for D2H in get_output() before the next replay.
        # Async scheduling needs an owned flag while the prior copy is in flight.
        if self.scheduler_config.async_scheduling:
            valid = valid.clone()
        audio, event = self.decoder.decode(denormalized, slots, self.mimi_positions[slots])
        self.mimi_positions[slots] += self.model.mimi.upsample_ratio
        self.audio_output = PhononAudioOutput(audio, valid, list(self._rows), count, event)

    def _flow_postprocess(self, hidden, noise, slots):
        lm = self.model.flow_lm
        latent, eos = lm.sample(hidden, noise)
        valid = self.active[slots]
        previous = self.remaining_tail[slots]
        tail_frames = self.tail_frames[slots]
        tail = torch.where(
            (previous < 0) & (eos.flatten() > self.model.phonon_config["eos_threshold"]), tail_frames, previous
        )
        ended = tail == 0
        self.remaining_tail[slots] = torch.where(tail >= 0, tail - 1, tail)
        keep = valid & ~ended
        self.active[slots] = keep
        feedback = lm.input_linear(latent) + self.condition
        self.next_embeddings[slots] = feedback
        denormalized = (latent * lm.emb_std + lm.emb_mean).float().unsqueeze(-1)
        return valid, feedback, denormalized, keep

    def postprocess_model_output(self, model_output, input_batch, req_states):
        return model_output, {"phonon_audio_pending": True}

    def prepare_streaming_audio_output(self, input_batch, req_states, outputs):
        return self.audio_output
