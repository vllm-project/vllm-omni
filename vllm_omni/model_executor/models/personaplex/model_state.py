# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MRv2 lifecycle for PersonaPlex's model-owned duplex frame state."""

import torch

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.utils.device_copy import index_to_device
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class PersonaPlexModelState(OmniModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        if vllm_config.speculative_config is not None:
            raise ValueError("PersonaPlex MRv2 requires non-speculative sampling")
        super().__init__(vllm_config, model, encoder_cache, device)
        self._turn_codes = None

    def run_preprocess(self, input_batch, model_inputs, req_states=None, mtp_batch_descriptor_dispatcher=None) -> None:
        # Write into the runner's captured embedding buffer. The model's batch
        # hook owns Mimi state and can fill complete live-append rows directly.
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            input_ids = input_batch.input_ids
        embeds = model_inputs.get("inputs_embeds")
        if embeds is None:
            embeds = self.model.embed_input_ids(input_ids)
            model_inputs["inputs_embeds"] = embeds
        req_ids = input_batch.req_ids
        slots = input_batch.idx_mapping_np[: input_batch.num_reqs].tolist()
        buffers = dict(zip(req_ids, (self.intermediate_buffer.buffers[slot] for slot in slots), strict=True))
        starts = input_batch.query_start_loc_np[: input_batch.num_reqs].tolist()
        counts = input_batch.num_scheduled_tokens[: input_batch.num_reqs].tolist()
        computed = input_batch.num_computed_tokens_np[: input_batch.num_reqs].tolist()
        prompt_lens = input_batch.prefill_len_np[: input_batch.num_reqs].tolist()
        handled = self.model.preprocess_batch(
            req_ids=req_ids,
            model_intermediate_buffer=buffers,
            device=embeds.device,
            input_ids=input_ids,
            inputs_embeds=embeds,
            token_offsets=starts,
            num_scheduled_tokens=counts,
            num_computed_tokens=computed,
            prompt_lens=prompt_lens,
        )
        decode = []
        for row, (req_id, slot, start, count, offset, prompt_len) in enumerate(
            zip(req_ids, slots, starts, counts, computed, prompt_lens, strict=True)
        ):
            if req_id in handled:
                continue
            info = dict(buffers[req_id])
            info.update(
                request_id=req_id,
                duplex_token_offset=offset,
                duplex_prompt_len=prompt_len,
                _omni_num_computed_tokens=offset,
                _omni_prompt_len=prompt_len,
                _omni_is_prefill=offset < prompt_len,
            )
            ids, embedding, updates = self.model.preprocess(
                input_ids[start : start + count], embeds[start : start + count], **info
            )
            input_ids[start : start + count].copy_(ids)
            embeds[start : start + count].copy_(embedding)
            mtp_inputs = updates.pop("mtp_inputs", None)
            self.intermediate_buffer.update(slot, updates, self.model.gpu_resident_buffer_keys)
            if offset >= prompt_len and count == 1 and mtp_inputs is not None:
                decode.append((row, slot, start, mtp_inputs))
        self._turn_codes = None
        if decode:
            indices = index_to_device([entry[2] for entry in decode], embeds.device)
            hidden = torch.cat([entry[3][0] for entry in decode])
            text_step = torch.cat([entry[3][1] for entry in decode])
            frame_embeds, codes = self.model.talker_mtp(
                input_ids.index_select(0, indices), embeds.index_select(0, indices), hidden, text_step
            )
            embeds.index_copy_(0, indices, frame_embeds)
            self._turn_codes = (indices, codes)
            for entry, code in zip(decode, codes.split(1), strict=True):
                self.intermediate_buffer.update(
                    entry[1], {"codes": {"audio": code}}, self.model.gpu_resident_buffer_keys
                )

    def postprocess_model_output(self, model_output, input_batch, req_states):
        hidden = model_output.text_hidden_states if isinstance(model_output, OmniOutput) else model_output
        if (
            getattr(self.model.vllm_config.model_config, "session_mode", "turn") != "duplex"
            and self._turn_codes is None
        ):
            return hidden, {}
        codes = torch.zeros(
            (input_batch.num_tokens, self.model.num_active_codebooks), dtype=torch.long, device=hidden.device
        )
        valid = torch.zeros(input_batch.num_tokens, dtype=torch.bool, device=hidden.device)
        if self._turn_codes is not None:
            indices, values = self._turn_codes
            self._turn_codes = None
            codes.index_copy_(0, indices, values[:, : self.model.num_active_codebooks])
            valid.index_fill_(0, indices, True)
        return hidden, {"codes": {"audio": codes}, "meta": {"codec_frame_valid": valid}}

    def post_sample(
        self,
        input_batch,
        text_hidden,
        sampled_token_ids,
        multimodal_outputs,
        mtp_batch_descriptor_dispatcher=None,
    ) -> None:
        if getattr(self.model.vllm_config.model_config, "session_mode", "turn") != "duplex":
            return
        if sampled_token_ids.ndim != 2 or sampled_token_ids.shape[1] != 1:
            raise ValueError("PersonaPlex duplex requires one sampled text token per request")
        # This CPU metadata determines the same kept rows as the non-speculative sampler.
        rows = [
            row
            for row in range(input_batch.num_reqs)
            if input_batch.num_computed_tokens_np[row] + input_batch.num_scheduled_tokens[row]
            >= input_batch.prefill_len_np[row]
            and self.intermediate_buffer.buffers[int(input_batch.idx_mapping_np[row])]
            .get("duplex", {})
            .get("data_plane")
            is True
            and not self.intermediate_buffer.buffers[int(input_batch.idx_mapping_np[row])]
            .get("duplex", {})
            .get("stage0_stale", False)
        ]
        if not rows:
            return
        req_ids = [input_batch.req_ids[row] for row in rows]
        infos = [self.intermediate_buffer.buffers[int(input_batch.idx_mapping_np[row])] for row in rows]
        indices = index_to_device(rows, text_hidden.device)
        last_tokens = input_batch.query_start_loc.index_select(0, indices + 1).long() - 1
        if rows == list(range(len(rows))):
            text_tokens = sampled_token_ids[: len(rows)].reshape(-1)
        else:
            text_tokens = sampled_token_ids.index_select(0, indices).reshape(-1)
        if input_batch.num_tokens == input_batch.num_reqs and len(rows) == input_batch.num_reqs:
            hidden = text_hidden[: len(rows)]
        else:
            hidden = text_hidden.index_select(0, last_tokens)
        # This entry point retains the model-owned teacher gather, depformer
        # graph replay and runtime commit. Its output is copied into this step's
        # owned tensor before another replay can reuse model-local buffers.
        codes = self.model.post_sample_talker_mtp(
            input_ids=text_tokens,
            hidden_states=hidden,
            req_ids=req_ids,
            req_infos=infos,
        )
        multimodal_outputs["codes"]["audio"].index_copy_(0, last_tokens, codes)
        multimodal_outputs["meta"]["codec_frame_valid"].index_fill_(0, last_tokens, True)
