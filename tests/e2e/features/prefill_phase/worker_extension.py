# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Observe the real runner/model boundary without replacing inference."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner as LegacyRunner
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner


class PhaseContractWorkerExtension:
    model_runner: "LegacyRunner | OmniARModelRunner"

    def start_phase_contract_probe(self):
        runner = self.model_runner
        self.phase_contract_events: list[tuple[int, int, int, bool]] = []
        self.phase_contract_codes: list[list[list[int]]] = []
        tail_embeddings = {}
        original_preprocess = runner.model.preprocess
        model_state = getattr(runner, "model_state", None)
        is_v2 = model_state is not None
        original_mtp = model_state.post_sample if model_state is not None else runner._talker_mtp_forward

        def preprocess(input_ids, input_embeds, **info):
            rid = info["req_id"] if is_v2 else info["request_id"]
            if is_v2:
                prompt_len = info["_omni_prompt_len"]
                computed = info["_omni_num_computed_tokens"]
            else:
                index = runner.input_batch.req_id_to_index[rid]
                prompt_len = len(runner.requests[rid].prompt_token_ids)
                computed = int(runner.input_batch.num_computed_tokens_cpu[index])
            assert info["_omni_prompt_len"] == prompt_len
            assert info["_omni_num_computed_tokens"] == computed
            assert info["_omni_is_prefill"] is (computed < prompt_len)
            result = original_preprocess(input_ids=input_ids, input_embeds=input_embeds, **info)
            span = input_ids.numel()
            self.phase_contract_events.append((prompt_len, computed, span, info["_omni_is_prefill"]))
            if computed == prompt_len - 1 and span == 1:
                if is_v2:
                    tail_embeddings[rid] = result[1].detach().clone()
                else:
                    offset = int(runner.query_start_loc.cpu[index])
                    tail_embeddings[offset] = result[1].detach().clone()
            return result

        def legacy_mtp(decode_req_ids, inputs_embeds, start_offsets=None):
            for rid in decode_req_ids:
                index = runner.input_batch.req_id_to_index[rid]
                assert int(runner.input_batch.num_computed_tokens_cpu[index]) >= len(
                    runner.requests[rid].prompt_token_ids
                ), "prefill row was routed through MTP"
            original_mtp(decode_req_ids, inputs_embeds, start_offsets)
            for offset, expected in tail_embeddings.items():
                torch.testing.assert_close(inputs_embeds[offset : offset + 1], expected, rtol=0, atol=0)
            tail_embeddings.clear()
            for rid in decode_req_ids:
                self.phase_contract_codes.append(runner.model_intermediate_buffer[rid]["codes"]["audio"].tolist())

        def v2_mtp(input_batch, hidden, sampled, outputs, *args, **kwargs):
            assert model_state is not None
            recorded = model_state._mtp_state.recorded
            entries = recorded[1].entries if recorded is not None else []
            for batch_index, req_index, _req_id, _prefill in entries:
                computed = model_state._get_input_batch_num_computed(input_batch, req_index, batch_index)
                prompt_len = model_state._get_req_state_value(runner.req_states.prompt_len, req_index)
                scheduled = int(input_batch.num_scheduled_tokens[batch_index])
                assert computed is not None and prompt_len is not None and computed + scheduled >= prompt_len, (
                    "unfinished prefill row was routed through MTP"
                )
            original_mtp(input_batch, hidden, sampled, outputs, *args, **kwargs)
            for batch_index, _req_index, _req_id, _prefill in entries:
                last = int(input_batch.query_start_loc_np[batch_index + 1]) - 1
                if outputs["meta"]["codec_frame_valid"][last]:
                    self.phase_contract_codes.append(outputs["codes"]["audio"][last : last + 1].tolist())

        runner.model.preprocess = preprocess
        if model_state is not None:
            original_run_preprocess = model_state.run_preprocess

            def run_preprocess(input_batch, model_inputs, *args, **kwargs):
                original_run_preprocess(input_batch, model_inputs, *args, **kwargs)
                embeds = model_inputs["inputs_embeds"]
                for rid, expected in tail_embeddings.items():
                    batch_index = input_batch.req_ids.index(rid)
                    offset = int(input_batch.query_start_loc_np[batch_index])
                    torch.testing.assert_close(embeds[offset : offset + 1], expected, rtol=0, atol=0)
                tail_embeddings.clear()

            model_state.run_preprocess = run_preprocess
            model_state.post_sample = v2_mtp
        else:
            runner._talker_mtp_forward = legacy_mtp
        return True

    def get_phase_contract_probe(self):
        return {"events": self.phase_contract_events, "codes": self.phase_contract_codes}
