# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Observe the real runner/model boundary without replacing inference."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


class PhaseContractWorkerExtension:
    model_runner: "OmniGPUModelRunner"

    def start_phase_contract_probe(self):
        runner = self.model_runner
        self.phase_contract_events: list[tuple[int, int, int, bool]] = []
        self.phase_contract_codes: list[list[list[int]]] = []
        tail_embeddings = {}
        original_preprocess = runner.model.preprocess
        original_mtp = runner._talker_mtp_forward

        def preprocess(input_ids, input_embeds, **info):
            rid = info["request_id"]
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
                offset = int(runner.query_start_loc.cpu[index])
                tail_embeddings[offset] = result[1].detach().clone()
            return result

        def mtp(decode_req_ids, inputs_embeds, start_offsets=None):
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

        runner.model.preprocess = preprocess
        setattr(runner, "_talker_mtp_forward", mtp)
        return True

    def get_phase_contract_probe(self):
        return {"events": self.phase_contract_events, "codes": self.phase_contract_codes}
