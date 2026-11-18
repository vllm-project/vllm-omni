# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""BAGEL image-block positions on Model Runner V2."""

from vllm.v1.worker.gpu.mm.rope import RopeState

from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class BagelModelState(OmniModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)
        # BAGEL uses 1D RoPE with compressed image blocks. Reuse the native
        # prompt-position storage and decode offsets without declaring 3D RoPE.
        self.rope_state = RopeState(
            num_dims=1,
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_num_tokens=self.max_num_tokens,
            max_model_len=self.model_config.max_model_len,
            device=device,
        )

    def prepare_inputs(self, input_batch, req_states):
        inputs = super().prepare_inputs(input_batch, req_states)
        inputs["positions"] = inputs["positions"][0]
        # The multimodal runner otherwise supplies embeddings without IDs.
        # Img2img layout routing also needs the actual placeholder tokens.
        inputs["input_ids"] = input_batch.input_ids
        return inputs

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        inputs["positions"] = inputs["positions"][0]
        return inputs
