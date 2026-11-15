# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generic Omni request state with Ascend attention metadata and K-step output."""

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.model_executor.models.minicpmo_4_5.talker_kstep_contract import talker_kstep_frames
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class NPUOmniModelState(OmniModelState, AscendModelState):
    """Ascend attention metadata with K-step frame-loop output routing."""

    uses_ascend_attn_metadata = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = getattr(self, "vllm_config", None) or (args[0] if args else None)
        stage = getattr(getattr(cfg, "model_config", None), "model_stage", None)
        self._npu_kstep_frames = talker_kstep_frames(stage, getattr(cfg, "speculative_config", None))

    def postprocess_model_output(self, model_output, input_batch, req_states):
        if self._npu_kstep_frames > 1 and not isinstance(model_output, OmniOutput):
            buffer_list = self.intermediate_buffer.gather(input_batch)
            model_output = self.model.make_omni_output(
                model_output,
                model_intermediate_buffer=buffer_list,
                runtime_additional_information=buffer_list,
                request_token_spans=self._request_token_spans(input_batch),
            )
        if isinstance(model_output, OmniOutput):
            return model_output.text_hidden_states, dict(model_output.multimodal_outputs or {})
        return super().postprocess_model_output(model_output, input_batch, req_states)
