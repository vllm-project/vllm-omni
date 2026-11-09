# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Generic Omni request state with Ascend attention metadata and K-step output."""

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState


class NPUOmniModelState(OmniModelState, AscendModelState):
    """Omni state whose attention metadata is the Ascend V2 one.

    Request buffers and the Omni lifecycle stay on ``OmniModelState``;
    ``AscendModelState`` builds attention metadata from ``attn_state`` and
    ``seq_lens_np``. When the talker has the K-step block armed, the output
    postprocess is routed to the model's V1 frame loop (one forward produces
    K rows per request; the loop samples each codec frame in order) instead of
    the one-frame-per-step ``make_omni_output_mrv2`` hook.
    """

    uses_ascend_attn_metadata = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = getattr(self, "vllm_config", None) or (args[0] if args else None)
        spec = getattr(cfg, "speculative_config", None)
        n = int(getattr(spec, "num_speculative_tokens", 0) or 0) if spec is not None else 0
        self._npu_kstep_frames = n + 1 if n > 0 else 0

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
