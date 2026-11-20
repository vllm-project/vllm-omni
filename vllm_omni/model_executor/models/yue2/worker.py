# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native AR execution with request-owned acoustic work after slot retirement."""

import copy
import time
from dataclasses import replace

from vllm.v1.outputs import AsyncModelRunnerOutput

from vllm_omni.worker.gpu_ar_worker import GPUARWorker
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner


def _attach_audio(output, started, ready):
    # The no-forward output may be shared across steps.
    output = copy.copy(output)
    output.acoustic_started = started
    output.acoustic_ready = ready
    return output


class _Yue2AsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, output, started, ready):
        self.output = output
        self.started = started
        self.ready = ready
        self.copy_event = getattr(output, "copy_event", None)

    def get_output(self):
        return _attach_audio(self.output.get_output(), self.started, self.ready)

    def get_kv_connector_output(self):
        return self.output.get_kv_connector_output()


class Yue2ModelRunner(OmniARModelRunner):
    def finish_requests(self, scheduler_output):
        if replaced := getattr(scheduler_output, "replaced_req_ids", None):
            scheduler_output = replace(scheduler_output, finished_req_ids=scheduler_output.finished_req_ids | replaced)
        super().finish_requests(scheduler_output)
        for req_id in sorted(getattr(scheduler_output, "acoustic_req_ids", set()) - scheduler_output.finished_req_ids):
            # Retiring AR state must not invoke the model's acoustic cancellation hook.
            self._remove_request(req_id)

    def _with_audio(self, output, *, idle=False):
        ready = self.model.poll_audio()
        started, self.model.acoustic_started = self.model.acoustic_started, []
        if not ready and not started:
            if idle:
                time.sleep(0.001)
            return output
        if isinstance(output, AsyncModelRunnerOutput):
            return _Yue2AsyncOutput(output, started, ready)
        return _attach_audio(output, started, ready)

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
        context_len=0,
        valid_dummy_state_slots=False,
        randomize_inputs=False,
    ):
        output = super().execute_model(
            scheduler_output,
            intermediate_tensors,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            is_profile=is_profile,
            context_len=context_len,
            valid_dummy_state_slots=valid_dummy_state_slots,
            randomize_inputs=randomize_inputs,
        )
        if output is not None and not dummy_run and not is_profile:
            return self._with_audio(output, idle=scheduler_output.total_num_scheduled_tokens == 0)
        return output

    def sample_tokens(self, grammar_output):
        output = super().sample_tokens(grammar_output)
        return self._with_audio(output) if output is not None else None

    def shutdown(self):
        try:
            if hasattr(self, "model"):
                self.model.close()
        finally:
            super().shutdown()


class Yue2Worker(GPUARWorker):
    model_runner_cls = Yue2ModelRunner
