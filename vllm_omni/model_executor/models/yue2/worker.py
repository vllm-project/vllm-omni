# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2 V2 worker: release AR slots while retaining acoustic jobs."""

import time
from dataclasses import fields, replace
from typing import Any

import torch
from vllm.sequence import IntermediateTensors
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from vllm_omni.worker.gpu_ar_worker import GPUARWorker
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

from .runtime import Yue2ModelRunnerOutput, Yue2SchedulerOutput


class Yue2ModelRunner(OmniARModelRunner):
    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        if not isinstance(scheduler_output, Yue2SchedulerOutput):
            return super().finish_requests(scheduler_output)
        # An automatic ABC -> semantic transition reuses the request ID but
        # requires fresh model state and sampling parameters for the new phase.
        super().finish_requests(
            replace(
                scheduler_output,
                finished_req_ids=scheduler_output.finished_req_ids | scheduler_output.replaced_req_ids,
            )
        )
        # Detachment only releases the native slot. A genuine finish/abort above
        # also notifies the model, even if its AR slot was released earlier.
        for req_id in sorted(scheduler_output.acoustic_req_ids):
            self._remove_request(req_id)

    def _with_audio(self, output: ModelRunnerOutput | AsyncModelRunnerOutput) -> Yue2ModelRunnerOutput:
        if isinstance(output, AsyncModelRunnerOutput):
            output = output.get_output()
        # V2 also attaches transient attributes (e.g. _async_chunk) which are
        # not dataclass constructor arguments. Preserve the declared fields.
        result = Yue2ModelRunnerOutput(**{item.name: getattr(output, item.name) for item in fields(output)})
        result.acoustic_ready = self.model.poll_audio()
        result.acoustic_started = list(self.model.acoustic_started)
        self.model.acoustic_started.clear()
        return result

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        is_profile: bool = False,
        context_len: int = 0,
        valid_dummy_state_slots: bool = False,
        randomize_inputs: bool = False,
    ) -> Any:
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
        if output is not None and not dummy_run:
            output = self._with_audio(output)
            if not scheduler_output.total_num_scheduled_tokens and not output.acoustic_ready:
                # Dedicated worker thread, matching the existing acoustic poll
                # cadence without busy-spinning while no AR work is scheduled.
                time.sleep(0.001)
        return output

    @torch.inference_mode()
    def sample_tokens(self, grammar_output: GrammarOutput | None) -> Yue2ModelRunnerOutput | None:
        output = super().sample_tokens(grammar_output)
        return None if output is None else self._with_audio(output)


class Yue2Worker(GPUARWorker):
    model_runner_cls = Yue2ModelRunner
