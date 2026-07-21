# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stepwise diffusion runner using state-based pipeline atoms."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch.profiler import record_function

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.interface import StagePayload, supports_step_execution
from vllm_omni.diffusion.sched.interface import DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import (
    BatchRunnerOutput,
    DiffusionRequestState,
    RunnerOutput,
    attach_stage_durations,
    clear_pipeline_stage_durations,
    consume_pipeline_stage_durations,
    merge_stage_durations,
)
from vllm_omni.platforms import current_omni_platform


@dataclass
class _StepRequestView:
    request_id: str
    prompt: object
    sampling_params: object
    kv_sender_info: dict | None


class DiffusionModelRunnerV2(DiffusionModelRunner):
    """Runner-side execution for diffusion v2 atoms.

    The runner owns state cache, batching, step loop, scatter/gather, metrics,
    and local stage execution. The pipeline owns model math and state atoms.
    """

    def supports_step_mode(self) -> bool:
        return self.pipeline is not None and supports_step_execution(self.pipeline)

    def run_encode_stage(
        self,
        state: DiffusionRequestState,
        payload: StagePayload | None = None,
    ) -> DiffusionRequestState:
        """Build a DiT-ready state from local request data or a received payload."""
        clear_pipeline_stage_durations(self.pipeline)
        with record_function("pipeline_encode_stage"):
            state = self.pipeline.init_state(state)
            if payload is None:
                state = self.pipeline.check_inputs(state)
                state = self.pipeline.encode(state)
            else:
                state = self.pipeline.unpack_stage_state(payload, state)
            state = self.pipeline.prepare(state)
        merge_stage_durations(
            state,
            consume_pipeline_stage_durations(self.pipeline),
        )
        return state

    def _update_states(
        self, scheduler_output: DiffusionSchedulerOutput
    ) -> tuple[list[DiffusionRequestState], list[str]]:
        """Step-before update: cleanup finished requests and resolve running states."""
        for request_id in scheduler_output.finished_req_ids:
            self.state_cache.pop(request_id, None)

        resolved: list[DiffusionRequestState] = []
        new_request_ids: list[str] = []
        for sched_new_req in scheduler_output.scheduled_new_reqs:
            request_id = sched_new_req.request_id
            new_request_ids.append(request_id)
            if request_id in self.state_cache:
                raise ValueError(f"Received duplicate new-request payload for cached request {request_id}.")
            req = sched_new_req.req
            sampling = deepcopy(req.sampling_params)
            new_state = DiffusionRequestState(
                request_id=request_id,
                sampling=sampling,
                prompt=getattr(req, "prompt", None),
                kv_sender_info=getattr(req, "kv_sender_info", None),
            )
            state_req = _StepRequestView(
                request_id=request_id,
                prompt=new_state.prompt,
                sampling_params=sampling,
                kv_sender_info=new_state.kv_sender_info,
            )
            self.kv_transfer_manager.receive_multi_kv_cache_distributed(
                state_req,
                cfg_kv_collect_func=getattr(self.od_config, "cfg_kv_collect_func", None),
                target_device=self.target_device,
            )
            self.state_cache[request_id] = new_state
            resolved.append(new_state)

        for request_id in scheduler_output.scheduled_cached_reqs.request_ids:
            state = self.state_cache.get(request_id)
            if state is None:
                raise ValueError(f"Missing cached state for request {request_id}.")
            resolved.append(state)

        return resolved, new_request_ids

    def _prepare_generator(self, state: DiffusionRequestState) -> None:
        if state.sampling.generator is not None or state.sampling.seed is None:
            return
        if state.sampling.generator_device is not None:
            gen_device = state.sampling.generator_device
        elif self.device.type == "cpu":
            gen_device = "cpu"
        else:
            gen_device = self.device
        state.sampling.generator = torch.Generator(device=gen_device).manual_seed(state.sampling.seed)

    def _prepare_batch_inputs(self, states: list[DiffusionRequestState], new_request_ids: list[str]) -> InputBatch:
        for state_index, state in enumerate(states):
            if state.request_id not in new_request_ids:
                continue
            self._prepare_generator(state)
            prepared_state = self.run_encode_stage(state)
            if prepared_state.request_id != state.request_id:
                raise ValueError("Pipeline atom changed request_id during encode stage.")
            states[state_index] = prepared_state
            self.state_cache[prepared_state.request_id] = prepared_state

        input_batch = self.pipeline.build_step_batch(
            states,
            cached_batch=getattr(self, "input_batch", None),
        )
        self.input_batch = input_batch
        return input_batch

    def _update_states_after(
        self,
        states: list[DiffusionRequestState],
        input_batch: InputBatch,
        interrupted: bool = False,
    ) -> None:
        self.input_batch = input_batch

        for state in states:
            if interrupted or state.request_denoise_completed:
                self.state_cache.pop(state.request_id, None)

    def run_denoise_stage(
        self,
        input_batch: InputBatch,
    ) -> tuple[dict[str, RunnerOutput], bool]:
        """Run one DiT denoise step and request-local scheduler updates."""
        clear_pipeline_stage_durations(self.pipeline)
        with record_function("pipeline_denoise_stage"):
            noise_pred = self.pipeline.denoise_step(input_batch)

        stage_results: dict[str, DiffusionOutput] = {}
        if noise_pred is None and getattr(self.pipeline, "interrupt", False):
            for state in input_batch.states:
                stage_results[state.request_id] = DiffusionOutput(error="stepwise denoise interrupted")
            pipeline_interrupted = True
        elif noise_pred is None:
            raise RuntimeError("denoise_step returned None without pipeline interrupt.")
        else:
            pipeline_interrupted = False
            offset = 0
            for state in input_batch.states:
                if state.latents is None:
                    raise ValueError(f"Request {state.request_id} has no latents for denoise scheduling.")
                row_num = state.latents.shape[0]
                step_noise_pred = noise_pred[offset : offset + row_num]
                updated_state = self.pipeline.step_scheduler(state, step_noise_pred)
                if updated_state is not state:
                    raise RuntimeError("step_scheduler must update the existing DiffusionRequestState.")
                offset += row_num

            if offset != noise_pred.shape[0]:
                raise ValueError(
                    f"Stepwise noise_pred consumed {offset} rows, "
                    f"but batched noise_pred has {noise_pred.shape[0]} rows."
                )

        denoise_stage_durations = consume_pipeline_stage_durations(self.pipeline)
        for state in input_batch.states:
            merge_stage_durations(state, denoise_stage_durations)

        states_by_id = {state.request_id: state for state in input_batch.states}
        stage_outputs: dict[str, RunnerOutput] = {}
        for request_id, result in stage_results.items():
            state = states_by_id.get(request_id)
            if state is not None:
                attach_stage_durations(state, result)
            stage_outputs[request_id] = RunnerOutput(
                request_id=request_id,
                step_index=state.step_index if state is not None else 0,
                finished=True,
                result=result,
            )
        return stage_outputs, pipeline_interrupted

    def run_decode_stage(
        self,
        state: DiffusionRequestState,
        payload: StagePayload | None = None,
    ) -> DiffusionOutput:
        """Decode final/chunk latents through pipeline atoms."""
        clear_pipeline_stage_durations(self.pipeline)
        with record_function("pipeline_decode_stage"):
            if payload is not None:
                state = self.pipeline.unpack_stage_state(payload, state)
            updated_state = self.pipeline.decode(state)
            if updated_state is not state:
                raise RuntimeError("decode must update the existing DiffusionRequestState.")
            result = self.pipeline.postprocess(state)
        merge_stage_durations(
            state,
            consume_pipeline_stage_durations(self.pipeline),
        )
        attach_stage_durations(state, result)
        return result

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> BatchRunnerOutput:
        """Execute one scheduled diffusion step batch."""
        assert self.pipeline is not None, "Model not loaded. Call load_model() first."
        if not self.supports_step_mode():
            raise ValueError("Current pipeline does not support step execution.")
        if self.od_config.cache_backend not in (None, "none"):
            raise ValueError("Step mode does not support cache_backend yet.")

        use_hsdp = self.od_config.parallel_config.use_hsdp
        grad_context = torch.no_grad() if use_hsdp else torch.inference_mode()
        with grad_context:
            had_active_states = bool(self.state_cache)
            states, new_request_ids = self._update_states(scheduler_output)
            if not states:
                return BatchRunnerOutput.from_list([])
            is_primary = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if new_request_ids and not had_active_states and is_primary and current_omni_platform.is_available():
                current_omni_platform.reset_peak_memory_stats()
            input_batch = self._prepare_batch_inputs(states, new_request_ids)
            attn_metadata = self.pipeline.build_step_attention_metadata(input_batch)

            with set_forward_context(
                vllm_config=self.vllm_config,
                omni_diffusion_config=self.od_config,
                attn_metadata=attn_metadata,
            ):
                denoise_outputs, pipeline_interrupted = self.run_denoise_stage(input_batch)
                runner_output_list: list[RunnerOutput] = []
                for state in states:
                    denoise_output = denoise_outputs.get(state.request_id)
                    if denoise_output is not None:
                        runner_output_list.append(denoise_output)
                        continue

                    should_decode = (
                        state.chunk_denoise_completed if self.od_config.streaming_output else state.denoise_completed
                    )

                    result = self.run_decode_stage(state) if should_decode else None
                    finished = (
                        state.request_denoise_completed if self.od_config.streaming_output else state.denoise_completed
                    )
                    runner_output_list.append(
                        RunnerOutput(
                            request_id=state.request_id,
                            step_index=state.step_index,
                            finished=finished,
                            result=result,
                        )
                    )

                if is_primary:
                    batch_peak_memory_mb = self._sample_peak_memory_mb()
                    states_by_id = {state.request_id: state for state in states}
                    for state in states:
                        state.peak_memory_mb = max(state.peak_memory_mb, batch_peak_memory_mb)
                    for runner_output in runner_output_list:
                        if runner_output.result is None:
                            continue
                        state = states_by_id.get(runner_output.request_id)
                        if state is None:
                            continue
                        runner_output.result.peak_memory_mb = max(
                            runner_output.result.peak_memory_mb,
                            state.peak_memory_mb,
                        )

                self._update_states_after(states, input_batch, pipeline_interrupted)
                return BatchRunnerOutput.from_list(runner_output_list)
