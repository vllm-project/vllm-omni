# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned Realization state through the official diffusion session lifecycle RPC."""

from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.inputs.data import OmniInteractionPrompt


class DuoModelRunner(DiffusionModelRunner):
    def load_model(self, *args, **kwargs):
        from vllm.utils.gc_utils import freeze_gc_heap

        super().load_model(*args, **kwargs)
        freeze_gc_heap()

    def _prepare_batch_inputs(self, states, new_request_ids):
        from vllm_omni.diffusion.data import DiffusionOutput
        from vllm_omni.diffusion.worker.diffusion_model_runner import _dit_any_rank_failed
        from vllm_omni.diffusion.worker.utils import RunnerOutput

        live, aborted = [], []
        for state in states:
            c = state.prepared_layout
            sid, fence = getattr(c, "session_id", None), getattr(c, "fence", None)
            retired = type(fence) is int and self.pipeline.retired_fences.get(sid, -1) >= fence
            # Cancellation can retire the executing chunk before the engine
            # dequeues an already-prefetched successor. Agree across TP ranks
            # before skipping preparation, so no stale KV or collectives escape.
            if _dit_any_rank_failed(retired):
                self.state_cache.pop(state.request_id, None)
                aborted.append(
                    RunnerOutput(
                        request_id=state.request_id,
                        step_index=state.step_index,
                        finished=True,
                        result=DiffusionOutput(aborted=True, abort_message="Realization epoch retired"),
                    )
                )
            else:
                live.append(state)
        prepared, batch, errors = super()._prepare_batch_inputs(live, new_request_ids)
        return prepared, batch, aborted + errors

    def _execute_stepwise_core(self, *args, **kwargs):
        output = super()._execute_stepwise_core(*args, **kwargs)
        deferred = self.pipeline.options.get("defer_clean_until_codec", False)
        if not deferred and not self.pipeline.options.get("joint_execution", False):
            return output
        from vllm_omni.diffusion.worker.utils import StepWait

        for row in output.runner_outputs:
            state = self.state_cache.get(row.request_id)
            deferred = state is not None and state.sampling.extra_args.get(
                "realization_defer_clean_until_codec", self.pipeline.options.get("defer_clean_until_codec", False)
            )
            if (
                state is not None
                and not row.finished
                and state.extra.get("realization_latent_ready_emitted")
                and not state.extra["realization_ack"]
                and (
                    deferred
                    or (
                        self.pipeline.options.get("joint_execution", False)
                        and not state.extra["realization_condition"].final
                    )
                )
            ):
                row.wait_for_interaction = (
                    StepWait() if deferred else StepWait(self.pipeline.options.get("joint_wait_ms", 8.0) / 1000)
                )
        return output

    def submit_interaction(self, request_id: str, interaction: OmniInteractionPrompt) -> None:
        resume = interaction["event"].get("realization_resume_clean")
        if resume is None:
            return super().submit_interaction(request_id, interaction)
        state = self.state_cache[request_id]
        c = state.extra["realization_condition"]
        if (
            not state.sampling.extra_args.get(
                "realization_defer_clean_until_codec", self.pipeline.options.get("defer_clean_until_codec", False)
            )
            or (c.session_id, c.fence, c.block) != tuple(resume)
            or self.pipeline.retired_fences.get(c.session_id, -1) >= c.fence
            or not state.extra.get("realization_latent_ready_emitted")
            or state.extra["realization_ack"]
        ):
            raise ValueError("Stale or invalid deferred-clean release")
        # A successful official interaction RPC wakes StepScheduler, including
        # the race where the VAE finished before StepWait was installed.

    def close_session(self, session_id):
        session = self.pipeline.sessions.get(session_id)
        if session is not None:
            self.pipeline.close_session(session_id, session.fence)
        # Queued steps can outlive the close RPC in serial and joint execution.
        # Keep the fence so they are aborted before touching released KV.

    def reset_session(self, session_id):
        self.close_session(session_id)
