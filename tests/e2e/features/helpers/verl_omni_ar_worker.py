# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""verl-omni AR worker-extension mirror and runner probes for RLHF E2E tests.

Lives in its own importable module (like ``qwen3_omni_merge_probe``) because
``worker_extension_cls`` is resolved by qualified name inside the stage worker
process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from tests.e2e.features.helpers.custom_pipeline import set_death_signal

if TYPE_CHECKING:
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner


def _tensor_to_bytes(tensor: torch.Tensor) -> dict[str, Any]:
    """msgspec-safe CPU copy of a tensor (collective_rpc results cross msgpack)."""
    flat = tensor.detach().to("cpu").contiguous().flatten()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": flat.view(torch.uint8).numpy().tobytes(),
    }


def tensor_from_bytes(payload: dict[str, Any]) -> torch.Tensor:
    """Inverse of :func:`_tensor_to_bytes`."""
    flat = torch.frombuffer(bytearray(payload["data"]), dtype=getattr(torch, payload["dtype"]))
    return flat.reshape(payload["shape"])


class vLLMOmniColocateWorkerExtensionForARTest:
    """Mirror of ``vLLMOmniColocateWorkerExtension`` (verl-omni) on the AR worker.

    The production ``__new__`` installs verl's death signal on every vLLM worker
    process and enables the teacher hidden-state transport. That transport is
    native on the worker-v2 runner, so only the death signal is replicated.
    """

    model_runner: OmniARModelRunner
    _recorded_steps: list[dict[str, Any]] | None = None

    def __new__(cls, **kwargs):
        set_death_signal()
        return super().__new__(cls)

    # -- Test-only probes, reached through collective_rpc. --

    def runner_class(self) -> str:
        runner_type = type(self.model_runner)
        return f"{runner_type.__module__}.{runner_type.__qualname__}"

    def start_hidden_state_recording(self) -> bool:
        """Record each step's forward output and batch layout before sampling."""
        runner = self.model_runner
        sample_tokens = runner.sample_tokens

        def recording_sample_tokens(grammar_output):
            self._record_step()
            return sample_tokens(grammar_output)

        self._recorded_steps = []
        runner.sample_tokens = recording_sample_tokens
        return True

    def take_hidden_state_recording(self) -> list[dict[str, Any]]:
        steps = self._recorded_steps or []
        self._recorded_steps = None
        return steps

    def _record_step(self) -> None:
        steps = self._recorded_steps
        if steps is None:
            return
        state = self.model_runner.execute_model_state
        if state is None:
            return
        batch = state.input_batch
        num_reqs = batch.num_reqs
        steps.append(
            {
                "req_ids": list(batch.req_ids[:num_reqs]),
                "query_start_loc": batch.query_start_loc_np[: num_reqs + 1].tolist(),
                "prefill_len": batch.prefill_len_np[:num_reqs].tolist(),
                "num_computed_prefill": batch.num_computed_prefill_tokens_np[:num_reqs].tolist(),
                "num_scheduled": batch.num_scheduled_tokens[:num_reqs].tolist(),
                "hidden": _tensor_to_bytes(state.hidden_states),
            }
        )
