# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import replace
from typing import Any, ClassVar

from vllm_omni.config.stage_config import (
    DiffusionStage,
    DiffusionStageOperator,
    DiffusionStageRole,
    resolve_diffusion_stage_role,
)
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState


class DiffusionStageMixin:
    """Authorize an operation by role, then dispatch to role-free model code.

    Request-level operators use the conventional pipeline entry points
    (``forward``, ``encode_batch``, and ``decode_batch``). Step-level operators
    use the methods from the existing step-execution protocol. A pipeline only
    needs ``_stage_operator_methods`` when it uses a non-standard method name.
    Explicit operators have the same meaning for every role. The default
    request operator is a compatibility policy, not the role's permission set.
    """

    supports_stage_operator_dispatch = True
    _stage_operator_methods: ClassVar[Mapping[DiffusionStageOperator, str]] = {}
    _stage_dummy_methods: ClassVar[Mapping[DiffusionStageOperator, str]] = {}
    _stage_operator_validators: ClassVar[Mapping[DiffusionStageOperator, str]] = {}

    def __init__(self, *args: Any, od_config: Any | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if od_config is not None:
            self.od_config = od_config
            self.stage_role = resolve_diffusion_stage_role(
                getattr(od_config, "stage_role", None),
                getattr(od_config, "model_stage", None),
            )
            self.initialize_stage_execution()

    def initialize_stage_execution(self, *, step_execution: bool | None = None) -> None:
        """Validate deployment requirements without restricting role permissions."""
        forward_name = self._stage_operator_methods.get(DiffusionStageOperator.FORWARD)
        if (
            forward_name is not None
            and getattr(getattr(self, forward_name, None), "__func__", None) is DiffusionStageMixin.forward
        ):
            raise TypeError(
                "FORWARD cannot map to DiffusionStageMixin.forward; override it or use default composition."
            )
        if step_execution is None:
            od_config = getattr(self, "od_config", None)
            step_execution = bool(
                getattr(od_config, "step_execution", False) or getattr(od_config, "streaming_output", False)
            )

        stage_role = self._diffusion_stage_role()
        operators = stage_role.required_operators(step_execution=step_execution)
        missing = []
        for operator in operators:
            missing.extend(self._missing_stage_methods(operator))

        if missing:
            mode = "step-execution" if step_execution else "request-batching"
            raise TypeError(
                f"{type(self).__name__} does not implement the required {mode} methods for "
                f"role {stage_role.value!r}: {', '.join(sorted(missing))}."
            )

        self._stage_execution_step_mode = step_execution
        self.stage_uses_step_execution = step_execution and stage_role.allows(DiffusionStage.DENOISE)
        self._stage_execution_initialized = True

    def forward(self, batch: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Keep direct pipeline calls on the same dispatch path as the runner."""
        return self.run_stage(batch)

    def run_stage(
        self,
        payload: DiffusionRequestBatch | InputBatch | StepRequestState,
        *,
        operator: DiffusionStageOperator | str | None = None,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run one role-authorized request-level or step-level operation."""
        stage_role = self._diffusion_stage_role()
        is_request_payload = isinstance(payload, DiffusionRequestBatch)
        operator = self._resolve_operator(stage_role, operator, is_request_payload)
        operator = self._resolve_step_boundary_operator(stage_role, operator)

        if operator not in stage_role.allowed_operators:
            raise ValueError(f"Operator {operator.value!r} is not allowed for role {stage_role.value!r}.")

        expected_type = (
            DiffusionRequestBatch
            if operator.is_request_operator
            else InputBatch
            if operator is DiffusionStageOperator.DENOISE_STEP
            else StepRequestState
        )
        if not isinstance(payload, expected_type):
            raise TypeError(f"{operator.value} requires {expected_type.__name__}, got {type(payload).__name__}.")

        missing = self._missing_stage_methods(operator)
        if missing:
            raise NotImplementedError(f"{type(self).__name__} does not implement {', '.join(missing)}.")

        if is_request_payload:
            outputs = self._run_request_operator(payload, operator, **kwargs)
            # Only a boundary output needs host materialization. Composed G->D
            # calls keep their intermediate latents on the producing device.
            if operator in (DiffusionStageOperator.ENCODE, DiffusionStageOperator.DIFFUSE):
                return [replace(out, to_cpu=True) if not out.to_cpu else out for out in outputs]
            return outputs

        if operator is DiffusionStageOperator.DENOISE_STEP and states is not None:
            kwargs["states"] = states

        return getattr(self, self._resolve_stage_method_name(operator))(payload, **kwargs)

    @staticmethod
    def _resolve_step_boundary_operator(
        stage_role: DiffusionStageRole, operator: DiffusionStageOperator
    ) -> DiffusionStageOperator:
        """Select role-specific methods for generic step boundaries."""
        if operator is DiffusionStageOperator.STEP_PREPARE:
            return (
                DiffusionStageOperator.PREPARE_ENCODE
                if stage_role.allows(DiffusionStage.ENCODE)
                else DiffusionStageOperator.PREPARE_DENOISE
            )
        if operator is DiffusionStageOperator.STEP_FINALIZE:
            return (
                DiffusionStageOperator.POST_DECODE
                if stage_role.allows(DiffusionStage.DECODE)
                else DiffusionStageOperator.STEP_OUTPUT
            )
        return operator

    def _request_operator_plan(self, operator: DiffusionStageOperator) -> tuple[DiffusionStageOperator, ...]:
        # A concrete mapping opts out of the generic composition. This keeps
        # compound operations extensible for pipelines whose native entry point
        # already owns the combined request contract.
        if operator in self._stage_operator_methods:
            return ()
        if operator is DiffusionStageOperator.DIFFUSE_DECODE:
            return (DiffusionStageOperator.DIFFUSE, DiffusionStageOperator.DECODE)
        if (
            operator is DiffusionStageOperator.FORWARD
            and self._resolve_stage_method_name(operator) == "forward"
            and type(self).forward is DiffusionStageMixin.forward
        ):
            return (DiffusionStageOperator.ENCODE, DiffusionStageOperator.DIFFUSE, DiffusionStageOperator.DECODE)
        return ()

    def _missing_stage_methods(self, operator: DiffusionStageOperator) -> list[str]:
        plan = self._request_operator_plan(operator)
        if plan:
            return [missing for op in plan for missing in self._missing_stage_methods(op)]
        name = self._resolve_stage_method_name(operator)
        return [] if callable(getattr(self, name, None)) else [f"{operator.value} ({name})"]

    def _run_request_operator(
        self, batch: DiffusionRequestBatch, operator: DiffusionStageOperator, **kwargs: Any
    ) -> list[DiffusionOutput]:
        if not batch.requests:
            return []
        plan = self._request_operator_plan(operator)
        if plan:
            for op in plan:
                validator_name = self._stage_operator_validators.get(op)
                if validator_name is not None:
                    getattr(self, validator_name)(batch)
            for index, op in enumerate(plan):
                outputs = self._run_request_operator(batch, op, **kwargs)
                if index < len(plan) - 1:
                    batch = self._batch_with_stage_payload(batch, outputs)
            return outputs

        prepare = getattr(self, "_prepare_stage_request", None)
        if operator is not DiffusionStageOperator.FORWARD and callable(prepare):
            prepare(batch, operator)
        is_dummy_run = callable(getattr(batch.requests[0], "is_dummy_run_request_id", None)) and batch.is_dummy_run()
        if is_dummy_run:
            self._prepare_dummy_stage_payload(batch, operator=operator)
        name = self._resolve_stage_method_name(operator)
        method = getattr(self, name)
        if name in ("encode", "decode"):
            return [method(req, **kwargs) for req in batch.requests]
        return method(batch, **kwargs)

    def _prepare_dummy_stage_payload(
        self,
        batch: DiffusionRequestBatch,
        *,
        operator: DiffusionStageOperator | None = None,
    ) -> None:
        """Prepare a dummy request through the selected operator adapter."""
        if operator is None:
            operator = self._diffusion_stage_role().default_request_operator
        method_name = self._stage_dummy_methods.get(operator)
        if method_name is not None:
            getattr(self, method_name)(batch)

    @staticmethod
    def _batch_with_stage_payload(
        batch: DiffusionRequestBatch, outputs: list[DiffusionOutput]
    ) -> DiffusionRequestBatch:
        """Overlay an in-process intermediate without copying tensors or inputs."""
        if len(outputs) != batch.num_reqs:
            raise ValueError(f"Stage produced {len(outputs)} outputs for {batch.num_reqs} requests.")
        requests = []
        for request, output in zip(batch.requests, outputs):
            staged_request = copy(request)
            prompt = dict(request.prompt) if isinstance(request.prompt, dict) else {"prompt": request.prompt}
            additional = dict(prompt.get("additional_information") or {})
            for name, value in output.custom_output.items():
                prompt.pop(name, None)
                additional[name] = value
            prompt["additional_information"] = additional
            staged_request.prompt = prompt
            requests.append(staged_request)
        return DiffusionRequestBatch(requests=requests)

    def step_output(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Return denoise latents for a downstream decode stage."""
        del kwargs
        if state.latents is None:
            raise ValueError("Denoise stage has no latents to transfer.")
        return DiffusionOutput(
            output=None,
            custom_output={"latents": state.latents},
            to_cpu=True,
        )

    def _resolve_stage_method_name(self, operator: DiffusionStageOperator) -> str:
        override = self._stage_operator_methods.get(operator)
        if override is not None:
            return override

        if operator is DiffusionStageOperator.FORWARD:
            return "forward"
        if operator is DiffusionStageOperator.ENCODE:
            return "encode_batch" if callable(getattr(self, "encode_batch", None)) else "encode"
        if operator is DiffusionStageOperator.DIFFUSE:
            return "diffuse_batch"
        if operator is DiffusionStageOperator.DECODE:
            return "decode_batch" if callable(getattr(self, "decode_batch", None)) else "decode"
        return operator.value

    def _resolve_operator(
        self,
        stage_role: DiffusionStageRole,
        operator: DiffusionStageOperator | str | None,
        is_request_payload: bool,
    ) -> DiffusionStageOperator:
        if operator is None:
            if not is_request_payload:
                raise ValueError("operator is required for step-level stage execution.")
            return stage_role.default_request_operator
        if isinstance(operator, DiffusionStageOperator):
            return operator
        try:
            return DiffusionStageOperator(operator)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown diffusion stage operator: {operator!r}.") from exc

    def _diffusion_stage_role(self) -> DiffusionStageRole:
        """Resolve the role from the pipeline or its stage configuration."""
        stage_role = getattr(self, "stage_role", None)
        od_config = getattr(self, "od_config", None)
        model_stage = getattr(self, "model_stage", None) or getattr(od_config, "model_stage", None)
        return resolve_diffusion_stage_role(stage_role, model_stage)
