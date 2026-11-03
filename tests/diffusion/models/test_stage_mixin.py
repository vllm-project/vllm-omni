# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.config.stage_config import DiffusionStageOperator, DiffusionStageRole
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.models.stage_mixin import DiffusionStageMixin
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("stage_role", "prepare_operator", "output_operator"),
    [
        (
            DiffusionStageRole.FULL,
            DiffusionStageOperator.PREPARE_ENCODE,
            DiffusionStageOperator.POST_DECODE,
        ),
        (
            DiffusionStageRole.DENOISE_DECODE,
            DiffusionStageOperator.PREPARE_DENOISE,
            DiffusionStageOperator.POST_DECODE,
        ),
        (
            DiffusionStageRole.DENOISE,
            DiffusionStageOperator.PREPARE_DENOISE,
            DiffusionStageOperator.STEP_OUTPUT,
        ),
    ],
)
def test_step_mode_requires_role_specific_preparation(stage_role, prepare_operator, output_operator):
    assert stage_role.required_operators(step_execution=True) == {
        prepare_operator,
        DiffusionStageOperator.DENOISE_STEP,
        DiffusionStageOperator.STEP_SCHEDULER,
        output_operator,
    }


@pytest.mark.parametrize("stage_role", list(DiffusionStageRole))
def test_request_mode_requires_only_the_default_operator(stage_role):
    assert stage_role.required_operators(step_execution=False) == {stage_role.default_request_operator}


@pytest.mark.parametrize("stage_role", [DiffusionStageRole.ENCODE, DiffusionStageRole.DECODE])
def test_encoder_and_decoder_remain_atomic_in_step_mode(stage_role):
    assert stage_role.required_operators(step_execution=True) == {stage_role.default_request_operator}


def _request_batch(*, num_requests=1, encoded=True, latents=False) -> DiffusionRequestBatch:
    requests = []
    for index in range(num_requests):
        prompt = {"prompt": f"prompt-{index}"}
        if encoded:
            prompt["prompt_embeds"] = torch.full((1, 2), float(index + 1))
        if latents:
            prompt["latents"] = torch.full((1, 2), float(index + 11))
        requests.append(
            SimpleNamespace(
                request_id=f"request-{index}",
                is_dummy_run_request_id=lambda request_id: False,
                prompt=prompt,
                sampling_params=OmniDiffusionSamplingParams(num_outputs_per_prompt=1, output_type="latent"),
            )
        )
    return DiffusionRequestBatch(requests=requests)


def _step_state() -> StepRequestState:
    return StepRequestState(
        request_id="request-0",
        prompt="prompt",
        sampling=OmniDiffusionSamplingParams(),
        latents=torch.zeros(1),
        timesteps=torch.tensor([1.0]),
    )


def _input_batch(state: StepRequestState) -> InputBatch:
    batch = object.__new__(InputBatch)
    batch.states = (state,)
    return batch


class _RequestOperations:
    def __init__(self, stage_role: DiffusionStageRole | str, *, step_execution=False):
        self.stage_role = stage_role
        self.calls = []
        self.inputs = {}
        self.outputs = {}
        self.initialize_stage_execution(step_execution=step_execution)

    def encode_batch(self, batch):
        self.calls.append("encode")
        self.inputs["encode"] = batch
        outputs = [
            DiffusionOutput(custom_output={"prompt_embeds": torch.full((1, 2), float(index + 1))})
            for index in range(batch.num_reqs)
        ]
        self.outputs["encode"] = outputs
        return outputs

    def diffuse_batch(self, batch):
        self.calls.append("diffuse")
        self.inputs["diffuse"] = batch
        outputs = []
        for prompt in batch.prompts:
            condition = DiffusionRequestBatch.get_prompt_field(prompt, "prompt_embeds")
            assert isinstance(condition, torch.Tensor)
            outputs.append(DiffusionOutput(custom_output={"latents": condition + 10}))
        self.outputs["diffuse"] = outputs
        return outputs

    def decode_batch(self, batch):
        self.calls.append("decode")
        self.inputs["decode"] = batch
        outputs = []
        for prompt in batch.prompts:
            latents = DiffusionRequestBatch.get_prompt_field(prompt, "latents")
            assert isinstance(latents, torch.Tensor)
            outputs.append(DiffusionOutput(output=latents * 2, stage_durations=getattr(self, "stage_durations", {})))
        self.outputs["decode"] = outputs
        return outputs


class _RequestPipeline(_RequestOperations, DiffusionStageMixin):
    def forward(self, batch):
        del batch
        self.calls.append("forward")
        return [DiffusionOutput(output="forward")]


class _ComposedRequestPipeline(_RequestOperations, DiffusionStageMixin):
    pass


@pytest.mark.parametrize(
    ("stage_role", "expected_calls"),
    [
        (DiffusionStageRole.ENCODE, ["encode"]),
        (DiffusionStageRole.DENOISE, ["diffuse"]),
        (DiffusionStageRole.DENOISE_DECODE, ["diffuse", "decode"]),
        (DiffusionStageRole.DECODE, ["decode"]),
        (DiffusionStageRole.FULL, ["forward"]),
    ],
)
def test_default_request_dispatch_preserves_role_compatibility(stage_role, expected_calls):
    pipeline = _RequestPipeline(stage_role)

    outputs = pipeline.run_stage(_request_batch(latents=True))

    assert pipeline.calls == expected_calls
    assert len(outputs) == 1
    if stage_role is DiffusionStageRole.FULL:
        assert outputs[0].output == "forward"
    elif stage_role is DiffusionStageRole.DENOISE:
        assert torch.equal(outputs[0].custom_output["latents"], torch.full((1, 2), 11.0))
    elif stage_role is DiffusionStageRole.ENCODE:
        assert torch.equal(outputs[0].custom_output["prompt_embeds"], torch.ones(1, 2))
    else:
        assert torch.equal(outputs[0].output, torch.full((1, 2), 22.0))


@pytest.mark.parametrize("stage_role", [DiffusionStageRole.FULL, DiffusionStageRole.DENOISE_DECODE])
def test_explicit_diffuse_returns_latents_without_decoding_even_when_vae_is_loaded(stage_role):
    pipeline = _RequestPipeline(stage_role)
    pipeline.vae = object()

    outputs = pipeline.run_stage(_request_batch(), operator=DiffusionStageOperator.DIFFUSE)

    assert pipeline.calls == ["diffuse"]
    assert outputs[0].output is None
    assert torch.equal(outputs[0].custom_output["latents"], torch.full((1, 2), 11.0))


@pytest.mark.parametrize(
    ("operator", "expected_call"),
    [
        (DiffusionStageOperator.ENCODE, "encode"),
        (DiffusionStageOperator.DIFFUSE, "diffuse"),
        (DiffusionStageOperator.DECODE, "decode"),
    ],
)
def test_full_role_allows_explicit_request_operations_beyond_its_default(operator, expected_call):
    pipeline = _RequestPipeline(DiffusionStageRole.FULL)

    pipeline.run_stage(_request_batch(latents=True), operator=operator)

    assert pipeline.calls == [expected_call]


@pytest.mark.parametrize("stage_role", [DiffusionStageRole.FULL, DiffusionStageRole.DENOISE_DECODE])
def test_explicit_diffuse_decode_composes_only_generator_and_decoder(stage_role):
    pipeline = _RequestPipeline(stage_role)

    outputs = pipeline.run_stage(_request_batch(), operator=DiffusionStageOperator.DIFFUSE_DECODE)

    assert pipeline.calls == ["diffuse", "decode"]
    assert torch.equal(outputs[0].output, torch.full((1, 2), 22.0))


@pytest.mark.parametrize(
    ("stage_role", "operator"),
    [
        (DiffusionStageRole.ENCODE, DiffusionStageOperator.DIFFUSE),
        (DiffusionStageRole.ENCODE, DiffusionStageOperator.DECODE),
        (DiffusionStageRole.DENOISE, DiffusionStageOperator.ENCODE),
        (DiffusionStageRole.DENOISE, DiffusionStageOperator.DECODE),
        (DiffusionStageRole.DENOISE, DiffusionStageOperator.DIFFUSE_DECODE),
        (DiffusionStageRole.DENOISE_DECODE, DiffusionStageOperator.ENCODE),
        (DiffusionStageRole.DECODE, DiffusionStageOperator.DIFFUSE),
    ],
)
def test_role_rejects_unauthorized_operation_before_model_call(stage_role, operator):
    pipeline = _RequestPipeline(stage_role)

    with pytest.raises(ValueError, match=stage_role.value):
        pipeline.run_stage(_request_batch(latents=True), operator=operator)

    assert pipeline.calls == []


def test_composed_forward_hands_off_tensors_without_mutating_requests():
    pipeline = _ComposedRequestPipeline(DiffusionStageRole.FULL)
    batch = _request_batch(num_requests=2, encoded=False)
    original_prompts = [request.prompt for request in batch.requests]
    original_sampling = [request.sampling_params for request in batch.requests]

    outputs = pipeline.run_stage(batch)

    assert pipeline.calls == ["encode", "diffuse", "decode"]
    for index, request in enumerate(batch.requests):
        assert request.prompt is original_prompts[index]
        assert request.prompt == {"prompt": f"prompt-{index}"}
        assert request.sampling_params is original_sampling[index]
        generator_request = pipeline.inputs["diffuse"].requests[index]
        decoder_request = pipeline.inputs["decode"].requests[index]
        assert generator_request.request_id == decoder_request.request_id == request.request_id
        assert generator_request.sampling_params is decoder_request.sampling_params is request.sampling_params
        assert (
            DiffusionRequestBatch.get_prompt_field(generator_request.prompt, "prompt_embeds")
            is pipeline.outputs["encode"][index].custom_output["prompt_embeds"]
        )
        assert (
            DiffusionRequestBatch.get_prompt_field(decoder_request.prompt, "latents")
            is pipeline.outputs["diffuse"][index].custom_output["latents"]
        )
        assert torch.equal(outputs[index].output, torch.full((1, 2), float((index + 11) * 2)))


@pytest.mark.parametrize("stage_role", list(DiffusionStageRole))
def test_inherited_forward_uses_the_default_dispatch_without_recursion(stage_role):
    pipeline = _ComposedRequestPipeline(stage_role)

    outputs = pipeline.forward(_request_batch(latents=True))

    assert len(outputs) == 1
    assert (
        pipeline.calls
        == {
            DiffusionStageRole.ENCODE: ["encode"],
            DiffusionStageRole.DENOISE: ["diffuse"],
            DiffusionStageRole.DECODE: ["decode"],
            DiffusionStageRole.DENOISE_DECODE: ["diffuse", "decode"],
            DiffusionStageRole.FULL: ["encode", "diffuse", "decode"],
        }[stage_role]
    )


def test_native_forward_remains_the_pipeline_implementation():
    pipeline = _RequestPipeline(DiffusionStageRole.FULL)

    assert pipeline.forward(_request_batch())[0].output == "forward"
    assert pipeline.calls == ["forward"]


@pytest.mark.parametrize("operator", [DiffusionStageOperator.ENCODE, DiffusionStageOperator.DIFFUSE])
def test_missing_optional_operation_fails_at_call_without_falling_back_to_forward(operator):
    class ForwardOnlyPipeline(DiffusionStageMixin):
        def forward(self, batch):
            pytest.fail("An explicit operation must not fall back to native forward")

    pipeline = ForwardOnlyPipeline()
    pipeline.stage_role = DiffusionStageRole.FULL
    pipeline.initialize_stage_execution(step_execution=False)

    with pytest.raises(NotImplementedError, match=operator.value):
        pipeline.run_stage(_request_batch(), operator=operator)


def test_decode_preserves_pipeline_stage_durations():
    pipeline = _RequestPipeline(DiffusionStageRole.DECODE)
    pipeline.stage_durations = {"vae.decode": 1.25}

    outputs = pipeline.run_stage(_request_batch(latents=True))

    assert outputs[0].stage_durations == {"vae.decode": 1.25}


class _StepPipeline(_RequestPipeline):
    supports_step_execution = True

    def __init__(self, stage_role=DiffusionStageRole.DENOISE, *, step_execution=True):
        self.seen_states = None
        super().__init__(stage_role, step_execution=step_execution)

    def prepare_encode(self, state, **kwargs):
        del kwargs
        self.calls.append("prepare_encode")
        return state

    def prepare_denoise(self, state, **kwargs):
        del kwargs
        self.calls.append("prepare_denoise")
        return state

    def denoise_step(self, batch, *, states=None, **kwargs):
        del batch, kwargs
        self.calls.append("denoise_step")
        self.seen_states = states
        return torch.ones(1)

    def step_scheduler(self, state, noise_pred, **kwargs):
        del kwargs
        self.calls.append("step_scheduler")
        state.latents = state.latents + noise_pred
        state.step_index += 1

    def post_decode(self, state, **kwargs):
        del kwargs
        self.calls.append("post_decode")
        return DiffusionOutput(output=state.latents * 2)


def test_full_request_mode_does_not_block_an_implemented_denoise_step():
    pipeline = _StepPipeline(DiffusionStageRole.FULL, step_execution=False)
    batch = _input_batch(_step_state())

    result = pipeline.run_stage(batch, operator=DiffusionStageOperator.DENOISE_STEP, states=batch.states)

    assert torch.equal(result, torch.ones(1))
    assert pipeline.calls == ["denoise_step"]
    assert pipeline.seen_states is batch.states


@pytest.mark.parametrize(
    "stage_role", [DiffusionStageRole.FULL, DiffusionStageRole.DENOISE, DiffusionStageRole.DENOISE_DECODE]
)
def test_prepare_denoise_does_not_invoke_encoder_preparation(stage_role):
    pipeline = _StepPipeline(stage_role)
    state = _step_state()

    result = pipeline.run_stage(state, operator=DiffusionStageOperator.PREPARE_DENOISE)

    assert result is state
    assert pipeline.calls == ["prepare_denoise"]


def test_prepare_encode_keeps_the_combined_preparation_contract():
    pipeline = _StepPipeline(DiffusionStageRole.FULL)
    state = _step_state()

    result = pipeline.run_stage(state, operator=DiffusionStageOperator.PREPARE_ENCODE)

    assert result is state
    assert pipeline.calls == ["prepare_encode"]


@pytest.mark.parametrize(
    ("stage_role", "expected_prepare", "expected_finalize"),
    [
        (DiffusionStageRole.FULL, "prepare_encode", "post_decode"),
        (DiffusionStageRole.DENOISE_DECODE, "prepare_denoise", "post_decode"),
        (DiffusionStageRole.DENOISE, "prepare_denoise", "step_output"),
    ],
)
def test_run_stage_selects_role_specific_step_methods(stage_role, expected_prepare, expected_finalize):
    pipeline = _StepPipeline(stage_role)
    state = _step_state()

    pipeline.run_stage(state, operator=DiffusionStageOperator.STEP_PREPARE)
    output = pipeline.run_stage(state, operator=DiffusionStageOperator.STEP_FINALIZE)

    assert pipeline.calls == [expected_prepare] + ([] if expected_finalize == "step_output" else [expected_finalize])
    if expected_finalize == "step_output":
        assert output.output is None
        torch.testing.assert_close(output.custom_output["latents"], state.latents)


@pytest.mark.parametrize("stage_role", [DiffusionStageRole.DENOISE, DiffusionStageRole.DENOISE_DECODE])
def test_prepare_encode_requires_encoder_permission(stage_role):
    pipeline = _StepPipeline(stage_role)

    with pytest.raises(ValueError, match=stage_role.value):
        pipeline.run_stage(_step_state(), operator=DiffusionStageOperator.PREPARE_ENCODE)

    assert pipeline.calls == []


def test_scheduler_operation_updates_only_the_supplied_request_state():
    pipeline = _StepPipeline(DiffusionStageRole.FULL)
    state = _step_state()

    pipeline.run_stage(state, operator=DiffusionStageOperator.STEP_SCHEDULER, noise_pred=torch.ones(1))

    assert pipeline.calls == ["step_scheduler"]
    assert state.step_index == 1
    assert torch.equal(state.latents, torch.ones(1))


@pytest.mark.parametrize("stage_role", [DiffusionStageRole.FULL, DiffusionStageRole.DENOISE_DECODE])
def test_post_decode_consumes_one_request_state(stage_role):
    pipeline = _StepPipeline(stage_role)
    state = _step_state()

    result = pipeline.run_stage(state, operator=DiffusionStageOperator.POST_DECODE)

    assert torch.equal(result.output, state.latents * 2)
    assert pipeline.calls == ["post_decode"]


def test_step_output_transports_latents_without_decode():
    pipeline = _StepPipeline()
    state = _step_state()

    result = pipeline.run_stage(state, operator=DiffusionStageOperator.STEP_OUTPUT)

    assert result.output is None
    torch.testing.assert_close(result.custom_output["latents"], state.latents)
    assert pipeline.calls == []


@pytest.mark.parametrize(
    ("operator", "payload_type"),
    [
        (DiffusionStageOperator.ENCODE, "state"),
        (DiffusionStageOperator.DIFFUSE, "input_batch"),
        (DiffusionStageOperator.DECODE, "state"),
        (DiffusionStageOperator.PREPARE_ENCODE, "input_batch"),
        (DiffusionStageOperator.PREPARE_DENOISE, "input_batch"),
        (DiffusionStageOperator.DENOISE_STEP, "state"),
        (DiffusionStageOperator.STEP_SCHEDULER, "input_batch"),
        (DiffusionStageOperator.STEP_OUTPUT, "input_batch"),
        (DiffusionStageOperator.POST_DECODE, "input_batch"),
    ],
)
def test_explicit_operations_reject_the_wrong_payload_type_before_model_call(operator, payload_type):
    pipeline = _StepPipeline(DiffusionStageRole.FULL)
    state = _step_state()
    payload = state if payload_type == "state" else _input_batch(state)

    with pytest.raises(TypeError, match=operator.value):
        pipeline.run_stage(payload, operator=operator)

    assert pipeline.calls == []


def test_initialization_rejects_missing_required_step_methods():
    class UnsupportedStepPipeline(DiffusionStageMixin):
        stage_role = DiffusionStageRole.DENOISE
        supports_step_execution = True

    pipeline = UnsupportedStepPipeline()

    assert not supports_step_execution(pipeline)
    with pytest.raises(TypeError, match="required step-execution methods"):
        pipeline.initialize_stage_execution(step_execution=True)


def test_supports_step_execution_requires_the_runtime_protocol():
    assert not supports_step_execution(SimpleNamespace(supports_step_execution=True))
    assert supports_step_execution(_StepPipeline())


def test_request_pipeline_does_not_gain_step_support_from_stage_dispatch():
    pipeline = _RequestPipeline(DiffusionStageRole.DENOISE)

    with pytest.raises(TypeError, match="required step-execution methods"):
        pipeline.initialize_stage_execution(step_execution=True)


@pytest.mark.parametrize("role", list(DiffusionStageRole))
def test_empty_stage_request_batch_returns_without_model_calls(role):
    pipeline = _ComposedRequestPipeline(role)
    assert pipeline.run_stage(DiffusionRequestBatch(requests=[])) == []
    assert pipeline.calls == []


@pytest.mark.parametrize("method_name", ["forward", "forward_alias"])
def test_explicit_forward_mapping_rejects_recursive_mixin_entrypoint(method_name):
    class RecursivePipeline(DiffusionStageMixin):
        _stage_operator_methods = {DiffusionStageOperator.FORWARD: method_name}
        forward_alias = DiffusionStageMixin.forward

    pipeline = RecursivePipeline()
    pipeline.stage_role = DiffusionStageRole.FULL
    with pytest.raises(TypeError, match="FORWARD cannot map"):
        pipeline.initialize_stage_execution()
