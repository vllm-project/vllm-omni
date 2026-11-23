# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise OmniVoice chunk boundaries through the real runner and scheduler."""

import weakref
from types import SimpleNamespace

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import OmniVoicePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import DiffusionRequestStatus, StepScheduler
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.configs.omnivoice import OmniVoiceConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Generator:
    """Replace neural computation while retaining pipeline packing and state updates."""

    _cuda_graph_fwd = None

    def __init__(self):
        self.draws = []

    def _prepare_embeddings(self, input_ids, audio_masks):
        assert input_ids.shape[0] == audio_masks.shape[0]
        return input_ids

    def _transformer_forward(self, inputs_embeds, cu_seqs, max_seqlen):
        assert cu_seqs[-1].item() == inputs_embeds.shape[0]
        return inputs_embeds

    def _get_logits(self, hidden_states):
        return torch.zeros(8, hidden_states.shape[0], 1)

    def _unmask_one_request(self, c_logits, u_logits, tokens, *, generator, num_to_unmask, **kwargs):
        assert c_logits.shape[2] == u_logits.shape[2] == tokens.shape[-1]
        assert isinstance(num_to_unmask, int)
        self.draws.append((generator.initial_seed(), torch.rand((), generator=generator).item()))
        tokens.fill_(generator.initial_seed())


@pytest.fixture
def runtime(monkeypatch):
    # Only model loading and hardware accounting are replaced. Preparation,
    # packing, chunk transitions, runner cleanup and admission are real.
    pipeline = object.__new__(OmniVoicePipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.config = OmniVoiceConfig(audio_chunk_duration=8.0, audio_chunk_threshold=10.0, frame_rate=1)
    pipeline.device = torch.device("cpu")
    pipeline.sample_rate = 10
    pipeline.tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[1]))
    pipeline.duration_estimator = SimpleNamespace(estimate_duration=lambda text, ref_text, ref_len: len(text))
    for name in (
        "num_step",
        "guidance_scale",
        "t_shift",
        "layer_penalty_factor",
        "position_temperature",
        "class_temperature",
    ):
        setattr(pipeline, name, getattr(pipeline.config, name))
    pipeline.generator = _Generator()
    pipeline.decoder = lambda tokens: tokens[:, :1].float()
    for method in ("is_available", "max_memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(current_omni_platform, method, lambda: 0)
    monkeypatch.setattr(OmniDiffusionConfig, "_resolve_master_port", lambda self: 29500)
    config = OmniDiffusionConfig(model="test", step_execution=True, max_num_seqs=8)
    runner = DiffusionModelRunner(VllmConfig(device_config=DeviceConfig(device="cpu")), config, pipeline.device)
    runner.pipeline = pipeline
    runner.kv_transfer_manager = None  # No distributed KV transfer in this CPU test.
    scheduler = StepScheduler()
    scheduler.initialize(config)
    return pipeline, runner, scheduler


def _request(request_id, text, seed=3):
    return OmniDiffusionRequest(
        request_id=request_id,
        prompt=text,
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, seed=seed),
    )


def _step(runner, scheduler):
    scheduled = scheduler.schedule()
    output = runner.execute_stepwise(scheduled)
    for request_id in scheduler.update_from_output(scheduled, output):
        scheduler.pop_request_state(request_id)
    return scheduled, output


def test_short_requests_finish_and_new_requests_join_between_long_chunks(runtime):
    pipeline, runner, scheduler = runtime
    scheduler.add_request(_request("long", "One. Two. Three. Four."))
    scheduler.add_request(_request("short", "Hi", seed=7))

    _step(runner, scheduler)
    state = runner.state_cache["long"]
    generator = state.extra["generator"]
    chunks = state.extra["prepared"].chunks
    chunk_count = len(chunks.texts)
    assert chunk_count > 2
    _, output = _step(runner, scheduler)
    assert output.get_request_output("short").finished
    torch.testing.assert_close(output.get_request_output("short").result.output, torch.full((1, 1, 2), 7.0))
    assert not output.get_request_output("long").finished
    assert output.get_request_output("long").result is None
    assert state.step_index == 0
    assert "short" not in runner.state_cache
    reference = chunks.ref_audio_tokens
    torch.testing.assert_close(reference, torch.full((8, len(chunks.texts[0])), 3, dtype=torch.long))

    # Admit during a later chunk's second step, not only at a chunk boundary.
    _step(runner, scheduler)
    assert state.step_index == 1
    scheduler.add_request(_request("late", "Bye", seed=9))
    scheduled, output = _step(runner, scheduler)
    assert set(scheduled.scheduled_request_ids) == {"long", "late"}
    assert not output.get_request_output("long").finished
    assert not output.get_request_output("late").finished
    _, output = _step(runner, scheduler)
    assert output.get_request_output("late").finished
    assert not output.get_request_output("long").finished
    assert set(runner.state_cache) == {"long"}

    for _ in range(2 * chunk_count):
        assert state.extra["generator"] is generator
        assert chunks.ref_audio_tokens is reference
        assert chunks.ref_text == chunks.texts[0]
        _, output = _step(runner, scheduler)
        final = output.get_request_output("long")
        if final.finished:
            break
    else:
        pytest.fail("long request did not complete all chunks")
    assert final.result.error is None
    assert final.result.output.shape[-1] > len(chunks.texts[0])
    assert chunks.index == chunk_count
    expected = torch.Generator().manual_seed(3)
    draws = [value for seed, value in pipeline.generator.draws if seed == 3]
    assert len(draws) == chunk_count * 2
    assert draws == [torch.rand((), generator=expected).item() for _ in draws]
    assert not runner.state_cache
    assert runner.input_batch is None

    # Retire the scheduler's terminal bookkeeping before reusing an ID.
    _step(runner, scheduler)
    scheduler.add_request(_request("long", "New", seed=5))
    _, output = _step(runner, scheduler)
    assert output.get_request_output("long").step_index == 1
    assert not output.get_request_output("long").finished
    _, output = _step(runner, scheduler)
    assert output.get_request_output("long").finished
    torch.testing.assert_close(output.get_request_output("long").result.output, torch.full((1, 1, 3), 5.0))


def test_cached_batch_reuses_storage_across_steps_and_chunk_resets(runtime):
    _, runner, scheduler = runtime
    scheduler.add_request(_request("long", "One. Two. End."))
    _step(runner, scheduler)
    batch = runner.input_batch
    state = runner.state_cache["long"]
    _step(runner, scheduler)
    assert state.step_index == 0
    assert runner.input_batch is batch
    assert torch.all(state.extra["tokens"] == runner.pipeline.config.audio_mask_id)
    _step(runner, scheduler)
    storage = batch.latents.data_ptr()
    assert runner.input_batch is batch
    _step(runner, scheduler)
    assert runner.input_batch is batch
    assert batch.latents.data_ptr() == storage
    assert state.step_index == 0


def test_zero_count_steps_skip_sampling_and_still_finish(runtime):
    pipeline, runner, scheduler = runtime
    request = _request("short", "Hi")
    request.sampling_params.num_inference_steps = 32
    scheduler.add_request(request)
    _step(runner, scheduler)
    state = runner.state_cache["short"]
    schedule = state.extra["schedules"]
    assert 0 in schedule
    for step in range(1, len(schedule)):
        draws_before = len(pipeline.generator.draws)
        _, output = _step(runner, scheduler)
        assert len(pipeline.generator.draws) - draws_before == int(schedule[step] > 0)
        assert output.get_request_output("short").finished == (step == len(schedule) - 1)
    assert len(pipeline.generator.draws) == sum(count > 0 for count in schedule)
    assert output.get_request_output("short").result.error is None
    assert not runner.state_cache


def test_abort_releases_completed_chunks_and_preserves_other_request(runtime):
    _, runner, scheduler = runtime
    scheduler.add_request(_request("abort", "One. Two. Three. Four."))
    _step(runner, scheduler)
    _step(runner, scheduler)
    state = runner.state_cache["abort"]
    chunks = state.extra["prepared"].chunks
    references = [weakref.ref(state), weakref.ref(chunks), weakref.ref(chunks.audio[0])]
    del state, chunks
    scheduler.finish_requests("abort", DiffusionRequestStatus.FINISHED_ABORTED)
    scheduler.add_request(_request("survivor", "Hi", seed=7))
    scheduled, output = _step(runner, scheduler)
    assert "abort" in scheduled.finished_req_ids
    assert scheduled.scheduled_request_ids == ["survivor"]
    assert all(reference() is None for reference in references)
    assert not output.get_request_output("survivor").finished
    _, output = _step(runner, scheduler)
    assert output.get_request_output("survivor").finished
    assert not runner.state_cache
    assert runner.input_batch is None
