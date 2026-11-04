# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.ming3_tts.model_state import Ming3ModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

CONTINUE, STOP = 7, 9
LATENT_DIM, HIDDEN = 4, 6


@pytest.mark.parametrize("stage,hybrid", [("llm", False), ("llm", True), ("audio_vae", False)])
def test_model_state_factory_preserves_ming3_stage_dispatch(monkeypatch, stage, hybrid):
    from vllm_omni.config.model import OmniModelConfig
    from vllm_omni.model_executor.models.ming3_tts.ming3_tts import Ming3TTSForConditionalGeneration
    from vllm_omni.worker_v2.model_states import init_omni_model_state
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    class StageConfig(OmniModelConfig):
        is_hybrid = hybrid

    model = Ming3TTSForConditionalGeneration.__new__(Ming3TTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.model_stage = stage
    model.have_multimodal_outputs = True
    monkeypatch.setattr(Ming3ModelState, "__init__", lambda *_args: None)
    monkeypatch.setattr(OmniModelState, "__init__", lambda *_args: None)
    config = SimpleNamespace(model_config=object.__new__(StageConfig))

    state = init_omni_model_state(config, model, None, torch.device("cpu"))

    # The generic hybrid fallback cannot replace Tiny's flow/AudioVAE lifecycle.
    assert type(state) is (Ming3ModelState if stage == "llm" else OmniModelState)


class _FakeFlow:
    """Flow head stand-in: latent and next embedding come from the condition row, stop probability from a table."""

    latent_patch_key = "ming3_latent_patch"
    has_patch_key = "ming3_has_patch"
    fm_dtype = torch.float32

    def __init__(self, stop_probs: dict[float, float]):
        self.ming_config = SimpleNamespace(latent_dim=LATENT_DIM)
        self.stop_probs = stop_probs
        self.calls: list[tuple[float, float, float, int]] = []

    def _sample_flow_rows(self, cond, history, generators, *, cfg_scale, sigma, temperature, check_finite=True):
        assert not check_finite
        self.calls.append((cfg_scale, sigma, temperature, cond.shape[0]))
        latent = cond[:, None, :LATENT_DIM].clone()
        stop = torch.tensor([self.stop_probs.get(float(row[0]), 0.1) for row in cond])
        return latent, cond.clone(), stop, None


def _state(slots: int, stop_probs: dict[float, float], *, max_steps=5, min_steps=2) -> Ming3ModelState:
    state = object.__new__(Ming3ModelState)
    state.llm = _FakeFlow(stop_probs)
    state.history = torch.zeros((slots, 1, LATENT_DIM))
    state.next_embed = torch.zeros((slots, HIDDEN))
    state.active = torch.ones(slots, dtype=torch.bool)
    state.min_steps = torch.full((slots,), min_steps, dtype=torch.long)
    state.max_steps = torch.full((slots,), max_steps, dtype=torch.long)
    state._sde = [(1.0, 0.25, 1.5)] * slots
    state._seeds = [None] * slots
    state._flow_generators = [None] * slots
    state._continue_token, state._stop_token = CONTINUE, STOP
    state._threshold = 0.5
    state._patch_size = 1
    state._partitioned = True
    state._step_vae = None
    state.draining = torch.zeros(slots, dtype=torch.bool)
    return state


def _batch(slots, *, computed, scheduled, prompt_len, seq_lens):
    n = len(slots)
    return SimpleNamespace(
        num_reqs=n,
        req_ids=[f"req-{slot}" for slot in slots],
        idx_mapping_np=np.asarray(slots),
        idx_mapping=torch.tensor(slots, dtype=torch.int32),
        num_computed_tokens_np=np.asarray(computed),
        num_scheduled_tokens=np.asarray(scheduled),
        prefill_len_np=np.asarray(prompt_len),
        logits_indices=torch.arange(n),
        seq_lens=torch.tensor(seq_lens),
    )


def _hidden(values: list[float]) -> torch.Tensor:
    # Row i's first feature identifies it; the rest is arbitrary but finite.
    return torch.tensor([[value] + [0.5] * (HIDDEN - 1) for value in values])


def _run(state, batch, hidden, prompt_len_by_slot):
    req_states = SimpleNamespace(prompt_len=SimpleNamespace(gpu=torch.tensor(prompt_len_by_slot)))
    _, snapshot = state.postprocess_model_output(hidden, batch, req_states)
    return state._tokens, snapshot


def test_stop_head_and_max_steps_decide_tokens_on_device():
    # Slot 0 emits its 3rd latent with a firing stop head; slot 1 its 5th (= max) with a quiet one;
    # slot 2 its 1st (below the minimum) with a firing stop head.
    state = _state(3, {1.0: 0.9, 2.0: 0.1, 3.0: 0.9})
    prompt = [10, 10, 10]
    batch = _batch([0, 1, 2], computed=[11, 13, 9], scheduled=[1, 1, 1], prompt_len=prompt, seq_lens=[12, 14, 10])
    tokens, snapshot = _run(state, batch, _hidden([1.0, 2.0, 3.0]), prompt)

    assert tokens.tolist() == [STOP, STOP, CONTINUE]
    assert state.active.tolist() == [False, False, True]
    # Every live row emits the latent that triggered its decision.
    assert snapshot["ming3_has_patch"].tolist() == [True, True, True]
    torch.testing.assert_close(state.history[:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(state.next_embed[:, 0], torch.tensor([1.0, 2.0, 3.0]))

    # Async scheduling may run stopped rows once more; only the live row advances.
    batch = _batch([0, 1, 2], computed=[12, 14, 10], scheduled=[1, 1, 1], prompt_len=prompt, seq_lens=[13, 15, 11])
    tokens, snapshot = _run(state, batch, _hidden([4.0, 5.0, 6.0]), prompt)
    assert tokens.tolist() == [STOP, STOP, CONTINUE]
    assert snapshot["ming3_has_patch"].tolist() == [False, False, True]
    torch.testing.assert_close(state.history[:, 0, 0], torch.tensor([1.0, 2.0, 6.0]))
    torch.testing.assert_close(state.next_embed[:, 0], torch.tensor([1.0, 2.0, 6.0]))


def test_partial_prefill_and_non_finite_rows_are_not_emitted():
    state = _state(3, {1.0: 0.1, 2.0: 0.1})
    prompt = [10, 40, 10]
    # Row 1 is a non-final prefill chunk; row 2's flow output is not finite.
    batch = _batch([0, 1, 2], computed=[11, 0, 12], scheduled=[1, 16, 1], prompt_len=prompt, seq_lens=[12, 16, 13])
    tokens, snapshot = _run(state, batch, _hidden([1.0, 2.0, float("nan")]), prompt)

    assert snapshot["ming3_has_patch"].tolist() == [True, False, False]
    assert tokens.tolist() == [CONTINUE, STOP, STOP]
    assert state.active.tolist() == [True, True, False]
    assert state.llm.calls == [(1.0, 0.25, 1.5, 2)]
    failed, reason = state.get_request_errors()
    assert failed.tolist() == [False, False, True]
    assert "Non-finite Ming3" in reason
    # Exercise host partitioning on a real output, including non-emitting rows.
    host = snapshot.copy_to_cpu(lambda tensor: tensor.clone())
    assert [row["ming3_has_patch"].tolist() for row in host.inter_stage] == [[True], [False], [False]]
    torch.testing.assert_close(host.inter_stage[0]["ming3_latent_patch"], snapshot["ming3_latent_patch"][:1])


def test_rows_with_different_sampling_controls_are_grouped():
    state = _state(3, {1.0: 0.1, 2.0: 0.1, 3.0: 0.1})
    state._sde = [(1.0, 0.25, 1.5), (2.0, 0.25, 1.5), (1.0, 0.25, 1.5)]
    prompt = [10, 10, 10]
    batch = _batch([0, 1, 2], computed=[11, 11, 11], scheduled=[1, 1, 1], prompt_len=prompt, seq_lens=[12, 12, 12])
    tokens, snapshot = _run(state, batch, _hidden([1.0, 2.0, 3.0]), prompt)

    assert sorted(state.llm.calls) == [(1.0, 0.25, 1.5, 2), (2.0, 0.25, 1.5, 1)]
    assert tokens.tolist() == [CONTINUE] * 3
    torch.testing.assert_close(snapshot["ming3_latent_patch"][:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_seeded_flow_noise_uses_a_fresh_generator_per_request():
    # As on the V1 runner, flow-head noise starts from the request seed
    # regardless of the draws its prompt-latent posterior made at prefill.
    state = _state(2, {})
    state._seeds = [43, None]
    first = state._generator(0, torch.device("cpu"))
    assert state._generator(0, torch.device("cpu")) is first
    assert first.initial_seed() == 43
    assert torch.equal(torch.randn(3, generator=first), torch.randn(3, generator=torch.Generator().manual_seed(43)))
    assert state._generator(1, torch.device("cpu")) is None


class _FakeStepVAE:
    """Records the single-stage decode calls; every decoded row yields 3 samples."""

    scale = 4
    hop = 2
    stream = None

    def __init__(self):
        self.primed, self.steps = [], []

    def prime(self, slots, latents):
        self.primed.append(slots.tolist())

    def step(self, slots, positions, first, latents, last, valid):
        self.steps.append((slots.tolist(), positions.tolist(), first.tolist(), last.tolist(), valid.tolist()))
        rows = slots.shape[0]
        audio = torch.arange(rows * 5, dtype=torch.float32).view(rows, 5)
        start = torch.zeros(rows, dtype=torch.long)
        return audio, start, torch.where(valid, 3, 0)


def test_single_stage_rows_drain_one_step_before_stopping():
    from vllm_omni.model_executor.models.ming3_tts.model_state import _StepAudioOutput

    state = _state(2, {1.0: 0.9, 2.0: 0.1})
    state._step_vae = vae = _FakeStepVAE()
    state._audio_chunks = {}
    state._chunk_latents = (4, 25)
    state._samples_per_latent = 8
    state._vae_positions = torch.zeros(2, dtype=torch.long)
    state._sample_rate = torch.tensor(44100)
    prompt = [10, 10]
    # Slot 0 decodes its 3rd latent with a firing stop head; slot 1 finishes its prompt (first latent).
    batch = _batch([0, 1], computed=[11, 8], scheduled=[1, 2], prompt_len=prompt, seq_lens=[12, 10])
    tokens, _ = _run(state, batch, _hidden([1.0, 2.0]), prompt)

    # Halting continues for one flush step; the first latent is held, not decoded.
    assert tokens.tolist() == [CONTINUE, CONTINUE]
    assert state.draining.tolist() == [True, False]
    assert vae.primed == [[1]]
    assert vae.steps[-1] == ([0], [0], [True], [False], [True])
    output = state.prepare_streaming_audio_output(batch, None, {})
    assert isinstance(output, _StepAudioOutput)
    assert [None if item is None else item["model_outputs"].tolist() for item in output.get_output()] == [
        [0.0, 1.0, 2.0],
        None,
    ]

    # Next step: slot 0 flushes and stops; slot 1 decodes its first output.
    batch = _batch([0, 1], computed=[12, 10], scheduled=[1, 1], prompt_len=prompt, seq_lens=[13, 11])
    tokens, _ = _run(state, batch, _hidden([1.0, 2.0]), prompt)
    assert tokens.tolist() == [STOP, CONTINUE]
    assert state.draining.tolist() == [False, False]
    assert vae.steps[-1] == ([0, 1], [4, 0], [False, True], [True, False], [True, True])
    # Both rows release their audio: slot 0 its final chunk, slot 1 its first.
    output = state.prepare_streaming_audio_output(batch, None, {}).get_output()
    assert [item["model_outputs"].tolist() for item in output] == [[0.0, 1.0, 2.0], [5.0, 6.0, 7.0]]


def test_single_stage_audio_is_released_in_growing_chunks():
    from vllm_omni.model_executor.models.ming3_tts.model_state import _AudioChunks

    chunks = _AudioChunks(samples_per_latent=2, chunk_latents=(2, 3))
    pushes = [(torch.ones(1), False)] + [(torch.ones(2), False)] * 6 + [(torch.ones(1), True)]
    released = [None if (out := chunks.push(pcm, ended)) is None else out.numel() for pcm, ended in pushes]
    # First audio at once, then 2 and 3 latents (4 and 6 samples), then the remainder at the end.
    assert released == [1, None, 4, None, None, 6, None, 3]


def test_single_stage_vae_positions_follow_emission_and_flush():
    state = _state(3, {})
    state._step_vae = vae = _FakeStepVAE()
    state._audio_chunks = {}
    state._chunk_latents = (4, 25)
    state._samples_per_latent = 8
    state._vae_positions = torch.tensor([0, 12, 20])
    state._sample_rate = torch.tensor(44100)
    # Reordered slots: live, paused, and flushing. A paused row can resume
    # without leaving a hole in its KV cache; flush advances exactly once.
    slots = np.array([2, 0, 1])
    batch = _batch(slots, computed=[10] * 3, scheduled=[1] * 3, prompt_len=[10] * 3, seq_lens=[11] * 3)
    latent = torch.ones(3, 1, LATENT_DIM)
    state._decode_step_audio(
        batch, np.arange(3), slots, latent, torch.tensor([True, False, False]), torch.tensor([False, False, True])
    )
    assert state._vae_positions.tolist() == [0, 16, 24]
    assert vae.steps[-1] == ([2, 0, 1], [20, 0, 12], [False, True, False], [False, False, True], [True, False, True])

    state._decode_step_audio(
        batch, np.arange(3), slots, latent, torch.tensor([True, True, False]), torch.zeros(3, dtype=torch.bool)
    )
    assert state._vae_positions.tolist() == [4, 16, 28]
    assert vae.steps[-1][1:3] == ([24, 0, 16], [False, True, False])

    # Slot reuse starts with a held first latent and resets the device cursor.
    batch = _batch([1], computed=[9], scheduled=[1], prompt_len=[10], seq_lens=[10])
    state._decode_step_audio(
        batch, np.array([0]), np.array([1]), latent[:1], torch.tensor([True]), torch.tensor([False])
    )
    assert state._vae_positions.tolist() == [4, 0, 28]
    assert vae.primed == [[1]]


def test_zero_initial_chunk_uses_regular_size_without_initializing_two_stage_pcm(monkeypatch):
    from vllm_omni.model_executor.models.common.ming.audio_vae import AudioVAEConfig
    from vllm_omni.model_executor.models.ming3_tts.config_ming3_tts import (
        BailingMoeV3Config,
        Ming3TTSRuntimeConfig,
    )
    from vllm_omni.model_executor.stage_input_processors.ming3_tts import _chunk_sizes, _chunk_target
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    def base_init(self, config, model, encoder_cache, device):
        self.scheduler_config = SimpleNamespace(max_num_seqs=1)
        self.dtype = torch.float32

    monkeypatch.setattr(OmniModelState, "__init__", base_init)
    config = SimpleNamespace(
        model_config=SimpleNamespace(async_chunk=True, stage_connector_config=None),
        additional_config={"initial_latent_chunk_size": 0},
    )
    model = SimpleNamespace(
        model=_FakeFlow({}),
        ming_config=Ming3TTSRuntimeConfig(
            llm_hidden_size=HIDDEN,
            llm_vocab_size=10,
            llm_config=BailingMoeV3Config(),
            audio_tokenizer_config=AudioVAEConfig(),
        ),
        step_vae=_FakeStepVAE(),
    )
    state = Ming3ModelState(config, model, None, torch.device("cpu"))
    assert state._chunk_latents == (25,) and state._samples_per_latent == 8
    manager = SimpleNamespace(connector=SimpleNamespace(config={"extra": config.additional_config}))
    chunk, initial, ramp = _chunk_sizes(manager)
    assert _chunk_target(0, chunk, initial, ramp) == state._chunk_latents[0]

    # Both consumers reject the same invalid bounds; connector options still win.
    for steady, initial in ((0, 0), (25, -1), (25, 26)):
        extra = {"latent_chunk_size": steady, "initial_latent_chunk_size": initial}
        config.model_config.stage_connector_config = manager.connector.config = {"extra": extra}
        with pytest.raises(ValueError, match="Invalid Ming3 chunk sizes"):
            Ming3ModelState(config, model, None, torch.device("cpu"))
        with pytest.raises(ValueError, match="Invalid Ming3 chunk sizes"):
            _chunk_sizes(manager)
    config.model_config.stage_connector_config = {"extra": {"initial_latent_chunk_size": 4}}
    assert Ming3ModelState(config, model, None, torch.device("cpu"))._chunk_latents == (4, 8, 16, 25)
    config.model_config.stage_connector_config = {"extra": {"initial_latent_chunk_size": 0}}
    model.step_vae = None
    state = Ming3ModelState(config, model, None, torch.device("cpu"))
    assert not hasattr(state, "_chunk_latents") and not hasattr(state, "_samples_per_latent")


@pytest.mark.parametrize("cfg,temperature", [(0.0, 0.0), (2.0, 1.5), (4.0, 0.5)])
def test_api_controls_reach_mrv2_admission_and_final_flush(monkeypatch, cfg, temperature):
    import asyncio
    from types import MethodType
    from unittest.mock import AsyncMock

    from vllm import SamplingParams

    from tests.model_executor.models.ming3_tts.test_ming3_contract import FakeTokenizer
    from vllm_omni.core.sched.output import OmniNewRequestData
    from vllm_omni.engine.serialization import serialize_additional_information
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
    from vllm_omni.entrypoints.openai.tts_adapters import ming3_tts as adapter_module
    from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
    from vllm_omni.entrypoints.openai.tts_adapters.ming3_tts import Ming3TTSAdapter
    from vllm_omni.model_executor.models.ming3_tts.ming3_tts import Ming3TTSForConditionalGeneration
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
    from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

    monkeypatch.setattr(adapter_module, "_extract_speaker_embeddings", lambda *args: torch.zeros(2, 192))
    server = SimpleNamespace(
        engine_client=SimpleNamespace(
            model_config=SimpleNamespace(
                model="fake", hf_config=SimpleNamespace(tts_config={"task_decode_alignment": {"podcast": "old_shift"}})
            )
        ),
        _tts_tokenizer=FakeTokenizer(),
        _apply_uploaded_speaker=lambda request: None,
        _resolve_ref_audio_array=AsyncMock(return_value=(np.zeros(16000), 16000, "reference")),
        _get_resolved_ref_audio_artifact_key=lambda key: key,
    )
    adapter = Ming3TTSAdapter(SpeechServingContext(server=server))
    request = OpenAICreateSpeechRequest(
        input="speaker_1: Hello, how are you doing on this lovely day?\nspeaker_2: I am doing very well, thank you.",
        ref_audio=["one.wav", "two.wav"],
        ref_text="speaker_1: First reference.\nspeaker_2: Second reference.",
        max_new_tokens=100,
        seed=43,
        extra_params={"ming3_cfg": cfg, "temperature": temperature},
    )
    prepared = asyncio.run(adapter.build(request, [], False))
    sampling = adapter.apply_sampling_overrides([SamplingParams()], request, prepared.prompt, "api-id")[0]
    minimum = prepared.prompt["additional_information"]["ming3_min_decode_steps"]
    assert minimum > 13
    state = _state(1, {1.0: 0.9})
    state.device = torch.device("cpu")
    state.llm.ming_config = SimpleNamespace(
        latent_dim=LATENT_DIM,
        cfg=1.0,
        sigma=0.25,
        temperature=1.5,
        max_decode_steps=800,
        stop_head_min_steps=11,
    )
    state.model = SimpleNamespace()
    state.model._copy_request_state = MethodType(Ming3TTSForConditionalGeneration._copy_request_state, state.model)
    state.intermediate_buffer = OmniIntermediateBuffer(1)
    state._pending = {}
    monkeypatch.setattr(
        OmniModelState, "add_request", lambda self, index, data: self.intermediate_buffer.add_request(index, data)
    )
    ids = prepared.prompt["prompt_token_ids"]
    data = OmniNewRequestData(
        req_id="runner-id",
        prompt_token_ids=ids,
        prefill_token_ids=ids,
        mm_features=[],
        sampling_params=sampling,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        additional_information=serialize_additional_information(prepared.prompt["additional_information"]),
    )
    state.add_request(0, data)
    assert state._sde[0] == (cfg, 0.25, temperature)
    assert state._pending == {0: (minimum, 100)}
    assert state._seeds == [43] and sampling.max_tokens == 101
    state._flush_pending()
    state._step_vae = _FakeStepVAE()
    state._audio_chunks = {}
    state._chunk_latents = (4, 25)
    state._samples_per_latent = 8
    state._vae_positions = torch.zeros(1, dtype=torch.long)
    state._sample_rate = torch.tensor(44100)
    prompt = [len(ids)]
    # The text-derived minimum suppresses an early stop; reaching the cap
    # still reserves the API's extra token for a final PCM flush.
    for step, score, token, draining in (
        (minimum - 1, 0.9, CONTINUE, False),
        (100, 0.1, CONTINUE, True),
        (101, 0.1, STOP, False),
    ):
        state.llm.stop_probs[1.0] = score
        batch = _batch(
            [0], computed=[len(ids) + step - 2], scheduled=[1], prompt_len=prompt, seq_lens=[len(ids) + step - 1]
        )
        tokens, _ = _run(state, batch, _hidden([1.0]), prompt)
        assert tokens.tolist() == [token]
        assert state.draining.tolist() == [draining]
        output = state.prepare_streaming_audio_output(batch, None, {}).get_output()
    assert state.llm.calls == [(cfg, 0.25, temperature, 1)] * 3
    assert state._step_vae.steps[-1][3:] == ([True], [True])
    assert output[0]["model_outputs"].numel() > 0


def test_async_model_error_discards_only_failed_rows(monkeypatch):
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    from tests.worker_v2.test_omni_ar_model_runner import _async_output

    monkeypatch.setattr(torch.cuda, "set_stream", lambda _stream: None)
    batch = SimpleNamespace(query_start_loc_np=np.array([0, 1, 2]), num_scheduled_tokens=[1, 1], num_reqs=2)
    flags = torch.tensor([True, False])
    sampler = SamplerOutput(torch.tensor([[7], [8]]), None, None, torch.tensor([1, 1]), torch.tensor([0, 0]))
    pending = _async_output(
        req_ids=("bad", "good"),
        sampler_output=sampler,
        input_batch=batch,
        async_chunk=True,
        multimodal_outputs={"codes.audio": torch.tensor([[1], [2]])},
        request_errors=(flags, "Non-finite flow output"),
    )
    # The next step may replace device state; the deferred output owns its copy.
    flags.zero_()
    output = pending.get_output()
    assert output.request_errors == {"bad": "Non-finite flow output"}
    assert output.sampled_token_ids == [[], [8]]
    assert output.inter_stage_outputs[0] is None
    assert output.inter_stage_outputs[1]["codes.audio"].item() == 2
