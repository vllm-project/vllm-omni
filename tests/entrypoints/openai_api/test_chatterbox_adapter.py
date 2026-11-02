# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox's public speech request and conditioning contracts.

The adapters run inside a real ``OmniOpenAIServingSpeech``, built from the
deploy file's stage configs, and condition on references with the real
reference encoders, loaded from a model directory in the released files'
layout. Only the engine client is not the engine: ``AsyncOmni`` runs its
stages in other processes, so the client here carries the real configs it
exposes and nothing else.
"""

import asyncio
import base64
import io
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from vllm import SamplingParams
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels

from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from tests.model_executor.models.chatterbox.helpers import (
    DEPLOY,
    stage_vllm_config,
    write_model_dir,
    write_s3gen_checkpoint,
    write_voice_encoder,
)
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.config.speech_cache import SpeechCacheConfig
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy, resolve_deploy_yaml
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type
from vllm_omni.entrypoints.openai.tts_adapters.chatterbox import ChatterboxAdapter, ChatterboxOriginalAdapter
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioner, VoiceConditioning
from vllm_omni.model_executor.models.chatterbox.pipeline import CHATTERBOX_TURBO_PIPELINE
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

BUILTIN = {
    "t3": {"cond_prompt_speech_tokens": torch.tensor([[1, 2, 3]]), "speaker_emb": torch.ones(1, 256)},
    "gen": {
        "prompt_token": torch.tensor([[4, 5]]),
        "prompt_feat": torch.ones(1, 4, 80),
        "embedding": torch.ones(1, 192),
    },
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A Chatterbox model directory with every file the serving adapters read.

    The built-in voice (``conds.pt``), Turbo's GPT-2 BPE text tokenizer files
    with a vocabulary of a few pieces, and the reference encoders' weights in
    both checkpoints' S3Gen files, from the module definitions with seeded
    initial weights.
    """
    directory = write_model_dir(tmp_path_factory.mktemp("chatterbox"))
    torch.save(BUILTIN, Path(directory) / "conds.pt")
    vocab = {"H": 0, "e": 1, "l": 2, "o": 3, "He": 4, "ll": 5, "llo": 6, "Hello": 42, ".": 43, "<|endoftext|>": 44}
    (Path(directory) / "vocab.json").write_text(json.dumps(vocab))
    (Path(directory) / "merges.txt").write_text("#version: 0.2\nH e\nl l\nll o\nHe llo\n")
    write_voice_encoder(directory, seed=0)
    for variant in ("turbo", "original"):
        write_s3gen_checkpoint(directory, ChatterboxConfig(variant), seed=1, encoders_only=True)
    return directory


def engine_client(model_dir: str, deploy: str, legacy: bool = False) -> SimpleNamespace:
    """What the serving layer reads from ``AsyncOmni``, with the deploy file's real stage configs.

    ``legacy`` gives the stage configs in their legacy form, whose fields sit under ``engine_args``.
    """
    vllm_config = stage_vllm_config(model_dir, deploy, 0)
    path = deploy if Path(deploy).is_absolute() else get_deploy_config_path(deploy)
    stage_configs = (
        [stage.to_omegaconf() for stage in merge_pipeline_deploy(CHATTERBOX_TURBO_PIPELINE, load_deploy_config(path))]
        if legacy
        else list(
            resolve_omni_config(
                model_dir,
                trust_remote_code=None,
                deploy_config_path=path,
                cli_overrides={},
                stage_overrides=None,
                strategy_config_path=None,
            ).stage_configs
        )
    )
    return SimpleNamespace(
        model=model_dir,
        model_config=vllm_config.model_config,
        vllm_config=vllm_config,
        renderer=None,
        input_processor=None,
        stage_configs=stage_configs,
        errored=False,
        default_sampling_params_list=[SamplingParams(max_tokens=1000)],
        tts_batch_max_items=32,
    )


def serving(client: SimpleNamespace, speech_cache: SpeechCacheConfig | None = None) -> OmniOpenAIServingSpeech:
    """The speech server, with the deploy file's ``speech_cache`` budgets when given."""
    models = OpenAIServingModels(client, [BaseModelPath(name=client.model, model_path=client.model)])
    return OmniOpenAIServingSpeech(
        engine_client=client, models=models, request_logger=None, speech_cache_config=speech_cache
    )


@pytest.fixture
def server(
    model_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_speaker_cache: None
) -> Iterator[OmniOpenAIServingSpeech]:
    """The speech server for Turbo's deploy file; uploaded voices are kept under the test's directory.

    Every server shares the process-wide speaker cache, which the repository's
    ``clean_speaker_cache`` fixture resets around the test.
    """
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    speech = serving(engine_client(model_dir, deploy_on_cpu(DEPLOY["turbo"])))
    yield speech
    speech.shutdown()


@pytest.fixture
def adapter(server: OmniOpenAIServingSpeech) -> ChatterboxAdapter:
    return server._adapter


@pytest.fixture(scope="module")
def validator(model_dir: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[ChatterboxAdapter]:
    """One Turbo adapter shared by the tests that only check requests, which change none of its state.

    Building a server loads the reference encoders; the CPU job has no time for one per case.
    It leaves the process-wide speaker cache alone, so the tests that count its entries do
    not depend on it: building only fetches the cache (``clean_speaker_cache`` gives those
    tests a new one), and ``shutdown`` clears only voices this server uploaded, which is none.
    """
    with pytest.MonkeyPatch.context() as patch:
        # The server reads the uploaded voices' directory once, when it is built.
        patch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path_factory.mktemp("speakers")))
        speech = serving(engine_client(model_dir, deploy_on_cpu(DEPLOY["turbo"])))
    yield speech._adapter
    speech.shutdown()


@pytest.fixture(scope="module")
def original_model_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Original's own model directory: ``tokenizer.json`` takes the place of Turbo's GPT-2 files.

    The two cannot share a directory, since Turbo's AutoTokenizer would read a
    ``tokenizer.json`` in preference to the GPT-2 files.
    """
    directory = write_model_dir(tmp_path_factory.mktemp("chatterbox-original"))
    torch.save(BUILTIN, Path(directory) / "conds.pt")
    Tokenizer(WordLevel({"[UNK]": 0, "Hello.": 7}, unk_token="[UNK]")).save(str(Path(directory) / "tokenizer.json"))
    write_voice_encoder(directory, seed=0)
    write_s3gen_checkpoint(directory, ChatterboxConfig("original"), seed=1, encoders_only=True)
    return directory


@pytest.fixture
def original(
    original_model_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_speaker_cache: None
) -> Iterator[ChatterboxOriginalAdapter]:
    """Original's adapter in the speech server for its deploy file."""
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    speech = serving(engine_client(original_model_dir, deploy_on_cpu(DEPLOY["original"])))
    yield speech._adapter
    speech.shutdown()


def reference(seconds: float, seed: int, sample_rate: int = 24000, mime: str = "audio/wav") -> str:
    """A reference recording as a client sends it: a base64 WAV data URL."""
    samples = (0.1 * np.random.default_rng(seed).standard_normal(round(seconds * sample_rate))).astype(np.float32)
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, sample_rate, format="WAV")
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode()}"


# References the batching tests admit at once: a whole admission of the shipped
# profiles, 64, is too much CPU encoding for a shared runner.
ADMISSION = 8


def deploy_on_cpu(name: str, max_num_seqs: int | None = None) -> str:
    """A copy of deploy file ``name`` with its reference encoders on this host's CPU.

    Every CUDA profile conditions on ``cuda:0``; these tests build servers on a
    CPU host. ``max_num_seqs`` replaces stage 0's, which sizes the encoder batch.
    """
    stage: dict[str, object] = {"additional_config.conditioning_device": "cpu"}
    if max_num_seqs is not None:
        stage["max_num_seqs"] = max_num_seqs
    return modify_stage_config(get_deploy_config_path(name), updates={"stages": {0: stage}})


def encoder(adapter: ChatterboxAdapter) -> VoiceConditioner:
    """The adapter's reference encoders."""
    return adapter.conditioner


class EncoderBatches:
    """Each encoder batch's size, and how many ran at once, from the S3 tokenizer's encoder.

    Every batch quantizes its conditioning tokens and then its prompt
    tokens, one encoder call each. ``hold`` keeps a batch from finishing
    until ``release``.
    """

    def __init__(self, conditioner: VoiceConditioner, hold: bool = False) -> None:
        self.sizes: list[int] = []
        self.active = 0
        self.most_active = 0
        self.lock = threading.Lock()
        self.started = threading.Event()
        self.released = threading.Event()
        if not hold:
            self.released.set()
        module = conditioner.tokenizer.encoder
        self.hooks = [module.register_forward_pre_hook(self.enter), module.register_forward_hook(self.leave)]

    def enter(self, module: nn.Module, inputs: tuple) -> None:
        with self.lock:
            self.sizes.append(inputs[0].shape[0])
            self.active += 1
            self.most_active = max(self.most_active, self.active)
        self.started.set()
        self.released.wait(timeout=30)

    def leave(self, module: nn.Module, inputs: tuple, output: object) -> None:
        with self.lock:
            self.active -= 1

    @property
    def batches(self) -> list[int]:
        return self.sizes[::2]

    def remove(self) -> None:
        self.released.set()
        for hook in self.hooks:
            hook.remove()


async def wait_for_admission(adapter: ChatterboxAdapter, requests: list[asyncio.Task]) -> None:
    """Wait at most 30 s until every request, each its own reference, holds an admission slot.

    The caller holds the encoder batch, so no request can end first: one that
    does raises its own error here instead of leaving the wait to time out.
    """

    async def admitted() -> None:
        while len(adapter.conditioning_pending) < len(requests):
            if ended := [request for request in requests if request.done()]:
                # A failed request raises its error; a cancelled one falls through to the assertion.
                if failed := [request for request in ended if not request.cancelled()]:
                    failed[0].result()
                raise AssertionError(f"{len(ended)} of {len(requests)} requests ended before all were admitted")
            await asyncio.sleep(0.01)

    await asyncio.wait_for(admitted(), 30)


def test_entry_stage_selects_chatterbox():
    assert detect_tts_model_type("chatterbox_t3", None) == "chatterbox"


@pytest.mark.parametrize(
    "fields,error",
    [
        ({"input": "  "}, "empty"),
        ({"language": "French"}, "English"),
        ({"instructions": "Whisper"}, "instructions"),
        ({"task_type": "VoiceDesign"}, "VoiceDesign"),
        ({"x_vector_only_mode": True}, "full reference"),
        ({"extra_params": {"cfg_weight": 0.5}}, "Original"),
        ({"voice": "alloy"}, "default"),
        ({"ref_audio": ["data:a", "data:b"]}, "one reference"),
        ({"ref_audio_2": "data:a"}, "one reference"),
        ({"ref_audio": "clip.wav"}, "ref_audio must be a URL"),
    ],
)
def test_unsupported_request_is_rejected(validator, fields, error):
    request = OpenAICreateSpeechRequest(**{"input": "Hello.", **fields})
    assert error in validator.validate(request)


@pytest.mark.parametrize(
    "fields", [{}, {"voice": "default"}, {"language": "en"}, {"ref_audio": "data:audio/wav;base64,UklGRg=="}]
)
def test_builtin_and_reference_requests_are_accepted(validator, fields):
    assert validator.validate(OpenAICreateSpeechRequest(input="Hello.", **fields)) is None


def test_builtin_voice_loads_checkpoint_tensors(model_dir: str):
    voice = VoiceConditioning.from_builtin(model_dir)
    assert torch.equal(voice.cond_tokens, BUILTIN["t3"]["cond_prompt_speech_tokens"])
    assert torch.equal(voice.speaker_emb, BUILTIN["t3"]["speaker_emb"])
    assert torch.equal(voice.prompt_token, BUILTIN["gen"]["prompt_token"])
    assert torch.equal(voice.prompt_feat, BUILTIN["gen"]["prompt_feat"])
    assert torch.equal(voice.embedding, BUILTIN["gen"]["embedding"])


@pytest.mark.asyncio
async def test_builtin_request_uses_shared_prompt_builder_without_reference_encoders(adapter):
    """The model directory's built-in voice and tokenizer, loaded by the adapter itself."""
    request = OpenAICreateSpeechRequest(input="hello")
    prepared = await adapter.build(request, [SamplingParams()], False)
    assert prepared.prompt["prompt_token_ids"] == [6561] * 7
    assert prepared.prompt["additional_information"]["ids"] == {"prompt": [42, 43], "speech_token": [1, 2, 3]}
    assert adapter.conditioning_queue.empty() and not adapter.conditioning_pending
    assert prepared.model_type == "chatterbox"


def test_request_token_cap_does_not_mutate_deploy_defaults(validator):
    defaults = [SamplingParams(max_tokens=1000), SamplingParams(max_tokens=1)]
    result = validator.apply_sampling_overrides(defaults, OpenAICreateSpeechRequest(input="Hi.", max_new_tokens=50))
    assert [params.max_tokens for params in result] == [50, 1]
    assert defaults[0].max_tokens == 1000


def test_original_owns_its_stage_and_controls(original):
    assert detect_tts_model_type("chatterbox_original_t3", None) == "chatterbox_original"
    assert isinstance(original, ChatterboxOriginalAdapter) and original.config.variant == "original"
    request = OpenAICreateSpeechRequest(input="Hello.", extra_params={"cfg_weight": 0, "exaggeration": 0.7})
    assert original.validate(request) is None
    assert original.validate(OpenAICreateSpeechRequest(input="Hello.", language="French")) == (
        "Chatterbox Original supports English only"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_params", "controls"),
    [(None, {"exaggeration": 0.5, "cfg_weight": 0.5}), ({"cfg_weight": 0}, {"exaggeration": 0.5, "cfg_weight": 0})],
    ids=["deploy defaults", "request override"],
)
async def test_original_controls_default_to_the_deploy_files(original, extra_params, controls):
    """A request's missing controls come from stage 0's ``additional_config``, upstream's defaults there."""
    prepared = await original.build(OpenAICreateSpeechRequest(input="hello", extra_params=extra_params), [], False)
    assert prepared.prompt["additional_information"]["chatterbox"] == controls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("variant", "adapter_cls", "seconds", "max_num_seqs"),
    [("turbo", ChatterboxAdapter, 5.04, ADMISSION), ("original", ChatterboxOriginalAdapter, 1.0, 2 * ADMISSION)],
)
async def test_a_whole_admission_of_references_is_encoded_together_and_cached(
    request, tmp_path, monkeypatch, clean_speaker_cache, variant, adapter_cls, seconds, max_num_seqs
):
    """Everything queued while the encoders are busy goes through them as one batch, with no cap of its own.

    Stage 0 admits ``ADMISSION`` requests (Original's twice as many sequences
    are that many guided pairs), so that many references arriving together
    are one admission. The first batch is held so that the other references queue
    behind it; the batch after it is all of them. Each clip is 40 ms (one
    prompt token) longer than the one before, so the prompt lengths show
    every voice is its own reference's. A second wave of the same references
    is served from the cache.
    """
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    model_dir = request.getfixturevalue("original_model_dir" if variant == "original" else "model_dir")
    speech = serving(engine_client(model_dir, deploy_on_cpu(DEPLOY[variant], max_num_seqs)))
    concurrent = speech._adapter
    assert isinstance(concurrent, adapter_cls)
    assert concurrent.conditioning_capacity == ADMISSION
    batches = EncoderBatches(encoder(concurrent), hold=True)
    sources = [reference(seconds + 0.04 * index, seed=index) for index in range(ADMISSION)]
    try:
        requests = [asyncio.create_task(concurrent.resolve_voice(source)) for source in sources]
        assert await asyncio.to_thread(batches.started.wait, 30)
        await wait_for_admission(concurrent, requests)
        batches.released.set()
        voices = await asyncio.wait_for(asyncio.gather(*requests), 120)
        encoded = list(batches.batches)
        cached = await asyncio.gather(*(concurrent.resolve_voice(source) for source in sources))
    finally:
        batches.remove()
        speech.shutdown()

    lengths = [voice.prompt_token.shape[1] for voice in voices]
    assert lengths == list(range(lengths[0], lengths[0] + ADMISSION))
    assert encoded == [encoded[0], ADMISSION - encoded[0]]
    assert batches.most_active == 1
    # Served from the speaker cache: the same tensors, and no further encoder batch.
    assert all(left.prompt_feat is right.prompt_feat for left, right in zip(voices, cached, strict=True))
    assert batches.batches == encoded
    assert not concurrent.conditioning_pending


@pytest.mark.asyncio
async def test_a_prompt_build_does_not_wait_behind_an_encoder_batch(adapter):
    """A built-in-voice request is prepared while a reference is still in the encoders."""
    batches = EncoderBatches(encoder(adapter), hold=True)
    try:
        reference_request = asyncio.create_task(adapter.resolve_voice(reference(6, seed=7)))
        assert await asyncio.to_thread(batches.started.wait, 30)
        prepared = await asyncio.wait_for(adapter.build(OpenAICreateSpeechRequest(input="hello"), [], False), 30)
        assert not reference_request.done()
        batches.released.set()
        await reference_request
    finally:
        batches.remove()

    assert prepared.prompt["additional_information"]["ids"]["prompt"] == [42, 43]


@pytest.mark.asyncio
async def test_reference_singleflight_survives_waiter_cancellation(adapter):
    """Two locators of one waveform share one encoding, which outlives the waiter that started it."""
    batches = EncoderBatches(encoder(adapter), hold=True)
    try:
        first = asyncio.create_task(adapter.resolve_voice(reference(6, seed=1)))
        assert await asyncio.to_thread(batches.started.wait, 30)
        second = asyncio.create_task(adapter.resolve_voice(reference(6, seed=1, mime="audio/x-wav")))
        await asyncio.sleep(0.1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        batches.released.set()
        voice = await second
    finally:
        batches.remove()

    assert isinstance(voice, VoiceConditioning)
    assert batches.batches == [1]
    assert adapter.ctx.server._speaker_cache.stats()["entries"] == 1


@pytest.mark.asyncio
async def test_short_reference_is_repeated_and_encoded_with_its_batch(adapter):
    """A two-second clip, which upstream Turbo refuses, is repeated past five seconds and encoded with the rest.

    A held first batch lets the short and the long clip queue together, so
    the second batch holds both.
    """
    batches = EncoderBatches(encoder(adapter), hold=True)
    try:
        blocker = asyncio.create_task(adapter.resolve_voice(reference(6, seed=9)))
        assert await asyncio.to_thread(batches.started.wait, 30)
        pair = [
            asyncio.create_task(adapter.resolve_voice(reference(2, seed=2))),
            asyncio.create_task(adapter.resolve_voice(reference(6, seed=3))),
        ]
        await asyncio.sleep(0.1)
        batches.released.set()
        await blocker
        short, long = await asyncio.wait_for(asyncio.gather(*pair), 60)
    finally:
        batches.remove()

    assert isinstance(short, VoiceConditioning) and isinstance(long, VoiceConditioning)
    # Three repeats of two seconds give six seconds: 150 decoder prompt tokens at 25 per second.
    assert short.prompt_token.shape == (1, 150) == long.prompt_token.shape
    assert batches.batches == [1, 2]
    assert adapter.ctx.server._speaker_cache.stats()["entries"] == 3
    assert not adapter.conditioning_pending


@pytest.mark.asyncio
async def test_a_zero_speaker_cache_budget_encodes_every_request(
    model_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_speaker_cache: None
) -> None:
    """The deploy file's ``speech_cache.speaker_max_bytes`` bounds the conditioning cache; zero disables it."""
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    speech = serving(engine_client(model_dir, deploy_on_cpu(DEPLOY["turbo"])), SpeechCacheConfig(speaker_max_bytes=0))
    batches = EncoderBatches(encoder(speech._adapter))
    source = reference(6, seed=6)
    try:
        for _ in range(2):
            await speech._adapter.resolve_voice(source)
    finally:
        batches.remove()
        speech.shutdown()

    assert batches.batches == [1, 1]


@pytest.mark.asyncio
async def test_cancelled_wave_cannot_overfill_encoder_queue(
    original_model_dir, tmp_path, monkeypatch, clean_speaker_cache
):
    """Cancelled waiters keep their encoder work, and with it the slots a new wave would take."""
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    speech = serving(engine_client(original_model_dir, deploy_on_cpu(DEPLOY["original"], max_num_seqs=2 * ADMISSION)))
    original = speech._adapter
    assert isinstance(original, ChatterboxOriginalAdapter)
    admission = original.conditioning_capacity
    batches = EncoderBatches(encoder(original), hold=True)
    try:
        first = [asyncio.create_task(original.resolve_voice(reference(1, seed=index))) for index in range(admission)]
        assert await asyncio.to_thread(batches.started.wait, 30)
        # Every first-wave request holds a slot before the wave is cancelled.
        await wait_for_admission(original, first)
        for request in first:
            request.cancel()
        await asyncio.gather(*first, return_exceptions=True)
        second = [
            asyncio.create_task(original.resolve_voice(reference(1, seed=index)))
            for index in range(admission, 2 * admission)
        ]
        await asyncio.sleep(0.1)
        assert len(original.conditioning_pending) == admission
        assert original.conditioning_queue.qsize() <= admission
        assert not any(request.done() for request in second)
        batches.released.set()
        voices = await asyncio.wait_for(asyncio.gather(*second), 60)
    finally:
        batches.remove()
        speech.shutdown()

    assert all(isinstance(voice, VoiceConditioning) for voice in voices)
    assert not original.conditioning_pending


@pytest.mark.asyncio
async def test_shutdown_cancels_encoder_waiters(adapter):
    batches = EncoderBatches(encoder(adapter), hold=True)
    try:
        request = asyncio.create_task(adapter.resolve_voice(reference(6, seed=4)))
        assert await asyncio.to_thread(batches.started.wait, 30)
        adapter.shutdown()
        adapter.shutdown()
        with pytest.raises(asyncio.CancelledError):
            await request
        with pytest.raises(RuntimeError, match="shut down"):
            await adapter.resolve_voice(reference(6, seed=5))
    finally:
        batches.remove()
    assert not adapter.conditioning_pending
    assert adapter.conditioning_queue.empty()


@pytest.mark.parametrize("deploy", list(DEPLOY.values()), ids=list(DEPLOY))
def test_shipped_profiles_condition_on_the_stages_gpu(deploy: str) -> None:
    """Both deploy files put the reference encoders on the stages' GPU."""
    stage = resolve_deploy_yaml(get_deploy_config_path(deploy))["stages"][0]
    assert stage["additional_config"]["conditioning_device"] == "cuda:0"


@pytest.mark.parametrize("variant", list(DEPLOY))
def test_shipped_profiles_admit_64_references(request, tmp_path, monkeypatch, clean_speaker_cache, variant):
    """Both deploy files encode 64 references together: Original's 128 sequences are 64 guided pairs."""
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    model_dir = request.getfixturevalue("original_model_dir" if variant == "original" else "model_dir")
    speech = serving(engine_client(model_dir, deploy_on_cpu(DEPLOY[variant])))
    try:
        assert speech._adapter.conditioning_capacity == 64
    finally:
        speech.shutdown()


@pytest.mark.parametrize("legacy", [False, True], ids=["runtime_config", "engine_args"])
def test_conditioning_device_comes_from_either_stage_config_form(
    model_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_speaker_cache: None, legacy: bool
) -> None:
    """The deploy file's device reaches the adapter whether the stage config is typed or legacy."""
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    speech = serving(engine_client(model_dir, deploy_on_cpu(DEPLOY["turbo"]), legacy))
    try:
        assert speech._adapter.conditioning_device == torch.device("cpu")
    finally:
        speech.shutdown()


@pytest.mark.parametrize(
    ("device", "error"),
    [("meta", "must be cpu or cuda"), (None, "additional_config.conditioning_device")],
    ids=["unsupported", "missing"],
)
def test_the_conditioning_device_must_be_a_supported_one_and_set(
    model_dir: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_speaker_cache: None,
    device: str | None,
    error: str,
):
    """No device is assumed: a deploy file without one fails at startup."""
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    deploy = modify_stage_config(
        get_deploy_config_path(DEPLOY["turbo"]),
        updates={"stages": {0: {"additional_config.conditioning_device": device}}} if device is not None else None,
        deletes={"stages": {0: ["additional_config"]}} if device is None else None,
    )
    client = engine_client(model_dir, deploy)
    with pytest.raises(ValueError, match=error):
        serving(client)


def test_original_controls_must_be_set_in_stage_config(
    model_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_speaker_cache: None
) -> None:
    """An Original deploy file whose additional_config names no controls fails at startup with a sentence."""
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    deploy = modify_stage_config(
        get_deploy_config_path(DEPLOY["original"]),
        updates={"stages": {0: {"additional_config": {"conditioning_device": "cpu"}}}},
    )
    with pytest.raises(
        ValueError, match="Original deploy files set stage 0 additional_config.cfg_weight and exaggeration"
    ):
        serving(engine_client(model_dir, deploy))


@pytest.mark.parametrize("variant", ["adapter", "original"])
def test_public_speech_token_cap_is_upstreams_generation_cap(request, variant):
    """Both checkpoints accept at most 1000 new speech tokens, the cap upstream generates with."""
    configured = request.getfixturevalue(variant)
    assert configured.validate(OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=1000)) is None
    assert "cannot exceed 1000" in configured.validate(OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=1001))
