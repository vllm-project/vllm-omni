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
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
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
        write_s3gen_checkpoint(directory, ChatterboxConfig(variant), seed=1)
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


def serving(client: SimpleNamespace) -> OmniOpenAIServingSpeech:
    models = OpenAIServingModels(client, [BaseModelPath(name=client.model, model_path=client.model)])
    return OmniOpenAIServingSpeech(engine_client=client, models=models, request_logger=None)


@pytest.fixture
def server(
    model_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_speaker_cache: None
) -> Iterator[OmniOpenAIServingSpeech]:
    """The speech server for Turbo's deploy file; uploaded voices are kept under the test's directory.

    Every server shares the process-wide speaker cache, which the repository's
    ``clean_speaker_cache`` fixture resets around the test.
    """
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    speech = serving(engine_client(model_dir, DEPLOY["turbo"]))
    yield speech
    speech.shutdown()


@pytest.fixture
def adapter(server: OmniOpenAIServingSpeech) -> ChatterboxAdapter:
    return server._adapter


def reference(seconds: float, seed: int, sample_rate: int = 24000, mime: str = "audio/wav") -> str:
    """A reference recording as a client sends it: a base64 WAV data URL."""
    samples = (0.1 * np.random.default_rng(seed).standard_normal(round(seconds * sample_rate))).astype(np.float32)
    buffer = io.BytesIO()
    soundfile.write(buffer, samples, sample_rate, format="WAV")
    return f"data:{mime};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def encoder(adapter: ChatterboxAdapter) -> VoiceConditioner:
    """The adapter's reference encoders, loaded as its first encoder batch loads them."""
    assert adapter.prepare_voices([]) == []
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
def test_unsupported_request_is_rejected(adapter, fields, error):
    request = OpenAICreateSpeechRequest(**{"input": "Hello.", **fields})
    assert error in adapter.validate(request)


@pytest.mark.parametrize(
    "fields", [{}, {"voice": "default"}, {"language": "en"}, {"ref_audio": "data:audio/wav;base64,UklGRg=="}]
)
def test_builtin_and_reference_requests_are_accepted(adapter, fields):
    assert adapter.validate(OpenAICreateSpeechRequest(input="Hello.", **fields)) is None


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
    assert adapter.conditioner is None
    assert prepared.model_type == "chatterbox"


def test_request_token_cap_does_not_mutate_deploy_defaults(adapter):
    defaults = [SamplingParams(max_tokens=1000), SamplingParams(max_tokens=1)]
    result = adapter.apply_sampling_overrides(defaults, OpenAICreateSpeechRequest(input="Hi.", max_new_tokens=50))
    assert [params.max_tokens for params in result] == [50, 1]
    assert defaults[0].max_tokens == 1000


@pytest.mark.parametrize("value", [-1, float("inf"), "0.5"])
def test_original_rejects_invalid_guidance_controls(adapter, value):
    original = ChatterboxOriginalAdapter(adapter.ctx)
    request = OpenAICreateSpeechRequest(input="Hello.", extra_params={"cfg_weight": value})
    assert "finite nonnegative" in original.validate(request)


def test_original_owns_its_stage_and_controls(adapter):
    assert detect_tts_model_type("chatterbox_original_t3", None) == "chatterbox_original"
    original = ChatterboxOriginalAdapter(adapter.ctx)
    assert original.config.variant == "original"
    request = OpenAICreateSpeechRequest(input="Hello.", extra_params={"cfg_weight": 0, "exaggeration": 0.7})
    assert original.validate(request) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_cls", "seconds"), [(ChatterboxAdapter, 5.04), (ChatterboxOriginalAdapter, 1.0)])
async def test_64_distinct_references_are_batched_and_cached(adapter, adapter_cls, seconds):
    """64 references arriving together are encoded at most eight at a time, one batch at a time.

    Each clip is 40 ms (one prompt token) longer than the one before, so the
    prompt lengths show every voice is its own reference's. A second wave of
    the same references is served from the cache.
    """
    concurrent = adapter_cls(adapter.ctx)
    batches = EncoderBatches(encoder(concurrent))
    sources = [reference(seconds + 0.04 * index, seed=index) for index in range(64)]
    try:
        voices = await asyncio.gather(*(concurrent.resolve_voice(source) for source in sources))
        encoded = list(batches.batches)
        cached = await asyncio.gather(*(concurrent.resolve_voice(source) for source in sources))
    finally:
        batches.remove()

    lengths = [voice.prompt_token.shape[1] for voice in voices]
    assert lengths == list(range(lengths[0], lengths[0] + 64))
    assert sum(encoded) == 64
    assert max(encoded) <= 8
    assert len(encoded) < 64
    assert batches.most_active == 1
    assert all(left is right for left, right in zip(voices, cached, strict=True))
    assert batches.batches == encoded
    assert not concurrent.conditioning_pending


@pytest.mark.asyncio
async def test_reference_singleflight_survives_waiter_cancellation(adapter):
    """Two locators of one waveform share one encoding, which outlives the waiter that started it."""
    batches = EncoderBatches(encoder(adapter), hold=True)
    try:
        first = asyncio.create_task(adapter.resolve_voice(reference(6, seed=1)))
        await asyncio.to_thread(batches.started.wait, 30)
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
    assert list(adapter.conditioning_cache.values()) == [voice]


@pytest.mark.asyncio
async def test_bad_reference_does_not_poison_batch(adapter):
    """A clip Turbo refuses fails alone; its batch's other clip is encoded and cached."""
    batches = EncoderBatches(encoder(adapter))
    bad, good = reference(2, seed=2), reference(6, seed=3)
    try:
        results = await asyncio.gather(adapter.resolve_voice(bad), adapter.resolve_voice(good), return_exceptions=True)
    finally:
        batches.remove()

    assert isinstance(results[0], ValueError) and "longer than 5 s" in str(results[0])
    assert isinstance(results[1], VoiceConditioning)
    assert batches.batches == [1]
    assert list(adapter.conditioning_cache.values()) == [results[1]]
    assert not adapter.conditioning_pending


@pytest.mark.asyncio
async def test_cancelled_wave_cannot_overfill_encoder_queue(adapter):
    """Cancelled waiters keep their encoder work, and with it the slots a new wave would take."""
    original = ChatterboxOriginalAdapter(adapter.ctx)
    batches = EncoderBatches(encoder(original), hold=True)
    try:
        first = [asyncio.create_task(original.resolve_voice(reference(1, seed=index))) for index in range(64)]
        await asyncio.to_thread(batches.started.wait, 30)
        for request in first:
            request.cancel()
        await asyncio.gather(*first, return_exceptions=True)
        second = [asyncio.create_task(original.resolve_voice(reference(1, seed=index))) for index in range(64, 128)]
        await asyncio.sleep(0.1)
        assert len(original.conditioning_pending) == 64
        assert original.conditioning_queue.qsize() <= 64
        assert not any(request.done() for request in second)
        batches.released.set()
        voices = await asyncio.wait_for(asyncio.gather(*second), 60)
    finally:
        batches.remove()

    assert all(isinstance(voice, VoiceConditioning) for voice in voices)
    assert not original.conditioning_pending


@pytest.mark.asyncio
async def test_shutdown_cancels_encoder_waiters(adapter):
    batches = EncoderBatches(encoder(adapter), hold=True)
    try:
        request = asyncio.create_task(adapter.resolve_voice(reference(6, seed=4)))
        await asyncio.to_thread(batches.started.wait, 30)
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


@pytest.mark.parametrize("legacy", [False, True], ids=["runtime_config", "engine_args"])
def test_conditioning_device_is_explicit_in_stage_config(
    model_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_speaker_cache: None, legacy: bool
) -> None:
    """The high-concurrency deploy file puts the reference encoders on the GPU, in either stage config form."""
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    speech = serving(engine_client(model_dir, "chatterbox_turbo_high_concurrency.yaml", legacy))
    try:
        assert speech._adapter.conditioning_device == torch.device("cuda:0")
    finally:
        speech.shutdown()


def test_unsupported_conditioning_device_is_rejected(
    model_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_speaker_cache: None
):
    monkeypatch.setenv("SPEAKER_SAMPLES_DIR", str(tmp_path / "speakers"))
    high_concurrency = Path(get_deploy_config_path("chatterbox_turbo_high_concurrency.yaml"))
    deploy = modify_stage_config(
        str(high_concurrency),
        updates={
            # The copy lives elsewhere, so it names its base file by path.
            "base_config": str(high_concurrency.parent / DEPLOY["turbo"]),
            "stages": {0: {"additional_config.conditioning_device": "meta"}},
        },
    )
    client = engine_client(model_dir, deploy)
    with pytest.raises(ValueError, match="must be cpu or cuda"):
        serving(client)


@pytest.mark.parametrize("adapter_cls", [ChatterboxAdapter, ChatterboxOriginalAdapter])
def test_public_speech_token_cap_is_enforced_before_admission(adapter, adapter_cls):
    configured = adapter_cls(adapter.ctx)
    assert configured.validate(OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=4096)) is None
    assert "4096" in configured.validate(OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=4097))
