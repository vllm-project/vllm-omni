# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tier 2 integration smoke test for ``AceStepPipeline``.

Instantiates the pipeline via ``object.__new__`` to bypass the heavy
``__init__`` (which calls ``from_pretrained`` on tokenizer / text_encoder /
vae / scheduler — all real-checkpoint paths), then attaches:

  * tiny real ``AceStepTransformer1DModel`` and ``AceStepConditionEncoder``
    (random init) for the actual ACE-Step math,
  * mock tokenizer / text encoder / VAE for the I/O surface,
  * a real ``FlowMatchEulerDiscreteScheduler`` (lightweight, no checkpoint).

Runs ``forward`` end-to-end with a minimal ``OmniDiffusionRequest`` so all
eight pipeline steps execute on CPU in a few seconds. Catches wiring bugs
that would otherwise cost GPU hours: shape mismatches between components,
scheduler misuse, request unpacking errors, missing methods.

What this DOES verify:
    - All 8 forward steps run without crashing
    - Component-to-component shapes line up
    - Scheduler config consumed correctly
    - OmniDiffusionRequest unpacking works
    - Output is a populated DiffusionOutput with the right rank/shape

What this does NOT verify:
    - Audio quality (mock VAE returns randn; tiny DiT/condition encoder
      have random init weights — output is junk by design).
    - Real text encoder behaviour (mock embeds a fake vocab).
    - Numerical stability of the real model at full scale.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.models.ace_step.ace_step_transformer import (
    AceStepTransformer1DModel,
)
from vllm_omni.diffusion.models.ace_step.modeling_ace_step import (
    AceStepConditionEncoder,
)
from vllm_omni.diffusion.models.ace_step.pipeline_ace_step import AceStepPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# --------------------------------------------------------------------------- #
#                           single-process TP init                              #
# --------------------------------------------------------------------------- #
# Mirrors test_ace_step_transformer.py — ReplicatedLinear inside the tiny
# transformer / condition encoder needs the TP group set up before construction.


@pytest.fixture(scope="module")
def _init_single_process_tp():
    import os

    from vllm.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29504")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        init_distributed_environment(world_size=1, rank=0, local_rank=0)
        initialize_model_parallel(tensor_model_parallel_size=1)
    yield


# --------------------------------------------------------------------------- #
#                              I/O mocks                                       #
# --------------------------------------------------------------------------- #


class _MockTokenizerOutput:
    """Mimics the ``transformers.BatchEncoding`` surface our pipeline uses."""

    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class _MockTokenizer:
    """Returns fixed-length input_ids / all-ones attention_mask of length 8."""

    def __call__(
        self,
        strs,
        padding=None,
        truncation=None,
        max_length=None,
        return_tensors=None,
    ):
        batch_size = len(strs)
        seq_len = 8
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        return _MockTokenizerOutput(input_ids, attention_mask)


class _MockTextEncoderOutput:
    def __init__(self, last_hidden_state: torch.Tensor):
        self.last_hidden_state = last_hidden_state


class _MockTextEncoder(nn.Module):
    """Stub Qwen3-Embedding: an Embedding layer that doubles as ``get_input_embeddings``."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 32):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)

    def forward(self, input_ids=None):
        hidden = self.embed_tokens(input_ids)
        return _MockTextEncoderOutput(last_hidden_state=hidden)

    def get_input_embeddings(self):
        return self.embed_tokens


class _MockVAEDecodeOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


class _MockVAE(nn.Module):
    """Stub ``AutoencoderOobleck``: ``decode`` upsamples by ``hop_length``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        hop_length: int = 4,
        sampling_rate: int = 20,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hop_length = hop_length
        self.config = SimpleNamespace(sampling_rate=sampling_rate)
        self.dtype = torch.float32

    def decode(self, x: torch.Tensor) -> _MockVAEDecodeOutput:
        batch_size, _, latent_len = x.shape
        audio = torch.randn(batch_size, self.out_channels, latent_len * self.hop_length)
        return _MockVAEDecodeOutput(sample=audio)


# --------------------------------------------------------------------------- #
#                             tiny model configs                               #
# --------------------------------------------------------------------------- #
# Same proportions as the existing forward-shape tests; the DiT and condition
# encoder are real ACE-Step modules, just with small dimensions so the test
# finishes in seconds on CPU.


_TINY_DIT_KWARGS = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    in_channels=24,
    audio_acoustic_hidden_dim=8,
    patch_size=2,
    sliding_window=4,
)


_TINY_COND_KWARGS = dict(
    hidden_size=64,
    intermediate_size=64,
    text_hidden_dim=32,
    # ``timbre_hidden_dim`` must equal the DiT's ``audio_acoustic_hidden_dim``
    # in production (the converter writes silence_latent at that width and the
    # pipeline slices it directly into the ``src_latents`` half of
    # ``context_latents``). Keep them aligned here so the tiny test exercises
    # the real wiring.
    timbre_hidden_dim=_TINY_DIT_KWARGS["audio_acoustic_hidden_dim"],
    num_lyric_encoder_hidden_layers=2,
    num_timbre_encoder_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    sliding_window=4,
)


# --------------------------------------------------------------------------- #
#                           the integration test                                #
# --------------------------------------------------------------------------- #


def _build_pipeline_with_mocks() -> AceStepPipeline:
    """Construct a pipeline whose __init__ would normally call from_pretrained.

    Uses object.__new__ + manual attribute assignment so we never touch a real
    checkpoint. Mirrors the parallelism-contract tests' use of object.__new__.
    """
    pipeline = object.__new__(AceStepPipeline)
    nn.Module.__init__(pipeline)

    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(
        dtype=torch.float32,
        enable_diffusion_pipeline_profiler=False,
    )

    pipeline._guidance_scale = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.is_turbo = True
    # 5 latent frames per second → timbre_fix_frame = ceil(30 * 5) = 150, fits
    # well inside the condition encoder's silence_latent buffer (15000 frames).
    pipeline.latents_per_second = 5.0

    pipeline.transformer = AceStepTransformer1DModel(**_TINY_DIT_KWARGS).eval()
    pipeline.condition_encoder = AceStepConditionEncoder(**_TINY_COND_KWARGS).eval()

    pipeline.tokenizer = _MockTokenizer()
    pipeline.text_encoder = _MockTextEncoder(
        vocab_size=100,
        hidden_dim=_TINY_COND_KWARGS["text_hidden_dim"],
    )
    pipeline.vae = _MockVAE(
        in_channels=_TINY_DIT_KWARGS["audio_acoustic_hidden_dim"],
        out_channels=2,
        hop_length=4,
        sampling_rate=20,
    )

    # FlowMatchEulerDiscreteScheduler instantiated standalone — no checkpoint
    # needed since the pipeline supplies its own sigmas via set_timesteps.
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1, shift=1.0)

    return pipeline


def test_pipeline_forward_runs_end_to_end_on_cpu(_init_single_process_tp):
    """All 8 forward steps execute and the output has the expected shape."""
    torch.manual_seed(0)

    pipeline = _build_pipeline_with_mocks()

    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        seed=0,
        extra_args={
            "audio_duration": 2.0,
            "lyrics": "",
            "vocal_language": "en",
            "shift": 3.0,
        },
    )
    req = OmniDiffusionRequest(
        prompts=["test prompt for ace-step pipeline smoke test"],
        sampling_params=sampling_params,
        request_id="ace-step-smoke-test-0",
    )

    with torch.no_grad():
        output = pipeline.forward(req)

    audio = output.output

    # Expected derived shape:
    #   latent_length = ceil(audio_duration * latents_per_second)
    #                 = ceil(2.0 * 5.0) = 10
    #   audio_length  = latent_length * vae.hop_length = 10 * 4 = 40
    expected_batch = 1
    expected_channels = 2  # MockVAE outputs stereo
    expected_min_samples = 30  # ~40, allow slack for any boundary handling

    assert audio is not None, "forward returned a DiffusionOutput with no audio payload"
    assert audio.ndim == 3, f"expected [B, C, T] audio, got rank {audio.ndim}"
    assert audio.shape[0] == expected_batch, f"expected batch {expected_batch}, got {audio.shape[0]}"
    assert audio.shape[1] == expected_channels, f"expected {expected_channels} audio channels, got {audio.shape[1]}"
    assert audio.shape[2] >= expected_min_samples, (
        f"expected at least {expected_min_samples} samples, got {audio.shape[2]}"
    )


def test_pipeline_forward_returns_latent_when_requested(_init_single_process_tp):
    """``output_type='latent'`` short-circuits the VAE decode + normalization."""
    torch.manual_seed(0)

    pipeline = _build_pipeline_with_mocks()

    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        seed=0,
        extra_args={
            "audio_duration": 2.0,
            "lyrics": "",
            "vocal_language": "en",
            "shift": 3.0,
        },
    )
    req = OmniDiffusionRequest(
        prompts=["another test prompt"],
        sampling_params=sampling_params,
        request_id="ace-step-smoke-test-1",
    )

    with torch.no_grad():
        output = pipeline.forward(req, output_type="latent")

    latents = output.output

    # Expected: [B, latent_length, audio_acoustic_hidden_dim]
    #         = [1, 10, 8]
    assert latents is not None
    assert latents.ndim == 3, f"expected [B, T, C] latents, got rank {latents.ndim}"
    assert latents.shape[0] == 1
    assert latents.shape[1] == 10  # ceil(2.0 * 5.0)
    assert latents.shape[2] == _TINY_DIT_KWARGS["audio_acoustic_hidden_dim"]
