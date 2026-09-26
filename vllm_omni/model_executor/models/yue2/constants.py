# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Token/protocol constants for YuE2-3B, pinned by the checkpoint.

Values mirror the upstream ``yue2/protocol.py`` (bd90e4c) exactly; changing any
of them desynchronizes generation from the reference implementation. The model
is single-codebook: one codec token per latent frame, 25 frames per second,
so ``max_tokens`` in engine terms equals frames/25 seconds of music.
"""

from __future__ import annotations

EOD = 151643
ABC_START, ABC_END = 151847, 151848
MUSIC_START, MUSIC_END = 151851, 151852
CODEC_OFFSET, CODEC_SIZE = 151853, 32768
LATENT_START, LATENT_END, LATENT_PAD = 184621, 184622, 184623
VOCAB_SIZE, CONTEXT = 184704, 24576

# Engine-side stop ids: the union of both phase ends. The model's own sampler
# only ever emits the end token of the request's current phase, so a request
# never stops on the other phase's end by accident.
STOP_TOKEN_IDS = [ABC_END, MUSIC_END]

SAMPLE_RATE = 48000
AUDIO_CHANNELS = 2
FRAMES_PER_SECOND = 25
SAMPLES_PER_FRAME = 1920
LATENT_DIM = 64

# Reference sampling presets (upstream yue2_generation_config.json).
ABC_SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 30,
    "repetition_penalty": 1.005,
    "penalty_window": 100,
    "min_tokens": 32,
    "max_tokens": 4096,
}
SEMANTIC_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 100,
    "repetition_penalty": 1.2,
    "penalty_window": 50,
    "min_tokens": 200,
    "max_tokens": 9000,
}

ODE_STEPS = 32
ODE_METHOD = "midpoint"
VAE_CORE_FRAMES = 1024
VAE_HALO_FRAMES = 16
DEFAULT_VAE_ID = "m-a-p/YuE2-Vae"

# extra_args keys the model reads per request.
KEY_PHASE = "yue2_phase"  # "abc" | "semantic"
KEY_TEMPERATURE = "yue2_temperature"
KEY_TOP_P = "yue2_top_p"
KEY_TOP_K = "yue2_top_k"
KEY_REPETITION_PENALTY = "yue2_repetition_penalty"
KEY_PENALTY_WINDOW = "yue2_penalty_window"
KEY_MIN_TOKENS = "yue2_min_tokens"
KEY_MAX_AUDIO_FRAMES = "yue2_max_audio_frames"
KEY_SEED = "yue2_seed"
KEY_SKIP_SYNTHESIS = "yue2_skip_synthesis"  # abc phase: tokens only, no audio
# Full prompt token ids, shipped by the driver. The NAR conditioning needs the
# whole prefix, but under a KV prefix-cache hit the engine schedules only the
# uncached tail, so the scheduled input_ids slice cannot rebuild it.
KEY_PREFIX_IDS = "yue2_prefix_ids"

__all__ = [name for name in dir() if name.isupper()]
