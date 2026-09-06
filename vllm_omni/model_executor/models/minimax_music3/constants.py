# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-contract constants for MiniMax Music 3.

These values are fixed by the checkpoint, not tuning knobs. Changing any of
them changes the audio, so they are declared once here and imported
everywhere rather than re-derived per call site.
"""

# --- Generation limits -----------------------------------------------------

# 25 audio frames per second, capped at six minutes of song.
AUDIO_FRAME_RATE = 25
MAX_AUDIO_FRAMES = 9_000
DEFAULT_MAX_AUDIO_FRAMES = 9_000

# The backbone was trained with a 10,240-token context; the prompt is capped
# well below it so there is room for the audio frames that follow.
MAX_PROMPT_TOKENS = 5_000

# --- Frame layout ----------------------------------------------------------

# Eight RVQ codebooks per frame: c0 lives in the backbone vocabulary, c1..c7
# in the depth decoder's own 1024-entry books.
NUM_CODEBOOKS = 8
C0_VOCAB_SIZE = 16_384
AUDIO_VOCAB_SIZE = 1_024

# The acoustic stage is conditioned on the backbone hidden state concatenated
# with the seven depth-decoder hidden states.
BACKBONE_HIDDEN_SIZE = 4_096
AR_HIDDEN_SIZE = BACKBONE_HIDDEN_SIZE * NUM_CODEBOOKS  # 32768

# --- Sampling --------------------------------------------------------------

# Classifier-free guidance is not optional and has no request field: this is
# how the reference implementation samples this checkpoint.
AR_CFG_SCALE = 1.5
AR_CFG_TOP_K = 50
AR_TOP_K = 50

# --- Acoustic stage --------------------------------------------------------

DEFAULT_DIT_STEPS = 30
DEFAULT_DIT_CFG_SCALE = 1.7
DIT_LATENT_CHANNELS = 128

# Windowing: the AR stage hands the acoustic stage 200-frame windows that
# overlap by half, and the acoustic stage crops the overlap back off.
AR_CHUNK_FRAMES = 200
AR_CHUNK_HOP_FRAMES = 100

# DAV runs at 44.1 kHz; the response is resampled to 32 kHz stereo.
DAV_SAMPLE_RATE = 44_100
OUTPUT_SAMPLE_RATE = 32_000
OUTPUT_CHANNELS = 2

# Condition resampling: AR frames land on a 24 kHz / 960-hop grid and are
# stretched onto the DAV latent's 44.1 kHz / 512-hop grid.
CONDITION_INPUT_SAMPLE_RATE = 24_000
CONDITION_INPUT_HOP = 960
CONDITION_OUTPUT_SAMPLE_RATE = 44_100
CONDITION_OUTPUT_HOP = 512

# Mel frames per AR frame: 44100/24000 * 960/512 = 3.4453125. So a 100-frame
# hop is 344 mel frames and a 200-frame window is 689.
#
# HOP_MEL_FRAMES is the hop, not the window: the crop and carry-forward
# arithmetic is expressed in units of the hop, and a window is two of them
# plus one frame of rounding. Reading this as the window length is the easy
# mistake here, and it is off by a factor of two.
HOP_MEL_FRAMES = 344
BLEND_MEL_FRAMES = HOP_MEL_FRAMES // 4
DAV_HOP_SAMPLES = 512

# Only these two dtypes are supported for the acoustic stage. float32 is the
# default: TF32 keeps it affordable, and bfloat16 measurably degrades the
# solver.
SUPPORTED_ACOUSTIC_DTYPES = frozenset({"float32", "bfloat16"})


__all__ = [
    "AR_CFG_SCALE",
    "AR_CFG_TOP_K",
    "AR_CHUNK_FRAMES",
    "AR_CHUNK_HOP_FRAMES",
    "AR_HIDDEN_SIZE",
    "AR_TOP_K",
    "AUDIO_FRAME_RATE",
    "AUDIO_VOCAB_SIZE",
    "BACKBONE_HIDDEN_SIZE",
    "BLEND_MEL_FRAMES",
    "C0_VOCAB_SIZE",
    "CONDITION_INPUT_HOP",
    "CONDITION_INPUT_SAMPLE_RATE",
    "CONDITION_OUTPUT_HOP",
    "CONDITION_OUTPUT_SAMPLE_RATE",
    "DAV_HOP_SAMPLES",
    "DAV_SAMPLE_RATE",
    "DEFAULT_DIT_CFG_SCALE",
    "DEFAULT_DIT_STEPS",
    "DEFAULT_MAX_AUDIO_FRAMES",
    "DIT_LATENT_CHANNELS",
    "HOP_MEL_FRAMES",
    "MAX_AUDIO_FRAMES",
    "MAX_PROMPT_TOKENS",
    "NUM_CODEBOOKS",
    "OUTPUT_CHANNELS",
    "OUTPUT_SAMPLE_RATE",
    "SUPPORTED_ACOUSTIC_DTYPES",
]
