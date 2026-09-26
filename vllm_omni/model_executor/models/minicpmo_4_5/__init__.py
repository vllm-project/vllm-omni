# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# NOTE: Do not import model classes in this file. Importing any
# submodule in this package triggers __init__.py execution, and
# both the model registry and pipeline registry import submodules
# directly — heavy imports here would be loaded as a side effect
# even though nothing depends on these re-exports.

# One MiniCPMTTS.generate_chunk emits 25 codec frames plus its terminator.
MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK = 26
# Codec budget for the Talker chunk that carries the Thinker's turn end. The
# Talker paces ~1 s of speech per unit and emits its chunk EOS at the 25-frame
# cadence, while the text it still owes at turn end can exceed one unit. The
# turn-end chunk may run for several units so the tail of the turn is spoken
# (see ``_native_duplex_chunk_budget`` and the boundary EOS mask in the Talker).
MINICPMO45_DUPLEX_TURN_END_CODEC_TOKENS = MINICPMO45_DUPLEX_CODEC_TOKENS_PER_CHUNK * 4
