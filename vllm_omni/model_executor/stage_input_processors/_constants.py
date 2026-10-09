# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Canonical cross-model constant definitions.

Special-token ids shared across model pipelines / processors / model
implementations live here so no module hard-codes a magic number that could
drift out of sync.  This module is deliberately **dependency-free** (no torch /
vllm imports) so pure-config modules can import it without pulling in the
runtime stack.

**Scope note:** only constants genuinely shared across more than one module
belong here.  Model-specific codec ids keep their module-top home with the
processor as source and the HF config as the runtime single source of truth.
"""

# Qwen3-family talker codec stop (EOS) token id; used as the AR talker
# stage's ``stop_token_ids`` across the qwen3_omni / qwen3_tts / aura_omni
# pipelines.
QWEN3_CODEC_EOS_TOKEN_ID = 2150
