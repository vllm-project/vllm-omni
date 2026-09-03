# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Canonical cross-model constant definitions (RFC #4872).

Special-token ids that are shared across model pipelines / processors /
model implementations live here so no module hard-codes a magic number that
could drift out of sync.  This module is deliberately **dependency-free**
(no torch / vllm imports) so pure-config modules (e.g. pipeline topologies)
can import it without pulling in the runtime stack.

**Scope note (conservative):** only constants genuinely shared across more
than one module belong here.  Model-specific codec ids keep their module-top
home with the processor as source and the HF config as the runtime single
source of truth (see ``qwen3_omni._assert_codec_token_ids_consistent`` and
``qwen2_5_omni.TALKER_CODEC_END_TOKEN_ID``).
"""

# Qwen3-family talker codec stop (EOS) token id.  Used as the AR talker
# stage's ``stop_token_ids`` across the qwen3_omni / qwen3_tts / aura_omni
# pipelines (three pipeline files previously shared the literal ``2150``).
QWEN3_CODEC_EOS_TOKEN_ID = 2150
