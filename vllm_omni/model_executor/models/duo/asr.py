# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-ASR with a text-only handoff for the Duo streaming chain."""

from vllm.model_executor.models.qwen3_asr import Qwen3ASRForConditionalGeneration


class DuoASRForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    # Realization consumes the transcript, never ASR's decoder hidden states. Keep the
    # internal forward/logits unchanged and omit their synchronous D2H payload.
    omni_pooler_payload_include_hidden = False
