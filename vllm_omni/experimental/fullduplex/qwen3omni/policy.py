# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Turn-policy prompts for the Qwen3-Omni duplex adapter.

Qwen3-Omni has no model-native listen/speak tokens (MiniCPM-specific), so the
turn policy is expressed as a system prompt injected on every turn, plus a
per-turn interruption note injected only after a barge-in.
"""

SYSTEM_PROMPT = (
    "You are a voice assistant in a real-time duplex conversation.\n"
    "- Answer in short spoken turns; do not recite long text.\n"
    "- If the user speaks over you or your reply is interrupted, stop immediately.\n"
    "- Never continue a reply that was interrupted.\n"
    "- Respond in natural speech; no markdown, no lists, no code blocks."
)

INTERRUPTION_NOTE = (
    "Note: your previous reply was interrupted by the user. Discard it.\nRespond only to the user's latest input."
)
