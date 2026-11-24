# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Turn-policy prompts for Qwen3-Omni's chat fallback."""

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

__all__ = ["INTERRUPTION_NOTE", "SYSTEM_PROMPT"]
