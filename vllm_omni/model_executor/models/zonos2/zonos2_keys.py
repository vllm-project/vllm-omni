# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned runtime keys shared by frontend, sampler and stage handoff."""

FRAMES = "zonos2_frames"
SPEAKER_EMBEDDING = "zonos2_speaker_embedding"
SPEAKER_POSITION = "zonos2_speaker_position"
STATE = "zonos2"
TARGET_FRAMES = "zonos2_target"
TOKEN_BUDGET = "zonos2_token_budget"
SCHEDULED_SPAN = "_zonos2_scheduled_span"
TERMINAL = "_zonos2_terminal"
