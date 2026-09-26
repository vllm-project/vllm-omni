# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""YuE2-3B text-to-music model package."""

from .constants import STOP_TOKEN_IDS
from .pipeline import YUE2_PIPELINE

__all__ = ["STOP_TOKEN_IDS", "YUE2_PIPELINE"]
