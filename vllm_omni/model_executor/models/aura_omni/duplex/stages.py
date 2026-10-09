# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Stage ids of the AURA duplex roles.

The duplex plugin, data plane and sentence TTS address stages by role
instead of by number, so a pipeline variant that inserts a stage (the
judge after ASR) only has to declare a different layout.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuraStageLayout:
    asr: int = 0
    aura: int = 1
    talker: int = 2
    code2wav: int = 3
    # Optional response judge between ASR and AURA.
    judge: int | None = None


AURA_STAGE_LAYOUT = AuraStageLayout()
AURA_JUDGED_STAGE_LAYOUT = AuraStageLayout(asr=0, judge=1, aura=2, talker=3, code2wav=4)


__all__ = ["AURA_JUDGED_STAGE_LAYOUT", "AURA_STAGE_LAYOUT", "AuraStageLayout"]
