# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Resolve AURA duplex stage ids from pipeline roles."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.config.stage_config import PipelineConfig
from vllm_omni.model_executor.models.aura_omni.pipeline import AURA_OMNI_JUDGED_PIPELINE, AURA_OMNI_PIPELINE


@dataclass(frozen=True, slots=True)
class AuraStageLayout:
    asr: int
    aura: int
    talker: int
    code2wav: int
    judge: int | None = None

    @classmethod
    def from_pipeline(cls, pipeline_config: PipelineConfig) -> AuraStageLayout:
        roles: dict[str, int] = {}
        stage_ids: set[int] = set()
        required = {"asr", "aura", "qwen3_tts", "code2wav"}
        for stage in pipeline_config.stages:
            if stage.stage_id in stage_ids:
                raise ValueError(f"AURA pipeline has duplicate stage id {stage.stage_id}")
            stage_ids.add(stage.stage_id)
            if stage.model_stage not in required | {"response_judge"}:
                continue
            if stage.model_stage in roles:
                raise ValueError(f"AURA pipeline has duplicate role {stage.model_stage!r}")
            roles[stage.model_stage] = stage.stage_id
        missing = required - roles.keys()
        if missing:
            raise ValueError(f"AURA pipeline is missing required roles: {', '.join(sorted(missing))}")
        return cls(
            asr=roles["asr"],
            aura=roles["aura"],
            talker=roles["qwen3_tts"],
            code2wav=roles["code2wav"],
            judge=roles.get("response_judge"),
        )


AURA_STAGE_LAYOUT = AuraStageLayout.from_pipeline(AURA_OMNI_PIPELINE)
AURA_JUDGED_STAGE_LAYOUT = AuraStageLayout.from_pipeline(AURA_OMNI_JUDGED_PIPELINE)


__all__ = ["AURA_JUDGED_STAGE_LAYOUT", "AURA_STAGE_LAYOUT", "AuraStageLayout"]
