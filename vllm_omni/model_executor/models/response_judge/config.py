# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared prompt/head configuration for LAYA, without model imports."""

from collections.abc import Mapping

LAYA_QTYPES = {"choice": 0, "score": 1, "noul": 2}


def laya_question_type(config: object) -> str:
    model_qtype = getattr(config, "laya_question_type", None)
    judge = getattr(config, "response_judge", None)
    judge_qtype = judge.get("question_type") if isinstance(judge, Mapping) else None
    for value in (model_qtype, judge_qtype):
        if value is not None and (not isinstance(value, str) or value not in LAYA_QTYPES):
            raise ValueError("LAYA question_type must be a name: choice, score or noul; integer IDs are not supported")
    if model_qtype is not None and judge_qtype is not None and model_qtype != judge_qtype:
        raise ValueError(
            f"laya_question_type={model_qtype!r} does not match response_judge.question_type={judge_qtype!r}"
        )
    return model_qtype or judge_qtype or "choice"
