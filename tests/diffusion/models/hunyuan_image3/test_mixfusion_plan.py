# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm_omni.diffusion.models.hunyuan_image3.mixfusion import (
    build_mixfusion_sequence_plan,
    validate_mixfusion_sequence_plan,
)


def test_mixfusion_rejects_small_gcd_many_chunks() -> None:
    plan = build_mixfusion_sequence_plan([(64, 64), (52, 76)])

    use_mixfusion, reason = validate_mixfusion_sequence_plan(plan)

    assert not use_mixfusion
    assert plan.chunk_size == 16
    assert plan.chunk_count == 503
    assert "chunk_size=16" in reason


def test_mixfusion_accepts_large_gcd_few_chunks() -> None:
    plan = build_mixfusion_sequence_plan([(64, 64), (32, 32)])

    use_mixfusion, reason = validate_mixfusion_sequence_plan(plan)

    assert use_mixfusion
    assert reason == "ok"
    assert plan.chunk_size == 1024
    assert plan.chunk_count == 5
