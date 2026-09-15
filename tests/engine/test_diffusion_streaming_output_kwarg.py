# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``diffusion_streaming_output`` must reach a diffusion stage as ``streaming_output``.

The public AsyncOmni/serve kwarg and the stage schema field have different
names, and the two config paths translate it in different places: the
unregistered single-stage fallback writes it into the default stage config,
while a registered pipeline resolves through ``StageConfigFactory``, which drops
keys the stage schema does not know. Without the mirror in
``_resolve_stage_configs`` a registered pipeline (LingBot World stepwise) serves
only its final chunk instead of streaming one per step.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _resolve(monkeypatch, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Run ``_resolve_stage_configs`` and return the kwargs it forwarded."""

    seen: dict[str, Any] = {}

    def _fake_load(model, forwarded_kwargs, **_: Any):
        seen.update(forwarded_kwargs)
        return "config.yaml", [], None

    monkeypatch.setattr(
        "vllm_omni.engine.async_omni_engine.load_and_resolve_stage_configs",
        _fake_load,
    )
    engine = SimpleNamespace(
        _create_default_diffusion_stage_cfg=AsyncOmniEngine._create_default_diffusion_stage_cfg,
        _apply_strategy_lb_policy=lambda *_args, **_kw: None,
    )
    AsyncOmniEngine._resolve_stage_configs(
        engine,
        "some/model",
        kwargs,
        trust_remote_code=None,
    )
    return seen


def test_registered_pipeline_receives_streaming_output(monkeypatch) -> None:
    forwarded = _resolve(monkeypatch, {"diffusion_streaming_output": True})

    assert forwarded["streaming_output"] is True


def test_disabled_kwarg_does_not_override_the_deploy_config(monkeypatch) -> None:
    # ``serve`` always carries the flag's ``False`` default; mirroring it would
    # silently turn streaming off for a deploy YAML that asks for it.
    forwarded = _resolve(monkeypatch, {"diffusion_streaming_output": False})

    assert "streaming_output" not in forwarded


def test_explicit_streaming_output_wins(monkeypatch) -> None:
    forwarded = _resolve(
        monkeypatch,
        {"diffusion_streaming_output": True, "streaming_output": False},
    )

    assert forwarded["streaming_output"] is False


@pytest.mark.parametrize("enabled", [True, False])
def test_unregistered_fallback_stage_carries_the_flag(enabled: bool) -> None:
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg({"diffusion_streaming_output": enabled})[0]

    assert stage_cfg["engine_args"]["streaming_output"] is enabled
