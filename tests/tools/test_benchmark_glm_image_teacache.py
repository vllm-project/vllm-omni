# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for GLM-Image TeaCache benchmark session reuse / warmup order."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import yaml

from tools.teacache.benchmark_glm_image import (
    OmniSession,
    deploy_config_with_teacache,
    sampling_params_for_engine,
    warm_and_time,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeOmni:
    constructions = 0
    closes = 0

    def __init__(self, **kwargs):
        type(self).constructions += 1
        self.kwargs = kwargs
        self.closed = False
        self.calls: list[str] = []

    def generate(self, prompt, sampling_params):
        self.calls.append(f"generate:{prompt}")
        return [SimpleNamespace(images=["img"])]

    def close(self):
        self.closed = True
        type(self).closes += 1


@pytest.fixture(autouse=True)
def _reset_fake_omni():
    _FakeOmni.constructions = 0
    _FakeOmni.closes = 0
    yield


def test_omni_session_warms_before_timed_repeats_on_one_engine():
    with OmniSession(_FakeOmni, model="fake", cache_backend="tea_cache") as session:
        assert _FakeOmni.constructions == 1
        times, last = warm_and_time(
            session,
            prompt="p0",
            sampling_params=SimpleNamespace(),
            repeats=3,
        )
        # 1 warm + 3 timed generates on the same instance.
        assert session.generate_calls == 4
        assert len(times) == 3
        assert last[0].images == ["img"]
        assert session.omni.calls[0] == "generate:p0"
        assert len(session.omni.calls) == 4

    assert _FakeOmni.constructions == 1
    assert _FakeOmni.closes == 1
    assert session.omni.closed is True


def test_two_stage_generate_keeps_ar_params_and_replaces_dit():
    class _TwoStage:
        num_stages = 2
        default_sampling_params_list = ["ar-default", "dit-default"]

        def __init__(self):
            self.seen = None

        def generate(self, prompt, sampling_params):
            self.seen = sampling_params
            return []

    omni = _TwoStage()
    session = OmniSession(lambda **kwargs: omni)
    session.generate("p", "dit-override")
    assert omni.seen == ["ar-default", "dit-override"]
    assert sampling_params_for_engine(SimpleNamespace(num_stages=1), "only") == "only"


def test_glm_two_stage_params_pass_live_sampling_resolver():
    """Regression for the registered GLM AR+DiT contract (#9715 review).

    A single OmniDiffusionSamplingParams must fail the live resolver on a
    two-stage engine. The benchmark helper must produce a list that reaches
    generation instead of argument validation.
    """
    from unittest.mock import MagicMock

    from vllm.sampling_params import SamplingParams

    from vllm_omni.entrypoints.omni_base import OmniBase
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    base = object.__new__(OmniBase)
    base.engine = MagicMock()
    base.engine.num_stages = 2
    ar_default = SamplingParams(max_tokens=128, seed=42)
    dit_default = OmniDiffusionSamplingParams(
        height=1024,
        width=1024,
        num_inference_steps=50,
        seed=42,
    )
    base.default_sampling_params_list = [ar_default, dit_default]
    base.sampling_constraints_list = [{}, {}]

    dit_override = OmniDiffusionSamplingParams(
        height=512,
        width=512,
        num_inference_steps=8,
        seed=7,
        extra_args={"prior_token_ids": [1, 2, 3]},
    )

    with pytest.raises(ValueError, match="Expected 2 sampling params"):
        base.resolve_sampling_params_list(dit_override)

    # Fake engine used by OmniSession.generate: expose the same stage contract.
    fake = SimpleNamespace(
        num_stages=2,
        default_sampling_params_list=base.default_sampling_params_list,
    )
    params = sampling_params_for_engine(fake, dit_override)
    resolved = base.resolve_sampling_params_list(params)
    assert len(resolved) == 2
    assert resolved[0] is ar_default or resolved[0].max_tokens == 128
    assert resolved[1].height == 512
    assert resolved[1].width == 512
    assert resolved[1].num_inference_steps == 8
    assert resolved[1].extra_args["prior_token_ids"] == [1, 2, 3]


def test_omni_session_constructs_once_per_config():
    with OmniSession(_FakeOmni, model="fake") as baseline:
        warm_and_time(baseline, prompt="a", sampling_params=None, repeats=2)
    with OmniSession(_FakeOmni, model="fake", cache_backend="tea_cache") as cached:
        warm_and_time(cached, prompt="a", sampling_params=None, repeats=2)

    assert _FakeOmni.constructions == 2
    assert _FakeOmni.closes == 2


def test_deploy_config_with_teacache_patches_last_stage(tmp_path):
    base = tmp_path / "glm_image.yaml"
    base.write_text(
        yaml.safe_dump(
            {
                "stages": [
                    {"stage_id": 0, "stage_type": "llm"},
                    {"stage_id": 1, "stage_type": "diffusion"},
                ]
            },
            sort_keys=False,
        )
    )
    out = deploy_config_with_teacache(base, rel_l1_thresh=0.2)
    patched = yaml.safe_load(out.read_text())
    assert patched["stages"][0].get("cache_backend") is None
    assert patched["stages"][1]["cache_backend"] == "tea_cache"
    assert patched["stages"][1]["cache_config"] == {"rel_l1_thresh": 0.2}
