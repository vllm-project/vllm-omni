# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cache branch and scheduler metadata contracts shared by image pipelines."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed import cfg_parallel
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Pipeline(CFGParallelMixin):
    def __init__(self):
        self.context = None
        self.contexts = []
        self.calls = []
        self._cache_context_factory = self.cache_context

    @contextmanager
    def cache_context(self, name):
        previous = self.context
        self.context = name
        self.contexts.append(name)
        try:
            yield
        finally:
            self.context = previous

    def predict_noise(self, *, value, fail=False):
        self.calls.append((self.context, value))
        if fail:
            raise RuntimeError("transformer failed")
        return torch.full((1, 2, 3), value)


@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1)])
def test_cache_context_selects_executed_branch(monkeypatch, world_size, rank):
    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: world_size)
    monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_rank", lambda: rank)
    monkeypatch.setattr(
        cfg_parallel,
        "get_cfg_group",
        lambda: SimpleNamespace(all_gather=lambda prediction, **_: [prediction, prediction]),
    )
    pipeline = _Pipeline()

    negative_kwargs = {"value": 1.0} if world_size > 1 else None
    pipeline.predict_noise_maybe_with_cfg(world_size > 1, 3.0, {"value": 2.0}, negative_kwargs, False)

    assert pipeline.contexts == [["cond"], ["uncond"]][rank]
    assert pipeline.calls == [(["cond", "uncond"][rank], [2.0, 1.0][rank])]
    assert pipeline.context is None


@pytest.mark.parametrize(
    "world_size,rank,do_true_cfg,expected",
    [
        (1, 0, False, [("cond", "positive", False)]),
        (1, 0, True, [("cond", "positive", True), ("uncond", "negative", True)]),
        (2, 0, True, [("cond", "positive", True)]),
        (2, 1, True, [("uncond", "negative", True)]),
    ],
)
def test_cache_context_and_transformer_branch_match(monkeypatch, world_size, rank, do_true_cfg, expected):
    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: world_size)
    monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_rank", lambda: rank)
    monkeypatch.setattr(
        cfg_parallel,
        "get_cfg_group",
        lambda: SimpleNamespace(all_gather=lambda prediction, **_: [prediction, prediction]),
    )
    pipeline = _Pipeline()
    pipeline.transformer = SimpleNamespace(cfg_branch="outer")
    observations = []

    def predict_noise(*, value):
        observations.append((pipeline.context, pipeline.transformer.cfg_branch, pipeline.transformer.do_true_cfg))
        return torch.full((1, 2, 3), value)

    monkeypatch.setattr(pipeline, "predict_noise", predict_noise)
    pipeline.predict_noise_maybe_with_cfg(do_true_cfg, 3.0, {"value": 2.0}, {"value": 1.0}, False)

    assert observations == expected
    assert pipeline.context is None
    assert pipeline.transformer.cfg_branch == "outer"


def test_cache_context_and_transformer_branch_restore_on_failure(monkeypatch):
    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: 1)
    pipeline = _Pipeline()
    pipeline.context = "outer-cache"
    pipeline.transformer = SimpleNamespace(cfg_branch="outer")
    observations = []

    def predict_noise(*, value):
        observations.append((pipeline.context, pipeline.transformer.cfg_branch, pipeline.transformer.do_true_cfg))
        raise RuntimeError("transformer failed")

    monkeypatch.setattr(pipeline, "predict_noise", predict_noise)
    with pytest.raises(RuntimeError, match="transformer failed"):
        pipeline.predict_noise_maybe_with_cfg(True, 3.0, {"value": 2.0}, {"value": 1.0}, False)

    assert observations == [("cond", "positive", True)]
    assert pipeline.context == "outer-cache"
    assert pipeline.transformer.cfg_branch == "outer"


@pytest.mark.parametrize("world_size", [1, 2])
def test_missing_negative_arguments_fail_before_entering_cache_context(monkeypatch, world_size):
    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: world_size)
    pipeline = _Pipeline()

    with pytest.raises(ValueError, match="CFG requires negative prediction arguments"):
        pipeline.predict_noise_maybe_with_cfg(True, 3.0, {"value": 2.0}, None, False)

    assert pipeline.contexts == []
    assert pipeline.calls == []


def test_pipeline_owned_explicit_context_is_preserved():
    class ExplicitContextPipeline(_Pipeline):
        def predict_noise(self, *, value, fail=False, _cache_context="cond"):
            # Cosmos3 handles this key itself, including cond_no_control.
            assert self.transformer.cfg_branch == "positive"
            with self._cache_context_factory(_cache_context):
                return super().predict_noise(value=value, fail=fail)

    pipeline = ExplicitContextPipeline()
    pipeline.transformer = SimpleNamespace(cfg_branch="outer")
    pipeline.predict_noise_maybe_with_cfg(False, 1.0, {"_cache_context": "cond_no_control", "value": 2.0}, None)

    assert pipeline.contexts == ["cond_no_control"]
    assert pipeline.calls == [("cond_no_control", 2.0)]
    assert pipeline.context is None
    assert pipeline.transformer.cfg_branch == "outer"


def test_without_cache_factory_no_metadata_or_new_model_arguments_are_required():
    pipeline = _Pipeline()
    del pipeline._cache_context_factory

    # No scheduler is needed by the uncached path.
    with pipeline._cache_step_metadata(0, 5):
        pipeline.predict_noise_maybe_with_cfg(False, 1.0, {"value": 2.0}, None)

    assert pipeline.calls == [(None, 2.0)]
    assert pipeline.current_step_index is None
    assert pipeline.current_sigma is None


def test_scheduler_metadata_uses_resolved_sigma_and_cleans_up_on_failure(monkeypatch):
    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: 1)
    pipeline = _Pipeline()
    pipeline.scheduler = SimpleNamespace(
        sigmas=torch.tensor([1.0, 0.91, 0.63, 0.37, 0.17, 0.0]),
        begin_index=2,
        step_index=None,
    )

    with pipeline._cache_step_metadata(0, 2):
        assert pipeline.current_step_index == 0
        assert pipeline.current_sigma.item() == pytest.approx(0.63)
        assert pipeline._num_timesteps == 2

    # The live cursor deliberately differs from begin_index + loop index.
    pipeline.scheduler.step_index = 4
    with pytest.raises(RuntimeError, match="transformer failed"), pipeline._cache_step_metadata(1, 2):
        assert pipeline.current_step_index == 1
        assert pipeline.current_sigma.item() == pytest.approx(0.17)
        pipeline.predict_noise_maybe_with_cfg(True, 3.0, {"value": 2.0}, {"value": 1.0, "fail": True})

    assert pipeline.contexts == ["cond", "uncond"]
    assert pipeline.context is None
    assert pipeline.current_step_index is None
    assert pipeline.current_sigma is None


@pytest.mark.parametrize("family", ["flux", "qwen"])
def test_image_diffuse_publishes_scheduler_sigmas_and_actual_steps(monkeypatch, family):
    from vllm_omni.diffusion.models.flux.pipeline_flux import FluxPipeline
    from vllm_omni.diffusion.models.qwen_image.cfg_parallel import QwenImageCFGParallelMixin

    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: 1)
    pipeline = _Pipeline()
    pipeline.interrupt = False
    pipeline.joint_attention_kwargs = None
    pipeline.transformer = SimpleNamespace()
    pipeline.scheduler = SimpleNamespace(sigmas=torch.tensor([0.91, 0.17]), step_index=None)
    pipeline.scheduler.set_begin_index = lambda index: setattr(pipeline.scheduler, "begin_index", index)
    pipeline.progress_bar = lambda **_: nullcontext(SimpleNamespace(update=lambda: None))
    pipeline.scheduler_step_maybe_with_cfg = lambda noise_pred, timestep, latents, do_cfg: latents
    observations = []

    def predict_noise(**kwargs):
        observations.append(
            (pipeline.context, pipeline.current_step_index, float(pipeline.current_sigma), pipeline._num_timesteps)
        )
        return torch.ones_like(kwargs["hidden_states"])

    monkeypatch.setattr(pipeline, "predict_noise", predict_noise)
    latents = torch.zeros(1, 4, 8)
    embeddings = torch.zeros(1, 2, 8)
    shared = {
        "prompt_embeds": embeddings,
        "negative_prompt_embeds": embeddings,
        "latents": latents,
        # These deliberately do not encode the scheduler's sigmas.
        "timesteps": torch.tensor([900.0, 100.0]),
        "do_true_cfg": True,
        "guidance": None,
        "true_cfg_scale": 3.0,
        "cfg_normalize": False,
    }
    if family == "flux":
        FluxPipeline.diffuse(
            pipeline,
            pooled_prompt_embeds=torch.zeros(1, 8),
            negative_pooled_prompt_embeds=torch.zeros(1, 8),
            latent_image_ids=torch.zeros(4, 3),
            text_ids=torch.zeros(2, 3),
            negative_text_ids=torch.zeros(2, 3),
            **shared,
        )
    else:
        QwenImageCFGParallelMixin.diffuse(
            pipeline,
            prompt_embeds_mask=torch.ones(1, 2),
            negative_prompt_embeds_mask=torch.ones(1, 2),
            img_shapes=[[(1, 2, 2)]],
            txt_seq_lens=[2],
            negative_txt_seq_lens=[2],
            **shared,
        )

    assert observations == [
        ("cond", 0, pytest.approx(0.91), 2),
        ("uncond", 0, pytest.approx(0.91), 2),
        ("cond", 1, pytest.approx(0.17), 2),
        ("uncond", 1, pytest.approx(0.17), 2),
    ]
    assert pipeline.current_step_index is None
    assert pipeline.current_sigma is None
