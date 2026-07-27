# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _make_hidream_sampling(**overrides):
    values = {
        "height": 32,
        "width": 32,
        "num_inference_steps": 2,
        "sigmas": None,
        "max_sequence_length": 128,
        "generator": None,
        "true_cfg_scale": None,
        "guidance_scale": 5.0,
        "guidance_scale_provided": True,
        "num_outputs_per_prompt": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_hidream_pipeline() -> HiDreamImagePipeline:
    pipeline = object.__new__(HiDreamImagePipeline)
    nn.Module.__init__(pipeline)
    pipeline.vae_scale_factor = 8
    pipeline.default_sample_size = 128
    pipeline.transformer = SimpleNamespace(in_channels=4)
    pipeline.vae = SimpleNamespace(
        config=SimpleNamespace(scaling_factor=1.0, shift_factor=0.0),
        decode=lambda latents, return_dict=False: (torch.zeros(latents.shape[0], 3, 8, 8),),
    )
    pipeline._interrupt = False
    return pipeline


@contextmanager
def _noop_progress_bar(*args, **kwargs):
    del args, kwargs

    class _Bar:
        def update(self) -> None:
            return None

    yield _Bar()


def _fake_encode_outputs(batch_size: int = 1):
    prompt_embeds_t5 = torch.zeros(batch_size, 4, 8)
    negative_prompt_embeds_t5 = torch.full((batch_size, 4, 8), 1.0)
    prompt_embeds_llama3 = torch.zeros(2, batch_size, 4, 8)
    negative_prompt_embeds_llama3 = torch.full((2, batch_size, 4, 8), 2.0)
    pooled_prompt_embeds = torch.zeros(batch_size, 16)
    negative_pooled_prompt_embeds = torch.full((batch_size, 16), 3.0)
    return (
        prompt_embeds_t5,
        negative_prompt_embeds_t5,
        prompt_embeds_llama3,
        negative_prompt_embeds_llama3,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


def test_forward_passes_separate_embeds_to_diffuse():
    pipeline = _make_hidream_pipeline()

    class StopAfterDiffuseError(Exception):
        pass

    diffuse_call = {}

    def _fake_encode_prompt(**kwargs):
        del kwargs
        return _fake_encode_outputs(batch_size=1)

    def _fake_diffuse(
        prompt_embeds_t5,
        prompt_embeds_llama3,
        pooled_prompt_embeds,
        negative_prompt_embeds_t5,
        negative_prompt_embeds_llama3,
        negative_pooled_prompt_embeds,
        latents,
        timesteps,
        do_true_cfg,
        true_cfg_scale,
    ):
        diffuse_call.update(
            {
                "prompt_embeds_t5": prompt_embeds_t5,
                "prompt_embeds_llama3": prompt_embeds_llama3,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds_t5": negative_prompt_embeds_t5,
                "negative_prompt_embeds_llama3": negative_prompt_embeds_llama3,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                "do_true_cfg": do_true_cfg,
                "true_cfg_scale": true_cfg_scale,
            }
        )
        raise StopAfterDiffuseError

    pipeline.check_inputs = lambda *args, **kwargs: None
    pipeline.encode_prompt = _fake_encode_prompt
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 4, 2, 2)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.tensor([1.0, 2.0]), 2)
    pipeline.check_cfg_parallel_validity = lambda *args, **kwargs: True
    pipeline.diffuse = _fake_diffuse

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="hidream-prompt-a",
                prompt={"prompt": "prompt-a", "negative_prompt": "negative-a"},
                sampling_params=_make_hidream_sampling(),
            )
        ]
    )

    with pytest.raises(StopAfterDiffuseError):
        pipeline.forward(batch)

    expected = _fake_encode_outputs(batch_size=1)
    torch.testing.assert_close(diffuse_call["prompt_embeds_t5"], expected[0])
    torch.testing.assert_close(diffuse_call["prompt_embeds_llama3"], expected[2])
    torch.testing.assert_close(diffuse_call["pooled_prompt_embeds"], expected[4])
    torch.testing.assert_close(diffuse_call["negative_prompt_embeds_t5"], expected[1])
    torch.testing.assert_close(diffuse_call["negative_prompt_embeds_llama3"], expected[3])
    torch.testing.assert_close(diffuse_call["negative_pooled_prompt_embeds"], expected[5])
    assert diffuse_call["do_true_cfg"] is True
    assert diffuse_call["true_cfg_scale"] == 5.0


def test_predict_noise_forwards_return_dict_once():
    pipeline = _make_hidream_pipeline()
    captured: dict[str, object] = {}

    def _fake_transformer(**kwargs):
        captured["kwargs"] = kwargs
        return (torch.zeros(kwargs["hidden_states"].shape),)

    pipeline.transformer = _fake_transformer  # type: ignore[assignment]

    positive_kwargs = {
        "hidden_states": torch.zeros((1, 4, 2, 2), dtype=torch.float32),
        "timesteps": torch.tensor([1], dtype=torch.int64),
        "encoder_hidden_states_t5": torch.zeros(1, 4, 8),
        "encoder_hidden_states_llama3": torch.zeros(2, 1, 4, 8),
        "pooled_embeds": torch.zeros(1, 16),
        "return_dict": False,
    }

    noise_pred = pipeline.predict_noise(**positive_kwargs)

    assert captured["kwargs"]["return_dict"] is False
    assert noise_pred.shape == positive_kwargs["hidden_states"].shape


def test_diffuse_calls_predict_noise_maybe_with_cfg_per_timestep():
    pipeline = _make_hidream_pipeline()
    pipeline.progress_bar = _noop_progress_bar

    latents = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    timesteps = torch.tensor([7, 3], dtype=torch.int64)
    embeds = _fake_encode_outputs(batch_size=1)

    predict_calls: list[dict[str, object]] = []
    scheduler_calls: list[tuple[bool, float]] = []

    def _fake_predict_noise_maybe_with_cfg(
        do_true_cfg,
        true_cfg_scale,
        positive_kwargs,
        negative_kwargs,
        cfg_normalize=False,
    ):
        predict_calls.append(
            {
                "do_true_cfg": do_true_cfg,
                "true_cfg_scale": true_cfg_scale,
                "positive_kwargs": positive_kwargs,
                "negative_kwargs": negative_kwargs,
                "cfg_normalize": cfg_normalize,
            }
        )
        timestep = positive_kwargs["timesteps"]
        assert isinstance(timestep, torch.Tensor)
        return torch.full_like(latents, float(timestep[0].item()))

    def _fake_scheduler_step_maybe_with_cfg(noise_pred, t, current_latents, do_true_cfg):
        scheduler_calls.append((do_true_cfg, float(t.item())))
        return current_latents + noise_pred

    pipeline.predict_noise_maybe_with_cfg = _fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = _fake_scheduler_step_maybe_with_cfg  # type: ignore[method-assign]

    result = pipeline.diffuse(
        embeds[0],
        embeds[2],
        embeds[4],
        embeds[1],
        embeds[3],
        embeds[5],
        latents,
        timesteps,
        do_true_cfg=True,
        true_cfg_scale=5.0,
    )

    assert len(predict_calls) == 2
    assert predict_calls[0]["do_true_cfg"] is True
    assert predict_calls[0]["true_cfg_scale"] == 5.0
    assert predict_calls[0]["cfg_normalize"] is False
    assert set(predict_calls[0]["positive_kwargs"]) == {
        "hidden_states",
        "timesteps",
        "encoder_hidden_states_t5",
        "encoder_hidden_states_llama3",
        "pooled_embeds",
        "return_dict",
    }
    assert predict_calls[0]["negative_kwargs"] is not None
    assert set(predict_calls[0]["negative_kwargs"]) == set(predict_calls[0]["positive_kwargs"])
    assert scheduler_calls == [(True, 7.0), (True, 3.0)]
    assert torch.equal(result, torch.full_like(latents, 10.0))


def test_guidance_scale_triggers_sequential_cfg_path(monkeypatch: pytest.MonkeyPatch):
    pipeline = _make_hidream_pipeline()
    pipeline.progress_bar = _noop_progress_bar

    _cfg_parallel = "vllm_omni.diffusion.distributed.cfg_parallel"
    monkeypatch.setattr(f"{_cfg_parallel}.get_classifier_free_guidance_world_size", lambda: 1)
    monkeypatch.setattr(f"{_cfg_parallel}.get_classifier_free_guidance_rank", lambda: 0)

    predict_calls: list[dict[str, torch.Tensor | None]] = []

    def _fake_predict_noise(**kwargs):
        predict_calls.append(
            {
                "encoder_hidden_states_t5": kwargs.get("encoder_hidden_states_t5"),
                "encoder_hidden_states_llama3": kwargs.get("encoder_hidden_states_llama3"),
                "pooled_embeds": kwargs.get("pooled_embeds"),
            }
        )
        return torch.zeros_like(kwargs["hidden_states"])

    pipeline.predict_noise = _fake_predict_noise  # type: ignore[method-assign]
    pipeline.scheduler = MagicMock()
    pipeline.scheduler.step = MagicMock(side_effect=lambda noise_pred, t, latents, **kwargs: (latents,))

    latents = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    timesteps = torch.tensor([5], dtype=torch.int64)
    embeds = _fake_encode_outputs(batch_size=1)

    pipeline.diffuse(
        embeds[0],
        embeds[2],
        embeds[4],
        embeds[1],
        embeds[3],
        embeds[5],
        latents,
        timesteps,
        do_true_cfg=True,
        true_cfg_scale=5.0,
    )

    assert len(predict_calls) == 2
    torch.testing.assert_close(predict_calls[0]["encoder_hidden_states_t5"], embeds[0])
    torch.testing.assert_close(predict_calls[0]["encoder_hidden_states_llama3"], embeds[2])
    torch.testing.assert_close(predict_calls[0]["pooled_embeds"], embeds[4])
    torch.testing.assert_close(predict_calls[1]["encoder_hidden_states_t5"], embeds[1])
    torch.testing.assert_close(predict_calls[1]["encoder_hidden_states_llama3"], embeds[3])
    torch.testing.assert_close(predict_calls[1]["pooled_embeds"], embeds[5])


def test_forward_uses_request_guidance_scale_when_true_cfg_scale_unset():
    pipeline = _make_hidream_pipeline()

    class StopAfterDiffuseError(Exception):
        pass

    diffuse_call = {}

    def _fake_diffuse(*args, **kwargs):
        del args
        diffuse_call.update(kwargs)
        raise StopAfterDiffuseError

    pipeline.check_inputs = lambda *args, **kwargs: None
    pipeline.encode_prompt = lambda **kwargs: _fake_encode_outputs(batch_size=1)
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 4, 2, 2)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.tensor([1.0]), 1)
    pipeline.check_cfg_parallel_validity = lambda *args, **kwargs: True
    pipeline.diffuse = _fake_diffuse

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="hidream-guidance-scale",
                prompt={"prompt": "prompt-a", "negative_prompt": "negative-a"},
                sampling_params=_make_hidream_sampling(guidance_scale=4.2, true_cfg_scale=None),
            )
        ]
    )

    with pytest.raises(StopAfterDiffuseError):
        pipeline.forward(batch)

    assert diffuse_call["true_cfg_scale"] == 4.2
    assert diffuse_call["do_true_cfg"] is True


def test_forward_enables_cfg_with_precomputed_negative_embeds():
    pipeline = _make_hidream_pipeline()

    class StopAfterDiffuseError(Exception):
        pass

    diffuse_call = {}

    def _fake_diffuse(*args, **kwargs):
        del args
        diffuse_call.update(kwargs)
        raise StopAfterDiffuseError

    embeds = _fake_encode_outputs(batch_size=1)

    pipeline.check_inputs = lambda *args, **kwargs: None
    pipeline.encode_prompt = lambda **kwargs: embeds
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 4, 2, 2)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.tensor([1.0]), 1)
    pipeline.check_cfg_parallel_validity = lambda *args, **kwargs: True
    pipeline.diffuse = _fake_diffuse

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="hidream-precomputed-neg",
                prompt={"prompt": "prompt-a"},
                sampling_params=_make_hidream_sampling(),
            )
        ]
    )

    with pytest.raises(StopAfterDiffuseError):
        pipeline.forward(
            batch,
            negative_prompt_embeds_t5=embeds[1],
            negative_prompt_embeds_llama3=embeds[3],
            negative_pooled_prompt_embeds=embeds[5],
        )

    assert diffuse_call["do_true_cfg"] is True
    assert diffuse_call["negative_prompt_embeds_t5"] is not None
