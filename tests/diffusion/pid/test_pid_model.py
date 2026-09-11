# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-function tests for PidInferenceModel (t schedule + velocity->x0 math).

``PidInferenceModel.__init__`` constructs a real ``PidNet`` + downloads the
Gemma encoder, so these tests bypass ``__init__`` via ``object.__new__``.
"""

import pytest
import torch

from vllm_omni.diffusion.pid.config import PID_SAMPLING_CONFIG
from vllm_omni.diffusion.pid.pid_model import PidInferenceModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _bare_model():
    """Bypass __init__ (no PidNet / Gemma), expose only the sampling config."""
    model = object.__new__(PidInferenceModel)
    model._cfg = type("Cfg", (), dict(PID_SAMPLING_CONFIG))()
    return model


def test_t_list_default_matches_config():
    model = _bare_model()
    t_list = model._get_t_list(torch.device("cpu"), num_steps=4)
    assert t_list.shape == (5,)
    assert torch.isclose(t_list[0], torch.tensor(0.999))


def test_t_list_resampled_for_other_steps():
    model = _bare_model()
    t_list = model._get_t_list(torch.device("cpu"), num_steps=2)
    assert t_list.shape == (3,)


def test_velocity_to_x0_math():
    model = _bare_model()
    x_t = torch.tensor([1.0, 2.0, 3.0])
    v = torch.tensor([0.5, 0.5, 0.5])
    t = torch.tensor([0.5, 0.5, 0.5])
    x0 = model._velocity_to_x0(x_t, v, t)  # x0 = x_t - t * v
    torch.testing.assert_close(x0, torch.tensor([0.75, 1.75, 2.75]))


# -- caption broadcast for num_outputs_per_prompt > 1 -------------------------


class _FakeTextEncoder:
    def encode(self, captions):
        return torch.randn(len(captions), 8, 16)


def _model_for_generate():
    """Bypass __init__; stub encoder/net so generate_samples_from_batch runs."""
    model = _bare_model()
    model.tensor_kwargs = {"dtype": torch.float32}
    model.text_encoder = _FakeTextEncoder()
    model.net = type("FakeNetAttrs", (), {"txt_max_length": 300})()

    def fake_net(noise, t, emb, *, lq_latent, degrade_sigma):
        assert noise.shape[0] == lq_latent.shape[0], "noise/latent batch mismatch"
        return torch.zeros_like(noise)

    model._maybe_compile_net = lambda *a, **k: fake_net
    return model


def test_generate_broadcasts_single_caption_to_latent_batch():
    """n>=2 per request: latents arrive as [n, ...] with one caption string."""
    model = _model_for_generate()
    lq_latent = torch.randn(2, 32, 8, 8)
    out = model.generate_samples_from_batch(lq_latent, caption="a cat", output_size=(16, 16), num_steps=1)
    assert out.shape == (2, 3, 16, 16)


def test_generate_caption_list_must_match_latent_batch():
    """Explicit caption lists must match the latent batch size (fail loud)."""
    model = _model_for_generate()
    with pytest.raises(ValueError, match="caption count"):
        model.generate_samples_from_batch(
            torch.randn(2, 32, 8, 8),
            caption=["a", "b", "c"],
            output_size=(16, 16),
            num_steps=1,
        )
