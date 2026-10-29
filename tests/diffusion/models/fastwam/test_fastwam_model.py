# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastWAM model math against the original implementation, on tiny CPU models.

The ``REFERENCE_*`` values were produced by running the original FastWAM code
(AMD-AGI/faster-action-models ``FastWAM/src`` at a5fa731: FastWAM @7faa711 plus
the CAE action VAE) in float32 on CPU with the deterministic weights of
``tiny.py``; the port matched it bitwise there. Here they are compared with a
tolerance that absorbs CPU-kernel differences between machines.
"""

from __future__ import annotations

import pytest
import torch

from tests.diffusion.models.fastwam import tiny
from vllm_omni.diffusion.compile import _matches_repeated_block
from vllm_omni.diffusion.models.fastwam.modeling.fastwam import build_action_schedule, build_mot_attention_masks
from vllm_omni.diffusion.models.fastwam.modeling.umt5_encoder import WanUMT5Encoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# fmt: off
REFERENCE_ACTIONS_CAE = [0.1357816, 0.1762859, -0.3261685, -0.2935095, -0.02558176, 0.01555358, 0.1079478, 0.3072937, 0.1868789, -0.3172565, -0.2000479, 0.1349833, -0.2210764, 0.3204797, 0.03524231, 0.1912034, -0.2590585, -0.1637773, 0.1598454, -0.101173, 0.09595426, 0.4630314, 0.3569058, -0.4035035, -0.1292705, 0.3817377, -0.2768378, 0.3007706, 0.3054285, 0.3844333, -0.247989, -0.07689642, 0.04748471, -0.1909903, 0.1110959, 0.09098859, 0.27454, -0.1888258, 0.149367, 0.256557, -0.1670644, 0.2549682, 0.2036792, 0.28971, -0.356282, -0.3227215, -0.007567056, -0.04722682, 0.04630401, 0.1815649, 0.2708476, -0.3457063, -0.1538643, 0.270199, -0.08767492, 0.1833039]  # noqa: E501
REFERENCE_ACTIONS_FASTWAM = [-1.142308, -1.640262, -0.03483908, 0.6979551, 1.43875, 0.889914, -0.1464548, -2.332016, -0.138795, -0.64858, 1.204217, 1.29847, 0.1278638, 1.394014, 0.5806944, -0.6427516, -0.541957, -1.409015, 0.9745977, 0.7043942, 0.7342483, -1.393938, -0.6384302, 1.906161, 2.128777, 0.2581047, 0.1709964, 0.2671712, 1.169614, 1.722392, 1.463915, -0.3120363, 0.1452194, 0.08219089, -0.5125172, 0.1821077, 0.3758532, 0.3757374, 1.623587, 1.237905, 0.4192192, -0.4398957, -0.1655857, -0.6272367, 0.2544463, 2.99276, 1.378189, -0.07760128, -1.21335, -0.3039949, -0.5363917, 0.2931868, -1.085739, 0.3687195, 0.3523751, 1.45963]  # noqa: E501
REFERENCE_T5_ROW0 = [0.9016908, 1.004278, 0.7919861, -0.2355499, 0.3081923, -1.269029, 0.5437894, -1.338839, -1.19675, -0.1453693, 0.6659905, 0.5862063, -0.09353648, 0.6634319, 2.212651, 1.412896]  # noqa: E501
REFERENCE_T5_ROW4 = [0.5248817, 0.7666228, -0.4395221, 0.05599654, -1.523001, -1.464152, 0.595072, 1.158628, -0.797891, 0.9345892, 2.12395, -0.07260945, -0.9503231, 0.6664471, -0.05830791, -0.1068166]  # noqa: E501
# fmt: on
ATOL = 1e-4


def _infer(model, cfg, *, seed: int = 0, steps: int = 4, suite_id: int | None = 1):
    inputs = tiny.tiny_inputs(cfg, seed=1)
    return model.infer_action(
        inputs["image"],
        inputs["context"],
        inputs["context_mask"],
        inputs["proprio"],
        num_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        suite_id=suite_id,
    )


@pytest.mark.parametrize(
    ("make_cfg", "suite_id", "reference"),
    [
        (tiny.tiny_cae_config, 1, REFERENCE_ACTIONS_CAE),
        (tiny.tiny_fastwam_config, None, REFERENCE_ACTIONS_FASTWAM),
    ],
    ids=["cae", "fastwam"],
)
def test_infer_action_matches_reference(make_cfg, suite_id, reference):
    cfg = make_cfg()
    actions = _infer(tiny.build_tiny_model(cfg), cfg, suite_id=suite_id)
    assert actions.dtype == torch.float32 and actions.device.type == "cpu"
    assert actions.shape == (cfg.action_horizon, cfg.action_dim)
    torch.testing.assert_close(actions.reshape(-1), torch.tensor(reference), atol=ATOL, rtol=ATOL)


def test_umt5_encoder_matches_reference():
    encoder = WanUMT5Encoder(**tiny.TINY_T5)
    tiny.fill_deterministic(encoder, seed=2)
    ids = torch.tensor([[5, 9, 3, 17, 1, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    with torch.no_grad():
        context = encoder(ids, mask)
    torch.testing.assert_close(context[0, 0], torch.tensor(REFERENCE_T5_ROW0), atol=ATOL, rtol=ATOL)
    torch.testing.assert_close(context[0, 4], torch.tensor(REFERENCE_T5_ROW4), atol=ATOL, rtol=ATOL)


def test_encode_prompt_zeroes_padding_and_unmasks():
    model = tiny.build_tiny_model(tiny.tiny_cae_config())
    ids = torch.tensor([[5, 9, 3, 1, 0, 0, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])
    context, context_mask = model.encode_prompt(ids, mask)
    assert context.shape == (1, 8, tiny.TINY_T5["dim"])
    assert torch.all(context[0, 4:] == 0) and torch.any(context[0, :4] != 0)
    assert context_mask.dtype == torch.bool and bool(context_mask.all())


def test_same_seed_reproduces_and_different_seed_differs():
    cfg = tiny.tiny_cae_config()
    model = tiny.build_tiny_model(cfg)
    first, again, other = _infer(model, cfg, seed=7), _infer(model, cfg, seed=7), _infer(model, cfg, seed=8)
    assert torch.equal(first, again)
    assert not torch.allclose(first, other)


def test_suite_conditioning_changes_the_decoded_chunk():
    cfg = tiny.tiny_cae_config()
    model = tiny.build_tiny_model(cfg)
    assert not torch.allclose(_infer(model, cfg, suite_id=0), _infer(model, cfg, suite_id=1))
    with pytest.raises(ValueError, match="suite id is required"):
        _infer(model, cfg, suite_id=None)


def test_num_steps_changes_the_chunk():
    cfg = tiny.tiny_cae_config()
    model = tiny.build_tiny_model(cfg)
    assert not torch.allclose(_infer(model, cfg, steps=2), _infer(model, cfg, steps=4))


def test_schedule_matches_reference_bfloat16_values():
    # Values from the reference CAE eager trace (10 steps, shift 5, rounded to bfloat16).
    timesteps, deltas = build_action_schedule(10, 5.0, 1000, torch.device("cpu"), torch.bfloat16)
    assert timesteps.dtype == deltas.dtype == torch.bfloat16
    assert timesteps.float().tolist() == [1000, 980, 952, 920, 884, 832, 768, 680, 556, 358]
    expected = [-0.02173, -0.02588, -0.03125, -0.03882, -0.04907, -0.06396, -0.08740, -0.12598, -0.19824, -0.35742]
    torch.testing.assert_close(deltas.float(), torch.tensor(expected), atol=2e-5, rtol=0)
    with pytest.raises(ValueError, match="positive"):
        build_action_schedule(0, 5.0, 1000, torch.device("cpu"), torch.float32)


def test_single_frame_masks_are_all_true():
    video_mask, action_mask = build_mot_attention_masks(6, 2, 6, torch.device("cpu"))
    assert video_mask.shape == (6, 6) and action_mask.shape == (2, 8)
    assert bool(video_mask.all()) and bool(action_mask.all())


def test_infer_action_rejects_bad_image():
    cfg = tiny.tiny_cae_config()
    model = tiny.build_tiny_model(cfg)
    inputs = tiny.tiny_inputs(cfg)
    with pytest.raises(ValueError, match="multiples of 16"):
        model.infer_action(
            inputs["image"][..., :24],
            inputs["context"],
            inputs["context_mask"],
            inputs["proprio"],
            num_steps=1,
            generator=None,
            suite_id=0,
        )


def test_regional_compile_targets_every_expert_layer():
    cfg = tiny.tiny_cae_config()
    model = tiny.build_tiny_model(cfg)
    blocks = [
        name
        for name, module in model.named_modules()
        if _matches_repeated_block(name, module, model._repeated_blocks, [])
    ]
    # Both experts' layers and nothing else (the text encoder and VAE stay eager).
    assert len(blocks) == 2 * cfg.num_layers
    assert all(name.startswith("mot.mixtures.") for name in blocks)
