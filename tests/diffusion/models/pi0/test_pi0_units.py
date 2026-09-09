# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for the π0 (Pi-Zero) VLA model.

Deliberately lightweight: pure utility code (config plumbing, mask building,
sinusoidal embeddings, normalization, image preprocessing, weight-load remap)
that runs on CPU with no weights and no lerobot. The full LeRobot parity check
lives in ``tests/diffusion/models/pi0/test_pi0_parity.py`` and requires a model download.

    pytest tests/diffusion/models/pi0/test_pi0_units.py -v
"""

from __future__ import annotations

import logging
import math

import pytest
import torch

from vllm_omni.diffusion.models.pi0.config import Pi0Config
from vllm_omni.diffusion.models.pi0.modeling_pi0 import (
    OPENPI_ATTENTION_MASK_VALUE,
    Pi0ForActionPrediction,
    _apply_norm,
    _build_norm_buffers,
    create_sinusoidal_pos_embedding,
    get_gemma_config,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)
from vllm_omni.diffusion.models.pi0.processor_pi0 import (
    Pi0ImageProcessor,
    build_model_inputs,
    pil_image_to_tensor,
    resize_with_pad,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ----------------------------------------------------------------------------
# Pi0Config dataclass + checkpoint resolver
# ----------------------------------------------------------------------------
_LEROBOT_CFG = {
    "type": "pi0",
    "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m",
    "chunk_size": 50,
    "max_action_dim": 32,
    "max_state_dim": 32,
    "num_inference_steps": 10,
    "image_resolution": [224, 224],
    "tokenizer_max_length": 48,
    "dtype": "float32",
    # Training-only keys that must be filtered out of the dataclass.
    "optimizer_lr": 2.5e-5,
    "scheduler_warmup_steps": 1000,
    "input_features": {
        "observation.images.base_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.left_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.images.right_wrist_0_rgb": {"type": "VISUAL", "shape": [3, 224, 224]},
        "observation.state": {"type": "STATE", "shape": [32]},
    },
}

_EXPECTED_CAMERA_ORDER = [
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
]


def test_config_camera_order_from_input_features():
    c = Pi0Config.from_model_config(_LEROBOT_CFG)
    assert c.image_feature_keys == _EXPECTED_CAMERA_ORDER


def test_config_filters_training_only_keys():
    c = Pi0Config.from_model_config(_LEROBOT_CFG)
    assert not hasattr(c, "optimizer_lr")
    assert not hasattr(c, "scheduler_warmup_steps")


def test_config_coerces_image_resolution_to_tuple():
    c = Pi0Config.from_model_config(_LEROBOT_CFG)
    assert c.image_resolution == (224, 224)
    assert isinstance(c.image_resolution, tuple)


def test_config_non_square_resolution_raises():
    with pytest.raises(ValueError):
        Pi0Config(image_resolution=(224, 256))


@pytest.mark.parametrize("bad", [(224,), (224, 224, 3), 224])
def test_config_malformed_resolution_raises(bad):
    with pytest.raises(ValueError):
        Pi0Config(image_resolution=bad)


def test_config_defaults_and_empty():
    c = Pi0Config.from_model_config(None)
    assert c.chunk_size == 50
    assert c.max_action_dim == 32
    assert c.max_state_dim == 32
    assert c.num_inference_steps == 10
    assert c.image_resolution == (224, 224)
    assert c.tokenizer_max_length == 48
    assert c.max_cameras == 3
    assert c.norm_stats is None
    assert c.image_key_map == {}


def test_config_deploy_yaml_override():
    """Deploy-yaml model_config is authoritative (e.g. force dtype/variant)."""
    c = Pi0Config.from_model_config({**_LEROBOT_CFG, "dtype": "bfloat16", "num_inference_steps": 4})
    assert c.dtype == "bfloat16"
    assert c.num_inference_steps == 4


# ----------------------------------------------------------------------------
# Kernel utilities: sinusoidal timestep embedding
# ----------------------------------------------------------------------------
def test_sinusoidal_embedding_shape():
    emb = create_sinusoidal_pos_embedding(torch.tensor([0.0, 0.5, 1.0]), dimension=64)
    assert emb.shape == (3, 64)


def test_sinusoidal_embedding_values_at_zero():
    emb = create_sinusoidal_pos_embedding(torch.tensor([0.0]), dimension=4)
    assert torch.allclose(emb[0, :2].float(), torch.zeros(2), atol=1e-6)
    assert torch.allclose(emb[0, 2:].float(), torch.ones(2), atol=1e-6)


def test_sinusoidal_embedding_odd_dim_raises():
    with pytest.raises(ValueError):
        create_sinusoidal_pos_embedding(torch.tensor([1.0]), dimension=3)


def test_sinusoidal_embedding_non_1d_raises():
    with pytest.raises(ValueError):
        create_sinusoidal_pos_embedding(torch.tensor([[0.0, 0.5]]), dimension=4)


def test_sinusoidal_embedding_distinct_timesteps_differ():
    emb = create_sinusoidal_pos_embedding(torch.tensor([0.1, 0.2, 0.5, 0.9]), dimension=16)
    for i in range(emb.shape[0]):
        for j in range(i + 1, emb.shape[0]):
            assert not torch.allclose(emb[i], emb[j])


# ----------------------------------------------------------------------------
# Kernel utilities: attention mask construction
# ----------------------------------------------------------------------------
def test_make_att_2d_masks_bidirectional():
    result = make_att_2d_masks(torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.long))
    assert result.all()


def test_make_att_2d_masks_causal():
    result = make_att_2d_masks(torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 4, dtype=torch.long))
    expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    assert torch.equal(result[0], expected)


def test_make_att_2d_masks_prefix_lm():
    result = make_att_2d_masks(torch.ones(1, 5, dtype=torch.bool), torch.tensor([[0, 0, 0, 1, 1]], dtype=torch.long))
    assert result[0, 0, 0]
    assert not result[0, 0, 3]
    assert result[0, 3, 0]
    assert not result[0, 3, 4]


def test_make_att_2d_masks_respects_padding():
    pad = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)  # last token is padding
    att = torch.zeros(1, 4, dtype=torch.long)  # full bidirectional
    result = make_att_2d_masks(pad, att)
    assert not result[0, :, 3].any(), "Nothing should attend to padded key"
    assert not result[0, 3, :].any(), "Padded query should attend to nothing"


def test_prepare_attention_masks_4d():
    mask_2d = torch.tensor([[[True, False], [True, True]]])
    mask_4d = prepare_attention_masks_4d(mask_2d)
    assert mask_4d.shape == (1, 1, 2, 2)
    assert mask_4d[0, 0, 0, 0].item() == 0.0
    assert mask_4d[0, 0, 1, 1].item() == 0.0
    expected = torch.tensor(OPENPI_ATTENTION_MASK_VALUE, dtype=mask_4d.dtype)
    assert torch.equal(mask_4d[0, 0, 0, 1], expected)


def test_attention_mask_value_finite_and_negative():
    assert math.isfinite(OPENPI_ATTENTION_MASK_VALUE)
    assert OPENPI_ATTENTION_MASK_VALUE < -1e30


# ----------------------------------------------------------------------------
# Normalization round-trips
# ----------------------------------------------------------------------------
def test_build_norm_buffers_missing():
    assert _build_norm_buffers(None, "state") is None
    assert _build_norm_buffers({}, "state") is None
    assert _build_norm_buffers({"state": None}, "state") is None


def test_build_norm_buffers_mean_std():
    stats = {"state": {"mode": "mean_std", "mean": [1.0, 2.0], "std": [0.5, 0.5]}}
    out = _build_norm_buffers(stats, "state")
    assert out["mode"] == "mean_std"
    assert torch.equal(out["mean"], torch.tensor([1.0, 2.0]))
    assert torch.equal(out["std"], torch.tensor([0.5, 0.5]))


def test_build_norm_buffers_min_max():
    stats = {"action": {"mode": "min_max", "min": [-1.0, 0.0], "max": [1.0, 2.0]}}
    out = _build_norm_buffers(stats, "action")
    assert out["mode"] == "min_max"
    assert torch.equal(out["min"], torch.tensor([-1.0, 0.0]))
    assert torch.equal(out["max"], torch.tensor([1.0, 2.0]))


def test_apply_norm_mean_std_forward_inverse():
    stats = _build_norm_buffers({"s": {"mode": "mean_std", "mean": [0.5, -0.2], "std": [2.0, 0.1]}}, "s")
    x = torch.tensor([[1.0, 0.3]])
    z = _apply_norm(x, stats, inverse=False)
    assert torch.allclose(z, torch.tensor([[0.25, 5.0]]), atol=1e-5)
    x_back = _apply_norm(z, stats, inverse=True)
    assert torch.allclose(x_back, x, atol=1e-5)


def test_apply_norm_min_max_forward_inverse():
    stats = _build_norm_buffers({"s": {"mode": "min_max", "min": [0.0, -1.0], "max": [1.0, 1.0]}}, "s")
    x = torch.tensor([[0.25, 0.5]])
    z = _apply_norm(x, stats, inverse=False)
    assert torch.allclose(z, torch.tensor([[-0.5, 0.5]]), atol=1e-5)
    x_back = _apply_norm(z, stats, inverse=True)
    assert torch.allclose(x_back, x, atol=1e-5)


def test_apply_norm_preserves_padded_tail():
    stats = _build_norm_buffers({"s": {"mode": "mean_std", "mean": [0.0, 0.0], "std": [1.0, 1.0]}}, "s")
    x = torch.tensor([[1.0, 2.0, 99.0, -99.0]])
    z = _apply_norm(x, stats, inverse=False)
    assert torch.allclose(z[:, :2], x[:, :2])
    assert torch.allclose(z[:, 2:], x[:, 2:])


def test_apply_norm_noop_when_stats_none():
    x = torch.randn(2, 5)
    assert torch.equal(_apply_norm(x, None, inverse=False), x)
    assert torch.equal(_apply_norm(x, None, inverse=True), x)


# ----------------------------------------------------------------------------
# Gemma variant configs
# ----------------------------------------------------------------------------
def test_gemma_2b_dims():
    c = get_gemma_config("gemma_2b")
    assert c.width == 2048
    assert c.depth == 18


def test_gemma_300m_dims():
    c = get_gemma_config("gemma_300m")
    assert c.width == 1024
    assert c.depth == 18


def test_gemma_unknown_variant_raises():
    with pytest.raises(ValueError):
        get_gemma_config("gemma_7b")


# ----------------------------------------------------------------------------
# load_weights remap rules (tiny model; needs torch + transformers)
# ----------------------------------------------------------------------------
def _tiny_pi0_model():
    """Construct a tiny π0 model for remap tests (full Gemma depth but small
    action/state dims — the PaliGemma backbone is always full-size)."""
    cfg = Pi0Config(max_action_dim=8, max_state_dim=8, chunk_size=2)
    return Pi0ForActionPrediction(cfg)


@pytest.fixture(autouse=True)
def _silence_pi0_loader(caplog):
    # The remap tests feed a single synthetic weight; the loader's
    # "missing params" warning is expected noise here.
    logging.getLogger("vllm_omni.diffusion.models.pi0.modeling_pi0").setLevel(logging.ERROR)
    yield


@pytest.mark.slow
def test_remap_paligemma_submodules_to_nested_layout():
    model = _tiny_pi0_model()
    params = dict(model.named_parameters())
    target = next(k for k in params if k.startswith("paligemma_with_expert.paligemma.model.vision_tower."))
    flat_key = "model." + target.replace(
        "paligemma_with_expert.paligemma.model.vision_tower.",
        "paligemma_with_expert.paligemma.vision_tower.",
    )
    payload = torch.full(params[target].shape, 0.12345)
    before = params[target].detach().clone()
    model.load_weights([(flat_key, payload)])
    after = dict(model.named_parameters())[target]
    assert not torch.equal(before, after), "flat→nested remap broken"
    assert torch.allclose(after, payload)


@pytest.mark.slow
def test_lm_head_remap_to_embed_tokens():
    model = _tiny_pi0_model()
    params = dict(model.named_parameters())
    target = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    assert target in params
    payload = torch.full(params[target].shape, 0.5)
    before = params[target].detach().clone()
    model.load_weights([("model.paligemma_with_expert.paligemma.lm_head.weight", payload)])
    after = dict(model.named_parameters())[target]
    assert not torch.equal(before, after), "lm_head→embed_tokens remap broken"
    assert torch.allclose(after, payload)


@pytest.mark.slow
def test_vision_tower_version_robust_remap():
    """transformers >=5.4 flattens SigLIP from vision_tower.vision_model.* to
    vision_tower.*; the lerobot/pi0_base checkpoint uses the vision_model infix.
    load_weights must adapt so vision weights aren't silently dropped (which
    would leave the vision encoder at random init)."""
    model = _tiny_pi0_model()
    params = dict(model.named_parameters())
    # A representative vision-tower param as it exists on the model.
    target = next(
        k
        for k in params
        if k.startswith("paligemma_with_expert.paligemma.model.vision_tower.") and "patch_embedding.weight" in k
    )
    vt = "paligemma_with_expert.paligemma.model.vision_tower."
    rest = target[len(vt) :]
    # Build the checkpoint key in whichever layout differs from the model's.
    if rest.startswith("vision_model."):
        ckpt_key = vt + rest[len("vision_model.") :]
    else:
        ckpt_key = vt + "vision_model." + rest
    payload = torch.full(params[target].shape, 0.073)
    before = params[target].detach().clone()
    model.load_weights([("model." + ckpt_key, payload)])
    after = dict(model.named_parameters())[target]
    assert not torch.equal(before, after), "vision_tower version-robust remap broken"
    assert torch.allclose(after, payload)


# ----------------------------------------------------------------------------
# Image processor
# ----------------------------------------------------------------------------
def test_resize_with_pad_square():
    out = resize_with_pad(torch.ones(1, 3, 224, 224), 224, 224)
    assert out.shape == (1, 3, 224, 224)


def test_resize_with_pad_wide():
    """Wide source (H < W): padding applied to top/bottom (rows)."""
    out = resize_with_pad(torch.zeros(1, 3, 100, 200), 224, 224)
    assert out.shape == (1, 3, 224, 224)
    assert out[0, 0, 0, 112].item() == -1.0  # top row is padded


def test_resize_with_pad_tall():
    """Tall source (H > W): padding applied to left/right (cols)."""
    out = resize_with_pad(torch.zeros(1, 3, 400, 100), 224, 224)
    assert out.shape == (1, 3, 224, 224)
    assert out[0, 0, 112, 0].item() == -1.0  # left-most column padded


def test_resize_with_pad_rejects_non_4d():
    with pytest.raises(ValueError):
        resize_with_pad(torch.zeros(3, 100, 200), 224, 224)


def test_pil_to_tensor():
    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.full((10, 10, 3), 128, dtype=np.uint8))
    t = pil_image_to_tensor(img)
    assert t.shape == (1, 3, 10, 10)
    expected = 128.0 / 255.0 * 2.0 - 1.0
    assert abs(t[0, 0, 5, 5].item() - expected) < 1e-4


def test_pil_to_tensor_rgba_to_rgb():
    import numpy as np
    from PIL import Image

    rgba = np.full((8, 8, 4), 200, dtype=np.uint8)
    img = Image.fromarray(rgba, mode="RGBA")
    t = pil_image_to_tensor(img)
    assert t.shape == (1, 3, 8, 8)


def test_make_empty_image():
    proc = Pi0ImageProcessor(image_size=224)
    empty = proc.make_empty_image()
    assert empty.shape == (1, 3, 224, 224)
    assert (empty == -1.0).all()


def test_preprocess_single_resizes_ndarray():
    import numpy as np

    proc = Pi0ImageProcessor(image_size=224)
    arr = np.full((120, 120, 3), 100, dtype=np.uint8)
    t = proc.preprocess_single(arr)
    assert t.shape == (1, 3, 224, 224)


# ----------------------------------------------------------------------------
# build_model_inputs: camera ordering, padding, masks, state
# ----------------------------------------------------------------------------
class _FakeTokenizer:
    """Minimal stand-in for the PaliGemma tokenizer (no download)."""

    def __call__(
        self, text, padding=None, max_length=None, truncation=None, add_special_tokens=None, return_tensors=None
    ):
        ids = [2] + [ord(c) % 100 for c in text][: max_length - 1]
        ids = ids[:max_length]
        attn = [1] * len(ids)
        while len(ids) < max_length:
            ids.append(0)
            attn.append(0)
        return {"input_ids": ids, "attention_mask": attn}


def test_build_model_inputs_camera_order_and_padding():
    import numpy as np

    config = Pi0Config.from_model_config(_LEROBOT_CFG)  # 3-camera config
    device = torch.device("cpu")
    # Provide only 2 of the 3 declared cameras.
    robot_obs = {
        "images": {
            "observation.images.base_0_rgb": np.full((224, 224, 3), 50, dtype=np.uint8),
            "observation.images.left_wrist_0_rgb": np.full((224, 224, 3), 80, dtype=np.uint8),
        },
        "state": [0.1] * 14,  # raw state shorter than max_state_dim=32
        "prompt": "pick up the block",
    }
    images, image_masks, lang_tokens, lang_masks, state = build_model_inputs(
        robot_obs, config, _FakeTokenizer(), device
    )
    assert len(images) == 3
    assert all(img.shape == (1, 3, 224, 224) for img in images)
    masks = [bool(m.item()) for m in image_masks]
    assert masks == [True, True, False]
    # The padded (3rd) camera must be all -1.
    assert (images[2] == -1.0).all()
    assert lang_tokens.shape == (1, 48)
    assert lang_masks.shape == (1, 48)
    assert state.shape == (1, 32)
    # State padded with zeros beyond the 14 real entries.
    assert torch.allclose(state[0, 14:], torch.zeros(18))


def test_build_model_inputs_state_truncated():
    config = Pi0Config.from_model_config(_LEROBOT_CFG)
    robot_obs = {"images": {}, "state": [0.5] * 40, "prompt": "x"}
    _, _, _, _, state = build_model_inputs(robot_obs, config, _FakeTokenizer(), torch.device("cpu"))
    assert state.shape == (1, 32)


def test_build_model_inputs_image_key_map():
    import numpy as np

    cfg = Pi0Config.from_model_config({**_LEROBOT_CFG, "image_key_map": {"cam_high": "observation.images.base_0_rgb"}})
    robot_obs = {
        "images": {"cam_high": np.full((224, 224, 3), 60, dtype=np.uint8)},
        "prompt": "go",
    }
    _, image_masks, _, _, _ = build_model_inputs(robot_obs, cfg, _FakeTokenizer(), torch.device("cpu"))
    masks = [bool(m.item()) for m in image_masks]
    assert masks == [True, False, False]  # only base_0_rgb (mapped) is present


# ----------------------------------------------------------------------------
# Version-stability / self-consistency (gated on a real pi0_base checkpoint)
#
# The LeRobot parity gate (test_pi0_parity.py) only validates correctness at
# transformers 5.3.0 (where LeRobot loads pi0_base cleanly). Production vllm-omni
# ships transformers 5.6.2, and LeRobot is not a usable reference there (it
# incorrectly loads the SigLIP vision tower at >=5.4). So we pin the kernel's deterministic
# output to golden values established at 5.3.0 and re-check them here: running on
# the production transformers proves the port is version-stable and guards the
# vision_tower / embed_scale version-robust paths against regressions.
#
#   PI0_SELFCONSIST_CKPT=/path/to/pi0_base \
#       pytest tests/diffusion/models/pi0/test_pi0_units.py -k version_stable -v -s
# ----------------------------------------------------------------------------
import os  # noqa: E402

_SELFCONSIST_CKPT = os.environ.get("PI0_SELFCONSIST_CKPT")

# Golden values established at transformers 5.3.0 (where parity = 7.15e-07).
_GOLDEN_SUM = -67.808113
_GOLDEN_STD = 0.252789
_GOLDEN_MEAN = -0.021190
_GOLDEN_EMBED_IMAGE_STD = 2.17603


def _selfconsist_inputs(device):
    # NB: draw order is load-bearing — the golden values were established with
    # state drawn first, then the 3 images, then lang (shared RNG, seed 0).
    g = torch.Generator(device="cpu").manual_seed(0)
    b = 2
    state = torch.randn(b, 32, generator=g)
    images = [torch.rand(b, 3, 224, 224, generator=g) for _ in range(3)]
    masks = [torch.ones(b, dtype=torch.bool) for _ in range(3)]
    lang = torch.randint(0, 1000, (b, 48), generator=g)
    lmask = torch.ones(b, 48, dtype=torch.bool)
    ng = torch.Generator(device="cpu").manual_seed(42)
    noise = torch.randn(b, 50, 32, generator=ng)
    return (
        [im.to(device) for im in images],
        [m.to(device) for m in masks],
        lang.to(device),
        lmask.to(device),
        state.to(device),
        noise.to(device),
    )


def _selfconsist_model(device):
    import safetensors.torch

    model = Pi0ForActionPrediction(
        Pi0Config(max_action_dim=32, max_state_dim=32, chunk_size=50, num_inference_steps=10, dtype="float32")
    )
    state = safetensors.torch.load_file(os.path.join(_SELFCONSIST_CKPT, "model.safetensors"))
    filled = model.load_weights(list(state.items()))
    expected = {n for n, _ in model.named_parameters() if "rotary_emb" not in n and not n.endswith(".inv_freq")}
    missing = expected - (filled or set())
    assert not missing, f"{len(missing)} params received no weight (e.g. {sorted(missing)[:3]})"
    return model.to(device).eval()


@pytest.mark.slow
@pytest.mark.skipif(not _SELFCONSIST_CKPT, reason="Set PI0_SELFCONSIST_CKPT to a pi0_base dir.")
@torch.no_grad()
def test_version_stable_actions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _selfconsist_model(device)
    images, masks, lang, lmask, state, noise = _selfconsist_inputs(device)
    actions = (
        model.sample_actions(
            images=images,
            image_masks=masks,
            lang_tokens=lang,
            lang_masks=lmask,
            state=state,
            noise=noise,
            num_steps=10,
        )
        .float()
        .cpu()
    )
    s, m, std = actions.sum().item(), actions.mean().item(), actions.std().item()
    assert abs(s - _GOLDEN_SUM) < 0.5, f"action sum drifted: {s} vs {_GOLDEN_SUM}"
    assert abs(std - _GOLDEN_STD) < 0.01, f"action std drifted: {std} vs {_GOLDEN_STD}"
    assert abs(m - _GOLDEN_MEAN) < 0.01, f"action mean drifted: {m} vs {_GOLDEN_MEAN}"


@pytest.mark.slow
@pytest.mark.skipif(not _SELFCONSIST_CKPT, reason="Set PI0_SELFCONSIST_CKPT to a pi0_base dir.")
@torch.no_grad()
def test_version_stable_embed_image():
    """Directly guards the SigLIP vision-tower load across the >=5.4 flattening."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _selfconsist_model(device)
    g = torch.Generator(device="cpu").manual_seed(7)
    img = torch.rand(1, 3, 224, 224, generator=g, dtype=torch.float32).to(device)
    std = model.paligemma_with_expert.embed_image(img).float().std().item()
    assert abs(std - _GOLDEN_EMBED_IMAGE_STD) < 0.05, (
        f"embed_image std drifted: {std} vs {_GOLDEN_EMBED_IMAGE_STD} — "
        "likely a vision_tower weight-load regression under this transformers version."
    )
