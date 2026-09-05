# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GR00T-N1.7 eval image transform: chosen from ``processor_config.json`` and bit-exact with Isaac-GR00T.

``GOLDEN`` holds sha256 digests of the ``(C, H, W)`` uint8 output of Isaac-GR00T's own eval transform
(``build_image_transformations_albumentations(...)[1]`` applied through ``apply_with_replay``) for the
synthetic frames built below, at Isaac-GR00T 51d4c89 / albumentations 1.4.18 / opencv 5.0.0.93. The frame
generators are integer-only so they are identical on every platform; keep them byte-identical to the
regeneration script attached to the PR, which also cross-checks this module against Isaac-GR00T.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as transforms

from vllm_omni.diffusion.models.gr00t.modeling import processing_gr00t_n1d7 as processing
from vllm_omni.diffusion.models.gr00t.modeling.processing_gr00t_n1d7 import (
    AlbumentationsEvalImageTransform,
    Gr00tN1d7Processor,
    LetterBoxTransform,
    _build_eval_image_transform,
    _fractional_center_crop,
    _smallest_edge_resize,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# nvidia/GR00T-N1.7-3B and nvidia/GR00T-N1.7-DROID both ship exactly these processor_kwargs.
TARGET, CROP, EDGE, FRACTION = [256, 256], [230, 230], 256, 0.95
MODALITY_CONFIGS = {
    "new_embodiment": {
        "video": {"delta_indices": [0], "modality_keys": ["cam"]},
        "state": {"delta_indices": [0], "modality_keys": ["joint"]},
        "action": {"delta_indices": [0, 1], "modality_keys": ["joint"]},
    }
}

# Isaac-GR00T eval transform output for synthetic_frames(); key = "<frame>|letterbox=<flag>".
GOLDEN: dict[str, tuple[tuple[int, int, int], str]] = {
    "gradient_256x256|letterbox=False": (
        (3, 256, 256),
        "4e90a1a3c49539b3648aad752bba59af452bd7231e5712e871ab941cbea1d715",
    ),
    "checker_256x256|letterbox=False": (
        (3, 256, 256),
        "d2225b94efea5ab3f1c3e6dbae52b7afa3e0f7172d5de24439a2e729ab56080c",
    ),
    "noise_256x256|letterbox=False": (
        (3, 256, 256),
        "a427ae3b84ebcc6e87470c79c0e45cf684770d6e31158a49f8736685e0da72c6",
    ),
    "noise_512x768|letterbox=False": (
        (3, 256, 383),
        "45bc3a309f3d06c3f31051911bc8a544ff3f067b023b999dd9c2b7e6029ecc93",
    ),
    "gradient_256x256|letterbox=True": (
        (3, 256, 256),
        "4e90a1a3c49539b3648aad752bba59af452bd7231e5712e871ab941cbea1d715",
    ),
    "checker_256x256|letterbox=True": (
        (3, 256, 256),
        "d2225b94efea5ab3f1c3e6dbae52b7afa3e0f7172d5de24439a2e729ab56080c",
    ),
    "noise_256x256|letterbox=True": ((3, 256, 256), "a427ae3b84ebcc6e87470c79c0e45cf684770d6e31158a49f8736685e0da72c6"),
    "noise_512x768|letterbox=True": ((3, 256, 256), "52a3db185e6ee4f00617b87711dd198b8f2e2ad53865713d34e56225a4ec61a2"),
}


def _hash_noise(height: int, width: int, seed: int) -> np.ndarray:
    """Integer-only pseudo-noise (murmur-style mixing on uint32); no RNG or float, so identical everywhere."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.uint32)
    channels = []
    for channel in range(3):
        v = xs * np.uint32(73856093) ^ ys * np.uint32(19349663) ^ np.uint32((seed * 3 + channel) * 83492791)
        v ^= v >> np.uint32(13)
        v *= np.uint32(0x5BD1E995)
        v ^= v >> np.uint32(15)
        channels.append((v & np.uint32(0xFF)).astype(np.uint8))
    return np.stack(channels, axis=-1)


def synthetic_frames() -> dict[str, np.ndarray]:
    """Deterministic HWC uint8 RGB frames: smooth, hard-edged, noisy, and a non-square noisy one.

    The non-square frame is 512x768 so both INTER_AREA downscales (2x without letterbox, 3x with it)
    are integer factors, which OpenCV computes with exact integer sums: the goldens do not depend on
    the CPU architecture. The 243->256 upscale is a fixed-point bilinear on uint8, also exact.
    """
    ys, xs = np.mgrid[0:256, 0:256].astype(np.int64)
    frames = {
        "gradient_256x256": np.stack([xs % 256, ys % 256, (xs + ys) // 2 % 256], axis=-1).astype(np.uint8),
        "checker_256x256": np.stack(
            [((ys // 16 + xs // 16) % 2) * 255, ((ys // 8 + xs // 8) % 2) * 255, ((ys // 32) % 2) * 255],
            axis=-1,
        ).astype(np.uint8),
        "noise_256x256": _hash_noise(256, 256, seed=1),
        "noise_512x768": _hash_noise(512, 768, seed=2),
    }
    return frames


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(np.ascontiguousarray(tensor.numpy()).tobytes()).hexdigest()


class _FakeTokenizer:
    padding_side = "right"


class _FakeVlmProcessor:
    """The two attributes ``Gr00tN1d7Processor`` and its collator touch on the Qwen3-VL processor."""

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, *_args: object, **_kwargs: object) -> str:
        return "<prompt>"


@pytest.fixture(autouse=True)
def fake_vlm_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real Qwen3-VL processor is a gated download and irrelevant to the image pipeline."""
    monkeypatch.setattr(processing, "build_processor", lambda _model_name, _kwargs: _FakeVlmProcessor())


def _processor(**kwargs) -> Gr00tN1d7Processor:
    return Gr00tN1d7Processor(
        modality_configs=MODALITY_CONFIGS, image_crop_size=CROP, image_target_size=TARGET, **kwargs
    )


def _has_letterbox(compose: transforms.Compose) -> bool:
    return any(isinstance(step, LetterBoxTransform) for step in compose.transforms)


def test_processor_builds_the_pipeline_named_by_processor_config():
    albumentations = _processor(use_albumentations=True, shortest_image_edge=EDGE, crop_fraction=FRACTION)
    transform = albumentations.eval_image_transform
    assert isinstance(transform, AlbumentationsEvalImageTransform)
    assert (transform.max_size, transform.fraction, transform.letter_box_transform) == (EDGE, FRACTION, False)
    assert albumentations.use_albumentations is True

    torchvision = _processor(use_albumentations=False)
    assert isinstance(torchvision.eval_image_transform, transforms.Compose)
    assert not _has_letterbox(torchvision.eval_image_transform)

    letterboxed = _processor(use_albumentations=False, letter_box_transform=True)
    assert _has_letterbox(letterboxed.eval_image_transform)
    assert _processor(use_albumentations=True, letter_box_transform=True).eval_image_transform.letter_box_transform


def test_legacy_config_without_the_flags_keeps_the_torchvision_recipe():
    """Checkpoints predating the flags behave like Isaac-GR00T's defaults; training-only keys stay ignored."""
    processor = _processor(state_dropout_prob=0.2, extra_augmentation_config=None)
    assert processor.use_albumentations is False
    assert processor.letter_box_transform is False
    assert isinstance(processor.eval_image_transform, transforms.Compose)
    assert not _has_letterbox(processor.eval_image_transform)


def test_fraction_and_max_size_fall_back_like_isaac_gr00t():
    transform = _build_eval_image_transform(
        TARGET, CROP, shortest_image_edge=None, crop_fraction=None, use_albumentations=True
    )
    assert isinstance(transform, AlbumentationsEvalImageTransform)
    assert transform.fraction == CROP[0] / TARGET[0]
    assert transform.max_size == TARGET[0]
    with pytest.raises(ValueError, match="crop_fraction is None"):
        _build_eval_image_transform(TARGET, None, crop_fraction=None, use_albumentations=True)


def test_geometry_follows_isaac_rounding_rules():
    frame = _hash_noise(540, 640, seed=3)
    resized = _smallest_edge_resize(frame, EDGE)
    assert resized.shape == (256, 303, 3)  # round() per dimension, shortest edge -> 256
    cropped = _fractional_center_crop(resized, FRACTION)
    assert cropped.shape == (243, 287, 3)  # int() truncation: 243.2 -> 243, 287.85 -> 287
    assert np.array_equal(cropped, resized[6:249, 8:295])  # (256-243)//2, (303-287)//2
    square = frame[:256, :256]
    assert _smallest_edge_resize(square, EDGE) is square  # scale == 1.0 is a no-op, not a resample


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_albumentations_recipe_is_bit_exact_with_isaac_gr00t(case: str):
    name, flag = case.split("|letterbox=")
    transform = _build_eval_image_transform(
        TARGET,
        CROP,
        shortest_image_edge=EDGE,
        crop_fraction=FRACTION,
        use_albumentations=True,
        letter_box_transform=(flag == "True"),
    )
    out = transform(synthetic_frames()[name])
    expected_shape, expected_digest = GOLDEN[case]
    assert out.dtype == torch.uint8
    assert tuple(out.shape) == expected_shape
    assert digest(out) == expected_digest


def test_letterbox_is_gated_on_both_recipes():
    frame = _hash_noise(540, 640, seed=3)

    def albumentations(letterbox: bool) -> torch.Tensor:
        return _build_eval_image_transform(
            TARGET,
            CROP,
            shortest_image_edge=EDGE,
            crop_fraction=FRACTION,
            use_albumentations=True,
            letter_box_transform=letterbox,
        )(frame)

    padded, unpadded = albumentations(True), albumentations(False)
    assert tuple(padded.shape) == (3, 256, 256)  # 540x640 -> 640x640 -> 256x256 -> 243x243 -> 256x256
    assert tuple(unpadded.shape) == (3, 256, 302)  # aspect preserved: 256x303 -> 243x287 -> 256x302

    tv_padded = _build_eval_image_transform(TARGET, CROP, use_albumentations=False, letter_box_transform=True)(frame)
    tv_unpadded = _build_eval_image_transform(TARGET, CROP, use_albumentations=False, letter_box_transform=False)(frame)
    assert tuple(tv_padded.shape) == tuple(tv_unpadded.shape) == (3, 256, 256)
    assert not torch.equal(tv_padded, tv_unpadded)
    assert (tv_padded[:, :2, :] == 0).all() and not (tv_unpadded[:, :2, :] == 0).all()  # black bars only when gated


def test_albumentations_recipe_rejects_frames_outside_the_contract():
    transform = _build_eval_image_transform(TARGET, CROP, use_albumentations=True)
    with pytest.raises(ValueError, match="HWC uint8"):
        transform(np.zeros((256, 256, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="HWC uint8"):
        transform(np.zeros((3, 256, 256), dtype=np.uint8))
    with pytest.raises(ValueError, match="HWC uint8"):
        transform(np.zeros((256, 256), dtype=np.uint8))


def test_get_vlm_inputs_runs_the_selected_transform_per_frame():
    processor = _processor(use_albumentations=True, shortest_image_edge=EDGE, crop_fraction=FRACTION)
    frames = np.stack([_hash_noise(540, 640, seed=3)] * 2)  # (T, H, W, C) like policy.py hands over
    vlm_inputs = processor._get_vlm_inputs(
        image_keys=["cam"],
        images={"cam": frames},
        masks=None,
        image_transform=processor.eval_image_transform,
        language="x",
    )
    images = vlm_inputs["vlm_content"]["images"]
    assert len(images) == 2
    assert all(img.size == (302, 256) for img in images)  # PIL reports (W, H); no letterbox for this checkpoint


def test_flags_round_trip_through_save_pretrained_and_from_pretrained(tmp_path):
    saved = _processor(
        use_albumentations=True, letter_box_transform=True, shortest_image_edge=EDGE, crop_fraction=FRACTION
    )
    saved.save_pretrained(tmp_path)
    kwargs = json.loads((tmp_path / "processor_config.json").read_text())["processor_kwargs"]
    assert kwargs["use_albumentations"] is True
    assert kwargs["letter_box_transform"] is True

    loaded = Gr00tN1d7Processor.from_pretrained(tmp_path)
    assert loaded.use_albumentations is True
    assert loaded.letter_box_transform is True
    assert isinstance(loaded.eval_image_transform, AlbumentationsEvalImageTransform)
    assert loaded.eval_image_transform.letter_box_transform is True


def test_from_pretrained_defaults_missing_flags_to_isaac_gr00t_defaults(tmp_path):
    (tmp_path / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "Gr00tN1d7Processor",
                "processor_kwargs": {
                    "modality_configs": MODALITY_CONFIGS,
                    "image_crop_size": CROP,
                    "image_target_size": TARGET,
                },
            }
        )
    )
    (tmp_path / "statistics.json").write_text("{}")
    loaded = Gr00tN1d7Processor.from_pretrained(tmp_path)
    assert loaded.use_albumentations is False
    assert loaded.letter_box_transform is False
    assert isinstance(loaded.eval_image_transform, transforms.Compose)
