# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end WAN check for the typed pre-D2H media contract.

Runs the whole engine twice with the same seed, once per flag state, and compares
the frames a client receives. WAN denormalizes historical bfloat16 output in the
input dtype while device preparation computes in float32, so the paths may differ
by one 255th.

Run with a WAN2.2 checkpoint::

    VLLM_OMNI_DEVICE_POSTPROCESS_MODEL=Wan-AI/Wan2.2-TI2V-5B-Diffusers \
        pytest -s tests/e2e/offline_inference/test_device_postprocess_equivalence.py
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("VLLM_OMNI_DEVICE_POSTPROCESS_MODEL")

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not MODEL,
        reason="Set VLLM_OMNI_DEVICE_POSTPROCESS_MODEL to a WAN2.2 checkpoint to run this.",
    ),
]

_SEED = 1234
_STEPS = int(os.environ.get("VLLM_OMNI_DEVICE_POSTPROCESS_STEPS", "2"))
_SIZE = int(os.environ.get("VLLM_OMNI_DEVICE_POSTPROCESS_SIZE", "256"))
_FRAMES = int(os.environ.get("VLLM_OMNI_DEVICE_POSTPROCESS_FRAMES", "17"))


def _generate(enable_device_postprocess: bool) -> np.ndarray:
    from vllm_omni.entrypoints.omni import Omni

    engine = Omni(
        model=MODEL,
        num_gpus=1,
        video_output_transport={"enable_device_postprocess": enable_device_postprocess},
    )
    try:
        # output_type must travel with the request: the pipelines read it from the
        # sampling params, and the reduction only covers the "np" path.
        sampling_params = OmniDiffusionSamplingParams(
            output_type="np",
            seed=_SEED,
            num_inference_steps=_STEPS,
            height=_SIZE,
            width=_SIZE,
            num_frames=_FRAMES,
        )
        outputs = engine.generate({"prompt": "a robot waving"}, sampling_params)
    finally:
        engine.close()

    video = outputs[0].images[0]
    return video.numpy() if hasattr(video, "numpy") else np.asarray(video)


def _as_uint8_frames(video: np.ndarray) -> np.ndarray:
    """Reproduce what the API server does before encoding.

    A float payload is scaled and rounded there, so that is the only fair place to
    compare it against a payload the worker already reduced.
    """
    if video.dtype == np.uint8:
        return video
    return np.rint(np.clip(video, 0.0, 1.0) * 255.0).astype(np.uint8)


# A dtype difference moves ~20% of pixels by one; dropping the round moves ~50%.
_MAX_DIFFERING_FRACTION = 0.35


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_device_postprocess_produces_equivalent_frames() -> None:
    float_path = _generate(False)
    device_path = _generate(True)

    # Guard against the run silently taking the float path twice, which would make
    # the comparison meaningless.
    assert float_path.dtype != np.uint8, "the float path returned uint8; the gate may be stuck open"
    assert device_path.dtype == np.uint8, "the WAN typed-media path did not prepare uint8 frames"
    assert float_path.shape == device_path.shape

    expected = _as_uint8_frames(float_path)
    deviation = np.abs(device_path.astype(np.int16) - expected.astype(np.int16))
    differing = float((deviation > 0).mean())
    print(f"maxdiff={deviation.max()} differing={differing:.4f} ({int((deviation > 0).sum())}/{deviation.size})")

    assert deviation.max() <= 1, "the reduction moved a pixel by more than one 255th"
    assert differing < _MAX_DIFFERING_FRACTION, (
        f"{differing:.1%} of pixels differ, more than a denormalize-precision difference explains"
    )
