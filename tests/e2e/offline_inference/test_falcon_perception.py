"""
End-to-end test for Falcon Perception (image + text -> boxes and instance masks).

Verifies that the thinker -> segmentation pipeline reproduces a golden token
stream and instance count, and that the masks match a golden mask summary.

Model Hub repo id: ``tiiuae/Falcon-Perception``.
Deploy config: ``get_deploy_config_path("falcon_perception.yaml")``
  -> ``vllm_omni/deploy/falcon_perception.yaml``

The golden is the ``_GOLDEN`` constant below, inlined rather than stored as a
fixture file. Regenerate it with::

  UPDATE_GOLDEN=1 pytest tests/e2e/offline_inference/test_falcon_perception.py \
      -m 'advanced_model and cuda' --run-level 'advanced_model'

which prints a replacement block to paste over ``_GOLDEN``.

Masks are asserted by per-instance area ratio and centroid distance rather than
matching in a bit-exact manner. This is because bf16 attention is not batch-invariant
So mask boundaries move slightly between runs (characterised as 0.98-0.99 IoU at 1px tolerance).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from vllm.assets.image import ImageAsset
from vllm.sampling_params import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path

MODEL_PATH = "tiiuae/Falcon-Perception"
DEPLOY_CONFIG = get_deploy_config_path("falcon_perception.yaml")

_OMNI_RUNNER_PARAM = (MODEL_PATH, DEPLOY_CONFIG)

# Golden, inlined rather than kept in a fixture file: it is small enough to read
# in a diff, and a reviewer can see the expected values without opening a second
# file. Recorded with the compiled backbone and compiled AnyUp from the
# canonical deploy profile on tiiuae/Falcon-Perception, 1x A100-80GB,
# vLLM 0.26.0, greedy. AnyUp compilation can move mask boundaries enough to
# change the area-ordered NMS output, so regenerate this golden when that
# setting changes.
# Regenerate with UPDATE_GOLDEN=1 -- it prints a replacement block to paste here.
#
# The token stream is one <|coord|>/<|size|>/<|seg|> triple (240, 241, 262) per
# candidate, bracketed by 268 and the 263 stop token. Mask NMS removes one
# overlapping candidate below, so the final mask count is intentionally lower.
# fmt: off
_GOLDEN = {
    "tokens": [
        268,
        240, 241, 262,  240, 241, 262,  240, 241, 262,  240, 241, 262,  240, 241, 262,
        240, 241, 262,  240, 241, 262,  240, 241, 262,  240, 241, 262,  240, 241, 262,
        263,
    ],
    "n_instances": 9,
    "masks": [
        {"area": 806, "cy": 296.207, "cx": 634.453},
        {"area": 725, "cy": 269.399, "cx": 654.549},
        {"area": 663, "cy": 318.572, "cx": 471.002},
        {"area": 594, "cy": 297.677, "cx": 770.177},
        {"area": 585, "cy": 313.125, "cx": 682.398},
        {"area": 474, "cy": 322.285, "cx": 573.530},
        {"area": 517, "cy": 298.781, "cx": 792.294},
        {"area": 417, "cy": 347.374, "cx": 642.914},
        {"area": 340, "cy": 339.488, "cx": 719.497},
    ],
}
# fmt: on

# Reference stop tokens: EOS and <|end_of_query|>.
_STOP_TOKEN_IDS = [11, 263]
# A multi-instance query on purpose. A single-instance golden is a weak test:
# 1 area + 1 centroid is cheap to match by accident. 10 instances means ten
# areas and ten centroids must all land inside tolerance, which a broken mask
# head cannot do. Measured alternatives on this same asset (each bit-identical
# across two consecutive runs): "the stop sign" 1, "the stone lion statues" 2,
# "the red lanterns" 9, "the lanterns" 10, "the signs" 27. Denser was available
# but not taken -- "the signs" is a fuzzy referent in a busy street scene, and a
# longer exact-matched token stream is more exposed to cross-GPU bf16 drift.
_QUERY = "the lanterns"


def _build_prompt(query: str) -> str:
    return f"<|image|>Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>"


def _test_image():
    """A stock vLLM asset, so no image is checked into the repo.

    Despite the name it is a busy Chinatown street scene, which is why it can
    support a multi-instance query. It must be a real photograph containing the
    queried object: synthetic noise makes the detector return zero instances,
    which would make the golden vacuous -- the mask comparison would iterate an
    empty list and pass even with the whole segmentation stage broken.
    """
    return ImageAsset("stop_sign").pil_image


def _mask_summary(masks: np.ndarray) -> list[dict]:
    """Per-instance area and centroid — stable enough to store, strict enough to catch drift."""
    out = []
    for mask in masks:
        binary = np.asarray(mask) > 0
        ys, xs = np.nonzero(binary)
        out.append(
            {
                "area": int(binary.sum()),
                "cy": round(float(ys.mean()), 3) if ys.size else 0.0,
                "cx": round(float(xs.mean()), 3) if xs.size else 0.0,
            }
        )
    return out


# Both tests share the runner param; the run-level marks are per-test because the
# two levels need different weights. See each test's docstring.
pytestmark = [
    pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True),
]

# Under dummy weights the thinker emits noise, so cap the decode instead of
# letting it run to the 2048-token ceiling for nothing.
_SMOKE_MAX_TOKENS = 64


def _run_pipeline(omni_runner: OmniRunner, image, *, max_tokens: int):
    """Drive the 2-stage pipeline once -> (tokens, masks, boxes, saw_text_stage).

    This calls ``omni.generate`` directly rather than an
    ``OmniRunnerHandler.send_*_request`` helper: no helper in
    ``tests/helpers/runtime.py`` models the boxes+masks multimodal payload this
    pipeline returns. Tracked as debt -- add ``send_falcon_perception_request``
    there if a second Falcon test needs the same plumbing.
    """
    params = [
        SamplingParams(temperature=0.0, max_tokens=max_tokens, detokenize=True, stop_token_ids=_STOP_TOKEN_IDS),
        SamplingParams(temperature=0.0, max_tokens=1, detokenize=False),
    ]

    tokens: list[int] = []
    masks = np.zeros((0, 1, 1), dtype=np.uint8)
    boxes = np.zeros((0, 4), dtype=np.float32)
    saw_text_stage = False
    for stage_output in omni_runner.omni.generate(
        [{"prompt": _build_prompt(_QUERY), "multi_modal_data": {"image": image}}], params
    ):
        completion = stage_output.request_output.outputs[0]
        if stage_output.final_output_type == "text":
            saw_text_stage = True
            tokens = [int(t) for t in completion.token_ids]
        multimodal = getattr(completion, "multimodal_output", None)
        if not multimodal:
            continue
        if "masks" in multimodal and hasattr(multimodal["masks"], "shape"):
            masks = np.asarray(multimodal["masks"].to(torch.uint8).cpu())
        if "boxes" in multimodal and hasattr(multimodal["boxes"], "shape"):
            boxes = np.asarray(multimodal["boxes"].float().cpu())
    return tokens, masks, boxes, saw_text_stage


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"})
def test_falcon_perception_pipeline_smoke(omni_runner: OmniRunner):
    """L2 wiring check: both stages run and the output payload keeps its contract.

    Runs at ``--run-level core_model``, where the harness rewrites every stage to
    ``load_format: dummy``. The weights are therefore random and *nothing about
    the content* of the output is meaningful -- no instance count, no geometry.
    What this does catch, on every PR and without the checkpoint, is wiring:
    the architecture resolves through the registry, the 2-stage pipeline builds
    from the shipped YAML, stage 0 surfaces text, the stage bridge hands off, and
    any instances stage 1 does emit satisfy the documented shape contract.

    Accuracy is asserted separately by ``test_falcon_perception_masks_e2e`` (L3).
    """
    image = _test_image()
    tokens, masks, boxes, saw_text_stage = _run_pipeline(omni_runner, image, max_tokens=_SMOKE_MAX_TOKENS)

    assert saw_text_stage, "stage 0 produced no text output; the pipeline did not run end to end"
    assert isinstance(tokens, list), "token stream should be a list of ids"
    assert masks.ndim == 3, f"masks should be (n_instances, H, W), got {masks.shape}"

    # Random weights may legitimately yield zero instances, so the count is not
    # asserted. Whatever *is* returned must still honour the contract.
    if masks.shape[0]:
        assert masks.shape[1:] == (image.size[1], image.size[0]), (
            f"masks are {masks.shape[1:]}, expected the original {(image.size[1], image.size[0])}"
        )
        assert boxes.shape == (masks.shape[0], 4), f"expected one (cx, cy, w, h) box per mask, got {boxes.shape}"
        assert masks.max() <= 1, "masks should be binary"


# L3 (``advanced_model``), not L2: at ``--run-level core_model`` the harness
# patches every stage to ``load_format: dummy``, and a golden token stream means
# nothing under random weights. The golden comparison needs the real checkpoint.
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"})
def test_falcon_perception_masks_e2e(omni_runner: OmniRunner):
    """Image + query -> token stream, boxes and instance masks.

    Verifies:
      - the 2-stage pipeline initialises from the shipped YAML
      - the thinker's token stream matches the golden exactly
      - the instance count matches, and masks are at the original resolution
      - per-instance masks match the golden summary within tolerance
    """
    image = _test_image()
    tokens, masks, boxes, _ = _run_pipeline(omni_runner, image, max_tokens=2048)

    # Masks come back at the ORIGINAL image resolution, not the patch-aligned
    # smart_resize size — this is what the reference's finalize_masks does.
    if masks.shape[0]:
        assert masks.shape[1:] == (image.size[1], image.size[0]), (
            f"masks are {masks.shape[1:]}, expected the original {(image.size[1], image.size[0])}"
        )
        assert boxes.shape[0] == masks.shape[0], "one box per instance"

    current = {
        "tokens": tokens,
        "n_instances": int(masks.shape[0]),
        "masks": _mask_summary(masks),
    }

    # A zero-instance result would make every mask assertion below vacuous: the
    # comparison would zip two empty lists and pass with the mask head entirely
    # broken. Refuse to record or trust such a golden.
    assert current["n_instances"] > 0, (
        f"model returned no instances for {_QUERY!r}; a golden built from this would assert nothing about segmentation"
    )

    if os.environ.get("UPDATE_GOLDEN"):
        print(f"\n=== replace _GOLDEN in {Path(__file__).name} with ===\n_GOLDEN = {json.dumps(current, indent=4)}\n")
        pytest.skip("golden printed above; paste it into _GOLDEN")

    golden = _GOLDEN
    assert golden["n_instances"] > 0, "stored golden is vacuous; regenerate it"

    # Exact: the token stream was validated byte-identical against the reference.
    assert tokens == golden["tokens"], "thinker token stream drifted from the golden"
    assert current["n_instances"] == golden["n_instances"], "instance count changed"

    # Tolerant: sub-pixel boundary jitter is expected, structural drift is not.
    for i, (got, want) in enumerate(zip(current["masks"], golden["masks"], strict=True)):
        area_ratio = got["area"] / want["area"]
        assert 0.9 <= area_ratio <= 1.1, f"instance {i} area moved by {area_ratio:.3f}x"
        assert abs(got["cy"] - want["cy"]) <= 2.0, f"instance {i} centroid moved vertically"
        assert abs(got["cx"] - want["cx"]) <= 2.0, f"instance {i} centroid moved horizontally"
