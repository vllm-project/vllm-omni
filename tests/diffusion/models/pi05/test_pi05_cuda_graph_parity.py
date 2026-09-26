# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5 CUDA Graph path against the eager baseline, bit-exact.

Loads the real checkpoint once per serving dtype through ``Pi05Pipeline`` with
``enforce_eager=False`` and runs ``sample_actions`` on simulated robot
observations with 1, 2 and 3 real camera views, each at the configured
denoising-step count and at a per-request override. Every case runs on the
eager path first, then on the CUDA Graph path (twice, in both case orders, so a
region replaying stale buffers from the previous case is caught). The saved
chunks are compared with ``torch.equal`` at the end.

Both paths share one model; only ``model.cuda_graphs`` is swapped. The float32
checkpoint alone is ~14.5 GB, so a second copy would not fit a 16 GB card.

Needs a CUDA GPU and the real checkpoint::

    python -m pytest tests/diffusion/models/pi05/test_pi05_cuda_graph_parity.py -v -s

``PI05_PARITY_MODEL_PATH`` points at a local checkpoint to skip the HF download.
``PI05_CUDA_GRAPH_PARITY_DTYPES`` (comma separated, default
``float32,bfloat16``) selects the serving dtypes.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.diffusion,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="The CUDA Graph path needs a CUDA GPU."),
]

MODEL_PATH = os.environ.get("PI05_PARITY_MODEL_PATH", "lerobot/pi05_base")
DTYPES = os.environ.get("PI05_CUDA_GRAPH_PARITY_DTYPES", "float32,bfloat16").split(",")
DEPLOY_CONFIG = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "pi05.yaml"

NUM_VIEWS = (1, 2, 3)
# ``None`` is the configured default (10 in pi05.yaml); 4 is a per-request override.
NUM_STEPS = (None, 4)
CASES = [(views, steps) for views in NUM_VIEWS for steps in NUM_STEPS]
PROMPT = "pick up the red block and place it in the bin"


def _resolve_checkpoint_dir() -> str:
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_PATH, repo_type="model")


def _deploy_model_config() -> dict:
    """The serving ``model_config``, so the test runs the deployed shapes."""
    deploy = yaml.safe_load(DEPLOY_CONFIG.read_text())
    (stage,) = deploy["stages"]
    return stage["model_config"]


def _robot_obs(config, num_views: int, seed: int) -> dict:
    """A simulated OpenPI observation with the first ``num_views`` cameras.

    Camera frames are larger than 224x224 and non-square, so preprocessing
    resizes and pads them exactly as it does for a real robot.
    """
    rng = np.random.default_rng(seed)
    images = {
        key: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for key in config.image_feature_keys[:num_views]
    }
    state = rng.uniform(-1.0, 1.0, size=config.state_dim).astype(np.float32)
    return {"images": images, "state": state, "prompt": PROMPT}


def _sample(model, inputs, num_steps: int | None, seed: int) -> torch.Tensor:
    images, image_masks, lang_tokens, lang_masks = inputs
    generator = torch.Generator(device=lang_tokens.device).manual_seed(seed)
    # inference_mode, as in Pi05Pipeline.forward.
    with torch.inference_mode():
        actions = model.sample_actions(
            images=images,
            image_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            num_steps=num_steps,
            generator=generator,
        )
    return actions.cpu()


@pytest.fixture(scope="module", params=DTYPES)
def pipeline(request):
    od_config = OmniDiffusionConfig(
        model=_resolve_checkpoint_dir(),
        dtype=request.param,
        model_config=_deploy_model_config(),
        enforce_eager=False,
    )
    pipe = Pi05Pipeline(od_config=od_config)
    yield pipe
    # pytest still holds the fixture value during teardown, so release the
    # weights explicitly before the next dtype loads its copy.
    del pipe.model
    gc.collect()
    torch.accelerator.empty_cache()


@pytest.fixture(scope="module")
def outputs(pipeline):
    """Run every case eagerly, then on the CUDA Graph path; compare later."""
    model = pipeline.model
    graphs = model.cuda_graphs
    assert graphs is not None, "enforce_eager=False on a CUDA device must install the CUDA Graph path."

    inputs = {
        case: pipeline.processor.build_model_inputs(_robot_obs(pipeline.config, case[0], seed=index))
        for index, case in enumerate(CASES)
    }

    eager: dict = {}
    model.cuda_graphs = None
    try:
        for index, (views, steps) in enumerate(CASES):
            eager[(views, steps)] = _sample(model, inputs[(views, steps)], steps, seed=index)
    finally:
        model.cuda_graphs = graphs

    optimized: dict = {}
    for round_index, order in enumerate((CASES, CASES[::-1])):
        for views, steps in order:
            index = CASES.index((views, steps))
            optimized[(round_index, views, steps)] = _sample(model, inputs[(views, steps)], steps, seed=index)
    return eager, optimized


@pytest.mark.parametrize("num_steps", NUM_STEPS, ids=lambda steps: f"steps={steps or 'default'}")
@pytest.mark.parametrize("num_views", NUM_VIEWS, ids=lambda views: f"views={views}")
def test_cuda_graph_path_is_bit_exact(outputs, pipeline, num_views, num_steps):
    eager, optimized = outputs
    reference = eager[(num_views, num_steps)]
    assert reference.shape == (1, pipeline.config.chunk_size, pipeline.config.max_action_dim)
    assert torch.isfinite(reference).all()

    for round_index in (0, 1):
        actual = optimized[(round_index, num_views, num_steps)]
        max_abs_diff = (actual.double() - reference.double()).abs().max().item()
        assert torch.equal(actual, reference), (
            f"CUDA Graph path differs from eager (round {round_index}): max |diff| = {max_abs_diff:.3e}"
        )


def test_cases_differ_from_each_other(outputs):
    """Guards the parity check itself: identical chunks across cases would let a
    path that ignores its inputs, or replays the previous case, pass."""
    eager, _ = outputs
    chunks = list(eager.values())
    for i, first in enumerate(chunks):
        for second in chunks[i + 1 :]:
            assert not torch.equal(first, second)
