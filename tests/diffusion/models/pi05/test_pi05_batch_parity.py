# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A formed batch of π0.5 requests against the requests run alone (R2 acceptance).

Loads the real checkpoint once per serving dtype through ``Pi05Pipeline`` with
``enforce_eager=False`` and the graphs captured for every size under test, and
runs ``sample_actions`` on batches of simulated robot observations whose rows
differ in camera count, frames, instruction length, state and noise. In
float32, each row of a batch must be the chunk its request gets alone within
the R2 bound. In bfloat16 the layout's own rounding moves a chunk from the
float32 one by several percent, and a batch rounds differently than a request
alone, so a bfloat16 row is held to the float32 reference instead: no farther
from it than the farthest request alone in bfloat16. In both, the other rows
must not move when one row's request is replaced.

Needs a CUDA GPU with memory for the largest batch (80 GB for 32 in float32)
and the real checkpoint::

    python -m pytest tests/diffusion/models/pi05/test_pi05_batch_parity.py -v -s

``PI05_PARITY_MODEL_PATH`` points at a local checkpoint to skip the HF download.
``PI05_BATCH_PARITY_DTYPES`` (default ``float32,bfloat16``) selects the serving
dtypes; ``PI05_BATCH_PARITY_SIZES`` (default ``1,2,4,8,16,32``) the batch sizes,
which are also the sizes the graphs are captured for.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from vllm.utils.torch_utils import set_default_torch_dtype

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.pi05.pipeline_pi05 import Pi05Pipeline

pytestmark = [
    pytest.mark.local_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "H100"}),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="The batch parity test needs a CUDA GPU."),
]

MODEL_PATH = os.environ.get("PI05_PARITY_MODEL_PATH", "lerobot/pi05_base")
DTYPES = os.environ.get("PI05_BATCH_PARITY_DTYPES", "float32,bfloat16").split(",")
SIZES = tuple(sorted({int(size) for size in os.environ.get("PI05_BATCH_PARITY_SIZES", "1,2,4,8,16,32").split(",")}))
DEPLOY_CONFIG = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "pi05.yaml"

# float32: how far a row of a batch may move from its request alone (relative
# L2, largest absolute difference): the R2 acceptance bound. bfloat16 has no
# constant: a row may be as far from the float32 reference as the farthest of
# the requests alone in bfloat16 is (up to 7.8% and 0.077 on an A100).
FLOAT32_TOLERANCE = (1e-4, 1e-4)

# Instructions of different lengths, so rows differ in valid text tokens and padding.
PROMPTS = (
    "turn on the stove",
    "open the middle drawer of the cabinet",
    "pick up the red block and place it in the bin",
    "put both the alphabet soup and the tomato sauce in the basket",
    "pick up the black bowl between the plate and the ramekin and place it on the plate",
)


def _resolve_checkpoint_dir() -> str:
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_PATH, repo_type="model")


def _deploy_model_config() -> dict:
    deploy = yaml.safe_load(DEPLOY_CONFIG.read_text())
    (stage,) = deploy["stages"]
    return stage["model_config"]


def _robot_obs(config, index: int) -> dict:
    """Request ``index``: ``1 + index % 3`` cameras, its own frames, state and instruction."""
    rng = np.random.default_rng(index)
    num_views = 1 + index % 3
    images = {
        key: rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8) for key in config.image_feature_keys[:num_views]
    }
    state = rng.uniform(-1.0, 1.0, size=config.state_dim).astype(np.float32)
    return {"images": images, "state": state, "prompt": PROMPTS[index % len(PROMPTS)]}


class _Request:
    """One request's model inputs and the fixed noise it denoises from."""

    def __init__(self, pipeline, index: int):
        model = pipeline.model
        self.index = index
        self.inputs = pipeline.processor.build_model_inputs(_robot_obs(pipeline.config, index))
        device = self.inputs[2].device
        self.noise = torch.randn(
            1,
            model.action_horizon,
            model.action_dim,
            dtype=torch.float32,
            device=device,
            generator=torch.Generator(device=device).manual_seed(1000 + index),
        )
        self.views = int(sum(mask.item() for mask in self.inputs[1]))


def _batch(requests: list[_Request]) -> dict:
    """One ``sample_actions`` call over the rows of ``requests``."""
    parts = [request.inputs for request in requests]
    return dict(
        images=[torch.cat([part[0][cam] for part in parts]) for cam in range(len(parts[0][0]))],
        image_masks=[torch.cat([part[1][cam] for part in parts]) for cam in range(len(parts[0][1]))],
        lang_tokens=torch.cat([part[2] for part in parts]),
        lang_masks=torch.cat([part[3] for part in parts]),
        noise=torch.cat([request.noise for request in requests]),
    )


def _sample(model, inputs: dict) -> torch.Tensor:
    with torch.inference_mode():
        return model.sample_actions(**inputs).cpu()


@pytest.fixture(scope="module", params=DTYPES)
def pipeline(request):
    od_config = OmniDiffusionConfig(
        model=_resolve_checkpoint_dir(),
        dtype=request.param,
        model_config=_deploy_model_config(),
        enforce_eager=False,
        max_num_seqs=max(SIZES),
    )
    # Constructed the way DiffusersLoader does it: under the serving dtype as
    # torch's default dtype, which is also when the CUDA graphs are captured.
    with set_default_torch_dtype(od_config.dtype):
        pipe = Pi05Pipeline(od_config=od_config)
    assert pipe.model.cuda_graphs is not None, "enforce_eager=False on a CUDA device must install the graph path."
    yield pipe
    del pipe.model
    gc.collect()
    torch.accelerator.empty_cache()


@pytest.fixture(scope="module")
def requests(pipeline) -> list[_Request]:
    """``max(SIZES) + 1`` distinct requests; the last one replaces a row in the contamination test."""
    return [_Request(pipeline, index) for index in range(max(SIZES) + 1)]


# The float32 chunks of the requests alone, kept from the float32 pipeline for
# the bfloat16 one (the dtypes run in DTYPES order).
_FLOAT32_ALONE: dict[int, torch.Tensor] = {}


@pytest.fixture(scope="module")
def alone(pipeline, requests) -> dict[int, torch.Tensor]:
    """Every request's chunk alone, on the optimized path of the serving dtype."""
    chunks = {request.index: _sample(pipeline.model, _batch([request])) for request in requests}
    if pipeline._torch_dtype == torch.float32:
        _FLOAT32_ALONE.update(chunks)
    return chunks


@pytest.fixture(scope="module")
def reference(pipeline, requests, alone) -> dict[int, torch.Tensor]:
    """Every request's chunk alone in float32: ``alone`` itself for the float32
    pipeline, else the float32 pipeline's, computed eagerly here if that dtype
    did not run first."""
    if pipeline._torch_dtype == torch.float32:
        return alone
    if not _FLOAT32_ALONE:
        od_config = OmniDiffusionConfig(
            model=_resolve_checkpoint_dir(), dtype="float32", model_config=_deploy_model_config(), enforce_eager=True
        )
        with set_default_torch_dtype(torch.float32):
            float32 = Pi05Pipeline(od_config=od_config)
        for request in requests:
            _FLOAT32_ALONE[request.index] = _sample(float32.model, _batch([_Request(float32, request.index)]))
        del float32
        gc.collect()
        torch.accelerator.empty_cache()
    return _FLOAT32_ALONE


def _distance(mine: torch.Tensor, other: torch.Tensor) -> tuple[float, float]:
    """``(relative L2, largest absolute difference)`` of ``mine`` from ``other``."""
    mine, other = mine.double(), other.double()
    return ((mine - other).norm() / other.norm()).item(), (mine - other).abs().max().item()


def _assert_rows(pipeline, alone, reference, actual: torch.Tensor, rows: list[_Request], label: str) -> None:
    """Each row of ``actual`` is its request's chunk: in float32 the one it gets
    alone within ``FLOAT32_TOLERANCE``; in bfloat16 no farther from the float32
    reference than the farthest of all the requests alone in bfloat16."""
    dtype = pipeline._torch_dtype
    mine = [actual[position : position + 1] for position in range(len(rows))]
    if dtype == torch.float32:
        max_rel, max_abs = FLOAT32_TOLERANCE
        distances = [_distance(row, alone[request.index]) for row, request in zip(mine, rows)]
        against = "the request alone"
    else:
        by_themselves = [_distance(alone[index], reference[index]) for index in alone]
        max_rel, max_abs = max(d[0] for d in by_themselves), max(d[1] for d in by_themselves)
        distances = [_distance(row, reference[request.index]) for row, request in zip(mine, rows)]
        against = "the float32 reference"
    for position, (request, (rel, abs_diff)) in enumerate(zip(rows, distances)):
        assert rel <= max_rel and abs_diff <= max_abs, (
            f"{label}: row {position} (request {request.index}, {request.views} views) is rel L2 {rel:.3e}, "
            f"max |diff| {abs_diff:.3e} from {against}; allowed {max_rel:.3e}, {max_abs:.3e}"
        )
    worst_rel, worst_abs = max(d[0] for d in distances), max(d[1] for d in distances)
    note = f"{dtype} {label}: worst row rel L2 {worst_rel:.3e}, max |diff| {worst_abs:.3e} from {against}"
    if dtype != torch.float32:
        batched = sum(d[0] for d in distances) / len(rows)
        by_itself = sum(_distance(alone[request.index], reference[request.index])[0] for request in rows) / len(rows)
        note += f" (allowed {max_rel:.3e}, {max_abs:.3e}); mean rel L2 {batched:.3e} batched, {by_itself:.3e} alone"
    print("\n" + note)


@pytest.mark.parametrize("size", SIZES, ids=lambda size: f"batch={size}")
def test_formed_batch_rows_match_the_requests_alone(pipeline, requests, alone, reference, size):
    """Every row of a formed batch of a captured size is the chunk its request
    gets alone (see ``_assert_rows``), with rows of 1, 2 and 3 cameras and of
    different instruction lengths in the same batch."""
    rows = requests[:size]
    actual = _sample(pipeline.model, _batch(rows))
    assert actual.shape == (size, pipeline.config.chunk_size, pipeline.config.max_action_dim)
    assert torch.isfinite(actual).all()
    _assert_rows(pipeline, alone, reference, actual, rows, f"batch={size}")


def test_batch_between_captured_sizes_rows_match_the_requests_alone(pipeline, requests, alone, reference):
    """A batch of an undeclared size (its prefix regions without a graph, its
    steps on the next larger graph's padded rows) gives each request its chunk."""
    if len(SIZES) < 2:
        pytest.skip("needs two captured sizes to have a size between them")
    size = (SIZES[-2] + SIZES[-1]) // 2
    if size in SIZES:
        pytest.skip(f"no size between {SIZES[-2]} and {SIZES[-1]}")
    rows = requests[:size]
    actual = _sample(pipeline.model, _batch(rows))
    assert actual.shape[0] == size
    _assert_rows(pipeline, alone, reference, actual, rows, f"batch={size} (between {SIZES[-2]} and {SIZES[-1]})")


@pytest.mark.parametrize("size", [size for size in SIZES if size > 1], ids=lambda size: f"batch={size}")
def test_replacing_one_row_leaves_the_others_bit_identical(pipeline, requests, alone, reference, size):
    """No cross-request contamination: with one row's request replaced by
    another (of a different camera count where possible), every other row's
    chunk is bit for bit what it was, and the new row is its request alone."""
    rows = requests[:size]
    before = _sample(pipeline.model, _batch(rows))
    replaced = size // 2
    substitute = requests[max(SIZES)]  # not among the rows of any batch under test
    rows_after = [substitute if position == replaced else request for position, request in enumerate(rows)]
    after = _sample(pipeline.model, _batch(rows_after))
    others = [position for position in range(size) if position != replaced]
    assert torch.equal(after[others], before[others]), (
        f"batch={size}: replacing row {replaced} ({rows[replaced].views} -> {substitute.views} views) changed "
        f"other rows by up to {(after[others].double() - before[others].double()).abs().max().item():.3e}"
    )
    _assert_rows(pipeline, alone, reference, after[replaced : replaced + 1], [substitute], f"batch={size} new row")
