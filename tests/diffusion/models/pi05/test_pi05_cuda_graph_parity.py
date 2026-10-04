# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""π0.5's optimized path (fused Triton kernels under CUDA graphs) against the eager baseline.

Loads the real checkpoint once per serving dtype through ``Pi05Pipeline`` with
``enforce_eager=False``, which enables the fused kernels and captures the CUDA
graphs over them at init, and runs ``sample_actions`` on simulated robot
observations with 1, 2 and 3 real camera views, each at the configured
denoising-step count and at a per-request override. Every case runs on the
eager baseline, then with the fused kernels outside the graphs, then on the
optimized path (twice, in both case orders, so a region replaying stale
buffers from the previous case is caught).

Two expectations follow from what changes the computation:

* Graph capture and replay change nothing: the optimized path is bit-exact
  with the same fused kernels run eagerly, and every region must have replayed
  its graph rather than fallen back to eager.
* The fused kernels reorder reductions (GEMM accumulation, RMS variance,
  softmax sums) but keep eager's operations and rounding points
  (``test_pi05_fused_kernels.py``), so the optimized path is held to eager by
  a tolerance. float32 drifts by reduction-order noise; bfloat16 by the
  rounding flips that noise sets off, which is the same size as the drift
  between eager bfloat16 and eager float32.

Further checks: a region falling back to eager mixes bit-exactly with
replayed ones, no graph replays under a default dtype other than the float32
it was captured under, ``sample_actions`` never syncs with the host, and it
fills the KV cache the pipeline preallocates.

All paths share one model; only ``model.cuda_graphs`` and the backbone's
``fused_kernels`` switch are swapped. The float32 checkpoint alone is
~14.5 GB, so a second copy would not fit a 16 GB card.

Needs a CUDA GPU and the real checkpoint::

    python -m pytest tests/diffusion/models/pi05/test_pi05_cuda_graph_parity.py -v -s

``PI05_PARITY_MODEL_PATH`` points at a local checkpoint to skip the HF download.
``PI05_CUDA_GRAPH_PARITY_DTYPES`` (comma separated, default
``float32,bfloat16``) selects the serving dtypes.
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from vllm.utils.torch_utils import set_default_torch_dtype

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

# How far the optimized path may move a chunk from the eager baseline, per
# serving dtype: (relative L2, largest absolute difference).
# * float32: reduction-order noise, measured at most 1.5e-6 and 3.3e-6 on these
#   cases; the bound leaves room for other GPUs' cuBLAS kernel choices and is
#   still far below any real error.
# * bfloat16: no further than the bfloat16 layout itself moves a chunk from
#   float32, which eager bfloat16 does by up to 2.4% and 0.049 on these cases.
#   Measured: at most 1.4% and 0.021, and fused bfloat16 is as close to eager
#   float32 as eager bfloat16 is.
TOLERANCE = {torch.float32: (1e-4, 1e-4), torch.bfloat16: (2.5e-2, 5e-2)}


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


def _sample_on_device(model, inputs, num_steps: int | None, seed: int) -> torch.Tensor:
    images, image_masks, lang_tokens, lang_masks = inputs
    generator = torch.Generator(device=lang_tokens.device).manual_seed(seed)
    # inference_mode, as in Pi05Pipeline.forward.
    with torch.inference_mode():
        return model.sample_actions(
            images=images,
            image_masks=image_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            num_steps=num_steps,
            generator=generator,
        )


def _sample(model, inputs, num_steps: int | None, seed: int) -> torch.Tensor:
    return _sample_on_device(model, inputs, num_steps, seed).cpu()


@pytest.fixture(scope="module", params=DTYPES)
def pipeline(request):
    od_config = OmniDiffusionConfig(
        model=_resolve_checkpoint_dir(),
        dtype=request.param,
        model_config=_deploy_model_config(),
        enforce_eager=False,
    )
    # Constructed the way DiffusersLoader does it: under the serving dtype as
    # torch's default dtype, which is also when the CUDA graphs are captured.
    # Requests then run under the float32 default.
    with set_default_torch_dtype(od_config.dtype):
        pipe = Pi05Pipeline(od_config=od_config)
    yield pipe
    # pytest still holds the fixture value during teardown, so release the
    # weights explicitly before the next dtype loads its copy.
    del pipe.model
    gc.collect()
    torch.accelerator.empty_cache()


@contextmanager
def _path(model, *, graphs, fused_kernels: bool):
    """Run ``model`` on the given execution path; restore the optimized one after."""
    optimized = model.cuda_graphs
    model.cuda_graphs = graphs
    model.paligemma_with_expert.fused_kernels = fused_kernels
    try:
        yield
    finally:
        model.cuda_graphs = optimized
        model.paligemma_with_expert.fused_kernels = True


@pytest.fixture(scope="module")
def outputs(pipeline):
    """Run every case eagerly, with the fused kernels eagerly, then on the optimized path."""
    model = pipeline.model
    graphs = model.cuda_graphs
    assert graphs is not None, "enforce_eager=False on a CUDA device must install the CUDA Graph path."
    assert model.paligemma_with_expert.fused_kernels, "enforce_eager=False must enable the fused kernels."

    inputs = {
        case: pipeline.processor.build_model_inputs(_robot_obs(pipeline.config, case[0], seed=index))
        for index, case in enumerate(CASES)
    }

    def run_all():
        return {case: _sample(model, inputs[case], case[1], seed=index) for index, case in enumerate(CASES)}

    with _path(model, graphs=None, fused_kernels=False):
        eager = run_all()
    with _path(model, graphs=None, fused_kernels=True):
        fused_eager = run_all()

    optimized: dict = {}
    replays_before = graphs.num_replays.copy()
    for round_index, order in enumerate((CASES, CASES[::-1])):
        for views, steps in order:
            index = CASES.index((views, steps))
            optimized[(round_index, views, steps)] = _sample(model, inputs[(views, steps)], steps, seed=index)
    replays = graphs.num_replays - replays_before
    return SimpleNamespace(eager=eager, fused_eager=fused_eager, optimized=optimized, replays=replays, inputs=inputs)


@pytest.mark.parametrize("num_steps", NUM_STEPS, ids=lambda steps: f"steps={steps or 'default'}")
@pytest.mark.parametrize("num_views", NUM_VIEWS, ids=lambda views: f"views={views}")
def test_cuda_graph_path_is_bit_exact(outputs, pipeline, num_views, num_steps):
    """Capture and replay change no computation: the optimized path equals its
    kernels run eagerly, bit for bit, in both rounds."""
    reference = outputs.fused_eager[(num_views, num_steps)]
    assert reference.shape == (1, pipeline.config.chunk_size, pipeline.config.max_action_dim)
    assert torch.isfinite(reference).all()

    for round_index in (0, 1):
        actual = outputs.optimized[(round_index, num_views, num_steps)]
        max_abs_diff = (actual.double() - reference.double()).abs().max().item()
        assert torch.equal(actual, reference), (
            f"CUDA Graph path differs from its kernels run eagerly (round {round_index}): "
            f"max |diff| = {max_abs_diff:.3e}"
        )


@pytest.mark.parametrize("num_steps", NUM_STEPS, ids=lambda steps: f"steps={steps or 'default'}")
@pytest.mark.parametrize("num_views", NUM_VIEWS, ids=lambda views: f"views={views}")
def test_optimized_path_matches_eager(outputs, pipeline, num_views, num_steps):
    """The fused kernels reorder reductions only, so the optimized chunk stays
    within reduction-order drift of the eager baseline (``TOLERANCE``)."""
    reference = outputs.eager[(num_views, num_steps)].double()
    actual = outputs.optimized[(0, num_views, num_steps)].double()
    assert torch.isfinite(reference).all()
    max_rel, max_abs = TOLERANCE[pipeline._torch_dtype]
    rel = ((actual - reference).norm() / reference.norm()).item()
    abs_diff = (actual - reference).abs().max().item()
    print(f"\n{pipeline._torch_dtype} views={num_views} steps={num_steps}: rel L2 {rel:.3e}, max |diff| {abs_diff:.3e}")
    assert rel <= max_rel and abs_diff <= max_abs, (
        f"optimized path moved the chunk by rel L2 {rel:.3e}, max |diff| {abs_diff:.3e}; "
        f"allowed {max_rel:.1e}, {max_abs:.1e}"
    )


def test_cases_differ_from_each_other(outputs):
    """Guards the parity check itself: identical chunks across cases would let a
    path that ignores its inputs, or replays the previous case, pass."""
    chunks = list(outputs.eager.values())
    for i, first in enumerate(chunks):
        for second in chunks[i + 1 :]:
            assert not torch.equal(first, second)


def test_cuda_graph_path_replays_every_region(outputs, pipeline):
    """Guards the parity check itself: a region that silently fell back to eager
    would still be bit-exact. Each call replays regions 1 and 2 once and region
    3 once per step, over two rounds of every case."""
    replays = outputs.replays
    calls = 2 * len(CASES)
    steps = 2 * sum(pipeline.config.num_inference_steps if steps is None else steps for _, steps in CASES)
    assert replays == {"embed_prefix": calls, "prefix_forward": calls, "denoise_step": steps}


def test_eager_fallback_mixes_with_replayed_regions(outputs, pipeline):
    """Without the preallocated KV cache, ``sample_actions`` takes a one-off
    one, so regions 2 and 3 fall back to eager (still through the fused
    kernels) while region 1 still replays and hands them its graph outputs.
    The chunk must not change."""
    model = pipeline.model
    graphs = model.cuda_graphs
    case = (3, None)
    index = CASES.index(case)

    kv_cache = model.kv_cache
    replays_before = graphs.num_replays.copy()
    model.kv_cache = None
    try:
        actual = _sample(model, outputs.inputs[case], case[1], seed=index)
    finally:
        model.kv_cache = kv_cache

    assert graphs.num_replays - replays_before == {"embed_prefix": 1}
    assert torch.equal(actual, outputs.fused_eager[case])


@pytest.mark.parametrize("path", ["eager", "fused_eager", "optimized"])
def test_sample_actions_never_syncs_with_the_host(pipeline, path):
    """Every region is device-only work, the condition for capturing it, and a
    replay adds only device-to-device copies: a host-device copy or a stream
    sync anywhere in ``sample_actions`` fails this."""
    model = pipeline.model
    inputs = pipeline.processor.build_model_inputs(_robot_obs(pipeline.config, 3, seed=0))
    graphs = model.cuda_graphs if path == "optimized" else None
    with _path(model, graphs=graphs, fused_kernels=path != "eager"):
        torch.cuda.set_sync_debug_mode("error")
        try:
            _sample_on_device(model, inputs, num_steps=None, seed=0)
        finally:
            torch.cuda.set_sync_debug_mode("default")


def test_sample_actions_writes_the_preallocated_kv_cache(pipeline):
    """The pipeline allocates the KV cache once at init for the deployed
    prefix, and ``sample_actions`` fills every slot of it."""
    model = pipeline.model
    cache = model.kv_cache
    inputs = pipeline.processor.build_model_inputs(_robot_obs(pipeline.config, 3, seed=0))
    with torch.inference_mode():
        prefix_len = model.embed_prefix(*inputs)[0].shape[1]
    assert cache is not None and cache.fits(1, prefix_len)

    cache.key.fill_(float("nan"))
    cache.value.fill_(float("nan"))
    _sample_on_device(model, inputs, num_steps=None, seed=0)
    assert model.kv_cache is cache
    assert torch.isfinite(cache.key).all() and torch.isfinite(cache.value).all()


def test_other_default_dtype_runs_eagerly(pipeline):
    """The eager float mask follows torch's default dtype, so a graph captured
    under float32 must not replay under another default."""
    model = pipeline.model
    graphs = model.cuda_graphs
    inputs = pipeline.processor.build_model_inputs(_robot_obs(pipeline.config, 3, seed=0))
    replays_before = graphs.num_replays.copy()
    with set_default_torch_dtype(torch.bfloat16):
        _sample(model, inputs, num_steps=None, seed=0)
    assert graphs.num_replays == replays_before
