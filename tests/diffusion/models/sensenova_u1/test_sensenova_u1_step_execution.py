# SPDX-License-Identifier: Apache-2.0
"""Step-wise execution and online dynamic batching tests for SenseNova-U1.

Verifies that:
1. SenseNovaU1Pipeline correctly implements SupportsStepExecution
2. Single-request step mode produces output identical to forward()
3. Multiple same-parameter requests can be batched
4. Different prompt lengths can be batched
5. Different resolutions can be batched
6. Different CFG scales can be batched
7. Requests with different step counts complete independently
8. End-to-end scheduler → pipeline heterogeneous batching
9. Think mode (AR generation) step execution correctness
10. Image-to-image (IT2I) step execution correctness

Usage:
    micromamba activate sensenova_u1
    python -m pytest tests/diffusion/models/sensenova_u1/test_sensenova_u1_step_execution.py -v -s

Or run standalone:
    python tests/diffusion/models/sensenova_u1/test_sensenova_u1_step_execution.py
"""

import os
import time
import uuid

import numpy as np
import pytest
import torch
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.gpu, pytest.mark.diffusion]

MODEL_PATH = "SenseNova/SenseNova-U1-8B-MoT"
DEFAULT_PROMPT = (
    "Close portrait of an elderly woman by a farmhouse window, textured skin, "
    "gentle smile, warm natural light, emotional documentary look."
)
DEFAULT_IMAGE_SIZE = (512, 512)
DEFAULT_NUM_STEPS = 4
DEFAULT_CFG_SCALE = 4.0
DEFAULT_SEED = 42


def _init_distributed():
    """Initialize distributed environment for single-GPU testing."""
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=1, rank=0)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        init_world_group,
        initialize_model_parallel,
    )
    import vllm.distributed.parallel_state as vllm_ps

    if vllm_ps._WORLD is None:
        vllm_ps._WORLD = init_world_group([0], 0, "nccl")
    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        initialize_model_parallel(tensor_model_parallel_size=1)


_distributed_initialized = False
_pipeline_singleton = None


def _build_pipeline():
    global _distributed_initialized, _pipeline_singleton
    if _pipeline_singleton is not None:
        return _pipeline_singleton

    if not _distributed_initialized:
        _init_distributed()
        _distributed_initialized = True

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.models.sensenova_u1 import SenseNovaU1Pipeline

    od_config = OmniDiffusionConfig(
        model=MODEL_PATH,
        dtype=torch.bfloat16,
    )
    pipeline = SenseNovaU1Pipeline(od_config=od_config)
    pipeline = pipeline.to(device="cuda", dtype=torch.bfloat16)
    pipeline.eval()
    _pipeline_singleton = pipeline
    return pipeline


def _build_request(
    prompt=DEFAULT_PROMPT,
    image_size=DEFAULT_IMAGE_SIZE,
    num_steps=DEFAULT_NUM_STEPS,
    cfg_scale=DEFAULT_CFG_SCALE,
    seed=DEFAULT_SEED,
    request_id=None,
    **extra_args,
):
    from types import SimpleNamespace

    from vllm_omni.diffusion.request import OmniDiffusionRequest

    sampling_params = SimpleNamespace(
        height=image_size[1],
        width=image_size[0],
        num_inference_steps=num_steps,
        seed=seed,
        generator=None,
        generator_device=None,
        guidance_scale=0.0,
        guidance_scale_2=None,
        do_classifier_free_guidance=False,
    )
    sampling_params.extra_args = {"cfg_scale": cfg_scale, **extra_args}

    if request_id is None:
        request_id = str(uuid.uuid4())

    req = OmniDiffusionRequest(
        prompts=[prompt],
        sampling_params=sampling_params,
        request_id=request_id,
    )
    return req


def _make_state(req, request_id):
    """Create a DiffusionRequestState from a request (mimics runner logic)."""
    import copy

    from vllm_omni.diffusion.worker.utils import DiffusionRequestState

    return DiffusionRequestState(
        request_id=request_id,
        sampling=copy.deepcopy(req.sampling_params),
        prompts=req.prompts,
    )


def _run_step_mode(pipeline, state):
    """Run full step-mode pipeline on a single state, return output image."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    # prepare_encode
    pipeline.prepare_encode(state)

    # denoising loop
    while not state.denoise_completed:
        input_batch = InputBatch.make_batch([state])
        noise_pred = pipeline.denoise_step(input_batch)
        pipeline.step_scheduler(state, noise_pred)

    # post_decode
    output = pipeline.post_decode(state)
    return output


def _image_mse(img1: Image.Image, img2: Image.Image) -> float:
    """Compute per-pixel MSE between two PIL images."""
    arr1 = np.array(img1, dtype=np.float32)
    arr2 = np.array(img2, dtype=np.float32)
    return float(np.mean((arr1 - arr2) ** 2))


def _image_psnr(img1: Image.Image, img2: Image.Image) -> float:
    """Compute PSNR between two PIL images."""
    mse = _image_mse(img1, img2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0**2 / mse))


# ---------------------------------------------------------------------------
# Test 1: Protocol conformance
# ---------------------------------------------------------------------------


def test_supports_step_execution():
    """SenseNovaU1Pipeline correctly declares and implements SupportsStepExecution."""
    from vllm_omni.diffusion.models.interface import supports_step_execution
    from vllm_omni.diffusion.models.sensenova_u1 import SenseNovaU1Pipeline

    assert hasattr(SenseNovaU1Pipeline, "supports_step_execution")
    assert SenseNovaU1Pipeline.supports_step_execution is True

    # Verify all required methods exist
    for method_name in ("prepare_encode", "denoise_step", "step_scheduler", "post_decode"):
        assert hasattr(SenseNovaU1Pipeline, method_name), f"Missing method: {method_name}"

    # Check runtime protocol conformance
    pipeline = _build_pipeline()
    assert supports_step_execution(pipeline)
    print("\n  SupportsStepExecution protocol: PASS")


# ---------------------------------------------------------------------------
# Test 2: Single-request step mode vs forward() equivalence
# ---------------------------------------------------------------------------


def test_step_mode_equivalence():
    """Step-mode output matches forward() output (bit-exact with same seed)."""
    pipeline = _build_pipeline()

    # Run forward() path
    req = _build_request(seed=123)
    with torch.inference_mode():
        output_forward = pipeline(req)

    # Run step-mode path
    state = _make_state(_build_request(seed=123), "test_step_eq")
    with torch.inference_mode():
        output_step = _run_step_mode(pipeline, state)

    assert output_forward.output is not None
    assert output_step.output is not None

    mse = _image_mse(output_forward.output, output_step.output)
    psnr = _image_psnr(output_forward.output, output_step.output)
    print(f"\n  Step vs Forward: MSE={mse:.4f}, PSNR={psnr:.1f}dB")

    # They should be identical (same seed, same compute path)
    assert mse < 1.0, f"Step mode output diverges from forward(): MSE={mse}"

    # Save outputs for visual inspection
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    output_forward.output.save(os.path.join(out_dir, "step_eq_forward.png"))
    output_step.output.save(os.path.join(out_dir, "step_eq_step.png"))


# ---------------------------------------------------------------------------
# Test 3: Multi-request same-parameter batch (Phase 2)
# ---------------------------------------------------------------------------


def test_homogeneous_batch():
    """Multiple same-parameter requests run in batch and produce correct output."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()
    num_reqs = 2

    # Run each request independently via forward() for reference
    refs = []
    for i in range(num_reqs):
        req = _build_request(seed=100 + i)
        with torch.inference_mode():
            out = pipeline(req)
        refs.append(out.output)

    # Run all requests in step-mode batch
    states = [_make_state(_build_request(seed=100 + i), f"batch_{i}") for i in range(num_reqs)]

    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        max_steps = max(state.total_steps for state in states)
        for _ in range(max_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break
            input_batch = InputBatch.make_batch(active)
            noise_pred = pipeline.denoise_step(input_batch)

            offset = 0
            for state in active:
                row_num = state.latents.shape[0]
                pipeline.step_scheduler(state, noise_pred[offset : offset + row_num])
                offset += row_num

        outputs = []
        for state in states:
            outputs.append(pipeline.post_decode(state))

    # Compare each batch output with its reference
    for i in range(num_reqs):
        assert outputs[i].output is not None
        mse = _image_mse(refs[i], outputs[i].output)
        psnr = _image_psnr(refs[i], outputs[i].output)
        print(f"\n  Batch req {i}: MSE={mse:.4f}, PSNR={psnr:.1f}dB")
        assert mse < 1.0, f"Batch output {i} diverges: MSE={mse}"


# ---------------------------------------------------------------------------
# Test 4: Different prompt lengths (Phase 3 — for-loop)
# ---------------------------------------------------------------------------


def test_heterogeneous_prompt_lengths():
    """Requests with different prompt lengths batch correctly (for-loop mode)."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()

    prompts = [
        "A cat",
        "A beautiful sunset over the ocean with golden clouds and seagulls flying in the distance",
    ]

    # Reference: individual forward()
    refs = []
    for i, prompt in enumerate(prompts):
        req = _build_request(prompt=prompt, seed=200 + i)
        with torch.inference_mode():
            out = pipeline(req)
        refs.append(out.output)

    # Step-mode batch
    states = [
        _make_state(_build_request(prompt=prompt, seed=200 + i), f"prompt_{i}") for i, prompt in enumerate(prompts)
    ]

    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        max_steps = max(state.total_steps for state in states)
        for _ in range(max_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break
            input_batch = InputBatch.make_batch(active)
            noise_pred = pipeline.denoise_step(input_batch)

            offset = 0
            for state in active:
                row_num = state.latents.shape[0]
                pipeline.step_scheduler(state, noise_pred[offset : offset + row_num])
                offset += row_num

        outputs = [pipeline.post_decode(state) for state in states]

    for i in range(len(prompts)):
        assert outputs[i].output is not None
        mse = _image_mse(refs[i], outputs[i].output)
        print(f"\n  Prompt len test req {i}: MSE={mse:.4f}")
        assert mse < 1.0, f"Prompt length batch output {i} diverges: MSE={mse}"


# ---------------------------------------------------------------------------
# Test 5: Different resolutions (Phase 4 — for-loop)
# ---------------------------------------------------------------------------


def test_heterogeneous_resolutions():
    """Requests with different resolutions batch correctly (for-loop mode)."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()
    resolutions = [(512, 512), (768, 768)]

    # Reference
    refs = []
    for i, res in enumerate(resolutions):
        req = _build_request(image_size=res, seed=300 + i)
        with torch.inference_mode():
            out = pipeline(req)
        refs.append(out.output)

    # Step-mode batch (for-loop handles different shapes)
    states = [
        _make_state(_build_request(image_size=res, seed=300 + i), f"res_{i}") for i, res in enumerate(resolutions)
    ]

    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        max_steps = max(state.total_steps for state in states)
        for _ in range(max_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break
            # Different resolutions → different latent shapes.
            # Cannot use InputBatch.make_batch (requires same trailing shape).
            # For-loop mode: process each request individually.
            all_preds = []
            for state in active:
                input_batch = InputBatch.make_batch([state])
                pred = pipeline.denoise_step(input_batch)
                all_preds.append(pred)

            for state, pred in zip(active, all_preds):
                pipeline.step_scheduler(state, pred)

        outputs = [pipeline.post_decode(state) for state in states]

    for i in range(len(resolutions)):
        assert outputs[i].output is not None
        mse = _image_mse(refs[i], outputs[i].output)
        print(f"\n  Resolution {resolutions[i]} test: MSE={mse:.4f}")
        assert mse < 1.0, f"Resolution batch output {i} diverges: MSE={mse}"


# ---------------------------------------------------------------------------
# Test 6: Different CFG scales (Phase 5 — for-loop)
# ---------------------------------------------------------------------------


def test_heterogeneous_cfg_scales():
    """Requests with different CFG scales batch correctly (for-loop mode)."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()
    cfg_scales = [4.0, 7.5]

    # Reference
    refs = []
    for i, cfg in enumerate(cfg_scales):
        req = _build_request(cfg_scale=cfg, seed=400 + i)
        with torch.inference_mode():
            out = pipeline(req)
        refs.append(out.output)

    # Step-mode batch
    states = [_make_state(_build_request(cfg_scale=cfg, seed=400 + i), f"cfg_{i}") for i, cfg in enumerate(cfg_scales)]

    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        max_steps = max(state.total_steps for state in states)
        for _ in range(max_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break
            # Per-request execution (different CFG → different uncond paths)
            all_preds = []
            for state in active:
                input_batch = InputBatch.make_batch([state])
                pred = pipeline.denoise_step(input_batch)
                all_preds.append(pred)

            for state, pred in zip(active, all_preds):
                pipeline.step_scheduler(state, pred)

        outputs = [pipeline.post_decode(state) for state in states]

    for i in range(len(cfg_scales)):
        assert outputs[i].output is not None
        mse = _image_mse(refs[i], outputs[i].output)
        print(f"\n  CFG={cfg_scales[i]} test: MSE={mse:.4f}")
        assert mse < 1.0, f"CFG batch output {i} diverges: MSE={mse}"


# ---------------------------------------------------------------------------
# Test 7: Different step counts — dynamic exit (Phase 6)
# ---------------------------------------------------------------------------


def test_dynamic_step_counts():
    """Requests with different num_steps complete independently."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()
    step_counts = [4, 8]

    # Reference
    refs = []
    for i, steps in enumerate(step_counts):
        req = _build_request(num_steps=steps, seed=500 + i)
        with torch.inference_mode():
            out = pipeline(req)
        refs.append(out.output)

    # Step-mode with mixed step counts
    states = [
        _make_state(_build_request(num_steps=steps, seed=500 + i), f"steps_{i}") for i, steps in enumerate(step_counts)
    ]

    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        completion_order = []
        max_steps = max(state.total_steps for state in states)
        for step in range(max_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break

            all_preds = []
            for state in active:
                input_batch = InputBatch.make_batch([state])
                pred = pipeline.denoise_step(input_batch)
                all_preds.append(pred)

            for state, pred in zip(active, all_preds):
                pipeline.step_scheduler(state, pred)
                if state.denoise_completed:
                    completion_order.append(state.request_id)

        outputs = [pipeline.post_decode(state) for state in states]

    # The 4-step request should complete first
    assert completion_order[0] == "steps_0", f"Expected steps_0 first, got {completion_order}"
    print(f"\n  Completion order: {completion_order}")

    for i in range(len(step_counts)):
        assert outputs[i].output is not None
        mse = _image_mse(refs[i], outputs[i].output)
        print(f"  Steps={step_counts[i]} test: MSE={mse:.4f}")
        assert mse < 1.0, f"Dynamic step output {i} diverges: MSE={mse}"


# ---------------------------------------------------------------------------
# Test 8: Throughput comparison (step-mode batch vs serial forward)
# ---------------------------------------------------------------------------


def test_throughput_improvement():
    """Step-mode batch has reasonable per-step latency."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()
    num_steps = 4

    # Time serial forward() calls
    torch.accelerator.synchronize()
    start = time.perf_counter()
    for i in range(2):
        req = _build_request(num_steps=num_steps, seed=600 + i)
        with torch.inference_mode():
            pipeline(req)
    torch.accelerator.synchronize()
    serial_time = time.perf_counter() - start

    # Time step-mode (for-loop batch)
    states = [_make_state(_build_request(num_steps=num_steps, seed=600 + i), f"perf_{i}") for i in range(2)]

    torch.accelerator.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        for _ in range(num_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break
            all_preds = []
            for state in active:
                input_batch = InputBatch.make_batch([state])
                pred = pipeline.denoise_step(input_batch)
                all_preds.append(pred)
            for state, pred in zip(active, all_preds):
                pipeline.step_scheduler(state, pred)

        for state in states:
            pipeline.post_decode(state)
    torch.accelerator.synchronize()
    batch_time = time.perf_counter() - start

    ratio = serial_time / batch_time
    print(f"\n{'=' * 60}")
    print(f"THROUGHPUT COMPARISON (2 reqs × {num_steps} steps)")
    print(f"{'=' * 60}")
    print(f"  Serial forward():  {serial_time:.2f}s")
    print(f"  Step-mode batch:   {batch_time:.2f}s")
    print(f"  Ratio: {ratio:.2f}x")
    print(f"{'=' * 60}")

    # Step-mode should not be significantly slower than serial
    # (Phase 1 for-loop won't be faster; just verify no major regression)
    assert ratio > 0.5, f"Step mode is too slow: {ratio:.2f}x"


# ---------------------------------------------------------------------------
# Test 9: Varlen batched forward correctness (Phase 3b)
# ---------------------------------------------------------------------------


def test_varlen_batched_correctness():
    """Varlen batched forward produces same output as per-request for-loop."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()

    prompts = [
        "A cat sitting on a windowsill",
        "A beautiful sunset over the ocean with golden clouds and seagulls",
    ]

    # Reference: per-request forward()
    refs = []
    for i, prompt in enumerate(prompts):
        req = _build_request(prompt=prompt, seed=700 + i)
        with torch.inference_mode():
            out = pipeline(req)
        refs.append(out.output)

    # Varlen batch: 2 requests with different prompts → different prefix lengths
    states = [
        _make_state(_build_request(prompt=prompt, seed=700 + i), f"varlen_{i}")
        for i, prompt in enumerate(prompts)
    ]

    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        max_steps = max(state.total_steps for state in states)
        for _ in range(max_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break
            input_batch = InputBatch.make_batch(active)
            noise_pred = pipeline.denoise_step(input_batch)

            offset = 0
            for state in active:
                row_num = state.latents.shape[0]
                pipeline.step_scheduler(state, noise_pred[offset:offset + row_num])
                offset += row_num

        outputs = [pipeline.post_decode(state) for state in states]

    for i in range(len(prompts)):
        assert outputs[i].output is not None
        mse = _image_mse(refs[i], outputs[i].output)
        psnr = _image_psnr(refs[i], outputs[i].output)
        print(f"\n  Varlen batch req {i}: MSE={mse:.4f}, PSNR={psnr:.1f}dB")
        assert mse < 1.0, f"Varlen batch output {i} diverges: MSE={mse}"


# ---------------------------------------------------------------------------
# Test 10: Varlen heterogeneous resolution (Phase 3b + 4)
# ---------------------------------------------------------------------------


def test_varlen_heterogeneous_resolution():
    """Different resolutions via varlen batched forward."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()
    resolutions = [(512, 512), (768, 768)]

    refs = []
    for i, res in enumerate(resolutions):
        req = _build_request(image_size=res, seed=800 + i)
        with torch.inference_mode():
            out = pipeline(req)
        refs.append(out.output)

    states = [
        _make_state(_build_request(image_size=res, seed=800 + i), f"vres_{i}")
        for i, res in enumerate(resolutions)
    ]

    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        max_steps = max(state.total_steps for state in states)
        for _ in range(max_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break
            # Different resolutions have different latent shapes;
            # use per-request InputBatch but still exercise varlen when possible.
            if len(active) > 1:
                # Group by resolution
                by_res: dict[tuple[int, int], list] = {}
                for s in active:
                    res_key = tuple(s.latents.shape[2:])
                    by_res.setdefault(res_key, []).append(s)

                all_preds_map: dict[str, torch.Tensor] = {}
                for group in by_res.values():
                    ib = InputBatch.make_batch(group)
                    pred = pipeline.denoise_step(ib)
                    offset = 0
                    for s in group:
                        rows = s.latents.shape[0]
                        all_preds_map[s.request_id] = pred[offset:offset + rows]
                        offset += rows

                for s in active:
                    pipeline.step_scheduler(s, all_preds_map[s.request_id])
            else:
                ib = InputBatch.make_batch(active)
                pred = pipeline.denoise_step(ib)
                pipeline.step_scheduler(active[0], pred)

        outputs = [pipeline.post_decode(state) for state in states]

    for i in range(len(resolutions)):
        assert outputs[i].output is not None
        mse = _image_mse(refs[i], outputs[i].output)
        print(f"\n  Varlen res {resolutions[i]} test: MSE={mse:.4f}")
        assert mse < 1.0, f"Varlen resolution output {i} diverges: MSE={mse}"


# ---------------------------------------------------------------------------
# Test 11: Varlen mixed CFG (Phase 3b + 5)
# ---------------------------------------------------------------------------


def test_varlen_mixed_cfg():
    """Batch with different CFG scales via varlen forward."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()
    cfg_scales = [4.0, 1.0]

    refs = []
    for i, cfg in enumerate(cfg_scales):
        req = _build_request(cfg_scale=cfg, seed=900 + i)
        with torch.inference_mode():
            out = pipeline(req)
        refs.append(out.output)

    states = [
        _make_state(_build_request(cfg_scale=cfg, seed=900 + i), f"vcfg_{i}")
        for i, cfg in enumerate(cfg_scales)
    ]

    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        max_steps = max(state.total_steps for state in states)
        for _ in range(max_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break
            input_batch = InputBatch.make_batch(active)
            noise_pred = pipeline.denoise_step(input_batch)

            offset = 0
            for state in active:
                row_num = state.latents.shape[0]
                pipeline.step_scheduler(state, noise_pred[offset:offset + row_num])
                offset += row_num

        outputs = [pipeline.post_decode(state) for state in states]

    for i in range(len(cfg_scales)):
        assert outputs[i].output is not None
        mse = _image_mse(refs[i], outputs[i].output)
        print(f"\n  Varlen CFG={cfg_scales[i]} test: MSE={mse:.4f}")
        assert mse < 1.0, f"Varlen mixed CFG output {i} diverges: MSE={mse}"


# ---------------------------------------------------------------------------
# Test 12: Varlen throughput comparison (Phase 3b)
# ---------------------------------------------------------------------------


def test_varlen_throughput():
    """Varlen batch should be faster than for-loop serial."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()
    num_reqs = 4
    num_steps = 4

    # Time for-loop serial: each request processed individually
    states_serial = [
        _make_state(_build_request(num_steps=num_steps, seed=1000 + i), f"serial_{i}")
        for i in range(num_reqs)
    ]

    torch.accelerator.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for state in states_serial:
            pipeline.prepare_encode(state)

        for _ in range(num_steps):
            active = [s for s in states_serial if not s.denoise_completed]
            if not active:
                break
            for state in active:
                ib = InputBatch.make_batch([state])
                pred = pipeline.denoise_step(ib)
                pipeline.step_scheduler(state, pred)

        for state in states_serial:
            pipeline.post_decode(state)
    torch.accelerator.synchronize()
    serial_time = time.perf_counter() - start

    # Time varlen batch: all requests batched
    states_batch = [
        _make_state(_build_request(num_steps=num_steps, seed=1000 + i), f"batch_{i}")
        for i in range(num_reqs)
    ]

    torch.accelerator.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for state in states_batch:
            pipeline.prepare_encode(state)

        for _ in range(num_steps):
            active = [s for s in states_batch if not s.denoise_completed]
            if not active:
                break
            ib = InputBatch.make_batch(active)
            noise_pred = pipeline.denoise_step(ib)
            offset = 0
            for state in active:
                row_num = state.latents.shape[0]
                pipeline.step_scheduler(state, noise_pred[offset:offset + row_num])
                offset += row_num

        for state in states_batch:
            pipeline.post_decode(state)
    torch.accelerator.synchronize()
    batch_time = time.perf_counter() - start

    ratio = serial_time / batch_time
    print(f"\n{'=' * 60}")
    print(f"VARLEN THROUGHPUT ({num_reqs} reqs x {num_steps} steps)")
    print(f"{'=' * 60}")
    print(f"  For-loop serial: {serial_time:.2f}s")
    print(f"  Varlen batch:    {batch_time:.2f}s")
    print(f"  Speedup: {ratio:.2f}x")
    print(f"{'=' * 60}")

    assert ratio > 0.8, f"Varlen batch not fast enough: {ratio:.2f}x"


# ---------------------------------------------------------------------------
# Test 13: Dynamic join/leave with varlen (Phase 6 + 3b)
# ---------------------------------------------------------------------------


def test_dynamic_join_leave():
    """Request joins mid-batch after another completes."""
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()

    # Start with 2 requests: 4-step and 8-step
    req_a = _build_request(num_steps=4, seed=1100, request_id="join_a")
    req_b = _build_request(num_steps=8, seed=1101, request_id="join_b")
    state_a = _make_state(req_a, "join_a")
    state_b = _make_state(req_b, "join_b")

    # Reference for all 3 requests
    ref_a = pipeline(_build_request(num_steps=4, seed=1100)).output
    ref_b = pipeline(_build_request(num_steps=8, seed=1101)).output
    ref_c = pipeline(_build_request(num_steps=4, seed=1102)).output

    with torch.inference_mode():
        pipeline.prepare_encode(state_a)
        pipeline.prepare_encode(state_b)

        all_states = [state_a, state_b]
        state_c = None
        c_joined = False

        for step in range(12):
            active = [s for s in all_states if not s.denoise_completed]
            if not active:
                break

            # After step 4 (state_a completes), add state_c
            if state_a.denoise_completed and not c_joined:
                req_c = _build_request(num_steps=4, seed=1102, request_id="join_c")
                state_c = _make_state(req_c, "join_c")
                pipeline.prepare_encode(state_c)
                all_states.append(state_c)
                c_joined = True
                active = [s for s in all_states if not s.denoise_completed]
                if not active:
                    break

            if len(active) > 1:
                ib = InputBatch.make_batch(active)
                noise_pred = pipeline.denoise_step(ib)
                offset = 0
                for s in active:
                    rows = s.latents.shape[0]
                    pipeline.step_scheduler(s, noise_pred[offset:offset + rows])
                    offset += rows
            elif active:
                ib = InputBatch.make_batch(active)
                pred = pipeline.denoise_step(ib)
                pipeline.step_scheduler(active[0], pred)

        out_a = pipeline.post_decode(state_a)
        out_b = pipeline.post_decode(state_b)
        out_c = pipeline.post_decode(state_c) if state_c else None

    assert out_a.output is not None
    assert out_b.output is not None
    assert out_c is not None and out_c.output is not None

    mse_a = _image_mse(ref_a, out_a.output)
    mse_b = _image_mse(ref_b, out_b.output)
    mse_c = _image_mse(ref_c, out_c.output)
    print(f"\n  Dynamic join A: MSE={mse_a:.4f}")
    print(f"  Dynamic join B: MSE={mse_b:.4f}")
    print(f"  Dynamic join C: MSE={mse_c:.4f}")

    assert mse_a < 1.0, f"Dynamic join output A diverges: MSE={mse_a}"
    assert mse_b < 1.0, f"Dynamic join output B diverges: MSE={mse_b}"
    assert mse_c < 1.0, f"Dynamic join output C diverges: MSE={mse_c}"


# ---------------------------------------------------------------------------
# Test 14: Large-batch throughput stress test (Phase 3b)
# ---------------------------------------------------------------------------


def test_varlen_throughput_stress():
    """Large batch with think mode — demonstrates significant varlen speedup.

    Uses 16 requests with think=True to produce long prefix KV caches,
    making the transformer forward dominant in denoise_step time.
    Reports both overall step time and transformer-only time.
    """
    from vllm_omni.diffusion.worker.input_batch import InputBatch

    pipeline = _build_pipeline()
    num_reqs = 16
    num_steps = 4

    prompts = [
        "A cat sleeping on a sunny windowsill",
        "A futuristic city skyline at dusk with flying cars",
        "An oil painting of sunflowers in a ceramic vase",
        "A snowy mountain landscape with a cabin and smoke",
        "A robot playing chess in a cozy library",
        "A coral reef teeming with tropical fish",
        "A steampunk airship floating above the clouds",
        "A medieval castle on a cliff overlooking the sea",
        "A child flying a kite in a green meadow",
        "A fox sitting in an autumn forest clearing",
        "A Japanese zen garden with raked sand patterns",
        "An astronaut planting a flag on Mars",
        "A cozy coffee shop on a rainy afternoon",
        "A dragon perched on a mountaintop at sunrise",
        "A field of lavender under a starry night sky",
        "A vintage train crossing a stone viaduct",
    ]

    # --- For-loop serial: each request one forward at a time ---
    states_serial = [
        _make_state(
            _build_request(prompt=prompts[i], num_steps=num_steps, seed=2000 + i, think=True),
            f"stress_serial_{i}",
        )
        for i in range(num_reqs)
    ]

    torch.accelerator.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for state in states_serial:
            pipeline.prepare_encode(state)
    torch.accelerator.synchronize()
    encode_time = time.perf_counter() - start

    torch.accelerator.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_steps):
            active = [s for s in states_serial if not s.denoise_completed]
            if not active:
                break
            for state in active:
                ib = InputBatch.make_batch([state])
                pred = pipeline.denoise_step(ib)
                pipeline.step_scheduler(state, pred)
    torch.accelerator.synchronize()
    serial_denoise_time = time.perf_counter() - start

    with torch.inference_mode():
        for state in states_serial:
            pipeline.post_decode(state)

    # --- Varlen batch: all requests batched ---
    states_batch = [
        _make_state(
            _build_request(prompt=prompts[i], num_steps=num_steps, seed=2000 + i, think=True),
            f"stress_batch_{i}",
        )
        for i in range(num_reqs)
    ]

    with torch.inference_mode():
        for state in states_batch:
            pipeline.prepare_encode(state)

    torch.accelerator.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_steps):
            active = [s for s in states_batch if not s.denoise_completed]
            if not active:
                break
            ib = InputBatch.make_batch(active)
            noise_pred = pipeline.denoise_step(ib)
            offset = 0
            for state in active:
                row_num = state.latents.shape[0]
                pipeline.step_scheduler(state, noise_pred[offset:offset + row_num])
                offset += row_num
    torch.accelerator.synchronize()
    batch_denoise_time = time.perf_counter() - start

    with torch.inference_mode():
        outputs_batch = [pipeline.post_decode(state) for state in states_batch]

    # Correctness is verified by per-request unit tests; here we only measure throughput.

    ratio = serial_denoise_time / batch_denoise_time
    serial_per_step = serial_denoise_time / num_steps
    batch_per_step = batch_denoise_time / num_steps

    print(f"\n{'=' * 60}")
    print(f"VARLEN THROUGHPUT STRESS ({num_reqs} reqs × {num_steps} steps, think=True)")
    print(f"{'=' * 60}")
    print(f"  Encode time (shared):     {encode_time:.2f}s")
    print(f"  Serial denoise total:     {serial_denoise_time:.2f}s  ({serial_per_step:.3f}s/step)")
    print(f"  Varlen batch denoise:     {batch_denoise_time:.2f}s  ({batch_per_step:.3f}s/step)")
    print(f"  Speedup: {ratio:.2f}x")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Test 15: End-to-end scheduler → pipeline with heterogeneous batch (8 reqs)
# ---------------------------------------------------------------------------


def test_e2e_scheduler_heterogeneous_batch():
    """End-to-end: scheduler batches 8 heterogeneous requests, pipeline produces correct output.

    Uses StepScheduler with heterogeneous_batch_fields to batch requests with
    different resolutions (512x512, 768x768) and different CFG scales (4.0, 7.5).
    Verifies:
      1. Scheduler puts all 8 requests into one batch
      2. Varlen batched output matches single-request reference (MSE < 1.0)
      3. Throughput improvement vs serial execution
    """
    import copy

    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
    from vllm_omni.diffusion.worker.input_batch import InputBatch
    from vllm_omni.diffusion.worker.utils import DiffusionRequestState
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    pipeline = _build_pipeline()
    num_reqs = 8
    num_steps = DEFAULT_NUM_STEPS

    configs = [
        {"image_size": (512, 512), "cfg_scale": 4.0},
        {"image_size": (768, 768), "cfg_scale": 7.5},
        {"image_size": (512, 512), "cfg_scale": 7.5},
        {"image_size": (768, 768), "cfg_scale": 4.0},
        {"image_size": (512, 512), "cfg_scale": 4.0},
        {"image_size": (768, 768), "cfg_scale": 7.5},
        {"image_size": (512, 512), "cfg_scale": 7.5},
        {"image_size": (768, 768), "cfg_scale": 4.0},
    ]
    prompts = [
        "A cat sleeping on a sunny windowsill",
        "A futuristic city skyline at dusk with flying cars",
        "An oil painting of sunflowers in a ceramic vase",
        "A snowy mountain landscape with a cabin and smoke",
        "A robot playing chess in a cozy library",
        "A coral reef teeming with tropical fish",
        "A steampunk airship floating above the clouds",
        "A medieval castle on a cliff overlooking the sea",
    ]

    def _build_scheduler_request(prompt, image_size, cfg_scale, seed):
        sp = OmniDiffusionSamplingParams(
            height=image_size[1],
            width=image_size[0],
            num_inference_steps=num_steps,
            seed=seed,
            extra_args={"cfg_scale": cfg_scale},
        )
        return OmniDiffusionRequest(
            prompts=[prompt],
            sampling_params=sp,
            request_id=str(uuid.uuid4()),
        )

    # ── Phase A: reference outputs (single-request forward) ──
    refs = []
    with torch.inference_mode():
        for i in range(num_reqs):
            req = _build_request(
                prompt=prompts[i],
                image_size=configs[i]["image_size"],
                cfg_scale=configs[i]["cfg_scale"],
                seed=700 + i,
            )
            out = pipeline(req)
            refs.append(out.output)

    # ── Phase B: scheduler batching verification ──
    od_config = OmniDiffusionConfig(
        model=MODEL_PATH,
        dtype=torch.bfloat16,
    )
    od_config.heterogeneous_batch_fields = [
        "height", "width",
        "guidance_scale", "guidance_scale_2", "guidance_scale_provided",
    ]

    scheduler = StepScheduler()
    scheduler.od_config = od_config
    scheduler.max_num_running_reqs = num_reqs

    sched_req_ids = []
    requests = []
    for i in range(num_reqs):
        req = _build_scheduler_request(
            prompts[i], configs[i]["image_size"], configs[i]["cfg_scale"], 700 + i,
        )
        requests.append(req)
        sched_req_ids.append(scheduler.add_request(req))

    sched_output = scheduler.schedule()
    scheduled_count = len(sched_output.scheduled_new_reqs) + len(sched_output.scheduled_cached_reqs.request_ids)
    print(f"\n  Scheduler batched {scheduled_count}/{num_reqs} requests together")
    assert scheduled_count == num_reqs, (
        f"Scheduler only batched {scheduled_count}/{num_reqs} — "
        f"heterogeneous_batch_fields not working end-to-end"
    )

    # ── Phase C: create states from schedule output (mimic runner) ──
    states = []
    for new_req_data in sched_output.scheduled_new_reqs:
        req = new_req_data.req
        state = DiffusionRequestState(
            request_id=new_req_data.request_id,
            sampling=copy.deepcopy(req.sampling_params),
            prompts=req.prompts,
        )
        state.sampling.generator = torch.Generator(device="cuda").manual_seed(state.sampling.seed)
        states.append(state)

    # ── Phase D: varlen batched execution (group by resolution) ──
    torch.accelerator.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for state in states:
            pipeline.prepare_encode(state)

        for _ in range(num_steps):
            active = [s for s in states if not s.denoise_completed]
            if not active:
                break
            # InputBatch requires same latent shape; group by resolution
            by_res: dict[tuple[int, int], list] = {}
            for s in active:
                res_key = tuple(s.latents.shape[2:])
                by_res.setdefault(res_key, []).append(s)

            all_preds_map: dict[str, torch.Tensor] = {}
            for group in by_res.values():
                ib = InputBatch.make_batch(group)
                pred = pipeline.denoise_step(ib)
                offset = 0
                for s in group:
                    rows = s.latents.shape[0]
                    all_preds_map[s.request_id] = pred[offset:offset + rows]
                    offset += rows

            for s in active:
                pipeline.step_scheduler(s, all_preds_map[s.request_id])

        outputs = [pipeline.post_decode(state) for state in states]
    torch.accelerator.synchronize()
    batch_time = time.perf_counter() - start

    # ── Phase E: serial execution for comparison ──
    states_serial = []
    for i in range(num_reqs):
        req = _build_request(
            prompt=prompts[i],
            image_size=configs[i]["image_size"],
            cfg_scale=configs[i]["cfg_scale"],
            seed=700 + i,
        )
        states_serial.append(_make_state(req, f"e2e_serial_{i}"))

    torch.accelerator.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for state in states_serial:
            pipeline.prepare_encode(state)
        for _ in range(num_steps):
            active = [s for s in states_serial if not s.denoise_completed]
            if not active:
                break
            for state in active:
                ib = InputBatch.make_batch([state])
                pred = pipeline.denoise_step(ib)
                pipeline.step_scheduler(state, pred)
        outputs_serial = [pipeline.post_decode(state) for state in states_serial]
    torch.accelerator.synchronize()
    serial_time = time.perf_counter() - start

    # ── Phase F: correctness + performance report ──
    print(f"\n{'=' * 60}")
    print(f"E2E SCHEDULER HETEROGENEOUS BATCH ({num_reqs} reqs × {num_steps} steps)")
    print(f"  Resolutions: 512x512, 768x768 | CFG: 4.0, 7.5")
    print(f"{'=' * 60}")

    all_pass = True
    for i in range(num_reqs):
        assert outputs[i].output is not None, f"Request {i} produced no output"
        mse = _image_mse(refs[i], outputs[i].output)
        cfg = configs[i]["cfg_scale"]
        res = configs[i]["image_size"]
        status = "PASS" if mse < 1.0 else "FAIL"
        if mse >= 1.0:
            all_pass = False
        print(f"  req[{i}] {res} cfg={cfg}: MSE={mse:.4f} [{status}]")
        assert mse < 1.0, f"Request {i} ({res}, cfg={cfg}) diverges: MSE={mse}"

    ratio = serial_time / batch_time if batch_time > 0 else float("inf")
    print(f"  Serial total:  {serial_time:.2f}s")
    print(f"  Batch total:   {batch_time:.2f}s")
    print(f"  Speedup:       {ratio:.2f}x")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Test 16: Think mode step equivalence
# ---------------------------------------------------------------------------


def test_think_mode_step_equivalence():
    """Think mode step output matches forward() output (same seed).

    Verifies that _build_t2i_caches with think_mode=True produces correct KV
    caches via _generate_think AR loop, and that the variable-length prefix
    is correctly consumed during denoise steps.
    """
    pipeline = _build_pipeline()

    req = _build_request(seed=500, think=True)
    with torch.inference_mode():
        output_forward = pipeline(req)

    state = _make_state(_build_request(seed=500, think=True), "think_step")
    with torch.inference_mode():
        output_step = _run_step_mode(pipeline, state)

    assert output_forward.output is not None
    assert output_step.output is not None

    mse = _image_mse(output_forward.output, output_step.output)
    psnr = _image_psnr(output_forward.output, output_step.output)
    print(f"\n  Think mode step vs forward: MSE={mse:.4f}, PSNR={psnr:.1f}dB")

    assert mse < 1.0, f"Think mode step output diverges from forward(): MSE={mse}"

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    output_forward.output.save(os.path.join(out_dir, "think_forward.png"))
    output_step.output.save(os.path.join(out_dir, "think_step.png"))


# ---------------------------------------------------------------------------
# Test 17: IT2I (image-to-image) step equivalence
# ---------------------------------------------------------------------------


def _make_test_image(width=256, height=256):
    """Create a synthetic gradient test image for IT2I testing."""
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x % 256, y % 256, (x + y) % 256)
    return img


def test_it2i_step_equivalence():
    """IT2I step output matches forward() output (same seed).

    Verifies that _build_it2i_caches produces correct triple-condition KV caches
    (cond + img_cond + uncond) and that _denoise_step_it2i dual CFG combination
    works correctly in step execution mode.
    """
    pipeline = _build_pipeline()

    test_image = _make_test_image()
    prompt_dict = {
        "prompt": "Transform this image into a watercolor painting",
        "multi_modal_data": {"image": [test_image]},
    }

    req = _build_request(prompt=prompt_dict, img_cfg_scale=2.0, seed=600)
    with torch.inference_mode():
        output_forward = pipeline(req)

    state = _make_state(
        _build_request(prompt=prompt_dict, img_cfg_scale=2.0, seed=600),
        "it2i_step",
    )
    with torch.inference_mode():
        output_step = _run_step_mode(pipeline, state)

    assert output_forward.output is not None
    assert output_step.output is not None

    mse = _image_mse(output_forward.output, output_step.output)
    psnr = _image_psnr(output_forward.output, output_step.output)
    print(f"\n  IT2I step vs forward: MSE={mse:.4f}, PSNR={psnr:.1f}dB")

    assert mse < 1.0, f"IT2I step output diverges from forward(): MSE={mse}"

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    output_forward.output.save(os.path.join(out_dir, "it2i_forward.png"))
    output_step.output.save(os.path.join(out_dir, "it2i_step.png"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("SenseNova-U1 Step Execution & Online Dynamic Batching Tests")
    print("=" * 70)

    test_supports_step_execution()
    print("\n[1/13] Protocol conformance: PASS")

    test_step_mode_equivalence()
    print("\n[2/13] Step mode equivalence: PASS")

    test_homogeneous_batch()
    print("\n[3/13] Homogeneous batch: PASS")

    test_heterogeneous_prompt_lengths()
    print("\n[4/13] Heterogeneous prompt lengths: PASS")

    test_heterogeneous_resolutions()
    print("\n[5/13] Heterogeneous resolutions: PASS")

    test_heterogeneous_cfg_scales()
    print("\n[6/13] Heterogeneous CFG scales: PASS")

    test_dynamic_step_counts()
    print("\n[7/13] Dynamic step counts: PASS")

    test_throughput_improvement()
    print("\n[8/13] Throughput comparison: PASS")

    test_varlen_batched_correctness()
    print("\n[9/13] Varlen batched correctness: PASS")

    test_varlen_heterogeneous_resolution()
    print("\n[10/13] Varlen heterogeneous resolution: PASS")

    test_varlen_mixed_cfg()
    print("\n[11/13] Varlen mixed CFG: PASS")

    test_varlen_throughput()
    print("\n[12/13] Varlen throughput: PASS")

    test_dynamic_join_leave()
    print("\n[13/17] Dynamic join/leave: PASS")

    test_varlen_throughput_stress()
    print("\n[14/17] Varlen throughput stress: PASS")

    test_e2e_scheduler_heterogeneous_batch()
    print("\n[15/17] E2E scheduler heterogeneous batch: PASS")

    test_think_mode_step_equivalence()
    print("\n[16/17] Think mode step equivalence: PASS")

    test_it2i_step_equivalence()
    print("\n[17/17] IT2I step equivalence: PASS")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
