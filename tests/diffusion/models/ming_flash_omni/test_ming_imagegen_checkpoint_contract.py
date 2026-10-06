# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in checkpoint verification, using one real worker and production schedulers.

Set VLLM_MING_TEST_MODEL to a local image-stage checkpoint. The thinker is not
loaded: declared, nonzero hidden states enter through its genuine output types
and stage bridge. This verifies diffusion integration, not prompt semantics or
image quality. No checkpoint download or model/runner/collective mock is used.
"""

import os
import socket
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors import safe_open

from tests.diffusion.models.ming_flash_omni.test_pipeline_ming_imagegen import stage_output
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.ming_flash_omni.pipeline_ming_imagegen import (
    get_ming_image_post_process_func,
    get_ming_image_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import RequestScheduler, StepScheduler
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import thinker2imagegen

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.slow]


@pytest.fixture(scope="module", params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def checkpoint_worker(request):
    model = os.environ.get("VLLM_MING_TEST_MODEL")
    if not model:
        pytest.skip("Set VLLM_MING_TEST_MODEL to execute the checkpoint contract")
    assert model is not None
    root = Path(model)
    for name in ("transformer", "connector", "mlp", "vae", "scheduler"):
        assert (root / name).is_dir(), f"Missing checkpoint component: {name}"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = OmniDiffusionConfig(
        model_class_name="MingImagePipeline",
        model=str(root),
        dtype=request.param,
        num_gpus=1,
        master_port=port,
        max_num_seqs=2,
        step_execution=True,
        enforce_eager=True,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    worker = DiffusionWorker(local_rank=0, rank=0, od_config=config)
    try:
        yield worker
    finally:
        worker.shutdown()


def checkpoint_request(worker, request_id, *, seed, reference=None, output_type="latent", latents=None, output_count=1):
    """Input source: MLP checkpoint input shape and actual thinker bridge contract.

    The published image-query contract has 256 query tokens plus an end token.
    A deterministic sinusoid provides finite nonzero precomputed hidden states;
    it is not claimed to be an actual thinker's semantic representation.
    """
    root = Path(worker.od_config.model)
    with safe_open(root / "mlp" / "model.safetensors", framework="pt") as weights:
        hidden_size = weights.get_slice("proj_in.weight").get_shape()[1]
    hidden = torch.sin(torch.arange(257 * hidden_size, dtype=torch.float32).reshape(257, hidden_size) / 1000)
    hidden += 0.25 if request_id == "B" else 0.0
    stages = [stage_output(request_id, hidden), stage_output(request_id + "__cfg_text", -hidden)]
    original: dict[str, object] = {"prompt": "paint a landscape", "modalities": ["image"]}
    if reference is not None:
        original["multi_modal_data"] = {"image": reference}
    prompt = thinker2imagegen(stages, prompt=original)[0]
    sampling = OmniDiffusionSamplingParams(
        height=256,
        width=256,
        num_inference_steps=3,
        guidance_scale=2.0,
        seed=seed,
        output_type=output_type,
        num_outputs_per_prompt=output_count,
        latents=latents,
        extra_args={"cfg_truncation": 0.3},
    )
    return get_ming_image_pre_process_func(worker.od_config)(
        OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=sampling,
            request_id=request_id,
            # Multiple images use the full-forward API contract, not STEP_BATCH.
            use_step_execution=output_count == 1,
        )
    )


def request_wave(worker, requests):
    """Actual RequestScheduler -> SchedulerOutput -> Worker -> Runner -> DiT."""
    scheduler = RequestScheduler()
    scheduler.initialize(worker.od_config)
    for request in requests:
        scheduler.add_request(request)
    scheduled = scheduler.schedule()
    assert scheduled.scheduled_request_ids == [r.request_id for r in requests]
    result = worker.execute_model_batch(scheduled, worker.od_config)
    assert result.request_ids == scheduled.scheduled_request_ids
    scheduler.update_from_output(scheduled, result)
    assert not scheduler.has_requests()
    for row in result.runner_outputs:
        assert row.finished and row.result is not None and row.result.error is None
        assert torch.isfinite(row.result.output).all()
    return {row.request_id: row.result for row in result.runner_outputs}


def step_session(worker, requests):
    """Actual StepScheduler admission/cached transitions, retirement and consumer."""
    scheduler = StepScheduler()
    scheduler.initialize(worker.od_config)
    scheduler.add_request(requests[0])
    outputs, waves = {}, []
    for tick in range(3 + len(requests) - 1):
        if tick == 1 and len(requests) == 2:
            scheduler.add_request(requests[1])
        scheduled = scheduler.schedule()
        waves.append(scheduled.scheduled_request_ids)
        result = worker.execute_stepwise(scheduled)
        assert result.request_ids == scheduled.scheduled_request_ids
        for row in result.runner_outputs:
            if row.result is not None:
                assert row.result.error is None
            if row.finished:
                assert row.result is not None and torch.isfinite(row.result.output).all()
                outputs[row.request_id] = row.result
        scheduler.update_from_output(scheduled, result)
    assert waves[0] == [requests[0].request_id]
    if len(requests) == 2:
        assert set(waves[1]) == {r.request_id for r in requests}
        assert waves[-1] == [requests[1].request_id]
    assert set(outputs) == {r.request_id for r in requests}
    assert not scheduler.has_requests() and not worker.model_runner.state_cache
    return outputs


def checkpoint_pair(worker):
    return [checkpoint_request(worker, "A", seed=11), checkpoint_request(worker, "B", seed=22)]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_checkpoint_wave_matches_single_request(checkpoint_worker):
    """Input: genuine bridge requests -> actual RequestScheduler/Worker/Runner.

    Expected source: deterministic singleton/batch row isolation with the same
    seeds and model. Regression: batch accessor failures, RNG/order drift.
    The original precision bound is retained after the compute-shape fix.
    It must not be loosened merely to produce a passing test result.
    """
    worker = checkpoint_worker
    batch = request_wave(worker, checkpoint_pair(worker))
    assert batch["A"].output.shape == batch["B"].output.shape == (1, 16, 32, 32)
    assert not torch.equal(batch["A"].output, batch["B"].output)
    for request in checkpoint_pair(worker):
        single = request_wave(worker, [request])[request.request_id]
        torch.testing.assert_close(batch[request.request_id].output, single.output, rtol=0.03, atol=0.03)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_checkpoint_late_admission_and_retirement(checkpoint_worker):
    """Input: actual StepScheduler admits B after A's first tick.

    Expected source: scheduler tick/retirement contract and shared B1/B2
    algorithm. Regression: reused scheduler, stale row/timestep, early completion.
    Euler/CFG correctness has a separate independent analytic oracle; B1/B2
    agreement alone is not taken as proof of algorithmic correctness.
    """
    worker = checkpoint_worker
    batch = request_wave(worker, checkpoint_pair(worker))
    steps = step_session(worker, checkpoint_pair(worker))
    for request_id, output in steps.items():
        torch.testing.assert_close(output.output, batch[request_id].output, rtol=0.03, atol=0.03)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_checkpoint_optional_latents_do_not_affect_peer(checkpoint_worker):
    """Input: API optional tensor/None latents -> real request wave.

    Expected source: request independence. Changing A's initial latent must not
    change B with the same batch geometry and seed. Regression: tensor+None
    collation failure or cross-request state/condition leakage.
    """
    worker = checkpoint_worker
    batch = request_wave(worker, checkpoint_pair(worker))
    explicit = torch.full((1, 16, 32, 32), 0.25)
    mixed = [checkpoint_request(worker, "A", seed=11, latents=explicit), checkpoint_request(worker, "B", seed=22)]
    mixed_output = request_wave(worker, mixed)
    torch.testing.assert_close(mixed_output["B"].output, batch["B"].output)
    assert not torch.equal(mixed_output["A"].output, batch["A"].output)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("output_count", [2, 3])
def test_checkpoint_multiple_outputs_preserve_rows_across_dit_chunks(checkpoint_worker, output_count):
    """Scenario: B1 has more image rows than the request-capacity model chunk.

    Input source: public num_outputs_per_prompt=2/3, actual RequestScheduler/worker.
    Why valid: B1 supports multiple images for each independently seeded request.
    Expected source: per-request sample identity and output count/order must not
    depend on admission alongside a peer. No new numeric bound is introduced.
    Regression: dropping rows beyond capacity, routing padding as real output,
    resetting a request's random generator or changing its model compute shape.
    Three images also place A's last row and B's first row in the same chunk,
    unlike singleton execution, which must preserve both requests' sample order.
    """
    worker = checkpoint_worker

    def requests():
        return [
            checkpoint_request(worker, rid, seed=seed, output_count=output_count)
            for rid, seed in [("A", 11), ("B", 22)]
        ]

    batch = request_wave(worker, requests())
    for request in requests():
        single = request_wave(worker, [request])[request.request_id]
        assert batch[request.request_id].output.shape == single.output.shape == (output_count, 16, 32, 32)
        torch.testing.assert_close(batch[request.request_id].output, single.output, rtol=0.03, atol=0.03)


def checkpoint_engine(worker):
    # Isolate process/transport construction only: use the genuine postprocessor
    # and DiffusionEngine.postprocess_output -> formatter -> OmniRequestOutput.
    engine = object.__new__(DiffusionEngine)
    engine.od_config = worker.od_config
    engine.post_process_func = get_ming_image_post_process_func(worker.od_config)
    engine._post_process_accepts_sampling_params = True
    return engine


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_checkpoint_reference_trajectory_and_image_consumer(checkpoint_worker):
    """Input: genuine bridge with PIL reference -> supported singleton schedule.

    Expected source: reference as an extra frame, shared full B1/B2 schedule and
    the public 256x256 PIL image contract. Regression: strength initialization,
    wrong reference/frame layout, duplicate normalization or broken output routing.
    This checks actual VAE decode without claiming semantic image quality.
    """
    worker = checkpoint_worker
    image = Image.new("RGB", (256, 256), "red")
    ref_wave = request_wave(worker, [checkpoint_request(worker, "R", seed=33, reference=image)])["R"]
    ref_step = step_session(worker, [checkpoint_request(worker, "R", seed=33, reference=image)])["R"]
    torch.testing.assert_close(ref_wave.output, ref_step.output, rtol=0.03, atol=0.03)
    decoded = request_wave(worker, [checkpoint_request(worker, "R", seed=33, reference=image, output_type="pil")])["R"]
    (image_api,) = checkpoint_engine(worker).postprocess_output(
        checkpoint_request(worker, "R", seed=33, reference=image, output_type="pil"), decoded
    )
    assert image_api.request_id == "R" and image_api.finished
    assert len(image_api.images) == 1 and image_api.images[0].size == (256, 256)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_checkpoint_runner_latents_reach_public_consumer(checkpoint_worker):
    """Input: real checkpoint StepScheduler/Worker results -> engine/formatter.

    Expected source: requested latent API payload, preserved identity and terminal
    lifecycle. Regression: latent converted to PIL or completion loses state.
    """
    worker = checkpoint_worker
    request = checkpoint_request(worker, "A", seed=11)
    output = step_session(worker, [request])["A"]
    (api,) = checkpoint_engine(worker).postprocess_output(request, output)
    assert api.request_id == "A" and api.finished
    assert api.images == [] and api.final_output_type == "latents"
    torch.testing.assert_close(api.latents, output.output)
