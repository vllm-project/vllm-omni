# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real GPU image-stage transport, engine, worker and output-consumer contracts.

No engine/worker/transport/model mocks. A real stage subprocess loads the local
image checkpoint. Genuine thinker output types carry declared precomputed
hidden states: this establishes image-stage E2E, not text/thinker semantics.
"""

import asyncio
import os
import socket
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors import safe_open
from vllm.utils import random_uuid

from tests.diffusion.models.ming_flash_omni.test_pipeline_ming_imagegen import stage_output
from tests.helpers.mark import hardware_test
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.diffusion.data import (
    AttentionConfig,
    AttentionSpec,
    OmniDiffusionConfig,
    is_diffusion_request_started_output,
)
from vllm_omni.diffusion.stage_diffusion_client import create_diffusion_client
from vllm_omni.engine.stage_init_utils import extract_legacy_stage_metadata
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.ming_flash_omni.pipeline import MING_FLASH_OMNI_IMAGE_PIPELINE
from vllm_omni.model_executor.stage_input_processors.ming_flash_omni import thinker2imagegen
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.slow]


def stage_prompt(root, request_id, reference=None, *, condition_id="A"):
    """Reuse the actual stage output/bridge layout and checkpoint hidden width."""
    with safe_open(root / "mlp" / "model.safetensors", framework="pt") as weights:
        hidden_size = weights.get_slice("proj_in.weight").get_shape()[1]
    hidden = torch.sin(torch.arange(257 * hidden_size, dtype=torch.float32).reshape(257, hidden_size) / 1000)
    hidden += 0.25 if condition_id == "B" else 0
    stages = [stage_output(request_id, hidden), stage_output(request_id + "__cfg_text", -hidden)]
    original: dict[str, object] = {"prompt": "paint a landscape", "modalities": ["image"]}
    if reference is not None:
        original["multi_modal_data"] = {"image": reference}
    return thinker2imagegen(stages, prompt=original)[0]


async def receive_finished(client, request_ids):
    """Consume the same public queue as the real orchestrator, with a bounded wait."""
    pending = set(request_ids)
    outputs, events = {}, []

    async def consume():
        while pending:
            output = client.get_diffusion_output_nowait()
            if output is None:
                await asyncio.sleep(0.01)
                continue
            assert isinstance(output, OmniRequestOutput)
            assert output.error is None, output.error
            if output.request_id not in pending:
                continue
            if is_diffusion_request_started_output(output):
                events.append((output.request_id, "started"))
            elif output.finished:
                events.append((output.request_id, "finished"))
                pending.remove(output.request_id)
                outputs[output.request_id] = output
        return outputs, events

    return await asyncio.wait_for(consume(), timeout=120)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("step_execution", [False, True], ids=["b1", "b2"])
def test_checkpoint_stage_process_preserves_requests_and_public_outputs(dtype, step_execution):
    """Regression: stage serialization/engine execution must not drop or mix rows.

    Input source: production Ming pipeline/deploy merge -> StageMetadata/client;
    genuine thinker outputs -> bridge -> ZMQ -> real engine/scheduler -> MP GPU
    worker/runner -> checkpoint DiT/VAE -> public output -> ZMQ/orchestrator queue.
    Why valid: legal 256x256 requests with independent seeds, precomputed hidden
    states, optional reference, real profile capacity four and B1/B2 selection.
    Expected source: public latent/PIL contracts, terminal lifecycle, deterministic
    per-request isolation and configured concurrent capacity. Independent Euler
    and source mutations in companion suites pin algorithmic correctness.
    This explicitly does not establish missing-thinker prompt/image semantics.
    """
    model = os.environ.get("VLLM_MING_TEST_MODEL")
    if not model:
        pytest.skip("Set VLLM_MING_TEST_MODEL to execute real stage transport")
    assert model is not None
    root = Path(model)
    profile = "ming_flash_omni_image_stepwise.yaml" if step_execution else "ming_flash_omni_image_high_throughput.yaml"
    repo = Path(__file__).resolve().parents[4]
    stages = merge_pipeline_deploy(
        MING_FLASH_OMNI_IMAGE_PIPELINE, load_deploy_config(repo / "vllm_omni" / "deploy" / profile)
    )
    image_stage = next(s for s in stages if s.stage_id == 1)
    metadata = extract_legacy_stage_metadata(image_stage.to_omegaconf())
    assert metadata.stage_type == "diffusion" and metadata.engine_input_source == [0]
    assert metadata.custom_process_input_func is thinker2imagegen
    engine_args = image_stage.to_omegaconf().engine_args
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = OmniDiffusionConfig(
        model=str(root),
        model_class_name="MingImagePipeline",
        dtype=dtype,
        num_gpus=1,
        master_port=port,
        max_num_seqs=engine_args.max_num_seqs,
        step_execution=step_execution,
        request_batch_max_wait_ms=20,
        distributed_executor_backend="mp",
        enforce_eager=True,
        diffusion_attention_config=AttentionConfig(default=AttentionSpec(backend="TORCH_SDPA")),
    )
    assert config.max_num_seqs == 4
    client = create_diffusion_client(str(root), config, metadata, stage_init_timeout=240, use_inline=False)
    try:
        assert client._proc_manager.proc.is_alive()

        async def scenario():
            def sampling(seed, output_type="latent"):
                return OmniDiffusionSamplingParams(
                    height=256,
                    width=256,
                    num_inference_steps=3,
                    guidance_scale=2,
                    seed=seed,
                    output_type=output_type,
                    # The orchestrator opts in to the started lifecycle event.
                    emit_request_lifecycle=True,
                    extra_args={"cfg_truncation": 0.3},
                )

            single = {}
            for rid, seed in [("A", 11), ("B", 22)]:
                # Public image-generation callers allocate a fresh UUID per
                # submission, including repeats with identical conditions/seeds.
                request_id = f"img_gen-{random_uuid()}"
                await client.add_request_async(
                    request_id, stage_prompt(root, request_id, condition_id=rid), sampling(seed)
                )
                single[rid] = (await receive_finished(client, [request_id]))[0][request_id]
            request_ids = {rid: f"img_gen-{random_uuid()}" for rid in ("A", "B")}
            for rid, seed in [("A", 11), ("B", 22)]:
                await client.add_request_async(
                    request_ids[rid], stage_prompt(root, request_ids[rid], condition_id=rid), sampling(seed)
                )
            batched, events = await receive_finished(client, request_ids.values())
            first_finish = next(i for i, (_, kind) in enumerate(events) if kind == "finished")
            assert {rid for rid, kind in events[:first_finish] if kind == "started"} == set(request_ids.values())
            for rid in ("A", "B"):
                output = batched[request_ids[rid]]
                assert output.finished and output.final_output_type == "latents" and output.images == []
                assert output.latents.shape == (1, 16, 32, 32)
                assert torch.isfinite(output.latents).all()
                torch.testing.assert_close(output.latents, single[rid].latents, rtol=0.03, atol=0.03)
            assert not torch.equal(batched[request_ids["A"]].latents, batched[request_ids["B"]].latents)
            reference_id = f"img_edit-{random_uuid()}"
            await client.add_request_async(
                reference_id,
                stage_prompt(root, reference_id, Image.new("RGB", (256, 256), "red")),
                sampling(33, "pil"),
            )
            image = (await receive_finished(client, [reference_id]))[0][reference_id]
            assert image.finished and image.final_output_type == "image"
            assert len(image.images) == 1 and isinstance(image.images[0], Image.Image)
            assert image.images[0].size == (256, 256)

        asyncio.run(scenario())
    finally:
        client.shutdown()
    assert not client._proc_manager.proc.is_alive()
