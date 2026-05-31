# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import vllm_omni.diffusion.worker.diffusion_model_runner as model_runner_module
import vllm_omni.diffusion.worker.diffusion_submodule_runner as submodule_runner_module
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput, NewRequestData
from vllm_omni.diffusion.stage_submodule_client import StageSubModuleClient
from vllm_omni.diffusion.stage_submodule_proc import StageSubModuleProc
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_submodule_runner import DiffusionSubmoduleRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.qwen_image import (
    denoise_to_decode,
    encode_to_denoise,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@contextmanager
def _noop_forward_context(*args, **kwargs):
    del args, kwargs
    yield


class _StepContractPipeline:
    supports_step_execution = True
    stage = "denoise"
    produces_intermediate_stage_output = True
    interrupt = False
    device = torch.device("cpu")

    def __init__(self, num_steps: int):
        self.num_steps = num_steps
        self.post_decode_calls = 0

    def prepare_encode(self, state):
        state.timesteps = torch.arange(self.num_steps)
        state.latents = torch.zeros((1, 1))
        state.prompt_embeds = torch.zeros((1, 1, 1))
        state.prompt_embeds_mask = torch.ones((1, 1), dtype=torch.long)
        state.img_shapes = [[(1, 1, 1)]]
        state.txt_seq_lens = [1]
        return state

    def denoise_step(self, input_batch):
        return torch.ones_like(input_batch.latents)

    def step_scheduler(self, state, noise_pred):
        state.latents = state.latents + noise_pred
        state.step_index += 1

    def post_decode(self, state):
        del state
        self.post_decode_calls += 1
        return DiffusionOutput(output="decoded")

    def post_intermediate_output(self, state):
        return DiffusionOutput(
            output=None,
            multimodal_output={
                "latents": state.latents.clone(),
                "step_index": state.step_index,
            },
        )


class _OptimizedStagePipeline(_StepContractPipeline):
    def __init__(self, num_steps: int):
        super().__init__(num_steps)
        self.execute_stage_model_calls = 0

    def execute_stage_model(self, req):
        del req
        self.execute_stage_model_calls += 1
        return DiffusionOutput(
            output=None,
            multimodal_output={
                "latents": torch.full((1, 1), 7.0),
                "step_index": 7,
            },
        )


class _SubmodulePipeline:
    context = torch.ones((1, 2, 4))
    image = object()

    def __init__(self, empty: bool = False):
        self.empty = empty
        self.calls: list[tuple[str, list[OmniDiffusionRequest]]] = []

    def execute_encode(self, requests: list[OmniDiffusionRequest]) -> list[dict]:
        self.calls.append(("encode", requests))
        return [] if self.empty else [{"req_id": "req-1", "context": self.context, "metadata": {}, "unused": None}]

    def execute_decode(self, requests: list[OmniDiffusionRequest]) -> list[dict]:
        self.calls.append(("decode", requests))
        return [] if self.empty else [{"req_id": "req-1", "image": self.image, "output_type": "pil", "metadata": {}}]


def _make_runner() -> DiffusionModelRunner:
    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = object()
    runner.device = torch.device("cpu")
    runner.cache_backend = None
    runner.offload_backend = None
    runner.prompt_embed_cache = None
    runner.od_config = SimpleNamespace(
        cache_backend="none",
        enable_cache_dit_summary=False,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )
    runner.kv_transfer_manager = SimpleNamespace(
        receive_multi_kv_cache_distributed=lambda req, cfg_kv_collect_func=None, target_device=None: None,
    )
    return runner


def _make_diffusion_request(num_steps: int = 1) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompts=[{"prompt": "a prompt", "additional_information": {}}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=num_steps),
        request_id="req-1",
    )


def test_submodule_model_stage_survives_config_filtering():
    cfg = OmniDiffusionConfig.from_kwargs(
        model="Qwen/Qwen-Image",
        model_class_name="QwenImagePipeline",
        model_stage="encode",
        ignored_unknown_field=True,
    )

    assert cfg.model_stage == "encode"
    assert not hasattr(cfg, "ignored_unknown_field")


def test_qwen_image_stage_components_are_model_specific():
    all_components = {"scheduler", "text_encoder", "tokenizer", "vae", "transformer"}

    assert QwenImagePipeline._STAGE_COMPONENTS["diffusion"] == all_components
    assert QwenImagePipeline._STAGE_COMPONENTS["encode"] == {"scheduler", "text_encoder", "tokenizer"}
    assert QwenImagePipeline._STAGE_COMPONENTS["denoise"] == {"scheduler", "transformer"}


def test_qwen_image_denoise_dummy_run_request_uses_intermediate_payload():
    od_config = SimpleNamespace(
        model="/path/that/does/not/exist",
        model_stage="denoise",
        dtype=torch.bfloat16,
        tf_model_config={"in_channels": 64, "joint_attention_dim": 3584},
    )

    request = QwenImagePipeline.build_dummy_run_request(
        od_config,
        height=1024,
        width=1024,
        num_inference_steps=1,
    )

    assert request is not None
    assert request.request_id == "dummy_req_id"
    info = request.prompts[0]["additional_information"]
    assert info["context"].shape == (1, 1, 3584)
    assert info["context_mask"].shape == (1, 1)
    assert info["latents"].shape == (1, 4096, 64)
    assert info["img_shapes"] == [[(1, 64, 64)]]
    assert info["num_inference_steps"] == 1
    assert request.sampling_params.num_inference_steps == 1


def test_qwen_image_diffusion_dummy_run_uses_safe_engine_prompt():
    od_config = SimpleNamespace(model_stage="diffusion")

    request = QwenImagePipeline.build_dummy_run_request(
        od_config,
        height=1024,
        width=1024,
        num_inference_steps=1,
    )

    assert request is not None
    assert request.prompts == [{"prompt": "dummy run"}]
    assert request.request_id == "dummy_req_id"
    assert request.sampling_params.num_inference_steps == 2
    assert request.sampling_params.true_cfg_scale == 1.0


def test_qwen_image_encode_to_denoise_payload():
    latents = torch.zeros((1, 4096, 64), dtype=torch.bfloat16)
    context_null = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
    image_latents = torch.ones((1, 4096, 64), dtype=torch.bfloat16)
    source_outputs = [
        SimpleNamespace(
            multimodal_output={
                "context": torch.ones((1, 4, 8), dtype=torch.bfloat16),
                "context_mask": torch.ones((1, 4), dtype=torch.long),
                "context_null": context_null,
                "latents": latents,
                "timesteps": torch.arange(2),
                "height": 1024,
                "width": 1024,
                "img_shapes": [[(1, 64, 64)]],
                "true_cfg_scale": 4.0,
                "do_true_cfg": True,
                "image_latents": image_latents,
            }
        )
    ]

    prompts = encode_to_denoise(source_outputs)

    assert len(prompts) == 1
    info = prompts[0]["additional_information"]
    assert info["context"].shape == (1, 4, 8)
    assert info["latents"] is latents
    assert info["height"] == 1024
    assert info["width"] == 1024
    assert info["context_null"] is context_null
    assert info["image_latents"] is image_latents
    assert info["true_cfg_scale"] == 4.0
    assert info["do_true_cfg"] is True


def test_qwen_image_denoise_to_decode_payload():
    latents = torch.ones((1, 4, 8))
    prompts = denoise_to_decode([SimpleNamespace(multimodal_output={"latents": latents, "height": 512, "width": 768})])

    assert prompts[0]["additional_information"] == {
        "latents": latents,
        "height": 512,
        "width": 768,
        "output_type": "pil",
    }


def test_qwen_image_denoise_to_decode_payload_requires_latents():
    source_outputs = [
        SimpleNamespace(
            multimodal_output={
                "height": 1024,
                "width": 1024,
            }
        )
    ]

    with pytest.raises(RuntimeError, match="missing required key 'latents'"):
        denoise_to_decode(source_outputs)


def test_stage_submodule_client_batch_contract():
    client = object.__new__(StageSubModuleClient)
    sampling_params = OmniDiffusionSamplingParams(seed=123)
    kv_sender_info = {1: {"addr": "stage-1"}}
    calls = []

    async def fake_add_request_async(request_id, prompt, sampling_params_arg, kv_sender_info=None):
        calls.append((request_id, prompt, sampling_params_arg, kv_sender_info))

    client.add_request_async = fake_add_request_async

    asyncio.run(
        StageSubModuleClient.add_batch_request_async(
            client,
            "req-1",
            [{"prompt": "single"}],
            sampling_params,
            kv_sender_info=kv_sender_info,
        )
    )

    assert calls == [("req-1", {"prompt": "single"}, sampling_params, kv_sender_info)]

    client._output_queue = asyncio.Queue()
    client.check_health = lambda: None
    asyncio.run(
        StageSubModuleClient.add_batch_request_async(
            client,
            "req-batch",
            [{"prompt": "first"}, {"prompt": "second"}],
            sampling_params,
        )
    )

    output = client._output_queue.get_nowait()
    assert output.request_id == "req-batch"
    assert output.error == "StageSubModuleClient only supports single-prompt batches, got 2 prompts."


def test_diffusion_engine_preserves_intermediate_multimodal_output():
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(
        model_stage="denoise",
        enable_cpu_offload=False,
        model_class_name="QwenImagePipeline",
    )
    engine.pre_process_func = None
    engine.post_process_func = None
    engine._post_process_accepts_sampling_params = False
    request = OmniDiffusionRequest(
        prompts=[{"prompt": "a cup of coffee"}],
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=1),
        request_id="req-1",
    )
    output = DiffusionOutput(
        output=None,
        multimodal_output={
            "latents": torch.ones((1, 4, 8), dtype=torch.float32),
            "height": 1024,
            "width": 1024,
        },
    )

    async def _async_response(req):
        del req
        return output

    async def _noop_start_loop():
        return None

    engine.async_add_req_and_wait_for_response = _async_response
    engine._check_and_start_background_loop = _noop_start_loop

    request_outputs = asyncio.run(DiffusionEngine.step(engine, request))

    assert len(request_outputs) == 1
    assert request_outputs[0].request_id == "req-1"
    assert request_outputs[0].final_output_type == "stage_denoise"
    assert torch.equal(request_outputs[0].multimodal_output["latents"], output.multimodal_output["latents"])


def test_execute_model_uses_optimized_stage_model_hook(monkeypatch):
    runner = _make_runner()
    runner.pipeline = _OptimizedStagePipeline(num_steps=2)
    runner._record_peak_memory = lambda output: None
    req = _make_diffusion_request(num_steps=2)

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)
    monkeypatch.setattr(model_runner_module, "cache_summary", lambda pipeline, details: None)
    monkeypatch.setattr(model_runner_module.current_omni_platform, "reset_peak_memory_stats", lambda: None)

    output = DiffusionModelRunner.execute_model(runner, req)

    assert output.output is None
    assert output.multimodal_output["step_index"] == 7
    assert torch.equal(output.multimodal_output["latents"], torch.full((1, 1), 7.0))
    assert runner.pipeline.execute_stage_model_calls == 1
    assert runner.pipeline.post_decode_calls == 0


@pytest.mark.parametrize("model_stage", ["encode", "decode"])
def test_diffusion_submodule_runner_dispatches_stage_and_shapes_output(monkeypatch, model_stage):
    pipeline = _SubmodulePipeline()
    runner = DiffusionSubmoduleRunner(
        vllm_config=object(),
        od_config=SimpleNamespace(model_stage=model_stage),
        device=torch.device("cpu"),
    )
    runner.pipeline = pipeline
    req = _make_diffusion_request(num_steps=1)
    req.sampling_params.seed = 123

    monkeypatch.setattr(submodule_runner_module, "set_forward_context", _noop_forward_context)

    output = runner.execute_model(req)

    assert pipeline.calls == [(model_stage, [req])]
    assert req.sampling_params.generator is not None
    assert req.sampling_params.generator.initial_seed() == 123
    assert output.output is (None if model_stage == "encode" else pipeline.image)
    assert output.multimodal_output == (
        {"context": pipeline.context} if model_stage == "encode" else {"image": pipeline.image, "output_type": "pil"}
    )


def test_diffusion_submodule_runner_returns_error_for_empty_stage_result(monkeypatch):
    runner = DiffusionSubmoduleRunner(
        vllm_config=object(),
        od_config=SimpleNamespace(model_stage="encode"),
        device=torch.device("cpu"),
    )
    runner.pipeline = _SubmodulePipeline(empty=True)

    monkeypatch.setattr(submodule_runner_module, "set_forward_context", _noop_forward_context)

    output = runner.execute_model(_make_diffusion_request(num_steps=1))

    assert output.error == "encode stage returned no outputs"


@pytest.mark.parametrize("model_stage", [None, "denoise"])
def test_diffusion_submodule_runner_rejects_non_encode_decode_stage(model_stage):
    runner = DiffusionSubmoduleRunner(
        vllm_config=object(),
        od_config=SimpleNamespace(model_stage=model_stage),
        device=torch.device("cpu"),
    )
    runner.pipeline = _SubmodulePipeline()

    with pytest.raises(ValueError, match="requires model_stage encode/decode"):
        runner.execute_model(_make_diffusion_request(num_steps=1))


def test_execute_stepwise_uses_denoise_stage_intermediate_output(monkeypatch):
    runner = _make_runner()
    runner.pipeline = _StepContractPipeline(num_steps=1)
    runner.state_cache = {}
    req = _make_diffusion_request(num_steps=1)
    scheduler_output = DiffusionSchedulerOutput(
        step_id=0,
        scheduled_new_reqs=[NewRequestData(request_id="req-1", req=req)],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )

    monkeypatch.setattr(model_runner_module, "set_forward_context", _noop_forward_context)

    batch_output = DiffusionModelRunner.execute_stepwise(runner, scheduler_output)
    output = batch_output["req-1"]

    assert output is not None
    assert output.request_id == "req-1"
    assert output.finished is True
    assert output.result is not None
    assert output.result.output is None
    assert output.result.multimodal_output["step_index"] == 1
    assert torch.equal(output.result.multimodal_output["latents"], torch.ones((1, 1)))
    assert runner.pipeline.post_decode_calls == 0


def test_diffusion_submodule_proc_uses_submodule_worker(monkeypatch):
    calls: list[tuple[str, object]] = []

    class FakeSubModuleWorker:
        def __init__(self, local_rank, rank, od_config):
            calls.append(("init", (local_rank, rank, od_config)))

        def load_model(self, load_format):
            calls.append(("load_model", load_format))

        def execute_submodule(self, request):
            calls.append(("execute_submodule", request))
            return DiffusionOutput(output=None, multimodal_output={"ok": True})

        def shutdown(self):
            calls.append(("shutdown", None))

    import vllm_omni.diffusion.stage_submodule_proc as proc_mod

    monkeypatch.setattr(proc_mod, "SubModuleWorker", FakeSubModuleWorker)
    od_config = SimpleNamespace(
        model="Qwen/Qwen-Image",
        model_class_name="QwenImagePipeline",
        model_stage="encode",
        tf_model_config=SimpleNamespace(params={"in_channels": 64}),
        diffusion_load_format="dummy",
        enrich_config=lambda: None,
        update_multimodal_support=lambda: None,
    )
    proc = StageSubModuleProc("Qwen/Qwen-Image", od_config)

    proc.initialize()
    output = asyncio.run(
        proc._process_request(
            "req-1",
            {"prompt": "test"},
            {"num_inference_steps": 1},
        )
    )
    proc.close()

    assert output.multimodal_output == {"ok": True}
    assert [name for name, _ in calls] == [
        "init",
        "load_model",
        "execute_submodule",
        "shutdown",
    ]
