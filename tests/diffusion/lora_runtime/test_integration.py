# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.base_scheduler import BaseScheduler
from vllm_omni.diffusion.sched.request_scheduler import (
    build_request_batch_sampling_params_key,
)
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _StepScheduler(BaseScheduler):
    def update_from_output(self, sched_output, output) -> set[str]:
        del sched_output, output
        return set()


def _request(name: str, scale: float) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        prompt="prompt",
        request_id=name,
        sampling_params=OmniDiffusionSamplingParams(
            num_inference_steps=4,
            diffusion_loras=({"name": name, "scale": scale},),
        ),
    )


def test_composition_is_part_of_both_batching_keys():
    turbo = _request("turbo", 1.0)
    style = _request("style", 0.5)

    assert build_request_batch_sampling_params_key(turbo) != build_request_batch_sampling_params_key(style)
    scheduler = _StepScheduler()
    assert scheduler._build_sampling_params_key(turbo) != scheduler._build_sampling_params_key(style)


def test_serve_cli_forwards_runtime_deployments():
    parser = TrackingArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)
    args = parser.parse_args(
        [
            "serve",
            "MiniMaxAI/MiniMax-H3",
            "--omni",
            "--enable-diffusion-lora",
            "--diffusion-lora",
            '{"name":"turbo","path":"lightx2v/Minimax-h3-Turbo"}',
        ]
    )

    explicit = args.get_explicit_kwargs_dict()
    engine_args = AsyncOmniEngine._create_default_diffusion_stage_cfg(explicit)[0]["engine_args"]
    assert engine_args["enable_diffusion_lora"] is True
    assert engine_args["diffusion_lora"] == ['{"name":"turbo","path":"lightx2v/Minimax-h3-Turbo"}']


def test_runtime_config_rejects_legacy_lora_path():
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    with pytest.raises(ValueError, match="legacy --lora-path"):
        OmniDiffusionConfig(
            model="MiniMaxAI/MiniMax-H3",
            enable_diffusion_lora=True,
            diffusion_lora=['{"name":"turbo","path":"adapter"}'],
            lora_path="legacy",
            parallel_config=SimpleNamespace(world_size=1),
        )


def test_engine_rejects_unavailable_lora_before_admission(mocker):
    engine = object.__new__(DiffusionEngine)
    engine.od_config = SimpleNamespace(enable_diffusion_lora=True)
    engine._registered_diffusion_lora_names = frozenset({"turbo"})
    engine.pre_process_func = mocker.Mock()
    engine._diffusion_kv_profile_limits = None

    with pytest.raises(OmniClientError, match="Unknown diffusion LoRA"):
        engine._prepare_request_for_admission(_request("unknown", 1.0))
    engine.pre_process_func.assert_not_called()

    engine.od_config.enable_diffusion_lora = False
    with pytest.raises(OmniClientError, match="did not enable"):
        engine._prepare_request_for_admission(_request("turbo", 1.0))
    engine.pre_process_func.assert_not_called()


def test_new_runtime_rejects_legacy_lora_management(mocker):
    worker = object.__new__(DiffusionWorker)
    worker.diffusion_lora_runtime = object()
    worker.lora_manager = None

    calls = (
        ("add_lora", (mocker.Mock(),)),
        ("remove_lora", (1,)),
        ("list_loras", ()),
        ("pin_lora", (1,)),
    )
    for method_name, args in calls:
        with pytest.raises(NotImplementedError, match="immutable after startup"):
            getattr(worker, method_name)(*args)
