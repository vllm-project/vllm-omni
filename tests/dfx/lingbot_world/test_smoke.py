# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Short real-checkpoint PR gate on the public streamed-video endpoint."""

import asyncio

import pytest

from benchmarks.lingbot_world.benchmark_lingbot_world_realtime import parse_args, run_benchmark
from benchmarks.lingbot_world.regression import read_json, sha256, write_json
from tests.dfx.lingbot_world.runner import (
    ENGINE_ENVIRONMENT,
    ROOT,
    _measured_metrics,
    checked_assets,
    hardware_contract,
    server_arguments,
    server_environment,
)
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
def test_cuda_smoke(lingbot_settings):
    from tests.e2e.online_serving import test_lingbot_world_v2_stepwise as lifecycle
    from tests.helpers.runtime import OmniServer

    settings = lingbot_settings
    assert settings["platform"] == "cuda" and settings["dies"] == 1
    write_json(
        settings["output"] / "configuration.json",
        {
            "profile": read_json(settings["profile_path"]),
            "deploy_sha256": sha256(ROOT / settings["deploy_config"]),
            "engine_environment": dict(ENGINE_ENVIRONMENT),
        },
    )
    write_json(settings["output"] / "hardware.json", hardware_contract(settings))
    model, images = checked_assets(settings)
    with OmniServer(
        str(model),
        server_arguments(settings),
        env_dict=server_environment(ROOT, cpu_threads=settings.get("cpu_threads_per_worker")),
        startup_timeout=settings["startup_timeout_s"],
    ) as server:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(lifecycle, "_IMAGE_PATH_OVERRIDE", str(images[settings["cases"][0]["image"]]))
            patch.setattr(lifecycle, "_HEIGHT", settings["height"])
            patch.setattr(lifecycle, "_WIDTH", settings["width"])
            output = settings["output"]
            document = asyncio.run(
                run_benchmark(
                    parse_args(
                        [
                            "--port",
                            str(server.port),
                            "--model",
                            str(model),
                            "--image",
                            str(images[settings["cases"][0]["image"]]),
                            "--prompt",
                            settings["cases"][0]["prompt"],
                            "--seed",
                            str(settings["cases"][0]["seed"]),
                            "--camera-pattern",
                            settings["cases"][0]["camera_pattern"],
                            "--height",
                            str(settings["height"]),
                            "--width",
                            str(settings["width"]),
                            "--num-chunks",
                            str(settings["num_chunks"]),
                            "--warmup-chunks",
                            str(settings["warmup_chunks"]),
                            "--fps",
                            str(settings["mux_fps"]),
                            "--target-fps",
                            str(settings["target_fps"]),
                            "--first-chunk-timeout",
                            str(settings["first_chunk_timeout_s"]),
                            "--chunk-timeout",
                            str(settings["chunk_timeout_s"]),
                            "--output-json",
                            str(output / "benchmark.json"),
                            "--save-video",
                            str(output / "rollout.mp4"),
                        ]
                    )
                )
            )
            write_json(output / "metrics.json", _measured_metrics(document, settings))
            lifecycle.test_lingbot_world_v2_stepwise_stop_during_paced_wait_releases_session(server)
