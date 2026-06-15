# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import concurrent.futures
import time

import numpy as np
import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import (
    OmniServer,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

MODEL = "Qwen/Qwen-Image"
NEGATIVE_PROMPT = "blurry, low quality"
SINGLE_CARD_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"})
pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, *SINGLE_CARD_FEATURE_MARKS]


def _server_args(*, stepwise: bool) -> list[str]:
    args = [
        "--enforce-eager",
        "--cache-backend",
        "none",
        "--stage-init-timeout",
        "600",
        "--init-timeout",
        "900",
        "--log-stats",
    ]
    if stepwise:
        args = [
            "--step-execution",
            "--max-num-seqs",
            "2",
            *args,
        ]
    return args


def _request_config(
    model: str,
    *,
    prompt: str,
    height: int,
    width: int,
    seed: int,
    true_cfg_scale: float,
) -> dict:
    return {
        "model": model,
        "messages": dummy_messages_from_mix_data(content_text=prompt),
        "extra_body": {
            "height": height,
            "width": width,
            "num_inference_steps": 4,
            "negative_prompt": NEGATIVE_PROMPT,
            "true_cfg_scale": true_cfg_scale,
            "seed": seed,
        },
    }


def _image(response):
    assert response.images and len(response.images) == 1
    return response.images[0].convert("RGB")


def _run_serial(openai_client: OpenAIClientHandler, configs: list[dict]) -> tuple[list, float]:
    start = time.perf_counter()
    responses = [openai_client.send_diffusion_request(config)[0] for config in configs]
    return [_image(response) for response in responses], time.perf_counter() - start


def _run_staggered_batch(openai_client: OpenAIClientHandler, configs: list[dict]) -> tuple[list, float]:
    def send_after_delay(index: int):
        if index > 0:
            time.sleep(0.02)
        return openai_client.send_diffusion_request(configs[index])[0]

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(configs)) as executor:
        futures = [executor.submit(send_after_delay, index) for index in range(len(configs))]
        responses = [future.result() for future in futures]
    return [_image(response) for response in responses], time.perf_counter() - start


def _assert_images_close(actual_images: list, expected_images: list) -> None:
    assert len(actual_images) == len(expected_images)
    for index, (actual, expected) in enumerate(zip(actual_images, expected_images, strict=True)):
        assert actual.size == expected.size
        actual_arr = np.asarray(actual, dtype=np.int16)
        expected_arr = np.asarray(expected, dtype=np.int16)
        diff = np.abs(actual_arr - expected_arr)
        max_abs = int(diff.max())
        mean_abs = float(diff.mean())
        p99_abs = float(np.quantile(diff, 0.99))
        assert mean_abs <= 1.0, f"image {index} mean_abs={mean_abs:.6f}"
        assert p99_abs <= 4.0, f"image {index} p99_abs={p99_abs:.6f}"
        assert max_abs <= 16, f"image {index} max_abs={max_abs}"


def test_qwen_image_stepwise_dynamic_matches_execute_model_for_staggered_pairs(
    model_prefix: str,
    run_level: str,
) -> None:
    prompts = [
        "A small red teapot on a white table, product photo.",
        "A blue ceramic vase beside a window, product photo with soft light.",
    ]
    size_pairs = [
        ((512, 512), (512, 512)),
        ((512, 512), (1024, 1024)),
        ((512, 512), (256, 256)),
    ]
    model = model_prefix + MODEL

    def configs_for(server_model: str, first_size: tuple[int, int], second_size: tuple[int, int]) -> list[dict]:
        return [
            _request_config(
                server_model,
                prompt=prompts[0],
                height=first_size[0],
                width=first_size[1],
                seed=123,
                true_cfg_scale=2.0,
            ),
            _request_config(
                server_model,
                prompt=prompts[1],
                height=second_size[0],
                width=second_size[1],
                seed=456,
                true_cfg_scale=7.0,
            ),
        ]

    execute_model_images_by_pair: dict[tuple[tuple[int, int], tuple[int, int]], list] = {}
    with OmniServer(model, _server_args(stepwise=False), use_omni=True) as execute_model_server:
        execute_client = OpenAIClientHandler(
            host=execute_model_server.host,
            port=execute_model_server.port,
            run_level=run_level,
            log_stats=execute_model_server.log_stats,
        )
        for first_size, second_size in size_pairs:
            configs = configs_for(execute_model_server.model, first_size, second_size)
            execute_model_images_by_pair[(first_size, second_size)], _ = _run_serial(execute_client, configs)

    with OmniServer(model, _server_args(stepwise=True), use_omni=True) as dynamic_server:
        dynamic_client = OpenAIClientHandler(
            host=dynamic_server.host,
            port=dynamic_server.port,
            run_level=run_level,
            log_stats=dynamic_server.log_stats,
        )
        for first_size, second_size in size_pairs:
            configs = configs_for(dynamic_server.model, first_size, second_size)
            expected_images = execute_model_images_by_pair[(first_size, second_size)]

            dynamic_single_images, dynamic_single_s = _run_serial(dynamic_client, configs)
            dynamic_multi_images, dynamic_multi_s = _run_staggered_batch(dynamic_client, configs)

            _assert_images_close(dynamic_single_images, expected_images)
            _assert_images_close(dynamic_multi_images, expected_images)
            _assert_images_close(dynamic_multi_images, dynamic_single_images)
            assert dynamic_multi_s <= dynamic_single_s * 1.10
