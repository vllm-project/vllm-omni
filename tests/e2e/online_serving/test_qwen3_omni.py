"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os

import pytest

from tests.helpers.mark import hardware_marks, hardware_test
from tests.helpers.mark import hardware_marks, hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"


models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]


_CI_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe.yaml")


def get_chunk_config(config_path: str | None = None):
    """Load the qwen3_omni CI deploy yaml with async_chunk modifications for streaming mode."""
    if config_path is None:
        config_path = _CI_DEPLOY
    # TODO: remove this workaround once legacy `stage_args` path is deleted.
    # The pipeline (qwen3_omni/pipeline.py) already wires
    # thinker2talker_async_chunk / talker2code2wav_async_chunk on stage 0/1,
    # so only async_chunk needs flipping. Writing nested `engine_args:` into
    # the new-schema overlay trips _parse_stage_deploy's legacy branch and
    # drops flat fields (load_format, max_num_seqs, ...).
    return modify_stage_config(config_path, updates={"async_chunk": True})


def get_pd_config(config_path: str | None = None):
    """Load the qwen3_omni CI deploy yaml with PD disaggregation enabled."""
    """Load the qwen3_omni CI deploy yaml with PD disaggregation enabled."""
    if config_path is None:
        config_path = _CI_DEPLOY
    return modify_stage_config(
        config_path,
        updates={
            "pd_disaggregation.enabled": True,
            "pd_disaggregation.async_chunk": False,
            "pd_disaggregation.enabled": True,
            "pd_disaggregation.async_chunk": False,
            "stages": {
                1: {"devices": "2"},
                2: {"devices": "2"},
            },
        },
    )


def get_prefix_caching_config(config_path: str):
    """Create a stage config with prefix caching enabled on the thinker (stage 0)."""
    path = modify_stage_config(
        config_path,
        updates={
            "stage_args": {
                0: {"engine_args.enable_prefix_caching": True},
            },
        },
    )
    return path


# Cover sync, async-chunk, and PD launch paths by default.
test_params = [
    pytest.param(
        OmniServerParams(model=model, stage_config_path=_CI_DEPLOY),
        id="default",
        marks=hardware_marks(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2),
    )
    for model in models
] + [
    pytest.param(
        OmniServerParams(model=model, stage_config_path=get_chunk_config()),
        id="async_chunk",
        marks=hardware_marks(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2),
    )
    for model in models
] + [
    pytest.param(
        OmniServerParams(model=model, stage_config_path=get_pd_config()),
        id="pd_default",
        marks=hardware_marks(res={"cuda": "H100", "rocm": "MI325"}, num_cards=3),
    )
    for model in models
]
prefix_caching_stage_configs = [get_prefix_caching_config(_CI_DEPLOY)]

# For prefix caching, we need to enable prompt token details so that we
# can determine if any tokens were cached.
prefix_test_params = [
    OmniServerParams(
        model=model,
        stage_config_path=stage_config,
        server_args=["--enable-prompt-tokens-details"],  # Enable prompt tokens details to get cached_tokens
    )
    for model in models
    for stage_config in prefix_caching_stage_configs
]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def get_prompt(prompt_type="text_only"):
    prompts = {
        "text_only": "What is the capital of China? Answer in 20 words.",
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
        "text_image": "What color are the squares in this image?",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test multi-modal input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + audio + video + image
    Output Modal: text + audio
    Input Setting: stream=True
    Datasets: single request
    """

    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text=get_prompt("mix"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {
            "audio": ["test"],
        },
    }

    # Test single completion
    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    """
    Test text input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: text
    Datasets: few requests
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", prefix_test_params, indirect=True)
def test_thinker_prefix_caching(omni_server, openai_client) -> None:
    """
    Test thinker prefix caching by sending identical requests with an image (i.e.,
    a large shared prefix) and verifying that the second request uses cached tokens
    & produces the same output with greedy decoding.

    NOTE: The seed for this test is used as a regression test for the issue linked below;
    https://github.com/vllm-project/vllm-omni/issues/2833; without passing the sampling
    params, this test will fail with the current default stage configs.
    """
    seed = 10
    img_res = generate_synthetic_image(224, 224, seed=seed)
    image_data_url = f"data:image/jpeg;base64,{img_res['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        image_data_url=image_data_url,
        content_text=get_prompt("text_image"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "sampling_params_list": [{"seed": seed, "temperature": 0, "max_tokens": 16}] * 3,
    }

    response_1 = openai_client.send_omni_request(request_config, request_num=1)[0]
    response_2 = openai_client.send_omni_request(request_config, request_num=1)[0]

    # We should cache the vast majority of the prompt (image + up to last full block),
    # and set seed + temperature, so the second request should give an identical
    # response for the generated input image, even if we use dummy weights
    assert response_2.cached_tokens is not None and response_2.cached_tokens > 0
    assert response_1.text_content == response_2.text_content
