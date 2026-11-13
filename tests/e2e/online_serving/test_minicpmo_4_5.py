# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""E2E online tests for MiniCPM-o 4.5 multimodal input and audio/text output.

Exercises async chunk streaming (``--async-chunk``) across separate Thinker,
Talker, and Code2Wav stages.
"""

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest
from vllm.logger import DEFAULT_LOGGING_CONFIG

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_MODEL = "openbmb/MiniCPM-o-4_5"
_NATIVE_CHAT_TEMPLATE = (
    Path(__file__).resolve().parents[3] / "vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja"
)
_CI_DEPLOY = modify_stage_config(
    get_deploy_config_path("minicpmo_4_5.yaml"),
    updates={
        "stages": {
            0: {"default_sampling_params.max_tokens": 64},
            1: {
                "default_sampling_params.max_tokens": 1024,
                # Content-consistency assertions must not depend on a random
                # codec trajectory: different valid TTS samples can append an
                # audible tail even when Thinker text is identical.
                "default_sampling_params.temperature": 0.0,
            },
        },
    },
)

_PROMPT_LOG_TEMP_DIR = tempfile.TemporaryDirectory(prefix="minicpmo45-prompt-selection-")
_PROMPT_SELECTION_LOG = Path(_PROMPT_LOG_TEMP_DIR.name) / "prompt-selection.log"
_PROMPT_LOGGING_CONFIG_PATH = Path(_PROMPT_LOG_TEMP_DIR.name) / "logging.json"
_PROMPT_LOGGING_CONFIG = deepcopy(DEFAULT_LOGGING_CONFIG)
_PROMPT_LOGGING_CONFIG["handlers"]["minicpmo_prompt_selection"] = {
    "class": "logging.FileHandler",
    "filename": str(_PROMPT_SELECTION_LOG),
    "formatter": "vllm",
    "level": "DEBUG",
}
_PROMPT_LOGGING_CONFIG["loggers"]["vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav"] = {
    "handlers": ["minicpmo_prompt_selection"],
    "level": "DEBUG",
    "propagate": True,
}
# The Talker logs once when it verifies draft codes.
_PROMPT_LOGGING_CONFIG["loggers"]["vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts"] = {
    "handlers": ["minicpmo_prompt_selection"],
    "level": "DEBUG",
    "propagate": True,
}
_PROMPT_LOGGING_CONFIG_PATH.write_text(json.dumps(_PROMPT_LOGGING_CONFIG), encoding="utf-8")

test_params = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_CI_DEPLOY,
            use_stage_cli=False,
            server_args=[
                "--trust-remote-code",
                "--async-chunk",
                "--chat-template",
                str(_NATIVE_CHAT_TEMPLATE),
                "--chat-template-content-format",
                "openai",
            ],
            env_dict={
                "VLLM_CONFIGURE_LOGGING": "1",
                "VLLM_LOGGING_CONFIG_PATH": str(_PROMPT_LOGGING_CONFIG_PATH),
            },
        ),
        id="async_chunk",
    ),
]

turn_test_params = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=modify_stage_config(
                _CI_DEPLOY,
                updates={
                    "session_mode": "turn",
                    # This smoke test uses the single-frame Talker.
                    "platforms.npu.stages.1.default_sampling_params.stop_token_ids": [],
                    "platforms.cuda.stages.1.default_sampling_params.stop_token_ids": [],
                },
                deletes={
                    "platforms.npu.stages.1.speculative_config": None,
                    "platforms.cuda.stages.1.speculative_config": None,
                },
            ),
            use_stage_cli=False,
            server_args=[
                "--trust-remote-code",
                "--async-chunk",
                "--chat-template",
                str(_NATIVE_CHAT_TEMPLATE),
                "--chat-template-content-format",
                "openai",
            ],
        ),
        id="turn_async_chunk",
    ),
]


# Talker DSpark draft, a test checkpoint only (its model card states no license).
# It keeps the DSpark fields under ``dflash_config`` and ships no codec embedding
# or head, so ``minicpmo_talker_draft`` writes a loadable copy before the server starts.
_TALKER_DRAFT = "Qun1023/MiniCPM-o-4_5-talker-DSpark"
_TALKER_DRAFT_REVISION = "0457ffd6980c2d1f19ebd3f1775f49764fa821af"
_TALKER_DRAFT_DIR = Path(_PROMPT_LOG_TEMP_DIR.name) / "talker-draft"
_TALKER_DRAFT_TOP_LEVEL_FIELDS = (
    "markov_rank",
    "markov_head_type",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "confidence_head_alpha",
    "mask_token_id",
    "target_layer_ids",
)
talker_dspark_params = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=modify_stage_config(
                get_deploy_config_path("minicpmo_4_5_turn.yaml"),
                updates={
                    "platforms.cuda.stages.1": {
                        "stage_id": 1,
                        "async_scheduling": True,
                        "default_sampling_params": {"stop_token_ids": []},
                        "speculative_config": {
                            "method": "dspark",
                            "model": str(_TALKER_DRAFT_DIR),
                            "num_speculative_tokens": 7,
                            "draft_sample_method": "probabilistic",
                        },
                    },
                    "stages": {0: {"default_sampling_params.max_tokens": 64}},
                },
            ),
            use_stage_cli=False,
            server_args=[
                "--trust-remote-code",
                "--chat-template",
                str(_NATIVE_CHAT_TEMPLATE),
                "--chat-template-content-format",
                "openai",
            ],
            env_dict={
                "VLLM_CONFIGURE_LOGGING": "1",
                "VLLM_LOGGING_CONFIG_PATH": str(_PROMPT_LOGGING_CONFIG_PATH),
            },
        ),
        id="turn_mrv2_talker_dspark",
    ),
]


def _repo_file(repo: str, filename: str, revision: str | None = None) -> Path:
    """A file of ``repo`` from the ``MODEL_PREFIX`` mirror when it has one, else the Hub."""
    from huggingface_hub import hf_hub_download

    from tests.helpers.runtime import get_model_prefix

    local = Path(get_model_prefix() + repo) / filename
    if get_model_prefix() and local.exists():
        return local
    return Path(hf_hub_download(repo, filename, revision=revision))


@pytest.fixture(scope="session")
def minicpmo_talker_draft() -> Path:
    """Write a copy of the Talker draft that vLLM can load, before its server starts.

    Qwen3DSparkModel reads the DSpark fields from the top level of the config.
    The Talker stage reports the text vocabulary, so vLLM does not share the
    Talker's codec embedding and head with this draft: they are added here.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    if (_TALKER_DRAFT_DIR / "model.safetensors").exists():
        return _TALKER_DRAFT_DIR
    config = json.loads(_repo_file(_TALKER_DRAFT, "config.json", _TALKER_DRAFT_REVISION).read_text())
    for key in _TALKER_DRAFT_TOP_LEVEL_FIELDS:
        config.setdefault(key, config["dflash_config"][key])
    weights = load_file(_repo_file(_TALKER_DRAFT, "model.safetensors", _TALKER_DRAFT_REVISION))

    weight_map = json.loads(_repo_file(_MODEL, "model.safetensors.index.json").read_text())["weight_map"]
    embed, head_g, head_v = (
        "tts.emb_code.0.weight",
        "tts.head_code.0.parametrizations.weight.original0",
        "tts.head_code.0.parametrizations.weight.original1",
    )
    talker = {}
    for name in (embed, head_g, head_v):
        with safe_open(_repo_file(_MODEL, weight_map[name]), framework="pt") as f:
            talker[name] = f.get_tensor(name)
    weights["embed_tokens.weight"] = talker[embed].to(torch.bfloat16).contiguous()
    # The head is weight-normed (dim 0); restore it as the Talker does at load.
    head = torch._weight_norm(talker[head_v], talker[head_g], dim=0)
    weights["lm_head.weight"] = head.to(torch.bfloat16).contiguous()

    _TALKER_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    (_TALKER_DRAFT_DIR / "config.json").write_text(json.dumps(config), encoding="utf-8")
    save_file(weights, _TALKER_DRAFT_DIR / "model.safetensors", metadata={"format": "pt"})
    return _TALKER_DRAFT_DIR


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are MiniCPM-o 4.5, a virtual human capable of perceiving "
                    "auditory and visual inputs, as well as generating text and speech."
                ),
            }
        ],
    }


def get_prompt(prompt_type: str = "text_only") -> str:
    prompts = {
        "text_only": "What is the capital of China? Answer in 20 words.",
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
        "text_image": "What color are the squares in this image?",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


# Close <think> and emit <|tts_bos|> so Talker speaks the answer, not reasoning.
_TTS_EXTRA_BODY = {
    "chat_template_kwargs": {
        "use_tts_template": True,
        "enable_thinking": False,
    }
}


def _prompt_selection_log_offset() -> int:
    return _PROMPT_SELECTION_LOG.stat().st_size if _PROMPT_SELECTION_LOG.exists() else 0


def _read_prompt_selection_log(offset: int) -> str:
    if not _PROMPT_SELECTION_LOG.exists():
        return ""
    with open(_PROMPT_SELECTION_LOG, "rb") as log_file:
        log_file.seek(offset)
        return log_file.read().decode("utf-8")


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params + turn_test_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """
    Test text-only input generating text + audio output via OpenAI API.
    This exercises Talker TTS region detection and the Code2Wav stage.
    Deploy Setting: default single GPU
    Input Modal: text
    Output Modal: text + audio
    Input Setting: stream=True
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"audio": ["Beijing"]},
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config)


# Merge selection (`-m 'advanced_model and cuda'`) reuses module-scoped
# ``omni_server`` only while the next collected item's param value matches.
# Keep every ``test_params`` case (id ``async_chunk``) in one contiguous block
# before Talker DSpark. ``turn_async_chunk`` is a different deploy
# (``session_mode: turn``); it stays the second parameter of
# ``test_text_to_text_001``, after this block.


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_with_reference_audio(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt()),
        "stream": True,
        "modalities": ["text", "audio"],
        "extra_body": {
            **_TTS_EXTRA_BODY,
            "ref_audio": f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}",
        },
    }

    log_offset = _prompt_selection_log_offset()
    responses = openai_client.send_omni_request(request_config)
    prompt_log = _read_prompt_selection_log(log_offset)

    assert responses[0].success
    assert responses[0].audio_bytes
    assert "prompt_cache_id=runtime-ref-" in prompt_log
    assert "minicpmo45_ref_" in prompt_log


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_with_default_reference(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt()),
        "stream": True,
        "modalities": ["text", "audio"],
        "extra_body": _TTS_EXTRA_BODY,
    }

    responses = openai_client.send_omni_request(request_config)

    assert responses[0].success
    assert responses[0].audio_bytes


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test multi-modal input (text + audio + video + image) generating text + audio output.
    Deploy Setting: default single GPU
    Input Modal: text + audio + video + image
    Output Modal: text + audio
    Input Setting: stream=True
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(24, 24, 20)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(24, 24)['base64']}"
    # A loop of the default single word "test" can make the answer repeat
    # to the token limit, leaving TTS/ASR to count a long run of identical words
    # (#7630). Exercise speech perception with a meaningful phrase instead.
    audio = generate_synthetic_audio(5, 1, phrase_text="The weather is sunny today.")
    audio_data_url = f"data:audio/wav;base64,{audio['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text=(
            "Briefly describe the image and video, and summarize what the voice says. "
            "Answer in one short sentence without repeating the spoken phrase."
        ),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params + turn_test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    """
    Test text-only input generating text output via OpenAI API.
    Deploy Setting: default single GPU
    Input Modal: text
    Output Modal: text
    Input Setting: stream=False
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["Beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.usefixtures("minicpmo_talker_draft")
@pytest.mark.parametrize("omni_server", talker_dspark_params, indirect=True)
def test_text_to_audio_talker_dspark(omni_server, openai_client) -> None:
    """
    Test the MRv2 Talker verifying DSpark draft codes in turn sessions.
    Deploy Setting: turn MRv2 profile, DSpark draft on stage 1
    Input Modal: text
    Output Modal: text + audio
    Input Setting: stream=True
    """
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt()),
        "stream": True,
        "modalities": ["text", "audio"],
        "key_words": {"audio": ["Beijing"]},
        "extra_body": _TTS_EXTRA_BODY,
    }

    responses = openai_client.send_omni_request(request_config, request_num=get_max_batch_size())

    assert all(response.success and response.audio_bytes for response in responses)
    # Only stage 1 with a loaded draft verifies codes. Read from the start: the line is logged once.
    assert "MiniCPM-o 4.5 Talker is verifying draft codes" in _read_prompt_selection_log(0)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test audio input generating text + audio output via OpenAI API.
    Deploy Setting: default single GPU
    Input Modal: text + audio
    Output Modal: text + audio
    Input Setting: stream=True
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        audio_data_url=audio_data_url,
        content_text=get_prompt(),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"text": ["Beijing"]},
        "modalities": ["text", "audio"],
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test image input generating text + audio output via OpenAI API.
    Deploy Setting: default single GPU
    Input Modal: text + image
    Output Modal: text + audio
    Input Setting: stream=True
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(24, 24)['base64']}"
    # Match the native image-chat layout explicitly; do not rely on the
    # renderer moving media placeholders ahead of text.
    messages = [
        get_system_prompt(),
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": get_prompt("text_image")},
            ],
        },
    ]

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test video input generating text + audio output via OpenAI API.
    Deploy Setting: default single GPU
    Input Modal: text + video
    Output Modal: text + audio
    Input Setting: stream=True
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(24, 24, 20)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text=get_prompt("mix"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())
