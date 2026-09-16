# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E offline tests for Audio8 TTS Preview (DualAR, 44.1 kHz codec).

The prompt is pre-tokenized here (not built from ``mm_processor_kwargs``)
because Audio8 TTS shares Fish Speech's protocol: text-only prompts are plain
token ids, and voice-clone prompts are a placeholder of the exact final length
whose real embeddings are built model-side from the encoded reference audio.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner, OmniRunnerHandler
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("AUDIO8_TTS_MODEL_PATH", "Audio8/Audio8-TTS-Preview-0.6b")
DEPLOY_CONFIG = get_deploy_config_path("audio8_tts.yaml")
SAMPLE_RATE = 44100
TEXT = "The weather is nice today, perfect for a walk in the park."

# Four distinct inputs for the concurrency test: distinct text is required so a
# leaked per-request codec buffer (which would make two requests decode to the
# same audio) actually fails the pairwise-difference assertion.
CONCURRENT_TEXTS = [
    "The weather is nice today, perfect for a walk in the park.",
    "She sold seashells by the seashore all through the summer.",
    "Our train departs at a quarter past nine tomorrow morning.",
    "He planted rows of tomatoes and basil behind the old house.",
]

# The checkpoint registers its own `arktts` config in vllm-omni, so remote code
# must stay disabled (transformers would otherwise prefer the checkpoint's
# auto_map and bypass the Qwen2-shaped view Qwen2Model needs).
_OMNI_RUNNER_PARAM = (MODEL, DEPLOY_CONFIG, {"trust_remote_code": False})

pytestmark = pytest.mark.parametrize("omni_runner", [_OMNI_RUNNER_PARAM], indirect=True)


def _text_only_prompt(text: str = TEXT) -> dict:
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.audio8_tts.prompt_utils import build_text_only_prompt_ids

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_ids, normalized_text = build_text_only_prompt_ids(tokenizer, text)
    return {"prompt_token_ids": prompt_ids, "additional_information": {"text": [normalized_text]}}


# At ``core_model`` (L1/L2) the harness patches every stage to
# ``load_format: dummy``, so the AR runs on random weights: it never emits EOS
# and always generates to ``max_tokens`` (1024 frames ~= 47.5 s). Duration
# bounds are therefore only meaningful once real weights load, at
# ``advanced_model`` / ``full_model``.
_REAL_WEIGHT_RUN_LEVELS = frozenset({"advanced_model", "full_model"})


def _duration_bounds(run_level: str) -> dict[str, float]:
    if run_level not in _REAL_WEIGHT_RUN_LEVELS:
        return {}
    return {"min_duration_s": 0.5, "max_duration_s": 30.0}


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_text_to_audio_001(omni_runner: OmniRunner, run_level: str) -> None:
    """Default deploy, single request, text-only synthesis.

    Deploy Setting: audio8_tts.yaml
    Input Modal: text
    Output Modal: audio
    """
    prompt = _text_only_prompt()
    request_config = {
        "input": TEXT,
        "prompt_token_ids": prompt["prompt_token_ids"],
        "additional_information": prompt["additional_information"],
        "response_format": "wav",
        "expected_sample_rate": SAMPLE_RATE,
        **_duration_bounds(run_level),
    }
    OmniRunnerHandler(omni_runner).send_tokenized_tts_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_text_to_audio_concurrent_002(omni_runner: OmniRunner, run_level: str) -> None:
    """Four concurrent requests with distinct text: stage 0 runs ``max_num_seqs: 4``.

    Per-request codec state is keyed by request id. The four inputs are
    deliberately distinct and the decoded outputs are asserted pairwise
    different, so leaked codec state (which would make two requests decode to
    identical audio) fails here, and only here.
    """
    prompts = [_text_only_prompt(text) for text in CONCURRENT_TEXTS]
    request_config = {
        "input": CONCURRENT_TEXTS[0],
        "prompt_token_ids": [p["prompt_token_ids"] for p in prompts],
        "additional_information": [p["additional_information"] for p in prompts],
        "response_format": "wav",
        "expected_sample_rate": SAMPLE_RATE,
        "assert_distinct_outputs": True,
        **_duration_bounds(run_level),
    }
    OmniRunnerHandler(omni_runner).send_tokenized_tts_request(request_config)
