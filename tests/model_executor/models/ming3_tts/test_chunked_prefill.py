# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Conditioning must be independent of the scheduler's prompt partition."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.ming3_tts.test_ming3_contract import FakeTokenizer, _config, _prefill_wrapper
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.model_executor.models.ming3_tts import ming3_tts as model_module
from vllm_omni.model_executor.models.ming3_tts.constants import KEY_PROMPT_AUDIO_START
from vllm_omni.model_executor.models.ming3_tts.ming3_tts_llm import Ming3TTSLLMModel
from vllm_omni.model_executor.models.ming3_tts.prompt_assembly import build_ming3_prompt
from vllm_omni.model_executor.models.ming_tts import config_ming_tts as state_keys

pytestmark = [pytest.mark.core_model, pytest.mark.tts]


class _Backbone(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embedding = torch.nn.Embedding(157184, 4, device=device)
        self.project_speaker_embedding = torch.nn.Linear(192, 4, device=device)
        self.linear_proj_audio = torch.nn.Linear(64, 4, device=device)
        self.fm_dtype = torch.float32
        self._request_generators = {}
        self._pending_postprocess_updates = {}

    def embed_input_ids(self, ids):
        return self.embedding(ids)

    _get_request_generator = Ming3TTSLLMModel._get_request_generator
    on_requests_finished = Ming3TTSLLMModel.on_requests_finished


def _wrapper(device="cpu"):
    from vllm_omni.model_executor.models.ming3_tts.config_ming3_tts import Ming3TTSRuntimeConfig

    wrapper = _prefill_wrapper(_Backbone(device), Ming3TTSRuntimeConfig.from_hf_config(_config()))
    wrapper.model_stage = "llm"
    wrapper.ming_config.llm_hidden_size = 4
    return wrapper


def _prompt(mode):
    speakers = 2 if mode == "podcast" else 1
    reference = None if mode in {"masked", "no_speaker"} else [torch.zeros((1, 44100))] * speakers
    text = "speaker_1:你好。\nspeaker_2:欢迎。" if speakers == 2 else "清晨的风穿过树林。"
    prompt = build_ming3_prompt(
        FakeTokenizer(),
        target_text=text,
        reference_waveform=reference,
        reference_sample_rate=[44100] * speakers if reference is not None else None,
        prompt_text=text if reference is not None else None,
        speaker_embedding=torch.arange(speakers * 192).reshape(speakers, 192) / 192 if reference is not None else None,
        instruction="自然清晰。" if mode == "masked" else None,
        speaker_slot=mode != "no_speaker",
    )
    info = deserialize_additional_information(serialize_additional_information(prompt["additional_information"]))
    info.update(request_id="request", tts_local_seed=17, _omni_prompt_len=len(prompt["prompt_token_ids"]))
    return torch.tensor(prompt["prompt_token_ids"]), info


@pytest.fixture
def reference_draws(monkeypatch):
    draws = []

    def encode(wrapper, info, *, generator=None):
        if "prompt_waveform" not in info:
            return None
        draws.append(info["request_id"])
        count = info["_omni_prompt_len"] - info[KEY_PROMPT_AUDIO_START]
        patches = torch.randn((count, 1, 64), device=wrapper.model.embedding.weight.device, generator=generator)
        return {"frames": patches.squeeze(1), "patches": patches}

    monkeypatch.setattr(model_module, "_resolve_prompt_latents", encode)
    return draws


def _run(wrapper, ids, info, cuts):
    result = []
    for start, end in zip(cuts, cuts[1:]):
        _, embeds, update = wrapper.preprocess(
            ids[start:end], None, **info, _omni_is_prefill=True, _omni_num_computed_tokens=start
        )
        result.append(embeds.detach().clone())
        info.update(update)
    return torch.cat(result)


@pytest.mark.parametrize("mode", ["clone", "podcast", "masked", "no_speaker"])
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "H100"}, num_cards=1)),
    ],
)
def test_chunked_prefill_matches_full(reference_draws, mode, device):
    from vllm.platforms import current_platform

    if device == "cuda" and not current_platform.is_cuda():
        pytest.skip("CUDA device required")
    wrapper = _wrapper(device)
    ids, initial_info = _prompt(mode)
    ids = ids.to(device)
    full_info = initial_info.copy()
    full = _run(wrapper, ids, full_info, [0, len(ids)])
    generator = wrapper.model._request_generators.get("request")
    rng_state = generator.get_state() if generator is not None else None
    wrapper.on_requests_finished({"request"})
    split_info = initial_info.copy()
    # Single-token chunks cross every speaker/audio boundary, including the tail.
    actual = _run(wrapper, ids, split_info, range(len(ids) + 1))
    torch.testing.assert_close(actual, full, rtol=0, atol=0)
    torch.testing.assert_close(split_info[state_keys.KEY_LATENT_HISTORY], full_info[state_keys.KEY_LATENT_HISTORY])
    # Recomputing a prefix must reuse the same sampled reference, not advance RNG.
    split_info[state_keys.KEY_DECODE_STEP] = 7
    saved_history = torch.ones_like(split_info[state_keys.KEY_LATENT_HISTORY])
    split_info[state_keys.KEY_LATENT_HISTORY] = saved_history
    replay = _run(wrapper, ids, split_info, [0, len(ids)])
    torch.testing.assert_close(replay, full, rtol=0, atol=0)
    assert split_info[state_keys.KEY_DECODE_STEP] == 7
    torch.testing.assert_close(split_info[state_keys.KEY_LATENT_HISTORY], saved_history)
    assert len(reference_draws) == (2 if "prompt_waveform" in initial_info else 0)
    if rng_state is not None:
        assert torch.equal(wrapper.model._request_generators["request"].get_state(), rng_state)
    wrapper.on_requests_finished({"request"})
    assert not wrapper._prefill_conditioning and not wrapper.model._request_generators


@pytest.mark.cpu
def test_interleaved_prefill_requests_are_isolated_and_cancelled(reference_draws):
    wrapper = _wrapper()
    ids, first = _prompt("clone")
    second = {**first, "request_id": "other", "tts_local_seed": 42}
    for start, end in [(0, len(ids) - 1), (len(ids) - 1, len(ids))]:
        for info in (first, second):
            _run(wrapper, ids, info, [start, end])
    assert reference_draws == ["request", "other"]
    assert len(wrapper._prefill_conditioning) == 2
    assert not torch.equal(
        wrapper._prefill_conditioning["request"].prompt_embeds,
        wrapper._prefill_conditioning["other"].prompt_embeds,
    )
    wrapper.on_requests_finished({"request"})
    assert set(wrapper._prefill_conditioning) == {"other"}
    wrapper.on_requests_finished({"other"})
    assert not wrapper._prefill_conditioning and not wrapper.model._request_generators
