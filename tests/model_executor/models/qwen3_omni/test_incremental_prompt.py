# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast

from vllm_omni.model_executor.models.qwen3_omni.duplex.incremental_prompt import QwenIncrementalPrompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        ["user assistant system hello world changed history audio image 工具天气\n"],
        trainers.BpeTrainer(vocab_size=200, special_tokens=["[UNK]", "<|im_start|>", "<|im_end|>"]),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )


@pytest.mark.parametrize("change", ["edit", "prune", "system", "assistant"])
def test_history_changes_remain_equivalent_to_full_encoding(tokenizer, change):
    incremental = QwenIncrementalPrompt(tokenizer)
    first = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    incremental.tokenize(first)
    prompt = first + "world<|im_end|>\n<|im_start|>user\naudio<|im_end|>\n<|im_start|>assistant\n"
    incremental.tokenize(prompt)
    changed = {
        "edit": prompt.replace("hello", "changed"),
        "prune": "<|im_start|>user\naudio<|im_end|>\n<|im_start|>assistant\n",
        "system": "<|im_start|>system\nchanged<|im_end|>\n" + prompt,
        "assistant": prompt.replace("world", "hello"),
    }[change]
    assert incremental.tokenize(changed) == tokenizer.encode(changed)


def test_implicit_special_tokens_disable_reuse(tokenizer):
    bos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
        single="<|im_end|> $A", special_tokens=[("<|im_end|>", bos_id)]
    )
    incremental = QwenIncrementalPrompt(tokenizer)
    first = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    for prompt in (first, first + "world<|im_end|>\n<|im_start|>assistant\n"):
        assert incremental.tokenize(prompt) == tokenizer.encode(prompt)
        assert not incremental._append_safe


@pytest.mark.parametrize("change", [None, "unlisted_start", "assistant", "unfinished"])
def test_turn_delta_keeps_sampled_tokens_and_only_new_media(tokenizer, change):
    from vllm.sampling_params import SamplingParams

    from vllm_omni.engine import OmniEngineCoreRequest
    from vllm_omni.model_executor.models.qwen3_omni.duplex.incremental_prompt import DELTA_KEY

    class UnlistedStartTokenizer:
        all_special_tokens = [token for token in tokenizer.all_special_tokens if token != "<|im_start|>"]

        def __getattr__(self, name):
            return getattr(tokenizer, name)

    cache = QwenIncrementalPrompt(UnlistedStartTokenizer() if change == "unlisted_start" else tokenizer)
    first = {"prompt": "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"}
    request = OmniEngineCoreRequest(
        request_id="r",
        prompt_token_ids=[11, 12, 13],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=10),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    cache.tokenize(first["prompt"])
    cache.complete_turn(first, request, None)
    generated = tokenizer.encode("hello", add_special_tokens=False)
    cache.record_output(generated, False)
    cache.record_output([tokenizer.convert_tokens_to_ids("<|im_end|>")], change != "unfinished")
    answer = tokenizer.decode(generated, skip_special_tokens=False)
    assert cache.can_append_turn("changed" if change == "assistant" else answer) == (
        change not in {"assistant", "unfinished"}
    )
    if change in {"assistant", "unfinished"}:
        assert cache.prepare_turn(first) == (first, None)
        return
    delta = {
        DELTA_KEY: True,
        "prompt": "<|im_end|>\n<|im_start|>user\nnew",
        "multi_modal_data": {"audio": ["new audio"]},
        "multi_modal_uuids": {"audio": ["rep0:new"]},
    }
    prepared, prefix = cache.prepare_turn(delta)
    assert prepared is delta
    assert prefix == [11, 12, 13] + generated
    request.prompt_token_ids = [21, 22]
    cache.complete_turn(delta, request, prefix)
    assert request.prompt_token_ids == prefix + [21, 22]
    assert request.for_engine_submission().prompt_token_ids == [21, 22]
    assert request.prompt_token_ids == prefix + [21, 22]
