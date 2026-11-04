# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import patch

import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, processors, trainers
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


def test_append_tokenizes_only_suffix_and_matches_full_encoding(tokenizer):
    incremental = QwenIncrementalPrompt(tokenizer)
    first = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    assert incremental.tokenize(first) == tokenizer.encode(first)
    next_prompt = first + "world<|im_end|>\n<|im_start|>user\n天气<|im_end|>\n<|im_start|>assistant\n"
    expected = tokenizer.encode(next_prompt)
    with patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode:
        assert incremental.tokenize(next_prompt) == expected
    assert encode.call_count == 1
    assert encode.call_args.args[0] == next_prompt[len(first) - len("assistant\n") :]


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


@pytest.mark.parametrize("options", [{"lstrip": True}, {"rstrip": True}, {"single_word": True}])
def test_context_dependent_boundary_disables_reuse(tokenizer, options):
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken("<|im_start|>", special=True, **options)]})
    incremental = QwenIncrementalPrompt(tokenizer)
    prompt = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    incremental.tokenize(prompt)
    prompt += "world<|im_end|>\n<|im_start|>assistant\n"
    expected = tokenizer.encode(prompt)
    with patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode:
        assert incremental.tokenize(prompt) == expected
    assert encode.call_args.args[0] == prompt


def test_no_chat_boundary_clears_previous_prefix(tokenizer):
    incremental = QwenIncrementalPrompt(tokenizer)
    prompt = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    incremental.tokenize(prompt)
    assert incremental.tokenize("hello world") == tokenizer.encode("hello world")
    with patch.object(tokenizer, "encode", wraps=tokenizer.encode) as encode:
        assert incremental.tokenize(prompt) == tokenizer.encode(prompt)
    assert encode.call_args_list[0].args[0] == prompt


@pytest.mark.parametrize("change", [None, "unlisted_start", "audio", "assistant", "system", "prune", "unfinished"])
def test_turn_delta_keeps_sampled_tokens_and_only_new_media(tokenizer, change):
    from vllm.sampling_params import SamplingParams

    from vllm_omni.engine import OmniEngineCoreRequest
    from vllm_omni.model_executor.models.qwen3_omni.duplex.incremental_prompt import HISTORY_KEY

    if change == "unlisted_start":

        class UnlistedStartTokenizer:
            # vLLM's real Qwen CachedTokenizer encodes im_start atomically but
            # does not list it among all_special_tokens.
            all_special_tokens = [token for token in tokenizer.all_special_tokens if token != "<|im_start|>"]

            def __getattr__(self, name):
                return getattr(tokenizer, name)

        cache = QwenIncrementalPrompt(UnlistedStartTokenizer())
    else:
        cache = QwenIncrementalPrompt(tokenizer)
    first: dict = {
        "prompt": "<|im_start|>user\naudio<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {"audio": ["old audio"]},
        HISTORY_KEY: [{"role": "user", "audio_payload": "old audio"}],
    }
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
    second: dict = {
        "prompt": first["prompt"] + answer + "<|im_end|>\n<|im_start|>user\nimage<|im_end|>\n<|im_start|>assistant\n",
        "multi_modal_data": {"audio": ["old audio", "new audio"], "image": ["new image"]},
        "multi_modal_uuids": {"audio": ["rep0:old", "rep0:new"], "image": ["rep0:image"]},
        HISTORY_KEY: first[HISTORY_KEY]
        + [{"role": "assistant", "content": answer}, {"role": "user", "content": "image"}],
    }
    if change == "audio":
        second[HISTORY_KEY][0] = {"role": "user", "audio_payload": "edited audio"}
    elif change == "assistant":
        second["prompt"] = second["prompt"].replace(answer, "changed answer")
    elif change == "system":
        second["prompt"] = "system change" + second["prompt"]
    elif change == "prune":
        second[HISTORY_KEY] = second[HISTORY_KEY][1:]
    prepared, prefix = cache.prepare_turn(second)
    if change not in {None, "unlisted_start"}:
        assert prefix is None
        assert prepared is second
        return
    assert prefix == [11, 12, 13] + generated
    assert prepared["multi_modal_data"] == {"audio": ["new audio"], "image": ["new image"]}
    assert prepared["multi_modal_uuids"] == {"audio": ["rep0:new"], "image": ["rep0:image"]}
    assert second["multi_modal_uuids"]["audio"] == ["rep0:old", "rep0:new"]
    assert prepared["prompt"].startswith("<|im_end|>\n<|im_start|>user")
    request.prompt_token_ids = [21, 22]
    cache.complete_turn(second, request, prefix)
    assert request.prompt_token_ids == prefix + [21, 22]
    assert request.for_engine_submission().prompt_token_ids == [21, 22]
    assert request.prompt_token_ids == prefix + [21, 22]  # frontend snapshot is immutable to transport
