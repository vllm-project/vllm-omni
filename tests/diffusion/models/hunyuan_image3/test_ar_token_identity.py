# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from dataclasses import dataclass

import pytest
import torch
from tokenizers import Tokenizer, decoders, models
from transformers import GenerationConfig, PreTrainedTokenizerFast

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerWrapper
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import ImageInfo
from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS
from vllm_omni.diffusion.models.hunyuan_image3.request_layout import (
    extract_hunyuan_ar_token_ids,
    prepare_hunyuan_layout,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import ar2diffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@dataclass
class _ARCompletion:
    cumulative_token_ids: list[int]
    text: str


@dataclass
class _AROutput:
    prompt_token_ids: list[int]
    outputs: list[_ARCompletion]


@dataclass
class _ImageProcessor:
    image_info: ImageInfo

    def build_image_info(
        self,
        image_size: tuple[int, int],
        *,
        add_guidance_token: bool = False,
        add_timestep_r_token: bool = False,
    ) -> ImageInfo:
        return self.image_info


@pytest.fixture
def wrapper():
    # Legal generated BPE sequences need not be canonical encodings of their
    # decoded text. abc changes length; def changes IDs at the same length.
    tokens = list(
        dict.fromkeys(
            [chr(i) for i in range(32, 127)]
            + ["\n", "ab", "bc", "abc", "de", "ef"]
            + list(HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS)
            + ["<pad>", "<timestep>"]
        )
    )
    tokenizer = Tokenizer(
        models.BPE(
            vocab={token: i for i, token in enumerate(tokens)},
            merges=[("a", "b"), ("ab", "c"), ("d", "e"), ("e", "f")],
        )
    )
    tokenizer.decoder = decoders.Fuse()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<|startoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<pad>",
        additional_special_tokens=list(HUNYUAN_IMAGE3_SPECIAL_TOKEN_IDS) + ["<timestep>"],
    )
    return TokenizerWrapper(hf_tokenizer)


def _image_info():
    return ImageInfo(
        image_type="gen_image",
        image_width=32,
        image_height=32,
        token_width=2,
        token_height=2,
        image_token_length=4,
        base_size=1024,
        ratio_index=0,
    )


def _prompt_ids(wrapper, system_prompt=None):
    output = wrapper.apply_chat_template(
        batch_prompt=["draw"],
        batch_system_prompt=[system_prompt],
        mode="gen_text",
        bot_task="think",
        sequence_template="instruct",
    )["output"]
    return output.tokens[0].tolist()


def _layout(wrapper, prompt_ids, generated_ids, **kwargs):
    kwargs.setdefault("validate_ar_kv_prefix", True)
    return wrapper.apply_chat_template(
        batch_prompt=["draw"],
        mode="gen_image",
        batch_gen_image_info=[_image_info()],
        sequence_template="instruct",
        batch_ar_token_ids=[(prompt_ids, generated_ids)],
        **kwargs,
    )["output"]


@pytest.mark.parametrize("pieces", [("a", "bc"), ("d", "ef")])
@pytest.mark.parametrize("cfg_factor", [1, 2])
def test_generated_bpe_ids_survive_ar_to_dit(wrapper, pieces, cfg_factor):
    prompt_ids = _prompt_ids(wrapper)
    original = [wrapper.tokenizer.convert_tokens_to_ids(piece) for piece in pieces]
    text = wrapper.decode(original, skip_special_tokens=False)
    assert wrapper.encode(text, add_special_tokens=False) != original
    generated = original + [wrapper.end_think_token_id]
    output = _layout(wrapper, prompt_ids, generated, cfg_factor=cfg_factor)

    boundary = len(prompt_ids) + len(original)
    assert output.think_recaption_end_pos[0] == [boundary]
    assert output.tokens[0, : boundary + 1].tolist() == prompt_ids + generated
    assert output.gen_timestep_scatter_index[0, 0] > boundary
    assert output.gen_image_mask.shape == output.tokens.shape
    assert output.real_pos[0, 0] == output.tokens.shape[1]


@pytest.mark.skipif(
    not os.environ.get("HUNYUAN_IMAGE3_TOKENIZER_PATH"), reason="Set HUNYUAN_IMAGE3_TOKENIZER_PATH to tokenizer.json"
)
@pytest.mark.parametrize(
    ("original", "rebuilt"),
    [([110236, 103550], [102111, 100420, 100406]), ([114980, 100449], [67494, 114745])],
)
@pytest.mark.parametrize("cfg_factor", [1, 2])
def test_reported_token_sequences_keep_original_ids(original, rebuilt, cfg_factor):
    # Read only the tokenizer data; loading checkpoint Python or weights is
    # unnecessary for the two saved counterexamples.
    wrapper = TokenizerWrapper(
        PreTrainedTokenizerFast(
            tokenizer_file=os.environ["HUNYUAN_IMAGE3_TOKENIZER_PATH"],
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<pad>",
        )
    )
    assert wrapper.encode(wrapper.decode(original), add_special_tokens=False) == rebuilt
    prompt_ids = _prompt_ids(wrapper)
    generated = original + [wrapper.end_think_token_id]
    output = _layout(wrapper, prompt_ids, generated, cfg_factor=cfg_factor)
    assert output.tokens[0, : len(prompt_ids) + len(generated)].tolist() == prompt_ids + generated
    assert output.think_recaption_end_pos[0] == [len(prompt_ids) + len(original)]


def test_roundtrip_stable_text_layout_is_unchanged(wrapper):
    generated = wrapper.encode("body</think>", add_special_tokens=False)
    raw = _layout(wrapper, _prompt_ids(wrapper), generated)
    text = wrapper.apply_chat_template(
        batch_prompt=["draw"],
        mode="gen_image",
        batch_gen_image_info=[_image_info()],
        sequence_template="instruct",
        batch_cot_text=["<think>body</think>"],
    )["output"]
    for name in ("tokens", "gen_image_mask", "gen_timestep_scatter_index", "real_pos"):
        assert torch.equal(getattr(raw, name), getattr(text, name))
    assert raw.think_recaption_end_pos == text.think_recaption_end_pos


def test_generated_image_metadata_without_cot_keeps_legacy_layout(wrapper):
    prompt_ids = _prompt_ids(wrapper)
    generated = wrapper.encode("<img_ratio_0>", add_special_tokens=False)
    raw = _layout(wrapper, prompt_ids, generated)
    legacy = wrapper.apply_chat_template(
        batch_prompt=["draw"],
        mode="gen_image",
        batch_gen_image_info=[_image_info()],
        sequence_template="instruct",
    )["output"]
    assert torch.equal(raw.tokens, legacy.tokens)
    assert raw.think_recaption_end_pos == [[None]]


def test_raw_ids_take_precedence_over_decoded_text_without_special_tokens(wrapper):
    generated = wrapper.encode("body</think>", add_special_tokens=False)
    output = _layout(wrapper, _prompt_ids(wrapper), generated, batch_cot_text=["body"])
    assert output.think_recaption_end_pos[0][0] == len(_prompt_ids(wrapper)) + len(generated) - 1


def test_recaption_boundary_and_image_suffix_are_not_duplicated(wrapper):
    prompt_ids = _prompt_ids(wrapper)
    generated = wrapper.encode(
        "body</think><recaption>image</recaption><answer><boi><img_size_1024><img_ratio_0>",
        add_special_tokens=False,
    )
    output = _layout(wrapper, prompt_ids, generated)
    assert output.think_recaption_end_pos[0][0] == len(prompt_ids) + generated.index(wrapper.end_recaption_token_id)
    timestep = int(output.gen_timestep_scatter_index[0, 0])
    assert output.tokens[0, :timestep].tolist() == prompt_ids + generated
    assert output.tokens[0].tolist().count(wrapper.boi_token_id) == 1


def test_custom_system_prompt_and_whitespace_preserve_original_prefix(wrapper):
    system = "  custom\n system  "
    prompt_ids = _prompt_ids(wrapper, system)
    generated = wrapper.encode("body</think>", add_special_tokens=False)
    output = _layout(wrapper, prompt_ids, generated, batch_system_prompt=[system])
    assert output.tokens[0, : len(prompt_ids) + len(generated)].tolist() == prompt_ids + generated


def test_equal_length_prompt_mismatch_is_rejected_before_kv_injection(wrapper):
    prompt_ids = _prompt_ids(wrapper)
    prompt_ids[1] += 1
    generated = wrapper.encode("body</think>", add_special_tokens=False)
    with pytest.raises(ValueError, match="prefix token IDs"):
        _layout(wrapper, prompt_ids, generated)


def test_dropping_cached_thought_is_rejected(wrapper):
    generated = wrapper.encode("body</think><recaption>image</recaption>", add_special_tokens=False)
    with pytest.raises(ValueError, match="prefix token IDs"):
        _layout(wrapper, _prompt_ids(wrapper), generated, drop_think=True)


def test_dropping_thought_without_kv_reuse_remains_supported(wrapper):
    generated = wrapper.encode("body</think><recaption>image</recaption>", add_special_tokens=False)
    raw = _layout(wrapper, _prompt_ids(wrapper), generated, drop_think=True, validate_ar_kv_prefix=False)
    legacy = wrapper.apply_chat_template(
        batch_prompt=["draw"],
        mode="gen_image",
        batch_gen_image_info=[_image_info()],
        sequence_template="instruct",
        batch_cot_text=["<think>body</think><recaption>image</recaption>"],
        drop_think=True,
    )["output"]
    assert torch.equal(raw.tokens, legacy.tokens)


def test_bridge_copies_original_ids_and_layout_planner_consumes_them(wrapper):
    prompt_ids = _prompt_ids(wrapper)
    generated = [
        wrapper.tokenizer.convert_tokens_to_ids("a"),
        wrapper.tokenizer.convert_tokens_to_ids("bc"),
        wrapper.end_think_token_id,
    ]
    source = _AROutput(
        prompt_token_ids=prompt_ids,
        outputs=[_ARCompletion(cumulative_token_ids=generated, text="abc")],
    )
    prompt = ar2diffusion([source], prompt={"prompt": "draw"})
    assert prompt["extra"]["ar_prompt_token_ids"] == prompt_ids
    assert prompt["extra"]["ar_prompt_token_ids"] is not prompt_ids
    assert prompt["extra"]["ar_generated_token_ids"] == generated
    assert prompt["extra"]["ar_generated_token_ids"] is not generated
    request = OmniDiffusionRequest(
        prompt=prompt,
        kv_transfer_params={"num_transfer_tokens": len(prompt_ids) + len(generated) - 1},
        request_id="ar-token-test",
        sampling_params=OmniDiffusionSamplingParams(
            height=32, width=32, guidance_scale=1.0, guidance_scale_provided=True, extra_args={}
        ),
    )
    prepared = prepare_hunyuan_layout(
        request,
        tokenizer_wrapper=wrapper,
        image_processor=_ImageProcessor(_image_info()),
        generation_config=GenerationConfig(sequence_template="instruct", drop_think=False),
        image_base_size=1024,
    )
    worker = _layout(wrapper, prompt_ids, generated)
    assert torch.equal(prepared.tokenizer_output.tokens, worker.tokens)
    assert prepared.tokenizer_output.think_recaption_end_pos == worker.think_recaption_end_pos


def test_text_only_requests_keep_legacy_path_and_partial_metadata_is_rejected():
    assert extract_hunyuan_ar_token_ids(["draw", {"prompt": "draw"}]) is None
    with pytest.raises(ValueError, match="both prompt and generated"):
        extract_hunyuan_ar_token_ids([{"extra": {"ar_prompt_token_ids": [1]}}])
