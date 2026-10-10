# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import (
    ChatterboxT3ForConditionalGeneration,
    T3Heads,
    prefill_embeds,
    prefill_slice,
    speech_logits,
    split_t3_weights,
)
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning, build_prompt
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(scope="module")
def config() -> ChatterboxConfig:
    cfg = ChatterboxConfig()
    cfg.hidden_size = 16  # the layout does not depend on width; this keeps the test small
    return cfg


@pytest.fixture(scope="module")
def heads(config: ChatterboxConfig) -> T3Heads:
    torch.manual_seed(0)
    return T3Heads(config)


@pytest.fixture(scope="module")
def talker(heads: T3Heads, config: ChatterboxConfig) -> ChatterboxT3ForConditionalGeneration:
    """The stage without its backbone: __init__ needs a vLLM config, the embedding hooks only these two."""
    model = object.__new__(ChatterboxT3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config, model.heads = config, heads
    return model


@pytest.mark.parametrize("cond_len", [150, 375])
def test_prefill_layout_follows_the_conditioning_length_001(
    heads: T3Heads, config: ChatterboxConfig, cond_len: int
) -> None:
    """Speaker slot, the prompt tokens, the text, one start-of-speech slot."""
    text_ids = torch.tensor([5, 6, 7])
    cond_tokens = torch.randint(0, 6561, (cond_len,))
    speaker = torch.randn(1, 256)

    embeds = prefill_embeds(heads, text_ids, cond_tokens, speaker, config.start_speech_token)

    assert embeds.shape == (1 + cond_len + 3 + 1, config.hidden_size)
    assert torch.equal(embeds[0], heads.spkr_enc(speaker)[0])
    assert torch.equal(embeds[1 : 1 + cond_len], heads.speech_emb(cond_tokens))
    assert torch.equal(embeds[1 + cond_len : -1], heads.text_emb(text_ids))
    assert torch.equal(embeds[-1], heads.speech_emb.weight[config.start_speech_token])


def test_prefill_split_with_one_token_tail_matches_one_call_001(heads: T3Heads, config: ChatterboxConfig) -> None:
    """However the scheduler splits the prompt, each span gets its own rows."""
    text_ids = torch.tensor([5, 6, 7])
    cond_tokens = torch.randint(0, 6561, (375,))
    speaker = torch.randn(1, 256)
    whole = prefill_embeds(heads, text_ids, cond_tokens, speaker, config.start_speech_token)
    total = whole.shape[0]

    head = prefill_slice(heads, config, text_ids, cond_tokens, speaker, 0, total - 1)
    tail = prefill_slice(heads, config, text_ids, cond_tokens, speaker, total - 1, 1)

    assert torch.equal(torch.cat([head, tail]), whole)
    # The one-token tail is the prompt's last row, not the embedding a decode
    # step would give the same placeholder id.
    assert torch.equal(tail[0], heads.speech_emb.weight[config.start_speech_token])


def test_prefill_span_past_the_prompt_returns_only_the_prompt_rows_001(
    heads: T3Heads, config: ChatterboxConfig
) -> None:
    """A preempted request is recomputed from zero with its generated tokens in the span."""
    text_ids = torch.tensor([5, 6, 7])
    cond_tokens = torch.randint(0, 6561, (375,))
    speaker = torch.randn(1, 256)
    whole = prefill_embeds(heads, text_ids, cond_tokens, speaker, config.start_speech_token)
    total = whole.shape[0]

    from_zero = prefill_slice(heads, config, text_ids, cond_tokens, speaker, 0, total + 2)
    inside = prefill_slice(heads, config, text_ids, cond_tokens, speaker, total - 3, 5)

    assert torch.equal(from_zero, whole)
    assert torch.equal(inside, whole[-3:])


def test_preprocess_embeds_the_generated_tokens_after_the_prompt_001(
    talker: ChatterboxT3ForConditionalGeneration, heads: T3Heads, config: ChatterboxConfig
) -> None:
    """The rows past the prompt are speech ids, embedded as a decode step would."""
    text_ids = [5, 6, 7]
    cond_tokens = torch.randint(0, 6561, (375,))
    speaker = torch.randn(1, 256)
    whole = prefill_embeds(heads, torch.tensor(text_ids), cond_tokens, speaker, config.start_speech_token)
    total = whole.shape[0]
    generated = torch.tensor([11, 12])
    input_ids = torch.cat([torch.full((total,), config.start_speech_token), generated])

    _, embeds, _ = talker.preprocess(
        input_ids,
        None,
        _omni_is_prefill=True,
        _omni_prompt_len=total,
        _omni_num_computed_tokens=0,
        ids={"prompt": text_ids, "speech_token": cond_tokens.tolist()},
        embed={"voice": speaker},
    )

    assert embeds.shape == (total + 2, config.hidden_size)
    assert torch.equal(embeds[:total], whole)
    assert torch.equal(embeds[total:], heads.speech_emb(generated))


@pytest.mark.parametrize("text_len", [1, 3, 671])
@pytest.mark.parametrize("cond_len", [125, 150, 375])
def test_the_placeholder_prompt_is_as_long_as_its_embeddings_001(
    heads: T3Heads, config: ChatterboxConfig, cond_len: int, text_len: int
) -> None:
    """``build_prompt`` counts the placeholders, ``prefill_embeds`` builds what replaces them.

    A longer placeholder span would be filled past the prompt with the
    start token's embedding, a shorter one would drop the start-of-speech
    slot; either ends generation wrongly with no error. Nothing checks it
    at run time, so the two are pinned to each other here, through the
    request payload ``preprocess`` is handed.
    """
    conditioning = VoiceConditioning(
        cond_tokens=torch.randint(0, 6561, (1, cond_len)),
        speaker_emb=torch.randn(1, 256),
        prompt_token=torch.randint(0, 6561, (1, 250)),
        prompt_feat=torch.randn(1, 500, 80),
        embedding=torch.randn(1, 192),
    )
    prompt = build_prompt(list(range(text_len)), conditioning, config)
    info = prompt["additional_information"]

    embeds = prefill_embeds(
        heads,
        torch.tensor(info["ids"]["prompt"]),
        torch.tensor(info["ids"]["speech_token"]),
        info["embed"]["voice"],
        config.start_speech_token,
    )

    assert len(prompt["prompt_token_ids"]) == embeds.shape[0]


def test_batched_decode_embedding_matches_per_request_preprocess_001(
    talker: ChatterboxT3ForConditionalGeneration, config: ChatterboxConfig
) -> None:
    """The runner hands a step's one-token decode rows to one call instead of one call each."""
    input_ids = torch.tensor([11, 6560, 0])
    req_infos = [
        {
            "_omni_is_prefill": False,
            "_omni_prompt_len": 381,
            "_omni_num_computed_tokens": 381 + row,
            "ids": {},
            "embed": {},
        }
        for row in range(3)
    ]

    ids, embeds, updates = talker.preprocess_decode_batch(input_ids=input_ids, req_infos=req_infos)

    each = [talker.preprocess(input_ids[row : row + 1], None, **info) for row, info in enumerate(req_infos)]
    assert ids is input_ids
    assert embeds.shape == (3, config.hidden_size)
    assert torch.equal(embeds, torch.cat([row_embeds for _, row_embeds, _ in each]))
    assert updates == [update for _, _, update in each] == [{}, {}, {}]


def test_forward_returns_the_backbones_tensor_unwrapped_001(
    talker: ChatterboxT3ForConditionalGeneration, config: ChatterboxConfig
) -> None:
    """Model Runner V2 sizes its CUDA graph's output from what ``forward`` returns."""
    hidden = torch.randn(3, config.hidden_size)
    seen: list[tuple] = []

    def backbone(*inputs: torch.Tensor | None) -> torch.Tensor:
        seen.append(inputs)
        return hidden

    talker.tfmr = backbone
    input_ids, positions, embeds = torch.tensor([1, 2, 3]), torch.arange(3), torch.randn(3, config.hidden_size)

    assert talker.forward(input_ids, positions, None, embeds, seq_token_counts=[3]) is hidden
    assert seen == [(input_ids, positions, None, embeds)]


@pytest.mark.parametrize("padded_rows", [7, 16], ids=["as scheduled", "padded by the graph"])
def test_every_request_of_a_step_gets_a_payload_from_the_runner_001(
    talker: ChatterboxT3ForConditionalGeneration, config: ChatterboxConfig, padded_rows: int
) -> None:
    """The runner-side transport calls the chunk processor only for a request the step has a payload for.

    Two decode rows and a five-token prefill, cut per request by the
    runner's own slicing, each end up with one key that is not a client
    output. It holds one false flag per token row and nothing else.
    """
    hidden = torch.randn(padded_rows, config.hidden_size)

    output = talker.make_omni_output(hidden, model_intermediate_buffer=[{}, {}, {}])

    assert output.text_hidden_states is hidden
    inter_stage, client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        output.multimodal_outputs, np.array([0, 1, 2]), np.array([1, 1, 5]), 3, 7, padded_rows
    )
    assert client is None
    assert [list(payload) for payload in inter_stage] == [["meta.codec_frame_valid"]] * 3
    assert [payload["meta.codec_frame_valid"].tolist() for payload in inter_stage] == [[False], [False], [False] * 5]


def model_state(talker: ChatterboxT3ForConditionalGeneration, payloads: list[dict]) -> OmniModelState:
    """Model Runner V2's per-model state around the talker, short of the engine.

    Only what ``run_preprocess`` reads is set; the state resolves the
    model's decode hook itself, as it does at load.
    """
    state = object.__new__(OmniModelState)
    state.model = talker
    state.has_preprocess = True
    state._static_inputs_embeds = None
    state._decode_preprocess_is_identity = False
    state.intermediate_buffer = OmniIntermediateBuffer(len(payloads))
    for slot, payload in enumerate(payloads):
        state.intermediate_buffer.buffers[slot] = {"req_id": f"request-{slot}", **payload}
    return state


def test_model_runner_v2_embeds_a_mixed_step_through_the_hooks_001(
    talker: ChatterboxT3ForConditionalGeneration, heads: T3Heads, config: ChatterboxConfig
) -> None:
    """V2 orders decode rows first, embeds every row itself, then calls the hooks.

    Rows: two decode rows; a one-token row of a request being recomputed
    after preemption, which is past its prompt and so goes to the decode
    hook too; a prefill that runs two generated tokens past its prompt; and
    a whole prompt. The state unpacks five values from the decode hook.
    """
    text_ids = [5, 6, 7]
    cond_tokens = torch.randint(0, 6561, (150,))
    speaker = torch.randn(1, 256)
    prompt = prefill_embeds(heads, torch.tensor(text_ids), cond_tokens, speaker, config.start_speech_token)
    total = prompt.shape[0]
    payload = {"ids": {"prompt": text_ids, "speech_token": cond_tokens.tolist()}, "embed": {"voice": speaker}}
    state = model_state(talker, [payload] * 5)
    placeholder = torch.full((total,), config.start_speech_token)
    spans = [
        torch.tensor([11]),
        torch.tensor([6560]),
        torch.tensor([12]),
        torch.cat([placeholder, torch.tensor([13, 14])]),
        placeholder,
    ]
    input_ids = torch.cat(spans)
    lengths = [span.numel() for span in spans]
    batch = SimpleNamespace(
        idx_mapping_np=np.arange(5),
        num_reqs=5,
        num_scheduled_tokens=lengths,
        query_start_loc_np=np.cumsum([0, *lengths]),
        num_tokens=input_ids.numel(),
        num_computed_tokens_np=np.array([total + 4, total + 9, total + 1, 0, 0]),
    )
    model_inputs = {"input_ids": input_ids.clone(), "inputs_embeds": talker.embed_input_ids(input_ids)}

    assert OmniModelState._resolve_decode_preprocess(talker) == talker.preprocess_decode_batch_mrv2
    state.run_preprocess(batch, model_inputs, SimpleNamespace(prompt_len=np.full(5, total)))

    assert torch.equal(model_inputs["input_ids"], input_ids)
    assert torch.equal(
        model_inputs["inputs_embeds"],
        torch.cat(
            [heads.speech_emb(torch.tensor([11, 6560, 12])), prompt, heads.speech_emb(torch.tensor([13, 14])), prompt]
        ),
    )
    # No hook wrote anything back into a request's payload.
    assert [set(buffer) for buffer in state.intermediate_buffer.buffers] == [{"req_id", "ids", "embed"}] * 5


def test_logits_are_the_speech_heads_with_the_start_token_masked_001(heads: T3Heads, config: ChatterboxConfig) -> None:
    """The sampler's vocabulary is the speech head's; the text vocabulary only sizes the text embedding."""
    logits = speech_logits(heads, torch.randn(2, config.hidden_size), config)

    assert logits.shape == (2, config.vocab_size) == (2, heads.speech_head.out_features)
    assert heads.text_emb.num_embeddings == config.text_vocab_size
    assert torch.isfinite(logits[:, : config.start_speech_token]).all()
    assert (logits[:, config.start_speech_token] == float("-inf")).all()
    assert torch.isfinite(logits[:, config.stop_speech_token]).all()


def test_weight_routing_sends_the_backbone_to_vllm_and_the_rest_to_the_heads_001() -> None:
    weights = [
        ("tfmr.h.0.attn.c_attn.weight", torch.zeros(1)),
        ("tfmr.wte.weight", torch.zeros(1)),
        ("tfmr.wpe.weight", torch.zeros(1)),
        ("text_head.weight", torch.zeros(1)),
        ("text_emb.weight", torch.zeros(1)),
        ("speech_emb.weight", torch.zeros(1)),
        ("speech_head.weight", torch.zeros(1)),
        ("speech_head.bias", torch.zeros(1)),
        ("cond_enc.spkr_enc.weight", torch.zeros(1)),
        ("cond_enc.spkr_enc.bias", torch.zeros(1)),
    ]

    backbone, heads = split_t3_weights(iter(weights))

    assert [name for name, _ in backbone] == ["h.0.attn.c_attn.weight", "wte.weight", "wpe.weight"]
    assert set(heads) == {
        "text_emb.weight",
        "speech_emb.weight",
        "speech_head.weight",
        "speech_head.bias",
        "spkr_enc.weight",
        "spkr_enc.bias",
    }


def test_weight_routing_rejects_a_key_it_does_not_know_001() -> None:
    with pytest.raises(KeyError, match="cond_enc.perceiver.weight"):
        split_t3_weights(iter([("cond_enc.perceiver.weight", torch.zeros(1))]))


def test_prefix_caching_is_refused_at_startup_001() -> None:
    """Every prompt is the same placeholder id, so a cache would match on length alone."""
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True))
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        ChatterboxT3ForConditionalGeneration(vllm_config=vllm_config)
