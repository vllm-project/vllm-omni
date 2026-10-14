# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.chatterbox.chatterbox_original_t3 import ChatterboxOriginalT3
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import (
    HEAD_KEYS,
    ChatterboxT3ForConditionalGeneration,
    T3Heads,
    prefill_embeds,
    speech_logits,
    split_t3_weights,
)
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning, build_prompt
from vllm_omni.model_executor.models.chatterbox.original_heads import OriginalHeads
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
def test_prefill_layout_follows_the_conditioning_length(
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


@pytest.mark.parametrize(
    "schedule",
    [
        lambda total: (0, total),
        lambda total: (0, total - 1),
        lambda total: (total - 1, 1),
        lambda total: (0, total + 2),
        lambda total: (total - 3, 5),
        lambda total: (10, total - 8),
    ],
    ids=[
        "whole prompt",
        "all but the last row",
        "one-row tail",
        "recomputed past the prompt",
        "across the end",
        "middle",
    ],
)
def test_preprocess_embeds_each_scheduled_span_as_one_call_would(
    talker: ChatterboxT3ForConditionalGeneration,
    heads: T3Heads,
    config: ChatterboxConfig,
    schedule: Callable[[int], tuple[int, int]],
) -> None:
    """However the scheduler splits or recomputes the prompt, a span gets exactly its own rows.

    Rows inside the prompt are the prompt's embedding; rows past it are the
    speech ids a preempted request had generated, embedded as a decode step
    would. ``schedule`` gives the tokens computed before the step and the
    step's span from the prompt's length.
    """
    torch.manual_seed(0)
    text_ids = [5, 6, 7]
    cond_tokens = torch.randint(0, 6561, (375,))
    speaker = torch.randn(1, 256)
    whole = prefill_embeds(heads, torch.tensor(text_ids), cond_tokens, speaker, config.start_speech_token)
    total = whole.shape[0]
    num_computed, span = schedule(total)
    generated = torch.tensor([11, 12])
    every_id = torch.cat([torch.full((total,), config.start_speech_token), generated])

    _, embeds, update = talker.preprocess(
        every_id[num_computed : num_computed + span],
        None,
        _omni_is_prefill=True,
        _omni_prompt_len=total,
        _omni_num_computed_tokens=num_computed,
        ids={"prompt": text_ids, "speech_token": cond_tokens.tolist()},
        embed={"voice": speaker},
    )

    assert embeds.shape == (span, config.hidden_size)
    assert torch.equal(embeds, torch.cat([whole, heads.speech_emb(generated)])[num_computed : num_computed + span])
    assert update == {}


@pytest.mark.parametrize("text_len", [1, 3, 671])
@pytest.mark.parametrize("cond_len", [125, 150, 375])
def test_the_placeholder_prompt_is_as_long_as_its_embeddings(
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


def test_batched_decode_embedding_matches_per_request_preprocess(
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
    # Both are the speech table's rows: a decode row is a sampled speech id.
    assert torch.equal(embeds, talker.heads.speech_emb.weight[input_ids])
    assert updates == [update for _, _, update in each] == [{}, {}, {}]


def test_forward_returns_the_backbones_tensor_unwrapped(
    talker: ChatterboxT3ForConditionalGeneration, config: ChatterboxConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model Runner V2 sizes its CUDA graph's output from what ``forward`` returns."""
    hidden = torch.randn(3, config.hidden_size)
    seen: list[tuple] = []

    def backbone(*inputs: torch.Tensor | None) -> torch.Tensor:
        seen.append(inputs)
        return hidden

    monkeypatch.setattr(talker, "tfmr", backbone, raising=False)
    input_ids, positions, embeds = torch.tensor([1, 2, 3]), torch.arange(3), torch.randn(3, config.hidden_size)

    assert talker.forward(input_ids, positions, None, embeds, seq_token_counts=[3]) is hidden
    assert seen == [(input_ids, positions, None, embeds)]


@pytest.mark.parametrize("padded_rows", [7, 16], ids=["as scheduled", "padded by the graph"])
def test_every_request_of_a_step_gets_a_payload_from_the_runner(
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


def test_model_runner_v2_embeds_a_mixed_step_through_the_hooks(
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


def test_logits_are_the_speech_heads_with_the_start_token_masked(heads: T3Heads, config: ChatterboxConfig) -> None:
    """The sampler's vocabulary is the speech head's; the text vocabulary only sizes the text embedding."""
    logits = speech_logits(heads, torch.randn(2, config.hidden_size), config)

    assert logits.shape == (2, config.vocab_size) == (2, heads.speech_head.out_features)
    assert heads.text_emb.num_embeddings == config.text_vocab_size
    assert torch.isfinite(logits[:, : config.start_speech_token]).all()
    assert (logits[:, config.start_speech_token] == float("-inf")).all()
    assert torch.isfinite(logits[:, config.stop_speech_token]).all()


def test_the_key_map_fills_every_head_parameter_and_nothing_else(config: ChatterboxConfig) -> None:
    assert sorted(HEAD_KEYS.values()) == sorted(T3Heads(config).state_dict())


def test_weight_routing_sends_the_backbone_to_vllm_and_the_rest_to_the_heads(config: ChatterboxConfig) -> None:
    """Every head tensor reaches its own parameter, by value; the backbone keeps its own."""
    torch.manual_seed(1)
    state = T3Heads(config).state_dict()
    checkpoint = {name: state[parameter] for name, parameter in HEAD_KEYS.items()}
    backbone_keys = ["tfmr.h.0.attn.c_attn.weight", "tfmr.wte.weight", "tfmr.wpe.weight"]
    weights = [(name, torch.full((1,), float(i))) for i, name in enumerate(backbone_keys)]
    weights += [("text_head.weight", torch.zeros(1)), *checkpoint.items()]

    backbone, routed = split_t3_weights(iter(weights))

    assert [(name, float(tensor)) for name, tensor in backbone] == [
        ("h.0.attn.c_attn.weight", 0.0),
        ("wte.weight", 1.0),
        ("wpe.weight", 2.0),
    ]
    heads = T3Heads(config)
    heads.load_state_dict(routed, strict=True)
    assert torch.equal(heads.speech_head.bias, checkpoint["speech_head.bias"])
    assert torch.equal(heads.spkr_enc.weight, checkpoint["cond_enc.spkr_enc.weight"])
    assert all(torch.equal(heads.state_dict()[parameter], state[parameter]) for parameter in HEAD_KEYS.values())


def test_weight_routing_rejects_a_key_it_does_not_know() -> None:
    with pytest.raises(KeyError, match="cond_enc.perceiver.weight"):
        split_t3_weights(iter([("cond_enc.perceiver.weight", torch.zeros(1))]))


def test_prefix_caching_is_refused_at_startup() -> None:
    """Every prompt is the same placeholder id, so a cache would match on length alone."""
    vllm_config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True))
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        ChatterboxT3ForConditionalGeneration(vllm_config=vllm_config)


class RecordingBackbone(nn.Module):
    """vLLM's backbone loader as the stage sees it: it returns the names it loaded."""

    def __init__(self) -> None:
        super().__init__()
        self.loaded: list[tuple[str, torch.Tensor]] = []

    def load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> set[str]:
        self.loaded = list(weights)
        return {name for name, _ in self.loaded}


@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_load_weights_fills_the_heads_and_names_every_parameter(variant: str) -> None:
    """vLLM fails the load on a parameter the returned set does not name.

    The heads are loaded strictly, from the checkpoint's own names and
    values; the backbone gets its keys without the ``tfmr.`` prefix.
    """
    config = ChatterboxConfig(variant)
    config.hidden_size = 8
    stage_cls, heads_cls = (
        (ChatterboxT3ForConditionalGeneration, T3Heads) if variant == "turbo" else (ChatterboxOriginalT3, OriginalHeads)
    )
    stage = object.__new__(stage_cls)
    nn.Module.__init__(stage)
    stage.heads = heads_cls(config)
    stage.tfmr = RecordingBackbone()
    torch.manual_seed(2)
    state = heads_cls(config).state_dict()
    checkpoint = {name: state[parameter] for name, parameter in HEAD_KEYS.items()} if variant == "turbo" else state

    loaded = stage.load_weights(
        [("tfmr.layers.0.weight", torch.ones(1)), ("text_head.weight", torch.zeros(1)), *checkpoint.items()]
    )

    assert loaded == {"tfmr.layers.0.weight"} | {f"heads.{name}" for name in state}
    assert [name for name, _ in stage.tfmr.loaded] == ["layers.0.weight"]
    assert all(torch.equal(stage.heads.state_dict()[name], tensor) for name, tensor in state.items())


def test_preprocess_embeds_in_the_heads_dtype_whatever_the_voices() -> None:
    """Stage 0 runs in bfloat16 while the request's voice arrives as float32."""
    config = ChatterboxConfig()
    config.hidden_size = 16
    torch.manual_seed(3)
    stage = object.__new__(ChatterboxT3ForConditionalGeneration)
    nn.Module.__init__(stage)
    stage.config, stage.heads = config, T3Heads(config).to(torch.bfloat16)
    speaker = torch.randn(1, 256)
    cond_tokens = [1, 2, 3]
    total = 1 + len(cond_tokens) + 2 + 1

    _, embeds, _ = stage.preprocess(
        torch.full((total,), config.start_speech_token),
        None,
        _omni_is_prefill=True,
        _omni_prompt_len=total,
        _omni_num_computed_tokens=0,
        ids={"prompt": [5, 6], "speech_token": cond_tokens},
        embed={"voice": speaker},
    )

    assert embeds.dtype == torch.bfloat16
    assert embeds.shape == (total, config.hidden_size)
    assert torch.equal(embeds[0], stage.heads.spkr_enc(speaker.to(torch.bfloat16))[0])
