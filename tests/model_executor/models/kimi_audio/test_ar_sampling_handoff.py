# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU request -> heads -> runner sampler -> next-input boundary tests.

Native SamplingParams/Metadata, LogitsProcessor, Omni request-context/dispatch/output
methods and scheduler stop checks execute here. Heads and hidden states are synthetic;
this does not start a GPU worker, attention kernels or an end-to-end engine.
"""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.model_executor.layers import logits_processor
from vllm.sampling_params import RepetitionDetectionParams, SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import RequestStatus
from vllm.v1.sample.metadata import SamplingMetadata

from tests.model_executor.models.kimi_audio.runtime import cpu_pp_group as cpu_pp_group
from tests.model_executor.models.kimi_audio.runtime import registered_model_runtime as registered_model_runtime
from vllm_omni.model_executor.models.kimi_audio.audio_processing import prepare_kimi_audio_inputs
from vllm_omni.model_executor.models.kimi_audio.kimi_audio import KimiAudioForConditionalGeneration
from vllm_omni.model_executor.models.kimi_audio.kimi_audio_ar_stage import KimiAudioARStage
from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioPromptBuilder, KimiAudioSpecialTokens
from vllm_omni.model_executor.models.kimi_audio.sampling import KimiAudioSamplingParams, sample_kimi_audio_step
from vllm_omni.model_executor.stage_input_processors.kimi_audio import prepare_kimi_audio_request
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.omni]
REFERENCE = json.loads((Path(__file__).parent / "fixtures/prompt_reference.json").read_text(encoding="utf-8"))
SPECIAL = KimiAudioSpecialTokens.from_vocab(REFERENCE["special_tokens"])
WORD = REFERENCE["text_tokens"]["你好"][0]
OTHER_WORD = REFERENCE["text_tokens"]["比较"][0]
CODE = REFERENCE["input_config"]["audio_token_offset"] + 17


class RequestSteps:
    """Supply bounded scheduled spans to the actual eager runner methods."""

    def __init__(self, model):
        stage = self.stage = model.model
        self.runner = object.__new__(GPUARModelRunner)
        self.runner.model = model
        self.runner.requests = {}
        self.runner.model_intermediate_buffer = {}
        self.runner.model_config = stage.vllm_config.model_config
        self.runner.vllm_config = stage.vllm_config
        self.runner._downstream_payload_cache = {}
        self.runner.omni_prefix_cache = None
        self.runner._async_chunk = False
        self.runner.supports_mm_inputs = False
        # Connector IO is outside this CPU boundary test. The real output
        # builder and client/inter-stage partitioning still execute.
        self.runner._should_accumulate_full_payload_output = lambda: False
        self.runner.get_omni_connector_output = lambda: None
        self.runner._omni_query_start_loc_model_kwarg = False
        self.runner.sampler = lambda **_: pytest.fail("Kimi-Audio fell back to the single-stream sampler")
        self.generators = {}
        self.outputs = []

    def add(self, rid, mode, params, text="你好，Kimi。"):
        builder = KimiAudioPromptBuilder(REFERENCE["text_tokens"].__getitem__, SPECIAL, **REFERENCE["input_config"])
        prepared = prepare_kimi_audio_inputs(
            [{"role": "user", "message_type": "text", "content": text}], builder, output_type=mode
        )
        # Supply the future stage's stop constraint, then exercise its
        # prompt_transform_func before the native admission EOS update.
        params = params.clone()
        params.stop_token_ids = [SPECIAL.msg_end]
        params.include_stop_str_in_output = True
        params.all_stop_token_ids.add(SPECIAL.msg_end)
        prepared = prepare_kimi_audio_request(prepared, [params])
        # The native Kimi tokenizer currently exposes 151644 as EOS. Resolve
        # it through native admission. Our stage terminates on msg_end.
        params.update_from_generation_config({}, eos_token_id=151644)
        req = SimpleNamespace(
            prompt_token_ids=prepared["prompt_token_ids"],
            output_token_ids=[],
            sampling_params=params,
            num_computed_tokens=0,
            num_tokens=len(prepared["prompt_token_ids"]),
            num_output_tokens=0,
            max_tokens=params.max_tokens,
            pooling_params=None,
        )
        self.runner.requests[rid] = req
        self.runner.model_intermediate_buffer[rid] = prepared["model_intermediate_buffer"]
        self.runner.model_intermediate_buffer[rid]["omni_final_stage_id"] = 0 if mode == "text" else 1
        self.runner._downstream_payload_cache.pop(rid, None)
        if params.temperature > 0 and params.seed is not None:
            self.generators[rid] = torch.Generator().manual_seed(params.seed)
        return req

    def step(self, order, spans, columns):
        runner = self.runner
        counts = np.array([count for _, count in spans])
        runner._omni_num_scheduled_tokens_np = counts
        runner.query_start_loc = SimpleNamespace(cpu=np.concatenate(([0], counts.cumsum())))
        params = [runner.requests[rid].sampling_params for rid in order]
        all_greedy = all(p.temperature == 0 for p in params)
        # Synthetic CPU batches use the native metadata type and its optional
        # tensor conventions; no native InputBatch/GPU scheduling is simulated.
        metadata = SamplingMetadata(
            temperature=None if all_greedy else torch.tensor([p.temperature for p in params]),
            all_greedy=all_greedy,
            all_random=all(p.temperature > 0 for p in params),
            top_p=None,
            top_k=(
                torch.tensor([p.top_k if p.top_k > 0 else self.stage.config.vocab_size for p in params])
                if any(p.top_k > 0 for p in params)
                else None
            ),
            generators={i: self.generators[rid] for i, rid in enumerate(order) if rid in self.generators},
            max_num_logprobs=None,
            no_penalties=all(p.repetition_penalty == 1 for p in params),
            prompt_token_ids=None,
            frequency_penalties=torch.zeros(len(order)),
            presence_penalties=torch.zeros(len(order)),
            repetition_penalties=torch.tensor([p.repetition_penalty for p in params]),
            output_token_ids=[],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=SimpleNamespace(),
        )
        runner.input_batch = SimpleNamespace(
            req_ids=order,
            sampling_metadata=metadata,
            update_async_output_token_ids=lambda: None,
        )
        embeddings, hidden = {}, []
        for rid, (offset, count), (text_column, audio_column) in zip(order, spans, columns, strict=True):
            req = runner.requests[rid]
            req.num_computed_tokens = offset
            info = runner.model_intermediate_buffer[rid]
            info.update(
                request_id=rid,
                _omni_is_prefill=offset < len(req.prompt_token_ids),
                _omni_num_computed_tokens=offset,
                _omni_prompt_len=len(req.prompt_token_ids),
                _omni_seed=req.sampling_params.seed,
                _omni_max_tokens=req.max_tokens,
            )
            scheduled = torch.tensor((req.prompt_token_ids + req.output_token_ids)[offset : offset + count])
            _, embeddings[rid], update = runner.model.preprocess(scheduled, None, **info)
            runner._update_intermediate_buffer(rid, update)
            rows = torch.zeros(count, 2 * self.stage.config.hidden_size)
            rows[-1, text_column] = rows[-1, self.stage.config.hidden_size + audio_column] = 1
            hidden.append(rows)
        kwargs = runner._build_model_kwargs_extra()
        assert "request_sampling_params" not in kwargs
        if runner.model_config.has_sampling_extra_args:
            assert kwargs["sampling_extra_args"] == [
                runner.requests[rid].sampling_params.extra_args or {} for rid in order
            ]
        output = runner.model.make_omni_output(torch.cat(hidden), **kwargs)
        packed, mm = runner.extract_multimodal_outputs(output)
        selected = packed[torch.tensor(runner.query_start_loc.cpu[1:] - 1)]
        logits = runner.model.compute_logits(selected, runner.input_batch.sampling_metadata)
        expected_text = torch.nn.functional.linear(selected[:, :4], self.stage.lm_head.weight)[
            :, : self.stage.config.vocab_size
        ]
        expected_audio = torch.nn.functional.linear(selected[:, 4:], self.stage.mimo_output.weight)[
            :, : self.stage.config.vocab_size
        ]
        torch.testing.assert_close(logits, expected_text)
        torch.testing.assert_close(self.stage._audio_logits, expected_audio)
        sampled = runner._sample(logits, None).sampled_token_ids
        assert sampled.shape == (len(order), 1)
        assert self.stage._sampling_context is None and self.stage._audio_logits is None
        for rid, token, eligible in zip(order, sampled[:, 0].tolist(), kwargs["request_sample_eligible"]):
            if eligible:
                req = runner.requests[rid]
                req.output_token_ids.append(token)
                req.num_output_tokens += 1
                req.num_tokens += 1
        self.outputs.append(
            runner._build_omni_model_runner_output_from_snapshot(
                scheduler_output=SimpleNamespace(total_num_scheduled_tokens=int(counts.sum())),
                hidden_states=packed,
                staged_hidden_states_cpu=None,
                multimodal_outputs=mm,
                req_ids_output_copy=list(order),
                req_id_to_index_output_copy={rid: i for i, rid in enumerate(order)},
                valid_sampled_token_ids=[
                    [token] if eligible else []
                    for token, eligible in zip(sampled[:, 0].tolist(), kwargs["request_sample_eligible"])
                ],
                logprobs_lists=None,
                prompt_logprobs_dict={},
                num_nans_in_logits=None,
                kv_connector_output=None,
                ec_connector_output=None,
                cudagraph_stats=None,
                kv_extracted_req_ids=None,
                num_scheduled_tokens_np=counts,
                query_start_loc_cpu=runner.query_start_loc.cpu,
            )
        )
        return sampled[:, 0].tolist(), embeddings

    def state(self, rid):
        return self.runner.model_intermediate_buffer[rid]["kimi_audio_generation"]


@pytest.fixture
def steps(monkeypatch, registered_model_runtime):
    stage = KimiAudioARStage.__new__(KimiAudioARStage)
    torch.nn.Module.__init__(stage)
    stage.config = SimpleNamespace(
        hidden_size=4,
        vocab_size=168448,
        kimia_mimo_audiodelaytokens=6,
        kimia_token_offset=152064,
        eos_token_ids=[151644, SPECIAL.msg_end],
    )
    stage.vllm_config = registered_model_runtime(
        model_config=SimpleNamespace(
            model_stage="kimi_audio_ar",
            hf_config=stage.config,
            max_model_len=8192,
            has_sampling_extra_args=True,
            engine_output_type="text",
            stage_connector_config={"extra": {"role": "sender"}},
        )
    )
    stage._sampling_context = stage._audio_logits = None
    stage.embed_tokens = torch.nn.Embedding(stage.config.vocab_size, 4)
    monkeypatch.setattr(logits_processor, "get_current_vllm_config", lambda: SimpleNamespace(model_config=None))
    stage.logits_processor = logits_processor.LogitsProcessor(stage.config.vocab_size)
    with torch.no_grad():
        for name, targets in (
            ("lm_head", (SPECIAL.kimia_text_eos, WORD, OTHER_WORD, 151644)),
            ("mimo_output", (CODE, SPECIAL.media_end, SPECIAL.msg_end, CODE + 6)),
        ):
            head = torch.nn.Linear(4, stage.config.vocab_size + 8, bias=False)
            head.weight.zero_()
            for column, token in enumerate(targets):
                head.weight[token, column] = 10
            # Native LogitsProcessor must remove vocabulary padding; otherwise
            # these artificially dominant padding rows would always win.
            head.weight[-8:].fill_(1000)
            head.tp_size = 1
            head.quant_method = SimpleNamespace(
                apply=lambda layer, hidden, bias=None: torch.nn.functional.linear(hidden, layer.weight, bias)
            )
            setattr(stage, name, head)

    # Substitute only stage construction; the unified entry exposes the real
    # preprocess/output/sample methods to the actual eager runner dispatch.
    def initialize_heads(self, *, vllm_config, prefix):
        self.__dict__.update(stage.__dict__)
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config

    monkeypatch.setattr(KimiAudioARStage, "__init__", initialize_heads)
    model = KimiAudioForConditionalGeneration(vllm_config=stage.vllm_config)
    return RequestSteps(model.eval())


@torch.inference_mode()
def test_profile_logits_do_not_leave_a_handoff_before_real_sampling(steps):
    model, stage = steps.runner.model, steps.stage
    for _ in range(2):
        hidden = torch.rand(2, 2 * stage.config.hidden_size)
        output = model.make_omni_output(hidden)
        assert not output.multimodal_outputs
        assert model.compute_logits(output.text_hidden_states).shape == (2, stage.config.vocab_size)
        assert stage._sampling_context is None and stage._audio_logits is None
    request = steps.add("after-profile", "text", SamplingParams(temperature=0, max_tokens=3))
    steps.step(["after-profile"], [(0, len(request.prompt_token_ids))], [(1, 0)])
    assert len(steps.runner.model_intermediate_buffer["after-profile"]["kimi_audio_generation"]["text_history"]) == 1


@torch.inference_mode()
def test_mixed_prefill_reorder_replay_and_native_stop(steps):
    a = steps.add("a", "both", SamplingParams(temperature=0, max_tokens=30, seed=7))
    b = steps.add("b", "text", SamplingParams(temperature=0, max_tokens=30), text="你好")
    pa, pb = len(a.prompt_token_ids), len(b.prompt_token_ids)
    steps.step(["a", "b"], [(0, 1), (0, pb)], [(0, 1), (1, 0)])
    assert steps.state("a")["text_history"] == []
    assert steps.state("a").get("rng_state") is None
    assert steps.state("b")["text_history"] == [WORD]
    first = steps.outputs[-1]
    assert first.multimodal_outputs[0]["ids.output"].tolist() == []
    assert first.multimodal_outputs[1]["ids.output"].tolist() == [WORD]
    assert not first.inter_stage_outputs[0]["meta.finished"].item()
    assert not check_stop(b, 8192)

    steps.step(["b", "a"], [(pb, 1), (1, pa - 1)], [(0, 0), (0, 1)])
    assert check_stop(b, 8192) and not check_stop(a, 8192)
    assert steps.state("a")["text_history"] == [SPECIAL.kimia_text_eos]
    assert a.output_token_ids == [SPECIAL.kimia_text_blank]
    assert steps.state("b")["text_history"] == [WORD, SPECIAL.kimia_text_eos]
    assert steps.outputs[-1].multimodal_outputs[0]["ids.output"].tolist() == []
    assert steps.outputs[-1].multimodal_outputs[0]["meta.finished"].item()

    _, embeds = steps.step(["a"], [(pa, 1)], [(1, 1)])
    expected = steps.stage.embed_tokens(torch.tensor([SPECIAL.kimia_text_eos])) + steps.stage.embed_tokens(
        torch.tensor([SPECIAL.kimia_text_blank])
    )
    torch.testing.assert_close(embeds["a"], expected)
    # Replay crosses the prompt/generation boundary but has not caught up
    # with the accepted history, so it must not sample a third output yet.
    previous = list(steps.state("a")["text_history"])
    _, replay = steps.step(["a"], [(pa - 1, 2)], [(1, 1)])
    torch.testing.assert_close(replay["a"][-1:], expected)
    assert steps.state("a")["text_history"] == previous
    assert steps.outputs[-1].inter_stage_outputs[0]["ids.output"].numel() == 0
    assert steps.outputs[-1].inter_stage_outputs[0]["codes.audio"].numel() == 0
    for step in range(2, 7):
        steps.step(["a"], [(pa + step - 1, 1)], [(1, 1)])
        assert check_stop(a, 8192) == (step == 6)
    assert steps.state("a")["audio_history"] == [SPECIAL.kimia_text_blank] * 6 + [SPECIAL.media_end]
    assert steps.state("a")["finished"]
    assert steps.outputs[-1].inter_stage_outputs[0]["codes.audio"].tolist() == [SPECIAL.media_end]
    assert steps.outputs[-1].multimodal_outputs[0]["meta.finished"].item()
    # A subsequent batch must not mutate earlier payloads through the handoff.
    assert first.multimodal_outputs[1]["ids.output"].tolist() == [WORD]
    assert not first.multimodal_outputs[1]["meta.finished"].item()
    # Reuse a finished request ID with a fresh runner buffer; the model holds
    # no old request history after the execute/sample handoff is consumed.
    fresh = steps.add("b", "text", SamplingParams(temperature=0, max_tokens=10), text="你好")
    steps.step(["b"], [(0, len(fresh.prompt_token_ids))], [(3, 0)])
    assert steps.state("b")["text_history"] == [fresh.sampling_params.eos_token_id]
    assert fresh.output_token_ids == [SPECIAL.kimia_text_blank]
    assert not check_stop(fresh, 8192)


@torch.inference_mode()
def test_request_overrides_and_seeded_audio_with_text_greedy(steps):
    with torch.no_grad():
        steps.stage.mimo_output.weight[CODE, 3] = 9.8
        steps.stage.lm_head.weight[OTHER_WORD, 1] = 9.7
    requests, oracles = {}, {}
    for rid, temperature, seed in (("audio_seed", 0.0, 103), ("native_seed", 0.8, 207), ("text_override", 0.0, 311)):
        overrides = {"audio_temperature": 0.7, "audio_top_k": 2, "audio_repetition_window_size": 2}
        if rid == "text_override":
            overrides.update(text_temperature=0.6, text_top_k=2)
        params = SamplingParams(
            temperature=temperature,
            top_k=2,
            repetition_penalty=1.1,
            max_tokens=20,
            seed=seed,
            extra_args={"kimi_audio": overrides},
        )
        requests[rid] = steps.add(rid, "both", params)
        oracles[rid] = ([], [], torch.Generator().manual_seed(seed))
    for index in range(8):
        order = list(requests) if index % 2 == 0 else list(reversed(requests))
        for rid in order:
            req = requests[rid]
            text, audio, generator = oracles[rid]
            result = sample_kimi_audio_step(
                steps.stage.lm_head.weight[:168448, 1],
                steps.stage.mimo_output.weight[:168448, 3],
                text_history=text,
                audio_history=audio,
                text_finished=False,
                output_type="both",
                special_tokens=SPECIAL,
                audio_delay=6,
                params=KimiAudioSamplingParams(
                    text_temperature=0.6 if rid == "text_override" else req.sampling_params.temperature,
                    text_top_k=2 if rid == "text_override" else req.sampling_params.top_k,
                    text_repetition_penalty=1.1,
                    audio_temperature=0.7,
                    audio_top_k=2,
                    audio_repetition_window_size=2,
                ),
                generator=generator,
            )
            text.append(result.text_token)
            audio.append(result.audio_token)
        spans = [
            (0, len(requests[rid].prompt_token_ids))
            if index == 0
            else (len(requests[rid].prompt_token_ids) + index - 1, 1)
            for rid in order
        ]
        # The override request runs alone to also cover all-greedy metadata:
        # native temperature/top-k can be absent while a custom stream samples.
        override_row = order.index("text_override")
        steps.step(["text_override"], [spans.pop(override_row)], [(1, 3)])
        order.pop(override_row)
        steps.step(order, spans, [(1, 3)] * 2)
        for rid in requests:
            text, audio, generator = oracles[rid]
            assert steps.state(rid)["text_history"] == text
            assert steps.state(rid)["audio_history"] == audio
            actual_rng = steps.generators[rid].get_state() if rid == "native_seed" else steps.state(rid)["rng_state"]
            assert torch.equal(actual_rng, generator.get_state())


@pytest.mark.parametrize("async_chunk", [False, True])
@torch.inference_mode()
def test_audio_end_delivers_simultaneous_text_through_runner_output(steps, async_chunk):
    steps.runner._async_chunk = async_chunk
    req = steps.add("both", "both", SamplingParams(temperature=0, max_tokens=20))
    prompt_len = len(req.prompt_token_ids)
    for index in range(8):
        span = (0, prompt_len) if index == 0 else (prompt_len + index - 1, 1)
        steps.step(["both"], [span], [(1, 0 if index < 7 else 1)])
    assert check_stop(req, 8192)
    assert req.output_token_ids == [SPECIAL.kimia_text_blank] * 7 + [SPECIAL.msg_end]
    assert steps.state("both")["text_history"] == [WORD] * 8
    assert steps.state("both")["audio_history"][-1] == SPECIAL.media_end
    client = [output.multimodal_outputs[0] for output in steps.outputs]
    inter = [output.inter_stage_outputs[0] for output in steps.outputs]
    assert all(set(payload) == {"ids.output", "codes.audio", "meta.finished"} for payload in client)
    assert torch.cat([payload["ids.output"] for payload in client]).tolist() == [WORD] * 8
    # Non-async stage bridging reads RequestOutput, not the connector channel.
    assert torch.cat([payload["codes.audio"] for payload in client]).tolist() == [CODE, SPECIAL.media_end]
    assert torch.cat([payload["codes.audio"] for payload in inter]).tolist() == [CODE, SPECIAL.media_end]
    assert [payload["meta.finished"].item() for payload in inter] == [False] * 7 + [True]
    assert client[-1]["ids.output"].tolist() == [WORD]


@pytest.mark.parametrize(
    "mode,budget,model_limit", [("text", 1, False), ("both", 6, False), ("both", 8, False), ("both", 8, True)]
)
@torch.inference_mode()
def test_length_limit_delivers_tail_without_fabricating_eos(steps, mode, budget, model_limit):
    req = steps.add("limited", mode, SamplingParams(temperature=0, max_tokens=20 if model_limit else budget))
    prompt_len = len(req.prompt_token_ids)
    if model_limit:
        steps.stage.vllm_config.model_config.max_model_len = prompt_len + budget
    for index in range(budget):
        span = (0, prompt_len) if index == 0 else (prompt_len + index - 1, 1)
        # In both mode the text ends first; audio keeps running until capped.
        steps.step(["limited"], [span], [(1 if index == 0 or mode == "text" else 0, 0)])
        assert check_stop(req, steps.stage.vllm_config.model_config.max_model_len) == (index == budget - 1)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert req.output_token_ids == [SPECIAL.kimia_text_blank] * budget
    assert steps.state("limited")["finished"]
    inter = [output.inter_stage_outputs[0] for output in steps.outputs]
    assert torch.cat([payload["ids.output"] for payload in inter]).tolist() == [WORD]
    assert torch.cat([payload["codes.audio"] for payload in inter]).tolist() == [CODE] * max(0, budget - 6)
    assert [payload["meta.finished"].item() for payload in inter] == [False] * (budget - 1) + [True]
    assert all(payload["codes.audio"].dtype == torch.long for payload in inter)


@pytest.mark.parametrize(
    "setting",
    [
        {"min_tokens": 2},
        {"top_p": 0.7},
        {"logprobs": 0},
        {"repetition_detection": RepetitionDetectionParams(min_pattern_size=1, max_pattern_size=1, min_count=3)},
    ],
)
def test_unsupported_request_options_fail_before_advancing_history(steps, setting):
    with pytest.raises(ValueError, match=next(iter(setting))):
        steps.add("bad", "text", SamplingParams(temperature=1, max_tokens=20, **setting))
    assert "bad" not in steps.runner.requests
    assert steps.stage._sampling_context is None


def test_admission_stop_contract_and_no_parameter_mutation():
    builder = KimiAudioPromptBuilder(REFERENCE["text_tokens"].__getitem__, SPECIAL, **REFERENCE["input_config"])
    prompt = prepare_kimi_audio_inputs(
        [{"role": "user", "message_type": "text", "content": "你好"}], builder, output_type="both"
    )
    params = SamplingParams(
        temperature=0, stop_token_ids=[SPECIAL.msg_end], include_stop_str_in_output=True, extra_args={"kimi_audio": {}}
    )
    before = params.clone()
    admitted = prepare_kimi_audio_request(prompt, [params])
    assert params == before
    assert "kimi_audio_request_validated" not in prompt["model_intermediate_buffer"]
    assert admitted["model_intermediate_buffer"]["kimi_audio_request_validated"] is True
    for stop_ids in ([], [WORD], [SPECIAL.msg_end, SPECIAL.kimia_text_blank]):
        with pytest.raises(ValueError, match="stop_token_ids"):
            prepare_kimi_audio_request(
                prompt, [SamplingParams(stop_token_ids=stop_ids, include_stop_str_in_output=True)]
            )


def test_missing_extra_args_channel_fails_explicitly(steps):
    req = steps.add("unconfigured", "text", SamplingParams(temperature=0))
    steps.runner.model_config.has_sampling_extra_args = False
    with pytest.raises(ValueError, match="sampling_extra_args"):
        steps.step(["unconfigured"], [(0, len(req.prompt_token_ids))], [(1, 0)])
    assert not steps.state("unconfigured")["text_history"]


@torch.inference_mode()
def test_pp_feedback_preserves_request_identity_and_next_input(steps, cpu_pp_group):
    a = steps.add("a", "both", SamplingParams(temperature=0, max_tokens=2, seed=7))
    b = steps.add("b", "text", SamplingParams(temperature=0, max_tokens=8), text="你好")
    pa, pb = len(a.prompt_token_ids), len(b.prompt_token_ids)
    # Both ranks initialize metadata even for a partial prefill.
    steps.step(["a", "b"], [(0, 1), (0, 1)], [(1, 0), (2, 0)])
    first_buffer = copy.deepcopy(steps.runner.model_intermediate_buffer)
    sent = None

    def broadcast(update, src):
        nonlocal sent
        assert src == 1
        if update is not None:
            sent = copy.deepcopy(update)
        return copy.deepcopy(sent)

    cpu_pp_group.broadcast_object = broadcast
    for spans in ([(1, pa - 1), (1, 1)], [(pa, 1), (2, pb - 2)]):
        # Existing fixture exercises real Kimi sampling with synthetic heads.
        cpu_pp_group.world_size = 1
        cpu_pp_group.is_first_rank = cpu_pp_group.is_last_rank = True
        steps.step(["a", "b"], spans, [(1, 0), (2, 0)])
        cpu_pp_group.world_size = 2
        cpu_pp_group.is_first_rank, cpu_pp_group.is_last_rank = False, True
        steps.runner.model.sync_pipeline_state(
            req_ids=["a", "b"], model_intermediate_buffer=steps.runner.model_intermediate_buffer
        )
        cpu_pp_group.is_first_rank, cpu_pp_group.is_last_rank = True, False
        steps.runner.model.sync_pipeline_state(req_ids=["b", "a"], model_intermediate_buffer=first_buffer)
        # Replaying a feedback payload must not duplicate accepted tokens.
        steps.runner.model.sync_pipeline_state(req_ids=["a", "b"], model_intermediate_buffer=first_buffer)
        for rid in ("a", "b"):
            first = first_buffer[rid]["kimi_audio_generation"]
            last = steps.state(rid)
            for key in ("text_history", "audio_history", "scheduler_history", "text_finished", "finished"):
                assert first[key] == last[key]
        if not first_buffer["a"]["kimi_audio_generation"]["finished"]:
            state = steps.state("a")
            info = dict(first_buffer["a"], _omni_is_prefill=False, _omni_num_computed_tokens=pa)
            _, embedding, _ = steps.stage.preprocess(torch.tensor([state["scheduler_history"][-1]]), None, **info)
            expected = steps.stage.embed_tokens(torch.tensor([state["text_history"][-1]]))
            expected += steps.stage.embed_tokens(torch.tensor([state["audio_history"][-1]]))
            torch.testing.assert_close(embedding, expected)
    assert first_buffer["a"]["kimi_audio_generation"]["finished"]
    assert len(first_buffer["b"]["kimi_audio_generation"]["scheduler_history"]) == 1
    sent["a"] = (100, *sent["a"][1:])
    with pytest.raises(ValueError, match="out of sync"):
        steps.runner.model.sync_pipeline_state(req_ids=["a", "b"], model_intermediate_buffer=first_buffer)


def test_pp_runner_syncs_after_sampling_and_skips_idle_calls(steps, cpu_pp_group):
    runner = steps.runner
    events = []
    runner.model.sync_pipeline_state = lambda **kwargs: events.append(("sync", kwargs["req_ids"]))
    runner.input_batch = SimpleNamespace(req_ids=["request"])
    runner.use_async_scheduling = False
    runner.attach_omni_connector_output = lambda output: output
    runner.kv_connector_output = None
    runner.execute_model_state = None
    cpu_pp_group.is_first_rank, cpu_pp_group.is_last_rank = True, False
    # No forward (e.g. a cleanup-only scheduler step) must not join a collective.
    runner.sample_tokens(None)
    assert events == []
    runner._pp_model_state_pending = True
    runner.sample_tokens(None)
    assert events == [("sync", ["request"])]
    runner.sample_tokens(None)
    assert len(events) == 1

    cpu_pp_group.is_first_rank, cpu_pp_group.is_last_rank = False, True
    runner.execute_model_state = (SimpleNamespace(),) + (None,) * 11

    def sample(*args):
        events.append(("sample", None))
        return SimpleNamespace(sampled_token_ids=torch.tensor([[1]]))

    class AfterSyncError(Exception):
        pass

    def stop_after_sync(*args):
        raise AfterSyncError

    runner._sample = sample
    runner._update_states_after_model_execute = stop_after_sync
    # Stop at the boundary under test; output building has separate coverage.
    with pytest.raises(AfterSyncError):
        runner.sample_tokens(None)
    assert events[-2:] == [("sample", None), ("sync", ["request"])]
