# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage 0 on Model Runner V2 and both samplers, on the GPU.

Each checkpoint's stage 0 runs in a real ``OmniARModelRunner``, set up as a
worker sets it up: vLLM's dummy loader fills the weights, the KV cache
layout is resolved as the engine core resolves it, and requests arrive as
the scheduler hands them over. Only the scheduler is played by the test.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace

import numpy as np
import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import get_current_attn_backends
from vllm.sampling_params import SamplingParams
from vllm.v1.attention.backends.utils import (
    get_supported_kv_cache_layouts,
    record_kv_cache_layout,
    resolve_kv_cache_layout,
)
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.core.sched.output import CachedRequestData
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import sort_batch_req_ids
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_input_batch import InputBatch as V1InputBatch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.chatterbox.helpers import (
    DEPLOY,
    SEED,
    build_stage,
    model_parallel,
    random_voice,
    stage_vllm_config,
)
from vllm_omni.core.sched.output import OmniNewRequestData, OmniSchedulerOutput
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.chatterbox.cfg_sampler import ChatterboxMRv2CFGSampler
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import S3GenDecoder
from vllm_omni.model_executor.models.chatterbox.chatterbox_t3 import prefill_embeds
from vllm_omni.model_executor.models.chatterbox.conditioning import build_prompt
from vllm_omni.model_executor.stage_input_processors.chatterbox import (
    expand_original_cfg_prompts,
    prepare_original_prompt,
)
from vllm_omni.request import OmniRequest
from vllm_omni.transformers_utils.configs.chatterbox import START_SPEECH_TOKEN, STOP_SPEECH_TOKEN, ChatterboxConfig
from vllm_omni.worker.sampling_utils import call_model_sampler
from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

pytestmark = [pytest.mark.core_model]

# Room for every request a test schedules; the deploy files' budget is for a whole GPU.
KV_CACHE_BYTES = 1 << 30
BLOCKS_PER_REQUEST = 32
Runner = tuple[OmniARModelRunner, VllmConfig]


@pytest.fixture(scope="module")
def runners(model_dir: str) -> Iterator[Callable[[str], Runner]]:
    """Model Runner V2 for a checkpoint's stage 0, set up once per module as a worker sets it up."""
    built: dict[str, Runner] = {}

    def runner(variant: str) -> Runner:
        if variant not in built:
            config = stage_vllm_config(model_dir, DEPLOY[variant], 0, load_format="dummy")
            with set_current_vllm_config(config):
                model_runner = OmniARModelRunner(config, torch.device("cuda:0"))
                model_runner.load_model()
                specs = model_runner.get_kv_cache_spec()
                layouts = [layout.name for layout in get_supported_kv_cache_layouts(get_current_attn_backends(config))]
                record_kv_cache_layout(
                    config.cache_config, resolve_kv_cache_layout(config, [layouts], specs.values()).name
                )
                # The engine core registers vLLM's own KV cache specs before sizing the
                # cache. The lazy registration is skipped once any spec is registered,
                # as a model module's own spec is when another test imports it.
                register_all_kvcache_specs(config)
                (kv_cache_config,) = get_kv_cache_configs(config, [specs], [KV_CACHE_BYTES])
                model_runner.initialize_kv_cache(kv_cache_config)
            built[variant] = model_runner, config
        return built[variant]

    with model_parallel():
        yield runner
        for model_runner, _ in built.values():
            model_runner.shutdown()


def new_request(
    req_id: str,
    prompt: dict,
    params: SamplingParams,
    *,
    slot: int = 0,
    external_id: str | None = None,
    generated: tuple[int, ...] = (),
    computed: int = 0,
) -> OmniNewRequestData:
    """A request as the scheduler hands it to the runner: new, or resumed after preemption.

    ``generated`` are the tokens a resumed request had sampled, recomputed
    with its prompt; ``computed`` how much of that the cache already holds.
    ``slot`` picks the request's KV cache blocks, distinct from those of the
    other requests of its test; every test finishes its requests.
    """
    request = OmniRequest(
        request_id=req_id,
        prompt_token_ids=prompt["prompt_token_ids"],
        sampling_params=params,
        pooling_params=None,
        external_req_id=external_id or req_id,
        additional_information=serialize_additional_information(prompt["additional_information"]),
    )
    request.num_computed_tokens = computed
    return OmniNewRequestData.from_request(
        request,
        (list(range(slot * BLOCKS_PER_REQUEST, (slot + 1) * BLOCKS_PER_REQUEST)),),
        prefill_token_ids=[*prompt["prompt_token_ids"], *generated],
    )


def step(
    new: Sequence[OmniNewRequestData] = (), decode: dict[str, int] | None = None, finished: set[str] | None = None
) -> OmniSchedulerOutput:
    """One scheduler step: new requests in full, and one token for each ``decode`` request at its progress."""
    decode = decode or {}
    num_scheduled = {data.req_id: len(data.prefill_token_ids) - data.num_computed_tokens for data in new}
    num_scheduled |= dict.fromkeys(decode, 1)
    return OmniSchedulerOutput(
        scheduled_new_reqs=list(new),
        scheduled_cached_reqs=CachedRequestData(
            req_ids=list(decode),
            resumed_req_ids=set(),
            new_token_ids=[[] for _ in decode],
            all_token_ids={},
            new_block_ids=[None for _ in decode],
            num_computed_tokens=list(decode.values()),
            num_output_tokens=[1 for _ in decode],
        ),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=sum(num_scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=finished or set(),
        free_encoder_mm_hashes=[],
    )


def run_step(runner: OmniARModelRunner, scheduler_output: OmniSchedulerOutput) -> dict[str, int]:
    """Execute and sample one step; the token each request sampled."""
    runner.execute_model(scheduler_output)
    output = runner.sample_tokens(None).get_output()
    return {req_id: tokens[0] for req_id, tokens in zip(output.req_ids, output.sampled_token_ids, strict=True)}


def greedy() -> SamplingParams:
    return SamplingParams(temperature=0.0, max_tokens=64)


def original_prompts(
    external_id: str, weight: float, params: SamplingParams | None = None
) -> list[tuple[dict, SamplingParams, str]]:
    """An Original request admitted as the engine admits it: with its companion when guided."""
    prompt = build_prompt(
        [12, 13, 14], random_voice(8, 250), ChatterboxConfig("original"), exaggeration=0.5, cfg_weight=weight
    )
    prompt["additional_information"]["global_request_id"] = [external_id]
    params = params or greedy()
    prepare_original_prompt(prompt, [params])
    admitted = [(prompt, params, external_id)]
    for expanded in expand_original_cfg_prompts(prompt, params):
        companion_params, _ = expanded.apply_overrides(params, [params])
        companion_id = external_id + expanded.request_id_suffix
        expanded.prompt["additional_information"]["global_request_id"] = [companion_id]
        admitted.append((expanded.prompt, companion_params, companion_id))
    return admitted


def stage0_request(variant: str, external_id: str) -> tuple[dict, SamplingParams, str]:
    """One unguided request of either checkpoint, admitted as the engine admits it."""
    if variant == "original":
        return original_prompts(external_id, 0.0)[0]
    return build_prompt([5, 6, 7], random_voice(150, 250), ChatterboxConfig()), greedy(), external_id


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_forward_returns_the_backbones_tensor_unwrapped(runners: Callable[[str], Runner], variant: str) -> None:
    """Model Runner V2 sizes its CUDA graph's output from what ``forward`` returns.

    The registered class's output is a bare hidden-state tensor, nothing
    wrapped around it, and the step keeps it and samples from it.
    """
    runner, config = runners(variant)
    outputs: list[object] = []

    def forward_output(module: torch.nn.Module, args: tuple, output: object) -> None:
        outputs.append(output)

    hook = runner.model.register_forward_hook(forward_output)
    req_id = f"forward-{variant}"
    prompt, params, external_id = stage0_request(variant, req_id)
    try:
        with set_current_vllm_config(config):
            runner.execute_model(step([new_request(req_id, prompt, params, external_id=external_id)]))
            kept = runner.execute_model_state.hidden_states
            sampled = runner.sample_tokens(None).get_output().sampled_token_ids
    finally:
        hook.remove()
        runner.finish_requests(step(finished={req_id}))

    (output,) = outputs
    assert type(output) is torch.Tensor
    assert output.shape[0] >= len(prompt["prompt_token_ids"])
    assert output.shape[1] == runner.model.model.config.hidden_size
    # The runner keeps forward's tensor as the step's hidden states and samples from it.
    assert kept.data_ptr() == output.data_ptr()
    assert torch.equal(kept, output[: kept.shape[0]])
    assert 0 <= sampled[0][0] <= STOP_SPEECH_TOKEN


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_model_runner_v2_embeds_a_mixed_step_through_the_hooks(runners: Callable[[str], Runner]) -> None:
    """V2 orders decode rows first, embeds every row itself, then calls the hooks.

    The second step holds two decode rows; a one-token row of a request
    recomputed after preemption, which is past its prompt and so goes to the
    decode hook too; a recomputed request whose prefill runs two generated
    tokens past its prompt; and a whole prompt. What reaches ``forward`` is
    compared with the embeddings each row should have, and no hook wrote
    anything into a request's payload.
    """
    runner, config = runners("turbo")
    heads, t3 = runner.model.model.heads, runner.model.model.config
    torch.manual_seed(6)
    prompts = {
        name: build_prompt([5, 6, 7], random_voice(cond, 250), t3)
        for name, cond in zip("abcde", (150, 140, 130, 120, 110))
    }
    lengths = {name: len(prompt["prompt_token_ids"]) for name, prompt in prompts.items()}
    embeds: dict[str, torch.Tensor] = {}

    def forward_inputs(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
        embeds["step"] = kwargs["inputs_embeds"].clone()

    def prompt_embeds(name: str) -> torch.Tensor:
        info = prompts[name]["additional_information"]
        return prefill_embeds(
            heads,
            torch.tensor(info["ids"]["prompt"], device="cuda"),
            torch.tensor(info["ids"]["speech_token"], device="cuda"),
            info["embed"]["voice"].to(device="cuda", dtype=heads.text_emb.weight.dtype),
            t3.start_speech_token,
        )

    req = {name: f"mixed-{name}" for name in prompts}
    with set_current_vllm_config(config):
        first = run_step(
            runner, step([new_request(req[name], prompts[name], greedy(), slot=slot) for slot, name in enumerate("ab")])
        )
        hook = runner.model.register_forward_pre_hook(forward_inputs, with_kwargs=True)
        try:
            second = step(
                [
                    new_request(req["c"], prompts["c"], greedy(), slot=2, generated=(12,), computed=lengths["c"]),
                    new_request(req["d"], prompts["d"], greedy(), slot=3, generated=(13, 14)),
                    new_request(req["e"], prompts["e"], greedy(), slot=4),
                ],
                decode={req["a"]: lengths["a"], req["b"]: lengths["b"]},
            )
            run_step(runner, second)
        finally:
            hook.remove()
            buffers = {
                req_id: set(runner.model_state.intermediate_buffer.buffers[index])
                for req_id, index in runner.req_states.req_id_to_index.items()
                if req_id in req.values()
            }
            runner.finish_requests(step(finished=set(req.values())))

    speech = heads.speech_emb
    expected = {
        req["a"]: speech(torch.tensor([first[req["a"]]], device="cuda")),
        req["b"]: speech(torch.tensor([first[req["b"]]], device="cuda")),
        req["c"]: speech(torch.tensor([12], device="cuda")),
        req["d"]: torch.cat([prompt_embeds("d"), speech(torch.tensor([13, 14], device="cuda"))]),
        req["e"]: prompt_embeds("e"),
    }
    order = sort_batch_req_ids(second.num_scheduled_tokens, {}, runner.decode_query_len)
    assert set(order[:3]) == {req["a"], req["b"], req["c"]}
    assert torch.equal(embeds["step"][: second.total_num_scheduled_tokens], torch.cat([expected[r] for r in order]))
    # The request's payload beside the runner's own two keys, as the requests arrived.
    assert buffers == dict.fromkeys(req.values(), {"req_id", "sampling_params", "ids", "embed"})


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_mrv2_sampling_transports_tokens_without_unused_hidden_states(
    runners: Callable[[str], Runner], variant: str
) -> None:
    """The stage's sampler returns the runner's own sampling, flagged token-only.

    Stage 1 reads the sampled ids, never the hidden states, and the runner
    keeps forward's per-token streaming payload.
    """
    runner, config = runners(variant)
    req_id = f"transport-{variant}"
    prompt, params, external_id = stage0_request(variant, req_id)
    with set_current_vllm_config(config):
        runner.execute_model(step([new_request(req_id, prompt, params, external_id=external_id)]))
        state = runner.execute_model_state
        try:
            standard = runner.sample(state.hidden_states, state.input_batch, None)
            result = runner.sampler.sample_step(
                state.hidden_states, state.input_batch, runner.req_states, None, runner.sample
            )
        finally:
            runner.sample_tokens(None)
            runner.finish_requests(step(finished={req_id}))

    assert torch.equal(result.sampler_output.sampled_token_ids, standard[0].sampled_token_ids)
    assert torch.equal(result.num_sampled, standard[1]) and torch.equal(result.num_rejected, standard[2])
    assert result.multimodal_outputs is None  # Preserve forward's per-token streaming metadata.
    assert result.include_hidden_states is False
    assert result.owns_multimodal_outputs is True


def guided_step(runner: OmniARModelRunner, config: VllmConfig, name: str, seed: int | None) -> tuple:
    """A guided Original pair prefilled in one step; its batch, before sampling."""
    params = greedy() if seed is None else SamplingParams(temperature=1.0, seed=seed, max_tokens=64)
    admitted = original_prompts(name, 0.5, params)
    if seed is not None:
        # The companion inherits its parent's seed; its own makes the delegate's
        # draws differ between the rows, so whose token they end with shows.
        admitted[1][1].seed = seed + 1
    new = [
        new_request(name + suffix, prompt, request_params, slot=slot, external_id=external_id)
        for slot, ((prompt, request_params, external_id), suffix) in enumerate(zip(admitted, ("", "-u"), strict=True))
    ]
    with set_current_vllm_config(config):
        runner.execute_model(step(new))
    return runner.execute_model_state.input_batch, [data.req_id for data in new]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_mrv2_guidance_precedes_delegate_and_tokens_follow_conditional(runners: Callable[[str], Runner]) -> None:
    """Both rows sample the guided logits, and the companion takes the conditional's token.

    The pair is registered by the stage's own preprocessing of the step.
    Greedy: either row alone would pick token 1, the guided logits pick 0.
    Seeded: the delegate draws differently for the two rows, and the
    companion still ends with the conditional row's token.
    """
    runner, config = runners("original")
    for name, seed in (("guided-greedy", None), ("guided-seeded", 9)):
        batch, (cond_id, uncond_id) = guided_step(runner, config, name, seed)
        rows = {req_id: row for row, req_id in enumerate(batch.req_ids)}
        logits = torch.zeros(2, runner.model.model.config.vocab_size, device="cuda")
        if seed is None:
            logits[rows[uncond_id], :2] = torch.tensor([-2.0, 1.5])
            logits[rows[cond_id], :2] = torch.tensor([1.0, 1.2])
        try:
            with set_current_vllm_config(config):
                guided, _ = runner.sampler.cfg_sampler.prepare_logits(logits, batch.req_ids)
                delegate = runner.sampler.base_sampler(guided.clone(), batch).sampled_token_ids.clone()
                tokens = runner.sampler(logits.clone(), batch).sampled_token_ids
        finally:
            runner.sample_tokens(None)
            runner.finish_requests(step(finished={cond_id, uncond_id}))
        if seed is None:
            assert logits.argmax(dim=1).tolist() == [1, 1]
            assert tokens.tolist() == [[0], [0]]
        else:
            assert delegate[rows[uncond_id]] != delegate[rows[cond_id]]
            assert tokens[rows[uncond_id]] == tokens[rows[cond_id]] == delegate[rows[cond_id]]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_mrv2_warmup_rows_sample_unpaired_and_masked(runners: Callable[[str], Runner]) -> None:
    """The runner's own warmup batches carry random ids and no prompt; kernel warmup uses ``_warmup_`` ids.

    Neither needs a guidance registration, both still get the speech-token
    mask (token 5 or 7 wins only if the start token and the ids past the
    stop token are masked), and each row keeps its own token. A batch that
    reports a real prompt for an unregistered id is refused.
    """
    runner, config = runners("original")
    sampler = ChatterboxMRv2CFGSampler(runner.sampler.base_sampler, {})
    vocab = runner.model.model.config.vocab_size
    logits = torch.ones(2, vocab, device="cuda")
    logits[:, START_SPEECH_TOKEN] = 100.0
    logits[:, STOP_SPEECH_TOKEN + 1 :] = 100.0
    logits[0, 5], logits[1, 7] = 50.0, 50.0
    with set_current_vllm_config(config):
        warmup = InputBatch.make_dummy(2, 2, runner.input_buffers)
        assert not warmup.prefill_len_np.any()
        assert sampler(logits.clone(), warmup).sampled_token_ids.tolist() == [[5], [7]]

        kernel = replace(
            InputBatch.make_dummy(1, 1, runner.input_buffers), req_ids=["_warmup_0_"], prefill_len_np=np.array([1])
        )
        assert sampler(logits[:1].clone(), kernel).sampled_token_ids.tolist() == [[5]]

        real = replace(InputBatch.make_dummy(1, 1, runner.input_buffers), prefill_len_np=np.array([42]))
        with pytest.raises(ValueError, match="Missing Chatterbox CFG registration"):
            sampler(logits[:1].clone(), real)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_v1_runner_hands_each_request_its_guidance_pair(runners: Callable[[str], Runner]) -> None:
    """The V1 runner calls the stage's sampler with each row's ``extra_args``, from its own input batch.

    The rows are the pair the engine admits, in the batch order the V1 input
    batch keeps; greedy picks the guided logits' best token for both.
    """
    runner, config = runners("original")
    stage = runner.model
    admitted = original_prompts("v1-pair", 0.5)
    vocab = stage.model.config.vocab_size
    batch = V1InputBatch(
        max_num_reqs=4,
        max_model_len=config.model_config.max_model_len,
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        device=torch.device("cuda"),
        vocab_size=vocab,
        block_sizes=[config.cache_config.block_size],
        max_num_blocks_per_req=[BLOCKS_PER_REQUEST],
    )
    requests = {}
    for prompt, params, external_id in reversed(admitted):
        requests[external_id] = CachedRequestState(
            req_id=external_id,
            prompt_token_ids=prompt["prompt_token_ids"],
            mm_features=[],
            sampling_params=params,
            generator=None,
            block_ids=([0],),
            num_computed_tokens=len(prompt["prompt_token_ids"]),
            output_token_ids=[],
        )
        batch.add_request(requests[external_id])
    batch.refresh_metadata()
    rows = {req_id: row for row, req_id in enumerate(batch.req_ids)}
    logits = torch.zeros(2, vocab, device="cuda")
    logits[rows["v1-pair__cfg_uncond"], :2] = torch.tensor([-2.0, 1.5])
    logits[rows["v1-pair"], :2] = torch.tensor([1.0, 1.2])

    with set_current_vllm_config(config):
        output = call_model_sampler(
            stage, stage.sample, logits, batch.sampling_metadata, input_batch=batch, requests=requests
        )

    assert batch.req_ids == ["v1-pair__cfg_uncond", "v1-pair"]
    assert output.sampled_token_ids.tolist() == [[0], [0]]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_stage_1_noise_is_the_same_draw_when_built_on_the_gpu(model_dir: str) -> None:
    """vLLM's loader builds stage 1 inside a GPU default-device context; its flow noise is the CPU's draw."""
    config = stage_vllm_config(model_dir, DEPLOY["turbo"], 1)
    on_cpu = S3GenDecoder(config.model_config.hf_config, config.model_config.max_model_len, SEED)

    stage = build_stage(config, "cuda").model

    assert stage.flow_noise.device.type == stage.trim_fade.device.type == "cuda"
    assert torch.equal(stage.flow_noise.cpu(), on_cpu.flow_noise)
