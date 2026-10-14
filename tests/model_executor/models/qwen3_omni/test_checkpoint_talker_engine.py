# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in full Talker GPU engine checks with controlled source conditioning.

Set VLLM_OMNI_TEST_TALKER_ENGINE=1 and VLLM_OMNI_TEST_QWEN3_MODEL to a
verified local checkpoint, and isolate an idle GPU with CUDA_VISIBLE_DEVICES.
These tests use the complete Talker transformer, native paged KV, runner and
scheduler, but synthetic Thinker outputs. They are not pipeline/WER/perf tests.
Exact cold/warm/resumed comparisons require VLLM_BATCH_INVARIANT=1 on the
qualified BF16 GPU backend. Codec-only recovery does not require Code2Wav;
audio comparisons also require its independent reduction-dtype fix.
Ordinary batch-shape numerical drift remains a
separate accuracy gate. Predictor compilation is disabled by default; set
VLLM_OMNI_TEST_TALKER_COMPILE=1 to exercise its compiled path. Set
VLLM_OMNI_TEST_TALKER_GRAPH=1 to also exercise the runner's decode/MTP graphs,
or =full to cover prefill graphs with Triton attention (FA2 only graphs decode).
"""

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from vllm.config import CUDAGraphMode
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.forward_context import get_forward_context
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.request import RequestStatus

from tests.model_executor.models.qwen3_omni.qualification_worker import (
    qualification_prefix_config as _qualification_prefix_config,
)
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.arg_utils import OmniEngineArgs
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import talker2code2wav_full_payload
from vllm_omni.platforms import current_omni_platform
from vllm_omni.request import OmniRequest
from vllm_omni.worker import gpu_model_runner

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.omni]


@pytest.fixture(
    scope="module",
    params=[(False, False), (True, False)],
    ids=["cache-off", "guarded-cache-sync"],
)
def talker_engine(request, tmp_path_factory):
    if os.environ.get("VLLM_OMNI_TEST_TALKER_ENGINE") != "1":
        pytest.skip("set VLLM_OMNI_TEST_TALKER_ENGINE=1 for full Talker GPU execution")
    checkpoint = Path(os.environ["VLLM_OMNI_TEST_QWEN3_MODEL"])
    assert checkpoint.is_dir() and torch.cuda.is_available()
    cache_enabled, async_scheduling = request.param
    graph_mode = os.environ.get("VLLM_OMNI_TEST_TALKER_GRAPH")
    graph_enabled = graph_mode in ("1", "full")
    full_graph = graph_mode == "full"
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    talker_keys = {key for key in weight_map if key.startswith("talker.")}
    assert talker_keys
    # A stage-only weight view avoids requiring unrelated Thinker shards.
    # No source checkpoint/index edits or duplicate weight storage. Native
    # loading still validates that every constructed model parameter is loaded.
    stage_weights = tmp_path_factory.mktemp("talker-weights")
    present_keys = set()
    for filename in sorted({weight_map[key] for key in talker_keys}):
        shard = checkpoint / filename
        assert shard.is_file(), f"missing full Talker shard: {shard}"
        with safe_open(shard, framework="pt", device="cpu") as handle:
            present_keys.update(handle.keys())
        (stage_weights / filename).symlink_to(shard)
    assert talker_keys <= present_keys
    for metadata in checkpoint.iterdir():
        if metadata.suffix in (".json", ".txt", ".jinja", ".model") and metadata.name != "model.safetensors.index.json":
            (stage_weights / metadata.name).symlink_to(metadata)
    args = OmniEngineArgs(
        model=str(stage_weights),
        tokenizer=str(checkpoint),
        model_stage="talker",
        model_arch="Qwen3OmniMoeForConditionalGeneration",
        hf_config_name="talker_config",
        worker_type="ar",
        engine_output_type="audio",
        stage_id=1,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=2,
        enforce_eager=not graph_enabled,
        attention_config={"backend": "TRITON_ATTN"} if full_graph else {},
        compilation_config={
            "mode": 0,
            "cudagraph_mode": "FULL" if full_graph else "FULL_DECODE_ONLY" if graph_enabled else "NONE",
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128] if full_graph else [1, 2],
        },
        enable_prefix_caching=cache_enabled,
        async_scheduling=async_scheduling,
        gpu_memory_utilization=0.35,
        kv_cache_memory_bytes=256 * 1024 * 1024,
        scheduler_cls=(
            "vllm_omni.core.sched.omni_ar_scheduler.OmniARAsyncScheduler"
            if async_scheduling
            else "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler"
        ),
        load_format="safetensors",
    )
    # Predictor compilation is a separate qualification axis: its wrapper
    # currently does not consume the outer engine's enforce_eager setting.
    with pytest.MonkeyPatch.context() as patch:
        if os.environ.get("VLLM_OMNI_TEST_TALKER_COMPILE") != "1":
            patch.setattr(current_omni_platform, "supports_torch_inductor", lambda: False)
        if cache_enabled:
            patch.setattr(gpu_model_runner, "stage_prefix_cache_config", _qualification_prefix_config)
        engine = None
        try:
            engine = EngineCore(args.create_engine_config(), UniProcExecutor, log_stats=False)
            # Same stage-local hook as StageEngineCoreProc.__init__; EngineCore
            # is used here only to avoid unrelated ZMQ/coordinator processes.
            StageEngineCoreProc._init_talker_request_hasher(engine)
            runner = engine.model_executor.driver_worker.worker.model_runner
            # Initialize real accumulator/cleanup state without transport I/O.
            # Individual tests opt into accumulation and capture the send edge.
            with pytest.MonkeyPatch.context() as transport:
                transport.setattr(runner, "_create_connector", lambda config: None)
                runner.init_omni_connectors(runner.model_config, runner.kv_transfer_manager)
            assert runner._omni_connector is None
            assert not runner._omni_cache_policy.needs_full_hidden_states
            if graph_enabled:
                assert isinstance(runner.talker_mtp, current_omni_platform.get_graph_wrapper_cls())
            yield engine
        finally:
            if engine is not None:
                engine.shutdown()
            cleanup_dist_env_and_memory()


def _controlled_payload(model):
    config = model.config
    # Forty source text rows; no actual Thinker inference is implied.
    user = [config.im_start_token_id, config.user_token_id, 198] + list(range(100, 140))
    user += [config.im_end_token_id, 198]
    header = [config.im_start_token_id, config.assistant_token_id, 198]
    prompt = user + header
    sequence = prompt + list(range(200, 208))
    prompt_ids = user + header + [config.tts_pad_token_id] * 4 + [config.tts_bos_token_id, 200]
    talker_vocab_size = model.talker_config.text_config.vocab_size
    assert any(0 <= token < talker_vocab_size for token in prompt_ids)
    assert any(token >= talker_vocab_size for token in prompt_ids)
    width = config.thinker_config.text_config.hidden_size
    generator = torch.Generator().manual_seed(456)

    def values(rows):
        return torch.randn(rows, width, generator=generator, dtype=torch.bfloat16) * 0.01

    return {
        "ids": {"prompt": prompt, "all": sequence},
        "embed": {
            "prefill": values(len(sequence)),
            # Full Thinker output accumulates one special-token snapshot per
            # step. Exercise that shape, including when resuming past text EOS.
            "tts_pad": values(1).unsqueeze(0).repeat(17, 1, 1),
            "tts_bos": values(1).unsqueeze(0).repeat(17, 1, 1),
            "tts_eos": values(1).unsqueeze(0).repeat(17, 1, 1),
        },
        "hidden_states": {"output": values(len(sequence))},
        "meta": {
            "talker_prefill_plan": [("user", 0, len(user)), ("assistant", len(user), len(sequence))],
            "next_stage_prompt_ids": prompt_ids,
            "next_stage_prompt_len": len(prompt_ids),
        },
    }


@pytest.fixture(scope="module")
def checkpoint_code2wav(talker_engine):
    config = talker_engine.model_executor.driver_worker.worker.model_runner.model.code2wav_config
    decoder = Qwen3OmniMoeCode2Wav(vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=config)))
    decoder = decoder.to(device="cuda", dtype=torch.bfloat16).eval()
    snapshot = Path(os.environ["VLLM_OMNI_TEST_QWEN3_MODEL"])
    weights = {}
    with safe_open(snapshot / "model-00015-of-00015.safetensors", framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key.startswith("code2wav."):
                weights[key] = handle.get_tensor(key)
    assert decoder.load_weights(weights.items()) == set(dict(decoder.named_parameters()))
    decoder.precompute_snake_caches()
    return decoder


@torch.inference_mode()
def _decode_emitted_codes(decoder, request, outputs):
    payload = talker2code2wav_full_payload(None, {"codes.audio": _emitted_codes(outputs)}, request)
    assert payload is not None
    codes = torch.tensor(payload["codes"]["audio"], device="cuda", dtype=torch.long).reshape(1, 16, -1)
    # Preserve the existing full-payload packer's length-cap behavior: it
    # tail-aligns against all sampled IDs, retaining one bootstrap zero row
    # plus the 17 executed generated inputs. Do not silently trim the oracle.
    assert codes.shape[2] == request.num_output_tokens
    assert not torch.count_nonzero(codes[:, :, 0])
    torch.testing.assert_close(codes[0, :, 1:].cpu().T, _emitted_codes(outputs)[-17:].cpu(), rtol=0, atol=0)
    waveform = decoder.chunked_decode(codes, chunk_size=300, left_context_size=25)[0].detach().cpu()
    assert waveform.numel() > 0 and torch.isfinite(waveform).all()
    assert torch.count_nonzero(waveform) and waveform.abs().max() <= 1
    return waveform


def _request(engine, req_id, *, temperature=0):
    model = engine.model_executor.driver_worker.worker.model_runner.model
    payload = _controlled_payload(model)
    return OmniRequest(
        request_id=req_id,
        prompt_token_ids=payload["meta"]["next_stage_prompt_ids"],
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=18,
            ignore_eos=True,
            repetition_penalty=1.05,
        ),
        pooling_params=None,
        model_intermediate_buffer=deepcopy(payload),
        block_hasher=engine.request_block_hasher,
        cache_salt=f"test-controlled-conditioning-456-temperature-{temperature}",
    )


def _assert_no_pending_talker_embeddings(runner):
    for state in runner.requests.values():
        assert getattr(state, "talker_next_input_embedding", None) is None


def _generate(engine, req_id, *, preempt_after=None, temperature=0, evict_prefix=False):
    model = engine.model_executor.driver_worker.worker.model_runner.model
    request = _request(engine, req_id, temperature=temperature)
    engine.add_request(request)
    outputs: list[OmniEngineCoreOutput] = []
    predictor_trace = []
    forward_trace = []
    predict = model.talker.code_predictor_forward
    runner = engine.model_executor.driver_worker.worker.model_runner
    forward = runner._model_forward

    def traced_forward(*args, **kwargs):
        # Observe scheduled rows before the graph boundary. A model.forward
        # hook is bypassed on replay, and padding is not a request input row.
        num_rows = int(runner.query_start_loc.cpu[runner.input_batch.num_reqs])
        if os.environ.get("VLLM_OMNI_TEST_TALKER_GRAPH") == "full":
            context = get_forward_context()
            assert context.cudagraph_runtime_mode == CUDAGraphMode.FULL
            assert context.batch_descriptor is not None
            assert context.batch_descriptor.num_tokens >= num_rows
        embeds = kwargs["inputs_embeds"][:num_rows].detach().cpu().clone()
        result = forward(*args, **kwargs)
        forward_trace.append(embeds)
        return result

    def traced_predict(*args, **kwargs):
        rng = torch.cuda.get_rng_state().clone()
        hidden = kwargs["last_talker_hidden"].detach().cpu().clone()
        result = predict(*args, **kwargs)
        predictor_trace.append((rng, hidden, result[0].detach().cpu().clone()))
        return result

    preempted = False
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(model.talker, "code_predictor_forward", traced_predict)
        patch.setattr(runner, "_model_forward", traced_forward)
        for _ in range(48):
            result, _ = engine.step_fn()
            outputs.extend(output for batch in (result or {}).values() for output in batch.outputs)
            _assert_no_pending_talker_embeddings(runner)
            if request.is_finished() and not engine.batch_queue:
                break
            if (
                preempt_after is not None
                and request.num_output_tokens >= preempt_after
                and not preempted
                and not request.is_finished()
            ):
                scheduler = engine.scheduler
                if engine.vllm_config.scheduler_config.async_scheduling:
                    assert request.num_in_flight_tokens > 0, "exercise real in-flight preemption, not a drained pause"
                scheduler.running.remove(request)
                scheduler._preempt_request(request, time.monotonic())
                assert scheduler.kv_cache_manager.get_computed_blocks(request)[1] >= request.num_prompt_tokens
                if evict_prefix:
                    # Pause this waiting request while real queued output
                    # drains its native block-free fence. Then simulate full
                    # eviction through the public reset, not block-pool edits.
                    scheduler.waiting.remove_request(request)
                    while engine.batch_queue:
                        drained, _ = engine.step_fn()
                        outputs.extend(output for batch in (drained or {}).values() for output in batch.outputs)
                        _assert_no_pending_talker_embeddings(runner)
                    # Forced preemption must also invalidate the scheduler's
                    # previous-batch token snapshot (native reset contract).
                    assert engine.reset_prefix_cache(reset_running_requests=True)
                    assert scheduler.kv_cache_manager.get_computed_blocks(request)[1] == 0
                    scheduler.waiting.add_request(request)
                preempted = True
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.num_output_tokens == 18
    assert sum(len(output.new_token_ids) for output in outputs) == 18
    assert all(0 <= token < model.talker_config.text_config.vocab_size for token in request.output_token_ids)
    assert preempted == (preempt_after is not None)
    return request, outputs, predictor_trace, forward_trace


def _emitted_codes(outputs):
    rows = [
        output.multimodal_output["codes.audio"]
        for output in outputs
        if output.multimodal_output and "codes.audio" in output.multimodal_output
    ]
    assert rows, "the real output route must deliver audio codes"
    return torch.cat(rows)


@pytest.fixture(scope="module")
def cache_off_reference():
    # The full qualification module runs cache-off first. Keep its actual
    # primary/code/wave outputs as an independent legacy-ordering oracle.
    return {}


@pytest.mark.parametrize("mixed_media", [False, True], ids=["text", "mixed-media"])
@torch.inference_mode()
def test_checkpoint_cpu_mask_matches_original_gpu_projection(talker_engine, mixed_media):
    model = talker_engine.model_executor.driver_worker.worker.model_runner.model
    payload = _controlled_payload(model)
    embeds = payload["embed"]["prefill"][:8].to("cuda")
    hidden = payload["hidden_states"]["output"][:8].to("cuda")
    cpu_mask = torch.tensor([False, mixed_media] * 4)
    gpu_mask = cpu_mask.to("cuda")

    # Independent original assembly: select GPU rows, project, then scatter.
    expected = torch.empty((8, model.talker_config.text_config.hidden_size), device="cuda", dtype=torch.bfloat16)
    if mixed_media:
        expected[gpu_mask] = model.talker.hidden_projection(hidden[gpu_mask])
    expected[~gpu_mask] = model.talker.text_projection(embeds[~gpu_mask])

    actual = model._get_talker_user_parts(0, 8, cpu_mask, hidden, embeds)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("temperature", [0, 0.7], ids=["greedy", "sampled"])
def test_full_checkpoint_talker_engine_generates_to_native_length_cap(
    talker_engine, checkpoint_code2wav, cache_off_reference, temperature
):
    engine = talker_engine
    torch.manual_seed(321)
    cold, cold_outputs, _, cold_inputs = _generate(engine, "checkpoint-cold", temperature=temperature)
    cold_codes = _emitted_codes(cold_outputs)
    assert cold_codes.shape == (cold.num_prompt_tokens + 17, 16)
    torch.testing.assert_close(
        cold_codes[cold.num_prompt_tokens :, 0],
        torch.tensor(cold.output_token_ids[:-1], device=cold_codes.device, dtype=cold_codes.dtype),
        rtol=0,
        atol=0,
    )
    assert cold_inputs[0].shape[0] == cold.num_prompt_tokens
    cold_wave = _decode_emitted_codes(checkpoint_code2wav, cold, cold_outputs)
    if engine.vllm_config.cache_config.enable_prefix_caching:
        assert temperature in cache_off_reference, "run the complete module to include its cache-off oracle"
        off_tokens, off_codes, off_wave = cache_off_reference[temperature]
        assert list(cold.output_token_ids) == off_tokens
        torch.testing.assert_close(cold_codes, off_codes, rtol=0, atol=0)
        torch.testing.assert_close(cold_wave, off_wave, rtol=0, atol=0)
        cache = engine.model_executor.driver_worker.worker.model_runner.omni_prefix_cache
        assert not cache._policy.needs_full_hidden_states
        assert cache._policy.hidden_key is None
        assert cache._policy.deferred_keys == frozenset({"codes.audio"})
        assert not hasattr(cold, "talker_codec_inputs")
        assert not hasattr(cold, "talker_terminal_input")
        torch.manual_seed(321)
        warm, warm_outputs, _, _ = _generate(engine, "checkpoint-warm", temperature=temperature)
        torch.testing.assert_close(_emitted_codes(warm_outputs), cold_codes, rtol=0, atol=0)
        torch.testing.assert_close(
            _decode_emitted_codes(checkpoint_code2wav, warm, warm_outputs), cold_wave, rtol=0, atol=0
        )
        assert any(
            output.prefill_stats is not None and output.prefill_stats.num_cached_tokens > 0 for output in warm_outputs
        )
        assert list(warm.output_token_ids) == list(cold.output_token_ids)
        assert not hasattr(warm, "talker_codec_inputs")
        torch.manual_seed(321)
        resumed, resumed_outputs, _, _ = _generate(
            engine, "checkpoint-resumed", preempt_after=12, temperature=temperature
        )
        torch.testing.assert_close(_emitted_codes(resumed_outputs), cold_codes, rtol=0, atol=0)
        torch.testing.assert_close(
            _decode_emitted_codes(checkpoint_code2wav, resumed, resumed_outputs), cold_wave, rtol=0, atol=0
        )
        assert list(resumed.output_token_ids) == list(cold.output_token_ids)
        assert not hasattr(resumed, "talker_codec_inputs")
        torch.manual_seed(321)
        evicted, evicted_outputs, _, _ = _generate(
            engine, "checkpoint-evicted", preempt_after=12, temperature=temperature, evict_prefix=True
        )
        assert list(evicted.output_token_ids) == list(cold.output_token_ids)
        torch.testing.assert_close(_emitted_codes(evicted_outputs), cold_codes, rtol=0, atol=0)
        torch.testing.assert_close(
            _decode_emitted_codes(checkpoint_code2wav, evicted, evicted_outputs), cold_wave, rtol=0, atol=0
        )
    else:
        cache_off_reference[temperature] = (list(cold.output_token_ids), cold_codes.clone(), cold_wave.clone())


@pytest.mark.parametrize("terminal_kind", ["eos", "stop"])
def test_full_checkpoint_mixed_batch_terminal_drain_keeps_survivor(talker_engine, terminal_kind, monkeypatch):
    engine = talker_engine
    if not engine.vllm_config.cache_config.enable_prefix_caching:
        pytest.skip("terminal history/drain protocol is specific to the guarded cache path")
    torch.manual_seed(321)
    reference, reference_outputs, _, _ = _generate(engine, f"reference-{terminal_kind}")
    # Make an actually sampled primary the request's EOS/stop token. No logits,
    # transformer, primary sampler or residual predictor is mocked.
    stopped = _request(engine, f"{terminal_kind}-first")
    survivor = _request(engine, f"{terminal_kind}-survivor")
    token = reference.output_token_ids[0]
    if terminal_kind == "eos":
        stopped.sampling_params.ignore_eos = False
        stopped.sampling_params._eos_token_id = token
    else:
        stopped.sampling_params.stop_token_ids = [token]
    runner = engine.model_executor.driver_worker.worker.model_runner
    predict = runner._predict_talker_codes
    predictor_batches = []

    def predict_live(*args, **kwargs):
        predictor_batches.append(args[0].shape[0])
        return predict(*args, **kwargs)

    # Count at the runner boundary: captured predictor kernels replay without
    # calling the Python model method, but terminal rows must still be excluded.
    monkeypatch.setattr(runner, "_predict_talker_codes", predict_live)
    outputs: dict[str, list[OmniEngineCoreOutput]] = {stopped.request_id: [], survivor.request_id: []}
    torch.manual_seed(321)
    engine.add_request(stopped)
    engine.add_request(survivor)
    for _ in range(48):
        result, _ = engine.step_fn()
        for batch in (result or {}).values():
            for output in batch.outputs:
                outputs[output.request_id].append(output)
        if stopped.is_finished() and survivor.is_finished() and not engine.batch_queue:
            break
    assert stopped.status == RequestStatus.FINISHED_STOPPED
    assert list(stopped.output_token_ids) == [token]
    assert not hasattr(stopped, "talker_codec_inputs")
    assert not hasattr(stopped, "talker_terminal_input")
    assert _emitted_codes(outputs[stopped.request_id]).shape == (stopped.num_prompt_tokens, 16)
    assert survivor.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert list(survivor.output_token_ids) == list(reference.output_token_ids)
    assert predictor_batches == [1] * 17
    torch.testing.assert_close(
        _emitted_codes(outputs[survivor.request_id]), _emitted_codes(reference_outputs), rtol=0, atol=0
    )


@pytest.mark.parametrize("evict_prefix", [False, True], ids=["prefix-retained", "prefix-evicted"])
@torch.inference_mode()
def test_full_checkpoint_codec_history_survives_preemption(talker_engine, evict_prefix, monkeypatch):
    engine = talker_engine
    if not engine.vllm_config.cache_config.enable_prefix_caching:
        pytest.skip("request-local codec replay requires the guarded cache path")
    runner = engine.model_executor.driver_worker.worker.model_runner
    predict = runner._predict_talker_codes
    predictor_rows = []

    def count_prediction(input_ids, *args, **kwargs):
        predictor_rows.append(input_ids.shape[0])
        return predict(input_ids, *args, **kwargs)

    monkeypatch.setattr(runner, "_predict_talker_codes", count_prediction)
    torch.manual_seed(321)
    reference, reference_outputs, _, _ = _generate(engine, "codec-reference")
    assert predictor_rows == [1] * 17
    predictor_rows.clear()
    torch.manual_seed(321)
    resumed, resumed_outputs, _, _ = _generate(engine, "codec-resumed", preempt_after=12, evict_prefix=evict_prefix)
    assert predictor_rows == [1] * 17, "recomputed codec inputs must not be sampled again"
    assert list(resumed.output_token_ids) == list(reference.output_token_ids)
    torch.testing.assert_close(_emitted_codes(resumed_outputs), _emitted_codes(reference_outputs), rtol=0, atol=0)


def _generate_pair(engine, name, *, preempt=False, evict=False, reorder=False):
    """Keep both requests live to exercise B2 MTP, including pending replay."""
    assert not engine.vllm_config.scheduler_config.async_scheduling
    requests = [_request(engine, f"{name}-{index}") for index in range(2)]
    outputs: dict[str, list[OmniEngineCoreOutput]] = {request.request_id: [] for request in requests}
    runner = engine.model_executor.driver_worker.worker.model_runner
    invoke = runner._invoke_talker_mtp
    forward = runner._model_forward
    batch_sizes = []
    batch_orders = []
    retained = []

    def invoke_live(batch_size, **kwargs):
        batch_sizes.append(batch_size)
        batch_orders.append(tuple(runner.input_batch.req_ids))
        return invoke(batch_size, **kwargs)

    def forward_live(*args, **kwargs):
        if os.environ.get("VLLM_OMNI_TEST_TALKER_GRAPH") == "full":
            assert get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.FULL
        return forward(*args, **kwargs)

    for request in requests:
        engine.add_request(request)
    preempted = False
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runner, "_invoke_talker_mtp", invoke_live)
        patch.setattr(runner, "_model_forward", forward_live)
        for _ in range(64):
            result, _ = engine.step_fn()
            for batch in (result or {}).values():
                for output in batch.outputs:
                    outputs[output.request_id].append(output)
            _assert_no_pending_talker_embeddings(runner)
            if all(request.is_finished() for request in requests) and not engine.batch_queue:
                break
            if preempt and not preempted and all(request.num_output_tokens >= 6 for request in requests):
                for request in requests:
                    state = runner.requests[request.request_id]
                    retained.append((state, torch.stack(state.talker_codec_inputs).detach().cpu().clone()))
                # Native preemption prepends requests to its waiting queue.
                # Preserve row order for exact stochastic-continuation checks;
                # separately exercise reordered recovery without that oracle.
                for request in requests if reorder else reversed(requests):
                    assert not request.is_finished()
                    engine.scheduler.running.remove(request)
                    engine.scheduler._preempt_request(request, time.monotonic())
                if evict:
                    assert engine.reset_prefix_cache(reset_running_requests=True)
                    assert all(
                        engine.scheduler.kv_cache_manager.get_computed_blocks(request)[1] == 0 for request in requests
                    )
                preempted = True
    assert preempted == preempt
    assert batch_sizes == [2] * 17, "pending inputs must stay batched and never be sampled twice"
    original_order = tuple(request.request_id for request in requests)
    expected_orders = [original_order] * 17
    if reorder:
        expected_orders[5:] = [tuple(reversed(original_order))] * 12
    assert batch_orders == expected_orders
    for state, accepted_codes in retained:
        assert len(state.talker_codec_inputs) == 17
        torch.testing.assert_close(
            torch.stack(state.talker_codec_inputs[: accepted_codes.shape[0]]).cpu(), accepted_codes, rtol=0, atol=0
        )
    for request in requests:
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert request.num_output_tokens == 18
        assert sum(len(output.new_token_ids) for output in outputs[request.request_id]) == 18
        assert _emitted_codes(outputs[request.request_id]).shape == (request.num_prompt_tokens + 17, 16)
    return [(request, outputs[request.request_id]) for request in requests]


def test_full_checkpoint_new_prefill_preserves_pending_history(talker_engine, monkeypatch):
    engine = talker_engine
    if not engine.vllm_config.cache_config.enable_prefix_caching:
        pytest.skip("request-local ordering is specific to the guarded cache path")
    assert engine.reset_prefix_cache()
    torch.manual_seed(321)
    accepted = _request(engine, "ongoing-before-prefill")
    fresh = _request(engine, "new-during-decode")
    runner = engine.model_executor.driver_worker.worker.model_runner
    outputs: dict[str, list[OmniEngineCoreOutput]] = {accepted.request_id: [], fresh.request_id: []}
    materialize = runner._materialize_pending_talker_history
    batch_sizes = []
    checked_mixed = False
    hidden_before = None
    codes_before = None

    def materialize_live(items):
        nonlocal checked_mixed
        if fresh.request_id in runner.requests and not runner.requests[fresh.request_id].output_token_ids:
            # Fresh prefill must already have initialized its own cursor, while
            # the other request's pending producer still sees its original state.
            assert (
                runner.model_intermediate_buffer[fresh.request_id]["meta"]["num_processed_tokens"]
                == fresh.num_prompt_tokens
            )
            torch.testing.assert_close(
                runner.model_intermediate_buffer[accepted.request_id]["hidden_states"]["last"],
                hidden_before,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                torch.stack(runner.requests[accepted.request_id].talker_codec_inputs),
                codes_before,
                rtol=0,
                atol=0,
            )
            checked_mixed = True
        pending_before = sum(
            len(runner.requests[req_id].talker_codec_inputs) < len(runner.requests[req_id].output_token_ids)
            for req_id, *_ in items
        )
        materialize(items)
        if pending_before:
            batch_sizes.append(pending_before)

    monkeypatch.setattr(runner, "_materialize_pending_talker_history", materialize_live)
    engine.add_request(accepted)
    added = False
    for _ in range(64):
        result, _ = engine.step_fn()
        for batch in (result or {}).values():
            for output in batch.outputs:
                outputs[output.request_id].append(output)
        _assert_no_pending_talker_embeddings(runner)
        if not added and accepted.num_output_tokens >= 6:
            hidden_before = runner.model_intermediate_buffer[accepted.request_id]["hidden_states"]["last"].clone()
            codes_before = torch.stack(runner.requests[accepted.request_id].talker_codec_inputs).clone()
            engine.add_request(fresh)
            added = True
        if added and accepted.is_finished() and fresh.is_finished() and not engine.batch_queue:
            break
    assert checked_mixed
    assert max(batch_sizes) == 2
    assert sum(batch_sizes) == 34, "each nonterminal accepted primary is materialized exactly once"
    for request in (accepted, fresh):
        assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
        assert sum(len(output.new_token_ids) for output in outputs[request.request_id]) == 18
        assert _emitted_codes(outputs[request.request_id]).shape == (request.num_prompt_tokens + 17, 16)


@pytest.mark.parametrize("evict_prefix", [False, True], ids=["prefix-retained", "prefix-evicted"])
@torch.inference_mode()
def test_full_checkpoint_batched_codec_history_survives_preemption(talker_engine, evict_prefix):
    engine = talker_engine
    if not engine.vllm_config.cache_config.enable_prefix_caching:
        pytest.skip("request-local codec replay requires the guarded cache path")
    assert engine.reset_prefix_cache()
    torch.manual_seed(321)
    reference = _generate_pair(engine, "codec-batch-reference")
    torch.manual_seed(321)
    resumed = _generate_pair(engine, "codec-batch-resumed", preempt=True, evict=evict_prefix)
    for (before, before_outputs), (after, after_outputs) in zip(reference, resumed, strict=True):
        assert list(after.output_token_ids) == list(before.output_token_ids)
        torch.testing.assert_close(_emitted_codes(after_outputs), _emitted_codes(before_outputs), rtol=0, atol=0)
    # Reordering can change future global RNG draws, but never accepted codes.
    torch.manual_seed(321)
    reordered = _generate_pair(engine, "codec-batch-reordered", preempt=True, evict=True, reorder=True)
    for (before, before_outputs), (after, after_outputs) in zip(reference, reordered, strict=True):
        assert list(after.output_token_ids[:6]) == list(before.output_token_ids[:6])
        prefix_end = before.num_prompt_tokens + 5
        torch.testing.assert_close(
            _emitted_codes(after_outputs)[:prefix_end], _emitted_codes(before_outputs)[:prefix_end], rtol=0, atol=0
        )


def test_full_checkpoint_two_live_requests_keep_batched_mtp(talker_engine, checkpoint_code2wav, cache_off_reference):
    engine = talker_engine
    assert engine.reset_prefix_cache()
    torch.manual_seed(321)
    cold = _generate_pair(engine, "batch-cold")
    reference = [
        (
            list(request.output_token_ids),
            _emitted_codes(outputs).clone(),
            _decode_emitted_codes(checkpoint_code2wav, request, outputs),
        )
        for request, outputs in cold
    ]
    if not engine.vllm_config.cache_config.enable_prefix_caching:
        cache_off_reference["batch"] = reference
        return
    assert "batch" in cache_off_reference, "run the complete module with its cache-off oracle"
    for before, after in zip(cache_off_reference["batch"], reference, strict=True):
        assert before[0] == after[0]
        torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
        torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
    for name, preempt, evict in (("warm", False, False), ("resume", True, False), ("evicted", True, True)):
        torch.manual_seed(321)
        actual = _generate_pair(engine, f"batch-{name}", preempt=preempt, evict=evict)
        if name == "warm":
            assert all(
                any(output.prefill_stats and output.prefill_stats.num_cached_tokens > 0 for output in outputs)
                for _, outputs in actual
            )
        for (request, outputs), (tokens, codes, wave) in zip(actual, reference, strict=True):
            assert list(request.output_token_ids) == tokens, name
            torch.testing.assert_close(_emitted_codes(outputs), codes, rtol=0, atol=0)
            torch.testing.assert_close(
                _decode_emitted_codes(checkpoint_code2wav, request, outputs), wave, rtol=0, atol=0
            )
    # Global residual sampling is row-ordered. Reordering may change future
    # samples, but must not regenerate accepted input decisions or duplicate
    # their delivery. The helper also checks the actual reordered MTP batches.
    torch.manual_seed(321)
    reordered = _generate_pair(engine, "batch-reordered", preempt=True, evict=True, reorder=True)
    for (request, outputs), (tokens, codes, _wave) in zip(reordered, reference, strict=True):
        assert list(request.output_token_ids[:6]) == tokens[:6]
        prefix_end = request.num_prompt_tokens + 5
        torch.testing.assert_close(_emitted_codes(outputs)[:prefix_end], codes[:prefix_end], rtol=0, atol=0)
        _decode_emitted_codes(checkpoint_code2wav, request, outputs)
