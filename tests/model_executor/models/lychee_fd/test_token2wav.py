# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.lychee_fd.token2wav import (
    LycheeSpeechOwner,
    LycheeSpeechState,
    LycheeToken2WavCore,
    LycheeToken2WavForConditionalGeneration,
    LycheeToken2WavSessionStore,
)
from vllm_omni.model_executor.stage_input_processors.lychee_fd import lychee2token2wav

pytestmark = [pytest.mark.core_model]


class FakeCore:
    sample_rate = 24000
    chunk_size = 25
    pre_lookahead_len = 3

    def __init__(self):
        self.calls = []
        self.flushes = 0

    def synthesize(self, tokens, state, *, final):
        self.calls.append((list(tokens), state.owner, final))
        state.stream_cache = {"owned": torch.ones(1)}
        state.hift_cache = {"speech": torch.tensor([9.0, 10.0])}
        return torch.tensor([float(len(tokens)), float(final)])

    def flush_tail(self, state):
        self.flushes += 1
        return state.hift_cache["speech"].clone()


def meta(seq=0, response="r0", epoch=0, **kwargs):
    return dict(
        session_id="s0",
        response_id=response,
        execution_epoch=epoch,
        chunk_seq=seq,
        request_id="engine0",
        response_number=kwargs.pop("response_number", 1 if response == "r0" else 2),
        **kwargs,
    )


def store():
    core = FakeCore()
    return LycheeToken2WavSessionStore(core, "prompt.wav"), core


def test_accumulate_codec_lookahead_and_consume_only_chunk():
    sessions, core = store()
    assert sessions.process(list(range(10)), meta())[0].numel() == 0
    assert sessions.process(list(range(10, 27)), meta(1))[0].numel() == 0
    waveform, _ = sessions.process([27], meta(2))
    assert waveform.tolist() == [28.0, 0.0]
    assert core.calls[0][0] == list(range(28))
    assert next(iter(sessions.states.values())).pending_tokens == [25, 26, 27]
    sessions.process(list(range(28, 53)), meta(3))
    assert core.calls[1][0] == list(range(25, 53))
    assert next(iter(sessions.states.values())).pending_tokens == [50, 51, 52]


def test_natural_final_synthesizes_remainder_once():
    sessions, core = store()
    sessions.process(list(range(28)), meta())
    waveform, owned = sessions.process([28], meta(1, final=True))
    assert core.calls[-1] == ([25, 26, 27, 28], LycheeSpeechOwner("s0", "r0", 0), True)
    assert waveform.tolist() == [4.0, 1.0]
    assert owned["final"] is True
    assert not sessions.states
    assert sessions.process([], meta(2, final=True))[1]["discarded"] is True
    assert len(core.calls) == 2


def test_empty_final_returns_retained_tail_once():
    sessions, core = store()
    sessions.process([], meta())
    state = next(iter(sessions.states.values()))
    state.stream_cache = {"owned": torch.ones(1)}
    state.hift_cache = {"speech": torch.tensor([9.0, 10.0])}
    waveform, _ = sessions.process([], meta(1, final=True))
    assert waveform.tolist() == [9.0, 10.0]
    assert core.flushes == 1
    sessions.process([], meta(2, final=True))
    assert core.flushes == 1


def test_empty_response_final_never_loads_or_synthesizes():
    sessions, core = store()
    assert sessions.process([], meta(final=True))[0].numel() == 0
    assert not core.calls and not sessions.states


def test_cancel_and_late_payload_never_flush():
    sessions, core = store()
    sessions.process(list(range(28)), meta())
    assert sessions.process([], meta(1, cancel=True))[0].numel() == 0
    assert sessions.process([3], meta(2, final=True))[1]["discarded"] is True
    assert core.flushes == 0 and len(core.calls) == 1


def test_new_epoch_discards_old_output_and_resets_codec_pending():
    sessions, core = store()
    sessions.process(list(range(20)), meta())
    sessions.process([11], meta(response="r1", epoch=1))
    old = sessions.process(list(range(28)), meta(1, final=True))[1]
    assert old["discarded"] is True
    state = next(iter(sessions.states.values()))
    assert state.pending_tokens == [11] and state.owner.execution_epoch == 1
    assert not core.calls


def test_response_switch_has_fresh_caches_even_in_same_epoch():
    sessions, core = store()
    sessions.process(list(range(28)), meta())
    sessions.process([14], meta(response="r1"))
    state = next(iter(sessions.states.values()))
    assert state.pending_tokens == [14] and state.stream_cache is None
    assert sessions.process([22], meta(1))[1]["discarded"] is True


def test_duplicate_chunk_is_idempotent_and_gap_poisoned():
    sessions, core = store()
    sessions.process(list(range(28)), meta())
    assert sessions.process(list(range(28)), meta())[1]["discarded"] is True
    assert len(core.calls) == 1
    with pytest.raises(ValueError, match="gap"):
        sessions.process([29], meta(3))
    assert not sessions.states
    assert sessions.process([30], meta(4))[1]["discarded"] is True


@pytest.mark.parametrize("token", [-1, 6561, True, 0.5])
def test_invalid_codec_poisons_response(token):
    sessions, _ = store()
    with pytest.raises(ValueError, match="codec IDs"):
        sessions.process([token], meta())
    assert not sessions.states


def test_engine_finish_drops_all_owned_response_caches_without_flush():
    sessions, core = store()
    sessions.process(list(range(28)), meta())
    sessions.finish_requests({"other"})
    assert sessions.states
    sessions.finish_requests({"engine0"})
    assert not sessions.states and core.flushes == 0


def test_synthesis_failure_releases_owned_caches():
    sessions, core = store()

    def fail(*args, **kwargs):
        raise RuntimeError("synthesis failed")

    core.synthesize = fail
    with pytest.raises(RuntimeError, match="synthesis failed"):
        sessions.process(list(range(28)), meta())
    assert not sessions.states


def test_b1_rejects_another_active_session():
    sessions, _ = store()
    sessions.process([3], meta())
    payload = meta(response="other")
    payload["session_id"] = "s1"
    with pytest.raises(RuntimeError, match="one active response"):
        sessions.process([3], payload)


def test_adapter_extracts_only_plugin_fenced_codec_delta():
    payload = meta(codec_token_ids=torch.tensor([0, 4, 5]))
    output = SimpleNamespace(
        multimodal_output={"lychee_t2w": payload, "lychee_speech_token_ids": torch.tensor([151699])}
    )
    prompts = lychee2token2wav([SimpleNamespace(outputs=[output])])
    assert prompts[0]["prompt_token_ids"] == [0, 4, 5]
    assert not prompts[0]["additional_information"]["lychee_t2w"]["empty"]
    assert "codec_token_ids" in payload
    assert "codec_token_ids" not in prompts[0]["additional_information"]["lychee_t2w"]


def test_adapter_empty_final_uses_explicit_placeholder():
    output = SimpleNamespace(multimodal_output={"lychee_t2w": meta(final=True)})
    prompts = lychee2token2wav([SimpleNamespace(outputs=[output])])
    assert prompts[0]["prompt_token_ids"] == [0]
    assert prompts[0]["additional_information"]["lychee_t2w"]["empty"]


def test_adapter_does_not_use_parked_engine_finished_as_response_eof():
    output = SimpleNamespace(multimodal_output={"lychee_t2w": meta()})
    assert lychee2token2wav([SimpleNamespace(outputs=[output], finished=True)]) == []


def test_wrapper_valid_zero_codec_is_not_profile():
    wrapper = LycheeToken2WavForConditionalGeneration.__new__(LycheeToken2WavForConditionalGeneration)
    nn.Module.__init__(wrapper)
    sessions, core = store()
    wrapper.sessions = sessions
    result = wrapper.forward(
        torch.tensor([0]), torch.tensor([0]), runtime_additional_information=[{"lychee_t2w": meta(final=True)}]
    )
    assert core.calls[0][0] == [0]
    assert result.multimodal_outputs["chunk.lychee_t2w.response_number"].item() == 1


def test_wrapper_empty_final_does_not_convert_placeholder_to_codec():
    wrapper = LycheeToken2WavForConditionalGeneration.__new__(LycheeToken2WavForConditionalGeneration)
    nn.Module.__init__(wrapper)
    sessions, core = store()
    wrapper.sessions = sessions
    result = wrapper.forward(
        torch.tensor([0]),
        torch.tensor([0]),
        runtime_additional_information=[{"lychee_t2w": meta(final=True, empty=True)}],
    )
    assert not core.calls
    assert result.multimodal_outputs["model_outputs"][0].numel() == 0


def test_core_hift_overlap_cache_and_estimator_trim_are_owned():
    core = LycheeToken2WavCore.__new__(LycheeToken2WavCore)
    nn.Module.__init__(core)
    core.device = torch.device("cpu")
    core.float16 = False
    core.speech_window = torch.from_numpy(__import__("numpy").hamming(7680)).float()
    cache = {"estimator_att_cache": torch.arange(160.0).reshape(1, 1, 1, 1, 160, 1).expand(10, 1, 1, 1, 160, 1)}

    class Flow:
        def inference_chunk(self, **kwargs):
            return torch.ones(1, 80, 50), cache

    class HiFT:
        def __call__(self, mel, source):
            return torch.ones(1, mel.shape[2] * 480), torch.ones(1, 1, mel.shape[2] * 480)

    core.flow = Flow()
    core.hift = HiFT()
    core.prepare_prompt = lambda _: (torch.ones(1, 10), torch.ones(1, 192), torch.ones(1, 20, 80))
    state = LycheeSpeechState(LycheeSpeechOwner("s", "r", 0), "prompt.wav", "req")
    state.stream_cache = cache
    state.hift_cache = {"mel": torch.zeros(1, 80, 0), "source": torch.zeros(1, 1, 0), "speech": torch.zeros(1, 0)}
    from torch.utils._python_dispatch import TorchDispatchMode

    class CloneShapes(TorchDispatchMode):
        def __init__(self):
            self.shapes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func == torch.ops.aten.clone.default:
                self.shapes.append(tuple(args[0].shape))
            return func(*args, **(kwargs or {}))

    clone_shapes = CloneShapes()
    with clone_shapes:
        waveform = core.synthesize(list(range(28)), state, final=False)
    assert (10, 1, 1, 1, 120, 1) not in clone_shapes.shapes
    assert waveform.numel() == 50 * 480 - 3840
    assert state.stream_cache["estimator_att_cache"].shape[4] == 120
    assert state.stream_cache["estimator_att_cache"].data_ptr() != cache["estimator_att_cache"].data_ptr()
    assert state.stream_cache["estimator_att_cache"][0, 0, 0, 0, :, 0].tolist() == (
        list(range(20)) + list(range(60, 160))
    )
    assert state.hift_cache["mel"].shape[-1] == 8
    assert state.hift_cache["speech"].shape[-1] == 3840
    waveform = core.synthesize([4, 5, 6], state, final=True)
    assert waveform.numel() == 58 * 480


def test_session_epoch_reopen_rejects_old_packets():
    sessions, core = store()
    sessions.process([10], meta(session_epoch=0))
    sessions.process([11], meta(session_epoch=1))
    assert sessions.process(list(range(28)), meta(1, session_epoch=0))[1]["discarded"]
    state = next(iter(sessions.states.values()))
    assert state.owner.session_epoch == 1 and state.pending_tokens == [11]


def test_adapter_rejects_tokenizer_only_codec_not_in_checkpoint():
    output = SimpleNamespace(multimodal_output={"lychee_t2w": meta(codec_token_ids=[6655])})
    with pytest.raises(ValueError, match="6561"):
        lychee2token2wav([SimpleNamespace(outputs=[output])])


def test_native_synthesis_load_restores_ar_default_dtype():
    from vllm.utils.torch_utils import set_default_torch_dtype

    core = LycheeToken2WavCore.__new__(LycheeToken2WavCore)
    nn.Module.__init__(core)
    core.flow = None
    seen = []
    core._load_models_fp32 = lambda: seen.append(torch.get_default_dtype())
    with set_default_torch_dtype(torch.bfloat16):
        core.load_models()
        assert torch.get_default_dtype() == torch.bfloat16
    assert seen == [torch.float32]


def test_engine_wire_and_runtime_buffer_preserve_complete_ownership_metadata():
    import msgspec

    from vllm_omni.engine import AdditionalInformationPayload
    from vllm_omni.engine.serialization import serialize_additional_information
    from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

    metadata = meta(response_number=3, session_epoch=2, tick=49, final=False, empty=False)
    wire = serialize_additional_information({"lychee_t2w": metadata})
    decoded_wire = msgspec.msgpack.decode(msgspec.msgpack.encode(wire), type=AdditionalInformationPayload)
    buffer = OmniIntermediateBuffer(1)
    buffer.add_request(0, SimpleNamespace(req_id="engine0", additional_information=decoded_wire, mm_features=[]))
    gathered = buffer.gather(SimpleNamespace(idx_mapping_np=[0]))
    assert gathered[0]["lychee_t2w"] == metadata
    assert gathered[0]["req_id"] == "engine0"


def test_native_weight_loader_does_not_consume_ar_checkpoint_iterator():
    wrapper = LycheeToken2WavForConditionalGeneration.__new__(LycheeToken2WavForConditionalGeneration)
    nn.Module.__init__(wrapper)
    loaded = []
    wrapper.core = SimpleNamespace(load_models=lambda: loaded.append(True))

    def unused_ar_weights():
        raise AssertionError("AR checkpoint must not be read by synthesis stage")
        yield

    assert wrapper.load_weights(unused_ar_weights()) == set()
    assert loaded == [True]


def test_three_hundred_responses_keep_only_session_highwater_and_fence_old_payload():
    sessions, core = store()
    for number in range(1, 301):
        sessions.process([number], meta(response=f"response-{number}", response_number=number, final=True))
    assert not sessions.states and not sessions.active
    assert len(sessions.highwater) == len(sessions.closed_highwater) == 1
    assert sessions.closed_highwater["s0"] == (0, 0, 300)
    assert sessions.process([4], meta(response="response-1", response_number=1, final=True))[1]["discarded"]
    assert len(core.calls) == 300
    sessions.finish_requests({"engine0"})
    assert not sessions.highwater and not sessions.closed_highwater
    assert not sessions.session_requests and not sessions.request_sessions
    assert sessions.process([4], meta(response="response-301", response_number=301, final=True))[1]["discarded"]


def test_retired_request_fences_are_bounded():
    sessions, _ = store()
    sessions.finish_requests({f"owner-{number}" for number in range(300)})
    assert len(sessions.retired_requests) == 128


def _write_flow_config(tmp_path, **changes):
    from copy import deepcopy

    import yaml

    from vllm_omni.model_executor.models.lychee_fd.token2wav import _FLOW_CONTRACT, _FLOW_TAGS

    contract = deepcopy(_FLOW_CONTRACT)
    for path, value in changes.items():
        parts = path.split("__")
        target = contract
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    source = yaml.safe_dump(contract, sort_keys=False)
    for path, tag in _FLOW_TAGS.items():
        key = path.split(".")[-1]
        source = source.replace(f"{key}:\n", f"{key}: {tag}\n")
    target = tmp_path / "flow.yaml"
    target.write_text(source)
    return target


def test_released_flow_contract_is_accepted_without_yaml_constructor_execution(tmp_path):
    from vllm_omni.model_executor.models.lychee_fd.token2wav import validate_native_flow_config

    validate_native_flow_config(_write_flow_config(tmp_path))


@pytest.mark.parametrize(
    "setting,value",
    [
        ("flow__encoder__pre_lookahead_len", 4),
        ("flow__encoder__up_stride", 3),
        ("flow__decoder__inference_cfg_rate", 0.0),
    ],
)
def test_nonweight_flow_configuration_is_rejected_before_cuda_model_allocation(tmp_path, setting, value):
    path = _write_flow_config(tmp_path, **{setting: value})
    with pytest.raises(ValueError, match=setting.split("__")[-1]):
        LycheeToken2WavCore(str(path.parent), device="cuda")


def test_unimplemented_flow_setting_is_rejected(tmp_path):
    from vllm_omni.model_executor.models.lychee_fd.token2wav import validate_native_flow_config

    path = _write_flow_config(tmp_path, flow__encoder__activation_type="relu")
    with pytest.raises(ValueError, match="unsupported.*activation_type"):
        validate_native_flow_config(path)


def test_yaml_executable_constructor_tag_is_rejected_without_execution(tmp_path):
    from vllm_omni.model_executor.models.lychee_fd.token2wav import validate_native_flow_config

    path = _write_flow_config(tmp_path)
    path.write_text(
        path.read_text().replace("!new:cosyvoice2.flow.flow.CausalMaskedDiffWithXvec", "!!python/object:os.system")
    )
    with pytest.raises(ValueError, match="constructor"):
        validate_native_flow_config(path)


def _materialize_generation_output(result, asynchronous):
    from vllm_omni.worker_v2.omni_ar_model_runner import _ensure_tensor_values
    from vllm_omni.worker_v2.omni_generation_model_runner import (
        OmniGenerationAsyncOutput,
        OmniGenerationModelRunner,
    )

    if not asynchronous:
        return _ensure_tensor_values(OmniGenerationModelRunner._build_pooler_output(result, 1)[0])
    # Exercise real D2H completion on CPU, replacing only the CUDA event.
    async_output = object.__new__(OmniGenerationAsyncOutput)
    synchronized = []
    async_output.copy_event = SimpleNamespace(synchronize=lambda: synchronized.append(True))
    async_output.pending_aux_output = None
    async_output._has_fault = None
    async_output.num_reqs = 1
    async_output.multimodal_outputs_cpu = result.multimodal_outputs
    async_output.model_runner_output = SimpleNamespace(multimodal_outputs=None)
    async_output.finalize_output = None
    payload = async_output.get_output().multimodal_outputs[0]
    assert synchronized == [True]
    return payload


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_numeric_ownership_survives_actual_generation_output_materializer(asynchronous):
    from vllm_omni.data_entry_keys import unflatten_payload

    wrapper = LycheeToken2WavForConditionalGeneration.__new__(LycheeToken2WavForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.sessions, _ = store()
    result = wrapper.forward(
        torch.tensor([0, 7]),
        torch.tensor([0, 1]),
        runtime_additional_information=[
            {"lychee_t2w": meta(epoch=4, session_epoch=3, response_number=9, tick=49, final=True)}
        ],
    )
    payload = _materialize_generation_output(result, asynchronous)
    assert all(isinstance(value, torch.Tensor) for value in payload.values())
    metadata = unflatten_payload(unflatten_payload(payload)["chunk"])["lychee_t2w"]
    assert {key: value.item() for key, value in metadata.items()} == {
        "session_epoch": 3,
        "execution_epoch": 4,
        "response_number": 9,
        "chunk_seq": 0,
        "tick": 49,
        "final": True,
        "discarded": False,
        "num_samples": 2,
    }
    assert payload["model_outputs"].tolist() == [2.0, 1.0]
    assert payload["sr"].item() == 24000
    assert not {"session_id", "response_id", "request_id"}.intersection(metadata)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_cumulative_audio_keeps_each_materialized_owner_and_sample_boundary(asynchronous):
    from vllm_omni.data_entry_keys import unflatten_payload
    from vllm_omni.outputs.mm_outputs import MultimodalPayload
    from vllm_omni.outputs.output_processor import OmniRequestState

    wrapper = LycheeToken2WavForConditionalGeneration.__new__(LycheeToken2WavForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.sessions, _ = store()
    request_state = object.__new__(OmniRequestState)
    request_state.mm_type = "audio"
    request_state.mm_accumulated = MultimodalPayload()
    updates = [
        (list(range(10)), meta(epoch=4, session_epoch=3, response_number=9, tick=10)),
        (list(range(10, 28)), meta(1, epoch=4, session_epoch=3, response_number=9, tick=11)),
        ([], meta(2, epoch=4, session_epoch=3, response_number=9, tick=12, final=True, empty=True)),
        ([7], meta(response="r1", epoch=4, session_epoch=3, response_number=10, tick=13, final=True)),
    ]
    for tokens, metadata in updates:
        result = wrapper.forward(
            torch.tensor(tokens or [0]),
            torch.arange(len(tokens) or 1),
            runtime_additional_information=[{"lychee_t2w": metadata}],
        )
        payload = _materialize_generation_output(result, asynchronous)
        request_state.add_multimodal_tensor(payload, "audio")
        # Match the actual CUMULATIVE output path after every generation step.
        request_state._consolidate_multimodal_tensors()

    payload = request_state.mm_accumulated
    metadata = unflatten_payload(unflatten_payload(dict(payload))["chunk"])["lychee_t2w"]
    assert {key: value.reshape(-1).tolist() for key, value in metadata.items()} == {
        "session_epoch": [3, 3, 3, 3],
        "execution_epoch": [4, 4, 4, 4],
        "response_number": [9, 9, 9, 10],
        "chunk_seq": [0, 1, 2, 0],
        "tick": [10, 11, 12, 13],
        "final": [False, False, True, True],
        "discarded": [False, False, False, False],
        "num_samples": [0, 2, 2, 2],
    }
    waveform = payload["audio"]
    assert waveform.numel() == int(metadata["num_samples"].sum())
    segments = list(waveform.split(metadata["num_samples"].tolist()))
    assert [part.tolist() for part in segments] == [[], [28.0, 0.0], [3.0, 1.0], [1.0, 1.0]]
    assert payload["sr"].item() == 24000


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
def test_resident_delta_generation_drains_waveform_and_each_owner_after_every_response(asynchronous):
    from vllm.sampling_params import RequestOutputKind

    from vllm_omni.data_entry_keys import unflatten_payload
    from vllm_omni.outputs.mm_outputs import MultimodalPayload
    from vllm_omni.outputs.output_processor import OmniRequestState

    wrapper = LycheeToken2WavForConditionalGeneration.__new__(LycheeToken2WavForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.sessions, _ = store()
    request_state = object.__new__(OmniRequestState)
    request_state.mm_type = "audio"
    request_state.mm_accumulated = MultimodalPayload()
    request_state.detokenizer = None
    request_state.request_index = 0
    request_state.output_kind = RequestOutputKind.DELTA
    first_output = None
    for number in range(1, 301):
        result = wrapper.forward(
            torch.tensor([number]),
            torch.tensor([0]),
            runtime_additional_information=[
                {"lychee_t2w": meta(response=f"response-{number}", response_number=number, final=True)}
            ],
        )
        request_state.add_multimodal_tensor(_materialize_generation_output(result, asynchronous), "audio")
        completion = request_state._new_completion_output([], None, None)
        emitted = completion.multimodal_output
        owner = unflatten_payload(emitted["chunk"])["lychee_t2w"]
        assert emitted["audio"].tolist() == [1.0, 1.0]
        assert all(isinstance(value, torch.Tensor) and value.numel() == 1 for value in owner.values())
        assert owner["response_number"].item() == number
        assert owner["num_samples"].item() == 2
        assert owner["final"].item()
        assert "audio" not in request_state.mm_accumulated
        assert not any(key.startswith("chunk.") for key in request_state.mm_accumulated)
        if first_output is None:
            first_output = emitted
    assert first_output is not None
    assert unflatten_payload(first_output["chunk"])["lychee_t2w"]["response_number"].item() == 1
    assert first_output["audio"].tolist() == [1.0, 1.0]
    assert request_state.mm_accumulated["sr"].item() == 24000


@pytest.mark.cpu
@pytest.mark.parametrize(
    "configured,environment,expected", [(None, None, 10), (25, None, 25), (25, "10", 10), (10, "25", 25)]
)
def test_stage_vocoder_hop_matches_released_online_service(monkeypatch, configured, environment, expected):
    import vllm_omni.model_executor.models.lychee_fd.token2wav as module

    monkeypatch.setattr(module, "validate_native_flow_config", lambda path: None)
    monkeypatch.delenv("LYCHEEFD_TTS_VOCODER_HOP_SIZE", raising=False)
    if environment is not None:
        monkeypatch.setenv("LYCHEEFD_TTS_VOCODER_HOP_SIZE", environment)
    settings = {} if configured is None else {"token2wav_vocoder_hop_size": configured}
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(**settings), model="/tmp/models/ar"),
        device_config=SimpleNamespace(device="cpu"),
    )
    stage = LycheeToken2WavForConditionalGeneration(vllm_config=config)
    assert stage.core.chunk_size == expected
    assert stage.core.pre_lookahead_len == 3
    assert stage.core.flow is None and stage.core.hift is None


@pytest.mark.cpu
@pytest.mark.parametrize("hop", [0, -1, 26, True, 10.0, "10"])
def test_vocoder_hop_rejects_invalid_values_before_loading_checkpoint(hop):
    with pytest.raises(ValueError, match="vocoder hop"):
        LycheeToken2WavCore("/missing/checkpoint", device="cpu", chunk_size=hop)


@pytest.mark.cpu
@pytest.mark.parametrize("hop", [10, 25])
def test_configured_vocoder_hop_retains_three_lookahead_and_final_once(hop):
    core = FakeCore()
    core.chunk_size = hop
    sessions = LycheeToken2WavSessionStore(core, "prompt.wav")
    assert sessions.process(list(range(hop + 2)), meta())[0].numel() == 0
    sessions.process([hop + 2], meta(1))
    assert core.calls[0][0] == list(range(hop + 3))
    assert next(iter(sessions.states.values())).pending_tokens == list(range(hop, hop + 3))
    sessions.process([hop + 3], meta(2, final=True))
    assert core.calls[-1][0] == list(range(hop, hop + 4))
    assert core.calls[-1][2] is True
    assert not sessions.states and not sessions.active
    assert sessions.process([], meta(3, final=True))[1]["discarded"] is True
