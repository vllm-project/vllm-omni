# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeTalkerConfig,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_omni.data_entry_keys import OmniPayload, to_dict
from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni as model_module
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import Qwen3OmniMoeForConditionalGeneration
from vllm_omni.model_executor.stage_input_processors.qwen3_omni import thinker2talker_async_chunk
from vllm_omni.worker_v2 import staged_writes
from vllm_omni.worker_v2.model_states.mtp_model_state import MTPState
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_H = 4
_Q = 3


class _Talker(nn.Module):
    num_code_groups = _Q

    def __init__(self):
        super().__init__()
        self.text_projection = nn.Identity()
        self.code_predictor = SimpleNamespace(
            _top_k=50,
            _top_p=0.8,
            set_sampling_params=lambda **kwargs: setattr(self, "sampling", kwargs),
        )


def _talker(*, async_chunk: bool = True) -> Qwen3OmniMoeForConditionalGeneration:
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "talker"
    model.talker = _Talker()
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(async_chunk=async_chunk, use_v2_model_runner=True))
    model.tts_eos_embed = torch.full((_H,), -1.0)
    model.tts_pad_embed = torch.full((_H,), -2.0)
    model._codec_codebook_size = 2048
    return model


def _mtp_state() -> MTPState:
    owner = object.__new__(OmniModelState)
    owner.model = _talker()
    owner.model.has_talker_mtp = False
    owner.device = torch.device("cpu")
    owner._embed_dim = 0
    return MTPState(owner)


def test_mtp_seed_prefers_explicit_tts_local_seed():
    params = SamplingParams(seed=42, extra_args={"tts_local_seed": 7})
    assert model_module.get_qwen3_omni_mtp_seed(params) == 7
    assert Qwen3OmniMoeForConditionalGeneration.get_mtp_seed(params) == 7


def test_mtp_seed_falls_back_to_domain_separated_talker_seed():
    residual_seed = model_module.get_qwen3_omni_mtp_seed(SamplingParams(seed=42))
    assert residual_seed == 0x5157_454E_334D_547A
    assert residual_seed != 42
    assert model_module.get_qwen3_omni_mtp_seed(SamplingParams()) is None


def test_identical_seeded_requests_own_equivalent_independent_mtp_streams():
    state = _mtp_state()
    params = SamplingParams(seed=42)
    first = state._get_mtp_generator("first", params, torch.device("cpu"))
    second = state._get_mtp_generator("second", params, torch.device("cpu"))

    assert first is not None and second is not None and first is not second
    assert first.initial_seed() == second.initial_seed()
    assert torch.equal(torch.rand(16, generator=first), torch.rand(16, generator=second))


def test_mtp_generator_state_persists_across_frames_and_batch_order():
    params = {
        "a": SamplingParams(seed=11),
        "b": SamplingParams(seed=29),
    }

    def draw(order: tuple[str, str]) -> tuple[dict[str, list[torch.Tensor]], dict[str, torch.Generator]]:
        state = _mtp_state()
        streams: dict[str, list[torch.Tensor]] = {req_id: [] for req_id in params}
        generators: dict[str, torch.Generator] = {}
        for frame_order in (order, order[::-1]):
            for req_id in frame_order:
                generator = state._get_mtp_generator(req_id, params[req_id], torch.device("cpu"))
                assert generator is not None
                if req_id in generators:
                    assert generator is generators[req_id]
                generators[req_id] = generator
                streams[req_id].append(torch.rand(8, generator=generator))
        return streams, generators

    forward, forward_generators = draw(("a", "b"))
    reversed_, reversed_generators = draw(("b", "a"))
    for req_id in params:
        assert forward_generators[req_id] is not reversed_generators[req_id]
        assert len(forward[req_id]) == len(reversed_[req_id]) == 2
        for expected, actual in zip(forward[req_id], reversed_[req_id], strict=True):
            assert torch.equal(expected, actual)


def _rows(*values: float) -> torch.Tensor:
    return torch.tensor([[value] * _H for value in values])


def _step(model, payload: dict) -> torch.Tensor:
    update: dict = {}
    text_step = model._thinker_decode_to_talker_decode(payload, torch.device("cpu"), update)
    for key, value in update.items():
        payload.setdefault(key, {}).update(value)
    return text_step.reshape(-1)


def test_mtp_frame_valid_rejects_codec_eos_and_special_ids():
    layer0 = torch.tensor([0, 2047, 2048, 2150, -1])
    assert _talker().mtp_frame_valid(layer0).tolist() == [True, True, False, False, False]


@pytest.mark.parametrize("async_chunk", [False, True])
@pytest.mark.parametrize("profile_only", [False, True])
def test_talker_tensor_forward_keeps_full_graph_capture(async_chunk, profile_only, monkeypatch, mocker):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=Qwen3OmniMoeConfig(),
            multimodal_config=SimpleNamespace(),
            model_stage="talker",
            use_v2_model_runner=True,
            async_chunk=async_chunk,
        ),
        additional_config={},
        with_hf_config=lambda *args, **kwargs: None,
    )
    talker = _Talker()
    hidden = _rows(1)
    mocker.patch.object(talker, "forward", return_value=hidden)
    monkeypatch.setattr(model_module, "init_vllm_registered_model", lambda **kwargs: talker)
    mocker.patch.object(Qwen3OmniMoeForConditionalGeneration, "_init_special_tokens_embeddings", return_value=set())
    model = Qwen3OmniMoeForConditionalGeneration(vllm_config=config)
    assert model.have_multimodal_outputs is True
    assert model.mtp_sample_uniforms is True
    assert model.mtp_sample_uniforms_preserve_eager_stream is True
    assert model.mtp_accepts_per_row_generators is True

    runner = object.__new__(OmniGPUModelRunner)
    runner._staged_writes = staged_writes.StagedWrites()
    runner.model = model
    runner.use_aux_hidden_state_outputs = False
    runner.model_state = SimpleNamespace()
    runner._configure_cudagraph_output_contract()
    full = BatchExecutionDescriptor(CUDAGraphMode.FULL, 1, None)
    piecewise = BatchExecutionDescriptor(CUDAGraphMode.PIECEWISE, 1, None)
    runner.cudagraph_manager = SimpleNamespace(
        _capture_descs={CUDAGraphMode.FULL: [full], CUDAGraphMode.PIECEWISE: [piecewise]},
        _candidates={(1, 0): [full, piecewise]},
    )

    def capture(_runner, *, profile_only=False):
        # Check the candidates that the real Omni capture entrypoint hands to vLLM.
        assert runner.cudagraph_manager._capture_descs[CUDAGraphMode.FULL] == [full]
        assert full in runner.cudagraph_manager._candidates[(1, 0)]
        assert model.forward(torch.zeros(1, dtype=torch.long), torch.zeros(1), inputs_embeds=hidden) is hidden
        return 0

    monkeypatch.setattr(GPUModelRunner, "capture_model", capture)
    assert runner.capture_model(profile_only=profile_only) == 0


@pytest.mark.parametrize("v2", [True, False])
def test_decode_rows_are_consumed_one_per_step_in_arrival_order(v2):
    model = _talker()
    model.vllm_config.model_config.use_v2_model_runner = v2
    # Two rows were already pending (cached at prefill); two more arrive at once.
    payload: dict = {"embed": {"cached_decode": _rows(1, 2), "decode": _rows(3, 4)}, "meta": {"decode_span": not v2}}
    payload.update(_omni_prompt_len=10, _omni_num_computed_tokens=10)
    steps = [_step(model, payload)[0].item()]
    for arrival in (None, _rows(5), None, None, None, None):
        payload["embed"]["decode"] = arrival
        steps.append(_step(model, payload)[0].item())
    assert steps == [1, 2, 3, 4, 5, -1, -2]
    assert payload["embed"]["cached_decode"] is None


@pytest.mark.parametrize(("v2", "resumable", "expected"), [(True, True, 3), (False, False, 2), (False, True, 3)])
def test_v1_and_resumable_requests_keep_the_indexed_cache_path(v2, resumable, expected):
    model = _talker()
    model.vllm_config.model_config.use_v2_model_runner = v2
    payload: dict = {"embed": {"cached_decode": _rows(1, 2, 3), "decode": None}, "meta": {"resumable": resumable}}
    payload["meta"]["num_processed_tokens"] = 2
    # Deliberately disagree: resumable requests must retain the model counter,
    # while ordinary V1 requests use the runner's generation offset.
    payload.update(_omni_prompt_len=10, _omni_num_computed_tokens=11)
    assert _step(model, payload)[0].item() == expected


_PROMPT_LEN = 28


def _prefilled_v1_payload(model, rows_during_prefill: list[torch.Tensor]) -> dict:
    """Request state after a Talker prefill, with Thinker rows that arrived during its chunks."""
    payload: dict = {"embed": {}, "meta": {"num_processed_tokens": _PROMPT_LEN, "prefill_consumed_text_tokens": 1}}
    for row in rows_during_prefill:
        update: dict = {}
        model._talker_cache_thinker_decode_embeds({**payload["embed"], "decode": row}, update)
        payload["embed"].update(update["embed"])
    return payload


def _v1_decode_text_steps(model, payload: dict, arrivals: list[torch.Tensor | None]) -> list[float]:
    """Run one V1 decode position per arrival through talker_preprocess, as the runner does."""
    steps = []
    for frame, arrival in enumerate(arrivals):
        payload["embed"]["decode"] = arrival
        payload.update(
            _omni_is_prefill=False, _omni_prompt_len=_PROMPT_LEN, _omni_num_computed_tokens=_PROMPT_LEN + frame
        )
        _, _, update = model.talker_preprocess(torch.zeros(1, dtype=torch.long), torch.zeros((1, _H)), **payload)
        steps.append(update.pop("mtp_inputs")[1].reshape(-1)[0].item())
        for key, value in update.items():
            payload.setdefault(key, {}).update(value)
    return steps


@pytest.mark.parametrize("rows_during_prefill", [0, 1, 2])
def test_v1_decode_speaks_thinker_rows_that_arrived_during_chunked_prefill(rows_during_prefill):
    model = _talker()
    model.vllm_config.model_config.use_v2_model_runner = False
    model.talker_config = Qwen3OmniMoeTalkerConfig(text_config={"hidden_size": _H})
    # Thinker text tokens 1..5; token 0 is the last row of the Talker prompt.
    rows = [_rows(value) for value in range(1, 6)]
    payload = _prefilled_v1_payload(model, rows[:rows_during_prefill])
    arrivals = rows[rows_during_prefill:] + [None] * (rows_during_prefill + 2)
    assert _v1_decode_text_steps(model, payload, arrivals) == [1, 2, 3, 4, 5, -1, -2]


@pytest.mark.parametrize("ordinary_steps", [0, 1, 3])
@pytest.mark.parametrize("rows_during_prefill", [0, 2])
def test_v1_first_speculative_chunk_does_not_replay_consumed_prefill_rows(ordinary_steps, rows_during_prefill):
    """Pass sender chunks to a Talker that cached rows during chunked prefill."""
    model = _talker()
    model.vllm_config.model_config.use_v2_model_runner = False
    payload = _prefilled_v1_payload(model, [_rows(11 + index) for index in range(rows_during_prefill)])
    manager = SimpleNamespace(
        put_req_chunk={"r": 1},
        request_payload={},
        config=SimpleNamespace(hf_config=SimpleNamespace(talker_config=SimpleNamespace(accept_hidden_layer=24))),
    )
    spoken = []
    for frame in range(ordinary_steps + rows_during_prefill + 5):
        if frame <= ordinary_steps + 1:
            first = 11 + rows_during_prefill + frame + (frame > ordinary_steps)
            speculative = frame == ordinary_steps
            tokens = [first, first + 1] if speculative else [first]
            delta = [token + 1 for token in tokens]
            request = SimpleNamespace(
                external_req_id="r",
                prompt_token_ids=[1, 2, 3],
                output_token_ids=list(range(10, delta[-1] + 1)),
                additional_information=None,
                resumable=False,
            )
            mm: OmniPayload = {"hidden_states": {"layers": {0: _rows(*tokens), 24: _rows(*tokens)}}}
            if speculative:
                mm["meta"] = {"decode_span": True, "consumed_input_span": True, "committed_token_count": len(tokens)}
                mm["ids"] = {"output": tokens}
            chunk = thinker2talker_async_chunk(manager, mm, request, new_token_ids=delta)
            assert chunk is not None
            manager.put_req_chunk["r"] += 1
            # The runner merges each received chunk into request-local state.
            for key, value in to_dict(chunk).items():
                if isinstance(value, dict):
                    payload.setdefault(key, {}).update(value)
        payload.update(_omni_prompt_len=_PROMPT_LEN, _omni_num_computed_tokens=_PROMPT_LEN + frame)
        spoken.append(_step(model, payload)[0].item())
    assert spoken == list(range(11, 14 + ordinary_steps + rows_during_prefill)) + [-1, -2]


def test_v1_decode_rejects_negative_generation_offset():
    model = _talker()
    model.vllm_config.model_config.use_v2_model_runner = False
    model.talker_config = Qwen3OmniMoeTalkerConfig(text_config={"hidden_size": _H})
    payload = _prefilled_v1_payload(model, [_rows(1), _rows(2)])
    payload.update(_omni_is_prefill=False, _omni_prompt_len=_PROMPT_LEN, _omni_num_computed_tokens=_PROMPT_LEN - 1)

    with pytest.raises(ValueError, match="non-negative generation offset"):
        model.talker_preprocess(torch.zeros(1, dtype=torch.long), torch.zeros((1, _H)), **payload)


def test_make_omni_output_aligns_codes_to_request_spans():
    model = _talker()
    hidden = torch.zeros((8, _H))  # 5 real tokens + graph padding
    decode_codes = torch.tensor([[5, 6, 7]])
    buffers = [
        {"req_id": "_warmup_0_"},  # no preprocess, no codes
        {"req_id": "a", "codes": {"audio": torch.zeros((3, _Q), dtype=torch.long)}},
        {"req_id": "b", "codes": {"audio": decode_codes}},
    ]
    out = model.make_omni_output(
        hidden, model_intermediate_buffer=buffers, request_token_spans=[(0, 1), (1, 4), (4, 5)]
    )
    codes = out.multimodal_outputs["codes"]["audio"]
    assert codes.shape == (5, _Q)
    assert codes[4].tolist() == [5, 6, 7]
    assert out.text_hidden_states.shape[0] == 5


def test_make_omni_output_without_spans_keeps_v1_concat():
    model = _talker()
    buffers = [{"codes": {"audio": torch.tensor([[1, 2, 3]])}}, {"codes": {"audio": torch.tensor([[4, 5, 6]])}}]
    out = model.make_omni_output(torch.zeros((4, _H)), model_intermediate_buffer=buffers)
    assert out.multimodal_outputs["codes"]["audio"].tolist() == [[1, 2, 3], [4, 5, 6]]
    assert out.text_hidden_states.shape[0] == 2


def test_talker_mtp_feeds_precomputed_sampling_noise_to_one_decode_step():
    """Residual-codebook noise must enter the predictor as a tensor.

    MRv2 captures ``talker_mtp`` in one CUDA graph. Drawing Gumbel noise
    inside that capture replays the graph RNG. The runner fills a static
    uniform buffer each step and the predictor has to consume it.
    """
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    recorded: dict = {}

    class _Predictor:
        def code_predictor_forward(self, input_ids, inputs_embeds, *, last_talker_hidden=None, **kwargs):
            recorded["seq"] = int(input_ids.shape[-1])
            recorded["kwargs"] = kwargs
            hidden = inputs_embeds.shape[-1]
            codes = torch.zeros(input_ids.shape[0], _Q, input_ids.shape[-1], dtype=torch.long)
            summed = torch.zeros(input_ids.shape[0], input_ids.shape[-1], hidden)
            return codes, summed

    model.talker = _Predictor()
    model.talker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=_H))
    uniforms = torch.rand(2, _Q - 1, 8)
    model.talker_mtp(
        torch.zeros(2, 1, dtype=torch.long),
        torch.zeros(2, _H),
        torch.zeros(2, 1, _H),
        torch.zeros(2, _H),
        sample_uniforms=uniforms,
        generator=object(),
    )
    assert recorded["seq"] == 1
    assert recorded["kwargs"]["sample_uniforms"] is uniforms
    assert recorded["kwargs"]["generator"] is None
    assert recorded["kwargs"]["generators"] is None

    model.talker_mtp(
        torch.zeros(1, 3, dtype=torch.long),
        torch.zeros(3, _H),
        torch.zeros(1, 1, _H),
        torch.zeros(3, _H),
        sample_uniforms=uniforms[:1],
    )
    assert recorded["seq"] == 3
    assert recorded["kwargs"]["sample_uniforms"] is None


def test_eager_output_is_token_major_with_invalid_rows():
    model = _talker()
    model.eager_frames_active = True
    hidden = torch.zeros((6, _H))
    buffers = [{"req_id": "a"}, {"req_id": "b"}]
    out = model.make_omni_output(hidden, model_intermediate_buffer=buffers, request_token_spans=[(0, 3), (3, 4)])
    mm = out.multimodal_outputs
    assert mm["codes"]["audio"].shape == (4, _Q)
    assert mm["meta"]["codec_frame_valid"].tolist() == [0, 0, 0, 0]
    assert out.text_hidden_states is hidden


def test_v1_resumable_speculative_span_uses_pending_queue():
    model = _talker()
    model.vllm_config.model_config.use_v2_model_runner = False
    payload = {
        "embed": {"cached_decode": _rows(1, 2), "decode": _rows(3, 4)},
        "meta": {"resumable": True, "decode_span": True, "num_processed_tokens": 20},
    }
    assert [_step(model, payload)[0].item() for _ in range(6)] == [1, 2, 3, 4, -1, -2]
