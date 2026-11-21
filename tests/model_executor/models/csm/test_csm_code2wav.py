# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the CSM stage-1 Mimi decoder against transformers' MimiModel."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from transformers import CsmConfig, MimiModel

from vllm_omni.entrypoints.openai.tts_adapters.csm import CsmAdapter
from vllm_omni.model_executor.models.csm.csm_code2wav import CsmCode2Wav, codec_weight_name
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.stage_input_processors.csm import talker2code2wav_async_chunk
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]

NUM_CODEBOOKS = 32
SAMPLES_PER_FRAME = 1920


@pytest.fixture(scope="module")
def config() -> CsmConfig:
    return CsmConfig()


def _random_codec(config: CsmConfig, *, layer_scale: float | None = None) -> MimiModel:
    torch.manual_seed(0)
    codec = MimiModel(config.codec_config).eval()
    with torch.no_grad():
        # A fresh codebook decodes every code to zero; give it content.
        for name, buffer in codec.named_buffers():
            if name.endswith("embed_sum"):
                buffer.normal_()
        if layer_scale is not None:
            for name, parameter in codec.named_parameters():
                if name.endswith("layer_scale.scale"):
                    parameter.fill_(layer_scale)
    return codec


@pytest.fixture(scope="module")
def reference(config: CsmConfig) -> MimiModel:
    return _random_codec(config)


@pytest.fixture(scope="module")
def strong_reference(config: CsmConfig) -> MimiModel:
    """A codec whose transformer matters: fresh layer scales are 0.01, which hides attention errors."""
    return _random_codec(config, layer_scale=1.0)


def _checkpoint(reference: MimiModel, *, flat: bool) -> dict[str, torch.Tensor]:
    """A CSM checkpoint's tensors: the codec under ``codec_model.`` plus other stages' weights."""
    state = {}
    for name, tensor in reference.state_dict().items():
        state["codec_model." + (codec_weight_name(name, nested=False) if flat else name)] = tensor
    state["lm_head.weight"] = torch.zeros(1)
    state["backbone_model.norm.weight"] = torch.zeros(1)
    return state


def _build(config: CsmConfig) -> CsmCode2Wav:
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=config))
    return CsmCode2Wav(vllm_config=vllm_config).eval()


@pytest.fixture
def model(config: CsmConfig, reference: MimiModel) -> CsmCode2Wav:
    model = _build(config)
    model.load_weights(_checkpoint(reference, flat=True).items())
    return model


def _codes(frames: int, seed: int) -> torch.Tensor:
    """Frame-major codes ``[frames, num_codebooks]``."""
    return torch.randint(0, 2048, (frames, NUM_CODEBOOKS), generator=torch.Generator().manual_seed(seed))


def _flat(codes: torch.Tensor) -> torch.Tensor:
    return codes.T.reshape(-1)


def _reference_audio(reference: MimiModel, codes: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return reference.decode(codes.T[None]).audio_values[0, 0]


def _audios(output: OmniOutput) -> list[torch.Tensor]:
    return output.multimodal_outputs["model_outputs"]


def _assert_same_audio(actual: torch.Tensor, expected: torch.Tensor) -> None:
    # Incremental decoding sums the same terms in another order, so allow rounding noise only.
    assert actual.shape == expected.shape
    rms = expected.pow(2).mean().sqrt()
    error = actual - expected
    assert float(error.pow(2).mean().sqrt() / rms) < 1e-4
    assert float(error.abs().max() / rms) < 1e-3


def _decode(model: CsmCode2Wav, request_id: str, codes: torch.Tensor) -> torch.Tensor:
    output = model(input_ids=_flat(codes), request_ids=[request_id])
    return _audios(output)[0]


def test_weight_names_convert_between_mimi_layouts(reference: MimiModel) -> None:
    assert (
        codec_weight_name("encoder.layers.0.conv.weight", nested=True) == "encoder.encoder_seanet.layers.0.conv.weight"
    )
    assert codec_weight_name("decoder_transformer.layers.1.x", nested=True) == "decoder.decoder_transformer.layers.1.x"
    assert codec_weight_name("upsample.conv.weight", nested=True) == "decoder.upsample.conv.weight"
    assert codec_weight_name("downsample.conv.weight", nested=True) == "encoder.downsample.conv.weight"
    assert codec_weight_name("quantizer.x", nested=True) == "quantizer.x"
    # Names already in the target layout pass through.
    assert codec_weight_name("encoder.encoder_seanet.layers.0.conv.weight", nested=True).count("encoder_seanet") == 1
    assert codec_weight_name("decoder.layers.0.conv.weight", nested=False) == "decoder.layers.0.conv.weight"

    installed = list(reference.state_dict())
    flat = [codec_weight_name(name, nested=False) for name in installed]
    assert len(set(flat)) == len(installed)
    assert [codec_weight_name(name, nested=True) for name in flat] == installed


@pytest.mark.parametrize("flat", [True, False])
def test_checkpoint_loads_in_either_layout(config: CsmConfig, reference: MimiModel, flat: bool) -> None:
    model = _build(config)
    loaded = model.load_weights(_checkpoint(reference, flat=flat).items())

    expected = reference.state_dict()
    actual = model.codec_model.state_dict()
    assert set(actual) == set(expected)
    for name, tensor in expected.items():
        assert torch.equal(actual[name], tensor), name
    assert {f"codec_model.{name}" for name, _ in model.codec_model.named_parameters()} <= loaded
    assert all(name.startswith("codec_model.") for name in loaded)


def test_codec_runs_in_float32(model: CsmCode2Wav) -> None:
    assert {parameter.dtype for parameter in model.codec_model.parameters()} == {torch.float32}


def test_whole_sequence_matches_transformers(model: CsmCode2Wav, reference: MimiModel) -> None:
    codes = _codes(frames=6, seed=1)
    output = model(input_ids=_flat(codes), runtime_additional_information=[{}])

    (audio,) = _audios(output)
    assert audio.dtype == torch.float32 and audio.shape == (6 * SAMPLES_PER_FRAME,)
    _assert_same_audio(audio, _reference_audio(reference, codes))
    assert [int(rate) for rate in output.multimodal_outputs["sr"]] == [24000]
    assert output.text_hidden_states is None


@pytest.mark.parametrize(("frames", "chunk"), [(12, 1), (40, 7)])
def test_chunks_match_decoding_the_whole_sequence(
    model: CsmCode2Wav, reference: MimiModel, frames: int, chunk: int
) -> None:
    codes = _codes(frames, seed=2)
    pieces = [_decode(model, "a", codes[start : start + chunk]) for start in range(0, frames, chunk)]

    assert all(piece.shape[0] % SAMPLES_PER_FRAME == 0 for piece in pieces)
    _assert_same_audio(torch.cat(pieces), _reference_audio(reference, codes))


# The decoder transformer attends to 250 positions, which is 125 frames; these run well past it.
@pytest.mark.parametrize(("frames", "chunk"), [(300, 1), (300, 25), (600, 1), (600, 25), (300, 300), (300, 130)])
def test_chunks_match_past_the_attention_window(
    config: CsmConfig, strong_reference: MimiModel, frames: int, chunk: int
) -> None:
    model = _build(config)
    model.load_weights(_checkpoint(strong_reference, flat=True).items())
    codes = _codes(frames, seed=15)
    pieces = [_decode(model, "a", codes[start : start + chunk]) for start in range(0, frames, chunk)]

    _assert_same_audio(torch.cat(pieces), _reference_audio(strong_reference, codes))


def test_chunks_built_by_the_stage_input_processor_decode_to_the_whole_sequence(
    model: CsmCode2Wav, reference: MimiModel
) -> None:
    manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 5, "initial_codec_chunk_frames": 1}}),
    )
    codes = _codes(frames=23, seed=11)
    payloads = []
    for index, frame in enumerate(codes):
        last = index == len(codes) - 1
        request = SimpleNamespace(external_req_id="a", is_finished=lambda last=last: last)
        step = {"codes": {"audio": frame[None]}, "meta": {"codec_frame_valid": torch.ones(1, dtype=torch.int8)}}
        payloads.append(talker2code2wav_async_chunk(manager, step, request))
    chunks = [payload.codes.audio for payload in payloads if payload is not None]

    assert [chunk.numel() // NUM_CODEBOOKS for chunk in chunks] == [1, 5, 5, 5, 5, 2]
    pieces = [_audios(model(input_ids=chunk, request_ids=["a"]))[0] for chunk in chunks]
    _assert_same_audio(torch.cat(pieces), _reference_audio(reference, codes))


def test_requests_in_one_batch_keep_separate_state(model: CsmCode2Wav, reference: MimiModel) -> None:
    first, second = _codes(frames=9, seed=3), _codes(frames=6, seed=4)
    audio: dict[str, list[torch.Tensor]] = {"a": [], "b": []}
    # Rows change order between steps, and "b" joins late.
    steps = [[("a", first[:3])], [("b", second[:2]), ("a", first[3:5])], [("a", first[5:]), ("b", second[2:])]]
    for step in steps:
        segments = [_flat(codes) for _, codes in step]
        output = model(
            input_ids=torch.cat(segments),
            runtime_additional_information=[{} for _ in step],
            seq_token_counts=[segment.numel() for segment in segments],
            request_ids=[request_id for request_id, _ in step],
        )
        for (request_id, _), piece in zip(step, _audios(output)):
            audio[request_id].append(piece)

    _assert_same_audio(torch.cat(audio["a"]), _reference_audio(reference, first))
    _assert_same_audio(torch.cat(audio["b"]), _reference_audio(reference, second))


def test_finished_requests_release_their_state(model: CsmCode2Wav, reference: MimiModel) -> None:
    _decode(model, "a", _codes(frames=4, seed=5))
    assert set(model._streams) == {"a"}
    model.on_requests_finished({"a"})
    assert model._streams == {}

    # The same id starts from a fresh state.
    codes = _codes(frames=3, seed=6)
    _assert_same_audio(_decode(model, "a", codes), _reference_audio(reference, codes))


def test_requests_without_an_id_keep_no_state(model: CsmCode2Wav, reference: MimiModel) -> None:
    codes = _codes(frames=3, seed=7)
    for _ in range(2):
        output = model(input_ids=_flat(codes), runtime_additional_information=[{}])
        _assert_same_audio(_audios(output)[0], _reference_audio(reference, codes))
    assert model._streams == {}


def test_invalid_codes_fail_only_their_request(model: CsmCode2Wav, reference: MimiModel) -> None:
    good, bad = _codes(frames=4, seed=9), _codes(frames=4, seed=10)
    bad[1, 5] = 2048

    def step(good_codes: torch.Tensor, bad_codes: torch.Tensor) -> list[torch.Tensor]:
        segments = [_flat(bad_codes), _flat(good_codes)]
        output = model(
            input_ids=torch.cat(segments),
            runtime_additional_information=[{}, {}],
            seq_token_counts=[segment.numel() for segment in segments],
            request_ids=["bad", "good"],
        )
        return _audios(output)

    first = step(good[:2], bad[:2])
    assert first[0].numel() == 0 and model._failed == {"bad"} and "bad" not in model._streams
    # Later chunks of the failed request stay silent even when their codes are valid.
    second = step(good[2:], _codes(frames=2, seed=11))
    assert second[0].numel() == 0
    _assert_same_audio(torch.cat((first[1], second[1])), _reference_audio(reference, good))

    model.on_requests_finished({"bad", "good"})
    assert model._failed == set() and model._streams == {}


def test_a_failed_request_reaches_the_serving_adapter_as_a_synthesis_error(model: CsmCode2Wav) -> None:
    """The flag is split per request the way the generation runner does it, then read by the adapter."""
    good, bad = _codes(frames=4, seed=9), _codes(frames=4, seed=10)
    bad[1, 5] = 2048
    adapter = object.__new__(CsmAdapter)

    def reported(output: OmniOutput, num_requests: int) -> list[bool]:
        flags = []
        for payload in OmniGenerationModelRunner._build_pooler_output(output, num_requests):
            collect: dict = {}
            adapter.collect_response_metadata(payload, collect)
            flags.append(collect.get("audio_synthesis_error", False))
        return flags

    def step(bad_codes: torch.Tensor, good_codes: torch.Tensor) -> OmniOutput:
        segments = [_flat(bad_codes), _flat(good_codes)]
        return model(
            input_ids=torch.cat(segments),
            seq_token_counts=[segment.numel() for segment in segments],
            request_ids=["bad", "good"],
        )

    first = step(bad[:2], good[:2])
    assert [int(flag) for flag in first.multimodal_outputs["meta"]["error"]] == [1, 0]
    assert reported(first, 2) == [True, False]
    # The request stays marked on its later chunks, whose own codes are valid.
    assert reported(step(_codes(frames=2, seed=11), good[2:]), 2) == [True, False]
    # A request alone in its batch, and a step without input.
    assert reported(model(input_ids=_flat(bad), request_ids=["alone"]), 1) == [True]
    assert reported(model(input_ids=torch.empty(0, dtype=torch.long)), 1) == [False]


@pytest.mark.parametrize("value", [-1, 2048, 2050])
def test_codes_outside_the_codec_vocabulary_are_rejected(model: CsmCode2Wav, value: int) -> None:
    codes = _codes(frames=2, seed=12)
    codes[0, 0] = value
    with pytest.raises(ValueError, match="out of range"):
        model._validate_codes(_flat(codes))
    model._validate_codes(_flat(_codes(frames=2, seed=12).clamp(max=2047)))
    with pytest.raises(ValueError, match="not a multiple"):
        model._validate_codes(torch.zeros(5, dtype=torch.long))


def test_bad_rows_and_profiling_rows_are_skipped(model: CsmCode2Wav, reference: MimiModel) -> None:
    codes = _codes(frames=2, seed=13)
    malformed = torch.zeros(5, dtype=torch.long)
    profile = torch.zeros(7, dtype=torch.long)
    segments = [_flat(codes), malformed, profile]
    infos = [{}, {}, *model.get_dummy_runtime_additional_information(1)]

    output = model(
        input_ids=torch.cat(segments),
        runtime_additional_information=infos,
        seq_token_counts=[segment.numel() for segment in segments],
    )

    audios = _audios(output)
    assert len(audios) == 3 and len(output.multimodal_outputs["sr"]) == 3
    _assert_same_audio(audios[0], _reference_audio(reference, codes))
    assert audios[1].numel() == 0 and audios[2].numel() == 0


def test_connector_payload_takes_precedence_over_token_ids(model: CsmCode2Wav, reference: MimiModel) -> None:
    codes = _codes(frames=3, seed=14)
    placeholder = torch.zeros(3, dtype=torch.long)
    output = model(input_ids=placeholder, model_intermediate_buffer=[{"codes": {"audio": _flat(codes)}}])
    _assert_same_audio(_audios(output)[0], _reference_audio(reference, codes))


def test_non_sampling_stage_contract(model: CsmCode2Wav) -> None:
    empty = model(input_ids=torch.zeros(0, dtype=torch.long))
    assert _audios(empty)[0].numel() == 0
    assert model.compute_logits(torch.zeros(1, 1)) is None
    assert model.embed_input_ids(torch.zeros(4, dtype=torch.long)).shape == (4, 1)
    assert model.make_omni_output(empty) is empty
    with pytest.raises(TypeError, match="returns OmniOutput"):
        model.make_omni_output(torch.zeros(1))
    assert model.requires_exact_input_shape and model.requires_request_ids and model.have_multimodal_outputs
    assert model.allow_patterns_overrides == ["transformers-*.safetensors"]
