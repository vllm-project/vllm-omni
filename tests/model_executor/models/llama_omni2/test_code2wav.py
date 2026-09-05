# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.llama_omni2.llama_omni2_code2wav import (
    LlamaOmni2Code2Wav,
    LlamaOmni2Code2WavCore,
    _load_cosy2_modules,
    load_default_speaker_embedding,
    validate_cosy2_decoder_dir,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeFlow(nn.Module):
    token_mel_ratio = 1
    pre_lookahead_len = 1

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def inference(self, *, token, finalize, **kwargs):
        self.last_embedding = kwargs["embedding"].detach().clone()
        self.batch_sizes.append(int(token.shape[0]))
        del kwargs, finalize
        mel = token.to(torch.float32).unsqueeze(1)
        return mel, None


class _LookaheadFlow(_FakeFlow):
    pre_lookahead_len = 3

    def __init__(self):
        super().__init__()
        self.calls = []

    def inference(self, *, token, finalize, **kwargs):
        self.calls.append((token.detach().clone(), finalize))
        if not finalize and token.shape[1] <= self.pre_lookahead_len:
            raise AssertionError("non-final input lacks the lookahead window")
        return super().inference(token=token, finalize=finalize, **kwargs)


class _FailingFlow(_FakeFlow):
    def __init__(self):
        super().__init__()
        self.fail_token_length = None

    def inference(self, *, token, finalize, **kwargs):
        if self.fail_token_length == token.shape[1]:
            raise RuntimeError("injected flow failure")
        return super().inference(
            token=token,
            finalize=finalize,
            **kwargs,
        )


class _ForwardOnlyFlow(nn.Module):
    token_mel_ratio = 1
    pre_lookahead_len = 1

    def forward(
        self,
        *,
        token,
        token_len,
        prompt_feat,
        prompt_feat_len,
        embedding,
        streaming,
        finalize,
    ):
        del token_len, prompt_feat, prompt_feat_len, embedding
        self.last_streaming = streaming
        self.last_finalize = finalize
        return token.to(torch.float32).unsqueeze(1), None


class _FakeHift(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def inference(self, *, speech_feat, cache_source):
        del cache_source
        self.batch_sizes.append(int(speech_feat.shape[0]))
        speech = speech_feat.flatten(1)
        source = speech.unsqueeze(1)
        return speech, source


class _LoadableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded_state = None
        self.loaded_strict = None
        self.target_device = None
        self.eval_called = False

    def load_state_dict(self, state_dict, strict=True):
        self.loaded_state = state_dict
        self.loaded_strict = strict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def to(self, device):
        self.target_device = torch.device(device)
        return self

    def eval(self):
        self.eval_called = True
        return self


class _LoadableIncrementalFlow(_LoadableModule):
    def __init__(self):
        super().__init__()
        self.decoder = SimpleNamespace(fp16=True)

    def setup_cache(self, token, mel, spk, n_timesteps=10):
        del token, mel, spk, n_timesteps

    def inference_chunk(
        self,
        token,
        spk,
        cache,
        last_chunk=False,
        n_timesteps=10,
    ):
        del token, spk, cache, last_chunk, n_timesteps


class _LoadableLegacyFlow(_LoadableModule):
    def __init__(self):
        super().__init__()
        self.decoder = SimpleNamespace(fp16=True)


class _LoadableHift(_LoadableModule):
    def forward(self, speech_feat, cache_source):
        return speech_feat + 1, cache_source + 2


def test_validate_decoder_dir_names_missing_artifact(tmp_path):
    for name in ("cosyvoice.yaml", "flow.pt"):
        (tmp_path / name).write_bytes(b"fixture")

    with pytest.raises(FileNotFoundError, match="hift.pt"):
        validate_cosy2_decoder_dir(tmp_path)


def test_decoder_loader_constructs_incremental_flow_from_yaml(
    tmp_path,
    monkeypatch,
):
    for name in ("flow.yaml", "flow.pt", "hift.pt"):
        (tmp_path / name).write_bytes(b"fixture")

    fake_flow = _LoadableIncrementalFlow()
    fake_hift = _LoadableHift()
    hyperpyyaml = ModuleType("hyperpyyaml")
    hyperpyyaml.load_hyperpyyaml = lambda handle: {"flow": fake_flow}
    flashcosyvoice = ModuleType("flashcosyvoice")
    modules = ModuleType("flashcosyvoice.modules")
    hifigan_module = ModuleType("flashcosyvoice.modules.hifigan")
    hifigan_module.HiFTGenerator = lambda: fake_hift
    monkeypatch.setitem(sys.modules, "hyperpyyaml", hyperpyyaml)
    monkeypatch.setitem(sys.modules, "flashcosyvoice", flashcosyvoice)
    monkeypatch.setitem(sys.modules, "flashcosyvoice.modules", modules)
    monkeypatch.setitem(
        sys.modules,
        "flashcosyvoice.modules.hifigan",
        hifigan_module,
    )

    def fake_load(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        if Path(path).name == "flow.pt":
            return {"flow.weight": torch.ones(1)}
        return {"generator.hift.weight": torch.ones(1)}

    monkeypatch.setattr(torch, "load", fake_load)

    flow, hift = _load_cosy2_modules(tmp_path, torch.device("cpu"))

    assert flow is fake_flow
    assert callable(flow.setup_cache)
    assert callable(flow.inference_chunk)
    assert "inference" not in flow.__dict__
    assert hift is fake_hift
    assert flow.loaded_state == {"flow.weight": torch.ones(1)}
    assert hift.loaded_state == {"hift.weight": torch.ones(1)}
    assert flow.loaded_strict is True
    assert hift.loaded_strict is True
    assert flow.decoder.fp16 is False
    assert flow.target_device == torch.device("cpu")
    assert hift.target_device == torch.device("cpu")
    assert flow.eval_called
    assert hift.eval_called
    speech, source = hift.inference(
        speech_feat=torch.ones(1, 80, 2),
        cache_source=torch.ones(1, 1, 2),
    )
    assert torch.equal(speech, torch.full((1, 80, 2), 2.0))
    assert torch.equal(source, torch.full((1, 1, 2), 3.0))


def test_decoder_loader_uses_strict_flash_flow_for_legacy_cosyvoice_yaml(
    tmp_path,
    monkeypatch,
):
    for name in ("cosyvoice.yaml", "flow.pt", "hift.pt"):
        (tmp_path / name).write_bytes(b"fixture")

    fake_flow = _LoadableLegacyFlow()
    fake_hift = _LoadableHift()
    hyperpyyaml = ModuleType("hyperpyyaml")

    def reject_full_config(_handle):
        raise AssertionError("legacy cosyvoice.yaml must not parse unrelated LLM config")

    hyperpyyaml.load_hyperpyyaml = reject_full_config
    flashcosyvoice = ModuleType("flashcosyvoice")
    modules = ModuleType("flashcosyvoice.modules")
    flow_module = ModuleType("flashcosyvoice.modules.flow")
    hifigan_module = ModuleType("flashcosyvoice.modules.hifigan")
    flow_module.CausalMaskedDiffWithXvec = lambda: fake_flow
    hifigan_module.HiFTGenerator = lambda: fake_hift
    monkeypatch.setitem(sys.modules, "hyperpyyaml", hyperpyyaml)
    monkeypatch.setitem(sys.modules, "flashcosyvoice", flashcosyvoice)
    monkeypatch.setitem(sys.modules, "flashcosyvoice.modules", modules)
    monkeypatch.setitem(sys.modules, "flashcosyvoice.modules.flow", flow_module)
    monkeypatch.setitem(
        sys.modules,
        "flashcosyvoice.modules.hifigan",
        hifigan_module,
    )

    def fake_load(path, **kwargs):
        assert kwargs == {"map_location": "cpu", "weights_only": True}
        if Path(path).name == "flow.pt":
            return {"flow.weight": torch.ones(1)}
        return {"generator.hift.weight": torch.ones(1)}

    monkeypatch.setattr(torch, "load", fake_load)

    flow, hift = _load_cosy2_modules(tmp_path, torch.device("cpu"))

    assert flow is fake_flow
    assert "inference" not in flow.__dict__
    assert hift is fake_hift
    assert flow.loaded_state == {"flow.weight": torch.ones(1)}
    assert flow.loaded_strict is True


def test_default_english_speaker_embedding_is_packaged_and_nonzero():
    embedding = load_default_speaker_embedding()

    assert embedding.shape == (1, 192)
    assert embedding.dtype == torch.float32
    assert torch.isfinite(embedding).all()
    assert torch.count_nonzero(embedding).item() > 0


def test_streaming_core_uses_default_english_speaker_embedding():
    flow = _FakeFlow()
    core = LlamaOmni2Code2WavCore(
        flow=flow,
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )

    core.process("request-default-speaker", [1, 2], finished=True)

    assert torch.equal(flow.last_embedding, load_default_speaker_embedding())


def test_streaming_core_calls_legacy_forward_without_monkey_patch():
    flow = _ForwardOnlyFlow()
    core = LlamaOmni2Code2WavCore(
        flow=flow,
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )

    chunk = core.process("request-forward-only", [1, 2], finished=True)

    assert chunk.audio.tolist() == [1.0, 2.0]
    assert flow.last_streaming is False
    assert flow.last_finalize is True


def test_streaming_core_buffers_short_nonfinal_lookahead_window_until_decodable():
    flow = _LookaheadFlow()
    core = LlamaOmni2Code2WavCore(
        flow=flow,
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )

    buffered = core.process("request-lookahead", [1], finished=False)
    emitted = core.process("request-lookahead", [2, 3, 4], finished=False)

    assert buffered is None
    assert len(flow.calls) == 1
    assert flow.calls[0][0].tolist() == [[1, 2, 3, 4]]
    assert flow.calls[0][1] is False
    assert emitted is not None
    assert emitted.sequence_index == 0
    assert emitted.consumed_units == 4


def test_streaming_core_finalizes_a_short_buffered_lookahead_window():
    flow = _LookaheadFlow()
    core = LlamaOmni2Code2WavCore(
        flow=flow,
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )

    assert core.process("request-final-lookahead", [1], finished=False) is None
    final = core.process("request-final-lookahead", [], finished=True)

    assert final is not None
    assert final.finished
    assert final.consumed_units == 1
    assert flow.calls[0][0].tolist() == [[1]]
    assert flow.calls[0][1] is True
    assert "request-final-lookahead" not in core


def test_streaming_core_emits_multiple_finite_24khz_chunks_and_flushes_once():
    flow = _FakeFlow()
    speaker_embedding = torch.arange(192, dtype=torch.float32).reshape(1, 192)
    core = LlamaOmni2Code2WavCore(
        flow=flow,
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
        speaker_embedding=speaker_embedding,
    )

    first = core.process("request-a", [1, 2, 3], finished=False)
    second = core.process("request-a", [4, 5], finished=True)

    assert first.sample_rate == 24000
    assert second.sample_rate == 24000
    assert first.sequence_index == 0
    assert second.sequence_index == 1
    assert not first.finished
    assert second.finished
    assert torch.isfinite(first.audio).all()
    assert torch.isfinite(second.audio).all()
    assert first.consumed_units == 3
    assert second.consumed_units == 5
    assert torch.equal(flow.last_embedding, speaker_embedding)
    assert "request-a" not in core

    with pytest.raises(ValueError, match="already finished"):
        core.process("request-a", [], finished=True)


def test_streaming_core_isolates_and_cancels_requests():
    core = LlamaOmni2Code2WavCore(
        flow=_FakeFlow(),
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )

    first_a = core.process("request-a", [1, 2], finished=False)
    first_b = core.process("request-b", [8, 9], finished=False)
    core.cancel("request-a")
    final_b = core.process("request-b", [10], finished=True)

    assert first_a.audio.tolist() != first_b.audio.tolist()
    assert "request-a" not in core
    assert final_b.consumed_units == 3
    assert final_b.finished


def test_model_forward_uses_request_scoped_runtime_payloads():
    core = LlamaOmni2Code2WavCore(
        flow=_FakeFlow(),
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/unused",
            get_hidden_size=lambda: 1,
        ),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )
    model = LlamaOmni2Code2Wav(vllm_config=config, core=core)

    output = model(
        input_ids=torch.tensor([0]),
        positions=torch.tensor([0]),
        runtime_additional_information=[
            {
                "codes": {"audio": torch.tensor([3, 4])},
                "meta": {
                    "request_id": "request-a",
                    "finished": torch.tensor(True),
                },
            }
        ],
    )

    assert len(output.multimodal_outputs["model_outputs"]) == 1
    assert output.multimodal_outputs["sr"][0].item() == 24000
    assert output.multimodal_outputs["finished"][0].item()
    assert output.multimodal_outputs["sequence_index"][0].item() == 0
    assert output.multimodal_outputs["codec_units"][0].tolist() == [3, 4]


def test_model_batches_equal_shape_requests_in_one_flow_and_hift_call():
    flow = _FakeFlow()
    hift = _FakeHift()
    core = LlamaOmni2Code2WavCore(
        flow=flow,
        hift=hift,
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/unused",
            get_hidden_size=lambda: 1,
        ),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )
    model = LlamaOmni2Code2Wav(vllm_config=config, core=core)

    output = model(
        input_ids=torch.tensor([0]),
        positions=torch.tensor([0]),
        runtime_additional_information=[
            {
                "codes": {"audio": torch.tensor([1, 2])},
                "meta": {
                    "request_id": "request-a",
                    "finished": torch.tensor(True),
                },
            },
            {
                "codes": {"audio": torch.tensor([8, 9])},
                "meta": {
                    "request_id": "request-b",
                    "finished": torch.tensor(True),
                },
            },
        ],
    )

    assert flow.batch_sizes == [2]
    assert hift.batch_sizes == [2]
    assert output.multimodal_outputs["model_outputs"][0].tolist() == [1.0, 2.0]
    assert output.multimodal_outputs["model_outputs"][1].tolist() == [8.0, 9.0]


def test_profiler_ranges_report_batched_flow_and_hift(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("VLLM_OMNI_LLAMA_OMNI2_PROFILE_BATCHES", "1")
    ranges = []

    @contextmanager
    def record_function(name):
        ranges.append(name)
        yield

    monkeypatch.setattr(torch.profiler, "record_function", record_function)
    core = LlamaOmni2Code2WavCore(
        flow=_FakeFlow(),
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )

    core.process_batch(
        [
            ("request-a", [1, 2], True),
            ("request-b", [8, 9], True),
        ]
    )

    assert ranges == [
        "llama_omni2.code2wav.flow[batch=2]",
        "llama_omni2.code2wav.hift[batch=2]",
        "llama_omni2.code2wav.d2h[batch=2]",
    ]


def test_core_mixed_final_work_uses_separate_exact_shape_buckets():
    flow = _FakeFlow()
    hift = _FakeHift()
    core = LlamaOmni2Code2WavCore(
        flow=flow,
        hift=hift,
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )

    chunks = core.process_batch(
        [
            ("request-a", [1, 2], False),
            ("request-b", [8, 9], True),
        ]
    )

    assert flow.batch_sizes == [1, 1]
    assert hift.batch_sizes == [1, 1]
    assert not chunks[0].finished
    assert chunks[1].finished
    assert "request-a" in core
    assert "request-b" not in core


def test_core_reordered_requests_keep_state_ownership():
    core = LlamaOmni2Code2WavCore(
        flow=_FakeFlow(),
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )

    core.process_batch(
        [
            ("request-a", [1, 2], False),
            ("request-b", [8, 9], False),
        ]
    )
    chunks = core.process_batch(
        [
            ("request-b", [10, 11], False),
            ("request-a", [3, 4], False),
        ]
    )

    assert [chunk.request_id for chunk in chunks] == ["request-b", "request-a"]
    assert core._states["request-a"].units == [1, 2, 3, 4]
    assert core._states["request-b"].units == [8, 9, 10, 11]
    assert core._states["request-a"].sequence_index == 2
    assert core._states["request-b"].sequence_index == 2


def test_core_failed_later_bucket_rolls_back_every_request():
    flow = _FailingFlow()
    core = LlamaOmni2Code2WavCore(
        flow=flow,
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )
    core.process_batch(
        [
            ("request-a", [1, 2], False),
            ("request-b", [8, 9], False),
        ]
    )
    before = {
        request_id: (
            list(state.units),
            state.token_offset,
            state.sequence_index,
            state.mel_cache.clone(),
            state.source_cache.clone(),
            state.speech_cache.clone(),
        )
        for request_id, state in core._states.items()
    }
    flow.fail_token_length = 5

    with pytest.raises(RuntimeError, match="injected flow failure"):
        core.process_batch(
            [
                ("request-a", [3, 4], False),
                ("request-b", [10, 11, 12], False),
            ]
        )

    for request_id, state in core._states.items():
        expected = before[request_id]
        assert state.units == expected[0]
        assert state.token_offset == expected[1]
        assert state.sequence_index == expected[2]
        assert torch.equal(state.mel_cache, expected[3])
        assert torch.equal(state.source_cache, expected[4])
        assert torch.equal(state.speech_cache, expected[5])


def test_core_rejects_duplicate_chunk_seq_without_state_mutation():
    flow = _FakeFlow()
    core = LlamaOmni2Code2WavCore(
        flow=flow,
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )
    core.process_batch(
        [("request-a", [1, 2], False)],
        chunk_seqs=[0],
    )
    before = core._clone_state(core._states["request-a"])

    with pytest.raises(ValueError, match="chunk_seq"):
        core.process_batch(
            [("request-a", [3, 4], False)],
            chunk_seqs=[0],
        )

    current = core._states["request-a"]
    assert current.units == before.units
    assert current.token_offset == before.token_offset
    assert current.sequence_index == before.sequence_index
    assert current.chunk_seq == before.chunk_seq
    assert flow.batch_sizes == [1]


def test_core_split_cache_tensors_do_not_alias_between_requests():
    core = LlamaOmni2Code2WavCore(
        flow=_FakeFlow(),
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )

    core.process_batch(
        [
            ("request-a", [1, 2], False),
            ("request-b", [8, 9], False),
        ]
    )

    state_a = core._states["request-a"]
    state_b = core._states["request-b"]
    for name in ("mel_cache", "source_cache", "speech_cache"):
        cache_a = getattr(state_a, name)
        cache_b = getattr(state_b, name)
        assert cache_a.untyped_storage().data_ptr() != cache_b.untyped_storage().data_ptr()


def test_model_forward_exposes_codec_delta_without_audio_snapshot_while_buffering():
    core = LlamaOmni2Code2WavCore(
        flow=_LookaheadFlow(),
        hift=_FakeHift(),
        device="cpu",
        mel_cache_len=1,
        source_cache_len=1,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            model="/unused",
            get_hidden_size=lambda: 1,
        ),
        device_config=SimpleNamespace(device=torch.device("cpu")),
    )
    model = LlamaOmni2Code2Wav(vllm_config=config, core=core)

    output = model(
        input_ids=torch.tensor([0]),
        positions=torch.tensor([0]),
        runtime_additional_information=[
            {
                "codes": {"audio": torch.tensor([3])},
                "meta": {
                    "request_id": "request-buffered",
                    "finished": torch.tensor(False),
                },
            }
        ],
    )

    assert output.multimodal_outputs["codec_units"][0].tolist() == [3]
    assert output.multimodal_outputs["model_outputs"][0] is None
    assert output.multimodal_outputs["sr"][0] is None
    assert output.multimodal_outputs["finished"][0] is None
    assert output.multimodal_outputs["sequence_index"][0] is None
    assert output.multimodal_outputs["consumed_units"][0] is None
