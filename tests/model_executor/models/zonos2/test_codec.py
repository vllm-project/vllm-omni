# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P4 CPU boundary oracles; optional dependencies and network are not needed."""

from __future__ import annotations

import builtins
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.zonos2.zonos2_codec import DACStreamDecoder, LocalDAC, eos_boundary, shear
from vllm_omni.model_executor.models.zonos2.zonos2_speaker import Zonos2SpeakerEncoder
from vllm_omni.model_executor.stage_input_processors.zonos2 import talker2dac, talker2dac_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def delayed(n):
    aligned = torch.arange(n * 9).reshape(n, 9) % 1024
    raw = torch.full((n + 8, 9), 1025, dtype=torch.long)
    for j in range(9):
        raw[j : n + j, j] = aligned[:, j]
    return aligned, raw


def fake_decode(codes):
    return codes[0].float().repeat_interleave(512)


@pytest.mark.parametrize("n", [0, 1, 8, 15, 16, 17, 32, 33])
def test_shear_same_shape_and_alignment(n):
    aligned, raw = delayed(n)
    assert torch.equal(shear(raw, up=True)[:n], aligned)
    x = torch.cat((aligned, torch.full((8, 9), 1025, dtype=torch.long)))
    assert torch.equal(shear(x, up=False), raw)
    assert shear(aligned, up=True).shape == aligned.shape


@pytest.mark.parametrize("n", [0, 1, 15, 16, 17, 32, 33, 64, 65])
def test_stream_lengths_tail_flush_and_final_cleanup(n):
    aligned, raw = delayed(n)
    stream = DACStreamDecoder(fake_decode)
    output = []
    for end in range(24, len(raw), 16):
        output.append(stream.push("r", raw[:end], final=False, target=end - 8, sequence=end))
    output.append(stream.push("r", raw, final=True, target=n, sequence=len(raw)))
    wav = torch.cat(output)
    assert len(wav) == n * 512
    assert wav.dtype == torch.float32
    torch.testing.assert_close(wav, aligned[:, 0].float().repeat_interleave(512))
    assert not stream.states
    assert not stream.push("r", raw, final=True, target=n, sequence=len(raw)).numel()
    stream.cleanup(["r"])
    assert not stream.closed


@pytest.mark.parametrize("n", [0, 1, 15, 16, 17])
def test_min_chunk_threshold(n):
    _, raw = delayed(n)
    calls = []

    def decode(codes):
        calls.append(codes)
        return fake_decode(codes)

    stream = DACStreamDecoder(decode)
    wav = stream.push("r", raw, final=False, target=n, sequence=0)
    assert len(calls) == int(n >= 16)
    assert wav.numel() == (n - 4) * 512 if n >= 16 else wav.numel() == 0
    tail = stream.push("r", raw, final=True, target=n, sequence=1)
    assert len(wav) + len(tail) == n * 512


def test_raised_cosine_crossfade():
    outputs = iter((torch.ones(16 * 512), torch.zeros(20 * 512)))
    stream = DACStreamDecoder(lambda codes: next(outputs))
    _, raw = delayed(32)
    a = stream.push("r", raw[:24], final=False, target=16, sequence=0)
    b = stream.push("r", raw, final=True, target=32, sequence=1)
    assert len(a) == 12 * 512 and len(b) == 20 * 512
    expected = (0.5 * (1 + torch.cos(torch.linspace(0, torch.pi, 2048, dtype=torch.float64)))).float()
    torch.testing.assert_close(b[:2048], expected)
    assert b[0] == 1 and b[2047] == 0 and not b[2048:].any()


def test_duplicate_cancellation_reuse_and_error_cleanup():
    _, raw = delayed(32)
    stream = DACStreamDecoder(fake_decode)
    stream.push("a", raw[:24], final=False, target=16, sequence=0)
    stream.push("b", raw[:24], final=False, target=16, sequence=0)
    assert not stream.push("a", raw[:24], final=False, target=16, sequence=0).numel()
    stream.cleanup(["a"])
    assert set(stream.states) == {"b"}
    assert len(stream.push("a", raw, final=True, target=32, sequence=0)) == 32 * 512
    with pytest.raises(ValueError, match="regressed"):
        stream.push("b", raw[:24], final=True, target=0, sequence=1)
    assert not stream.states
    stream.cleanup(["a"])
    assert not stream.closed


@pytest.mark.parametrize("column", range(9))
def test_short_eos_alignment(column):
    raw = torch.zeros((12, 9), dtype=torch.long)
    raw[column + 1, column] = 1024
    assert eos_boundary(raw) == 1
    stream = DACStreamDecoder(fake_decode)
    wav = stream.push("r", raw, final=True, target=eos_boundary(raw), sequence=0)
    assert len(wav) == 512


def test_multi_codebook_eos_uses_highest_column():
    raw = torch.zeros((20, 9), dtype=torch.long)
    raw[10, [0, 3, 8]] = 1024
    assert eos_boundary(raw) == 2


def test_stage_async_terminal_frame_once_and_buffer_cleanup():
    manager = SimpleNamespace(code_prompt_token_ids=defaultdict(list), put_req_chunk=defaultdict(int))
    request = SimpleNamespace(external_req_id="r", is_finished=lambda: False)
    _, raw = delayed(32)
    chunks = []
    for i, frame in enumerate(raw):
        mm = {"codes": {"audio": frame[None]}, "meta": {"eos_frame": torch.tensor([-1])}}
        payload = talker2dac_async_chunk(manager, mm, request, new_token_ids=(0,))
        if payload:
            chunks.append(payload)
            manager.put_req_chunk["r"] += 1
    assert [p.meta.num_processed_tokens for p in chunks] == [16, 32]
    final = talker2dac_async_chunk(manager, mm, request, is_finished=True, new_token_ids=())
    assert final.codes.audio.shape == (9, len(raw))
    assert final.meta.finished
    assert "r" not in manager.code_prompt_token_ids


def test_sync_handoff_keeps_raw_frames_without_arbitrary_edge_trim():
    _, raw = delayed(17)
    source = SimpleNamespace(
        finished=True,
        outputs=[
            SimpleNamespace(
                multimodal_output={
                    "codes": {"audio": raw},
                    "meta": {"eos_frame": torch.tensor([17])},
                }
            )
        ],
    )
    prompt = talker2dac([source])[0]
    assert prompt["prompt_token_ids"] == [0]
    info = prompt["additional_information"]
    assert info["zonos2_target"] == 17
    assert torch.equal(info["codes"]["audio"], raw.T)


def test_imports_and_constructors_are_offline_and_lazy(monkeypatch):
    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"dac", "torchaudio", "transformers", "nemo_text_processing"}:
            raise AssertionError(f"Unexpected eager dependency: {name}")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    assert LocalDAC().codec is None
    assert Zonos2SpeakerEncoder()._model is None
    from vllm_omni.model_executor.models.zonos2.zonos2_textnorm import Zonos2TextNormalizer

    assert not Zonos2TextNormalizer()._normalizers


def test_missing_dac_checkpoint_is_explicit_no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_ZONOS2_DAC_PATH", str(tmp_path / "missing.pth"))
    with pytest.raises(FileNotFoundError, match="no automatic download"):
        LocalDAC().load()


def test_missing_dac_package(monkeypatch, tmp_path):
    path = tmp_path / "codec.pth"
    path.touch()
    monkeypatch.setenv("VLLM_ZONOS2_DAC_PATH", str(path))
    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "dac":
            raise ImportError("missing dac")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(ImportError, match="descript-audio-codec"):
        LocalDAC().load()


def test_corrupt_dac_checkpoint_and_dtype_restored(monkeypatch, tmp_path):
    path = tmp_path / "codec.pth"
    path.write_bytes(b"bad")
    monkeypatch.setenv("VLLM_ZONOS2_DAC_PATH", str(path))

    def fail(*args, **kwargs):
        assert kwargs["weights_only"] is True
        raise ValueError("bad")

    monkeypatch.setattr(torch, "load", fail)
    old = torch.get_default_dtype()
    with pytest.raises(RuntimeError, match="Cannot load"):
        LocalDAC().load()
    assert torch.get_default_dtype() == old


def test_speaker_missing_torchaudio(monkeypatch):
    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "torchaudio":
            raise ImportError("missing")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(ImportError, match="torchaudio"):
        Zonos2SpeakerEncoder().encode([0.0] * 1000, 24000)


def test_speaker_short_audio_error():
    with pytest.raises(ValueError, match="384"):
        Zonos2SpeakerEncoder().encode([0.0] * 10, 24000)


def test_speaker_uncached_weights_local_only(monkeypatch):
    import sys

    def fail(path, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is True
        raise OSError("Not cached")

    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=fail)))
    monkeypatch.delenv("VLLM_ZONOS2_SPEAKER_PATH", raising=False)
    with pytest.raises(RuntimeError, match="no automatic download"):
        Zonos2SpeakerEncoder()._load()


def test_speaker_missing_explicit_path(monkeypatch, tmp_path):
    import sys

    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=None))
    monkeypatch.setenv("VLLM_ZONOS2_SPEAKER_PATH", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Zonos2SpeakerEncoder()._load()


def test_nemo_missing_dependency_error(monkeypatch):
    from vllm_omni.model_executor.models.zonos2.zonos2_textnorm import Zonos2TextNormalizer

    real = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.startswith("nemo_text_processing"):
            raise ImportError("missing NeMo")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    with pytest.raises(ImportError, match="nemo_text_processing==1.2.0"):
        Zonos2TextNormalizer()._build("en")


def test_decoder_uses_transport_final_and_target_not_runtime_generated_len():
    from vllm_omni.model_executor.models.zonos2.zonos2_dac_decoder import (
        Zonos2Code2WavForConditionalGeneration,
    )

    cfg = SimpleNamespace(model_config=SimpleNamespace(model="unused"), device_config=SimpleNamespace(device="cpu"))
    decoder = Zonos2Code2WavForConditionalGeneration(vllm_config=cfg)
    decoder._stream = DACStreamDecoder(fake_decode)
    _, raw = delayed(16)
    info = {
        "codes": {"audio": raw.T},
        "generated_len": 0,
        "meta": {"last_chunk": False, "num_processed_tokens": 16, "chunk_seq": 0},
    }

    def run():
        return decoder.forward(
            torch.tensor([0]), runtime_additional_information=[info], request_ids=["r"], seq_token_counts=[1]
        ).multimodal_outputs

    first = run()
    assert len(first["model_outputs"][0]) == 12 * 512
    info["meta"] = {"last_chunk": True, "num_processed_tokens": 16, "chunk_seq": 1}
    last = run()
    assert len(last["model_outputs"][0]) == 4 * 512
    assert int(last["sr"][0]) == 44100
    decoder.on_requests_finished(["r"])
    assert not decoder._stream.states and not decoder._stream.closed


def test_decode_exception_drops_state():
    def fail(codes):
        raise RuntimeError("codec error")

    stream = DACStreamDecoder(fail)
    _, raw = delayed(16)
    with pytest.raises(RuntimeError, match="codec error"):
        stream.push("r", raw, final=False, target=16, sequence=0)
    assert not stream.states and not stream.closed


@pytest.mark.parametrize("metadata", [[], {"kwargs": []}])
def test_invalid_checkpoint_metadata_restores_default_dtype(monkeypatch, tmp_path, metadata):
    path = tmp_path / "invalid-metadata.pth"
    torch.save({"metadata": metadata, "state_dict": {}}, path)
    monkeypatch.setenv("VLLM_ZONOS2_DAC_PATH", str(path))
    old = torch.get_default_dtype()
    with pytest.raises(RuntimeError, match="Cannot load ZONOS2 DAC checkpoint"):
        LocalDAC().load()
    assert torch.get_default_dtype() == old
