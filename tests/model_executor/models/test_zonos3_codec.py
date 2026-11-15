# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed Codec loading and native transport lifecycle against stateless oracles."""

import hashlib
import json
import math
from dataclasses import replace
from unittest.mock import MagicMock

import pytest
import torch
from torch.nn import functional as F
from transformers import PretrainedConfig
from vllm.config import LoadConfig, ModelConfig, SchedulerConfig, VllmConfig

from vllm_omni.data_entry_keys import to_dict
from vllm_omni.model_executor.models.zonos3 import codec as codec_checkpoint
from vllm_omni.model_executor.models.zonos3.codec import (
    CodecManager,
    ZACCodecChunk,
    ZACDecoderConfig,
    ZACDecoderNetwork,
    ZACMossLayer,
    ZACPatch,
    ZACTransformerSpec,
    Zonos3CodecFiles,
    Zonos3CodecModel,
)
from vllm_omni.model_executor.models.zonos3.model_state import Zonos3CodecModelState
from vllm_omni.model_executor.stage_input_processors.zonos3 import Zonos3ChunkConfig, Zonos3RVQStageInputProcessor
from vllm_omni.worker.scheduling_metadata_adapter import DefaultSchedulingMetadataAdapter
from vllm_omni.worker_v2.omni_generation_model_runner import OmniGenerationModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ---------------------------------------------------------------- codec_layers


def spec(context=7) -> ZACTransformerSpec:
    return {
        "d_model": 16,
        "num_heads": 2,
        "context_frames": context,
        "dim_feedforward": 32,
        "layer_scale": 0.5,
        "input_dimension": 16,
        "output_dimension": 16,
        "num_layers": 1,
    }


def whole_sequence(layer: ZACMossLayer, x: torch.Tensor) -> torch.Tensor:
    """Stateless reference: whole-prefix local attention, following MossLayer.attention.

    No incremental caches, slots or per-row positions enter this oracle.
    """
    b, t, d = x.shape
    q, k, v = layer.qkv(layer.norm1(x)).reshape(b, t, 3, layer.heads, layer.head_dim).unbind(2)
    frequency = torch.exp(torch.arange(layer.head_dim // 2) * (-math.log(10000.0) * 2 / layer.head_dim))
    phase = torch.arange(t)[:, None] * frequency[None, :]
    cos, sin = phase.cos()[None, :, None, :], phase.sin()[None, :, None, :]
    rotated = []
    for z in (q, k):
        real, imag = z.float().reshape(b, t, layer.heads, layer.head_dim // 2, 2).unbind(-1)
        rotated.append(torch.stack((real * cos - imag * sin, real * sin + imag * cos), -1).flatten(-2))
    q, k = [z.to(x.dtype).transpose(1, 2) for z in rotated]
    delta = torch.arange(t)[:, None] - torch.arange(t)[None, :]
    output = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), attn_mask=(delta >= 0) & (delta < layer.context))
    x = x + layer.attention_scale * layer.attention_out(output.transpose(1, 2).reshape(b, t, d))
    return x + layer.ffn_scale * layer.ffn(layer.norm2(x))


# ---------------------------------------------------------------- codec_network


def small_config() -> ZACDecoderConfig:
    first = {**spec(), "input_dimension": 8, "output_dimension": 16, "num_layers": 2}
    final = {**spec(), "input_dimension": 8, "output_dimension": 32}
    return ZACDecoderConfig(
        stages=(first, 2, final),
        stage_layers=((0, 2), None, (0,)),
        semantic_width=4,
        semantic_hidden=8,
        semantic_depth=2,
        codebook_dim=3,
        latent_width=8,
        head_depth=2,
        head_hidden=32,
        n_ffts=(8, 24),
        hop=2,
        fir_kernel=7,
        codebook_size=32,
        trainable_stages=(2,),
    )


def offline_conv(block, x):
    h = block.dwconv(F.pad(x, (6, 0))).transpose(1, 2)
    return x + block.pwconv2(F.gelu(block.pwconv1(block.norm(h)))).transpose(1, 2)


def whole_trunk(model: ZACDecoderNetwork, codes: torch.Tensor) -> list[torch.Tensor]:
    """Whole-utterance oracle with no streaming state, following ISTFTCodec.features."""
    x = model.quantizer.semantic_quantizer.quantizers[0](codes[:, :, 0])
    for block in model.semantic_head[:-1]:
        x = offline_conv(block, x)
    x = model.semantic_head[-1](x)
    for i, quantizer in enumerate(model.quantizer.quantizer.quantizers):
        code = codes[:, :, i + 1]
        x = x + torch.where(code[:, None] >= 0, quantizer(code.clamp(min=0)), 0.0)
    for stage in model.stages:
        if isinstance(stage, ZACPatch):
            x = stage(x)
        else:
            x = stage.input_proj(x.transpose(1, 2))
            for layer in stage.layers:
                x = whole_sequence(layer, x)
            x = stage.output_proj(x).transpose(1, 2)
    for block in model.head.blocks:
        x = offline_conv(block, x)
    x = model.head.norm(x.transpose(1, 2))
    return [linear(x) for linear in model.head.outs]


@torch.inference_mode()
def test_release_weight_mapping_pruning_and_head_override():
    torch.manual_seed(42)
    source = ZACDecoderNetwork(small_config()).eval()
    original = source.state_dict()
    generator, trainable = {}, {}
    for key, value in original.items():
        if key.startswith(("head.", "stages.2.")):
            trainable[key] = value
        if key.startswith("head."):
            continue
        target = key
        if key.startswith("stages."):
            parts = key.split(".")
            index = int(parts[1])
            if len(parts) > 3 and parts[2] == "layers":
                parts[3] = str(source.config.stage_layers[index][int(parts[3])])
            parts[0] = "decoder"
            target = ".".join(parts)
        generator[target] = value
    generator["decoder.0.layers.1.qkv.weight"] = torch.zeros((48, 16))  # pruned layer
    generator["encoder.0.unused"] = torch.zeros(1)
    generator["quantizer.quantizer.quantizers.0.in_proj.weight_g"] = torch.zeros(1)  # encode-only
    target = ZACDecoderNetwork(small_config())
    target.load_release_weights(generator, trainable)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="head checkpoint keys"):
        target.load_release_weights(generator, {})
    incomplete = {key: value for key, value in generator.items() if not key.startswith("semantic_head.")}
    with pytest.raises(ValueError, match="Incomplete ZAC generator"):
        target.load_release_weights(incomplete, trainable)


def test_release_config_selects_retained_stages_and_pruned_layer_indices():
    stages = []
    for i in range(9):
        if i % 2:
            stages.append({"type": "PatchedPretransform", "direction": "up", "patch_size": 2})
        else:
            stages.append(
                {
                    **spec(),
                    "type": "MossProjectedTransformer",
                    "num_layers": 32 if i == 0 else 12,
                    "frame_rate": 15 * 2 ** (i // 2),
                }
            )
    stages.append({"type": "PatchedPretransform", "direction": "up", "patch_size": 23})
    release = {
        "model": {
            "sample_rate": 44100,
            "internal_sample_rate": 44160,
            "frame_rate": 15,
            "internal_hop_length": 2944,
            "external_samples_per_frame": 2940,
            "residual_codebooks": 9,
            "semantic_codebook_size": 4096,
            "residual_codebook_size": 4096,
            "codebook_dim": 8,
            "decoder_stages": stages,
            "dual_encoding": {"width": 256, "hidden": 512, "depth": 3},
        },
        "head": {
            "type": "mr_istft",
            "n_ffts": [736, 2208],
            "depth": 4,
            "hidden": 1536,
            "stage_layers": {"0": [0, 31], "2": [0, 11]},
            "trainable_stages": list(range(9)),
        },
    }
    config = ZACDecoderConfig.from_release(release)
    assert config.hop == 184 and config.n_ffts == (736, 2208)
    assert config.stage_layers[0] == (0, 31) and config.stage_layers[2] == (0, 11)
    assert config.stages[0]["num_layers"] == 2
    assert config.stages[1] == 2 and len(config.stages) == 9
    release["head"]["trainable_stages"] = [0, 8]
    with pytest.raises(ValueError, match="contiguous suffix"):
        ZACDecoderConfig.from_release(release)
    release["head"]["trainable_stages"] = []
    release["head"]["stage_layers"]["0"] = [31, 0]
    with pytest.raises(ValueError, match="retained transformer layers"):
        ZACDecoderConfig.from_release(release)


# ---------------------------------------------------------------- codec_synthesis


# ---------------------------------------------------------------- codec_reference


# ---------------------------------------------------------------- codec_manager


def whole_audio(model, codes):
    branches = []
    for i, y in enumerate(whole_trunk(model, codes[None])):
        log_mag, phase = y[0].float().chunk(2, -1)
        mag = log_mag.clamp(max=model.head.max_log_magnitude).exp()
        spectrum = torch.complex(mag * phase.cos(), mag * phase.sin()).T
        branches.append(
            torch.istft(
                spectrum,
                model.config.n_ffts[i],
                hop_length=model.config.hop,
                window=getattr(model.head, f"window_{i}"),
                length=y.shape[1] * model.config.hop,
            )
        )
    audio = F.conv1d(torch.stack(branches)[None], model.head.mixer.weight, padding=model.config.fir_kernel // 2)[0, 0]
    return model.to_external(audio)


@torch.inference_mode()
def test_metadata_and_code_validation_precede_state_mutation():
    model = ZACDecoderNetwork(small_config()).eval()
    manager = CodecManager(model, 1)
    chunk = ZACCodecChunk("a", 0, 0, torch.zeros((2, 10), dtype=torch.int64))
    with pytest.raises(ValueError, match="at most one"):
        manager.decode([chunk, chunk])
    with pytest.raises(RuntimeError, match="No free"):
        manager.decode([chunk, replace(chunk, request_id="b")])
    assert not manager._streams
    manager.decode([chunk])
    with pytest.raises(ValueError, match="Out-of-order"):
        manager.decode([chunk])
    assert manager._streams["a"].frames == 2
    malformed = replace(chunk, chunk_seq=1, frame_offset=2, codes=torch.full((1, 10), -1))
    with pytest.raises(ValueError, match="Invalid ZAC code"):
        manager.decode([malformed])
    assert manager._streams["a"].chunk_seq == 1


@torch.inference_mode()
def test_failed_neural_cohort_reclaims_slots_before_retry(monkeypatch):
    model = ZACDecoderNetwork(small_config()).eval()
    manager = CodecManager(model, 2)
    forward = model.forward_packed

    def fail_after_state_update(*args):
        forward(*args)
        raise RuntimeError("injected neural failure")

    chunks = [ZACCodecChunk(str(i), 0, 0, torch.zeros((4, 10), dtype=torch.int64)) for i in range(2)]
    monkeypatch.setattr(model, "forward_packed", fail_after_state_update)
    with pytest.raises(RuntimeError, match="injected neural failure"):
        manager.decode(chunks)
    assert not manager._streams and len(manager._free) == 2
    monkeypatch.setattr(model, "forward_packed", forward)
    result = manager.decode([replace(chunk, finished=True) for chunk in chunks])
    for row, chunk in zip(result, chunks, strict=True):
        torch.testing.assert_close(row.audio, whole_audio(model, chunk.codes), atol=2e-7, rtol=3e-4)


# ---------------------------------------------------------------- codec_model


def release_config():
    stages = []
    for i in range(9):
        if i % 2:
            stages.append({"type": "PatchedPretransform", "direction": "up", "patch_size": 2})
        else:
            stages.append(
                {
                    **spec(),
                    "type": "MossProjectedTransformer",
                    "num_layers": 1,
                    "input_dimension": 8,
                    "output_dimension": 16,
                    "frame_rate": 15 * 2 ** (i // 2),
                }
            )
    stages.append({"type": "PatchedPretransform", "patch_size": 23})
    return {
        "model": {
            "sample_rate": 44100,
            "internal_sample_rate": 44160,
            "frame_rate": 15,
            "internal_hop_length": 2944,
            "external_samples_per_frame": 2940,
            "residual_codebooks": 9,
            "semantic_codebook_size": 4096,
            "residual_codebook_size": 4096,
            "codebook_dim": 3,
            "decoder_stages": stages,
            "dual_encoding": {"width": 4, "hidden": 8, "depth": 1},
        },
        "head": {
            "type": "mr_istft",
            "n_ffts": [736, 2208],
            "depth": 1,
            "hidden": 16,
            "fir_kernel": 7,
            "stage_layers": {},
            "trainable_stages": list(range(9)),
        },
    }


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    torch.manual_seed(73)
    files = Zonos3CodecFiles.for_variant("keep2_plain")
    (tmp_path / files.config).parent.mkdir(parents=True)
    config = release_config()
    (tmp_path / files.config).write_text(json.dumps(config))
    oracle = ZACDecoderNetwork(ZACDecoderConfig.from_release(config)).eval()
    generator, trainable = {}, {}
    for name, value in oracle.state_dict().items():
        if name.startswith(("head.", "stages.")):
            trainable[name] = value
        if not name.startswith("head."):
            generator[name.replace("stages.", "decoder.", 1) if name.startswith("stages.") else name] = value
    torch.save(
        {"step": 600000, "config": {"model": config["model"]}, "generator": generator}, tmp_path / files.generator
    )
    torch.save({"step": 50000, "trainable": trainable}, tmp_path / files.checkpoint)
    with (tmp_path / files.generator).open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    # This fixture tests loading, not release-weight parity: only its generated
    # tiny checkpoint is accepted, and production keeps the registry checksum.
    monkeypatch.setattr(codec_checkpoint, "GENERATOR_SHA256", digest)
    config = MagicMock(spec=VllmConfig)
    config.model_config = MagicMock(spec=ModelConfig)
    config.model_config.model = str(tmp_path)
    config.model_config.revision = None
    config.model_config.hf_config = PretrainedConfig(zonos3_decoder="keep2_plain")
    config.model_config.enforce_eager = True
    config.model_config.stage_connector_config = {
        "extra": {"codec_chunk_frames": 15, "codec_chunk_ramp": [1, 2, 4, 8, 15]}
    }
    config.load_config = LoadConfig()
    config.additional_config = {}
    config.scheduler_config = SchedulerConfig(
        max_model_len=16, is_encoder_decoder=False, max_num_seqs=8, max_num_batched_tokens=32
    )
    return config, oracle


def payload(external_id, codes, seq, offset, finished=False):
    return {
        "codes": {"audio": codes},
        "meta": {
            "request_id": external_id,
            "chunk_seq": seq,
            "num_processed_tokens": offset,
            "finished": torch.tensor(finished),
        },
    }


@pytest.mark.parametrize("capacity, expected_batches", [(1, [1]), (3, [1, 2, 3])])
@pytest.mark.parametrize("chunk_size, ramp", [(7, [1, 3, 7]), (5, [])])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_codec_graph_policy_uses_scheduler_and_connector(
    checkpoint, capacity, expected_batches, chunk_size, ramp, enforce_eager
):
    config, _ = checkpoint
    config.scheduler_config.max_num_seqs = capacity
    config.model_config.enforce_eager = enforce_eager
    config.model_config.stage_connector_config = {"extra": {"codec_chunk_frames": chunk_size, "codec_chunk_ramp": ramp}}
    model = Zonos3CodecModel(vllm_config=config)
    manager = model._codec_manager()
    assert manager.graph.batch_sizes == ([capacity] if enforce_eager else expected_batches)
    expected_keys = (
        [] if enforce_eager else [(b, b * t) for b in expected_batches for t in sorted({1, chunk_size, *ramp})]
    )
    assert manager.graph.capture_keys == sorted(expected_keys)
    assert not manager.graph.buckets  # CPU tests never pretend to capture CUDA.
    assert manager.graph.codec.state is manager.state


@pytest.mark.parametrize("termination", ["eos", "length", "length_before_frame", "cancel", "cancel_before_output"])
@torch.inference_mode()
def test_native_finalizer_transport_codec_lifecycle(checkpoint, termination):
    """Real finalizer, output worker and producer feed the incremental Codec.

    The transport backend records connector puts; neural execution uses the
    synthetic release checkpoint rather than replacing the decoder with a mock.
    """
    from types import SimpleNamespace

    from vllm import SamplingParams

    from tests.model_executor.models.test_zonos3_model_state import materialize_codes
    from tests.worker_v2.test_omni_data_plane import (
        _make_plane,
        _new_request,
        _Output,
        _start_save_thread,
        _stop_save_thread,
    )
    from vllm_omni.model_executor.models.zonos3.model_state import Zonos3CodeSnapshot
    from vllm_omni.model_executor.stage_input_processors.zonos3 import zonos3_rvq_async_chunk
    from vllm_omni.worker_v2.native_output_worker import NativeOutputWorker

    config, oracle = checkpoint
    config.scheduler_config.max_num_seqs = 2
    model = Zonos3CodecModel(vllm_config=config)
    model.load_weights(iter(()))
    frames = torch.arange(200).reshape(20, 10)
    cancelled = termination.startswith("cancel")
    length = {"cancel_before_output": 0, "length_before_frame": 4, "eos": 15}.get(termination, 14)
    rows = torch.full((length, 10), 4097, dtype=torch.int64)
    for f in range(5 if termination == "eos" else 20):
        for book in range(10):
            if f + book < length:
                rows[f + book, book] = frames[f, book]
    if termination == "eos":
        for book in range(10):
            rows[5 + book, book] = 4096
    sent = []

    def put(**kwargs):
        sent.append(kwargs["data"] if isinstance(kwargs["data"], dict) else to_dict(kwargs["data"]))
        return True, 1, None

    plane = _make_plane(False)
    connector = SimpleNamespace(
        put=put,
        config={
            "extra": {
                "codebook_sizes": [4096] * 10,
                "eoa_id": 4096,
                "audio_pad_id": 4097,
            }
        },
    )
    _start_save_thread(plane, connector)
    plane._custom_process_func = zonos3_rvq_async_chunk
    plane._custom_process_batch_func = None
    request = _new_request()
    request.resumable = False
    request.sampling_params = SamplingParams(
        max_tokens=length if termination.startswith("length") else 100, stop_token_ids=[2]
    )
    plane.register_request(request)
    plane._start_output_worker(max_pending_batches=2)
    worker = NativeOutputWorker(capacity=2)
    try:
        for i, row in enumerate(rows):
            end = termination == "eos" and i == length - 1
            snapshot = Zonos3CodeSnapshot(row[None])
            worker.submit(_Output(payload=materialize_codes(snapshot)[0], token=2 if end else 1), plane).get_output()
        plane.drain_outputs()
        # A speculative in-flight step after stop must not be published.
        if not cancelled:
            assert (
                plane.complete_outputs(
                    req_ids=["internal"],
                    inter_stage_outputs=[{"codes": {"audio": torch.full((1, 10), -999)}}],
                    sampled_token_ids=[[1]],
                )
                == 0
            )
            plane.request_terminal({"internal"})
        else:
            plane.abort_requests({"internal"})
        assert plane.record.cleaned == ["internal"]
        assert not plane.request_payload and not plane._native_requests
        assert not plane._native_outputs_in_flight and not plane._native_terminal_pending
        audio = []
        for chunk in sent:
            if cancelled and "codes" not in chunk:
                # Cancellation is an engine lifecycle notification, not a
                # fabricated final Codec frame or a synthesis flush.
                assert chunk == {"meta": {"finished": torch.tensor(True)}}
                model.on_requests_finished({"codec-internal"})
                continue
            result = model(request_ids=["codec-internal"], model_intermediate_buffer=[chunk])
            audio.append(result.multimodal_outputs["model_outputs"][0])
        if not cancelled:
            assert [bool(chunk["meta"]["finished"]) for chunk in sent] == [False] * (len(sent) - 1) + [True]
            assert [chunk["meta"]["chunk_seq"] for chunk in sent] == list(range(len(sent)))
            waveform = torch.cat(audio)
            expected_frames = 0 if termination == "length_before_frame" else 5
            assert waveform.numel() == expected_frames * 2940
            if expected_frames:
                torch.testing.assert_close(
                    waveform, whole_audio(oracle, frames[:expected_frames]), atol=5e-7, rtol=1e-3
                )
        else:
            assert "codes" not in sent[-1]
        assert not model._manager._streams and len(model._manager._free) == config.scheduler_config.max_num_seqs
        model.on_requests_finished({"codec-internal"})
        assert not model._internal_to_external and not model._external_to_internal and not model._manager._finished
        # Reusing the same identities allocates clean state after cleanup.
        model(request_ids=["codec-internal"], model_intermediate_buffer=[payload("external", frames[:1], 0, 0, True)])
        model.on_requests_finished({"codec-internal"})
    finally:
        worker.close()
        plane._stop_output_worker()
        _stop_save_thread(plane)


@pytest.mark.parametrize("concurrency", [1, 4, 8])
@torch.inference_mode()
def test_native_model_loading_pcm_materialization_and_identity_cleanup(checkpoint, concurrency):
    config, oracle = checkpoint
    model = Zonos3CodecModel(vllm_config=config)
    assert next(model.network.parameters()).dtype == torch.float32
    with pytest.raises(RuntimeError, match="not been loaded"):
        model(request_ids=[], model_intermediate_buffer=[])
    model.load_weights(iter(()))
    requests = [f"internal-{i}" for i in range(concurrency)]
    codes = torch.randint(0, 4096, (concurrency, 5, 10))
    projected_batch = []
    handle = model.network.stages[0].input_proj.register_forward_pre_hook(
        lambda _, args: projected_batch.append(args[0].shape[0])
    )
    try:
        first = model(
            request_ids=requests,
            model_intermediate_buffer=[payload(f"external-{i}", codes[i, :2], 0, 0) for i in range(concurrency)],
        )
        last = model(
            request_ids=requests,
            model_intermediate_buffer=[payload(f"external-{i}", codes[i, 2:], 1, 2, True) for i in range(concurrency)],
        )
    finally:
        handle.remove()
    assert projected_batch == [2 * concurrency, 3 * concurrency]
    a = OmniGenerationModelRunner._build_pooler_output(first, concurrency)
    b = OmniGenerationModelRunner._build_pooler_output(last, concurrency)
    for i in range(concurrency):
        waveform = torch.cat((a[i]["model_outputs"], b[i]["model_outputs"]))
        assert waveform.numel() == 5 * 2940
        torch.testing.assert_close(waveform, whole_audio(oracle, codes[i]), atol=5e-7, rtol=1e-3)
        assert int(b[i]["sr"]) == 44100
    model.on_requests_finished(set(requests))
    assert not model._internal_to_external and not model._external_to_internal
    # Stable external identity can be reused only after native completion cleanup.
    model(request_ids=[requests[0]], model_intermediate_buffer=[payload("external-0", codes[0], 0, 0, True)])


@torch.inference_mode()
def test_producer_delay_reconstruction_to_native_codec_model(checkpoint):
    config, _ = checkpoint
    model = Zonos3CodecModel(vllm_config=config)
    model.load_weights(iter(()))
    processor = Zonos3RVQStageInputProcessor(
        Zonos3ChunkConfig(
            codebook_sizes=(4096,) * 10,
            eoa_id=4096,
            pad_id=4097,
            chunk_size=3,
            chunk_ramp=(1, 2),
            first_chunk_codebooks=4,
        )
    )
    frames = torch.randint(0, 4096, (8, 10))
    delayed = torch.full((18, 10), 4097, dtype=torch.int64)
    for f in range(8):
        for cb in range(10):
            delayed[f + cb, cb] = frames[f, cb]
    for cb in range(10):
        delayed[8 + cb, cb] = 4096
    output = []
    for step in range(19):
        chunk = processor.push("external", delayed[step : step + 1] if step < 18 else None, finished=step == 18)
        if chunk is None:
            continue
        transport_payload = to_dict(chunk)
        update = DefaultSchedulingMetadataAdapter().extract(transport_payload, model_mode="generation")
        assert update.prompt_token_ids == (0,)
        result = model(request_ids=["internal"], model_intermediate_buffer=[transport_payload])
        output.append(result.multimodal_outputs["model_outputs"][0])
    assert torch.cat(output).numel() == 8 * 2940


def test_checkpoint_integrity_and_native_preemption_rejection(checkpoint, monkeypatch):
    config, _ = checkpoint
    model = Zonos3CodecModel(vllm_config=config)
    monkeypatch.setattr(codec_checkpoint, "GENERATOR_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="checksum mismatch"):
        model.load_weights(iter(()))
    assert not model._weights_loaded
    state = object.__new__(Zonos3CodecModelState)
    with pytest.raises(RuntimeError, match="does not support preemption"):
        state.on_request_preempted("request", 7)


@torch.inference_mode()
def test_weight_loading_is_once_and_restores_ar_default_dtype(checkpoint):
    config, _ = checkpoint
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = Zonos3CodecModel(vllm_config=config)
        assert torch.get_default_dtype() == torch.bfloat16
        assert all(parameter.dtype == torch.float32 for parameter in model.network.parameters())
    finally:
        torch.set_default_dtype(previous_dtype)
    model.load_weights(iter(()))
    with pytest.raises(RuntimeError, match="already loaded"):
        model.load_weights(iter(()))


@torch.inference_mode()
def test_producer_cancel_discards_codec_tail_without_affecting_other_batch_request(checkpoint):
    from tests.model_executor.stage_input_processors.test_zonos3 import Manager, Request
    from vllm_omni.model_executor.stage_input_processors.zonos3 import zonos3_rvq_async_chunk

    config, oracle = checkpoint
    model = Zonos3CodecModel(vllm_config=config)
    model.load_weights(iter(()))
    codes = torch.randint(0, 4096, (6, 10))
    model(request_ids=["native-a"], model_intermediate_buffer=[payload("a", codes, 0, 0)])
    manager = Manager()
    assert zonos3_rvq_async_chunk(manager, None, Request("a", aborted=True), True) is None
    model.on_requests_finished({"native-a"})
    calls = []
    handle = model.network.stages[0].input_proj.register_forward_pre_hook(
        lambda _, args: calls.append(args[0].shape[0])
    )
    try:
        output = model(
            request_ids=["native-b"],
            model_intermediate_buffer=[payload("b", codes, 0, 0, True)],
        )
    finally:
        handle.remove()
    assert calls == [len(codes)]  # cancelled request does not run a neural forward
    torch.testing.assert_close(
        output.multimodal_outputs["model_outputs"][0], whole_audio(oracle, codes), atol=5e-7, rtol=1e-3
    )
    assert "a" not in model._codec_manager()._streams
    assert "native-a" not in model._internal_to_external


# ---------------------------------------------------------------- codec_graph


def graph_network(device="cpu"):
    torch.manual_seed(73)
    return ZACDecoderNetwork(ZACDecoderConfig.from_release(release_config())).to(device).eval()
