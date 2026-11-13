# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Zonos3 codec module tests: layers, network, streaming synthesis, manager,
model and complete-graph arithmetic against stateless / official oracles.

The optional reference sections run only with ZONOS3_REFERENCE_ROOT pointing
at an unmodified Zyphra/ZONOS3-Mini checkout.
"""

import ast
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torchaudio.transforms import Resample
from transformers import PretrainedConfig
from vllm.config import LoadConfig, ModelConfig, SchedulerConfig, VllmConfig

from vllm_omni.data_entry_keys import to_dict
from vllm_omni.model_executor.models.registry import OmniModelRegistry
from vllm_omni.model_executor.models.zonos3 import codec as codec_checkpoint
from vllm_omni.model_executor.models.zonos3.codec import (
    CodecManager,
    ZACCodecChunk,
    ZACConvNeXt,
    ZACDecoderConfig,
    ZACDecoderNetwork,
    ZACISTFTStream,
    ZACMossLayer,
    ZACMossStage,
    ZACPatch,
    ZACResamplerStream,
    ZACTransformerSpec,
    Zonos3CodecFiles,
    Zonos3CodecModel,
)
from vllm_omni.model_executor.models.zonos3.cuda_graph_wrapper import ZACCodecGraphWrapper
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


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("context", [1, 7, 23])
@torch.inference_mode()
def test_batched_moss_layer_mixed_positions_and_chunk_sizes(concurrency, context):
    torch.manual_seed(41)
    layer = ZACMossLayer(spec(context)).eval()
    state = layer.allocate_state(concurrency, torch.float32, "cpu")
    inputs = [torch.randn(1, 40, 16) for _ in range(concurrency)]
    expected = [whole_sequence(layer, x) for x in inputs]
    positions = [0] * concurrency
    calls = []
    handle = layer.qkv.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
    try:
        # Prime different histories, then run one B=concurrency neural forward
        # per compatible chunk length. Requests enter/leave in a changing order.
        for i in range(concurrency):
            n = i % 5
            if n:
                got = layer(inputs[i][:, :n], torch.tensor([i]), torch.tensor([0]), state)
                torch.testing.assert_close(got, expected[i][:, :n], atol=2e-6, rtol=2e-5)
                positions[i] = n
        for length in [1, 2, 4, 8, 3, 9]:
            order = list(reversed(range(concurrency))) if length % 2 else list(range(concurrency))
            batch = torch.cat([inputs[i][:, positions[i] : positions[i] + length] for i in order])
            got = layer(batch, torch.tensor(order), torch.tensor([positions[i] for i in order]), state)
            for row, i in enumerate(order):
                torch.testing.assert_close(
                    got[row], expected[i][0, positions[i] : positions[i] + length], atol=2e-6, rtol=2e-5
                )
                positions[i] += length
        assert calls[-6:] == [concurrency] * 6  # actual batched projection execution
    finally:
        handle.remove()


@torch.inference_mode()
def test_moss_slot_reuse_does_not_attend_to_previous_request():
    torch.manual_seed(21)
    layer = ZACMossLayer(spec()).eval()
    state = layer.allocate_state(2, torch.float32, "cpu")
    slots = torch.tensor([1])
    layer(torch.randn(1, 12, 16), slots, torch.tensor([0]), state)
    state.reset(slots)
    x = torch.randn(1, 4, 16)
    torch.testing.assert_close(layer(x, slots, torch.tensor([0]), state), whole_sequence(layer, x))
    assert not state.keys[0].any()


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_convnext_batched_incremental_matches_whole_causal_convolution(concurrency):
    torch.manual_seed(12)
    block = ZACConvNeXt(8, 16).eval()
    history = block.allocate_state(concurrency, torch.float32, "cpu")
    x = torch.randn(concurrency, 8, 27)
    h = block.dwconv(F.pad(x, (6, 0))).transpose(1, 2)
    expected = x + block.pwconv2(F.gelu(block.pwconv1(block.norm(h)))).transpose(1, 2)
    outputs = []
    start = 0
    for n in [1, 2, 4, 8, 12]:
        order = torch.arange(concurrency - 1, -1, -1)
        y = block(x[order, :, start : start + n], order, history)
        outputs.append(y[order])
        start += n
    torch.testing.assert_close(torch.cat(outputs, -1), expected)


def test_patch_preserves_official_channel_to_time_order():
    patch = ZACPatch(2)
    x = torch.arange(24).reshape(2, 4, 3)
    expected = torch.tensor(
        [[[0, 3, 1, 4, 2, 5], [6, 9, 7, 10, 8, 11]], [[12, 15, 13, 16, 14, 17], [18, 21, 19, 22, 20, 23]]]
    )
    assert torch.equal(patch(x), expected)


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


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_neural_decoder_batches_different_chunk_positions(concurrency):
    torch.manual_seed(31)
    model = ZACDecoderNetwork(small_config()).eval()
    state = model.allocate_state(concurrency)
    codes = torch.randint(0, 32, (concurrency, 24, 10))
    # Official partial-codebook policy leaves absent residual contributions out.
    codes[:, 0, 4:] = -1
    expected = [whole_trunk(model, row[None]) for row in codes]
    positions = [0] * concurrency
    for i in range(concurrency):
        n = i % 4
        if n:
            model(codes[i : i + 1, :n], torch.tensor([i]), state)
            positions[i] = n
    projected_batches = []
    handles = [
        stage.input_proj.register_forward_pre_hook(lambda _, args: projected_batches.append(args[0].shape[0]))
        for stage in model.stages
        if isinstance(stage, ZACMossStage)
    ]
    try:
        for n in [1, 2, 4, 8, 3]:
            order = torch.arange(concurrency - 1, -1, -1)
            batch = torch.cat([codes[i : i + 1, positions[i] : positions[i] + n] for i in order.tolist()])
            output = model(batch, order, state)
            for row, i in enumerate(order.tolist()):
                for actual, oracle in zip(output, expected[i], strict=True):
                    torch.testing.assert_close(
                        actual[row], oracle[0, 2 * positions[i] : 2 * (positions[i] + n)], atol=3e-6, rtol=3e-5
                    )
                positions[i] += n
        assert projected_batches == [concurrency] * 10
        assert state.frames.tolist() == positions
    finally:
        for handle in handles:
            handle.remove()


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


@torch.inference_mode()
def test_reset_reclaims_all_neural_histories():
    torch.manual_seed(17)
    model = ZACDecoderNetwork(replace(small_config(), head_depth=1)).eval()
    state = model.allocate_state(2)
    slot = torch.tensor([1])
    model(torch.randint(0, 32, (1, 10, 10)), slot, state)
    state.reset(slot)
    codes = torch.randint(0, 32, (1, 3, 10))
    for actual, expected in zip(model(codes, slot, state), whole_trunk(model, codes), strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


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


@pytest.mark.parametrize("frames", [1, 2, 11, 35, 129])
@pytest.mark.parametrize("chunks", [(1,), (1, 2, 4, 8, 15), (32,)])
@torch.inference_mode()
def test_streaming_multi_resolution_istft_matches_full_synthesis(frames, chunks):
    torch.manual_seed(42)
    hop, n_ffts = 184, (736, 2208)
    windows = [torch.hann_window(n) for n in n_ffts]
    spectra = [torch.randn(frames, n + 2) * 0.2 for n in n_ffts]
    mixer = torch.randn(2, 127) * 0.1
    stream = ZACISTFTStream(n_ffts, hop, windows, mixer, math.log(100.0))
    output, branches = [], []
    for y, n, window in zip(spectra, n_ffts, windows, strict=True):
        log_mag, phase = y.chunk(2, dim=-1)
        mag = log_mag.clamp(max=math.log(100.0)).exp()
        complex_spectrum = torch.complex(mag * phase.cos(), mag * phase.sin()).T
        branches.append(torch.istft(complex_spectrum, n, hop_length=hop, window=window, length=frames * hop))
    reference = F.conv1d(torch.stack(branches)[None], mixer[None], padding=63).flatten()
    start = 0
    index = 0
    while start < frames:
        n = min(chunks[min(index, len(chunks) - 1)], frames - start)
        output.append(stream.push([y[start : start + n] for y in spectra]))
        start += n
        index += 1
        # Only finite overlap / mixer history remains, independent of utterance length.
        assert max(x.numel() for x in stream.ola) <= max(n_ffts)
        assert max(x.numel() for x in stream.mix_input) <= max(n_ffts) + 127
    output.append(stream.flush())
    actual = torch.cat(output)
    assert actual.shape == (frames * hop,)
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, reference, atol=2e-7, rtol=2e-4)
    assert stream.flush().numel() == 0
    with pytest.raises(RuntimeError, match="flushed"):
        stream.push(spectra)


@pytest.mark.parametrize("samples", [0, 1, 73, 1472, 2944, 29440])
@torch.inference_mode()
def test_incremental_exact_rate_resampling(samples):
    torch.manual_seed(1)
    resampler = Resample(44160, 44100, lowpass_filter_width=16, rolloff=0.99)
    stream = ZACResamplerStream(
        resampler.kernel[:, 0],
        resampler.orig_freq // resampler.gcd,
        resampler.new_freq // resampler.gcd,
        resampler.width,
    )
    x = torch.randn(samples) * 0.1
    out = []
    for start in range(0, samples, 137):
        out.append(stream.push(x[start : start + 137]))
        assert stream.buffer.numel() < 2 * stream.kernel_length
    out.append(stream.flush())
    actual = torch.cat(out)
    expected = resampler(x) if samples else x
    assert actual.numel() == math.ceil(samples * 44100 / 44160)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    assert stream.flush().numel() == 0


@torch.inference_mode()
def test_request_synthesis_states_do_not_share_buffers():
    windows = [torch.hann_window(n) for n in (8, 24)]
    mixer = torch.ones(2, 3) / 6
    a = ZACISTFTStream((8, 24), 2, windows, mixer, 4.0)
    b = ZACISTFTStream((8, 24), 2, windows, mixer, 4.0)
    a.push([torch.randn(3, n + 2) for n in (8, 24)])
    assert b.frames == 0 and all(x.numel() == 0 for x in b.ola)
    assert b.flush().numel() == 0


# ---------------------------------------------------------------- codec_reference


@pytest.fixture(scope="module")
def reference_root():
    root = os.environ.get("ZONOS3_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set ZONOS3_REFERENCE_ROOT for official-source CPU numerical checks")
    path = Path(root)
    if not path.is_dir():
        pytest.fail("ZONOS3_REFERENCE_ROOT does not name an official source checkout")
    return path


def definitions(root, relative, names):
    path = root / relative
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    namespace = {"torch": torch, "nn": nn, "F": F, "math": math}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@torch.inference_mode()
def test_official_convnext_causal_chunk_parity(reference_root):
    oracle = definitions(reference_root, "model/src/zonos3/codec/hybrid10_dual_encoding/model.py", {"SemanticConvNeXt"})
    torch.manual_seed(3)
    ref = oracle["SemanticConvNeXt"](8, 16).eval()
    native = ZACConvNeXt(8, 16).eval()
    native.load_state_dict(ref.state_dict(), strict=True)
    x = torch.randn(4, 8, 31)
    expected = ref(x)
    state = native.allocate_state(4, torch.float32, "cpu")
    outputs = []
    start = 0
    for n in (1, 2, 4, 8, 16):
        outputs.append(native(x[:, :, start : start + n], torch.arange(4), state))
        start += n
    torch.testing.assert_close(torch.cat(outputs, -1), expected)


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_official_incremental_moss_layer_batch_parity(reference_root, concurrency):
    oracle = definitions(reference_root, "model/src/zonos3/codec/hybrid10_dual_encoding/base_model.py", {"MossLayer"})
    oracle_step = definitions(reference_root, "model/src/zonos3/codec/istft_head_a_20261002/stream.py", {"_moss_layer"})
    torch.manual_seed(5)
    config = {**spec(), "attention_dropout": 0.0, "ffn_dropout": 0.0}
    ref = oracle["MossLayer"](config).eval()
    native = ZACMossLayer(config).eval()
    native.load_state_dict(ref.state_dict(), strict=True)
    state = native.allocate_state(concurrency, torch.float32, "cpu")
    caches, positions = [None] * concurrency, [0] * concurrency
    for row in range(concurrency):
        n = row % 5
        if n:
            prefix = torch.randn(1, n, 16)
            _, caches[row] = oracle_step["_moss_layer"](ref, prefix, torch.tensor(0), None)
            native(prefix, torch.tensor([row]), torch.tensor([0]), state)
            positions[row] = n
    for n in (1, 2, 4, 8):
        x = torch.randn(concurrency, n, 16)
        actual = native(x, torch.arange(concurrency), torch.tensor(positions), state)
        for row in range(concurrency):
            expected, caches[row] = oracle_step["_moss_layer"](
                ref, x[row : row + 1], torch.tensor(positions[row]), caches[row]
            )
            torch.testing.assert_close(actual[row : row + 1], expected, atol=2e-6, rtol=2e-5)
            positions[row] += n


@torch.inference_mode()
def test_official_stream_device_synthesis_and_resampler_parity(reference_root):
    oracle = definitions(
        reference_root, "model/src/zonos3/codec/stream_device.py", {"IstftHeadStream", "ResamplerStream"}
    )
    torch.manual_seed(7)
    n_ffts, hop = (736, 2208), 184
    windows = [torch.hann_window(n) for n in n_ffts]
    mixer = torch.randn(2, 127) * 0.1
    native = ZACISTFTStream(n_ffts, hop, windows, mixer, math.log(100.0))
    ref = oracle["IstftHeadStream"](n_ffts, hop, windows, mixer, math.log(100.0), "cpu")
    for frames in (1, 2, 4, 8, 15, 15):
        spectra = [torch.randn(frames, n + 2) * 0.2 for n in n_ffts]
        torch.testing.assert_close(native.push(spectra), ref.push(*spectra), atol=0, rtol=0)
    torch.testing.assert_close(native.flush(), ref.flush(), atol=0, rtol=0)
    rs = Resample(44160, 44100, lowpass_filter_width=16, rolloff=0.99)
    args = (rs.kernel[:, 0], rs.orig_freq // rs.gcd, rs.new_freq // rs.gcd, rs.width)
    native_rs = ZACResamplerStream(*args)
    reference_rs = oracle["ResamplerStream"](*args, "cpu")
    for n in (1, 1472, 2944, 5888, 131):
        x = torch.randn(n) * 0.1
        torch.testing.assert_close(native_rs.push(x), reference_rs.push(x), atol=0, rtol=0)
    torch.testing.assert_close(native_rs.flush(), reference_rs.flush(), atol=0, rtol=0)


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


@pytest.mark.parametrize("concurrency", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_incremental_batched_neural_decode_through_pcm_and_final_flush(concurrency):
    torch.manual_seed(32)
    model = ZACDecoderNetwork(small_config()).eval()
    manager = CodecManager(model, concurrency)
    codes = [torch.randint(0, 32, (17 + i % 4, 10)) for i in range(concurrency)]
    for row in codes:
        row[0, 4:] = -1
    expected = [whole_audio(model, row) for row in codes]
    offsets, seqs = [0] * concurrency, [0] * concurrency
    parts = [[] for _ in codes]
    finished = set()
    batch_sizes = []
    handle = model.register_forward_pre_hook(lambda _, args: batch_sizes.append(args[0].shape[0]))
    try:
        for step in range(40):
            chunks = []
            for i in reversed(range(concurrency)):
                if i in finished or step < i % 3:
                    continue
                n = min((1, 2, 4, 8)[min(seqs[i], 3)], len(codes[i]) - offsets[i])
                # Half the requests finish with data, half with an empty marker.
                last = n == 0 or (offsets[i] + n == len(codes[i]) and i % 2 == 0)
                chunks.append(ZACCodecChunk(str(i), seqs[i], offsets[i], codes[i][offsets[i] : offsets[i] + n], last))
                offsets[i] += n
                seqs[i] += 1
                if last:
                    finished.add(i)
            output = manager.decode(chunks)
            assert [row.request_id for row in output] == [chunk.request_id for chunk in chunks]
            for row in output:
                parts[int(row.request_id)].append(row.audio)
            if len(finished) == concurrency:
                break
        assert len(finished) == concurrency
        for got, want in zip(parts, expected, strict=True):
            audio = torch.cat(got)
            assert audio.shape == want.shape
            torch.testing.assert_close(audio, want, atol=2e-7, rtol=3e-4)
        if concurrency > 1:
            assert max(batch_sizes) > 1  # the actual neural forward is batched
        assert not manager._streams and len(manager._free) == concurrency
    finally:
        handle.remove()


@torch.inference_mode()
def test_cancel_reuse_and_output_ownership():
    torch.manual_seed(29)
    model = ZACDecoderNetwork(small_config()).eval()
    manager = CodecManager(model, 1)
    codes = torch.randint(0, 32, (16, 10))
    output = manager.decode([ZACCodecChunk("cancel", 0, 0, codes)])
    snapshot = output[0].audio.clone()
    manager.release("cancel")  # discards unfinished synthesis without flushing
    other = torch.randint(0, 32, (12, 10))
    got = manager.decode([ZACCodecChunk("next", 0, 0, other, True)])[0]
    torch.testing.assert_close(got.audio, whole_audio(model, other), atol=2e-7, rtol=3e-4)
    assert torch.equal(output[0].audio, snapshot)
    with pytest.raises(ValueError, match="already finished"):
        manager.decode([ZACCodecChunk("next", 0, 0, other, True)])
    manager.release("next")
    assert manager.decode([ZACCodecChunk("next", 0, 0, other, True)])[0].finished


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
def test_context_audio_is_decoded_but_skipped_once():
    torch.manual_seed(24)
    model = ZACDecoderNetwork(small_config()).eval()
    manager = CodecManager(model, 1)
    codes = torch.randint(0, 32, (32, 10))
    want = whole_audio(model, codes)[21:]
    a = manager.decode([ZACCodecChunk("a", 0, 0, codes[:8], skip_samples=21)])[0]
    b = manager.decode([ZACCodecChunk("a", 1, 8, codes[8:], True)])[0]
    torch.testing.assert_close(torch.cat((a.audio, b.audio)), want, atol=2e-7, rtol=3e-4)


@torch.inference_mode()
def test_failed_neural_cohort_reclaims_slots_before_retry(monkeypatch):
    model = ZACDecoderNetwork(small_config()).eval()
    manager = CodecManager(model, 2)
    forward = model.forward

    def fail_after_state_update(*args):
        forward(*args)
        raise RuntimeError("injected neural failure")

    chunks = [ZACCodecChunk(str(i), 0, 0, torch.zeros((4, 10), dtype=torch.int64)) for i in range(2)]
    monkeypatch.setattr(model, "forward", fail_after_state_update)
    with pytest.raises(RuntimeError, match="injected neural failure"):
        manager.decode(chunks)
    assert not manager._streams and len(manager._free) == 2
    monkeypatch.setattr(model, "forward", forward)
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
        max_model_len=16, is_encoder_decoder=False, max_num_seqs=32, max_num_batched_tokens=32
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


@pytest.mark.parametrize("capacity, expected_batches", [(1, [1]), (3, [1, 2, 3]), (8, [1, 2, 4, 8])])
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
    if enforce_eager:
        assert manager.graph is None
        assert manager.state.frames.shape[0] == capacity
    else:
        assert manager.graph.batch_sizes == expected_batches
        assert manager.graph.frame_sizes == sorted({1, chunk_size, *ramp})
        assert manager.state.frames.shape[0] == 2 * capacity
        assert manager.graph.frame_bucket(0) == 1
        assert manager.graph.frame_bucket(chunk_size - 1) == chunk_size


@pytest.mark.parametrize("termination", ["eos", "length", "length_before_frame", "cancel", "cancel_before_output"])
@pytest.mark.parametrize("complete_codec", [False, True])
@torch.inference_mode()
def test_native_finalizer_transport_codec_lifecycle(checkpoint, termination, complete_codec):
    """Real finalizer, output worker and producer feed the incremental Codec.

    The transport backend records connector puts; neural execution uses the
    synthetic release checkpoint rather than replacing the decoder with a mock.
    """
    from types import SimpleNamespace

    from vllm import SamplingParams

    from tests.worker_v2.test_omni_data_plane import (
        _make_plane,
        _new_request,
        _Output,
        _start_save_thread,
        _stop_save_thread,
    )
    from vllm_omni.model_executor.models.zonos3.sampler import finalize_zonos3_codes
    from vllm_omni.model_executor.stage_input_processors.zonos3 import zonos3_rvq_async_chunk
    from vllm_omni.worker_v2.native_output_worker import NativeOutputWorker

    config, oracle = checkpoint
    config.model_config.enforce_eager = not complete_codec
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
        sent.append(to_dict(kwargs["data"]))
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
            materialized = finalize_zonos3_codes(
                {
                    "codes": {"audio": row[None, None]},
                    "meta": {"codec_frame_valid": torch.tensor([True]), "codec_eos_valid": torch.tensor([end])},
                },
                [1],
            )
            worker.submit(_Output(payload=materialized.inter_stage[0], token=2 if end else 1), plane).get_output()
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
            assert sent[-1]["meta"]["aborted"] and not audio[-1].numel()
        assert not model._manager._streams and len(model._manager._free) == 32
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
    handle = model.network.register_forward_pre_hook(lambda _, args: projected_batch.append(args[0].shape[0]))
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
    assert projected_batch == [concurrency, concurrency]
    a = OmniGenerationModelRunner._build_pooler_output(first, concurrency)
    b = OmniGenerationModelRunner._build_pooler_output(last, concurrency)
    for i in range(concurrency):
        waveform = torch.cat((a[i]["model_outputs"], b[i]["model_outputs"]))
        assert waveform.numel() == 5 * 2940
        torch.testing.assert_close(waveform, whole_audio(oracle, codes[i]), atol=5e-7, rtol=1e-3)
        assert int(b[i]["sr"]) == 44100
    terminal = model(
        request_ids=requests,
        model_intermediate_buffer=[
            {"meta": {"request_id": f"external-{i}", "transport_terminal": True, "finished": torch.tensor(False)}}
            for i in range(concurrency)
        ],
    )
    terminal_audio = OmniGenerationModelRunner._build_pooler_output(terminal, concurrency)
    assert all(not row["model_outputs"].numel() for row in terminal_audio)
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
def test_codec_cancel_and_identity_validation(checkpoint):
    config, _ = checkpoint
    model = Zonos3CodecModel(vllm_config=config)
    model.load_weights(iter(()))
    codes = torch.zeros((2, 10), dtype=torch.int64)
    model(request_ids=["internal"], model_intermediate_buffer=[payload("external", codes, 0, 0)])
    with pytest.raises(ValueError, match="identity changed"):
        model(request_ids=["internal"], model_intermediate_buffer=[payload("other", codes, 1, 2)])
    model.on_requests_finished({"internal"})
    assert not model._codec_manager()._streams
    with pytest.raises(ValueError, match="external request identity"):
        model(request_ids=["bad"], model_intermediate_buffer=[{}])


def test_native_codec_architecture_is_lazily_registered():
    from vllm.model_executor.models.interfaces_base import is_text_generation_model

    assert OmniModelRegistry.models["Zonos3CodecModel"].load_model_cls() is Zonos3CodecModel
    assert is_text_generation_model(Zonos3CodecModel)


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
    aborted = zonos3_rvq_async_chunk(Manager(), None, Request("a", aborted=True), True)
    calls = []
    handle = model.network.register_forward_pre_hook(lambda _, args: calls.append(args[0].shape[0]))
    try:
        output = model(
            request_ids=["native-a", "native-b"],
            model_intermediate_buffer=[
                to_dict(aborted),
                payload("b", codes, 0, 0, True),
            ],
        )
    finally:
        handle.remove()
    assert calls == [1]  # cancelled request does not run a neural forward
    assert output.multimodal_outputs["model_outputs"][0].numel() == 0
    torch.testing.assert_close(
        output.multimodal_outputs["model_outputs"][1], whole_audio(oracle, codes), atol=5e-7, rtol=1e-3
    )
    assert "a" not in model._codec_manager()._streams
    assert "native-a" not in model._internal_to_external


# ---------------------------------------------------------------- codec_graph


def graph_network(device="cpu"):
    torch.manual_seed(73)
    return ZACDecoderNetwork(ZACDecoderConfig.from_release(release_config())).to(device).eval()


@pytest.mark.parametrize("batch", [1, 4, 8, 16, 32])
@torch.inference_mode()
def test_padded_complete_decode_mixed_progress_and_final_matches_eager(batch):
    network = graph_network()
    state = network.allocate_state(2 * batch)
    wrapper = ZACCodecGraphWrapper(network, state, batch, (batch,), (4,))
    oracle = CodecManager(network, batch)
    progress = [0] * batch
    closed = set()
    for step in range(6):
        codes = torch.randint(0, 4096, (batch, 4, 10))
        codes[:, 0, 3:] = -1
        lengths = torch.tensor([(i + step) % 4 + 1 if i not in closed else 0 for i in range(batch)])
        final = torch.tensor([step == 2 + i % 3 for i in range(batch)])
        slots = torch.tensor([batch + i if i in closed else i for i in range(batch)])
        audio, valid = wrapper(codes, slots, lengths, final)
        for i in range(batch):
            if i in closed:
                assert int(valid[i]) == 0
                continue
            chunk = ZACCodecChunk(str(i), step, progress[i], codes[i, : lengths[i]], bool(final[i]))
            expected = oracle.decode([chunk])[0].audio
            assert int(valid[i]) == len(expected)
            torch.testing.assert_close(audio[i, : valid[i]], expected, rtol=1e-4, atol=1e-6)
            progress[i] += int(lengths[i])
            if final[i]:
                closed.add(i)
    assert not state.frames[batch:].any()
    assert state.frames[:batch].tolist() == progress
    assert all(h.shape[1] == wrapper.synthesis.history_columns for h in wrapper.synthesis.history)


@torch.inference_mode()
def test_complete_decode_empty_final_skip_cancel_and_slot_reuse():
    network = graph_network()
    eager = CodecManager(network, 1)
    padded = CodecManager(network, 1, graph_batch_sizes=(1,), graph_frame_sizes=(1, 4))
    codes = torch.randint(0, 4096, (9, 10))
    kept = []
    for request in ("cancelled", "reused"):
        for seq, (lo, hi, done) in enumerate(((0, 1, False), (1, 4, False), (4, 9, False), (9, 9, True))):
            chunk = ZACCodecChunk(request, seq, lo, codes[lo:hi], done, 2940 if seq == 0 else 0)
            a, b = eager.decode([chunk])[0].audio, padded.decode([chunk])[0].audio
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
            kept.append((b, b.clone()))
            if request == "cancelled":
                eager.release(request)
                padded.release(request)
                break
        eager.release(request)
        padded.release(request)
    for audio, snapshot in kept:
        torch.testing.assert_close(audio, snapshot, rtol=0, atol=0)
    assert not padded._streams and padded._free == [0]
    empty = ZACCodecChunk("empty", 0, 0, torch.empty(0, 10, dtype=torch.long), True)
    assert padded.decode([empty])[0].audio.numel() == 0


@torch.inference_mode()
def test_manager_groups_different_lengths_and_final_into_shared_buckets():
    network = graph_network()
    oracle = CodecManager(network, 2)
    manager = CodecManager(network, 2, graph_batch_sizes=(2,), graph_frame_sizes=(1, 4))
    codes = torch.randint(0, 4096, (2, 6, 10))
    calls = []
    hook = network.register_forward_pre_hook(lambda _, args: calls.append(tuple(args[0].shape)))
    try:
        cohorts = (
            [ZACCodecChunk("a", 0, 0, codes[0, :2]), ZACCodecChunk("b", 0, 0, codes[1, :3])],
            [ZACCodecChunk("b", 1, 3, codes[1, 3:4]), ZACCodecChunk("a", 1, 2, codes[0, :0], True)],
            [ZACCodecChunk("b", 2, 4, codes[1, 4:], True)],
        )
        for chunks in cohorts:
            expected = oracle.decode(chunks)
            calls.clear()
            actual = manager.decode(chunks)
            # One network forward for the already-scheduled compatible cohort,
            # including a zero-frame final row beside a live continuation.
            assert len(calls) == 1 and calls[0][0] == len(chunks)
            for a, b in zip(actual, expected, strict=True):
                assert a.request_id == b.request_id and a.finished == b.finished
                torch.testing.assert_close(a.audio, b.audio, rtol=1e-4, atol=1e-6)
        assert not manager._streams and len(manager._free) == 2
    finally:
        hook.remove()
