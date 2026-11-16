# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed Codec checkpoint loading and independent stateless numerical oracles."""

import math
from typing import Any

import pytest
import torch
from torch.nn import functional as F

from vllm_omni.model_executor.models.zonos3.codec import (
    ZACDecoderConfig,
    ZACDecoderNetwork,
    ZACMossLayer,
    ZACPatch,
    ZACTransformerSpec,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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
    release: dict[str, Any] = {
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


def graph_network(device="cpu"):
    torch.manual_seed(73)
    return ZACDecoderNetwork(ZACDecoderConfig.from_release(release_config())).to(device).eval()


def write_codec_release(root, monkeypatch):
    """A complete paired release, retaining the real strict checkpoint loader."""
    import hashlib
    import json

    from vllm_omni.model_executor.models.zonos3 import codec

    config = release_config()
    network = ZACDecoderNetwork(ZACDecoderConfig.from_release(config))
    files = codec.Zonos3CodecFiles.for_variant("keep2_plain")
    for relative in (files.config, files.generator, files.checkpoint):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
    (root / files.config).write_text(json.dumps(config))
    generator, trainable = {}, {}
    for key, value in network.state_dict().items():
        if key.startswith(("head.", *(f"stages.{i}." for i in network.config.trainable_stages))):
            trainable[key] = value
        if not key.startswith("head."):
            generator[key.replace("stages.", "decoder.", 1)] = value
    torch.save({"step": 600000, "config": config, "generator": generator}, root / files.generator)
    torch.save({"step": 50000, "trainable": trainable}, root / files.checkpoint)
    monkeypatch.setattr(codec, "GENERATOR_SHA256", hashlib.sha256((root / files.generator).read_bytes()).hexdigest())
