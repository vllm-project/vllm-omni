# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
from vllm_omni.diffusion.models.minimax_h3.controlnet import (
    MiniMaxH3ControlNet,
    build_control_rows,
    fit_control_canvas,
    load_control_pixels,
    parse_control,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class CPUAttention(nn.Module):
    """Real SDPA, avoiding CUDA backend discovery in CPU-only model tests."""

    def __init__(self, **kwargs):
        super().__init__()
        self.attn_backend = SimpleNamespace(
            supports_packed_mask_free=lambda: False,
            supports_prefix_kv_slicing=False,
        )

    def forward(self, q, k, v, metadata):
        mask = metadata.attn_mask
        if mask is not None:
            mask = mask[:, None, None, :]
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        ).transpose(1, 2)


@pytest.fixture
def cpu_layers(monkeypatch):
    monkeypatch.setattr(blocks, "Attention", CPUAttention)
    with ExitStack() as stack:
        for module in ("vllm.model_executor.layers.linear", "vllm.model_executor.parameter", h3.__name__):
            for name, value in (("get_tensor_model_parallel_world_size", 1), ("get_tensor_model_parallel_rank", 0)):
                stack.enter_context(patch(module + "." + name, return_value=value, create=True))
        yield


def tiny_arch():
    return blocks.MiniMaxH3DiTArchConfig(
        num_layers=2,
        token_refiner_num_layers=0,
        hidden_size=8,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=16,
        time_embed_dim=4,
        time_embed_hidden_size=8,
        timestep_input_dim=8,
        adaln_out_features=144,
        final_adaln_out_features=16,
        text_dim=8,
        rope_inv_freq_len=1,
    )


def original_weights(arch, blocks=2):
    h, a, f, t = (
        arch.hidden_size,
        arch.num_attention_heads * arch.attention_head_dim,
        arch.ffn_hidden_size,
        arch.time_embed_dim,
    )
    shapes = {"control_proj_in.weight": (h, 196), "control_proj_in.bias": (h,)}
    for i in range(blocks):
        prefix = f"control_blocks.{i}."
        block_shapes = {
            "attn.to_q.weight": (a, h),
            "attn.to_k.weight": (a, h),
            "attn.to_v.weight": (a, h),
            "attn.to_out.0.weight": (h, a),
            "attn.norm_q.weight": (arch.attention_head_dim,),
            "attn.norm_k.weight": (arch.attention_head_dim,),
            "norm1.weight": (h,),
            "norm2.weight": (h,),
            "ff.net.0.proj.weight": (2 * f, h),
            "ff.net.2.weight": (h, f),
            "adaln_proj.linear.weight": (18 * h, t),
            "adaln_proj.linear.bias": (18 * h,),
            "after_proj.weight": (h, h),
            "after_proj.bias": (h,),
        }
        if i == 0:
            block_shapes.update({"before_proj.weight": (h, h), "before_proj.bias": (h,)})
        shapes.update({prefix + k: v for k, v in block_shapes.items()})
    generator = torch.Generator().manual_seed(17)
    return {
        name: (torch.randn(shape, generator=generator) * 0.15).to(
            torch.float32 if name.startswith("control_proj_in") else torch.bfloat16
        )
        for name, shape in shapes.items()
    }


def test_original_loader_qkv_and_swiglu_order(cpu_layers):
    model = MiniMaxH3ControlNet(tiny_arch(), (0, 1))
    weights = original_weights(model.arch)
    loaded = model.load_weights(weights.items())
    assert loaded == set(dict(model.named_parameters()))
    block = model.control_blocks[0]
    assert torch.equal(
        block.attn.qkv_proj.weight, torch.cat([weights[f"control_blocks.0.attn.to_{part}.weight"] for part in "qkv"])
    )
    up, gate = weights["control_blocks.0.ff.net.0.proj.weight"].chunk(2)
    assert torch.equal(block.mlp.fc1.weight, torch.cat((gate, up)))
    for subset in (list(weights.items())[:-1], [*weights.items(), next(iter(weights.items()))]):
        with pytest.raises(ValueError, match="Missing|Unexpected"):
            model.load_weights(subset)
    malformed = dict(weights)
    malformed["control_blocks.0.adaln_proj.linear.weight"] = torch.zeros(8, 4)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        model.load_weights(malformed.items())


def reference_block(x, temb, indices, weights, prefix, arch):
    """Independent full-width VideoX-Fun block math, using original weight names."""

    def linear(x, name, bias=False):
        return F.linear(x, weights[prefix + name + ".weight"], weights[prefix + name + ".bias"] if bias else None)

    def norm(x, name, eps):
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            * weights[prefix + name + ".weight"].float()
        ).to(x.dtype)

    table = linear(F.silu(temb).to(torch.bfloat16), "adaln_proj.linear", True)
    shift, scale, gate, shift_ff, scale_ff, gate_ff = table.reshape(-1, 6 * arch.hidden_size).chunk(6, -1)
    z = (norm(x, "norm1", arch.norm_eps).float() * (1 + scale[indices].float()) + shift[indices].float()).to(x.dtype)
    q, k, v = [
        linear(z, f"attn.to_{part}").view(1, len(x), arch.num_attention_heads, arch.attention_head_dim)
        for part in "qkv"
    ]
    q, k = norm(q, "attn.norm_q", arch.qk_norm_eps), norm(k, "attn.norm_k", arch.qk_norm_eps)
    # This fixture uses zero RoPE phases.
    a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
    a = linear(a.transpose(1, 2).reshape(len(x), -1), "attn.to_out.0")
    x = (x.float() + gate[indices].float() * a.float()).to(x.dtype)
    z = (norm(x, "norm2", arch.norm_eps).float() * (1 + scale_ff[indices].float()) + shift_ff[indices].float()).to(
        x.dtype
    )
    up, gate_up = linear(z, "ff.net.0.proj").chunk(2, -1)
    ff = linear(up * F.silu(gate_up), "ff.net.2")
    return (x.float() + gate_ff[indices].float() * ff.float()).to(x.dtype)


def test_control_stream_matches_reference_and_never_injects_audio(cpu_layers):
    model = MiniMaxH3ControlNet(tiny_arch(), (0, 1))
    weights = original_weights(model.arch)
    model.load_weights(weights.items())
    # Platform norm/rope dispatch is selected at construction; force native CPU implementations.
    for module in model.modules():
        if hasattr(module, "forward_native"):
            module._forward_method = module.forward_native
    hidden = torch.randn(5, 8, generator=torch.Generator().manual_seed(3)).bfloat16()
    rows = torch.randn(2, 196, generator=torch.Generator().manual_seed(4))
    video, audio = torch.tensor([1, 3]), torch.tensor([2])
    indices = torch.tensor([1, 0, 2, 0, 1])
    # The official control forward rounds the shared embedding before SiLU.
    temb = torch.randn(1, 4, generator=torch.Generator().manual_seed(5)).bfloat16()
    rope = torch.cat((torch.ones(5, 3), torch.zeros(5, 3)), -1).bfloat16()
    actual = model(
        hidden,
        rows,
        video,
        audio,
        t_emb=temb,
        combined_indices=indices,
        rope_table=rope,
        cu_seqlens=torch.tensor([0, 5], dtype=torch.int32),
        max_seqlen=5,
        packed_total=5,
    )
    projected = F.linear(rows, weights["control_proj_in.weight"], weights["control_proj_in.bias"]).bfloat16()
    stream = hidden.index_copy(0, video, projected)
    stream = (
        F.linear(stream, weights["control_blocks.0.before_proj.weight"], weights["control_blocks.0.before_proj.bias"])
        + hidden
    )
    for i in range(2):
        prefix = f"control_blocks.{i}."
        stream = reference_block(stream, temb, indices, weights, prefix, model.arch)
        skip = F.linear(stream, weights[prefix + "after_proj.weight"], weights[prefix + "after_proj.bias"])
        skip[audio] = 0
        torch.testing.assert_close(actual[i], skip, atol=0.01, rtol=0.03)
        assert actual[i][audio].count_nonzero() == 0
        assert actual[i][0].count_nonzero() > 0  # text residual is deliberately retained
    # A different request cannot mutate the first request's skips or hidden state.
    before = {i: v.clone() for i, v in actual.items()}
    model(
        hidden,
        rows + 1,
        video,
        audio,
        t_emb=temb,
        combined_indices=indices,
        rope_table=rope,
        cu_seqlens=torch.tensor([0, 5], dtype=torch.int32),
        max_seqlen=5,
        packed_total=5,
    )
    for i in actual:
        assert torch.equal(actual[i], before[i])


def test_canvas_mask_channel_order_and_zero_control():
    seen = []

    def encode(pixels):
        seen.append(pixels.clone())
        return pixels.mean(1, keepdim=True).repeat(1, 24, 1, 1, 1)

    control = torch.ones(1, 3, 2, 2, 2) * 0.25
    source = torch.ones(1, 3, 2, 2, 2) * 0.75
    mask = torch.tensor([[[[[0.0, 1.0], [0.5, 0.51]]]]])
    kwargs = dict(height=2, width=2, num_frames=2, latent_shape=(2, 2, 2), encode=encode, device=torch.device("cpu"))
    rows = build_control_rows(control, source, mask, **kwargs)
    assert rows.shape == (2, 196)
    assert torch.all(rows[:, :96] == 0.25)
    assert rows[:, 96:100].tolist() == [[1, 0, 1, 0], [1, 0, 1, 0]]
    assert torch.equal(seen[-1][0, 0, 0], torch.tensor([[0.75, 0], [0.75, 0]]))
    no_mask = build_control_rows(control, None, None, **kwargs)
    assert no_mask[:, 96:].count_nonzero() == 0
    mask_only = build_control_rows(None, None, mask, **kwargs)
    assert mask_only[:, :96].count_nonzero() == 0
    assert seen[-1].count_nonzero() == 0
    with pytest.raises(ValueError, match="source requires mask"):
        build_control_rows(control, source, None, **kwargs)


def test_canvas_temporal_hold_and_truncate():
    pixels = torch.arange(2).reshape(1, 1, 2, 1, 1).float()
    assert fit_control_canvas(pixels, 1, 1, 4).flatten().tolist() == [0, 1, 1, 1]
    assert fit_control_canvas(pixels, 1, 1, 1).flatten().tolist() == [0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True, "1"])
def test_invalid_control_scale(value):
    with pytest.raises(ValueError, match="finite"):
        parse_control({"canny": {"control_path": "x", "control_context_scale": value}})


def test_control_mode_errors_and_no_control():
    assert parse_control({}) is None
    assert parse_control({"canny": {"control_path": "x", "control_context_scale": 0}})["control_context_scale"] == 0
    for extra in (
        {"canny": {"source_path": "x"}},
        {"canny": {"mask_path": "x"}},
        {"inpaint": {"source_path": "x"}},
        {"inpaint": {"mask_path": "x", "unknown": 1}},
        {"canny": {"control_path": "x"}, "depth": {"control_path": "x"}},
    ):
        with pytest.raises(ValueError):
            parse_control(extra)


def test_lossless_temporal_mask_sampling(tmp_path):
    import av
    import numpy as np

    path = tmp_path / "mask.mkv"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=12)
        stream.width = stream.height = 8
        stream.pix_fmt = "gray"
        for i in range(2):
            frame = av.VideoFrame.from_ndarray(np.full((8, 8), 255 * i, dtype=np.uint8), format="gray")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    pixels = load_control_pixels(str(path), height=8, width=8, num_frames=5, mask=True)
    assert pixels[0, 0, :, 0, 0].tolist() == [0, 0, 1, 1, 1]


def test_transformer_baseline_zero_strength_and_request_isolation(cpu_layers):
    arch = tiny_arch()
    config = SimpleNamespace(
        tf_model_config=vars(arch),
        parallel_config=SimpleNamespace(ulysses_degree=1, ring_degree=1),
        controlnet_model_path=None,
    )
    model = h3.MiniMaxH3DiTModel(config)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    model.controlnet = MiniMaxH3ControlNet(arch, (0, 1))
    model.controlnet.load_weights(original_weights(arch).items())
    for module in model.modules():
        if hasattr(module, "forward_native"):
            module._forward_method = module.forward_native
    kwargs = {
        "x": torch.randn(1, 5, 96),
        "audio_x": torch.randn(1, 5, 32),
        "img_position_ids": torch.zeros(1, 5, 3, dtype=torch.long),
        "unique_timesteps": torch.tensor([0.2]),
        "inverse_indices": torch.zeros(5, dtype=torch.long),
        "update_mask": torch.ones(2, dtype=torch.bool),
        "token_tags": torch.tensor([1, 0, 2, 0, 1]),
        "prompt_embeds": torch.randn(2, 8),
        "img_pos_info": {"position_ids": torch.tensor([1, 3])},
        "img_pos_for_infer_output_info": {"position_ids": torch.tensor([1, 3])},
        "audio_pos_info": {"position_ids": torch.tensor([2])},
        "text_pos_info": {"position_ids": torch.tensor([0, 4])},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 5], dtype=torch.int32), "max_seqlen_q": 5},
        "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 2], dtype=torch.int32), "max_seqlen_q": 2},
    }
    adaln_dtypes = []
    for module in model.modules():
        if isinstance(module, blocks.MiniMaxH3AdalnProj):
            module.register_forward_pre_hook(lambda _module, args: adaln_dtypes.append(args[0].dtype))
    before = model(**kwargs)
    assert adaln_dtypes == [torch.float32] * 3  # two main blocks plus final head
    adaln_dtypes.clear()
    rows = torch.randn(2, 196)
    with patch.object(model.controlnet, "forward", side_effect=AssertionError("Control must be bypassed")):
        zero = model(**kwargs, control_rows=rows, control_context_scale=0)
    assert adaln_dtypes == [torch.float32] * 3
    adaln_dtypes.clear()
    for expected, actual in zip(before, zero, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    controlled = model(**kwargs, control_rows=rows, control_context_scale=1)
    assert adaln_dtypes == [torch.bfloat16] * 5  # control + main blocks and final head
    adaln_dtypes.clear()
    assert not torch.equal(before[0], controlled[0])
    after = model(**kwargs)
    assert adaln_dtypes == [torch.float32] * 3
    for expected, actual in zip(before, after, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("rank", [0, 1])
def test_original_loader_shards_tp2_without_truncating_invalid_weights(cpu_layers, rank):
    arch = tiny_arch()
    weights = original_weights(arch)
    with ExitStack() as stack:
        for module in ("vllm.model_executor.layers.linear", "vllm.model_executor.parameter"):
            stack.enter_context(patch(module + ".get_tensor_model_parallel_world_size", return_value=2))
            stack.enter_context(patch(module + ".get_tensor_model_parallel_rank", return_value=rank))
        model = MiniMaxH3ControlNet(arch, (0, 1))
        model.load_weights(weights.items())
    qkv = torch.cat([weights[f"control_blocks.0.attn.to_{part}.weight"].chunk(2)[rank] for part in "qkv"])
    assert torch.equal(model.control_blocks[0].attn.qkv_proj.weight, qkv)
    up, gate = weights["control_blocks.0.ff.net.0.proj.weight"].chunk(2)
    assert torch.equal(model.control_blocks[0].mlp.fc1.weight, torch.cat((gate.chunk(2)[rank], up.chunk(2)[rank])))
    out = weights["control_blocks.0.attn.to_out.0.weight"].chunk(2, dim=1)[rank]
    assert torch.equal(model.control_blocks[0].attn.out_proj.weight, out)
    invalid = dict(weights)
    invalid["control_blocks.0.attn.to_q.weight"] = torch.zeros(32, 8)
    with pytest.raises(ValueError, match="shape"):
        model.load_weights(invalid.items())


def test_vae_control_uses_normalized_posterior_mode():
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    class MomentEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.input = None

        def encode_temporal(self, pixels):
            self.input = pixels
            # Arbitrary log variance must not enter the returned conditioning.
            return torch.cat((torch.full((1, 24, 2, 2, 2), 2.0), torch.full((1, 24, 2, 2, 2), 100.0)), 1)

    vae = object.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(vae)
    vae.model = MomentEncoder()
    vae.config_dict = {"latent_channels": 24, "latents_mean": [0.5] * 24, "latents_std": [2.0] * 24}
    pixels = torch.ones(1, 3, 5, 32, 32) * 0.5
    actual = vae.encode_control_latents(pixels)
    expected_pixels = (
        pixels - pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None, None]
    ) / pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None, None]
    torch.testing.assert_close(vae.model.input, expected_pixels)
    torch.testing.assert_close(actual, torch.full((1, 24, 2, 2, 2), 0.75))


@pytest.mark.parametrize(
    "offload",
    [
        {"diffusion_offload_config": {"mode": "module", "components": ["dit"]}},
        {"diffusion_offload_config": {"mode": "layer", "components": ["dit"]}},
        {"enable_cpu_offload": True},
    ],
)
def test_startup_rejects_public_and_legacy_offload(tmp_path, monkeypatch, offload):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    path = tmp_path / "control.safetensors"
    path.touch()
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    config = SimpleNamespace(
        model="/unused/FL2VA",
        task_type="fl2va",
        model_loaded={"text_encoder": True},
        parallel_config=SimpleNamespace(cfg_parallel_size=1, ulysses_degree=1, ring_degree=1, allgather_degree=1),
        controlnet_model_path=str(path),
        **offload,
    )
    with pytest.raises(ValueError, match="no cache/offload"):
        pipeline_module.MiniMaxH3Pipeline(od_config=config)


@pytest.mark.parametrize("partition", ["combined", "ref2va"])
def test_control_rejects_secondary_or_ref2va_partition_before_loading(tmp_path, monkeypatch, partition):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    path = tmp_path / "control.safetensors"
    path.touch()
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(pipeline_module, "_minimax_h3_partition_for_task", lambda *args: partition)
    config = SimpleNamespace(
        model="/unused/MiniMax-H3",
        task_type="auto",
        model_loaded={"text_encoder": True},
        parallel_config=SimpleNamespace(cfg_parallel_size=1, ulysses_degree=1, ring_degree=1, allgather_degree=1),
        controlnet_model_path=str(path),
    )
    with patch.object(pipeline_module, "_resolve_minimax_h3_model_root") as resolve_root:
        with pytest.raises(ValueError, match="FL2VA checkpoint partition"):
            pipeline_module.MiniMaxH3Pipeline(od_config=config)
        resolve_root.assert_not_called()


def test_startup_accepts_resident_default_policy(tmp_path, monkeypatch):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    path = tmp_path / "control.safetensors"
    path.touch()
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    config = SimpleNamespace(
        model="/unused/FL2VA",
        task_type="fl2va",
        model_loaded={"text_encoder": True},
        parallel_config=SimpleNamespace(cfg_parallel_size=1, ulysses_degree=1, ring_degree=1, allgather_degree=1),
        controlnet_model_path=str(path),
        diffusion_offload_config=None,
        revision=None,
    )
    with patch.object(pipeline_module, "_resolve_minimax_h3_model_root", side_effect=RuntimeError("guard accepted")):
        with pytest.raises(RuntimeError, match="guard accepted"):
            pipeline_module.MiniMaxH3Pipeline(od_config=config)


@pytest.mark.parametrize("native", [False, True])
def test_existing_turbo_targets_cannot_bind_control_branch(cpu_layers, native):
    import re

    from vllm_omni.diffusion.models.minimax_h3.lora import _TURBO_TARGET_PATTERN
    from vllm_omni.diffusion.models.minimax_h3.npu.lora import _NATIVE_TARGET_PATTERN

    pattern = _NATIVE_TARGET_PATTERN if native else _TURBO_TARGET_PATTERN
    model = MiniMaxH3ControlNet(tiny_arch(), (0, 1))
    assert re.search(pattern, "transformer.blocks.0.mlp.fc1")
    for name, _ in model.named_modules():
        full_name = f"transformer.controlnet.{name}"
        assert re.search(pattern, full_name) is None
        # The manager expands fused QKV names before matching too.
        for part in "qkv":
            assert re.search(pattern, full_name.replace("qkv_proj", f"to_{part}")) is None


@pytest.mark.parametrize(
    ("frame_pts", "expected"),
    [
        ([0, 1, 2, 3, 4, 5], [0, 40, 80, 120, 160, 200, 200]),
        ([0, 1, 3], [0, 40, 40, 80, 80, 80, 80]),
    ],
)
def test_millisecond_container_timestamps_preserve_24fps_and_vfr_hold(tmp_path, frame_pts, expected):
    from fractions import Fraction

    import av
    import numpy as np

    path = tmp_path / "quantized-timestamps.mkv"
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=24)
        stream.width = stream.height = 8
        stream.pix_fmt = "gray"
        for i, pts in enumerate(frame_pts):
            frame = av.VideoFrame.from_ndarray(np.full((8, 8), 40 * i, dtype=np.uint8), format="gray")
            frame.time_base = Fraction(1, 24)
            frame.pts = pts
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as source:
        assert source.streams.video[0].time_base == Fraction(1, 1000)
    pixels = load_control_pixels(str(path), height=8, width=8, num_frames=7)
    assert (pixels[0, 0, :, 0, 0] * 255).round().int().tolist() == expected
    truncated = load_control_pixels(str(path), height=8, width=8, num_frames=2)
    assert (truncated[0, 0, :, 0, 0] * 255).round().int().tolist() == expected[:2]


def test_control_decode_resizes_each_frame_before_accumulation(tmp_path, monkeypatch):
    import av
    import numpy as np

    path = tmp_path / "control.mkv"
    y, x = np.indices((48, 64))
    originals = [np.stack(((x * 3 + i * 11) % 256, y * 5, (x + y) * 2), axis=-1).astype(np.uint8) for i in range(3)]
    with av.open(str(path), "w") as output:
        stream = output.add_stream("ffv1", rate=24)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "bgr0"
        for pixels in originals:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    expected = F.interpolate(
        torch.from_numpy(np.stack(originals)).permute(0, 3, 1, 2).float() / 255,
        (6, 8),
        mode="bilinear",
        align_corners=False,
    )
    expected = torch.cat((expected, expected[-1:].expand(2, -1, -1, -1))).permute(1, 0, 2, 3)[None]
    interpolate = F.interpolate
    from_numpy = torch.from_numpy
    resized_shapes = []

    def resize_one_frame(value, *args, **kwargs):
        resized_shapes.append(tuple(value.shape))
        assert value.shape == (1, 3, 48, 64)
        assert value.device.type == "cpu"
        return interpolate(value, *args, **kwargs)

    def convert_one_frame(value):
        # A whole source-sized video must never be stacked for tensor conversion.
        assert value.ndim <= 3
        return from_numpy(value)

    monkeypatch.setattr(F, "interpolate", resize_one_frame)
    monkeypatch.setattr(torch, "from_numpy", convert_one_frame)
    actual = load_control_pixels(str(path), height=6, width=8, num_frames=5)
    assert len(resized_shapes) == 3
    assert actual.device.type == "cpu"
    assert actual.shape == (1, 3, 5, 6, 8)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("static", [True, False], ids=["static-mask", "temporal-mask"])
def test_control_mask_threshold_resize_threshold_boundary(tmp_path, static):
    import av
    import numpy as np
    from PIL import Image

    pixels = np.tile(np.array([102, 230, 255, 255, 0, 0, 230, 102], dtype=np.uint8), (8, 1))
    path = tmp_path / ("mask.png" if static else "mask.mkv")
    if static:
        Image.fromarray(pixels).save(path)
    else:
        with av.open(str(path), "w") as output:
            stream = output.add_stream("ffv1", rate=24)
            stream.width = stream.height = 8
            stream.pix_fmt = "gray"
            for frame_pixels in (pixels, pixels[:, ::-1].copy()):
                for packet in stream.encode(av.VideoFrame.from_ndarray(frame_pixels, format="gray")):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    actual = load_control_pixels(str(path), height=2, width=4, num_frames=4, mask=True)
    expected = torch.tensor([[0, 1, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
    if static:
        assert actual.shape == (1, 1, 1, 2, 4)
        torch.testing.assert_close(actual[0, 0, 0], expected)
    else:
        assert actual.shape == (1, 1, 4, 2, 4)
        torch.testing.assert_close(actual[0, 0, 0], expected)
        torch.testing.assert_close(actual[0, 0, 1:], expected.flip(-1).expand(3, -1, -1))
    # Raw grayscale resize would mark the mixed 102/230 pairs as regenerated;
    # the reference first binarizes them, then thresholds their 0.5 average off.
    assert set(actual.flatten().tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("degree", ["ulysses_degree", "ring_degree", "allgather_degree"])
def test_control_rejects_sequence_parallel_before_weight_loading(tmp_path, monkeypatch, degree):
    from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as pipeline_module

    path = tmp_path / "control.safetensors"
    path.touch()
    monkeypatch.setattr(pipeline_module, "get_local_device", lambda: torch.device("cpu"))
    parallel = SimpleNamespace(cfg_parallel_size=1, ulysses_degree=1, ring_degree=1, allgather_degree=1)
    setattr(parallel, degree, 2)
    config = SimpleNamespace(
        model="/unused/FL2VA",
        task_type="fl2va",
        model_loaded={"text_encoder": True},
        parallel_config=parallel,
        controlnet_model_path=str(path),
    )
    with patch.object(pipeline_module, "_resolve_minimax_h3_model_root") as resolve_root:
        with pytest.raises(ValueError, match="no cache/offload/SP"):
            pipeline_module.MiniMaxH3Pipeline(od_config=config)
        resolve_root.assert_not_called()


def test_control_rows_and_padded_sequence_survive_denoise_preparation():
    from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    control_rows = torch.ones(12, 196)
    context = dict(
        task="t2va",
        text_embeddings=torch.ones(7, 2),
        text_tags=torch.ones(7, dtype=torch.long),
        seed=0,
        latent_t=2,
        latent_h=4,
        latent_w=6,
        audio_t=3,
        num_frames=5,
        num_steps=2,
        video_shift=12.0,
        audio_shift=3.0,
        base_schedule=None,
        visual_condition=None,
        visual_condition_shape=None,
        audio_condition=None,
        ref_audio_t=None,
        ref_blocks=None,
        visual_condition_shapes=None,
        audio_condition_lengths=None,
        keyframe_frame_indices=None,
        control_rows=control_rows,
        control_context_scale=0.75,
        pad_seq_len=192,
    )
    inputs = pipeline._build_denoise_inputs(**pipeline._denoise_kwargs(context))
    branch = inputs["branch"]
    assert branch.seq_len == 192
    assert branch.used_len < branch.seq_len
    assert branch.static_kwargs["control_rows"] is control_rows
    assert branch.static_kwargs["control_context_scale"] == 0.75
    assert inputs["video_rows"].shape == (12, 96)
