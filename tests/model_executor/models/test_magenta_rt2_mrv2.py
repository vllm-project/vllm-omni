# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Surgical unit tests for Magenta RealTime 2 MRv2 + full-duplex integration."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch
from vllm.config import CompilationConfig, LoadConfig, VllmConfig
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.models.magenta_rt2.conditioning import (
    ENCODER_NUM_CHANNELS,
    MagentaRT2ConditioningEncoder,
    build_conditioning_tokens,
    discretize_cfg,
    regular_channel_vocab_sizes,
)
from vllm_omni.model_executor.models.magenta_rt2.duplex.plugin import MagentaRT2DuplexPlugin
from vllm_omni.model_executor.models.magenta_rt2.magenta_rt2_code2wav import SpectroStreamCode2Wav
from vllm_omni.model_executor.models.magenta_rt2.magenta_rt2_talker import (
    MagentaRT2ForConditionalGeneration,
    MagentaRT2ModelState,
)
from vllm_omni.model_executor.models.magenta_rt2.spectrostream import (
    CausalConvTranspose2d,
    SpectroStreamModel,
)
from vllm_omni.model_executor.models.magenta_rt2.transformer import SlotRingBufferAttention
from vllm_omni.model_executor.models.registry import OmniModelRegistry
from vllm_omni.model_executor.stage_input_processors.magenta_rt2 import (
    talker2code2wav_async_chunk,
    talker2code2wav_full_payload,
    talker2code2wav_token_only,
)
from vllm_omni.transformers_utils.configs.magenta_rt2 import MagentaRT2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_mrt2_config() -> MagentaRT2Config:
    return MagentaRT2Config(
        variant="mrt2_small",
        encoder_dim=32,
        temporal_num_layers=2,
        temporal_hidden_size=32,
        temporal_ffn_dim=64,
        temporal_num_heads=4,
        temporal_head_dim=8,
        temporal_self_past_horizon=6,
        temporal_cross_past_horizon=6,
        temporal_num_sinks=1,
        depth_num_layers=2,
        depth_hidden_size=32,
        depth_ffn_dim=64,
        depth_num_heads=4,
        depth_head_dim=8,
        depth_past_horizon=12,
        num_codebooks=12,
        codebook_size=1024,
        num_extra_tokens=6,
        ss_num_quantizers=16,
        ss_active_quantizers=12,
        depthformer_cuda_graphs=False,
        spectrostream_cuda_graphs=False,
    )


def _tiny_vllm_config(cfg: MagentaRT2Config, *, async_chunk: bool = True, max_num_seqs: int = 4) -> VllmConfig:
    vllm_cfg = object.__new__(VllmConfig)
    vllm_cfg.model_config = SimpleNamespace(
        model="google/magenta-realtime-2",
        hf_config=cfg,
        dtype=torch.float32,
        max_model_len=512,
        uses_mrope=False,
        async_chunk=async_chunk,
        duplex_max_sessions=max_num_seqs,
        has_no_ops=True,
        multimodal_config=None,
        enable_prompt_embeds=False,
        enforce_eager=True,
        is_multimodal_model=False,
        get_inputs_embeds_size=lambda: cfg.temporal_hidden_size,
        get_vocab_size=lambda: cfg.vocab_size,
    )
    vllm_cfg.scheduler_config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=64,
    )
    vllm_cfg.device_config = SimpleNamespace(device=torch.device("cpu"))
    vllm_cfg.compilation_config = CompilationConfig()
    vllm_cfg.load_config = LoadConfig()
    vllm_cfg.cache_config = SimpleNamespace(block_size=16)
    vllm_cfg.parallel_config = SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1)
    vllm_cfg.lora_config = None
    vllm_cfg.quant_config = None
    vllm_cfg.speculative_config = None
    vllm_cfg.kv_transfer_config = None
    vllm_cfg.ec_transfer_config = None
    vllm_cfg.observability_config = None
    return vllm_cfg


def _build_synthetic_flax_checkpoint(cfg: MagentaRT2Config, path: Path) -> dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(0)

    def t(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.randn(shape, generator=rng, dtype=torch.float32).to(dtype=dtype) * 0.02

    tensors: dict[str, torch.Tensor] = {}
    ed = cfg.encoder_dim
    td = cfg.temporal_hidden_size
    th = cfg.temporal_num_heads
    thd = cfg.temporal_head_dim
    tffn = cfg.temporal_ffn_dim

    # Encoder
    tensors[
        "params/depthformer/encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/mulan_dequantizer/embedding"
    ] = t((12372, cfg.mulan_dim))
    tensors[
        "params/depthformer/encoder/body/layers_1/branched_mulan_embedder/mulan_embedder/depth_input_adapter/kernel"
    ] = t((cfg.mulan_dim, ed))
    tensors["params/depthformer/encoder/body/layers_1/branched_regular_embedder/regular_embedder/embedding"] = t(
        (1536, ed)
    )
    tensors["params/depthformer/encoder/body/encoder_ln/scale"] = torch.ones(ed, dtype=torch.bfloat16)
    tensors["params/depthformer/encoder/body/encoder_ln/bias"] = torch.zeros(ed, dtype=torch.bfloat16)

    # Decoder embedding + temporal blocks
    tensors["params/depthformer/decoder/decoder_embedding/embedding/embedding"] = t((cfg.vocab_size, td))
    for i in range(cfg.temporal_num_layers):
        p = f"params/depthformer/decoder/temporal_body/transformer/x_layers_{i}"
        tensors[f"{p}/self_attention/pre_norm/scale"] = torch.ones(td, dtype=torch.bfloat16)
        tensors[f"{p}/self_attention/post_norm/scale"] = torch.ones(td, dtype=torch.bfloat16)
        tensors[f"{p}/self_attention/output_projection/kernel"] = t((td, th, thd))
        tensors[f"{p}/self_attention/attention/query_key_value_projection/kernel"] = t((td, 3, th, thd))
        tensors[f"{p}/self_attention/attention/per_dim_scale"] = torch.zeros(thd, dtype=torch.bfloat16)
        tensors[f"{p}/self_attention/attention/sink_key_embeddings"] = t((cfg.temporal_num_sinks, th, thd))
        tensors[f"{p}/self_attention/attention/sink_value_embeddings"] = t((cfg.temporal_num_sinks, th, thd))

        tensors[f"{p}/cross_attention/pre_norm/scale"] = torch.ones(td, dtype=torch.bfloat16)
        tensors[f"{p}/cross_attention/post_norm/scale"] = torch.ones(td, dtype=torch.bfloat16)
        tensors[f"{p}/cross_attention/output_projection/kernel"] = t((td, th, thd))
        tensors[f"{p}/cross_attention/attention/query_projection/kernel"] = t((td, th, thd))
        tensors[f"{p}/cross_attention/attention/key_value_projection/kernel"] = t((ed, 2, th, thd))
        tensors[f"{p}/cross_attention/attention/per_dim_scale"] = torch.zeros(thd, dtype=torch.bfloat16)
        tensors[f"{p}/cross_attention/attention/sink_key_embeddings"] = t((cfg.temporal_num_sinks, th, thd))
        tensors[f"{p}/cross_attention/attention/sink_value_embeddings"] = t((cfg.temporal_num_sinks, th, thd))

        tensors[f"{p}/ffn/pre_norm/scale"] = torch.ones(td, dtype=torch.bfloat16)
        tensors[f"{p}/ffn/post_norm/scale"] = torch.ones(td, dtype=torch.bfloat16)
        tensors[f"{p}/ffn/ffn_layer1/kernel"] = t((td, tffn))
        tensors[f"{p}/ffn/ffn_layer1/bias"] = torch.zeros(tffn, dtype=torch.bfloat16)
        tensors[f"{p}/ffn/ffn_layer2/kernel"] = t((tffn, td))
        tensors[f"{p}/ffn/ffn_layer2/bias"] = torch.zeros(td, dtype=torch.bfloat16)

    # Depthformer
    dd = cfg.depth_hidden_size
    dh = cfg.depth_num_heads
    dhd = cfg.depth_head_dim
    dffn = cfg.depth_ffn_dim
    for i in range(cfg.depth_num_layers):
        p = f"params/depthformer/decoder/depth_body/transformer/x_layers_{i}"
        tensors[f"{p}/self_attention/pre_norm/scale"] = torch.ones(dd, dtype=torch.bfloat16)
        tensors[f"{p}/self_attention/post_norm/scale"] = torch.ones(dd, dtype=torch.bfloat16)
        tensors[f"{p}/self_attention/output_projection/kernel"] = t((dd, dh, dhd))
        tensors[f"{p}/self_attention/attention/query_key_value_projection/kernel"] = t((dd, 3, dh, dhd))
        tensors[f"{p}/self_attention/attention/per_dim_scale"] = torch.zeros(dhd, dtype=torch.bfloat16)
        tensors[f"{p}/ffn/pre_norm/scale"] = torch.ones(dd, dtype=torch.bfloat16)
        tensors[f"{p}/ffn/post_norm/scale"] = torch.ones(dd, dtype=torch.bfloat16)
        tensors[f"{p}/ffn/ffn_layer1/kernel"] = t((dd, dffn))
        tensors[f"{p}/ffn/ffn_layer1/bias"] = torch.zeros(dffn, dtype=torch.bfloat16)
        tensors[f"{p}/ffn/ffn_layer2/kernel"] = t((dffn, dd))
        tensors[f"{p}/ffn/ffn_layer2/bias"] = torch.zeros(dd, dtype=torch.bfloat16)

    tensors["params/depthformer/decoder/depth_body/final_ln/scale"] = torch.ones(dd, dtype=torch.bfloat16)
    tensors["params/depthformer/decoder/depth_body/final_ln/bias"] = torch.zeros(dd, dtype=torch.bfloat16)
    tensors["params/depthformer/decoder/depth_body/to_logits/kernel"] = t((dd, cfg.vocab_size))
    tensors["params/depthformer/decoder/depth_body/to_logits/bias"] = torch.zeros(cfg.vocab_size, dtype=torch.bfloat16)

    # Soundstream / SpectroStream
    tensors["params/soundstream/quantizer/embedding"] = t(
        (cfg.ss_num_quantizers, cfg.ss_codebook_size, cfg.ss_input_dim), dtype=torch.float32
    )
    sd = "params/soundstream/decoder"
    tensors[f"{sd}/input_layer/conv1x1_first/conv/kernel"] = t((1, 1, 256, 2560), dtype=torch.float32)
    tensors[f"{sd}/input_layer/conv1x1_first/conv/bias"] = torch.zeros(2560, dtype=torch.float32)
    tensors[f"{sd}/input_layer/shortcut_layer/conv1x1_b1/conv/kernel"] = t((1, 1, 256, 2560), dtype=torch.float32)
    tensors[f"{sd}/input_layer/shortcut_layer/conv1x1_b1/conv/bias"] = torch.zeros(2560, dtype=torch.float32)
    tensors[f"{sd}/input_layer/shortcut_layer/conv1x1_b2/conv/kernel"] = t((1, 1, 2560, 2560), dtype=torch.float32)
    tensors[f"{sd}/input_layer/shortcut_layer/conv1x1_b2/conv/bias"] = torch.zeros(2560, dtype=torch.float32)

    tensors[f"{sd}/input_layers_residual_unit/conv2d_3x3_a/conv/kernel"] = t((3, 3, 512, 512), dtype=torch.float32)
    tensors[f"{sd}/input_layers_residual_unit/conv2d_3x3_a/conv/bias"] = torch.zeros(512, dtype=torch.float32)
    tensors[f"{sd}/input_layers_residual_unit/conv2d_3x3/conv/kernel"] = t((3, 3, 512, 512), dtype=torch.float32)
    tensors[f"{sd}/input_layers_residual_unit/conv2d_3x3/conv/bias"] = torch.zeros(512, dtype=torch.float32)

    specs = [
        ("decoder_0", (4, 3), 512, 1024, True),
        ("decoder_1", (4, 4), 512, 256, True),
        ("decoder_2", (3, 4), 256, 256, False),
        ("decoder_3", (3, 4), 256, 256, False),
        ("decoder_4", (3, 6), 256, 128, True),
        ("decoder_5", (3, 4), 128, 128, False),
        ("decoder_6", (3, 4), 128, 64, True),
    ]
    for name, k_t, cin, cout, has_sc in specs:
        tensors[f"{sd}/{name}/conv2dtranspose/conv/kernel"] = t((k_t[0], k_t[1], cin, cout), dtype=torch.float32)
        tensors[f"{sd}/{name}/conv2dtranspose/conv/bias"] = torch.zeros(cout, dtype=torch.float32)
        tensors[f"{sd}/{name}/conv2d_0/conv/kernel"] = t((3, 3, cout, cout), dtype=torch.float32)
        tensors[f"{sd}/{name}/conv2d_0/conv/bias"] = torch.zeros(cout, dtype=torch.float32)
        if has_sc:
            tensors[f"{sd}/{name}/shortcut_layer/conv1x1/conv/kernel"] = t((1, 1, cin, cout), dtype=torch.float32)
            tensors[f"{sd}/{name}/shortcut_layer/conv1x1/conv/bias"] = torch.zeros(cout, dtype=torch.float32)
    tensors[f"{sd}/output_layer/base_conv_last/conv/kernel"] = t((7, 7, 64, 2), dtype=torch.float32)
    tensors[f"{sd}/output_layer/base_conv_last/conv/bias"] = torch.zeros(2, dtype=torch.float32)

    safetensors.torch.save_file(tensors, str(path))
    return tensors


def test_registry_pipeline_and_deploy_yaml() -> None:
    cfg = MagentaRT2Config()
    assert cfg.model_type == "magenta_realtime"
    assert cfg.architectures == ["MagentaRT2ForConditionalGeneration"]
    assert cfg.vocab_size == 6 + 12 * 1024
    assert cfg.samples_per_frame == 1920

    assert "MagentaRT2ForConditionalGeneration" in OmniModelRegistry.get_supported_archs()
    assert "SpectroStreamCode2Wav" in OmniModelRegistry.get_supported_archs()
    pipe = resolve_pipeline_config("magenta_realtime", cfg)
    assert pipe is not None
    assert pipe.stages[0].supports_native_mrv2_data_plane
    assert pipe.stages[0].supports_duplex_mrv2
    assert pipe.stages[1].retains_state_across_chunks
    assert pipe.stages[1].supports_duplex_mrv2

    deploy_path = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy" / "magenta_rt2_mrv2.yaml"
    deploy = load_deploy_config(str(deploy_path))
    assert deploy.session_mode == "duplex"
    assert deploy.stages[0].model_runner == "v2"
    assert deploy.stages[1].model_runner == "v2"
    stage_configs = merge_pipeline_deploy(pipe, deploy)
    assert len(stage_configs) == 2
    assert stage_configs[0].yaml_engine_args["use_v2_model_runner"] is True
    assert stage_configs[1].yaml_engine_args["use_v2_model_runner"] is True


def test_conditioning_tokens_and_encoder() -> None:
    assert sum(regular_channel_vocab_sizes()) == 1526
    assert discretize_cfg(-1.0, 0.2, 40) == 0
    assert discretize_cfg(3.0, 0.2, 40) == 20
    assert discretize_cfg(7.0, 1.0, 8) == 8

    toks = build_conditioning_tokens(cfg_scales={"musiccoca": 3.0, "notes": 1.0, "drums": 2.0})
    assert toks.shape == (ENCODER_NUM_CHANNELS,)
    # All masked channels (-1 + 7 = 6)
    assert np.all(toks[:12] == 6)
    assert np.all(toks[12:141] == 6)
    assert toks[141] == 20 + 7
    assert toks[142] == 10 + 7
    assert toks[143] == 3 + 7

    enc = MagentaRT2ConditioningEncoder(_tiny_mrt2_config())
    out = enc(torch.from_numpy(toks).unsqueeze(0))
    assert out.shape == (1, 32)
    assert torch.isfinite(out).all()


def test_slot_ring_attention_matches_sequence_and_isolates_slots() -> None:
    torch.manual_seed(0)
    attn = SlotRingBufferAttention(
        hidden_size=32,
        num_heads=4,
        head_dim=8,
        max_past_horizon=4,
        num_sinks=1,
        max_slots=3,
        dtype=torch.float32,
    )
    with torch.no_grad():
        attn.per_dim_scale.uniform_(-0.3, 0.3)
        attn.sink_key_embeddings.normal_()
        attn.sink_value_embeddings.normal_()

    x_seq = torch.randn(1, 7, 32, dtype=torch.float32)
    ref = attn.forward_sequence(x_seq)

    cap = attn.ring_cap
    step_outs = []
    for t in range(7):
        wp = torch.tensor([t % cap], dtype=torch.long)
        phys = torch.arange(cap, dtype=torch.long)
        dist = (t % cap - phys + cap) % cap
        t_pos = t - dist
        mask = ((t_pos >= 0) & (dist < cap)).unsqueeze(0)
        y_t = attn(x_seq[:, t, :], slot_ids=torch.tensor([1]), write_pos=wp, attn_mask=mask)
        step_outs.append(y_t)
    step_seq = torch.stack(step_outs, dim=1)
    assert torch.allclose(step_seq, ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(attn.ring_k[0], torch.zeros_like(attn.ring_k[0]))


def test_talker_flax_load_and_mrv2_state(tmp_path: Path) -> None:
    cfg = _tiny_mrt2_config()
    vcfg = _tiny_vllm_config(cfg)
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    ckpt = ckpt_dir / "mrt2_small.safetensors"
    tensors = _build_synthetic_flax_checkpoint(cfg, ckpt)

    vcfg_dir = _tiny_vllm_config(cfg)
    vcfg_dir.model_config.model = str(tmp_path)
    model = MagentaRT2ForConditionalGeneration(vllm_config=vcfg_dir)
    assert model.allow_patterns_overrides == ["checkpoints/mrt2_small.safetensors"]
    loaded = model.load_weights(tensors.items())
    assert set(model.state_dict().keys()).issubset(loaded)

    state = MagentaRT2ForConditionalGeneration.create_mrv2_model_state(model, vcfg, None, torch.device("cpu"))
    assert isinstance(state, MagentaRT2ModelState)

    # Simulate 2 concurrent active requests via InputBatch-like structure.
    idx_req0, idx_req1 = 0, 1
    state.intermediate_buffer.buffers[idx_req0] = {
        "req_id": "req-0",
        "duplex": {
            "runtime_config": {
                "cfg_scales": {"musiccoca": 3.0, "notes": 1.0, "drums": 1.0},
                "mulan_tokens": list(range(12)),
                "temperature": 0.0,
                "top_k": 1,
            }
        },
    }
    state.intermediate_buffer.buffers[idx_req1] = {
        "req_id": "req-1",
        "duplex": {"runtime_config": {"temperature": 1.0, "top_k": 8}},
    }
    state.intermediate_buffer.req_id_to_index["req-0"] = idx_req0
    state.intermediate_buffer.req_id_to_index["req-1"] = idx_req1

    fake_batch = SimpleNamespace(
        num_reqs=2,
        num_tokens=2,
        num_tokens_after_padding=4,
        req_ids=["req-0", "req-1"],
        idx_mapping_np=np.array([idx_req0, idx_req1], dtype=np.int32),
        query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
        num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
    )
    fake_req_states = SimpleNamespace(prompt_len=np.array([1, 1], dtype=np.int32))
    state.prepare_inputs(fake_batch, fake_req_states)  # type: ignore[arg-type]

    h = model(input_ids=torch.zeros(4, dtype=torch.long))
    assert h.shape == (4, 32)
    state.run_postprocess(h, fake_batch)  # type: ignore[arg-type]
    _, mm = state.postprocess_model_output(h, fake_batch, fake_req_states)  # type: ignore[arg-type]
    codes = mm["codes"]["audio"]
    assert codes.shape == (2, 12)
    for q in range(12):
        lo = 6 + q * 1024
        hi = lo + 1024
        assert bool(((codes[:, q] >= lo) & (codes[:, q] < hi)).all())

    sampler_out = state.sample_determined_tokens(fake_batch, None)  # type: ignore[arg-type]
    assert sampler_out is not None
    assert sampler_out.sampled_token_ids.shape == (2, 1)

    state.remove_request("req-0")
    assert "req-0" not in state.req_to_slot
    state.remove_request("req-1")
    assert "req-1" not in state.req_to_slot

    # Per-request seed reproducibility independent of batch order / unseeded peers.
    for req_a, req_b, order in (("s-a", "s-b", 0), ("s-b2", "s-a2", 1)):
        state.add_request(
            0,
            SimpleNamespace(  # type: ignore[arg-type]
                req_id=req_a,
                mm_features=None,
                sampling_params=SamplingParams(seed=42, temperature=1.1, top_k=8),
            ),
        )
        state.add_request(
            1,
            SimpleNamespace(  # type: ignore[arg-type]
                req_id=req_b,
                mm_features=None,
                sampling_params=SamplingParams(seed=99 if order == 0 else None, temperature=1.1, top_k=8),
            ),
        )
        state.intermediate_buffer.buffers[0] = {"req_id": req_a}
        state.intermediate_buffer.buffers[1] = {"req_id": req_b}
        b = SimpleNamespace(
            num_reqs=2,
            num_tokens=2,
            num_tokens_after_padding=2,
            req_ids=[req_a, req_b] if order == 0 else [req_b, req_a],
            idx_mapping_np=np.array([0, 1] if order == 0 else [1, 0], dtype=np.int32),
            query_start_loc_np=np.array([0, 1, 2], dtype=np.int32),
            num_scheduled_tokens=np.array([1, 1], dtype=np.int32),
        )
        state.prepare_inputs(b, fake_req_states)  # type: ignore[arg-type]
        h_s = model(input_ids=torch.zeros(2, dtype=torch.long))
        state.run_postprocess(h_s, b)  # type: ignore[arg-type]
        _, mm_s = state.postprocess_model_output(h_s, b, fake_req_states)  # type: ignore[arg-type]
        picked = mm_s["codes"]["audio"][0 if order == 0 else 1].clone()
        if order == 0:
            first_seeded = picked
        else:
            assert torch.equal(picked, first_seeded)
        state.on_requests_finished({req_a, req_b})


def test_spectrostream_streaming_parity_and_code2wav(tmp_path: Path) -> None:
    cfg = _tiny_mrt2_config()
    vcfg = _tiny_vllm_config(cfg)
    ckpt = tmp_path / "mrt2_small.safetensors"
    tensors = _build_synthetic_flax_checkpoint(cfg, ckpt)

    # Verify JAX Conv2DTranspose flip matches Conv2dTranspose spatial geometry.
    tconv = CausalConvTranspose2d(4, 4, (4, 3), strides=(2, 2))
    assert tconv.ola_buf == 2

    code2wav = SpectroStreamCode2Wav(vllm_config=vcfg)
    loaded = code2wav.load_weights(tensors.items())
    assert set(code2wav.state_dict().keys()).issubset(loaded)

    codec: SpectroStreamModel = code2wav.spectrostream
    rng = torch.Generator().manual_seed(7)
    codes = torch.randint(6, 6 + 1024, (1, 4, 12), generator=rng, dtype=torch.long)
    # Use global per-codebook range for realism
    for q in range(12):
        codes[:, :, q] = 6 + q * 1024 + (codes[:, :, q] % 1024)

    batch_wav = codec.forward(codes)
    # forward() drops first 4 STFT steps (1 codec frame), yielding 3 frames * 1920 samples.
    assert batch_wav.shape == (1, 3 * 1920, 2)

    st = codec.init_state(1, device=torch.device("cpu"))
    step_chunks = [codec.step(codes[:, f : f + 1, :], st) for f in range(4)]
    # Frame 0 is zeroed by Lookahead(4) countdown.
    assert torch.allclose(step_chunks[0], torch.zeros_like(step_chunks[0]))
    stream_tail = torch.cat(step_chunks[1:], dim=1)
    assert torch.allclose(stream_tail, batch_wav, atol=1e-4, rtol=1e-4)

    # Two concurrent Stage-1 requests: verify folded-row reset and inactive-row preservation.
    standalone_s1_step0 = (
        code2wav.forward(
            input_ids=codes[0, 1],
            seq_token_counts=[12],
            request_ids=["solo-s1"],
        )
        .multimodal_outputs["model_outputs"][0]
        .clone()
    )
    standalone_s1_step1 = (
        code2wav.forward(
            input_ids=codes[0, 2],
            seq_token_counts=[12],
            request_ids=["solo-s1"],
        )
        .multimodal_outputs["model_outputs"][0]
        .clone()
    )
    code2wav.on_requests_finished(["solo-s1"])

    req_codes = torch.cat([codes[0, 0], codes[0, 1]], dim=0)  # 24 tokens (1 frame each for 2 reqs)
    out = code2wav.forward(
        input_ids=req_codes,
        seq_token_counts=[12, 12],
        request_ids=["s-0", "s-1"],
    )
    assert len(out.multimodal_outputs["model_outputs"]) == 2
    assert out.multimodal_outputs["model_outputs"][0].shape == (1920, 2)
    assert torch.allclose(out.multimodal_outputs["model_outputs"][1], standalone_s1_step0, atol=1e-5, rtol=1e-5)

    # Tick where only s-0 runs: s-1 must keep its exact state untouched.
    _ = code2wav.forward(input_ids=codes[0, 0], seq_token_counts=[12], request_ids=["s-0"])
    out_s1_next = code2wav.forward(input_ids=codes[0, 2], seq_token_counts=[12], request_ids=["s-1"])
    assert torch.allclose(out_s1_next.multimodal_outputs["model_outputs"][0], standalone_s1_step1, atol=1e-5, rtol=1e-5)

    code2wav.on_requests_finished(["s-0", "s-1"])
    assert code2wav._request_rows == {}
    if code2wav._codec_state is not None and code2wav._codec_state.out_conv_state is not None:
        assert torch.allclose(
            code2wav._codec_state.out_conv_state,
            torch.zeros_like(code2wav._codec_state.out_conv_state),
        )

    # bfloat16 conv state matches bfloat16 codec weights without dtype mismatch.
    code2wav.spectrostream.bfloat16()
    bf16_out = code2wav.forward(input_ids=codes[0, 0], seq_token_counts=[12], request_ids=["bf16-s"])
    assert bf16_out.multimodal_outputs["model_outputs"][0].shape == (1920, 2)
    assert code2wav._codec_state is not None and code2wav._codec_state.out_conv_state is not None
    assert code2wav._codec_state.out_conv_state.dtype == torch.bfloat16
    code2wav.on_requests_finished(["bf16-s"])


def test_stage_input_processors_and_duplex_plugin() -> None:
    req = SimpleNamespace(
        request_id="req-proc",
        resumable=True,
        additional_information={"codes": {"audio": torch.arange(24).reshape(2, 12)}},
    )
    tm = SimpleNamespace(connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 1}}))
    chunk_pl = talker2code2wav_async_chunk(
        tm, {"codes": {"audio": torch.arange(12).reshape(1, 12)}}, req, is_finished=True
    )
    assert chunk_pl is not None
    assert chunk_pl.codes.audio.shape == (12,)
    full_pl = talker2code2wav_full_payload(request=req)
    assert full_pl.codes.audio.shape == (24,)

    fake_talker_out = SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(cumulative_token_ids=[0, 1, 2], token_ids=[0, 1, 2])],
    )
    tok_prompts = talker2code2wav_token_only([fake_talker_out])
    assert len(tok_prompts) == 1
    assert len(tok_prompts[0]["prompt_token_ids"]) == 24

    # Duplex plugin + stereo data plane
    encoded_calls: list[tuple[tuple[int, ...], int]] = []

    def fake_encode(audio: object, sr: int, fmt: str, speed: float | None) -> str | None:
        del fmt, speed
        if audio is None:
            return None
        arr = np.asarray(audio)
        encoded_calls.append((arr.shape, sr))
        return "AQID"

    plugin = MagentaRT2DuplexPlugin(fake_encode)
    caps = plugin.capabilities(max_sessions=4)
    assert caps.chunk_period_ms == 40
    assert caps.supports_multi_session

    dp_out = list(
        plugin.data_plane.project(
            {
                "data_plane_outputs": [
                    SimpleNamespace(
                        request_id="req-stereo",
                        multimodal_output={"audio": torch.zeros((1920, 2), dtype=torch.float32), "sr": 48000},
                    )
                ]
            },
            context=DuplexDataPlaneContext(response_format="wav", modalities=("audio",)),
        )
    )
    assert len(dp_out) == 1
    assert dp_out[0]["sample_rate_hz"] == 48000
    assert dp_out[0]["audio_duration_ms"] == 40
    assert encoded_calls == [((1920, 2), 48000)]

    sp0, sp1 = plugin.configure_sampling_params(
        runtime_config={"temperature": 1.1, "top_k": 32},
        defaults=(SamplingParams(temperature=1.3), SamplingParams(output_kind=RequestOutputKind.FINAL_ONLY)),
    )
    assert isinstance(sp0, SamplingParams) and sp0.max_tokens == 1 and sp0.temperature == 1.1
    assert isinstance(sp1, SamplingParams) and sp1.output_kind == RequestOutputKind.DELTA

    silence_b64 = base64.b64encode(np.zeros(1920, dtype=np.float32).tobytes()).decode("ascii")
    plan = plugin.plan_append(
        request_id="req-stereo",
        fence=DuplexFence(session_id="sess-1", epoch=0),
        session_config={},
        runtime_config={"cfg_scales": {"musiccoca": 4.0, "notes": 1.0, "drums": 1.0}},
        seq=1,
        turn_seq=1,
        payload={"format": "pcm_f32le", "sample_rate_hz": 48000, "audio": silence_b64},
        final=False,
        sampling_params=None,
    )
    assert plan.prompt["prompt_token_ids"] == [0]
    assert plan.prompt["model_intermediate_buffer"]["duplex"]["runtime_config"]["cfg_scales"]["musiccoca"] == 4.0

    # Live session update allowed for cfg_scales
    cfg = DuplexSessionConfig(extra_body={"cfg_scales": {"musiccoca": 5.0}})
    plugin.validate_client_extra_body(cfg.extra_body)
    updated = plugin.runtime_config_for_update(cfg, {"cfg_scales": {"musiccoca": 3.0}})
    assert updated["cfg_scales"]["musiccoca"] == 5.0
