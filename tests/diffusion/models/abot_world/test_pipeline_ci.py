# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small CPU tests for ABot-World tensor geometry and checkpoint mapping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.abot_world.abot_world_transformer import (
    ABotSimpleAdapter,
    ABotWorldCausalTransformer3DModel,
)
from vllm_omni.diffusion.models.abot_world.actions import parse_abot_camera_action_script
from vllm_omni.diffusion.models.abot_world.pipeline_abot_world import ABotWorldCausalPipeline, _wan_decode_cache_bytes
from vllm_omni.diffusion.models.abot_world.taew2_2 import TAEW2Decoder
from vllm_omni.experimental.ar_diffusion.streaming_decode import StreamingDecodeState, WanStreamingDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _tiny_transformer() -> ABotWorldCausalTransformer3DModel:
    return ABotWorldCausalTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=1,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=8,
        ffn_dim=16,
        num_layers=1,
        downscale_factor_control_adapter=2,
    )


def test_action_adapter_preserves_patch_token_geometry() -> None:
    adapter = ABotSimpleAdapter(dim=8, downscale_factor=2, control_in_dim=2)
    tokens = torch.zeros(1, 3 * 4 * 4, 8)
    actions = torch.zeros(1, 2, 3, 16, 16)

    output = adapter(tokens, actions, num_frames=3, spatial_tokens=16)

    assert output.shape == tokens.shape


def test_timestep_projection_expands_framewise_values_to_tokens() -> None:
    model = _tiny_transformer()
    temb, projection = model._timestep_embeddings(
        torch.tensor([[0.0, 500.0, 500.0]]),
        batch_size=1,
        frames=3,
        dtype=torch.float32,
    )

    assert temb.shape == (1, 3, 8)
    assert projection.shape == (1, 3, 6, 8)
    assert not torch.equal(projection[:, 0], projection[:, 1])


def test_temporal_rope_supports_the_35th_three_frame_tick() -> None:
    model = _tiny_transformer()

    cosine, sine = model._rotary_embedding(
        frames=3,
        height=2,
        width=2,
        start_frame=34 * 3,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert cosine.shape == sine.shape == (12, 4)


def test_unknown_checkpoint_weight_is_rejected() -> None:
    model = _tiny_transformer()

    with pytest.raises(KeyError, match="Unexpected ABot model weight"):
        model.load_weights([("model.not_a_real_parameter", torch.zeros(1))])


def test_official_generator_prefix_is_normalized() -> None:
    model = _tiny_transformer()
    weight = torch.ones_like(model.blocks[0].norm3.weight)

    loaded = model.load_weights([("generator.model.blocks.0.norm3.weight", weight)])

    assert "blocks.0.norm3.weight" in loaded
    assert torch.equal(model.blocks[0].norm3.weight, weight)


def test_realtime_vae_decode_keeps_state_across_chunks() -> None:
    class Decoder:
        dtype = torch.float32

        def new_decode_state(self, session_id):
            return StreamingDecodeState(session_id, [])

        def decode_chunk(self, latents, state):
            history = state.frames_decoded
            frames = 9 if not state.started else 12
            output = latents.new_full((*latents.shape[:2], frames, *latents.shape[3:]), history / 10)
            state.frames_decoded += latents.shape[2]
            return output

    vae = Decoder()
    pipeline = SimpleNamespace(vae=vae, _streaming_decoder=vae, _vae_backend="taew2_2")
    chunk = torch.zeros(1, 1, 3, 2, 2)

    first, cache = ABotWorldCausalPipeline._decode_realtime_chunk(pipeline, chunk, None)
    continued, _ = ABotWorldCausalPipeline._decode_realtime_chunk(pipeline, chunk, cache)
    restarted, _ = ABotWorldCausalPipeline._decode_realtime_chunk(pipeline, chunk, None)

    assert first.shape[2] == restarted.shape[2] == 9
    assert continued.shape[2] == 12
    assert continued[:, :, 0].eq(0.3).all()
    assert restarted[:, :, 0].eq(0).all()


def test_ar_diffusion_spec_reserves_persistent_vae_state(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.abot_world.pipeline_abot_world.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    pipeline = SimpleNamespace(
        transformer=SimpleNamespace(
            config=SimpleNamespace(
                patch_size=(1, 2, 2),
                num_attention_heads=1,
                out_channels=48,
                num_layers=1,
                attention_head_dim=8,
                text_dim=4096,
            ),
            dtype=torch.bfloat16,
        ),
        vae=SimpleNamespace(dtype=torch.bfloat16),
        _vae_backend="taew2_2",
        vae_scale_factor_spatial=16,
        vae_scale_factor_temporal=4,
        _ar_height=512,
        _ar_width=832,
        _num_frame_per_block=3,
        _AR_BRANCH="main",
        _AR_TEXT_CACHE="text",
    )

    spec = ABotWorldCausalPipeline.ar_diffusion_kv_cache_spec(pipeline)

    vae_cache_bytes = TAEW2Decoder.persistent_state_bytes(32, 52, torch.bfloat16)
    assert vae_cache_bytes == 17_891_328
    assert (
        spec.model_owned_state_bytes_per_session
        == 4_792_320 + vae_cache_bytes + 512 * 4096 * 2 + 32 * 3 * 512 * 832 * 2
    )


def test_stepwise_camera_script_has_one_three_frame_action_block_per_chunk() -> None:
    script = parse_abot_camera_action_script([[["w"], [], ["d"]], [[], ["i"], []]])

    assert script == ((("w",), (), ("d",)), ((), ("i",), ()))
    assert ABotWorldCausalPipeline.supports_step_execution is True


@pytest.mark.parametrize("backend", ["wan", "taew2_2"])
def test_streaming_decode_matches_whole_clip_and_isolates_sessions(backend, monkeypatch) -> None:
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d, WanDecoder3d

    if backend == "wan":
        module = WanDecoder3d(
            dim=4,
            z_dim=48,
            dim_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            temperal_upsample=[True, True, False],
            out_channels=12,
            is_residual=True,
        ).eval()
        vae = SimpleNamespace(
            decoder=module,
            post_quant_conv=torch.nn.Identity(),
            config=SimpleNamespace(patch_size=2),
            _cached_conv_counts={"decoder": sum(isinstance(m, WanCausalConv3d) for m in module.modules())},
        )
        decoder = WanStreamingDecoder(vae)
        # Simulate the GPU case: BF16 weights do not imply BF16 feature caches.
        budget = _wan_decode_cache_bytes(module, 2, 2, torch.bfloat16)
    else:
        monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
        monkeypatch.setattr(torch.nn.Sequential, "load_state_dict", lambda *args, **kwargs: None)
        decoder = TAEW2Decoder("unused", torch.float32)
        budget = decoder.persistent_state_bytes(2, 2, torch.float32)

    latent = torch.randn(1, 48, 6, 2, 2, generator=torch.Generator().manual_seed(42))
    a, b = decoder.new_decode_state("a"), decoder.new_decode_state("b")
    with torch.inference_mode():
        first = decoder.decode_chunk(latent[:, :, :3], a)
        other = decoder.decode_chunk(-latent[:, :, :3], b)
        second = decoder.decode_chunk(latent[:, :, 3:], a)
        whole_state = decoder.new_decode_state("whole")
        whole = decoder.decode_chunk(latent, whole_state)
    assert first.shape == other.shape == (1, 3, 9, 32, 32)
    assert second.shape == (1, 3, 12, 32, 32)
    assert torch.equal(torch.cat([first, second], dim=2), whole)
    assert 0 < a.nbytes() <= budget
    assert a.nbytes() == b.nbytes() == whole_state.nbytes()
    assert all(
        x.data_ptr() != y.data_ptr()
        for x, y in zip(a.feat_map, b.feat_map)
        if torch.is_tensor(x) and torch.is_tensor(y)
    )
    decoder.release(a)
    assert a.nbytes() == 0 and not a.started
    assert b.nbytes() > 0


def test_decode_failure_releases_partial_session_cache() -> None:
    state = StreamingDecodeState("failed", [torch.ones(2)], frames_decoded=3)

    class Decoder:
        def decode_chunk(self, latent, cache):
            cache.feat_map[0] = torch.ones(4)
            raise RuntimeError("decode failed")

    pipeline = SimpleNamespace(
        _streaming_decoder=Decoder(),
        _vae_backend="taew2_2",
        vae=SimpleNamespace(dtype=torch.float32),
    )
    with pytest.raises(RuntimeError, match="decode failed"):
        ABotWorldCausalPipeline._decode_realtime_chunk(pipeline, torch.zeros(1, 48, 3, 2, 2), state)
    assert state.nbytes() == 0


def test_session_close_releases_decode_state_without_touching_other_session() -> None:
    a = StreamingDecodeState("a", [torch.ones(2)], frames_decoded=3)
    b = StreamingDecodeState("b", [torch.ones(2)], frames_decoded=3)
    pipeline = SimpleNamespace(_ar_sessions={"a": object(), "b": object()}, _streaming_decode_states={"a": a, "b": b})
    ABotWorldCausalPipeline.close_ar_diffusion_session(pipeline, "a")
    ABotWorldCausalPipeline.close_ar_diffusion_session(pipeline, "a")
    assert a.nbytes() == 0 and b.nbytes() > 0
    assert list(pipeline._ar_sessions) == list(pipeline._streaming_decode_states) == ["b"]
