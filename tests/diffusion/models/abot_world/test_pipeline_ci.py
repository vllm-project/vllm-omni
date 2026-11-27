# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small CPU tests for ABot-World tensor geometry and checkpoint mapping."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.abot_world.abot_world_transformer import (
    ABotAttentionCache,
    ABotSimpleAdapter,
    ABotTransformerCache,
    ABotWorldCausalTransformer3DModel,
)
from vllm_omni.diffusion.models.abot_world.actions import parse_abot_camera_action_script
from vllm_omni.diffusion.models.abot_world.interaction import ABotCameraSession
from vllm_omni.diffusion.models.abot_world.pipeline_abot_world import (
    ABotWorldCausalPipeline,
    _ABotRequestInputs,
    _build_shifted_flow_schedule,
    _wan_decode_cache_bytes,
)
from vllm_omni.diffusion.models.abot_world.taew2_2 import TAEW2Decoder
from vllm_omni.diffusion.models.interface import SupportsInteractionApply
from vllm_omni.diffusion.offloader import sequential_backend
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.streaming_decode import StreamingDecodeState, WanStreamingDecoder
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

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
                use_relative_rope=False,
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
    pipeline = SimpleNamespace(_streaming_decode_states={"a": a, "b": b})
    ABotWorldCausalPipeline.close_ar_diffusion_session(pipeline, "a")
    ABotWorldCausalPipeline.close_ar_diffusion_session(pipeline, "a")
    assert a.nbytes() == 0 and b.nbytes() > 0
    assert list(pipeline._streaming_decode_states) == ["b"]


def test_pipeline_exposes_the_chunk_boundary_interaction_contract() -> None:
    assert issubclass(ABotWorldCausalPipeline, SupportsInteractionApply)


def _bare_pipeline(**attributes) -> ABotWorldCausalPipeline:
    """A real pipeline instance without checkpoint loading."""
    pipeline = ABotWorldCausalPipeline.__new__(ABotWorldCausalPipeline)
    torch.nn.Module.__init__(pipeline)
    for name, value in attributes.items():
        setattr(pipeline, name, value)
    return pipeline


@pytest.mark.parametrize(
    ("relative", "chunk_index", "fps", "expected"),
    [
        (False, 0, None, (9, 9.0)),
        (False, 1, None, (12, 12.0)),
        (False, 1, 16, (12, 16.0)),
        (True, 0, None, (13, 13.0)),
        (True, 1, None, (12, 12.0)),
    ],
)
def test_peek_chunk_media_reports_the_opening_and_steady_chunk_lengths(relative, chunk_index, fps, expected) -> None:
    pipeline = _bare_pipeline(
        _num_frame_per_block=3,
        vae_scale_factor_temporal=4,
        transformer=SimpleNamespace(config=SimpleNamespace(use_relative_rope=relative)),
    )
    state = StepRequestState(request_id="r", sampling=OmniDiffusionSamplingParams(fps=fps), chunk_index=chunk_index)

    media = pipeline.peek_chunk_media(state)

    assert (media.num_media_frames, media.fps, media.num_latent_frames) == (*expected, 3)


_NO_SESSION = object()
_SCRIPT = (((), (), ()), (("d",), (), ()))


def _stepwise_pipeline(seen: list) -> ABotWorldCausalPipeline:
    pipeline = _bare_pipeline(
        transformer=_tiny_transformer(),
        vae_scale_factor_spatial=16,
        device=torch.device("cpu"),
    )

    def build_action_tensor(actions, *args):
        seen.append(actions)
        return torch.zeros(1)

    pipeline._build_action_tensor = build_action_tensor
    pipeline._ar_text_caches = lambda *args, **kwargs: []
    return pipeline


def _stepwise_state(*, script=None, live=_NO_SESSION) -> StepRequestState:
    inputs = _ABotRequestInputs(
        prompt="p",
        image=torch.zeros(3, 32, 32),
        reference_images=None,
        camera_action_script=script,
        height=32,
        width=32,
        num_frames=33,
        num_latent_frames=9,
        num_frame_per_block=3,
        output_type="pil",
        max_sequence_length=512,
        flow_shift=5.0,
        generator=torch.Generator().manual_seed(0),
    )
    state = StepRequestState(
        request_id="r",
        sampling=OmniDiffusionSamplingParams(),
        prompt_embeds=torch.zeros(1, 1, 16),
        chunk_index=1,
        total_chunks=3,
        chunk_num_steps=4,
        extra={"inputs": inputs, "dtype": torch.float32, "schedule": ((1000, 1.0), (750, 0.75))},
    )
    if live is not _NO_SESSION:
        state.interaction_sessions["camera"] = ABotCameraSession(chunk_actions=live)
    return state


@pytest.mark.parametrize(
    ("script", "live", "expected"),
    [
        (None, _NO_SESSION, None),
        (None, None, None),
        (None, (("w",), (), ()), (("w",), (), ())),
        (_SCRIPT, _NO_SESSION, _SCRIPT[1]),
        (_SCRIPT, None, _SCRIPT[1]),
    ],
)
def test_next_chunk_takes_live_or_scripted_actions(script, live, expected) -> None:
    seen: list = []
    state = _stepwise_state(script=script, live=live)

    _stepwise_pipeline(seen).prepare_next_chunk(state)

    assert seen == [expected]
    assert state.extra["start_frame"] == 3
    assert state.latents.shape == (1, 4, 3, 2, 2)


def test_next_chunk_rejects_live_actions_for_a_scripted_request() -> None:
    state = _stepwise_state(script=_SCRIPT, live=(("w",), (), ()))
    with pytest.raises(ValueError, match="camera_action_script"):
        _stepwise_pipeline([]).prepare_next_chunk(state)


def test_finished_rollout_prepares_no_further_chunk() -> None:
    seen: list = []
    state = _stepwise_state()
    state.chunk_index = state.total_chunks

    _stepwise_pipeline(seen).prepare_next_chunk(state)

    assert seen == [] and state.latents is None


def test_request_mode_rejects_ar_diffusion_session_binding() -> None:
    pipeline = _bare_pipeline(_ar_diffusion_kv_state=object())
    request = OmniDiffusionRequest(prompt="p", sampling_params=OmniDiffusionSamplingParams(), request_id="r")
    with pytest.raises(ValueError, match="enable step execution"):
        pipeline.forward(DiffusionRequestBatch(requests=[request]))


class _RecordingTextCache:
    """Stands in for the runner-owned KV state; records where projection ran."""

    def __init__(self, moves: list) -> None:
        self._moves = moves
        self.moves_at_projection: list = []
        self._layers: list = []

    def is_cross_attention_populated(self, kv_branch: str, cache_name: str) -> bool:
        return bool(self._layers)

    def populate_cross_attention(self, kv_branch: str, cache_name: str, layer_kv) -> None:
        self._layers = [{"k": key, "v": value} for key, value in layer_kv]
        self.moves_at_projection = list(self._moves)

    def get_cross_attention_kv(self, kv_branch: str, cache_name: str) -> list:
        return self._layers


@pytest.mark.parametrize("module_offload", [True, False])
def test_text_cache_projection_activates_the_offloaded_transformer(monkeypatch, module_offload) -> None:
    transformer = _tiny_transformer()
    text_encoder = torch.nn.Linear(2, 2)
    names = {id(transformer): "transformer", id(text_encoder): "text_encoder"}
    moves: list = []

    def move_params(module, target_device, **kwargs):
        moves.append((names[id(module)], target_device.type))
        return False

    monkeypatch.setattr(sequential_backend.SequentialOffloadHook, "_move_params", staticmethod(move_params))
    monkeypatch.setattr(sequential_backend.current_omni_platform, "synchronize", lambda: None)
    monkeypatch.setattr(sequential_backend.current_omni_platform, "get_free_memory", lambda: 0)
    sequential_backend.apply_sequential_offload(
        dit_modules=[transformer],
        encoder_modules=[text_encoder],
        device=torch.device("cuda:0"),
        pin_memory=False,
    )
    cache = _RecordingTextCache(moves)
    pipeline = _bare_pipeline(transformer=transformer, _ar_diffusion_kv_state=cache, _module_offload=module_offload)

    caches = pipeline._ar_text_caches(torch.zeros(1, 4, 16), invalidate=False)

    assert len(caches) == 1
    if module_offload:
        assert cache.moves_at_projection == [("text_encoder", "cpu"), ("transformer", "cuda")]
        assert moves[2:] == [("transformer", "cpu")]
    else:
        assert moves == []


class _WordTokenizer:
    """One id per word plus EOS, padded with zeros like the UMT5 tokenizer."""

    def __call__(self, texts, *, max_length, **kwargs):
        words = texts[0].split()[: max_length - 1]
        ids = torch.zeros(1, max_length, dtype=torch.long)
        ids[0, : len(words) + 1] = torch.tensor([len(word) + 1 for word in words] + [1])
        return SimpleNamespace(input_ids=ids, attention_mask=(ids > 0).long())


def _text_encoder(input_ids, attention_mask):
    return SimpleNamespace(last_hidden_state=input_ids.float().unsqueeze(-1).expand(-1, -1, 2))


@pytest.mark.parametrize(("relative", "eos_at"), [(False, [3]), (True, [3, 384])])
def test_iris_prompt_is_a_global_row_and_an_empty_local_row(relative: bool, eos_at: list[int]) -> None:
    pipeline = _bare_pipeline(
        transformer=SimpleNamespace(config=SimpleNamespace(use_relative_rope=relative)),
        tokenizer=_WordTokenizer(),
        text_encoder=_text_encoder,
        device=torch.device("cpu"),
    )

    embeds = pipeline.encode_prompt("a red  car", max_sequence_length=512, dtype=torch.float32)

    assert embeds.shape == (1, 512, 2)
    assert embeds[0, :3, 0].tolist() == [2.0, 4.0, 4.0]
    assert torch.nonzero(embeds[0, :, 0] == 1.0).flatten().tolist() == eos_at
    assert int(torch.count_nonzero(embeds[0, :, 0])) == 3 + len(eos_at)


def test_text_cache_rebuild_keeps_the_reference_kv() -> None:
    model = _relative_transformer()
    state = _relative_kv_state(model)
    pipeline = _bare_pipeline(transformer=model, _ar_diffusion_kv_state=state, _module_offload=False)
    generator = torch.Generator().manual_seed(0)
    reference = torch.randn(1, _REF_TOKENS, 1, 8, generator=generator)
    state.populate_cross_attention("main", "ref", [(reference, -reference)])

    with torch.no_grad():
        before = pipeline._ar_text_caches(torch.randn(1, 5, 16, generator=generator), invalidate=False)[0].key.clone()
        after = pipeline._ar_text_caches(torch.randn(1, 5, 16, generator=generator), invalidate=True)[0].key

    assert not torch.equal(after, before)
    kept = pipeline._ar_reference_caches()[0]
    assert torch.equal(kept.key, reference) and torch.equal(kept.value, -reference)


@pytest.mark.parametrize("model_type", ["ti2v", "ci2v"])
def test_native_abot_checkpoint_resolves_without_a_pipeline_index(tmp_path, model_type: str) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"_class_name": "CausalWanModel", "model_type": model_type}))
    config = OmniDiffusionConfig(model=str(tmp_path), model_class_name="ABotWorldCausalPipeline")

    config.enrich_config()

    assert config.model_class_name == "ABotWorldCausalPipeline"


def _relative_transformer() -> ABotWorldCausalTransformer3DModel:
    model = ABotWorldCausalTransformer3DModel(
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
        use_relative_rope=True,
        local_attn_size=7,
    )
    # vLLM parallel linear layers allocate uninitialized weights.
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model.eval()


def test_relative_rope_rollout_stays_finite_past_the_window_and_keeps_reference_sink() -> None:
    model = _relative_transformer()
    refs = torch.randn(1, 2, 4, 1, 4, 4)
    mask = torch.tensor([[1.0, 0.0]])
    ref_tokens = 2 * 4
    cache = model.allocate_cache(
        batch_size=1,
        latent_height=4,
        latent_width=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        ref_tokens=ref_tokens,
    )
    text = torch.randn(1, 5, 16)

    def call(frames: int, start: int, t: float, commit: bool) -> torch.Tensor:
        return model(
            torch.randn(1, 4, frames, 4, 4),
            torch.full((1, frames), t),
            text,
            cache=cache,
            start_frame=start,
            update_cache=commit,
            ref_latents=refs,
            ref_mask=mask,
        )

    with torch.no_grad():
        assert call(1, 0, 0.0, True).shape == (1, 4, 1, 4, 4)
        sink = cache.self_attention[0].key[:, :ref_tokens].clone()
        for block in range(6):  # 1 + 18 frames: the 7-frame window wraps twice
            start = 1 + 3 * block
            assert torch.isfinite(call(3, start, 500.0, False)).all()
            assert torch.isfinite(call(3, start, 0.0, True)).all()

    assert torch.equal(cache.self_attention[0].key[:, :ref_tokens], sink)


def test_masked_reference_slot_contributes_no_tokens() -> None:
    model = _relative_transformer()
    refs = torch.randn(1, 2, 4, 1, 4, 4)

    tokens = model._ref_tokens(refs, torch.tensor([[1.0, 0.0]]))

    assert tokens.shape[1] == 2 * 4
    assert tokens[:, :4].abs().sum() > 0
    assert torch.count_nonzero(tokens[:, 4:]) == 0


def _parse(pipeline: ABotWorldCausalPipeline, *, num_frames: int, refs: tuple | None) -> _ABotRequestInputs:
    mm_data = {"image": torch.zeros(3, 64, 64), "reference_images": refs}
    sampling = OmniDiffusionSamplingParams(
        height=704, width=1280, num_frames=num_frames, generator=torch.Generator(), extra_args={}
    )
    request = OmniDiffusionRequest(
        prompt={"prompt": "p", "multi_modal_data": mm_data}, sampling_params=sampling, request_id="r"
    )
    return pipeline._parse_request(DiffusionRequestBatch(requests=[request]))


def _parsing_pipeline(*, relative: bool) -> ABotWorldCausalPipeline:
    config = SimpleNamespace(patch_size=(1, 2, 2), use_relative_rope=relative)
    return _bare_pipeline(
        transformer=SimpleNamespace(config=config),
        od_config=SimpleNamespace(),
        vae_scale_factor_spatial=16,
        vae_scale_factor_temporal=4,
        _num_frame_per_block=3,
    )


def test_reference_images_need_a_relative_rope_checkpoint() -> None:
    refs = (torch.zeros(3, 8, 8),)

    with pytest.raises(NotImplementedError, match="Iris"):
        _parse(_parsing_pipeline(relative=False), num_frames=9, refs=refs)

    assert _parse(_parsing_pipeline(relative=True), num_frames=13, refs=refs).reference_images == refs


@pytest.mark.parametrize(("num_frames", "ok"), [(13, True), (25, True), (9, False)])
def test_relative_rope_prefills_one_frame_then_whole_blocks(num_frames: int, ok: bool) -> None:
    pipeline = _parsing_pipeline(relative=True)

    if ok:
        assert _parse(pipeline, num_frames=num_frames, refs=None).num_latent_frames == (num_frames - 1) // 4 + 1
    else:
        with pytest.raises(ValueError, match="generated latent frames"):
            _parse(pipeline, num_frames=num_frames, refs=None)


def test_unfilled_reference_slots_are_zero_and_masked() -> None:
    pipeline = _bare_pipeline(
        transformer=SimpleNamespace(config=SimpleNamespace(in_channels=4)),
        vae_scale_factor_spatial=16,
        device=torch.device("cpu"),
    )
    pipeline._encode_first_frame = lambda image, height, width, dtype: torch.ones(1, 4, 1, 32, 32, dtype=dtype)

    encoded = pipeline._encode_references((torch.zeros(3, 8, 8), torch.zeros(3, 8, 8)), torch.float32)

    assert encoded["ref_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0]]
    assert encoded["ref_latents"].shape[:3] == (1, 5, 4)
    assert encoded["ref_latents"][:, 2:].abs().sum() == 0


_REF_TOKENS = 2 * 4  # two 4x4-latent reference slots, 2x2 patches


def _relative_kv_state(model: ABotWorldCausalTransformer3DModel) -> ARDiffusionKVState:
    kv = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=4, window_chunks=7, sink_chunks=0),
        num_layers=1,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        block_size=4,
        max_model_len=4096,
        available_bytes=1 << 24,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        frames_per_block=3,
        max_scratch_tokens_per_branch=3 * 4,
        cross_attention_lengths={"text": 5, "ref": _REF_TOKENS},
        device=torch.device("cpu"),
    )
    return ARDiffusionKVState(kv, "s", {"main": kv.begin_request("s")}, num_layers=1)


def test_paged_relative_rope_matches_the_dense_cache_past_the_window() -> None:
    model = _relative_transformer()
    generator = torch.Generator().manual_seed(0)
    refs = torch.randn(1, 2, 4, 1, 4, 4, generator=generator)
    text = torch.randn(1, 5, 16, generator=generator)
    # Source-frame prefill, then 1 + 18 frames: the 7-frame window slides twice.
    calls = [(1, 0, 0.0, True)] + [(3, 1 + 3 * block, t, t == 0.0) for block in range(6) for t in (750.0, 500.0, 0.0)]
    inputs = [torch.randn(1, 4, frames, 4, 4, generator=generator) for frames, *_ in calls]

    dense = model.allocate_cache(
        batch_size=1,
        latent_height=4,
        latent_width=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        ref_tokens=_REF_TOKENS,
    )
    state = _relative_kv_state(model)
    holder = ABotAttentionCache(key=torch.empty(0), value=torch.empty(0))
    paged = ABotTransformerCache(self_attention=[], cross_attention=[None], reference=[holder])

    with torch.no_grad():
        for (frames, start, t, commit), latent in zip(calls, inputs):
            paged.self_attention = state.get_kv_caches("main", seq_len=frames * 4, commit_current=commit)
            outputs = [
                model(
                    latent,
                    torch.full((1, frames), t),
                    text,
                    cache=cache,
                    start_frame=start,
                    update_cache=commit,
                    ref_latents=refs,
                    ref_mask=torch.ones(1, 2),
                )
                for cache in (dense, paged)
            ]
            if commit:
                state.commit_paged_context("main")
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=1e-6)
            sink = dense.self_attention[0].key[:, :_REF_TOKENS]
            assert torch.equal(holder.key, sink)


def test_paged_relative_rope_refuses_a_call_without_cached_reference_tokens() -> None:
    model = _relative_transformer()
    state = _relative_kv_state(model)
    holder = ABotAttentionCache(key=torch.empty(0), value=torch.empty(0))
    cache = ABotTransformerCache(
        self_attention=state.get_kv_caches("main", seq_len=12, commit_current=False),
        cross_attention=[None],
        reference=[holder],
    )

    with torch.no_grad(), pytest.raises(RuntimeError, match="cached reference tokens"):
        model(
            torch.randn(1, 4, 3, 4, 4),
            torch.full((1, 3), 500.0),
            torch.randn(1, 5, 16),
            cache=cache,
            start_frame=1,
            update_cache=False,
            ref_latents=torch.randn(1, 2, 4, 1, 4, 4),
        )


def test_stepwise_publishes_reference_kv_beside_text_and_runs_the_first_chunk() -> None:
    model = _relative_transformer()
    state = _relative_kv_state(model)
    pipeline = _bare_pipeline(
        transformer=model,
        device=torch.device("cpu"),
        vae_scale_factor_spatial=16,
        _ar_diffusion_kv_state=state,
        _module_offload=False,
    )
    pipeline._build_action_tensor = lambda *args: None
    generator = torch.Generator().manual_seed(0)
    inputs = _ABotRequestInputs(
        prompt="p",
        image=torch.zeros(3, 64, 64),
        reference_images=None,
        camera_action_script=None,
        height=64,
        width=64,
        num_frames=13,
        num_latent_frames=4,
        num_frame_per_block=3,
        output_type="latent",
        max_sequence_length=5,
        flow_shift=5.0,
        generator=generator,
    )
    ref_kwargs = {"ref_latents": torch.randn(1, 2, 4, 1, 4, 4, generator=generator), "ref_mask": torch.ones(1, 2)}
    step = StepRequestState(
        request_id="s",
        sampling=OmniDiffusionSamplingParams(),
        prompt_embeds=torch.randn(1, 5, 16, generator=generator),
        chunk_index=0,
        total_chunks=1,
        chunk_num_steps=4,
        extra={
            "inputs": inputs,
            "dtype": torch.float32,
            "schedule": _build_shifted_flow_schedule(flow_shift=5.0),
            "first_frame_latent": torch.randn(1, 4, 1, 4, 4, generator=generator),
            "ref_kwargs": ref_kwargs,
        },
    )

    with pytest.raises(RuntimeError, match="before it was populated"):
        pipeline._ar_reference_caches()
    with torch.no_grad():
        pipeline._prefill_source_frame(step, pipeline._ar_text_caches(step.prompt_embeds, invalidate=False))
        assert state.is_cross_attention_populated("main", "text")
        published = pipeline._ar_reference_caches()[0]

        dense = model.allocate_cache(
            batch_size=1,
            latent_height=4,
            latent_width=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            ref_tokens=_REF_TOKENS,
        )
        model(
            step.extra["first_frame_latent"],
            torch.zeros(1, 1),
            step.prompt_embeds,
            cache=dense,
            start_frame=0,
            update_cache=True,
            **ref_kwargs,
        )
        assert torch.equal(published.key, dense.self_attention[0].key[:, :_REF_TOKENS])
        assert torch.equal(published.value, dense.self_attention[0].value[:, :_REF_TOKENS])

        pipeline.prepare_next_chunk(step)
        assert step.extra["start_frame"] == 1
        for _ in range(step.chunk_num_steps):
            pipeline.step_scheduler(step, pipeline.denoise_step(None, states=[step]))
        output = pipeline.post_decode(step)

    latents = output.output["payload"]["latents"]
    assert latents.shape == (1, 4, 4, 4, 4)
    assert torch.equal(latents[:, :, :1], step.extra["first_frame_latent"])
    assert torch.equal(pipeline._ar_reference_caches()[0].key, published.key)
    state.close()
    assert not state.kv_cache.is_cross_attention_populated("s", "ref", "main")
