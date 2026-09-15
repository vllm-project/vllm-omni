# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.config.stage_config import DiffusionStageRole
from vllm_omni.diffusion.models.dreamzero.payload_dreamzero import DreamZeroStaleRequestError
from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import (
    DREAMZERO_MODEL_OWNED_STATE_BYTES_PER_SESSION,
    DreamZeroPipeline,
)
from vllm_omni.diffusion.models.dreamzero.state_dreamzero import DreamZeroState
from vllm_omni.diffusion.models.dreamzero.utils import (
    DREAMZERO_BOUNDARY_ENCODE_TO_DIT,
    DREAMZERO_PAYLOAD_VERSION,
    DREAMZERO_STAGE_PAYLOAD_KEY,
)
from vllm_omni.diffusion.models.interface import role_loads_component
from vllm_omni.experimental.ar_diffusion.tick_protocol import AR_DIFFUSION_TICK_KEY
from vllm_omni.experimental.world_models.session_state import SessionStateLostError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _empty_pipeline() -> DreamZeroPipeline:
    pipeline = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipeline._states = OrderedDict()
    return pipeline


def test_dreamzero_pipeline_state_is_session_keyed() -> None:
    pipeline = _empty_pipeline()

    session_a = pipeline._get_or_create_state("session-a")
    session_b = pipeline._get_or_create_state("session-b")
    session_a.call_count = 7
    session_b.call_count = 3

    assert pipeline._get_or_create_state("session-a") is session_a
    assert pipeline._get_or_create_state("session-b") is session_b
    assert session_a.call_count == 7
    assert session_b.call_count == 3


def test_dreamzero_kv_spec_accounts_for_model_owned_cuda_state(monkeypatch) -> None:
    pipeline = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipeline.transformer = SimpleNamespace(
        frame_seqlen=16,
        blocks=[
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    max_attention_size=96,
                    tp_num_heads=4,
                )
            )
        ],
        text_len=8,
        model_type="t2v",
        num_layers=2,
        num_heads=4,
        dim=256,
        num_frame_per_block=4,
        num_action_per_block=2,
        num_state_per_block=1,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero.get_classifier_free_guidance_world_size",
        lambda: 1,
    )

    spec = pipeline.ar_diffusion_kv_cache_spec()

    assert spec.model_owned_state_bytes_per_session == DREAMZERO_MODEL_OWNED_STATE_BYTES_PER_SESSION


def test_dreamzero_pipeline_state_follows_runner_lifecycle_notifications() -> None:
    pipeline = _empty_pipeline()

    session_a = pipeline._get_or_create_state("session-a")
    session_b = pipeline._get_or_create_state("session-b")
    pipeline.state = session_b

    pipeline.close_ar_diffusion_session("session-a")
    assert pipeline.state is session_b
    pipeline.reset_ar_diffusion_session("session-b")

    assert not pipeline._states
    assert pipeline.state is None
    assert pipeline._get_or_create_state("session-a") is not session_a
    assert pipeline._get_or_create_state("session-b") is not session_b


def test_close_session_clears_active_state_alias() -> None:
    pipeline = _empty_pipeline()
    session_a = pipeline._get_or_create_state("session-a")
    pipeline.state = session_a

    pipeline.close_ar_diffusion_session("session-a")

    assert not pipeline._states
    assert pipeline.state is None


def test_dreamzero_warmup_provider_builds_session_scoped_requests() -> None:
    """FULL role: the encoders are present, so warmup carries raw observations."""
    provider = SimpleNamespace(
        ar_diffusion_kv_cache_spec=lambda: SimpleNamespace(window_frames=5, frames_per_block=4),
        od_config=SimpleNamespace(
            ar_diffusion_kv_config=None,
            model_config={"policy_server_config": {"image_resolution": [8, 16]}},
        ),
        _load_encoders=True,
        _ar_warmup_robot_obs=DreamZeroPipeline._ar_warmup_robot_obs,
    )

    requests = list(DreamZeroPipeline.ar_diffusion_warmup_requests(provider, "warmup-session"))

    assert [request.prompt for request in requests] == ["warmup", "warmup"]
    assert all(request.sampling_params.extra_args["session_id"] == "warmup-session" for request in requests)
    # Only the first warmup forward begins the session.
    assert [request.sampling_params.extra_args["reset"] for request in requests] == [True, False]
    assert requests[0].sampling_params.extra_args["robot_obs"]["observation/exterior_image_0_left"].shape == (
        8,
        16,
        3,
    )
    assert requests[1].sampling_params.extra_args["robot_obs"]["observation/exterior_image_0_left"].shape == (
        4,
        8,
        16,
        3,
    )


def test_dreamzero_state_owns_no_kv_caches() -> None:
    """KV (self- and cross-attention) is engine-owned since the AR-Diffusion
    paged backend: the model-local cache accessors must be gone so nothing can
    silently bypass the engine pool."""
    state = DreamZeroState()

    for removed in ("get_kv_caches", "create_kv_caches", "update_kv_cache", "get_crossattn_caches"):
        assert not hasattr(state, removed)


# ---------------------------------------------------------------------------
# Role-aware fixtures
# ---------------------------------------------------------------------------

# Small but internally consistent geometry: h_lat*w_lat must be a positive
# multiple of 4 so frame_seqlen = h_lat*w_lat/4 is a usable token count.
WARMUP_HEIGHT, WARMUP_WIDTH = 32, 64
WARMUP_H_LAT, WARMUP_W_LAT = WARMUP_HEIGHT // 8, WARMUP_WIDTH // 8
WARMUP_FRAME_SEQLEN = WARMUP_H_LAT * WARMUP_W_LAT // 4
WARMUP_NFPB = 4
WARMUP_NUM_FRAMES = 5
WARMUP_LATENT_FRAMES = (WARMUP_NUM_FRAMES - 1) // 4 + 1
WARMUP_TEXT_LEN, WARMUP_TEXT_DIM = 4, 8
WARMUP_ACTION_HORIZON, WARMUP_ACTION_DIM = 3, 7
WARMUP_MAX_STATE_DIM = 6


def _denoise_warmup_provider(*, cfg_scale: float, warmup_capture_reset: bool = False) -> SimpleNamespace:
    """DENOISE role: no encoders, so warmup needs a synthetic upstream payload."""
    provider = SimpleNamespace(
        ar_diffusion_kv_cache_spec=lambda: SimpleNamespace(window_frames=5, frames_per_block=WARMUP_NFPB),
        od_config=SimpleNamespace(
            ar_diffusion_kv_config={"warmup_capture_reset": warmup_capture_reset},
            model_config={"policy_server_config": {"image_resolution": [WARMUP_HEIGHT, WARMUP_WIDTH]}},
        ),
        _load_encoders=False,
        num_frame_per_block=WARMUP_NFPB,
        num_frames=WARMUP_NUM_FRAMES,
        text_len=WARMUP_TEXT_LEN,
        text_dim=WARMUP_TEXT_DIM,
        action_horizon=WARMUP_ACTION_HORIZON,
        action_dim=WARMUP_ACTION_DIM,
        max_state_dim=WARMUP_MAX_STATE_DIM,
        cfg_scale=cfg_scale,
    )
    provider._warmup_encode_payload = DreamZeroPipeline._warmup_encode_payload.__get__(provider, SimpleNamespace)
    return provider


def _warmup_payloads(monkeypatch, **kwargs) -> tuple[list[dict], list]:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero.get_local_device",
        lambda: torch.device("cpu"),
    )
    provider = _denoise_warmup_provider(**kwargs)
    requests = list(DreamZeroPipeline.ar_diffusion_warmup_requests(provider, "warmup-session"))
    return [request.prompt[DREAMZERO_STAGE_PAYLOAD_KEY] for request in requests], requests


def test_denoise_warmup_uses_a_stage_payload_and_no_raw_observation(monkeypatch) -> None:
    payloads, requests = _warmup_payloads(monkeypatch, cfg_scale=1.0)

    assert len(requests) == 2
    for request in requests:
        # The prompt is the payload envelope, not a bare string.
        assert isinstance(request.prompt, dict)
        assert request.prompt["prompt"] == "warmup"
        # A denoise stage owns no encoders, so it must never be handed raw obs.
        assert "robot_obs" not in request.sampling_params.extra_args
        assert request.sampling_params.extra_args["session_id"] == "warmup-session"

    for payload in payloads:
        assert payload["boundary"] == DREAMZERO_BOUNDARY_ENCODE_TO_DIT
        assert payload["payload_version"] == DREAMZERO_PAYLOAD_VERSION


def test_denoise_warmup_carries_valid_ordering_metadata(monkeypatch) -> None:
    payloads, _ = _warmup_payloads(monkeypatch, cfg_scale=1.0)

    scalars = [payload["scalar_fields"] for payload in payloads]
    # One epoch, sequences strictly increasing from 1, no retries, and the
    # runner already released the session, so no stage repeats a reset.
    assert [scalar["epoch"] for scalar in scalars] == [1, 1]
    assert [scalar["sequence"] for scalar in scalars] == [1, 2]
    assert [scalar["attempt"] for scalar in scalars] == [0, 0]
    assert [scalar["reset_reason"] for scalar in scalars] == [None, None]
    # Warmup is not a coordinated rollout.
    assert [scalar["generation"] for scalar in scalars] == [0, 0]
    # Window start first, then a continuation whose window advanced by one block.
    assert [scalar["window_start"] for scalar in scalars] == [True, False]
    assert [scalar["current_start_frame"] for scalar in scalars] == [0, 1]
    assert [scalar["frame_seqlen"] for scalar in scalars] == [WARMUP_FRAME_SEQLEN] * 2
    assert [scalar["seq_len"] for scalar in scalars] == [WARMUP_FRAME_SEQLEN * WARMUP_NFPB] * 2


def test_denoise_warmup_covers_window_start_and_continuation_shapes(monkeypatch) -> None:
    payloads, _ = _warmup_payloads(monkeypatch, cfg_scale=1.0)

    window_start, continuation = (payload["tensor_fields"] for payload in payloads)

    # Window start carries one observation frame; a continuation carries a block.
    assert tuple(window_start["image_latents"].shape) == (1, 1, 16, WARMUP_H_LAT, WARMUP_W_LAT)
    assert tuple(continuation["image_latents"].shape) == (1, WARMUP_NFPB, 16, WARMUP_H_LAT, WARMUP_W_LAT)

    for tensors in (window_start, continuation):
        assert tuple(tensors["prompt_embeds"].shape) == (1, WARMUP_TEXT_LEN, WARMUP_TEXT_DIM)
        assert tuple(tensors["ys"].shape) == (1, 20, WARMUP_LATENT_FRAMES, WARMUP_H_LAT, WARMUP_W_LAT)
        assert tuple(tensors["clip_feas"].shape) == (1, 257, 1280)
        assert tuple(tensors["noise_obs"].shape) == (1, WARMUP_NFPB, 16, WARMUP_H_LAT, WARMUP_W_LAT)
        assert tuple(tensors["noise_action"].shape) == (1, WARMUP_ACTION_HORIZON, WARMUP_ACTION_DIM)
        assert tuple(tensors["state_features"].shape) == (1, 1, WARMUP_MAX_STATE_DIM)
        assert tuple(tensors["embodiment_id"].shape) == (1,)
        assert tensors["embodiment_id"].dtype == torch.long


def test_denoise_warmup_matches_cfg_configuration(monkeypatch) -> None:
    cfg_off, _ = _warmup_payloads(monkeypatch, cfg_scale=1.0)
    cfg_on, _ = _warmup_payloads(monkeypatch, cfg_scale=5.0)

    for payload in cfg_off:
        assert payload["scalar_fields"]["do_true_cfg"] is False
        assert "negative_prompt_embeds" not in payload["tensor_fields"]
    for payload in cfg_on:
        assert payload["scalar_fields"]["do_true_cfg"] is True
        assert tuple(payload["tensor_fields"]["negative_prompt_embeds"].shape) == (
            1,
            WARMUP_TEXT_LEN,
            WARMUP_TEXT_DIM,
        )


def test_warmup_capture_reset_adds_one_more_forward(monkeypatch) -> None:
    without_reset, _ = _warmup_payloads(monkeypatch, cfg_scale=1.0)
    with_reset, _ = _warmup_payloads(monkeypatch, cfg_scale=1.0, warmup_capture_reset=True)

    assert len(without_reset) == 2
    assert len(with_reset) == 3
    # The extra forward is another continuation, one block further along.
    assert [payload["scalar_fields"]["current_start_frame"] for payload in with_reset] == [
        0,
        1,
        1 + WARMUP_NFPB,
    ]
    assert [payload["scalar_fields"]["sequence"] for payload in with_reset] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Role gating
# ---------------------------------------------------------------------------


def _role_pipeline(role: str, *, with_transformer: bool) -> DreamZeroPipeline:
    """Pipeline shell with the role-derived component flags set explicitly."""
    pipeline = DreamZeroPipeline.__new__(DreamZeroPipeline)
    pipeline.stage_role = DiffusionStageRole(role)
    pipeline._load_encoders = role_loads_component(role, "encoder")
    pipeline._load_dit = role_loads_component(role, "dit")
    # DreamZero's VAE belongs to the encoder group: the trailing postprocess
    # stage emits latents and actions and needs no decoder.
    pipeline._load_vae = pipeline._load_encoders
    pipeline._holds_session_state = pipeline._load_encoders or pipeline._load_dit
    pipeline._states = OrderedDict()
    pipeline._session_generations = {}
    pipeline.state = None
    pipeline.vae = object() if pipeline._load_vae else None
    pipeline.transformer = (
        SimpleNamespace(
            frame_seqlen=16,
            blocks=[SimpleNamespace(self_attn=SimpleNamespace(max_attention_size=96, tp_num_heads=4))],
            text_len=8,
            model_type="t2v",
            num_layers=2,
            num_heads=4,
            dim=256,
            num_frame_per_block=4,
            num_action_per_block=2,
            num_state_per_block=1,
        )
        if with_transformer
        else None
    )
    return pipeline


@pytest.mark.parametrize("role", ["full", "denoise"])
def test_dit_owning_roles_publish_a_kv_spec(role, monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero.get_classifier_free_guidance_world_size",
        lambda: 1,
    )
    pipeline = _role_pipeline(role, with_transformer=True)

    spec = pipeline.ar_diffusion_kv_cache_spec()

    assert spec.num_layers == 2
    assert spec.tokens_per_frame == 16


@pytest.mark.parametrize("role", ["encode", "decode"])
def test_non_dit_roles_reject_a_kv_spec_request(role) -> None:
    pipeline = _role_pipeline(role, with_transformer=False)

    with pytest.raises(RuntimeError, match="does not own the DiT"):
        pipeline.ar_diffusion_kv_cache_spec()


def test_role_component_groups_exclude_what_a_stage_must_not_construct() -> None:
    assert role_loads_component("encode", "encoder") is True
    assert role_loads_component("encode", "dit") is False
    assert role_loads_component("denoise", "dit") is True
    assert role_loads_component("denoise", "encoder") is False
    assert role_loads_component("full", "encoder") is True
    assert role_loads_component("full", "dit") is True


def test_decode_role_constructs_neither_encoders_nor_dit() -> None:
    pipeline = _role_pipeline("decode", with_transformer=False)

    assert pipeline._load_encoders is False
    assert pipeline._load_dit is False
    # DreamZero's decode stage is a weightless action postprocess: no VAE either.
    assert pipeline._load_vae is False
    assert pipeline.vae is None
    assert pipeline.transformer is None
    assert pipeline._holds_session_state is False


def test_postprocess_stage_retains_no_per_session_state() -> None:
    pipeline = _role_pipeline("decode", with_transformer=False)

    # Lifecycle calls are idempotent no-ops that allocate nothing.
    pipeline.close_ar_diffusion_session("rollout-1")
    pipeline.reset_ar_diffusion_session("rollout-1")
    assert pipeline._states == OrderedDict()
    assert pipeline.state is None
    assert pipeline.has_session_state("rollout-1") is False

    # It can still carry the coordinator's generation: identity, not tensors.
    assert pipeline.register_session_generation("rollout-1", 4) is True
    assert pipeline.registered_session_generation("rollout-1") == 4
    assert pipeline._states == OrderedDict()


def test_decode_stage_refuses_to_act_as_a_vae_decoder() -> None:
    pipeline = _role_pipeline("decode", with_transformer=False)

    with pytest.raises(RuntimeError, match="owns no VAE"):
        pipeline.decode_accumulated_video_latents("rollout-1")


# ---------------------------------------------------------------------------
# Missing history and generation fencing
# ---------------------------------------------------------------------------


def test_continuation_without_resident_state_is_refused_read_only() -> None:
    pipeline = _empty_pipeline()

    with pytest.raises(SessionStateLostError, match="reset"):
        pipeline._require_session_state("rollout-1")
    # The lookup created nothing.
    assert pipeline._states == OrderedDict()

    created = pipeline._get_or_create_state("rollout-1")
    assert pipeline._require_session_state("rollout-1") is created


def test_export_after_close_fails_without_creating_state() -> None:
    pipeline = _empty_pipeline()
    pipeline.vae = object()
    pipeline._get_or_create_state("rollout-1")
    pipeline.close_ar_diffusion_session("rollout-1")

    with pytest.raises(SessionStateLostError):
        pipeline.decode_accumulated_video_latents("rollout-1")
    with pytest.raises(SessionStateLostError):
        pipeline.clear_accumulated_video_latents("rollout-1")
    assert pipeline._states == OrderedDict()


def test_clearing_exported_latents_keeps_the_session_continuable() -> None:
    pipeline = _empty_pipeline()
    state = pipeline._get_or_create_state("rollout-1")
    state.append_video_latents(torch.zeros(1, 2, 16, 4, 8))
    state.current_start_frame = 5

    pipeline.clear_accumulated_video_latents("rollout-1")

    # The export buffer is gone; the session and its window position are not.
    assert state.get_concatenated_video_latents() is None
    assert pipeline._states["rollout-1"] is state
    assert state.current_start_frame == 5


def test_request_begins_session_gives_the_typed_tick_precedence() -> None:
    typed_false_flat_true = {
        "reset": True,
        AR_DIFFUSION_TICK_KEY: {
            "session_id": "s",
            "request_id": "r",
            "chunk_index": 0,
            "reset": False,
        },
    }
    typed_true_flat_absent = {
        AR_DIFFUSION_TICK_KEY: {
            "session_id": "s",
            "request_id": "r",
            "chunk_index": 0,
            "reset": True,
        },
    }

    # A typed False must not be OR-ed with a stale flat True.
    assert DreamZeroPipeline._request_begins_session(typed_false_flat_true) is False
    assert DreamZeroPipeline._request_begins_session(typed_true_flat_absent) is True
    assert DreamZeroPipeline._request_begins_session({"reset": True}) is True
    assert DreamZeroPipeline._request_begins_session({}) is False


def _encoded(session_id: str, *, generation: int = 0, epoch: int = 1, sequence: int = 1, reset_reason=None):
    from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import (
        _DreamZeroEncoded,
        _DreamZeroPostprocessMeta,
    )

    return _DreamZeroEncoded(
        session_id=session_id,
        generation=generation,
        reset_reason=reset_reason,
        window_start=True,
        current_start_frame=0,
        epoch=epoch,
        sequence=sequence,
        attempt=0,
        frame_seqlen=8,
        seq_len=32,
        do_true_cfg=False,
        prompt_embeds=torch.zeros(1),
        negative_prompt_embeds=None,
        clip_feas=torch.zeros(1),
        ys=torch.zeros(1),
        image_latents=torch.zeros(1),
        noise_obs=torch.zeros(1),
        noise_action=torch.zeros(1),
        state_features=None,
        embodiment_id=torch.zeros(1, dtype=torch.long),
        postprocess_meta=_DreamZeroPostprocessMeta(embodiment_name="e", embodiment_key="k"),
    )


def _progress_pipeline() -> DreamZeroPipeline:
    pipeline = _empty_pipeline()
    pipeline._issued_progress = {}
    pipeline._committed_progress = {}
    pipeline._session_generations = {}
    return pipeline


def test_authorization_is_side_effect_free_when_it_rejects() -> None:
    pipeline = _progress_pipeline()
    state = pipeline._get_or_create_state("s")

    # A first chunk is authorized but nothing is committed yet.
    pipeline._authorize_stage_progress(_encoded("s", epoch=1, sequence=1), state)
    assert pipeline._committed_progress == {}

    # A rejected chunk must not leave partially advanced bookkeeping behind.
    with pytest.raises(DreamZeroStaleRequestError, match="must start at sequence 1"):
        pipeline._authorize_stage_progress(_encoded("s", epoch=2, sequence=3), state)
    assert pipeline._committed_progress == {}

    # Progress moves only after the chunk's work succeeded.
    enc = _encoded("s", epoch=1, sequence=1)
    pipeline._commit_stage_progress(enc)
    committed = pipeline._committed_progress["s"]
    assert (committed.epoch, committed.sequence, committed.attempt) == (1, 1, 0)


def test_authorization_still_rejects_duplicates_gaps_and_fenced_epochs() -> None:
    pipeline = _progress_pipeline()
    state = pipeline._get_or_create_state("s")
    pipeline._commit_stage_progress(_encoded("s", epoch=2, sequence=2))

    with pytest.raises(DreamZeroStaleRequestError, match="duplicate or"):
        pipeline._authorize_stage_progress(_encoded("s", epoch=2, sequence=2), state)
    with pytest.raises(DreamZeroStaleRequestError, match="leaves a gap"):
        pipeline._authorize_stage_progress(_encoded("s", epoch=2, sequence=5), state)
    with pytest.raises(DreamZeroStaleRequestError, match="was fenced"):
        pipeline._authorize_stage_progress(_encoded("s", epoch=1, sequence=1), state)

    # The next chunk in sequence is still accepted.
    pipeline._authorize_stage_progress(_encoded("s", epoch=2, sequence=3), state)


def test_a_payload_from_a_replaced_generation_is_rejected() -> None:
    pipeline = _progress_pipeline()
    state = pipeline._get_or_create_state("s")
    pipeline.register_session_generation("s", 7)

    # The generation this stage is registered for is accepted.
    pipeline._authorize_stage_progress(_encoded("s", generation=7), state)

    with pytest.raises(DreamZeroStaleRequestError, match="generation 3"):
        pipeline._authorize_stage_progress(_encoded("s", generation=3), state)
    # An uncoordinated payload carries no generation and is not fenced by one.
    pipeline._authorize_stage_progress(_encoded("s", generation=0), state)


def test_dropping_a_session_forgets_its_generation_and_progress() -> None:
    pipeline = _progress_pipeline()
    pipeline._get_or_create_state("s")
    pipeline.register_session_generation("s", 5)
    pipeline._commit_stage_progress(_encoded("s", epoch=3, sequence=4))

    pipeline.close_ar_diffusion_session("s")

    assert pipeline.registered_session_generation("s") == 0
    assert pipeline._committed_progress == {}
    assert pipeline._issued_progress == {}
