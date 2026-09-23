# SPDX-License-Identifier: Apache-2.0
"""Transfer request geometry, prompt, and sliding-history contracts."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.control_contract import (
    Cosmos3NanoSimBimanualControlVideoConditioning,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import (
    Cosmos3NanoSimBimanualGeometry,
    Cosmos3NanoSimBimanualResolutionPolicy,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_transfer import (
    Cosmos3NanoSimTransferPipeline,
    _TransferConditioning,
    format_cosmos3_nano_sim_transfer_prompt,
    resolve_cosmos3_nano_sim_transfer_geometry,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.state_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualSessionState,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _sampling_params(
    *,
    resolution: str = "480",
    height: int | None = None,
    width: int | None = None,
    **extra,
):
    return SimpleNamespace(
        height=height,
        width=width,
        extra_args={"resolution": resolution, **extra},
    )


def test_transfer_input_video_drives_bucket_before_hint_control() -> None:
    params = _sampling_params(edge={"control": {"height": 480, "width": 480}})
    prompt = {
        "multi_modal_data": {"video": {"height": 480, "width": 832}},
    }

    geometry = resolve_cosmos3_nano_sim_transfer_geometry(
        params,
        prompt,
        Cosmos3NanoSimBimanualResolutionPolicy(),
    )

    assert geometry.session_key == (480, 832)


def test_transfer_hint_control_drives_bucket_without_input_video() -> None:
    params = _sampling_params(depth={"control": {"height": 832, "width": 480}})

    geometry = resolve_cosmos3_nano_sim_transfer_geometry(
        params,
        {"prompt": "test"},
        Cosmos3NanoSimBimanualResolutionPolicy(),
    )

    assert geometry.session_key == (832, 480)


def test_transfer_serialized_dimensions_must_match_generated_bucket() -> None:
    params = _sampling_params(
        height=720,
        width=1280,
        edge={"control": {"height": 480, "width": 832}},
    )

    with pytest.raises(ValueError, match="serialized dimensions do not match"):
        resolve_cosmos3_nano_sim_transfer_geometry(
            params,
            {"prompt": "test"},
            Cosmos3NanoSimBimanualResolutionPolicy(),
        )


def test_transfer_generated_bucket_must_pass_bimanual_policy() -> None:
    params = _sampling_params(resolution="720", edge={"control": {"height": 480, "width": 832}})
    policy = Cosmos3NanoSimBimanualResolutionPolicy(
        default_resolution=(480, 832),
        max_pixels=480 * 832,
    )

    with pytest.raises(ValueError, match="max_pixels"):
        resolve_cosmos3_nano_sim_transfer_geometry(params, {"prompt": "test"}, policy)


def pipeline(chunk_size: int = 1) -> Cosmos3NanoSimTransferPipeline:
    instance = Cosmos3NanoSimTransferPipeline.__new__(Cosmos3NanoSimTransferPipeline)
    torch.nn.Module.__init__(instance)
    contract = Cosmos3NanoSimBimanualControlVideoConditioning(
        mode="control_video",
        hints=("edge", "blur", "depth", "seg"),
        transfer_control_attention_mode="causal_control_with_rgb_history",
        share_vision_temporal_positions=True,
        system_prompt_id="cosmos3_transfer_v1",
        emphasize_control_in_prompt=True,
        no_eviction=True,
    )
    instance.manifest = SimpleNamespace(
        chunk_size=chunk_size,
        window_frames=51 if chunk_size == 1 else 96,
        sink_frames=1 if chunk_size == 1 else 0,
        temporal_compression_factor=4,
        text_cache_max_len=512,
        require_control_video_conditioning=lambda: contract,
    )
    instance.dtype = torch.float32
    instance.device = torch.device("cpu")
    return instance


def params(frames: int = 121, **extra) -> SimpleNamespace:
    return SimpleNamespace(extra_args={"depth": {"control_path": "control.mp4"}, "num_frames": frames, **extra})


@pytest.mark.parametrize("frames", [1, 5, 121, 593])
def test_chunk1_admission(frames: int) -> None:
    p = pipeline()
    p._init_conditioning(SimpleNamespace(enforce_eager=True))
    request = p._validate_conditioning_request(
        params(frames, kv_cache_inference_size=30, attention_sink_size=3, emphasize_control_in_prompt=False),
        None,
    )
    assert (request.window_frames, request.sink_frames, request.emphasize_control_in_prompt) == (30, 3, False)


@pytest.mark.parametrize("window,sink", [(0, 0), (3, 3), (3, -1), (True, 0), (3.5, 0)])
def test_invalid_windows(window, sink) -> None:
    with pytest.raises(ValueError):
        pipeline()._validate_conditioning_request(
            params(kv_cache_inference_size=window, attention_sink_size=sink), None
        )


def test_chunk4_full_history_contract() -> None:
    p = pipeline(4)
    assert p._validate_conditioning_request(params(113), None).num_pixel_frames == 113
    with pytest.raises(ValueError, match="% 16"):
        p._validate_conditioning_request(params(121), None)
    with pytest.raises(ValueError, match="full history"):
        p._validate_conditioning_request(params(193), None)


@pytest.mark.parametrize("window,sink", [(1, 0), (2, 1), (4, 0), (4, 2), (30, 3), (51, 1)])
def test_rollout_history_matches_reference_visibility(window: int, sink: int) -> None:
    p = pipeline()
    state = Cosmos3NanoSimBimanualSessionState(session_id="test")
    geometry = Cosmos3NanoSimBimanualGeometry(height=32, width=32)
    total = 3 * window + 1
    request = p._validate_conditioning_request(
        params(1 + 4 * (total - 1), kv_cache_inference_size=window, attention_sink_size=sink),
        None,
    )
    conditioning = _TransferConditioning(request, torch.zeros(1, 48, total, 2, 2))
    reads = []

    def visible() -> list[int]:
        kv = state.dense_kv_by_branch.get("main")
        return [] if kv is None else [int(v) for v in kv[0][0].flatten()]

    def commit(label: int) -> None:
        kv = torch.tensor([[[float(label)]]])
        p._append_dense_kv(state, [(kv, kv)], geometry)

    def control_forward(*args, frame_start, **kwargs):
        reads.append(("control", frame_start, visible()))
        commit(2 * frame_start)

    def denoise(*args, chunk_start, **kwargs):
        for _ in range(4):
            reads.append(("denoise", chunk_start, visible()))
        return torch.zeros(1, 48, 1, 2, 2)

    def clean_commit(*args, frame_idx, **kwargs):
        reads.append(("clean", frame_idx, visible()))
        commit(2 * frame_idx + 1)

    p._transformer_forward = control_forward
    p._denoise_chunk = denoise
    p._commit_clean_frame = clean_commit
    for frame in range(total):
        p._run_chunk(
            state,
            geometry=geometry,
            chunk_start=frame,
            chunk_end=frame + 1,
            target_frame=total,
            terminal_request=True,
            request_start_frame=0,
            seed=42,
            text_kv=[],
            real_text_kv_len=1,
            fps=30,
            conditioning=conditioning,
            tick_durations={},
            measure_tick_latency=False,
        )
        assert len(visible()) <= 2 * window

    for phase, frame, actual in reads:
        # Independent temporal-frame visibility, then expand each past frame
        # to its control/latent pair. The current frame contributes only C_t.
        past = [j for j in range(frame) if j < sink or j >= frame - (window - sink - 1)]
        expected = [entry for j in past for entry in (2 * j, 2 * j + 1)]
        if phase != "control":
            expected.append(2 * frame)
        assert actual == expected, (phase, frame)
    assert not any(phase == "clean" and frame == total - 1 for phase, frame, _ in reads)


def test_emphasis_and_explicit_geometry() -> None:
    kwargs = dict(hint="depth", num_frames=121, fps=30, height=480, width=832)
    plain = format_cosmos3_nano_sim_transfer_prompt("A robot moves.", emphasize_control_in_prompt=False, **kwargs)
    emphasized = format_cosmos3_nano_sim_transfer_prompt("A robot moves.", **kwargs)
    assert "Follow the depth control video precisely" not in plain
    assert emphasized.startswith(plain)
    assert "Follow the depth control video precisely" in emphasized
    sp = SimpleNamespace(
        extra_args={"resolution": "480", "aspect_ratio": "16,9", "depth": {"control": {"height": 832, "width": 480}}}
    )
    assert resolve_cosmos3_nano_sim_transfer_geometry(sp, {}, Cosmos3NanoSimBimanualResolutionPolicy()).session_key == (
        480,
        832,
    )


def test_dense_long_text_override_does_not_mutate_artifact() -> None:
    p = pipeline()
    tokens = torch.ones(1, 1024, dtype=torch.long)
    p.transformer = SimpleNamespace(encode_und_kv=lambda *_: ([(tokens, tokens)], 1024))
    state = Cosmos3NanoSimBimanualSessionState(session_id="long")
    with pytest.raises(ValueError, match="Cosmos3NanoSimTransferPipeline prompt exceeds token limit"):
        p._ensure_text_kv(state, tokens, tokens)
    limit = p._prompt_token_limit(params(max_prompt_tokens=4096), {})
    assert p._ensure_text_kv(state, tokens, tokens, max_length=limit)[0][0].shape[1] == 1024
    assert p.manifest.text_cache_max_len == 512


def test_prompt_file_relative_to_record_and_cli_override(tmp_path: Path) -> None:
    script = (
        Path(__file__).resolve().parents[4]
        / "examples/offline_inference/cosmos3_nano_sim_bimanual/cosmos3_nano_sim_transfer.py"
    )
    spec = importlib.util.spec_from_file_location("transfer_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    caption = {"scene": "robot", "objects": ["arm", "cube"]}
    (tmp_path / "caption.json").write_text(json.dumps(caption, indent=2))
    record = {"prompt_path": "caption.json"}
    assert module._load_prompt(record, tmp_path) == json.dumps(caption)
    override = tmp_path / "caption.txt"
    override.write_text("A different scene.\n")
    assert module._load_prompt(record, tmp_path, override) == "A different scene."
