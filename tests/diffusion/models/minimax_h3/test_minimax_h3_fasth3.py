# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3.fasth3 import (
    FASTH3_BASE_MODEL,
    FASTH3_BASE_SCHEDULE,
    FASTH3_DENOISE_STEPS,
    FASTH3_FORMAT,
    FastH3AdapterError,
    FastH3WeightFusion,
    resolve_fasth3_fusion,
)
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import _reorder_grouped_qkv_to_qkv
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_RANK = 2
_HIDDEN = 4
_HEAD_DIM = 2
_HEADS = 3
_INNER = _HEAD_DIM * _HEADS  # attention inner size
_FFN = 5


def _factors(out_dim: int, in_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A rank-2 pair whose product is reproducible and non-symmetric."""
    a = torch.arange(_RANK * in_dim, dtype=torch.float32).reshape(_RANK, in_dim) / (in_dim * _RANK)
    b = torch.arange(out_dim * _RANK, dtype=torch.float32).reshape(out_dim, _RANK) / (out_dim * _RANK)
    return a, b


def _write_adapter(
    path,
    *,
    tensors=None,
    drop: str | None = None,
    blocks: int = 1,
    metadata: dict[str, str | None] | None = None,
) -> None:
    """Write an artifact in the published ``fastvideo-lora-v2`` shape."""
    payload: dict[str, torch.Tensor] = {}
    for block in range(blocks):
        for suffix, (out_dim, in_dim) in {
            "attn.to_q": (_INNER, _HIDDEN),
            "attn.to_k": (_INNER, _HIDDEN),
            "attn.to_v": (_INNER, _HIDDEN),
            "attn.to_out.0": (_HIDDEN, _INNER),
            "ff.net.0.proj": (2 * _FFN, _HIDDEN),
            "ff.net.2": (_HIDDEN, _FFN),
        }.items():
            a, b = _factors(out_dim, in_dim)
            payload[f"transformer_blocks.{block}.{suffix}.lora_A.weight"] = a
            payload[f"transformer_blocks.{block}.{suffix}.lora_B.weight"] = b
    payload.update(tensors or {})
    if drop is not None:
        payload.pop(drop, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    # The published writer records what it emitted; mirror it so the fixtures
    # are as self-describing as the real artifact.
    declared = {
        "format": FASTH3_FORMAT,
        "finetuned_model": "FastVideo/FastVideo-FastH3-Dense-4-step-v1",
        "base_model": FASTH3_BASE_MODEL,
        "rank": str(_RANK),
        "low_rank_tensors": str(sum(1 for key in payload if key.endswith((".lora_A.weight", ".lora_B.weight")))),
        "diff_tensors": str(sum(1 for key in payload if key.endswith((".diff", ".diff_b")))),
        "set_weight_tensors": str(sum(1 for key in payload if key.endswith(".set_weight"))),
    }
    # safetensors metadata is string-only, so a ``None`` override drops the key.
    declared.update(metadata or {})
    save_file(payload, str(path), metadata={key: value for key, value in declared.items() if value is not None})


def _claim(path, **kwargs) -> FastH3WeightFusion | None:
    """Whatever from_path decides, including declining the artifact."""
    kwargs.setdefault("head_dim", _HEAD_DIM)
    kwargs.setdefault("num_blocks", 1)
    kwargs.setdefault("num_refiner_blocks", 0)
    return FastH3WeightFusion.from_path(path, **kwargs)


def _load(path, **kwargs) -> FastH3WeightFusion:
    """The same, for the tests that only make sense on a claimed adapter."""
    fusion = _claim(path, **kwargs)
    assert fusion is not None, f"{path} was not claimed as a FastH3 adapter"
    return fusion


def test_only_an_artifact_carrying_the_release_identity_is_claimed(tmp_path):
    plain = tmp_path / "peft" / "adapter_model.safetensors"
    plain.parent.mkdir(parents=True)
    save_file({"transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones((_RANK, _HIDDEN))}, str(plain))
    # No fastvideo-lora-v2 metadata: this is somebody else's LoRA and has to
    # stay on the dynamic route rather than being fused.
    assert _claim(plain.parent) is None
    assert _claim(tmp_path / "missing") is None

    claimed = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(claimed)
    assert _claim(claimed.parent) is not None


@pytest.mark.parametrize(
    "identity",
    [
        # An ordinary H3 adapter out of FastVideo's own extraction tools.
        {"finetuned_model": "someone/minimax-h3-style-lora"},
        # Somebody else's adapter that merely names the student it imitates.
        {"finetuned_model": "someone/fasth3-style-lora-for-h3"},
        {"finetuned_model": ""},
        {"finetuned_model": None},
        # A FastH3 name over a different base model.
        {"base_model": "Wan-AI/Wan2.2-TI2V-5B"},
    ],
)
def test_a_generic_fastvideo_h3_adapter_stays_on_the_dynamic_route(identity, tmp_path):
    """The fixture edits every block, so only the identity separates it from FastH3."""
    path = tmp_path / "generic" / "adapter_model.safetensors"
    _write_adapter(path, metadata=identity)

    assert _claim(path.parent) is None


def test_a_claimed_artifact_must_declare_every_tensor_count(tmp_path):
    """Declining is for other people's adapters; a FastH3 file is held to its word."""
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, metadata={"diff_tensors": None})

    with pytest.raises(FastH3AdapterError, match="omits diff_tensors"):
        _claim(path.parent)


def test_the_published_bundle_root_is_refused_rather_than_guessed(tmp_path):
    root = tmp_path / "FastVideo-FastH3-4-step-Preview-v1-LoRA"
    for slug in ("dense-datafree", "vsa-datafree"):
        _write_adapter(root / slug / "adapter_model.safetensors")
    (root / "adapter_manifest.json").write_text(json.dumps({"schema_version": "fasth3-lora-bundle-v1"}))

    with pytest.raises(FastH3AdapterError, match="point --lora-path at one variant"):
        _load(root)
    # One variant inside it is unambiguous.
    assert _claim(root / "dense-datafree") is not None


def test_a_multi_shard_non_fasth3_adapter_stays_on_the_dynamic_route(tmp_path):
    # A plain PEFT LoRA saved as several shards must not hard-fail startup with a
    # FastH3-specific message; it has to fall through to the dynamic LoRA route.
    directory = tmp_path / "peft-sharded"
    directory.mkdir()
    for shard in ("adapter_model-00001-of-00002.safetensors", "adapter_model-00002-of-00002.safetensors"):
        save_file(
            {"transformer_blocks.0.attn.to_q.lora_A.weight": torch.ones((_RANK, _HIDDEN))}, str(directory / shard)
        )

    assert _claim(directory) is None


def test_low_rank_factors_reach_the_fused_projections(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    fusion = _load(path.parent)

    qkv = torch.zeros((3 * _INNER, _HIDDEN), dtype=torch.float32)
    fused = fusion.fuse("blocks.0.attn.qkv_proj.weight", qkv).cpu()
    # The checkpoint stores one head group at a time as [q, k, v], so the delta
    # has to survive the loader's own unpacking as three separate projections.
    q, k, v = torch.split(
        _reorder_grouped_qkv_to_qkv(fused, num_query_groups=_HEADS, heads_per_group=1, head_dim=_HEAD_DIM),
        [_INNER, _INNER, _INNER],
    )
    a, b = _factors(_INNER, _HIDDEN)
    expected = b @ a
    for got in (q, k, v):
        assert torch.allclose(got, expected, atol=1e-5)


def test_the_fused_mlp_delta_is_swapped_into_gate_first_order(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    fusion = _load(path.parent)

    fused = fusion.fuse("blocks.0.mlp.fc1.weight", torch.zeros((2 * _FFN, _HIDDEN))).cpu()
    a, b = _factors(2 * _FFN, _HIDDEN)
    value, gate = (b @ a).chunk(2, dim=0)
    # The diffusers export is value-first; H3's fc1 is gate-first.
    assert torch.allclose(fused, torch.cat((gate, value), dim=0), atol=1e-5)


def test_diff_and_diff_b_edit_weights_and_biases(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(
        path,
        tensors={
            "transformer_blocks.0.norm1.diff": torch.full((_HIDDEN,), 0.25),
            "transformer_blocks.0.adaln_proj.linear.diff_b": torch.full((_HIDDEN,), -0.5),
            "context_embedder.diff": torch.full((_HIDDEN, _HIDDEN), 0.125),
            "context_embedder.diff_b": torch.full((_HIDDEN,), 2.0),
        },
    )
    fusion = _load(path.parent)

    # A full-rank delta on an RMSNorm vector is exactly what a LoRA layer
    # cannot express, and why this release is fused instead of switched.
    assert torch.allclose(fusion.fuse("blocks.0.norm1.weight", torch.ones(_HIDDEN)).cpu(), torch.full((_HIDDEN,), 1.25))
    assert torch.allclose(
        fusion.fuse("blocks.0.adaln_proj.linear.bias", torch.ones(_HIDDEN)).cpu(), torch.full((_HIDDEN,), 0.5)
    )
    assert torch.allclose(
        fusion.fuse("condition_proj.weight", torch.zeros((_HIDDEN, _HIDDEN))).cpu(),
        torch.full((_HIDDEN, _HIDDEN), 0.125),
    )
    assert torch.allclose(fusion.fuse("condition_proj.bias", torch.zeros(_HIDDEN)).cpu(), torch.full((_HIDDEN,), 2.0))


def test_a_streamed_checkpoint_keeps_the_parameters_the_adapter_never_edits(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, tensors={"transformer_blocks.0.norm1.diff": torch.full((_HIDDEN,), 3.0)})
    fusion = _load(path.parent)

    untouched = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
    streamed = dict(
        fusion.apply([("blocks.0.norm1.weight", torch.zeros(_HIDDEN)), ("blocks.0.attn.q_norm.weight", untouched)])
    )
    assert torch.allclose(streamed["blocks.0.norm1.weight"].cpu(), torch.full((_HIDDEN,), 3.0))
    assert streamed["blocks.0.attn.q_norm.weight"] is untouched

    # validate_fully_applied() releases the deltas, so a second stream would
    # fuse nothing at all; it has to fail rather than serve base H3 weights.
    with pytest.raises(FastH3AdapterError, match="already been fused"):
        next(fusion.apply([("blocks.0.norm1.weight", torch.zeros(_HIDDEN))]))


def test_the_fused_result_keeps_the_parameter_dtype(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, tensors={"transformer_blocks.0.norm1.diff": torch.full((_HIDDEN,), 0.5)})
    fusion = _load(path.parent)

    fused = fusion.fuse("blocks.0.norm1.weight", torch.ones(_HIDDEN, dtype=torch.bfloat16))
    assert fused.dtype == torch.bfloat16


def test_a_vsa_variant_is_recognised_by_its_compression_gates(tmp_path):
    dense = tmp_path / "dense" / "adapter_model.safetensors"
    _write_adapter(dense)
    assert _load(dense.parent).requires_vsa is False

    sparse = tmp_path / "vsa" / "adapter_model.safetensors"
    _write_adapter(sparse, tensors={"transformer_blocks.0.attn.to_gate_compress.set_weight": torch.ones((2, 2))})
    assert _load(sparse.parent).requires_vsa is True


def test_a_tensor_naming_no_h3_parameter_is_an_error(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, tensors={"transformer_blocks.0.attn.norm_q.lora_A.weight": torch.ones((_RANK, _HIDDEN))})
    # Dropping it silently would load a model that is not the distilled student.
    with pytest.raises(FastH3AdapterError, match="name no known"):
        _load(path.parent)


def test_an_unpaired_low_rank_factor_is_an_error(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, drop="transformer_blocks.0.attn.to_q.lora_B.weight")

    with pytest.raises(FastH3AdapterError, match="unpaired factor"):
        _load(path.parent)


def test_a_delta_the_checkpoint_never_offered_is_reported(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    fusion = _load(path.parent)

    fusion.fuse("blocks.0.attn.qkv_proj.weight", torch.zeros((3 * _INNER, _HIDDEN)))
    # An unapplied delta is the failure that matters: the model would load and
    # generate, just not as the distilled student.
    with pytest.raises(FastH3AdapterError, match="never provided"):
        fusion.validate_fully_applied()

    for name, shape in (
        ("blocks.0.attn.out_proj.weight", (_HIDDEN, _INNER)),
        ("blocks.0.mlp.fc1.weight", (2 * _FFN, _HIDDEN)),
        ("blocks.0.mlp.fc2.weight", (_HIDDEN, _FFN)),
    ):
        fusion.fuse(name, torch.zeros(shape))
    fusion.validate_fully_applied()


def _pipeline_stub(fusion=None):
    from vllm_omni.diffusion.models.minimax_h3 import MiniMaxH3Pipeline

    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.partition = "combined"
    pipeline.supported_tasks = frozenset({"t2va", "fl2va", "ref2va"})
    pipeline._turbo_lora_adapter_ids = set()
    pipeline._fasth3 = fusion
    pipeline.od_config = SimpleNamespace()
    pipeline.default_video_shift, pipeline.default_audio_shift = 12.0, 3.0
    return pipeline


def _sampling(steps=FASTH3_DENOISE_STEPS, **kwargs) -> SimpleNamespace:
    fields = {"num_inference_steps": steps, "extra_args": {}, "lora_request": None}
    return SimpleNamespace(**{**fields, **kwargs})


def _check_request(fusion, sampling) -> None:
    fusion.check_request(sampling, video_shift=12.0, audio_shift=3.0)


def _fusion(tmp_path, **kwargs) -> FastH3WeightFusion:
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, **kwargs)
    return _load(path.parent)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_an_active_fasth3_fusion_restricts_requests_to_t2va(task, tmp_path):
    pipeline = _pipeline_stub(_fusion(tmp_path))

    with pytest.raises(OmniClientError, match="distills \\['t2va'\\] only"):
        pipeline._resolve_task(task, {})
    assert pipeline._resolve_task("t2va", {}) == "t2va"
    assert _pipeline_stub()._resolve_task(task, {}) == task


@pytest.mark.parametrize("num_inference_steps", [None, 5, 50])
def test_fasth3_requires_one_forward_per_sigma_interval(num_inference_steps, tmp_path):
    fusion = _fusion(tmp_path)

    # The five sigma points bound four forwards, and a request states forwards:
    # the interval contract the pinned schedules use, and the count the step
    # scheduler admits a request on before any pipeline hook runs.
    with pytest.raises(OmniClientError, match=f"num_inference_steps={FASTH3_DENOISE_STEPS}"):
        _check_request(fusion, _sampling(num_inference_steps))
    _check_request(fusion, _sampling())
    # The pinned branch coerces the string form; this one cannot disagree.
    _check_request(fusion, _sampling(str(FASTH3_DENOISE_STEPS)))


def test_the_request_denoises_on_the_release_ladder(tmp_path):
    pipeline = _pipeline_stub(_fusion(tmp_path))

    positions, num_steps = pipeline._resolve_sigma_positions("t2va", _sampling())

    # dmd_denoising_steps [999, 749, 500, 250] out of 1000, closed with the
    # terminal 0.0: the pre-shift positions the student was distilled at, not
    # the uniform ladder num_inference_steps would otherwise derive.
    assert positions == (0.999, 0.749, 0.5, 0.25, 0.0)
    # Five sigma points bound four transformer forwards, and forwards is the
    # unit num_steps carries downstream (Cache-DiT, quality hints).
    assert num_steps == FASTH3_DENOISE_STEPS == len(positions) - 1
    # H3's own per-modality shifts still apply on top, and are what turn those
    # positions into the noise levels the four forwards run at.
    assert minimax_h3_time_shift_sigmas(shift_scale=12.0, base_schedule=positions) == pytest.approx(
        [0.999917, 0.972833, 0.923077, 0.8, 0.0], abs=1e-6
    )
    assert minimax_h3_time_shift_sigmas(shift_scale=3.0, base_schedule=positions) == pytest.approx(
        [0.999666, 0.899520, 0.75, 0.5, 0.0], abs=1e-6
    )
    # Without a fused adapter the same request keeps the undistilled ladder.
    assert _pipeline_stub()._resolve_sigma_positions("t2va", _sampling()) == (None, FASTH3_DENOISE_STEPS)


def test_the_contract_leaves_the_checkpoint_shifts_alone(tmp_path):
    pipeline = _pipeline_stub(_fusion(tmp_path))
    pipeline.partition = "fl2va"

    pipeline._fasth3.check_serving_contract(
        partition=pipeline.partition,
        od_config=pipeline.od_config,
        video_shift=pipeline.default_video_shift,
        audio_shift=pipeline.default_audio_shift,
    )

    # The ladder is stated pre-shift, so H3's 12/3 shifts are what place the
    # student on the levels it was distilled at; the contract must not touch them.
    assert pipeline.default_video_shift == 12.0
    assert pipeline.default_audio_shift == 3.0


@pytest.mark.parametrize(
    "extra",
    [{"flow_shift": 1.0}, {"audio_flow_shift": 1.0}, {"flow_shift": "bad"}],
)
def test_a_request_may_not_move_the_student_off_its_rungs(extra, tmp_path):
    fusion = _fusion(tmp_path)

    with pytest.raises(OmniClientError, match="FastH3 requires"):
        _check_request(fusion, _sampling(extra_args=extra))

    _check_request(fusion, _sampling(extra_args={"flow_shift": 12.0, "audio_flow_shift": 3.0}))


def test_a_request_may_not_carry_a_lora_on_a_fused_server(tmp_path):
    fusion = _fusion(tmp_path)

    # The dynamic LoRA manager is skipped for a fused adapter, so an adapter
    # asked for per request would be neither applied nor reported.
    with pytest.raises(OmniClientError, match="per-request lora is unavailable"):
        _check_request(fusion, _sampling(lora_request=SimpleNamespace(lora_int_id=1)))


def test_offload_is_refused_because_it_bypasses_the_fusion(tmp_path):
    fusion = _fusion(tmp_path)
    for flag in ("enable_cpu_offload", "enable_layerwise_offload", "enable_distributed_layerwise_offload"):
        # A host-weight plan installs the transformer without load_weights(),
        # so the fusion and its completeness check would both be skipped.
        with pytest.raises(ValueError, match="cannot be combined with"):
            fusion.check_serving_contract(
                partition="fl2va",
                od_config=SimpleNamespace(**{flag: True}),
                video_shift=12.0,
                audio_shift=3.0,
            )


def test_adopting_the_contract_refuses_a_vsa_variant_and_ref2va(tmp_path):
    sparse = tmp_path / "vsa" / "adapter_model.safetensors"
    _write_adapter(sparse, tensors={"transformer_blocks.0.attn.to_gate_compress.set_weight": torch.ones((2, 2))})
    contract = {"od_config": SimpleNamespace(), "video_shift": 12.0, "audio_shift": 3.0}
    with pytest.raises(ValueError, match="Video Sparse Attention variant"):
        _load(sparse.parent).check_serving_contract(partition="fl2va", **contract)

    with pytest.raises(ValueError, match="cannot serve a Ref2VA partition"):
        _fusion(tmp_path).check_serving_contract(partition="ref2va", **contract)


def test_a_full_rank_delta_on_a_fused_parameter_is_refused(tmp_path):
    # Only low-rank factors are placed into H3's fused QKV and gate/up layouts,
    # so a .diff aimed at one would otherwise be added transposed.
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, tensors={"transformer_blocks.0.attn.to_q.diff": torch.ones((_INNER, _HIDDEN))})
    with pytest.raises(FastH3AdapterError, match="fused layout"):
        _claim(path.parent)


def test_an_adapter_that_leaves_blocks_untouched_is_refused(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, blocks=2)

    # Every tensor in this file is well formed and pairs up, so the fusion's
    # own completeness check passes; only the model's depth reveals that a
    # third of it would have been served as base H3 weights.
    with pytest.raises(FastH3AdapterError, match="missing=\\[2\\]"):
        _claim(path.parent, num_blocks=3)
    with pytest.raises(FastH3AdapterError, match="unknown=\\[1\\]"):
        _claim(path.parent, num_blocks=1)
    assert _claim(path.parent, num_blocks=2) is not None


def test_an_adapter_shorter_than_it_declares_is_refused(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path, metadata={"low_rank_tensors": "724", "diff_tensors": "85"})

    # The writer records what it emitted, which is the file's one statement
    # about its own completeness.
    with pytest.raises(FastH3AdapterError, match="declares low_rank_tensors=724 but carries 12"):
        _claim(path.parent)


def test_only_a_configured_adapter_path_is_read(tmp_path):
    path = tmp_path / "fasth3" / "adapter_model.safetensors"
    _write_adapter(path)
    transformer = SimpleNamespace(
        arch=SimpleNamespace(attention_head_dim=_HEAD_DIM, num_layers=1, token_refiner_num_layers=0)
    )
    unbuilt = object()  # a transformer whose arch must not be touched

    assert resolve_fasth3_fusion(SimpleNamespace(lora_path=str(path)), transformer) is not None
    assert resolve_fasth3_fusion(SimpleNamespace(lora_path=[str(path)]), transformer) is not None
    # Nothing configured, or more than one artifact, stays on the dynamic route
    # without reading anything off the model.
    assert resolve_fasth3_fusion(SimpleNamespace(lora_path=None), unbuilt) is None
    assert resolve_fasth3_fusion(SimpleNamespace(), unbuilt) is None
    assert resolve_fasth3_fusion(SimpleNamespace(lora_path=[str(path), str(path)]), unbuilt) is None


def test_a_pipeline_that_fused_its_adapter_needs_no_lora_manager(tmp_path):
    assert _pipeline_stub(_fusion(tmp_path)).lora_is_fused is True
    assert _pipeline_stub().lora_is_fused is False


def test_the_ladder_is_the_one_the_release_publishes():
    # FASTH3_DENOISE_STEPS is derived from it, so the two cannot drift apart.
    assert FASTH3_BASE_SCHEDULE.base_schedule == (0.999, 0.749, 0.5, 0.25, 0.0)
    assert FASTH3_BASE_SCHEDULE.num_inference_steps == FASTH3_DENOISE_STEPS == 4
