# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import hashlib
import multiprocessing as mp
import time
from pathlib import Path
from types import MethodType
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

import vllm_omni.diffusion.models.minimax_h3.temporal_chunks as temporal_chunks_module
import vllm_omni.diffusion.models.minimax_h3.vae as vae_module
from vllm_omni.diffusion.models.minimax_h3.temporal_chunks import (
    MINIMAX_H3_RELEASED_BUNDLE_SHA256,
)
from vllm_omni.diffusion.models.minimax_h3.vae import (
    MiniMaxH3ChunkCallbackPeerError,
    MiniMaxH3ChunkedDecodeUnsupportedError,
    MiniMaxH3VideoVAE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _IdentityProcessor:
    @staticmethod
    def revert_tensor(tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class _FakeCompatTemporalModel(nn.Module):
    use_3d_conv = True
    clip_length = 17
    tokens_chunk_size = 5
    token_overlap = 2
    token_drop = 3
    vae_ratio_t = 4
    frame_pre_padding = 3
    frame_overlap = 5
    isolated_first_frame = False
    isolated_last_frame = False
    vae_ratio = 1

    def __init__(
        self,
        latent_t: int,
        *,
        collective: bool = False,
        tile_count: int = 2,
    ) -> None:
        super().__init__()
        self.parallel_tiling = collective
        self.collective = collective
        self.tile_count = tile_count
        pseudo_tokens = latent_t + self.token_drop
        remainder = pseudo_tokens % self.tokens_chunk_size
        self.pad_tokens = 0 if remainder == 0 else self.tokens_chunk_size - remainder
        self.num_chunks = (pseudo_tokens + self.pad_tokens) // self.tokens_chunk_size - 1
        ideal_frames = self.num_chunks * 17 + 5
        output_frames = {
            33: 111,
            34: 115,
            35: 119,
            36: 120,
        }.get(latent_t, ideal_frames)
        values = torch.linspace(0, 1, ideal_frames).view(1, 1, ideal_frames, 1, 1)
        self.ideal = values.expand(1, 3, -1, 1, 1).contiguous()
        self.expected = self.ideal[:, :, :output_frames].clone()
        self.processor = _IdentityProcessor()
        self.decode_base_calls = 0
        self.adaptive_decode_calls = 0
        self.blend_calls = 0
        self.eval()

    def split_tiles(self, size: int, tiled: bool):
        del size, tiled
        return list(range(self.tile_count)), None, None

    def decode_base(self, latent: torch.Tensor) -> torch.Tensor:
        del latent
        self.decode_base_calls += 1
        return self.expected.clone()

    def _adaptive_decode(self, latent: torch.Tensor) -> torch.Tensor:
        del latent
        call_index = self.adaptive_decode_calls % self.num_chunks
        self.adaptive_decode_calls += 1
        if self.collective and self.parallel_tiling:
            collective_probe = torch.tensor([call_index], dtype=torch.int32)
            dist.all_reduce(collective_probe)
        decoded = torch.full((1, 3, 28, 1, 1), -1.0)
        body_start = call_index * 17
        decoded[:, :, 3:20].copy_(self.ideal[:, :, body_start : body_start + 17])
        if call_index == self.num_chunks - 1:
            decoded[:, :, 23:28].copy_(self.ideal[:, :, -5:])
        return decoded

    def _decode_temporal_output_frame_plan(
        self,
        latent: torch.Tensor,
        head: torch.Tensor | None,
        tail: torch.Tensor | None,
        num_chunks: int,
        pad_tokens: int,
    ) -> tuple[int, int, int]:
        del latent
        assert head is None
        assert tail is None
        assert num_chunks == self.num_chunks
        assert pad_tokens == self.pad_tokens
        total_frames = num_chunks * 17 + 5
        pad_frames = total_frames - int(self.expected.shape[2])
        return total_frames, pad_frames, int(self.expected.shape[2])

    def _decode_temporal_pad_frames(
        self,
        latent: torch.Tensor,
        pad_tokens: int,
    ) -> int:
        del latent
        assert pad_tokens == self.pad_tokens
        return int(self.ideal.shape[2] - self.expected.shape[2])

    def blend(
        self,
        previous: torch.Tensor,
        current: torch.Tensor,
        extent: int,
        *,
        dim: int,
    ) -> torch.Tensor:
        del previous
        assert extent == self.frame_overlap
        assert dim == -3
        self.blend_calls += 1
        return current


def _install_released_source_fingerprint() -> None:
    temporal_chunks_module._source_bundle_sha256 = lambda model: (
        "fake/video_vae",
        MINIMAX_H3_RELEASED_BUNDLE_SHA256,
    )


def _allow_released_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        temporal_chunks_module,
        "_source_bundle_sha256",
        lambda model: (
            "fake/video_vae",
            MINIMAX_H3_RELEASED_BUNDLE_SHA256,
        ),
    )


def _compat_vae(
    latent_t: int,
    *,
    parallel_size: int = 1,
    collective: bool = False,
    tile_count: int = 2,
) -> tuple[MiniMaxH3VideoVAE, _FakeCompatTemporalModel]:
    vae = object.__new__(MiniMaxH3VideoVAE)
    nn.Module.__init__(vae)
    model = _FakeCompatTemporalModel(
        latent_t,
        collective=collective,
        tile_count=tile_count,
    )
    vae.model = model
    vae.config_dict = {
        "latent_channels": 3,
        "latents_mean": [0.0, 0.0, 0.0],
        "latents_std": [1.0, 1.0, 1.0],
    }
    vae.parallel_size = parallel_size
    vae._chunk_decode_coordinator = None
    return vae, model


def _latent(latent_t: int = 37) -> torch.Tensor:
    return torch.zeros(1, 3, latent_t, 1, 1)


def test_default_decode_path_does_not_request_chunks() -> None:
    vae, model = _compat_vae(37)

    output = vae.decode_latent(_latent())

    torch.testing.assert_close(output, model.expected, rtol=0, atol=0)
    assert model.decode_base_calls == 1
    assert model.adaptive_decode_calls == 0


def test_default_decode_bypasses_compatibility_source_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vae, model = _compat_vae(37)
    monkeypatch.setattr(
        temporal_chunks_module,
        "_source_bundle_sha256",
        lambda model: pytest.fail("default decode fingerprinted remote code"),
    )

    output = vae.decode_latent(_latent())

    torch.testing.assert_close(output, model.expected, rtol=0, atol=0)
    assert model.decode_base_calls == 1
    assert model.adaptive_decode_calls == 0


@pytest.mark.parametrize(
    ("latent_t", "expected_chunk_sizes"),
    [
        (37, [17] * 7 + [5]),
        (62, [17] * 12 + [5]),
        (72, [17] * 14 + [5]),
        (107, [17] * 21 + [5]),
        (36, [17] * 7 + [1]),
        (35, [17] * 7),
        (34, [17] * 6 + [13]),
        (33, [17] * 6 + [9]),
    ],
)
def test_released_compatibility_chunks_reconstruct_full_decode(
    monkeypatch: pytest.MonkeyPatch,
    latent_t: int,
    expected_chunk_sizes: list[int],
) -> None:
    _allow_released_source(monkeypatch)
    vae, model = _compat_vae(latent_t)
    chunks: list[torch.Tensor] = []
    snapshots: list[torch.Tensor] = []
    metadata: list[tuple[int, int, int, bool]] = []

    def consume(
        frames: torch.Tensor,
        *,
        chunk_index: int,
        total_chunks: int,
        frame_start: int,
        is_final: bool,
    ) -> None:
        assert frames.dtype is torch.float32
        assert frames.is_contiguous()
        assert torch.all((0 <= frames) & (frames <= 1))
        chunks.append(frames)
        snapshots.append(frames.clone())
        metadata.append((chunk_index, total_chunks, frame_start, is_final))

    output = vae.decode_latent_with_chunks(_latent(latent_t), consume)

    torch.testing.assert_close(output, model.expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=2), output, rtol=0, atol=0)
    assert [int(chunk.shape[2]) for chunk in chunks] == expected_chunk_sizes
    expected_metadata = []
    frame_start = 0
    for chunk_index, chunk_size in enumerate(expected_chunk_sizes):
        expected_metadata.append(
            (
                chunk_index,
                len(expected_chunk_sizes),
                frame_start,
                chunk_index == len(expected_chunk_sizes) - 1,
            )
        )
        frame_start += chunk_size
    assert metadata == expected_metadata
    for chunk, snapshot in zip(chunks, snapshots, strict=True):
        torch.testing.assert_close(chunk, snapshot, rtol=0, atol=0)
    assert len({chunk.untyped_storage().data_ptr() for chunk in chunks}) == len(chunks)
    assert model.adaptive_decode_calls == model.num_chunks


def test_compat_callback_mutation_does_not_change_returned_full_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_released_source(monkeypatch)
    vae, model = _compat_vae(37)

    def mutate(frames: torch.Tensor, **metadata: Any) -> None:
        del metadata
        frames.fill_(1234)

    output = vae.decode_latent_with_chunks(_latent(), mutate)

    torch.testing.assert_close(output, model.expected, rtol=0, atol=0)


def test_compat_source_fingerprint_is_cached_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fingerprint(model):
        nonlocal calls
        del model
        calls += 1
        return "fake/video_vae", MINIMAX_H3_RELEASED_BUNDLE_SHA256

    monkeypatch.setattr(
        temporal_chunks_module,
        "_source_bundle_sha256",
        fingerprint,
    )
    vae, model = _compat_vae(37)

    for _ in range(2):
        output = vae.decode_latent_with_chunks(
            _latent(),
            lambda *args, **kwargs: None,
        )
        torch.testing.assert_close(output, model.expected, rtol=0, atol=0)

    assert calls == 1


def test_compat_callback_publishes_after_temporal_blending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_released_source(monkeypatch)
    vae, model = _compat_vae(37)
    chunks: list[torch.Tensor] = []

    def mark_overlap(
        previous: torch.Tensor,
        current: torch.Tensor,
        extent: int,
        *,
        dim: int,
    ) -> torch.Tensor:
        del previous
        assert extent == 5
        assert dim == -3
        blended = current.clone()
        blended[:, :, :extent].fill_(0.25)
        return blended

    monkeypatch.setattr(model, "blend", mark_overlap)
    output = vae.decode_latent_with_chunks(
        _latent(),
        lambda frames, **metadata: chunks.append(frames.clone()),
    )

    expected_overlap = torch.full_like(chunks[1][:, :, :5], 0.25)
    torch.testing.assert_close(chunks[1][:, :, :5], expected_overlap, rtol=0, atol=0)
    torch.testing.assert_close(output[:, :, 17:22], expected_overlap, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(chunks, dim=2), output, rtol=0, atol=0)


@pytest.mark.parametrize("fail_final", [False, True])
def test_compat_callback_failure_finishes_decode_and_recovers(
    monkeypatch: pytest.MonkeyPatch,
    fail_final: bool,
) -> None:
    _allow_released_source(monkeypatch)
    vae, model = _compat_vae(37)
    seen: list[tuple[int, bool]] = []

    def fail(
        frames: torch.Tensor,
        *,
        chunk_index: int,
        total_chunks: int,
        frame_start: int,
        is_final: bool,
    ) -> None:
        del frames, total_chunks, frame_start
        seen.append((chunk_index, is_final))
        if is_final == fail_final:
            raise LookupError("compat sink failed")

    with pytest.raises(LookupError, match="compat sink failed"):
        vae.decode_latent_with_chunks(_latent(), fail)

    assert model.adaptive_decode_calls == model.num_chunks
    if fail_final:
        assert seen == [(index, index == 7) for index in range(8)]
    else:
        assert seen == [(0, False)]

    recovered_chunks: list[torch.Tensor] = []
    recovered = vae.decode_latent_with_chunks(
        _latent(),
        lambda frames, **metadata: recovered_chunks.append(frames),
    )
    torch.testing.assert_close(recovered, model.expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.cat(recovered_chunks, dim=2),
        model.expected,
        rtol=0,
        atol=0,
    )
    assert model.adaptive_decode_calls == model.num_chunks * 2


def test_native_decoder_failure_propagates_on_single_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _allow_released_source(monkeypatch)
    vae, model = _compat_vae(37)
    callback_calls = 0

    def fail_decode(latent: torch.Tensor) -> torch.Tensor:
        del latent
        raise LookupError("native decoder failed")

    def consume(*args, **kwargs) -> None:
        nonlocal callback_calls
        del args, kwargs
        callback_calls += 1

    monkeypatch.setattr(model, "_adaptive_decode", fail_decode)
    with pytest.raises(LookupError, match="native decoder failed"):
        vae.decode_latent_with_chunks(_latent(), consume)

    assert callback_calls == 0


@pytest.mark.parametrize("drift", ["changed", "missing"])
def test_compat_source_drift_fails_closed_before_decode(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    _allow_released_source(monkeypatch)
    if drift == "changed":
        monkeypatch.setattr(
            temporal_chunks_module,
            "_source_bundle_sha256",
            lambda model: ("fake/video_vae", "0" * 64),
        )
    else:
        monkeypatch.setattr(
            temporal_chunks_module,
            "_source_bundle_sha256",
            lambda model: (None, None),
        )
    vae, model = _compat_vae(37)

    with pytest.raises(
        MiniMaxH3ChunkedDecodeUnsupportedError,
        match="42ed227e",
    ):
        vae.decode_latent_with_chunks(_latent(), lambda *args, **kwargs: None)

    assert model.adaptive_decode_calls == 0
    torch.testing.assert_close(vae.decode_latent(_latent()), model.expected, rtol=0, atol=0)


def test_uninspectable_compat_source_maps_to_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        temporal_chunks_module.inspect,
        "getsourcefile",
        lambda object_type: (_ for _ in ()).throw(TypeError("no source")),
    )
    vae, model = _compat_vae(37)

    with pytest.raises(MiniMaxH3ChunkedDecodeUnsupportedError, match="bundle=None"):
        vae.decode_latent_with_chunks(_latent(), lambda *args, **kwargs: None)

    assert model.adaptive_decode_calls == 0


def test_source_bundle_manifest_is_deterministic_and_excludes_init(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = tmp_path / "video_vae"
    package.mkdir()
    (package / "z.py").write_text("z = 1\n", encoding="utf-8")
    (package / "klvae.py").write_text("k = 2\n", encoding="utf-8")
    (package / "a.py").write_text("a = 3\n", encoding="utf-8")
    (package / "__init__.py").write_text("ignored = 1\n", encoding="utf-8")
    monkeypatch.setattr(
        temporal_chunks_module.inspect,
        "getsourcefile",
        lambda object_type: str(package / "klvae.py"),
    )

    expected = hashlib.sha256()
    for filename in ("a.py", "klvae.py", "z.py"):
        expected.update(filename.encode())
        expected.update(b"\0")
        expected.update((package / filename).read_bytes())
        expected.update(b"\0")
    _, first = temporal_chunks_module._source_bundle_sha256(nn.Linear(1, 1))
    assert first == expected.hexdigest()

    (package / "__init__.py").write_text("ignored = 2\n", encoding="utf-8")
    _, after_init = temporal_chunks_module._source_bundle_sha256(nn.Linear(1, 1))
    assert after_init == first
    (package / "a.py").write_text("a = 4\n", encoding="utf-8")
    _, after_change = temporal_chunks_module._source_bundle_sha256(nn.Linear(1, 1))
    assert after_change != first


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("tokens_chunk_size", 6),
        ("token_overlap", 1),
        ("isolated_first_frame", True),
        ("isolated_last_frame", True),
    ],
)
def test_compat_config_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: object,
) -> None:
    _allow_released_source(monkeypatch)
    vae, model = _compat_vae(37)
    setattr(model, attribute, value)

    with pytest.raises(
        MiniMaxH3ChunkedDecodeUnsupportedError,
        match="released temporal configuration",
    ):
        vae.decode_latent_with_chunks(_latent(), lambda *args, **kwargs: None)

    assert model.adaptive_decode_calls == 0
    torch.testing.assert_close(vae.decode_latent(_latent()), model.expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["rank", "empty_time", "training", "non_3d"])
def test_compat_structural_invalids_fail_before_decode(
    monkeypatch: pytest.MonkeyPatch,
    invalid: str,
) -> None:
    _allow_released_source(monkeypatch)
    vae, model = _compat_vae(37)
    latent = _latent()
    if invalid == "rank":
        latent = latent[:, :, 0]
    elif invalid == "empty_time":
        latent = latent[:, :, :0]
    elif invalid == "training":
        model.train()
    else:
        model.use_3d_conv = False

    with pytest.raises(MiniMaxH3ChunkedDecodeUnsupportedError):
        vae.decode_latent_with_chunks(latent, lambda *args, **kwargs: None)

    assert model.adaptive_decode_calls == 0


def _distributed_worker(
    rank: int,
    init_file: str,
    result_queue: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        vae_module.get_data_parallel_world_size = lambda: 1
        vae_module.model_parallel_is_initialized = lambda: True

        def attach_group(vae: MiniMaxH3VideoVAE) -> None:
            vae._native_parallel_state = MethodType(
                lambda self: {"sp_process_group": dist.group.WORLD},
                vae,
            )

        _install_released_source_fingerprint()
        source_vae, source_model = _compat_vae(
            37,
            parallel_size=2,
            collective=True,
        )
        attach_group(source_vae)
        if rank == 1:
            temporal_chunks_module._source_bundle_sha256 = lambda model: (
                "fake/video_vae",
                "0" * 64,
            )
        try:
            source_vae.decode_latent_with_chunks(
                _latent(),
                (lambda *args, **kwargs: None) if rank == 0 else None,
            )
        except BaseException as exc:  # noqa: BLE001
            source_mismatch_type = type(exc).__name__
        else:
            source_mismatch_type = "none"
        source_decode_calls = source_model.adaptive_decode_calls

        _install_released_source_fingerprint()
        planner_vae, planner_model = _compat_vae(
            37,
            parallel_size=2,
            collective=True,
        )
        attach_group(planner_vae)
        if rank == 1:

            def fail_plan(self, *args, **kwargs):
                del self, args, kwargs
                raise RuntimeError("rank-local planner failed")

            setattr(
                planner_model,
                "_decode_temporal_output_frame_plan",
                MethodType(fail_plan, planner_model),
            )
        try:
            planner_vae.decode_latent_with_chunks(
                _latent(),
                (lambda *args, **kwargs: None) if rank == 0 else None,
            )
        except BaseException as exc:  # noqa: BLE001
            planner_failure_type = type(exc).__name__
        else:
            planner_failure_type = "none"
        planner_decode_calls = planner_model.adaptive_decode_calls

        dtype_vae, dtype_model = _compat_vae(
            37,
            parallel_size=2,
            collective=True,
        )
        attach_group(dtype_vae)
        temporal_chunks_module.resolve_minimax_h3_temporal_cat_dtype = (
            (lambda model: torch.float16) if rank == 0 else (lambda model: None)
        )
        try:
            dtype_vae.decode_latent_with_chunks(
                _latent(),
                (lambda *args, **kwargs: None) if rank == 0 else None,
            )
        except BaseException as exc:  # noqa: BLE001
            dtype_mismatch_type = type(exc).__name__
        else:
            dtype_mismatch_type = "none"
        dtype_decode_calls = dtype_model.adaptive_decode_calls
        temporal_chunks_module.resolve_minimax_h3_temporal_cat_dtype = lambda model: None

        vae_module.get_data_parallel_world_size = lambda: 2
        dp_vae, dp_model = _compat_vae(37, parallel_size=2, collective=True)
        attach_group(dp_vae)
        try:
            dp_vae.decode_latent_with_chunks(
                _latent(),
                (lambda *args, **kwargs: None) if rank == 0 else None,
            )
        except BaseException as exc:  # noqa: BLE001
            dp_vae_type = type(exc).__name__
        else:
            dp_vae_type = "none"
        dp_decode_calls = dp_model.adaptive_decode_calls
        vae_module.get_data_parallel_world_size = lambda: 1

        vae, model = _compat_vae(37, parallel_size=2, collective=True)
        attach_group(vae)

        try:
            vae.decode_latent_with_chunks(
                _latent(),
                lambda *args, **kwargs: None,
            )
        except BaseException as exc:  # noqa: BLE001
            invalid_owner_type = type(exc).__name__
        else:
            invalid_owner_type = "none"

        def fail(*args, **kwargs) -> None:
            del args, kwargs
            raise LookupError("distributed sink failed")

        try:
            vae.decode_latent_with_chunks(
                _latent(),
                fail if rank == 0 else None,
            )
        except BaseException as exc:  # noqa: BLE001
            failure_type = type(exc).__name__
        else:
            failure_type = "none"

        success_calls = 0

        def succeed(*args, **kwargs) -> None:
            nonlocal success_calls
            del args, kwargs
            success_calls += 1

        output = vae.decode_latent_with_chunks(
            _latent(),
            succeed if rank == 0 else None,
        )

        fallback_vae, fallback_model = _compat_vae(
            37,
            parallel_size=2,
            collective=True,
            tile_count=1,
        )
        attach_group(fallback_vae)
        fallback_calls = 0

        def consume_fallback(*args, **kwargs) -> None:
            nonlocal fallback_calls
            del args, kwargs
            fallback_calls += 1

        fallback_output = fallback_vae.decode_latent_with_chunks(
            _latent(),
            consume_fallback if rank == 0 else None,
        )

        local_vae, local_model = _compat_vae(37, parallel_size=1)
        local_calls = 0

        def consume_local(*args, **kwargs) -> None:
            nonlocal local_calls
            del args, kwargs
            local_calls += 1

        local_output = local_vae.decode_latent_with_chunks(
            _latent(),
            consume_local if rank == 0 else None,
        )

        recovery = torch.tensor([rank + 1], dtype=torch.int32)
        dist.all_reduce(recovery)
        result_queue.put(
            {
                "rank": rank,
                "source_mismatch_type": source_mismatch_type,
                "source_decode_calls": source_decode_calls,
                "planner_failure_type": planner_failure_type,
                "planner_decode_calls": planner_decode_calls,
                "dtype_mismatch_type": dtype_mismatch_type,
                "dtype_decode_calls": dtype_decode_calls,
                "dp_vae_type": dp_vae_type,
                "dp_decode_calls": dp_decode_calls,
                "invalid_owner_type": invalid_owner_type,
                "failure_type": failure_type,
                "adaptive_calls": model.adaptive_decode_calls,
                "success_calls": success_calls,
                "output_matches": torch.equal(output, model.expected),
                "fallback_calls": fallback_calls,
                "fallback_decode_calls": fallback_model.adaptive_decode_calls,
                "fallback_output_matches": torch.equal(
                    fallback_output,
                    fallback_model.expected,
                ),
                "fallback_tiling_restored": fallback_model.parallel_tiling,
                "local_calls": local_calls,
                "local_output_matches": torch.equal(local_output, local_model.expected),
                "recovery": int(recovery.item()),
            }
        )
    finally:
        dist.destroy_process_group()


def _mismatched_plan_worker(
    rank: int,
    init_file: str,
    result_queue: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        vae_module.get_data_parallel_world_size = lambda: 1
        vae_module.model_parallel_is_initialized = lambda: True
        _install_released_source_fingerprint()

        vae, model = _compat_vae(
            37 if rank == 0 else 62,
            parallel_size=2,
            collective=True,
        )
        vae._native_parallel_state = MethodType(
            lambda self: {"sp_process_group": dist.group.WORLD},
            vae,
        )
        try:
            vae.decode_latent_with_chunks(
                _latent(37 if rank == 0 else 62),
                (lambda *args, **kwargs: None) if rank == 0 else None,
            )
        except BaseException as exc:  # noqa: BLE001
            failure_type = type(exc).__name__
        else:
            failure_type = "none"

        recovery = torch.tensor([rank + 1], dtype=torch.int32)
        dist.all_reduce(recovery)
        result_queue.put(
            {
                "rank": rank,
                "failure_type": failure_type,
                "adaptive_calls": model.adaptive_decode_calls,
                "recovery": int(recovery.item()),
            }
        )
    finally:
        dist.destroy_process_group()


def _wrong_owner_worker(
    rank: int,
    init_file: str,
    result_queue: Any,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=3,
    )
    try:
        vae_module.get_data_parallel_world_size = lambda: 1
        vae_module.model_parallel_is_initialized = lambda: True
        _install_released_source_fingerprint()
        vae, model = _compat_vae(
            37,
            parallel_size=3,
            collective=True,
            tile_count=3,
        )
        vae._native_parallel_state = MethodType(
            lambda self: {"sp_process_group": dist.group.WORLD},
            vae,
        )

        try:
            vae.decode_latent_with_chunks(
                _latent(),
                (lambda *args, **kwargs: None) if rank == 1 else None,
            )
        except BaseException as exc:  # noqa: BLE001
            failure_type = type(exc).__name__
        else:
            failure_type = "none"

        recovery = torch.tensor([rank + 1], dtype=torch.int32)
        dist.all_reduce(recovery)
        result_queue.put(
            {
                "rank": rank,
                "failure_type": failure_type,
                "adaptive_calls": model.adaptive_decode_calls,
                "recovery": int(recovery.item()),
            }
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parallel
def test_three_rank_wrong_callback_owner_is_symmetric_and_group_recovers(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "gloo-wrong-owner-init")
    processes = [
        context.Process(
            target=_wrong_owner_worker,
            args=(rank, init_file, result_queue),
        )
        for rank in range(3)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 90
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.join(timeout=1)
    hung = [process for process in processes if process.is_alive()]
    for process in hung:
        process.terminate()
        process.join(timeout=5)
    assert not hung, "three-rank callback-owner validation deadlocked"
    assert [process.exitcode for process in processes] == [0, 0, 0]

    results = sorted(
        [result_queue.get(timeout=5) for _ in range(3)],
        key=lambda result: result["rank"],
    )
    assert [result["failure_type"] for result in results] == [
        "ValueError",
        "ValueError",
        "ValueError",
    ]
    assert [result["adaptive_calls"] for result in results] == [0, 0, 0]
    assert [result["recovery"] for result in results] == [6, 6, 6]


@pytest.mark.parallel
def test_two_rank_mismatched_decode_plans_fail_before_decoder_collectives(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "gloo-mismatched-plan-init")
    processes = [
        context.Process(
            target=_mismatched_plan_worker,
            args=(rank, init_file, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 90
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.join(timeout=1)
    hung = [process for process in processes if process.is_alive()]
    for process in hung:
        process.terminate()
        process.join(timeout=5)
    assert not hung, "mismatched MiniMax-H3 decode plans deadlocked"
    assert [process.exitcode for process in processes] == [0, 0]

    results = sorted(
        [result_queue.get(timeout=5) for _ in range(2)],
        key=lambda result: result["rank"],
    )
    assert [result["failure_type"] for result in results] == [
        MiniMaxH3ChunkedDecodeUnsupportedError.__name__,
        MiniMaxH3ChunkedDecodeUnsupportedError.__name__,
    ]
    assert [result["adaptive_calls"] for result in results] == [0, 0]
    assert [result["recovery"] for result in results] == [3, 3]


@pytest.mark.parallel
def test_two_rank_callback_failure_is_symmetric_and_group_recovers(
    tmp_path: Path,
) -> None:
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "gloo-init")
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(rank, init_file, result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    deadline = time.monotonic() + 90
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.join(timeout=1)
    hung = [process for process in processes if process.is_alive()]
    for process in hung:
        process.terminate()
        process.join(timeout=5)
    assert not hung, "two-rank chunk callback test deadlocked"
    assert [process.exitcode for process in processes] == [0, 0]

    results = sorted(
        [result_queue.get(timeout=5) for _ in range(2)],
        key=lambda result: result["rank"],
    )
    assert [result["failure_type"] for result in results] == [
        "LookupError",
        MiniMaxH3ChunkCallbackPeerError.__name__,
    ]
    assert [result["source_mismatch_type"] for result in results] == [
        MiniMaxH3ChunkedDecodeUnsupportedError.__name__,
        MiniMaxH3ChunkedDecodeUnsupportedError.__name__,
    ]
    assert [result["source_decode_calls"] for result in results] == [0, 0]
    assert [result["planner_failure_type"] for result in results] == [
        MiniMaxH3ChunkedDecodeUnsupportedError.__name__,
        "RuntimeError",
    ]
    assert [result["planner_decode_calls"] for result in results] == [0, 0]
    assert [result["dtype_mismatch_type"] for result in results] == [
        MiniMaxH3ChunkedDecodeUnsupportedError.__name__,
        MiniMaxH3ChunkedDecodeUnsupportedError.__name__,
    ]
    assert [result["dtype_decode_calls"] for result in results] == [0, 0]
    assert [result["dp_vae_type"] for result in results] == [
        MiniMaxH3ChunkedDecodeUnsupportedError.__name__,
        MiniMaxH3ChunkedDecodeUnsupportedError.__name__,
    ]
    assert [result["dp_decode_calls"] for result in results] == [0, 0]
    assert [result["invalid_owner_type"] for result in results] == [
        "ValueError",
        "ValueError",
    ]
    assert [result["adaptive_calls"] for result in results] == [14, 14]
    assert [result["success_calls"] for result in results] == [8, 0]
    assert all(result["output_matches"] for result in results)
    assert [result["fallback_calls"] for result in results] == [8, 0]
    assert [result["fallback_decode_calls"] for result in results] == [7, 7]
    assert all(result["fallback_output_matches"] for result in results)
    assert all(result["fallback_tiling_restored"] for result in results)
    # With VAE parallelism disabled, the caller still designates one request owner.
    assert [result["local_calls"] for result in results] == [8, 0]
    assert all(result["local_output_matches"] for result in results)
    assert [result["recovery"] for result in results] == [3, 3]
