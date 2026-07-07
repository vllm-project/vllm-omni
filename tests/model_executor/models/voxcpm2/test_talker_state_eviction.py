# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for VoxCPM2 talker per-request state lifecycle."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


@functools.lru_cache(maxsize=1)
def _voxcpm2_talker_mod():
    """Defer talker import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        VoxCPM2TalkerForConditionalGeneration,
        _RequestState,
        _VoxCPM2RuntimeConfig,
    )

    return VoxCPM2TalkerForConditionalGeneration, _RequestState, _VoxCPM2RuntimeConfig


def _make_bare_talker():
    VoxCPM2TalkerForConditionalGeneration, _, _ = _voxcpm2_talker_mod()
    talker = VoxCPM2TalkerForConditionalGeneration.__new__(VoxCPM2TalkerForConditionalGeneration)
    talker._active_states = {}
    talker._current_request_id = None
    talker._pending_requests = []
    talker._results_queue = []
    talker._audio_queue = []
    talker._deferred_cleanup_ids = set()
    talker._max_batch_size = 4
    talker._active_state_warn_threshold = 512
    talker._active_state_warned = False
    talker._enable_profiling = False
    talker._audio_emit_every = 1
    talker._vae_decode_every = 1
    talker._enable_delayed_audio_copy = False
    talker._delayed_audio_copy_use_events = False
    talker._coalesce_audio_d2h = False
    talker._enable_batched_vae_decode = False
    talker._audio_copy_stream = None
    talker._enable_vae_cuda_graph = False
    talker._enable_cfm_cuda_graph = False
    talker._enable_cfm_prealloc_output = False
    talker._enable_batched_cfm = True
    talker._deterministic_cfm_noise = False
    talker._cfm_buffers = None
    talker._last_audio_output_req_ids = []
    talker._batched_fsq_fusion_max_batch = 32
    return talker


def _seed_cached_decode(talker, req_id: str):
    _, _RequestState, _ = _voxcpm2_talker_mod()
    state = _RequestState(request_id=req_id)
    state.prefill_completed = True
    state.decode_step_count = 5
    talker._active_states[req_id] = state
    return state


class TestStateEvictionContract:
    def test_runtime_config_normalizes_mutually_exclusive_paths(self) -> None:
        _, _, RuntimeConfig = _voxcpm2_talker_mod()

        cfg = RuntimeConfig(
            enable_batched_cfm=True,
            enable_cfm_cuda_graph=True,
            enable_cfm_prealloc_output=True,
            enable_batched_vae_decode=True,
            enable_delayed_audio_copy=True,
        )._normalized()

        assert cfg.enable_batched_cfm is True
        assert cfg.enable_cfm_cuda_graph is False
        assert cfg.enable_cfm_prealloc_output is False
        assert cfg.enable_batched_vae_decode is False
        assert cfg.enable_delayed_audio_copy is True

    def test_pending_requests_is_not_used_for_eviction(self) -> None:
        talker = _make_bare_talker()

        cached_ids = [f"req-{i}" for i in range(4)]
        for rid in cached_ids:
            _seed_cached_decode(talker, rid)

        walked_so_far = ["req-new", cached_ids[0], cached_ids[1]]
        talker._pending_requests = [(rid, False, None, 0) for rid in walked_so_far]

        for rid in cached_ids:
            assert rid in talker._active_states
            assert talker._active_states[rid].prefill_completed is True

    def test_compute_logits_does_not_sync_stop_bool_on_default_path(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker.config = SimpleNamespace(vocab_size=4)
        state = RState(request_id="req", precomputed_stop_logits=torch.tensor([[0.0, 1.0]]))
        talker._active_states["req"] = state
        talker._results_queue = [("req", state.precomputed_stop_logits)]

        logits = talker.compute_logits(torch.zeros(1, 1))

        assert logits[0, 0] == 0.0
        assert logits[0, 1] == 1.0
        assert state.is_stopping is False
        assert state.precomputed_stop_logits is None
        assert state.precomputed_is_stopping is None

    def test_compute_logits_reuses_cached_stop_bool_from_sparse_audio_path(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker.config = SimpleNamespace(vocab_size=4)
        state = RState(
            request_id="req",
            precomputed_stop_logits=torch.tensor([[0.0, 1.0]]),
            precomputed_is_stopping=True,
        )
        talker._active_states["req"] = state
        talker._results_queue = [("req", state.precomputed_stop_logits)]

        logits = talker.compute_logits(torch.zeros(1, 1))

        assert logits[0, 0] == 0.0
        assert logits[0, 1] == 1.0
        assert state.is_stopping is True
        assert state.precomputed_stop_logits is None
        assert state.precomputed_is_stopping is None

    def test_batched_stop_precompute_sets_audio_collect_cache(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required to exercise batched stop D2H precompute")

        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._audio_emit_every = 2
        states = [
            RState(request_id="stop", precomputed_stop_logits=torch.tensor([[0.0, 1.0]], device="cuda")),
            RState(request_id="keep", precomputed_stop_logits=torch.tensor([[1.0, 0.0]], device="cuda")),
        ]

        talker._precompute_stop_flags_for_audio_collect(states)

        assert states[0].precomputed_is_stopping is True
        assert states[0].is_stopping is True
        assert states[1].precomputed_is_stopping is False
        assert states[1].is_stopping is False

    def test_on_requests_finished_defers_cleanup(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")

        talker.on_requests_finished({"req-A"})

        assert "req-A" in talker._active_states
        assert "req-A" in talker._deferred_cleanup_ids

    def test_flush_deferred_cleanup_removes_only_finished(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")
        talker.on_requests_finished(["req-A"])

        talker._flush_deferred_cleanup()

        assert "req-A" not in talker._active_states
        assert "req-B" in talker._active_states
        assert talker._deferred_cleanup_ids == set()

    def test_current_request_id_cleared_when_matching(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        talker._current_request_id = "req-A"

        talker.on_requests_finished({"req-A"})
        talker._flush_deferred_cleanup()

        assert talker._current_request_id is None

    def test_current_request_id_preserved_when_not_finished(self) -> None:
        talker = _make_bare_talker()
        _seed_cached_decode(talker, "req-A")
        _seed_cached_decode(talker, "req-B")
        talker._current_request_id = "req-B"

        talker.on_requests_finished({"req-A"})
        talker._flush_deferred_cleanup()

        assert talker._current_request_id == "req-B"


class TestLeakWarnGuard:
    def test_warn_fires_once_over_threshold(self, monkeypatch) -> None:
        from vllm_omni.model_executor.models.voxcpm2 import voxcpm2_talker as tk

        calls: list[str] = []

        def _capture(msg, *args, **kwargs):
            calls.append(msg % args if args else msg)

        monkeypatch.setattr(tk.logger, "warning", _capture)

        talker = _make_bare_talker()
        talker._active_state_warn_threshold = 3

        _, RState, _ = _voxcpm2_talker_mod()
        for i in range(4):
            talker._active_states[f"seed-{i}"] = RState(request_id=f"seed-{i}")

        talker._get_or_create_state("new-1")
        talker._get_or_create_state("new-2")

        leak_warnings = [m for m in calls if "cleanup path leak" in m]
        assert len(leak_warnings) == 1
        assert talker._active_state_warned is True


class _NoopPerf:
    def start(self, name: str) -> None:
        pass

    def stop(self, name: str) -> None:
        pass


class _FakeTTS:
    feat_decoder = SimpleNamespace(estimator=SimpleNamespace(_compiled=False))

    def fsq_layer(self, x: torch.Tensor) -> torch.Tensor:
        return x + 10

    def fusion_concat_proj(self, x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return x[:, :half] * 2 + x[:, half:] * 3

    def feat_encoder(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def enc_to_lm_proj(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(1).mean(dim=1).repeat(1, 4)


class TestDecodeBatchContract:
    def test_forward_skip_pending_prefill_keeps_audio_queue_aligned(self) -> None:
        class FakeScaffold:
            def __call__(self, input_ids, positions, intermediate_tensors, inputs_embeds):
                return torch.zeros(input_ids.shape[0], 4)

        talker = _make_bare_talker()
        talker._perf = _NoopPerf()
        talker._tts = _FakeTTS()
        talker.model = FakeScaffold()
        talker._enable_cuda_graph = False
        talker._cuda_graph_warmup_steps = 0
        talker._cuda_graph_warmup_threshold = 3
        talker._max_cached_graphs = 4
        talker._scaffold_graphs = {}
        talker._residual_graphs = {}
        talker._pending_requests = [("req", False, None, 1)]

        out = talker.forward(
            torch.tensor([1]),
            torch.tensor([0]),
            inputs_embeds=torch.zeros(1, 4),
        )

        torch.testing.assert_close(out, torch.zeros(1, 4))
        assert talker._results_queue == [("req", None)]
        assert talker._audio_queue == [("req", None)]

    def _make_decode_talker(self) -> object:
        talker = _make_bare_talker()
        talker._perf = _NoopPerf()
        talker._tts = _FakeTTS()
        talker._side_dtype = torch.float32
        talker._patch_size = 2
        talker._feat_dim = 3

        def dit_proj(lm_h: torch.Tensor, res_h: torch.Tensor) -> torch.Tensor:
            return lm_h + 2 * res_h

        def stop_fn(lm_h: torch.Tensor) -> torch.Tensor:
            return torch.stack([lm_h.sum(dim=1), -lm_h.sum(dim=1)], dim=1)

        def run_cfm(dit_h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            return dit_h[:, :1].view(-1, 1, 1) + cond.transpose(1, 2) * 0.1

        talker._dit_proj_fn = dit_proj
        talker._stop_fn = stop_fn
        talker._run_cfm = run_cfm
        talker._run_cfm_for_state = lambda _state, dit_h, cond: talker._run_cfm(dit_h, cond)
        return talker

    def _make_decode_state(self, request_id: str, offset: float):
        _, RState, _ = _voxcpm2_talker_mod()
        state = RState(request_id=request_id)
        state.curr_prefix_feat_cond = torch.arange(6, dtype=torch.float32).reshape(2, 3) + offset
        return state

    def test_batched_fsq_fusion_matches_per_request_decode_prepare(self) -> None:
        seq_talker = self._make_decode_talker()
        batch_talker = self._make_decode_talker()
        seq_talker._max_decode_steps = 2000
        batch_talker._max_decode_steps = 2000

        seq_states = [self._make_decode_state("req-0", 0.0), self._make_decode_state("req-1", 10.0)]
        batch_states = [self._make_decode_state("req-0", 0.0), self._make_decode_state("req-1", 10.0)]
        for idx, (seq_state, batch_state) in enumerate(zip(seq_states, batch_states, strict=True)):
            prev = torch.full((1, 4), float(idx + 1))
            seq_state.prev_feat_embed = prev.clone()
            batch_state.prev_feat_embed = prev.clone()

        hiddens = [
            torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
        ]

        expected = [
            seq_talker._prepare_residual_decode(state, hidden, torch.device("cpu"))
            for state, hidden in zip(seq_states, hiddens, strict=True)
        ]
        actual = batch_talker._prepare_residual_decode_batch(batch_states, hiddens, torch.device("cpu"))

        for (expected_res, expected_meta), (actual_res, actual_meta) in zip(expected, actual, strict=True):
            torch.testing.assert_close(actual_res, expected_res)
            torch.testing.assert_close(actual_meta["new_lm_hidden"], expected_meta["new_lm_hidden"])
        assert [state.decode_step_count for state in batch_states] == [1, 1]

    def test_batch_decode_matches_per_request_state_updates(self) -> None:
        seq_talker = self._make_decode_talker()
        batch_talker = self._make_decode_talker()

        seq_states = [self._make_decode_state("req-0", 0.0), self._make_decode_state("req-1", 10.0)]
        batch_states = [self._make_decode_state("req-0", 0.0), self._make_decode_state("req-1", 10.0)]
        metas = [
            {"new_lm_hidden": torch.tensor([[1.0, 2.0, 3.0, 4.0]])},
            {"new_lm_hidden": torch.tensor([[5.0, 6.0, 7.0, 8.0]])},
        ]
        batch_out = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]])

        for state, meta, res_out in zip(seq_states, metas, batch_out, strict=True):
            seq_talker._finish_decode(state, meta, res_out.unsqueeze(0), torch.device("cpu"))

        batch_talker._finish_decode_batch(
            [(state, False, meta) for state, meta in zip(batch_states, metas, strict=True)],
            batch_out,
        )

        for expected, actual in zip(seq_states, batch_states, strict=True):
            torch.testing.assert_close(actual.precomputed_stop_logits, expected.precomputed_stop_logits)
            torch.testing.assert_close(actual.curr_embed_for_next, expected.curr_embed_for_next)
            torch.testing.assert_close(actual.prev_feat_embed, expected.prev_feat_embed)
            torch.testing.assert_close(actual.curr_prefix_feat_cond, expected.curr_prefix_feat_cond)
            torch.testing.assert_close(actual.last_audio_patch_gpu, expected.last_audio_patch_gpu)

    def test_batch_decode_preserves_per_request_cfm_rng_order(self) -> None:
        talker = self._make_decode_talker()
        talker._enable_batched_cfm = False
        states = [self._make_decode_state("req-0", 0.0), self._make_decode_state("req-1", 10.0)]
        metas = [
            {"new_lm_hidden": torch.tensor([[1.0, 2.0, 3.0, 4.0]])},
            {"new_lm_hidden": torch.tensor([[5.0, 6.0, 7.0, 8.0]])},
        ]
        batch_out = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]])

        captured_noise: list[torch.Tensor] = []
        torch.manual_seed(1234)
        expected_noise = [torch.empty(1, 3, 2).normal_().detach().clone() for _ in states]

        def run_cfm(
            dit_h: torch.Tensor,
            cond: torch.Tensor,
        ) -> torch.Tensor:
            captured_noise.append(torch.empty(1, 3, 2).normal_().detach().clone())
            return dit_h[:, :1].view(-1, 1, 1) + cond.transpose(1, 2) * 0.1

        talker._run_cfm = run_cfm

        torch.manual_seed(1234)
        talker._finish_decode_batch(
            [(state, False, meta) for state, meta in zip(states, metas, strict=True)],
            batch_out,
        )

        assert len(captured_noise) == len(states)
        for actual, expected in zip(captured_noise, expected_noise, strict=True):
            torch.testing.assert_close(actual, expected)


class _FakeAudioVAE:
    decode_chunk_size = 2


class _FakeAudioTTS:
    audio_vae = _FakeAudioVAE()


class TestAudioEmitCoalescing:
    def test_audio_emit_every_accumulates_gpu_chunks_without_changing_order(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._audio_emit_every = 2
        talker._feat_dim = 1
        talker._n_decode_pad_frames = 1
        talker._device = torch.device("cpu")
        talker._tts = _FakeAudioTTS()
        talker._perf = _NoopPerf()

        values = [torch.tensor([10.0, 11.0]), torch.tensor([99.0, 99.0, 20.0, 21.0])]

        def run_vae_decode(feat: torch.Tensor) -> torch.Tensor:
            return values.pop(0)

        talker._run_vae_decode = run_vae_decode
        state = RState(request_id="req")

        state.last_audio_patch_gpu = torch.tensor([[1.0]])
        assert talker._collect_audio(state) is None
        assert len(state.pending_audio_chunks_gpu) == 1

        state.last_audio_patch_gpu = torch.tensor([[2.0]])
        audio = talker._collect_audio(state)

        torch.testing.assert_close(audio, torch.tensor([10.0, 11.0, 20.0, 21.0]))
        assert state.pending_audio_chunks_gpu == []

    def test_audio_emit_every_flushes_on_stop(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._audio_emit_every = 8
        talker._feat_dim = 1
        talker._n_decode_pad_frames = 1
        talker._device = torch.device("cpu")
        talker._tts = _FakeAudioTTS()
        talker._perf = _NoopPerf()
        talker._run_vae_decode = lambda feat: torch.tensor([3.0, 4.0])
        state = RState(request_id="req", precomputed_stop_logits=torch.tensor([[0.0, 1.0]]))

        state.last_audio_patch_gpu = torch.tensor([[1.0]])
        audio = talker._collect_audio(state)

        torch.testing.assert_close(audio, torch.tensor([3.0, 4.0]))
        assert state.pending_audio_chunks_gpu == []

    def test_vae_decode_every_accumulates_latents_before_decode(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._vae_decode_every = 2
        talker._feat_dim = 1
        talker._n_decode_pad_frames = 1
        talker._device = torch.device("cpu")
        talker._tts = _FakeAudioTTS()
        talker._perf = _NoopPerf()

        seen_feats = []
        values = [torch.tensor([10.0, 11.0, 20.0, 21.0])]

        def run_vae_decode(feat: torch.Tensor) -> torch.Tensor:
            seen_feats.append(feat.detach().clone())
            return values.pop(0)

        talker._run_vae_decode = run_vae_decode
        state = RState(request_id="req")

        state.last_audio_patch_gpu = torch.tensor([[1.0]])
        assert talker._collect_audio(state) is None
        assert len(state.pending_vae_latents_gpu) == 1
        assert seen_feats == []

        state.last_audio_patch_gpu = torch.tensor([[2.0]])
        audio = talker._collect_audio(state)

        torch.testing.assert_close(audio, torch.tensor([10.0, 11.0, 20.0, 21.0]))
        assert state.pending_vae_latents_gpu == []
        torch.testing.assert_close(seen_feats[0].reshape(-1), torch.tensor([1.0, 2.0]))

    def test_vae_decode_every_flushes_pending_latent_on_stop(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._vae_decode_every = 4
        talker._feat_dim = 1
        talker._n_decode_pad_frames = 1
        talker._device = torch.device("cpu")
        talker._tts = _FakeAudioTTS()
        talker._perf = _NoopPerf()
        talker._run_vae_decode = lambda feat: torch.tensor([7.0, 8.0])
        state = RState(request_id="req", precomputed_stop_logits=torch.tensor([[0.0, 1.0]]))

        state.last_audio_patch_gpu = torch.tensor([[1.0]])
        audio = talker._collect_audio(state)

        torch.testing.assert_close(audio, torch.tensor([7.0, 8.0]))
        assert state.pending_vae_latents_gpu == []
        assert state.is_stopping is True

    def test_batched_vae_decode_collects_ready_requests_in_order(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._enable_batched_vae_decode = True
        talker._coalesce_audio_d2h = True
        talker._vae_decode_every = 2
        talker._feat_dim = 1
        talker._n_decode_pad_frames = 1
        talker._device = torch.device("cpu")
        talker._tts = _FakeAudioTTS()
        talker._perf = _NoopPerf()

        seen_feats: list[torch.Tensor] = []

        def run_vae_decode(feat: torch.Tensor) -> torch.Tensor:
            seen_feats.append(feat.detach().clone())
            # One output row per request. decode_chunk_size=2, and each ready
            # request has two new latent frames, so the full decoded length is 4.
            return torch.tensor(
                [
                    [[10.0, 11.0, 12.0, 13.0]],
                    [[20.0, 21.0, 22.0, 23.0]],
                ]
            )

        talker._run_vae_decode = run_vae_decode

        ready0 = RState(request_id="ready-0")
        ready0.pending_vae_latents_gpu.append(torch.tensor([[1.0]]))
        ready0.last_audio_patch_gpu = torch.tensor([[2.0]])

        pending = RState(request_id="pending")
        pending.last_audio_patch_gpu = torch.tensor([[99.0]])

        ready1 = RState(request_id="ready-1")
        ready1.pending_vae_latents_gpu.append(torch.tensor([[3.0]]))
        ready1.last_audio_patch_gpu = torch.tensor([[4.0]])

        outputs = talker._collect_audio_batch([ready0, pending, ready1])

        assert list(outputs) == ["ready-0", "pending", "ready-1"]
        torch.testing.assert_close(outputs["ready-0"], torch.tensor([10.0, 11.0, 12.0, 13.0]))
        assert outputs["pending"] is None
        torch.testing.assert_close(outputs["ready-1"], torch.tensor([20.0, 21.0, 22.0, 23.0]))
        assert len(seen_feats) == 1
        torch.testing.assert_close(seen_feats[0].reshape(2, -1), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert pending.pending_vae_latents_gpu
        assert ready0.pending_vae_latents_gpu == []
        assert ready1.pending_vae_latents_gpu == []

    def test_audio_emit_every_keeps_private_chunk_when_vae_output_storage_is_reused(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._audio_emit_every = 2
        talker._enable_vae_cuda_graph = True
        talker._feat_dim = 1
        talker._n_decode_pad_frames = 1
        talker._device = torch.device("cpu")
        talker._tts = _FakeAudioTTS()
        talker._perf = _NoopPerf()

        shared_output = torch.empty(4)
        values = [torch.tensor([10.0, 11.0, -1.0, -1.0]), torch.tensor([99.0, 99.0, 20.0, 21.0])]

        def run_vae_decode(feat: torch.Tensor) -> torch.Tensor:
            shared_output.copy_(values.pop(0))
            return shared_output

        talker._run_vae_decode = run_vae_decode
        state = RState(request_id="req")

        state.last_audio_patch_gpu = torch.tensor([[1.0]])
        assert talker._collect_audio(state) is None

        state.last_audio_patch_gpu = torch.tensor([[2.0]])
        audio = talker._collect_audio(state)

        torch.testing.assert_close(audio, torch.tensor([10.0, 11.0, 20.0, 21.0]))

    def test_audio_emit_every_keeps_private_chunk_when_compiled_vae_reuses_storage(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._audio_emit_every = 2
        talker._feat_dim = 1
        talker._n_decode_pad_frames = 1
        talker._device = torch.device("cpu")
        talker._tts = _FakeAudioTTS()
        talker._tts.audio_vae._compiled = True
        talker._perf = _NoopPerf()

        shared_output = torch.empty(4)
        values = [torch.tensor([10.0, 11.0, -1.0, -1.0]), torch.tensor([99.0, 99.0, 20.0, 21.0])]

        def run_vae_decode(feat: torch.Tensor) -> torch.Tensor:
            shared_output.copy_(values.pop(0))
            return shared_output

        talker._run_vae_decode = run_vae_decode
        state = RState(request_id="req")

        state.last_audio_patch_gpu = torch.tensor([[1.0]])
        assert talker._collect_audio(state) is None

        state.last_audio_patch_gpu = torch.tensor([[2.0]])
        audio = talker._collect_audio(state)

        torch.testing.assert_close(audio, torch.tensor([10.0, 11.0, 20.0, 21.0]))

    def test_delayed_audio_copy_emits_previous_chunk_and_flushes_on_stop(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._enable_delayed_audio_copy = True
        talker._audio_emit_every = 1
        talker._feat_dim = 1
        talker._n_decode_pad_frames = 1
        talker._device = torch.device("cpu")
        talker._tts = _FakeAudioTTS()
        talker._perf = _NoopPerf()

        values = [
            torch.tensor([10.0, 11.0]),
            torch.tensor([99.0, 99.0, 20.0, 21.0]),
            torch.tensor([88.0, 88.0, 30.0, 31.0]),
        ]

        def run_vae_decode(feat: torch.Tensor) -> torch.Tensor:
            return values.pop(0)

        talker._run_vae_decode = run_vae_decode
        state = RState(request_id="req")

        state.last_audio_patch_gpu = torch.tensor([[1.0]])
        assert talker._collect_audio(state) is None
        assert len(state.pending_audio_copies) == 1

        state.last_audio_patch_gpu = torch.tensor([[2.0]])
        audio = talker._collect_audio(state)
        torch.testing.assert_close(audio, torch.tensor([10.0, 11.0]))
        assert len(state.pending_audio_copies) == 1

        state.precomputed_stop_logits = torch.tensor([[0.0, 1.0]])
        state.last_audio_patch_gpu = torch.tensor([[3.0]])
        audio = talker._collect_audio(state)
        torch.testing.assert_close(audio, torch.tensor([20.0, 21.0, 30.0, 31.0]))
        assert state.pending_audio_copies == []
        assert state.is_stopping is True

    def test_delayed_audio_copy_flushes_on_forced_stop(self) -> None:
        _, RState, _ = _voxcpm2_talker_mod()
        talker = _make_bare_talker()
        talker._enable_delayed_audio_copy = True
        talker._audio_emit_every = 1
        talker._feat_dim = 1
        talker._n_decode_pad_frames = 1
        talker._device = torch.device("cpu")
        talker._tts = _FakeAudioTTS()
        talker._perf = _NoopPerf()

        values = [
            torch.tensor([10.0, 11.0]),
            torch.tensor([99.0, 99.0, 20.0, 21.0]),
        ]

        def run_vae_decode(feat: torch.Tensor) -> torch.Tensor:
            return values.pop(0)

        talker._run_vae_decode = run_vae_decode
        state = RState(request_id="req")

        state.last_audio_patch_gpu = torch.tensor([[1.0]])
        assert talker._collect_audio(state) is None

        state.is_stopping = True
        state.last_audio_patch_gpu = torch.tensor([[2.0]])
        audio = talker._collect_audio(state)

        torch.testing.assert_close(audio, torch.tensor([10.0, 11.0, 20.0, 21.0]))
        assert state.pending_audio_copies == []

    def test_make_omni_output_uses_sparse_req_ids_for_deferred_chunks(self) -> None:
        talker = _make_bare_talker()
        talker._audio_emit_every = 4
        talker._sample_rate = 16000
        talker._last_audio_output_req_ids = ["req-0", "req-1", "req-2"]
        talker._audio_queue = [
            ("req-0", None),
            ("req-1", torch.tensor([1.0, 2.0])),
            ("req-2", None),
        ]

        out = talker.make_omni_output(torch.zeros(3, 1))

        audio = out.multimodal_outputs["model_outputs"]
        sr = out.multimodal_outputs["sr"]
        assert out.multimodal_outputs["meta"]["req_id"] == ["req-1"]
        assert out.multimodal_outputs["meta"]["sparse_audio"] == ["1"]
        assert len(audio) == 1
        torch.testing.assert_close(audio[0], torch.tensor([1.0, 2.0]))
        assert int(sr[0]) == 16000

    def test_make_omni_output_emits_empty_sparse_marker_for_deferred_chunks(self) -> None:
        talker = _make_bare_talker()
        talker._audio_emit_every = 4
        talker._sample_rate = 16000
        talker._audio_queue = [("req-0", None), ("req-1", None)]

        out = talker.make_omni_output(torch.zeros(2, 1))

        assert out.multimodal_outputs["model_outputs"] == []
        assert out.multimodal_outputs["sr"] == []
        assert out.multimodal_outputs["meta"]["req_id"] == []
        assert out.multimodal_outputs["meta"]["sparse_audio"] == ["1"]

    def test_make_omni_output_uses_sparse_req_ids_for_delayed_audio_copy(self) -> None:
        talker = _make_bare_talker()
        talker._enable_delayed_audio_copy = True
        talker._sample_rate = 16000
        talker._audio_queue = [
            ("req-0", None),
            ("req-1", torch.tensor([1.0, 2.0])),
            ("req-2", None),
        ]

        out = talker.make_omni_output(torch.zeros(3, 1))

        assert out.multimodal_outputs["meta"]["req_id"] == ["req-1"]
        assert out.multimodal_outputs["meta"]["sparse_audio"] == ["1"]
        assert len(out.multimodal_outputs["model_outputs"]) == 1
        torch.testing.assert_close(out.multimodal_outputs["model_outputs"][0], torch.tensor([1.0, 2.0]))

    def test_make_omni_output_uses_sparse_req_ids_for_deferred_vae_decode(self) -> None:
        talker = _make_bare_talker()
        talker._vae_decode_every = 2
        talker._sample_rate = 16000
        talker._audio_queue = [
            ("req-0", None),
            ("req-1", torch.tensor([1.0, 2.0])),
            ("req-2", None),
        ]

        out = talker.make_omni_output(torch.zeros(3, 1))

        assert out.multimodal_outputs["meta"]["req_id"] == ["req-1"]
        assert out.multimodal_outputs["meta"]["sparse_audio"] == ["1"]
        assert len(out.multimodal_outputs["model_outputs"]) == 1
        torch.testing.assert_close(out.multimodal_outputs["model_outputs"][0], torch.tensor([1.0, 2.0]))

    def test_make_omni_output_coalesces_gpu_audio_to_cpu(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required to exercise D2H coalescing")
        talker = _make_bare_talker()
        talker._coalesce_audio_d2h = True
        talker._sample_rate = 16000
        talker._audio_queue = [
            ("req-0", torch.tensor([1.0, 2.0], device="cuda")),
            ("req-1", torch.tensor([3.0], device="cuda")),
        ]

        out = talker.make_omni_output(torch.zeros(2, 1))

        audio = out.multimodal_outputs["model_outputs"]
        assert len(audio) == 2
        assert all(not chunk.is_cuda for chunk in audio)
        torch.testing.assert_close(audio[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(audio[1], torch.tensor([3.0]))
