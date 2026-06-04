# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_omni.model_executor.models.ming_flash_omni.talker_module as talker_module
from vllm_omni.model_executor.models.ming_flash_omni.talker_module import (
    CFM,
    Aggregator,
    CFMGraphExecutor,
    CFMGraphExecutorPool,
    DiT,
    MingAudioGenerator,
    MingTalkerBatchPolicy,
    MingTalkerSlotTable,
)

torch = pytest.importorskip("torch")
pytest.importorskip("x_transformers")

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for graph capture"),
]

_LATENT_DIM = 8
_PATCH_SIZE = 4
_HIS_PATCH_SIZE = 8
_LLM_HIDDEN = 16
_DIT_HIDDEN = 32
_AGG_HIDDEN = 32
_NUM_HEADS = 4
_DEPTH = 2
_STEPS = 5
_DTYPE = torch.float32


def _warmup_pipeline(cfm: CFM, aggregator: Aggregator, stop_head: torch.nn.Linear, device: torch.device) -> None:
    llm_cond = torch.randn(1, 1, _LLM_HIDDEN, device=device, dtype=_DTYPE)
    lat_cond = torch.randn(1, _HIS_PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)
    y0 = torch.randn(1, _PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)
    t = torch.linspace(0.0, 1.0, _STEPS + 1, device=device, dtype=_DTYPE)
    sde_args = torch.tensor([2.0, 0.25, 0.0], device=device, dtype=_DTYPE)
    sde_rnd = torch.randn(_STEPS, 1, _PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)

    with torch.no_grad():
        gen_lat = cfm.sample(llm_cond, lat_cond, y0, t, sde_args, sde_rnd)
        aggregator(gen_lat)
        stop_head(llm_cond[:, -1, :]).softmax(dim=-1)
    torch.accelerator.synchronize(device)


def _build_pipeline():
    device = torch.device("cuda")
    dit = (
        DiT(
            in_channels=_LATENT_DIM,
            hidden_size=_DIT_HIDDEN,
            depth=_DEPTH,
            num_heads=_NUM_HEADS,
            mlp_ratio=2.0,
            llm_cond_dim=_LLM_HIDDEN,
        )
        .to(device=device, dtype=_DTYPE)
        .eval()
    )
    cfm = CFM(dit, steps=_STEPS, sway_sampling_coef=-1.0).to(device=device, dtype=_DTYPE).eval()
    aggregator = (
        Aggregator(
            in_channels=_LATENT_DIM,
            hidden_size=_AGG_HIDDEN,
            depth=_DEPTH,
            num_heads=_NUM_HEADS,
            mlp_ratio=2.0,
            llm_input_dim=_LLM_HIDDEN,
        )
        .to(device=device, dtype=_DTYPE)
        .eval()
    )
    stop_head = torch.nn.Linear(_LLM_HIDDEN, 2).to(device=device, dtype=_DTYPE).eval()

    config = SimpleNamespace(steps=_STEPS, patch_size=_PATCH_SIZE)
    _warmup_pipeline(cfm, aggregator, stop_head, device)
    return config, cfm, aggregator, stop_head, device


class TestCFMGraphExecutor:
    """Capture once, replay twice: outputs must stay consistently-shaped."""

    def test_execute_shapes_and_replay(self) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()
        executor = CFMGraphExecutor(config, cfm, aggregator, stop_head)

        bsz = 1
        input_tensor = torch.randn(bsz, 1, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        his_lat = torch.randn(bsz, _HIS_PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)

        gen_lat, inputs_embeds, stop_out = executor.execute(input_tensor, his_lat)
        torch.accelerator.synchronize()

        assert gen_lat.shape == (bsz, _PATCH_SIZE, _LATENT_DIM)
        assert inputs_embeds.shape == (bsz, 1, _LLM_HIDDEN)
        assert stop_out.shape == (bsz, 2)
        assert torch.isfinite(gen_lat).all()
        assert torch.isfinite(inputs_embeds).all()
        # stop_head output is softmax-normalized across the last dim.
        assert torch.allclose(stop_out.sum(dim=-1), torch.ones(bsz, device=device, dtype=_DTYPE), atol=1e-4)

        # Replay the captured graph with fresh inputs — shapes must match.
        new_input = torch.randn_like(input_tensor)
        new_his = torch.randn_like(his_lat)
        gen_lat2, inputs_embeds2, stop_out2 = executor.execute(new_input, new_his)
        torch.accelerator.synchronize()
        assert gen_lat2.shape == gen_lat.shape
        assert inputs_embeds2.shape == inputs_embeds.shape
        assert stop_out2.shape == stop_out.shape
        assert executor.initialized is True

    def test_execute_is_noninplace_on_inputs(self) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()
        executor = CFMGraphExecutor(config, cfm, aggregator, stop_head)

        input_tensor = torch.randn(1, 1, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        his_lat = torch.randn(1, _HIS_PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)
        snapshot_input = input_tensor.clone()
        snapshot_his = his_lat.clone()

        executor.execute(input_tensor, his_lat)
        torch.accelerator.synchronize()
        assert torch.equal(input_tensor, snapshot_input)
        assert torch.equal(his_lat, snapshot_his)


class TestCFMGraphExecutorPool:
    def test_pool_acquires_and_releases(self) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()
        pool = CFMGraphExecutorPool(config, cfm, aggregator, stop_head, pool_size=2)

        input_tensor = torch.randn(1, 1, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        his_lat = torch.randn(1, _HIS_PATCH_SIZE, _LATENT_DIM, device=device, dtype=_DTYPE)

        gen_lat, inputs_embeds, stop_out = pool.execute(input_tensor, his_lat)
        torch.accelerator.synchronize()
        assert gen_lat.shape == (1, _PATCH_SIZE, _LATENT_DIM)
        assert inputs_embeds.shape == (1, 1, _LLM_HIDDEN)
        assert stop_out.shape == (1, 2)
        assert pool.pool.qsize() == 2


class TestMingAudioGeneratorBatch:
    def test_generate_latents_batch_returns_one_latent_list_per_request(self) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()

        class TinyLLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty((), device=device, dtype=_DTYPE))

            def forward(self, *, inputs_embeds, **kwargs):
                return SimpleNamespace(last_hidden_state=inputs_embeds)

        generator = MingAudioGenerator(
            config=config,
            llm_config=SimpleNamespace(
                num_hidden_layers=1, num_attention_heads=1, hidden_size=_LLM_HIDDEN, num_key_value_heads=1
            ),
            model=TinyLLM(),
            cfm=cfm,
            aggregator=aggregator,
            stop_head=stop_head,
            audio_vae=None,
            patch_size=_PATCH_SIZE,
            his_patch_size=_HIS_PATCH_SIZE,
            latent_dim=_LATENT_DIM,
            cfg_strength=2.0,
            use_cuda_graphs=True,
        )

        single = torch.randn(1, 3, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        single_latents = generator.generate_latents_batch(single, max_steps=2, use_static_cache=False)
        torch.accelerator.synchronize()
        assert len(single_latents) == 1

        inputs_embeds = torch.randn(2, 3, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        latents = generator.generate_latents_batch(inputs_embeds, max_steps=2, use_static_cache=False)
        torch.accelerator.synchronize()

        assert len(latents) == 2
        assert all(len(item) == 2 for item in latents)
        assert all(lat.shape == (1, _PATCH_SIZE, _LATENT_DIM) for item in latents for lat in item)

    def test_generate_latents_batch_compacts_finished_rows(self) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()

        class TinyLLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty((), device=device, dtype=_DTYPE))

            def forward(self, *, inputs_embeds, **kwargs):
                return SimpleNamespace(last_hidden_state=inputs_embeds)

        class StopAfterFirst(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                out = torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)
                out[:, 0] = 1.0
                if self.calls == 0:
                    out[0, 1] = 2.0
                self.calls += 1
                return out

        generator = MingAudioGenerator(
            config=config,
            llm_config=SimpleNamespace(
                num_hidden_layers=1, num_attention_heads=1, hidden_size=_LLM_HIDDEN, num_key_value_heads=1
            ),
            model=TinyLLM(),
            cfm=cfm,
            aggregator=aggregator,
            stop_head=StopAfterFirst(),
            audio_vae=None,
            patch_size=_PATCH_SIZE,
            his_patch_size=_HIS_PATCH_SIZE,
            latent_dim=_LATENT_DIM,
            cfg_strength=2.0,
            use_cuda_graphs=False,
        )

        inputs_embeds = torch.randn(2, 3, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        latents = generator.generate_latents_batch(inputs_embeds, max_steps=3, min_new_token=-1, use_static_cache=False)
        torch.accelerator.synchronize()

        assert len(latents[0]) == 1
        assert len(latents[1]) == 3

    def test_generate_latents_batch_compacts_finished_rows_with_static_cache(self, monkeypatch) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()

        class TinyCache:
            def __init__(self):
                self.selected = []

            def get_seq_length(self):
                return 3

            def batch_select_indices(self, indices):
                self.selected.append(indices.detach().cpu().tolist())

        class TinyLLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty((), device=device, dtype=_DTYPE))

            def forward(self, *, inputs_embeds, **kwargs):
                return SimpleNamespace(last_hidden_state=inputs_embeds)

        class StopAfterFirst(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                out = torch.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)
                out[:, 0] = 1.0
                if self.calls == 0:
                    out[0, 1] = 2.0
                self.calls += 1
                return out

        generator = MingAudioGenerator(
            config=config,
            llm_config=SimpleNamespace(
                num_hidden_layers=1, num_attention_heads=1, hidden_size=_LLM_HIDDEN, num_key_value_heads=1
            ),
            model=TinyLLM(),
            cfm=cfm,
            aggregator=aggregator,
            stop_head=StopAfterFirst(),
            audio_vae=None,
            patch_size=_PATCH_SIZE,
            his_patch_size=_HIS_PATCH_SIZE,
            latent_dim=_LATENT_DIM,
            cfg_strength=2.0,
            use_cuda_graphs=False,
        )
        cache = TinyCache()
        monkeypatch.setattr(generator, "_init_batched_kv_cache", lambda *args, **kwargs: (cache, 2048))

        inputs_embeds = torch.randn(2, 3, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        latents = generator.generate_latents_batch(inputs_embeds, max_steps=3, min_new_token=-1, use_static_cache=True)
        torch.accelerator.synchronize()

        assert len(latents[0]) == 1
        assert len(latents[1]) == 3
        assert cache.selected == [[1]]

    def test_reusable_static_cache_recreated_after_batch_compaction(self, monkeypatch) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()

        class DummyLayer:
            def __init__(self, batch_size: int):
                self.keys = torch.empty(batch_size, 1, 1, 1, device=device)
                self.values = torch.empty(batch_size, 1, 1, 1, device=device)
                self.cumulative_length = torch.zeros(batch_size, device=device, dtype=torch.int32)

        class DummyStaticCache:
            def __init__(self, *, max_batch_size: int, **kwargs):
                self.layers = [DummyLayer(max_batch_size)]

        generator = MingAudioGenerator(
            config=config,
            llm_config=SimpleNamespace(
                num_hidden_layers=1, num_attention_heads=1, hidden_size=_LLM_HIDDEN, num_key_value_heads=1
            ),
            model=torch.nn.Linear(_LLM_HIDDEN, _LLM_HIDDEN).to(device=device, dtype=_DTYPE),
            cfm=cfm,
            aggregator=aggregator,
            stop_head=stop_head,
            audio_vae=None,
            patch_size=_PATCH_SIZE,
            his_patch_size=_HIS_PATCH_SIZE,
            latent_dim=_LATENT_DIM,
            cfg_strength=2.0,
            use_cuda_graphs=False,
        )
        monkeypatch.setattr(talker_module, "StaticCache", DummyStaticCache)

        first = generator._get_reusable_kv_cache(2, device, _DTYPE, 16)
        first.layers[0].keys = first.layers[0].keys[:1].contiguous()
        second = generator._get_reusable_kv_cache(2, device, _DTYPE, 16)

        assert second is not first
        assert second.layers[0].keys.shape[0] == 2

    def test_generate_latents_batch_disables_llm_graph_for_compaction_path(self, monkeypatch) -> None:
        config, cfm, aggregator, stop_head, device = _build_pipeline()

        class TinyLLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.empty((), device=device, dtype=_DTYPE))

            def forward(self, *, inputs_embeds, **kwargs):
                return SimpleNamespace(last_hidden_state=inputs_embeds)

        generator = MingAudioGenerator(
            config=config,
            llm_config=SimpleNamespace(
                num_hidden_layers=1, num_attention_heads=1, hidden_size=_LLM_HIDDEN, num_key_value_heads=1
            ),
            model=TinyLLM(),
            cfm=cfm,
            aggregator=aggregator,
            stop_head=stop_head,
            audio_vae=None,
            patch_size=_PATCH_SIZE,
            his_patch_size=_HIS_PATCH_SIZE,
            latent_dim=_LATENT_DIM,
            cfg_strength=2.0,
            use_cuda_graphs=False,
        )
        seen: list[bool] = []

        def fake_init_batched_kv_cache(*args, allow_llm_decode_graph: bool = True, **kwargs):
            seen.append(allow_llm_decode_graph)
            return None, 2048

        monkeypatch.setattr(generator, "_init_batched_kv_cache", fake_init_batched_kv_cache)

        inputs_embeds = torch.randn(2, 3, _LLM_HIDDEN, device=device, dtype=_DTYPE)
        generator.generate_latents_batch(inputs_embeds, max_steps=1, use_static_cache=True)
        torch.accelerator.synchronize()

        assert seen == [False]


class TestMingTalkerBatchPolicy:
    def test_policy_dispatches_on_full_batch_or_wait_budget(self) -> None:
        policy = MingTalkerBatchPolicy(max_batch_size=8, max_wait_ms=20.0)

        assert policy.should_dispatch(0, 100.0) is False
        assert policy.should_dispatch(4, 5.0) is False
        assert policy.should_dispatch(8, 0.0) is True
        assert policy.should_dispatch(2, 20.0) is True

    def test_policy_chooses_bucket_without_exceeding_max_batch(self) -> None:
        policy = MingTalkerBatchPolicy(max_batch_size=8, bucket_sizes=(1, 2, 4, 8, 16))

        assert policy.choose_bucket(0) == 0
        assert policy.choose_bucket(1) == 1
        assert policy.choose_bucket(3) == 4
        assert policy.choose_bucket(9) == 8


class TestMingTalkerSlotTable:
    def test_slot_table_allocates_reuses_and_frees_slots(self) -> None:
        table = MingTalkerSlotTable(max_slots=2)

        first = table.allocate("r1")
        assert table.allocate("r1") == first
        second = table.allocate("r2")
        assert {first, second} == {0, 1}

        table.free("r1")
        reused = table.allocate("r3")

        assert reused == first
        assert set(table.active_request_ids()) == {"r2", "r3"}

    def test_slot_table_reports_compact_active_indices(self) -> None:
        table = MingTalkerSlotTable(max_slots=4)
        table.allocate("r1")
        table.allocate("r2")
        table.allocate("r3")
        table.free("r2")

        assert table.active_request_ids() == ["r1", "r3"]
        assert table.active_slots() == [0, 2]


class TestMingTalkerBatchCompatibility:
    def test_can_batch_additional_info_rejects_mismatched_params(self) -> None:
        from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
            MingFlashOmniTalkerForConditionalGeneration,
        )

        talker = object.__new__(MingFlashOmniTalkerForConditionalGeneration)
        compatible = [{"text": "a", "max_decode_steps": 10}, {"text": "b", "max_decode_steps": 10}]
        incompatible = [{"text": "a", "max_decode_steps": 10}, {"text": "b", "max_decode_steps": 11}]

        assert talker._can_batch_additional_info(compatible) is True
        assert talker._can_batch_additional_info(incompatible) is False


def test_ming_inner_cfm_graph_config_overrides_enforce_eager():
    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
        _resolve_inner_cfm_graph_enabled,
    )
    from vllm_omni.transformers_utils.configs.ming_flash_omni import MingFlashOmniTalkerConfig

    config = MingFlashOmniTalkerConfig(enable_cfm_graph=True)

    assert _resolve_inner_cfm_graph_enabled(config, enforce_eager=True) is True


def test_ming_inner_cfm_graph_defaults_to_not_enforce_eager():
    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
        _resolve_inner_cfm_graph_enabled,
    )
    from vllm_omni.transformers_utils.configs.ming_flash_omni import MingFlashOmniTalkerConfig

    config = MingFlashOmniTalkerConfig(enable_cfm_graph=None)

    assert _resolve_inner_cfm_graph_enabled(config, enforce_eager=True) is False
    assert _resolve_inner_cfm_graph_enabled(config, enforce_eager=False) is True


def test_ming_full_vae_decode_config_changes_default_stream_decode():
    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
        MingFlashOmniTalkerForConditionalGeneration,
    )

    talker = object.__new__(MingFlashOmniTalkerForConditionalGeneration)
    talker.cfg_strength = 2.0
    talker.config = SimpleNamespace(enable_full_vae_decode=True)

    params = talker._resolve_generation_params({})

    assert params.stream_decode is False


def test_ming_explicit_stream_decode_overrides_full_vae_decode_config():
    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
        MingFlashOmniTalkerForConditionalGeneration,
    )

    talker = object.__new__(MingFlashOmniTalkerForConditionalGeneration)
    talker.cfg_strength = 2.0
    talker.config = SimpleNamespace(enable_full_vae_decode=True)

    params = talker._resolve_generation_params({"stream_decode": True})

    assert params.stream_decode is True


def test_decode_batch_latents_uses_stream_decode_by_default():
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
        MingFlashOmniTalkerForConditionalGeneration,
    )

    class FakeAudioVAE:
        config = SimpleNamespace(sample_rate=44100)

        def decode(self, batch_latents, **kwargs):
            raise AssertionError("stream decode must not use batched full VAE")

    class FakeAudioGenerator:
        def __init__(self):
            self.calls = []

        def decode_to_waveform(self, latents, *, stream_decode):
            self.calls.append((len(latents), stream_decode))
            return torch.zeros(1, 1, 8)

    talker = object.__new__(MingFlashOmniTalkerForConditionalGeneration)
    talker.config = SimpleNamespace(batch_vae_min_batch=4)
    talker.audio_vae = FakeAudioVAE()
    talker.audio_generator = FakeAudioGenerator()
    latents = [[torch.zeros(1, 2, 3)] for _ in range(4)]

    audios, sample_rate = talker.decode_batch_latents_for_runner(latents, stream_decode=True)

    assert sample_rate == 44100
    assert len(audios) == 4
    assert talker.audio_generator.calls == [(1, True), (1, True), (1, True), (1, True)]


def test_decode_batch_latents_transfers_batched_waveform_to_cpu_once():
    from types import SimpleNamespace

    import torch

    from vllm_omni.model_executor.models.ming_flash_omni.ming_flash_omni_talker import (
        MingFlashOmniTalkerForConditionalGeneration,
    )

    class FakeWaveform:
        def __init__(self, tensor, counter):
            self.tensor = tensor
            self.counter = counter
            self.shape = tensor.shape

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            self.counter["cpu"] += 1
            return self.tensor.cpu()

        def __getitem__(self, idx):
            return FakeWaveform(self.tensor[idx], self.counter)

    class FakeAudioGenerator:
        def trim_trailing_silence(self, waveform):
            return waveform

    class FakeAudioVAE:
        def __init__(self):
            self.config = SimpleNamespace(sample_rate=44100)
            self.counter = {"cpu": 0}

        def decode(self, batch_latents, **kwargs):
            waveform = FakeWaveform(torch.zeros(batch_latents.shape[0], 1, 8), self.counter)
            return waveform, None, None

    talker = object.__new__(MingFlashOmniTalkerForConditionalGeneration)
    talker.config = SimpleNamespace(batch_vae_min_batch=4)
    talker.audio_vae = FakeAudioVAE()
    talker.audio_generator = FakeAudioGenerator()
    latents = [[torch.zeros(1, 2, 3)] for _ in range(4)]

    audios, sample_rate = talker.decode_batch_latents_for_runner(latents, stream_decode=False)

    assert sample_rate == 44100
    assert len(audios) == 4
    assert talker.audio_vae.counter["cpu"] == 1
