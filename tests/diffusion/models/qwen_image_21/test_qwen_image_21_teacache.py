# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Small tensor tests of the decode-only boundary, using the real TeaCache hook."""

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.cache.teacache.backend import TeaCacheBackend
from vllm_omni.diffusion.cache.teacache.extractors import extract_qwen_image21_context
from vllm_omni.diffusion.hooks import HookRegistry
from vllm_omni.diffusion.models.qwen_image_21.teacache import (
    QwenImage21DenoisingAdapter,
    enable_qwen_image21_teacache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

KVCache = list[dict[str, dict[str, torch.Tensor]]]


@dataclass
class _ParallelConfig:
    tensor_parallel_size: int = 1
    sequence_parallel_size: int = 1
    cfg_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    world_size: int = 1
    use_hsdp: bool = False
    hsdp_shard_size: int = 1
    hsdp_replicate_size: int = 1


@dataclass
class _CacheConfig:
    rel_l1_thresh: object
    coefficients: object
    cfg_rel_l1_thresh: object = None
    skip_timestep_range: object = None


class _Block(nn.Module):
    """Deterministic stand-in for expensive attention/MLP, including in-place writes."""

    def __init__(self, value):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(2, elementwise_affine=False)
        self.weight = nn.Parameter(torch.ones(2))
        self.value = value
        self.calls = 0

    def forward(self, hidden_states, *, kv_cache, cache_branch, cache_write_len, **kwargs):
        self.calls += 1
        if cache_write_len is not None:
            kv_cache[cache_branch] = {"key": hidden_states[:, :cache_write_len].clone()}
        else:
            assert "key" in kv_cache[cache_branch]
        return hidden_states.add_(self.value if cache_branch == "cond" else -self.value)


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(2), _Block(3)])
        self.causal_condition = True
        self.parallel_config = _ParallelConfig()
        self.prefix_kv_cache_dtype = None
        self._decode_graph_manager = None
        self.enable_cuda_graph_decode = True

    def _run_transformer_blocks(self, hidden_states, modulation, **kwargs):
        caches = kwargs.pop("kv_cache")
        for index, block in enumerate(self.transformer_blocks):
            hidden_states = block(hidden_states, modulation=modulation, kv_cache=caches[index], **kwargs)
        return hidden_states


@dataclass
class _Config:
    # Actual OmniDiffusionConfig field names, without its GPU/vLLM setup.
    step_execution: bool = False
    quantization_config: object = None
    lora_path: str | None = None
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    enable_distributed_layerwise_offload: bool = False
    diffusion_offload_config: dict[str, object] | None = None


class QwenImage21Pipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = _Transformer()
        self.od_config = _Config()


def _config(coefficients=None, threshold=0.2):
    return _CacheConfig(rel_l1_thresh=threshold, coefficients=coefficients or [0, 0, 0, 0, 0.1])


def _kwargs(cache, branch="cond", prefill=False):
    return dict(
        freqs=torch.ones(2, 1),
        target_token_mask=None,
        attn_metadata=None,
        kv_cache=cache,
        cache_branch=branch,
        cache_write_len=1 if prefill else None,
        sp_prefix_len=0,
        sp_decode=False,
        prepared_modulation=None,
        _pointwise_validated=False,
    )


def _input():
    return torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])


def _invoke(adapter, cache, branch="cond", prefill=False, timestep=0.5):
    return adapter.run(
        _input(),
        torch.zeros(2, 8),
        is_decode=not prefill,
        timestep=torch.tensor(timestep),
        **_kwargs(cache, branch, prefill),
    )


@torch.inference_mode()
def test_prefill_writes_every_layer_and_decode_reuses_only_residual():
    pipe = QwenImage21Pipeline()
    before = set(pipe.state_dict())
    enable_qwen_image21_teacache(pipe, _config())
    adapter = pipe.denoising_transformer
    cache: KVCache = [{}, {}]
    torch.testing.assert_close(_invoke(adapter, cache, prefill=True), _input() + 5)
    torch.testing.assert_close(cache[0]["cond"]["key"], torch.tensor([[[1.0, 2.0]]]))
    torch.testing.assert_close(cache[1]["cond"]["key"], torch.tensor([[[3.0, 4.0]]]))
    snapshot = [item["cond"]["key"].clone() for item in cache]
    for _ in range(3):
        torch.testing.assert_close(_invoke(adapter, cache), _input() + 5)
    # Prefill, first decode, and accumulated-distance=0.2 refresh; one decode skips.
    assert [block.calls for block in pipe.transformer.transformer_blocks] == [3, 3]
    for item, expected in zip(cache, snapshot):
        torch.testing.assert_close(item["cond"]["key"], expected)
    assert set(pipe.state_dict()) == before
    assert not pipe.transformer.enable_cuda_graph_decode


@torch.inference_mode()
def test_branch_identity_does_not_depend_on_call_parity():
    pipe = QwenImage21Pipeline()
    enable_qwen_image21_teacache(pipe, _config(threshold=100))
    cache: KVCache = [{}, {}]
    adapter = pipe.denoising_transformer
    _invoke(adapter, cache, "cond", prefill=True)
    _invoke(adapter, cache, "uncond", prefill=True)
    for branch, offset in [("cond", 5), ("uncond", -5), ("uncond", -5), ("cond", 5)]:
        torch.testing.assert_close(_invoke(adapter, cache, branch), _input() + offset)
    assert [block.calls for block in pipe.transformer.transformer_blocks] == [4, 4]


@pytest.mark.parametrize("do_true_cfg", [False, True])
@torch.inference_mode()
def test_real_hook_keeps_distinct_cfg_states_and_residuals(do_true_cfg):
    pipe = QwenImage21Pipeline()
    # The pipeline writes this flag on the transformer, not the hook adapter.
    pipe.transformer.do_true_cfg = do_true_cfg
    enable_qwen_image21_teacache(pipe, _config(threshold=100))
    adapter = pipe.denoising_transformer
    assert not hasattr(adapter, "do_true_cfg")
    cache: KVCache = [{}, {}]
    for branch in ("cond", "uncond"):
        _invoke(adapter, cache, branch, prefill=True)
    for branch in ("cond", "uncond", "uncond", "cond"):
        expected = _input() + (5 if branch == "cond" else -5)
        torch.testing.assert_close(_invoke(adapter, cache, branch), expected, rtol=0, atol=0)

    hook = HookRegistry.get_or_create(adapter).get_hook("teacache")
    states = hook.state_manager._states
    assert set(states) == {"teacache_positive", "teacache_negative"}
    positive, negative = states["teacache_positive"], states["teacache_negative"]
    assert positive is not negative
    assert positive.previous_residual.data_ptr() != negative.previous_residual.data_ptr()
    torch.testing.assert_close(positive.previous_residual, torch.full_like(_input(), 5), rtol=0, atol=0)
    torch.testing.assert_close(negative.previous_residual, torch.full_like(_input(), -5), rtol=0, atol=0)
    assert positive.cnt == negative.cnt == 2
    # Two prefills and the first decode of each branch compute; both later calls hit.
    assert [block.calls for block in pipe.transformer.transformer_blocks] == [4, 4]


@pytest.mark.parametrize("cache_on_denoising_boundary", [False, True])
@torch.inference_mode()
def test_refresh_finds_the_adapter_hook_without_a_pipeline_class_name(cache_on_denoising_boundary):
    class WrappedPipeline(QwenImage21Pipeline):
        pass

    pipe = WrappedPipeline()
    HookRegistry.get_or_create(pipe.transformer)
    enable_qwen_image21_teacache(pipe, _config(threshold=100))
    weight_owner, adapter = pipe.transformer, pipe.denoising_transformer
    if not cache_on_denoising_boundary:
        # Also preserve the generic transformer-owned cache path when the
        # denoising object has only an unrelated registry.
        pipe.transformer, pipe.denoising_transformer = adapter, weight_owner
    cache: KVCache = [{}, {}]
    _invoke(adapter, cache, prefill=True)
    _invoke(adapter, cache)
    weight_owner.transformer_blocks[0].value = 7
    TeaCacheBackend(_config(threshold=100)).refresh(pipe, num_inference_steps=50, verbose=False)
    torch.testing.assert_close(_invoke(adapter, cache), _input() + 10)


@pytest.mark.parametrize("with_transformer_hooks", [False, True])
@torch.inference_mode()
def test_refresh_and_new_same_shape_prefill_clear_residuals(with_transformer_hooks):
    pipe = QwenImage21Pipeline()
    if with_transformer_hooks:
        HookRegistry.get_or_create(pipe.transformer)
    backend = TeaCacheBackend(_config(threshold=100))
    backend.enable(pipe)
    cache: KVCache = [{}, {}]
    adapter = pipe.denoising_transformer
    _invoke(adapter, cache, prefill=True)
    _invoke(adapter, cache)
    pipe.transformer.transformer_blocks[0].value = 7
    backend.refresh(pipe, num_inference_steps=50, verbose=False)
    torch.testing.assert_close(_invoke(adapter, cache), _input() + 10)
    pipe.transformer.transformer_blocks[0].value = 11
    fresh: KVCache = [{}, {}]
    _invoke(adapter, fresh, prefill=True)
    torch.testing.assert_close(_invoke(adapter, fresh), _input() + 14)


@torch.inference_mode()
def test_feature_matches_first_blocks_normalization_and_prepared_scale():
    transformer = _Transformer()
    adapter = QwenImage21DenoisingAdapter(transformer)
    cache = [{"cond": {"key": torch.zeros(1)}}, {"cond": {"key": torch.zeros(1)}}]
    kwargs = _kwargs(cache)
    kwargs["prepared_modulation"] = (torch.full((1, 2, 2), 3.0), None, None, None)
    context = extract_qwen_image21_context(adapter, _input(), torch.zeros(2, 8), **kwargs)
    torch.testing.assert_close(context.modulated_input, transformer.transformer_blocks[0].img_norm1(_input()) * 3)
    assert context.extra_states["teacache_branch"] == "positive"


@pytest.mark.parametrize("coefficients", [None, [], [1] * 4, [1, 2, 3, 4, float("nan")], [1, 2, 3, 4, "5"]])
def test_invalid_coefficients_rejected_before_mutating_pipeline(coefficients):
    pipe = QwenImage21Pipeline()
    with pytest.raises((ValueError, TypeError), match="coefficients"):
        enable_qwen_image21_teacache(pipe, _CacheConfig(coefficients=coefficients, rel_l1_thresh=0.2))
    assert pipe.transformer.enable_cuda_graph_decode
    assert not hasattr(pipe, "denoising_transformer")


@pytest.mark.parametrize("threshold", [None, 0, -1, float("nan"), float("inf"), "0.2", True])
def test_invalid_threshold_rejected(threshold):
    with pytest.raises((ValueError, TypeError), match="rel_l1_thresh"):
        enable_qwen_image21_teacache(QwenImage21Pipeline(), _config(threshold=threshold))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cfg_rel_l1_thresh", 0),
        ("cfg_rel_l1_thresh", float("nan")),
        ("cfg_rel_l1_thresh", "0.1"),
        ("cfg_rel_l1_thresh", True),
        ("skip_timestep_range", (0.3,)),
        ("skip_timestep_range", (0.8, 0.3)),
        ("skip_timestep_range", (-0.1, 0.5)),
        ("skip_timestep_range", (0.3, 1.5)),
        ("skip_timestep_range", (0.3, float("nan"))),
        ("skip_timestep_range", (True, 0.5)),
        ("skip_timestep_range", "0.3,0.8"),
    ],
)
def test_invalid_skip_policy_rejected_before_mutating_pipeline(field, value):
    pipe = QwenImage21Pipeline()
    config = _config()
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        enable_qwen_image21_teacache(pipe, config)
    assert pipe.transformer.enable_cuda_graph_decode
    assert not hasattr(pipe, "denoising_transformer")


@torch.inference_mode()
def test_steps_outside_timestep_window_compute_and_reset_accumulation():
    pipe = QwenImage21Pipeline()
    # Each decode adds a rescaled distance of 0.1; 0.25 allows two consecutive hits.
    config = _config(threshold=0.25)
    config.skip_timestep_range = (0.3, 0.8)
    enable_qwen_image21_teacache(pipe, config)
    adapter = pipe.denoising_transformer
    cache: KVCache = [{}, {}]
    _invoke(adapter, cache, prefill=True, timestep=1.0)
    blocks = pipe.transformer.transformer_blocks
    computed = []
    for timestep in (0.9, 0.7, 0.85, 0.6, 0.5, 0.4, 0.2):
        before = blocks[0].calls
        _invoke(adapter, cache, timestep=timestep)
        computed.append(blocks[0].calls > before)
    # 0.9: first decode. 0.85 and 0.2: outside the window. The forced step at
    # 0.85 resets the 0.2 accumulated so far, so 0.6 and 0.5 hit before 0.4 computes.
    assert computed == [True, False, True, False, False, True, True]


@pytest.mark.parametrize(("do_true_cfg", "block_calls"), [(False, 3), (True, 2)])
@torch.inference_mode()
def test_true_cfg_requests_use_their_own_threshold(do_true_cfg, block_calls):
    pipe = QwenImage21Pipeline()
    pipe.transformer.do_true_cfg = do_true_cfg
    config = _config(threshold=0.15)
    config.cfg_rel_l1_thresh = 100
    enable_qwen_image21_teacache(pipe, config)
    adapter = pipe.denoising_transformer
    cache: KVCache = [{}, {}]
    _invoke(adapter, cache, prefill=True, timestep=1.0)
    for _ in range(3):
        _invoke(adapter, cache)
    # Prefill and first decode always compute. Without true CFG the accumulated
    # 0.2 crosses 0.15 on the third decode; with it, both later decodes hit.
    assert [block.calls for block in pipe.transformer.transformer_blocks] == [block_calls] * 2


@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize(
    "offload",
    [
        {"enable_cpu_offload": True},
        {"enable_layerwise_offload": True},
        {"enable_distributed_layerwise_offload": True},
        {"diffusion_offload_config": {"mode": "layer", "components": ["dit"]}},
        {"diffusion_offload_config": {"mode": "module", "components": ["text_encoder"]}},
    ],
)
def test_unqualified_offload_is_rejected_before_installation(offload, calibration):
    from vllm_omni.diffusion.models.qwen_image_21.teacache import enable_qwen_image21_calibration

    pipe = QwenImage21Pipeline()
    pipe.od_config = _Config(**offload)
    with pytest.raises(ValueError, match="offload"):
        if calibration:
            enable_qwen_image21_calibration(pipe)
        else:
            enable_qwen_image21_teacache(pipe, _config())
    assert not hasattr(pipe, "denoising_transformer")
    assert pipe.transformer.enable_cuda_graph_decode


@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize("prefix_dtype", ["fp8_e4m3", "fp8_e4m3_v"])
def test_unqualified_fp8_prefix_storage_is_rejected(prefix_dtype, calibration):
    from vllm_omni.diffusion.models.qwen_image_21.teacache import enable_qwen_image21_calibration

    pipe = QwenImage21Pipeline()
    pipe.transformer.prefix_kv_cache_dtype = prefix_dtype
    with pytest.raises(ValueError, match="prefix"):
        if calibration:
            enable_qwen_image21_calibration(pipe)
        else:
            enable_qwen_image21_teacache(pipe, _config())
    assert not hasattr(pipe, "denoising_transformer")


@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize("already_wrapped", [False, True])
@torch.inference_mode()
def test_dynamic_lora_is_rejected_at_installation_or_before_prefill(monkeypatch, calibration, already_wrapped):
    from vllm_omni.diffusion.models.qwen_image_21 import decode_graph
    from vllm_omni.diffusion.models.qwen_image_21.teacache import enable_qwen_image21_calibration

    pipe = QwenImage21Pipeline()
    # Reuse the existing runtime LoRA-wrapper detector. This unit test isolates
    # the rejection boundary; it does not load an adapter or qualify LoRA math.
    monkeypatch.setattr(decode_graph, "_dynamic_lora_wrappers_present", lambda model: already_wrapped)

    def install():
        if calibration:
            enable_qwen_image21_calibration(pipe)
        else:
            enable_qwen_image21_teacache(pipe, _config())

    if already_wrapped:
        with pytest.raises(ValueError, match="LoRA"):
            install()
        assert not hasattr(pipe, "denoising_transformer")
    else:
        install()
        monkeypatch.setattr(decode_graph, "_dynamic_lora_wrappers_present", lambda model: True)
        cache: KVCache = [{}, {}]
        with pytest.raises(ValueError, match="LoRA"):
            _invoke(pipe.denoising_transformer, cache, prefill=True)
        assert cache == [{}, {}]
        assert [block.calls for block in pipe.transformer.transformer_blocks] == [0, 0]


@pytest.mark.parametrize("field", ["tensor_parallel_size", "sequence_parallel_size", "cfg_parallel_size"])
def test_parallelism_is_rejected(field):
    pipe = QwenImage21Pipeline()
    setattr(pipe.transformer.parallel_config, field, 2)
    with pytest.raises(ValueError, match="single GPU"):
        enable_qwen_image21_teacache(pipe, _config())


@pytest.mark.parametrize("calibration", [False, True])
def test_standalone_hsdp_is_rejected_by_both_enablers(calibration):
    from vllm_omni.diffusion.models.qwen_image_21.teacache import enable_qwen_image21_calibration

    pipe = QwenImage21Pipeline()
    # Legal standalone HSDP: all other sizes stay one, but resolved WORLD=2.
    parallel = pipe.transformer.parallel_config
    parallel.use_hsdp = True
    parallel.hsdp_shard_size = 2
    parallel.hsdp_replicate_size = 1
    parallel.world_size = 2
    with pytest.raises(ValueError, match="single GPU"):
        if calibration:
            enable_qwen_image21_calibration(pipe)
        else:
            enable_qwen_image21_teacache(pipe, _config())
    assert not hasattr(pipe, "denoising_transformer")
    assert pipe.transformer.enable_cuda_graph_decode


def test_step_execution_is_rejected():
    pipe = QwenImage21Pipeline()
    pipe.od_config.step_execution = True
    with pytest.raises(ValueError, match="request mode"):
        enable_qwen_image21_teacache(pipe, _config())


@torch.inference_mode()
def test_batching_is_rejected_before_prefix_write():
    pipe = QwenImage21Pipeline()
    enable_qwen_image21_teacache(pipe, _config())
    cache: KVCache = [{}, {}]
    with pytest.raises(ValueError, match="batch size 1"):
        pipe.denoising_transformer.run(
            _input().repeat(2, 1, 1),
            torch.zeros(3, 8),
            is_decode=False,
            timestep=torch.tensor(1.0),
            **_kwargs(cache, prefill=True),
        )
    assert cache == [{}, {}]


class _TimeEmbed(nn.Module):
    def forward(self, timestep, hidden_states):
        return timestep[:, None].repeat(1, 2)


class _OutputNorm(nn.Module):
    def forward(self, hidden_states, temb, mask):
        # This must run even on a cache hit: it depends on the current timestep.
        return hidden_states + temb[0, 0]


@pytest.mark.parametrize(("skip_timestep_range", "block_calls"), [(None, 4), ((0.3, 0.4), 5)])
@torch.inference_mode()
def test_real_transformer_forward_keeps_output_layers_outside_cached_blocks(skip_timestep_range, block_calls):
    from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
        QwenImage21Rope,
        QwenImage21SequencePrepare,
        QwenImage21Transformer2DModel,
    )

    model = QwenImage21Transformer2DModel.__new__(QwenImage21Transformer2DModel)
    nn.Module.__init__(model)
    model.parallel_config = None
    model.causal_condition = True
    model.causal_block = True
    model.prefix_kv_cache_dtype = None
    model.transformer_blocks = nn.ModuleList([_Block(2), _Block(3)])
    model._decode_graph_manager = None
    model.enable_cuda_graph_decode = False
    model.sequence_prepare = QwenImage21SequencePrepare(nn.Identity(), QwenImage21Rope(10000, [2, 2, 2]))
    model.txt_in = nn.Identity()
    model.time_text_embed = _TimeEmbed()
    model.modulation = nn.Linear(2, 8, bias=False)
    nn.init.zeros_(model.modulation.weight)
    model.norm_out = _OutputNorm()
    model.proj_out = nn.Identity()
    pipe = QwenImage21Pipeline()
    pipe.transformer = model
    inputs = dict(
        hidden_states=torch.tensor([[[1.0, 2.0]] * 4]),
        encoder_hidden_states=torch.tensor([[[3.0, 4.0]]]),
        img_shapes=[[(1, 2, 2)]],
        img_mask=torch.tensor([[False, True]]),
        attention_kwargs={"attn_path": "mask"},
        return_dict=False,
    )
    reference_cache: KVCache = [{}, {}]
    model(**inputs, timestep=torch.tensor([1.0]), kv_cache=reference_cache)
    expected = model(**inputs, timestep=torch.tensor([0.5]), kv_cache=reference_cache)[0]
    keys = set(model.state_dict())
    config = _config(threshold=100)
    config.skip_timestep_range = skip_timestep_range
    enable_qwen_image21_teacache(pipe, config)
    cache: KVCache = [{}, {}]
    model(**inputs, timestep=torch.tensor([1.0]), kv_cache=cache)
    first = model(**inputs, timestep=torch.tensor([0.5]), kv_cache=cache)[0]
    # A hit without a window; with one, t=0.25 is outside it and recomputes.
    last = model(**inputs, timestep=torch.tensor([0.25]), kv_cache=cache)[0]
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(last, expected - 0.25, rtol=0, atol=0)
    assert set(model.state_dict()) == keys
    assert [block.calls for block in model.transformer_blocks] == [block_calls] * 2


@torch.inference_mode()
def test_dense_collector_separates_branches_and_snapshots_inplace_outputs():
    from vllm_omni.diffusion.models.qwen_image_21.teacache import QwenImage21CalibrationHook

    transformer = _Transformer()
    adapter = QwenImage21DenoisingAdapter(transformer)
    collector = QwenImage21CalibrationHook(max_observations=3)
    HookRegistry.get_or_create(adapter).register_hook(collector._HOOK_NAME, collector)
    cache = [{"cond": {"key": torch.zeros(1)}, "uncond": {"key": torch.zeros(1)}} for _ in range(2)]
    modulation = torch.zeros(2, 8)
    timestep = torch.tensor(0.5)
    for branch in ("cond", "uncond"):
        output = adapter.run(_input(), modulation, is_decode=True, timestep=timestep, **_kwargs(cache, branch))
        output.fill_(999)  # Caller reuse must not corrupt the collector's history.
    # For cond, output means are 7.5 and 10, so rel-L1 = 2.5 / 7.5.
    doubled = _input() * 2
    actual = adapter.run(doubled, modulation, is_decode=True, timestep=timestep, **_kwargs(cache))
    torch.testing.assert_close(actual, _input() * 2 + 5)
    assert len(collector.observations) == 1
    observation = collector.observations[0]
    assert observation.branch == "positive"
    assert observation.input_rel_l1 == pytest.approx(0, abs=2e-5)
    assert observation.output_rel_l1 == pytest.approx(1 / 3)
    # Every call ran both blocks; collection never reuses a residual.
    assert [block.calls for block in transformer.transformer_blocks] == [3, 3]


@torch.inference_mode()
def test_collector_resets_pairs_at_prefill_and_honors_budget():
    from vllm_omni.diffusion.models.qwen_image_21.teacache import QwenImage21CalibrationHook

    transformer = _Transformer()
    adapter = QwenImage21DenoisingAdapter(transformer)
    collector = QwenImage21CalibrationHook(max_observations=1)
    HookRegistry.get_or_create(adapter).register_hook(collector._HOOK_NAME, collector)
    cache: KVCache = [{}, {}]
    _invoke(adapter, cache, prefill=True)
    _invoke(adapter, cache)
    _invoke(adapter, cache, prefill=True)
    _invoke(adapter, cache)
    assert collector.observations == []  # No pair across the request boundary.
    for _ in range(4):
        _invoke(adapter, cache)
    assert len(collector.observations) == 1
    assert [block.calls for block in transformer.transformer_blocks] == [8, 8]


def test_collector_cannot_be_installed_over_approximation_hook():
    from vllm_omni.diffusion.models.qwen_image_21.teacache import QwenImage21CalibrationHook

    pipe = QwenImage21Pipeline()
    enable_qwen_image21_teacache(pipe, _config())
    with pytest.raises(ValueError, match="dense"):
        HookRegistry.get_or_create(pipe.denoising_transformer).register_hook(
            "teacache_collector", QwenImage21CalibrationHook()
        )


@pytest.mark.parametrize("budget", [0, -1, 1.5, True, 100001])
def test_collector_rejects_invalid_budget(budget):
    from vllm_omni.diffusion.models.qwen_image_21.teacache import QwenImage21CalibrationHook

    with pytest.raises(ValueError, match="max_observations"):
        QwenImage21CalibrationHook(max_observations=budget)


def test_quartic_fit_recovers_known_relationship():
    from vllm_omni.diffusion.models.qwen_image_21.teacache import (
        QwenImage21Observation,
        fit_qwen_image21_coefficients,
    )

    # Independently specified y = 2*x + 0.1, with >5 unique x values.
    observations = [
        QwenImage21Observation("positive", x, y)
        for x, y in (
            (0.0, 0.1),
            (0.1, 0.3),
            (0.2, 0.5),
            (0.3, 0.7),
            (0.4, 0.9),
            (0.5, 1.1),
        )
    ]
    assert fit_qwen_image21_coefficients(observations) == pytest.approx([0, 0, 0, 2, 0.1], abs=1e-9)


@pytest.mark.parametrize("pairs", [[], [(0.1, 0.2)] * 6, [(0.1, float("nan"))] * 6, [(-1, 0.2)] * 6])
def test_fit_rejects_insufficient_nonfinite_or_degenerate_points(pairs):
    from vllm_omni.diffusion.models.qwen_image_21.teacache import (
        QwenImage21Observation,
        fit_qwen_image21_coefficients,
    )

    with pytest.raises(ValueError):
        fit_qwen_image21_coefficients([QwenImage21Observation("positive", x, y) for x, y in pairs])
