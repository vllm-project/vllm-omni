# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
from vllm_omni.diffusion.models.minimax_h3.adaln_cache import (
    FORMAT_VERSION,
    MODES,
    MiniMaxH3AdalnCache,
    MiniMaxH3RuntimeAdalnCache,
    build_cache,
    canonical_json,
    input_names,
    schedule_contract,
    tensor_digest,
    timestep_plans,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def tp1(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1)


def _contract(mode="t2va", **updates):
    values = dict(
        mode=mode, num_steps=4, base_schedule=(0.999, 0.749, 0.5, 0.25, 0.0), flow_shift=12.0, audio_flow_shift=3.0
    )
    return schedule_contract(**(values | updates))


def _fixture(tmp_path, mode="t2va", adapter=None, *, base_model=False):
    arch = h3.MiniMaxH3DiTArchConfig(
        num_layers=2, hidden_size=8, timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4
    )
    generator = torch.Generator().manual_seed(4)
    weights = {}
    for name, shape in {
        "time_embedder.proj_in.weight": (8, 4),
        "time_embedder.proj_in.bias": (8,),
        "time_embedder.proj_out.weight": (4, 8),
        "time_embedder.proj_out.bias": (4,),
    }.items():
        weights[name] = torch.randn(shape, generator=generator)
    for name in sorted(input_names(arch.num_layers)):
        if name.startswith("time_embedder."):
            continue
        width = (2 if name.startswith("final_layer.") else 18) * arch.hidden_size
        shape = (width, arch.time_embed_dim) if name.endswith("weight") else (width,)
        weights[name] = torch.randn(shape, generator=generator).to(torch.bfloat16)
    variant = "ref2va" if mode.startswith("ref2va") else "fl2va"
    payload, manifest = build_cache(
        arch,
        weights.items(),
        contract=_contract(mode, **({"num_steps": 50, "base_schedule": None} if base_model else {})),
        model_variant=variant,
        adapter_sha256=adapter,
        device=torch.device("cpu"),
    )
    path = tmp_path / "adaln.safetensors"
    _save(path, payload, manifest)
    return arch, weights, payload, manifest, path


def _save(path, payload, manifest):
    save_file(payload, str(path), metadata={"format_version": FORMAT_VERSION, "manifest": canonical_json(manifest)})


def _ready(arch, weights, path, variant="fl2va", adapter=None):
    cache = MiniMaxH3AdalnCache(arch, path=str(path), model_variant=variant)
    cache.bind_adapter(adapter)
    for name, weight in weights.items():
        cache.verify_weight(name, weight)
    cache.finish_loading(torch.device("cpu"))
    return cache


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("base_model", [False, True], ids=["explicit-ladder", "base-h3-50"])
def test_builder_projection_parity_all_tasks(tmp_path, mode, base_model):
    arch, weights, payload, manifest, path = _fixture(tmp_path, mode, base_model=base_model)
    cache = _ready(arch, weights, path, manifest["model_variant"])
    for i, plan in enumerate(timestep_plans(manifest["schedule"])):
        blocks, final = cache.lookup(plan)
        embedding = payload["time_embeddings"][i, : len(plan)]
        activated = torch.nn.functional.silu(embedding).to(torch.bfloat16)
        for block_index, pieces in enumerate(blocks):
            prefix = f"blocks.{block_index}.adaln_proj.linear"
            expected = torch.nn.functional.linear(activated, weights[prefix + ".weight"], weights[prefix + ".bias"])
            actual = torch.stack(pieces, dim=1).reshape(len(plan), -1)
            assert torch.equal(actual, expected)
        prefix = "final_layer.adaln_proj.linear"
        expected = torch.nn.functional.linear(activated, weights[prefix + ".weight"], weights[prefix + ".bias"])
        assert torch.equal(torch.stack(final, dim=1).reshape(len(plan), -1), expected)
    assert not cache.state_dict()


@pytest.mark.parametrize("change", ["variant", "architecture", "weights", "payload", "plans", "nan", "version"])
def test_reject_invalid_sidecar(tmp_path, change):
    arch, weights, payload, manifest, path = _fixture(tmp_path)
    payload = {key: value.clone() for key, value in payload.items()}
    manifest = copy.deepcopy(manifest)
    if change == "variant":
        manifest["model_variant"] = "ref2va"
    elif change == "architecture":
        manifest["architecture"]["hidden_size"] += 1
    elif change == "weights":
        manifest["weights"].pop(next(iter(weights)))
    elif change == "payload":
        payload["block_params"][0, 0, 0, 0] += 1
    elif change == "plans":
        payload["plan_timesteps"][0, 0] += 0.01
        manifest["payload"]["plan_timesteps"] = tensor_digest(payload["plan_timesteps"])
    elif change == "nan":
        payload["final_params"][0, 0, 0] = float("nan")
        manifest["payload"]["final_params"] = tensor_digest(payload["final_params"])
    _save(path, payload, manifest)
    if change == "version":
        save_file(payload, str(path), metadata={"format_version": "3"})
    with pytest.raises(ValueError):
        MiniMaxH3AdalnCache(arch, path=str(path), model_variant="fl2va")


def test_weight_adapter_schedule_and_uncovered_timestep_checks(tmp_path):
    arch, weights, _, manifest, path = _fixture(tmp_path, adapter="a" * 64)
    cache = MiniMaxH3AdalnCache(arch, path=str(path), model_variant="fl2va")
    with pytest.raises(ValueError, match="adapter mismatch"):
        cache.bind_adapter(None)
    cache.bind_adapter("a" * 64)
    with pytest.raises(ValueError, match="Missing"):
        cache.finish_loading(torch.device("cpu"))
    name = next(iter(weights))
    with pytest.raises(ValueError, match="effective weight mismatch"):
        cache.verify_weight(name, weights[name] + 1)
    for name, weight in weights.items():
        cache.verify_weight(name, weight)
    cache.finish_loading(torch.device("cpu"))
    for update in (
        {"flow_shift": 10.0},
        {"audio_flow_shift": 4.0},
        {"mode": "fl2va"},
        {"base_schedule": [0.99, 0.74, 0.5, 0.25, 0]},
    ):
        with pytest.raises(ValueError):
            cache.check_request(**(manifest["schedule"] | update))
    with pytest.raises(ValueError, match="does not cover"):
        cache.lookup(torch.tensor([0.123]))


def test_adapter_switch_invalidates_runtime_cache(mocker):
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    cache = mocker.Mock()
    pipeline.transformer = SimpleNamespace(adaln_cache=cache)
    from vllm.lora.request import LoRARequest

    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    sampling = OmniDiffusionSamplingParams(lora_request=LoRARequest("one", 1, "one"), lora_scale=1.0)
    pipeline._prepare_adaln_adapter(sampling)
    pipeline._prepare_adaln_adapter(sampling)
    assert cache.clear.call_count == 1
    sampling.lora_scale = 0.5
    pipeline._prepare_adaln_adapter(sampling)
    sampling.lora_request = None
    pipeline._prepare_adaln_adapter(sampling)
    assert cache.clear.call_count == 3


def test_bad_sidecar_never_drops_runtime_weights(tmp_path):
    arch, weights, _, _, path = _fixture(tmp_path)
    candidate = MiniMaxH3AdalnCache(arch, path=str(path), model_variant="fl2va")
    component = SimpleNamespace(_adaln_sidecar_candidate=candidate)
    changed = list(weights.items())
    changed[0] = changed[0][0], changed[0][1] + 1
    result = list(MiniMaxH3Pipeline._verify_adaln_weights(component, iter(changed)))
    assert len(result) == len(changed)
    assert all(tensor is source for (_, tensor), (_, source) in zip(result, changed))
    assert component._adaln_sidecar_candidate is None


@pytest.mark.parametrize("tp_size", [1, 2])
def test_native_constructor_enables_cache_by_default(monkeypatch, tp_size):
    from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import (
        _FakeAttention,
        _FakeLinear,
        _small_od_config,
    )

    for name in ("ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear", "QKVParallelLinear"):
        monkeypatch.setattr(h3, name, _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: tp_size)
    config = _small_od_config()
    model = h3.MiniMaxH3DiTModel(config, diffusers_weights=False)
    assert model.adaln_cache.max_bytes > 0
    assert model.blocks[0].adaln_proj._adaln_cache is model.adaln_cache
    assert model.final_layer.adaln_proj._adaln_cache is model.adaln_cache
    assert "blocks.0.adaln_proj.linear.weight" in dict(model.named_parameters())
    config.cache_config = {"minimax_h3_adaln_cache": False}
    disabled = h3.MiniMaxH3DiTModel(config, diffusers_weights=False)
    assert disabled.adaln_cache.max_bytes == 0


@pytest.fixture
def offload_model_config(monkeypatch):
    from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import (
        _FakeAttention,
        _FakeLinear,
        _small_od_config,
    )

    for name in ("ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear", "QKVParallelLinear"):
        monkeypatch.setattr(h3, name, _FakeLinear)
    monkeypatch.setattr(h3, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(h3.current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)
    monkeypatch.setattr(h3.current_omni_platform, "get_device_capability", lambda device_id: (12, 0))
    config = _small_od_config()
    config.enforce_eager = True
    config.cache_config = {"minimax_h3_adaln_offload": True}
    return config


def test_compiled_h3_projection_keeps_cache_and_tensor_graphs(monkeypatch, mocker):
    class Linear(torch.nn.Linear):
        def __init__(self, in_features, out_features, *, params_dtype, bias=True, **kwargs):
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)

        def forward(self, x):
            return super().forward(x), None

    monkeypatch.setattr(h3, "ColumnParallelLinear", Linear)
    arch = h3.MiniMaxH3DiTArchConfig(hidden_size=8, time_embed_dim=4)
    cache = MiniMaxH3RuntimeAdalnCache()
    proj = h3.MiniMaxH3AdalnProj(
        arch,
        18 * arch.hidden_size,
        None,
        expand_ratio=6,
        modality_num=3,
        prefix="blocks.0.adaln_proj",
        adaln_cache=cache,
    )
    calls = mocker.spy(proj.linear, "forward")
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    def run(embedding):
        # Include prepare in the compiled function to exercise full-model as
        # well as regional compilation's projection boundary.
        cache.prepare(embedding)
        return proj(embedding)[0].sin() + 1

    compiled = torch.compile(run, backend=backend, dynamic=True)
    embedding = torch.randn(2, arch.time_embed_dim)
    with torch.no_grad():
        expected = run(embedding)
        cache.clear()
        calls.reset_mock()
        first = compiled(embedding)
        graph_count = len(graphs)
        second = compiled(embedding.clone())
        assert graph_count > 0 and len(graphs) == graph_count
        assert calls.call_count == 1 and cache.hits == 1
        assert torch.equal(first, expected) and torch.equal(second, expected)
        proj.linear.weight.add_(0.125)
        changed = compiled(embedding)
        assert calls.call_count == 2
        cache.clear()
        assert torch.equal(changed, run(embedding))


@pytest.mark.parametrize("enforce_eager", [False, True], ids=["compiled", "eager"])
@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
@pytest.mark.parametrize("config_type", [dict, SimpleNamespace], ids=["mapping", "attributes"])
def test_adaln_offload_allocates_only_projection_weights_on_host(
    offload_model_config, explicit, config_type, enforce_eager
):
    offload_model_config.enforce_eager = enforce_eager
    offload_model_config.cache_config = config_type(**({"minimax_h3_adaln_offload": True} if explicit else {}))
    # A non-CPU construction context reproduces how the loader allocates the
    # model, without requiring a large checkpoint or a GPU in this CPU test.
    with torch.device("meta"):
        model = h3.MiniMaxH3DiTModel(offload_model_config)
    for name, parameter in model.named_parameters():
        assert parameter.device.type == ("cpu" if ".adaln_proj.linear." in name else "meta")
    assert model.adaln_cache.offload_weights
    assert model.adaln_cache.max_bytes > 0


@pytest.mark.parametrize("config_type", [dict, SimpleNamespace], ids=["mapping", "attributes"])
def test_adaln_offload_can_be_disabled_without_disabling_cache(offload_model_config, config_type):
    offload_model_config.cache_config = config_type(minimax_h3_adaln_offload=False)
    with torch.device("meta"):
        model = h3.MiniMaxH3DiTModel(offload_model_config)
    assert all(p.device.type == "meta" for p in model.parameters())
    assert not model.adaln_cache.offload_weights
    assert model.adaln_cache.max_bytes > 0


@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
@pytest.mark.parametrize("setting", ["module", "layer", "hsdp", "quantized", "non_cuda"])
def test_adaln_offload_preserves_incompatible_modes(offload_model_config, monkeypatch, setting, explicit):
    config = offload_model_config
    config.cache_config = {"minimax_h3_adaln_offload": True} if explicit else {}
    quant = None
    if setting in {"module", "layer"}:
        config.diffusion_offload_config = {"mode": setting, "components": ["dit"]}
    elif setting == "hsdp":
        config.parallel_config.use_hsdp = True
    elif setting == "non_cuda":
        monkeypatch.setattr(h3.current_omni_platform, "is_cuda", lambda: False)
    else:
        quant = object()
    if explicit:
        with pytest.raises(ValueError, match="[Aa]da[Ll][Nn]|adaln"):
            h3.MiniMaxH3DiTModel(config, quant_config=quant)
    else:
        model = h3.MiniMaxH3DiTModel(config, quant_config=quant)
        assert not model.adaln_cache.offload_weights
        assert model.adaln_cache.max_bytes > 0


@pytest.mark.parametrize("value", ["true", None, 1])
def test_adaln_offload_requires_boolean(offload_model_config, value):
    offload_model_config.cache_config["minimax_h3_adaln_offload"] = value
    with pytest.raises(ValueError, match="minimax_h3_adaln_offload must be a boolean"):
        h3.MiniMaxH3DiTModel(offload_model_config)


def test_adaln_offload_keeps_text_encoder_offload_independent(offload_model_config):
    offload_model_config.cache_config = {}
    offload_model_config.diffusion_offload_config = {"mode": "module", "components": ["text_encoder"]}
    model = h3.MiniMaxH3DiTModel(offload_model_config)
    assert model.adaln_cache.offload_weights


@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (10, 0), (12, 0), (12, 1), None])
@pytest.mark.parametrize("explicit", [False, True], ids=["default", "explicit"])
def test_adaln_offload_default_is_limited_to_worker_sm120(offload_model_config, monkeypatch, capability, explicit):
    seen = []

    def get_capability(device_id):
        seen.append(device_id)
        return capability

    monkeypatch.setattr(h3.current_omni_platform, "get_device_capability", get_capability)
    offload_model_config.cache_config = {"minimax_h3_adaln_offload": True} if explicit else {}
    with torch.device("meta"):
        model = h3.MiniMaxH3DiTModel(offload_model_config)
    expected = explicit or capability == (12, 0)
    assert model.adaln_cache.offload_weights is expected
    assert seen == [3]
    assert model.blocks[0].adaln_proj.linear.weight.device.type == ("cpu" if expected else "meta")
    assert model.adaln_cache.max_bytes > 0


def test_disabling_adaln_cache_keeps_weights_resident_by_default(offload_model_config):
    offload_model_config.cache_config = {"minimax_h3_adaln_cache": False}
    model = h3.MiniMaxH3DiTModel(offload_model_config)
    assert not model.adaln_cache.offload_weights
    assert model.adaln_cache.max_bytes == 0


def test_optional_sidecar_seeds_runtime_projection(tmp_path, mocker):
    arch, weights, payload, _, path = _fixture(tmp_path)
    sidecar = _ready(arch, weights, path)
    runtime = MiniMaxH3RuntimeAdalnCache()
    name = "blocks.0.adaln_proj.linear"
    linear = torch.nn.Linear(arch.time_embed_dim, 18 * arch.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(weights[name + ".weight"])
        linear.bias.copy_(weights[name + ".bias"])
        runtime.seed(sidecar, {name: linear})
        x = payload["time_embeddings"][1, :2].clone()
        runtime.prepare(x)
        compute = mocker.Mock(side_effect=lambda: linear(torch.nn.functional.silu(x).to(torch.bfloat16)))
        result = runtime.project(name, linear, x, compute)
        assert compute.call_count == 0
        assert torch.equal(result, compute())
        linear.weight.add_(1)
        runtime.project(name, linear, x, compute)
        assert compute.call_count == 2


def test_failed_seed_does_not_attach_sidecar(tmp_path):
    arch, weights, _, _, path = _fixture(tmp_path)
    sidecar = _ready(arch, weights, path)
    runtime = MiniMaxH3RuntimeAdalnCache()
    linear = torch.nn.Linear(arch.time_embed_dim, 18 * arch.hidden_size, dtype=torch.bfloat16)
    with torch.inference_mode():
        unversioned = torch.nn.Linear(arch.time_embed_dim, 2 * arch.hidden_size, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="Inference tensors do not track version counter"):
        runtime.seed(sidecar, {"blocks.0.adaln_proj.linear": linear, "final_layer.adaln_proj.linear": unversioned})

    assert runtime.sidecar is None
    assert runtime._sidecar_signatures == {}


@pytest.mark.parametrize("mode", MODES)
def test_sidecar_prepare_uses_one_fingerprint_without_tensor_scan(tmp_path, mocker, mode):
    import vllm_omni.diffusion.cache.exact_projection_cache as cache_module

    arch, weights, payload, manifest, path = _fixture(tmp_path, mode=mode, base_model=True)
    sidecar = _ready(arch, weights, path, manifest["model_variant"])
    runtime = MiniMaxH3RuntimeAdalnCache()
    name = "final_layer.adaln_proj.linear"
    linear = torch.nn.Linear(arch.time_embed_dim, 2 * arch.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(weights[name + ".weight"])
        linear.bias.copy_(weights[name + ".bias"])
        runtime.seed(sidecar, {name: linear})
        count = int(payload["plan_lengths"][-1])
        embedding = payload["time_embeddings"][-1, :count].clone()
        expected = linear(torch.nn.functional.silu(embedding).to(torch.bfloat16))
        comparisons = mocker.spy(torch, "equal")
        fingerprint = mocker.spy(cache_module, "tensor_digest")
        compute = mocker.Mock(side_effect=AssertionError("matching sidecar must seed the projection"))

        runtime.prepare(embedding)
        actual = runtime.project(name, linear, embedding, compute)

        comparisons.assert_not_called()
        fingerprint.assert_called_once()
        assert torch.equal(actual, expected)
        compute.assert_not_called()


def test_sidecar_rechecks_numerical_environment_at_projection(tmp_path, mocker):
    arch, weights, payload, _, path = _fixture(tmp_path)
    sidecar = _ready(arch, weights, path)
    runtime = MiniMaxH3RuntimeAdalnCache()
    name = "final_layer.adaln_proj.linear"
    linear = torch.nn.Linear(arch.time_embed_dim, 2 * arch.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(weights[name + ".weight"])
        linear.bias.copy_(weights[name + ".bias"])
        runtime.seed(sidecar, {name: linear})
        count = int(payload["plan_lengths"][-1])
        embedding = payload["time_embeddings"][-1, :count].clone()
        runtime.prepare(embedding)
        compute = mocker.Mock(side_effect=lambda: linear(torch.nn.functional.silu(embedding).to(torch.bfloat16)))
        precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high" if precision == "highest" else "highest")
            actual = runtime.project(name, linear, embedding, compute)
            compute.assert_called_once()
            assert torch.equal(actual, compute())
        finally:
            torch.set_float32_matmul_precision(precision)


def test_builder_rejects_missing_or_duplicate_weights(tmp_path):
    arch, weights, _, manifest, _ = _fixture(tmp_path)
    for source in (list(weights.items())[:-1], [*weights.items(), next(iter(weights.items()))]):
        with pytest.raises(ValueError):
            build_cache(
                arch,
                source,
                contract=manifest["schedule"],
                model_variant="fl2va",
                adapter_sha256=None,
                device=torch.device("cpu"),
            )


def test_native_checkpoint_builder_cli(tmp_path, monkeypatch):
    from tools.minimax_h3.build_adaln_cache import main

    arch, weights, _, _, _ = _fixture(tmp_path)
    root = tmp_path / "transformer"
    root.mkdir()
    (root / "config.json").write_text(canonical_json(asdict(arch)))
    save_file(weights, str(root / "model.safetensors"))
    output = tmp_path / "cli.safetensors"
    argv = [
        "build_adaln_cache",
        "--transformer-path",
        str(root),
        "--output",
        str(output),
        "--model-variant",
        "fl2va",
        "--num-inference-steps",
        "4",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    _ready(arch, weights, output)
    with pytest.raises(ValueError, match="overwrite"):
        main()


@pytest.mark.parametrize("base_model", [False, True], ids=["fasth3-four", "base-h3-50"])
def test_runtime_projection_matches_sidecar_using_actual_h3_forwards(tmp_path, monkeypatch, mocker, base_model):
    class Linear(torch.nn.Linear):
        def __init__(self, in_features, out_features, *, params_dtype, bias=True, **kwargs):
            super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)

        def forward(self, x):
            return super().forward(x), None

    monkeypatch.setattr(h3, "ColumnParallelLinear", Linear)
    monkeypatch.setattr(h3, "RowParallelLinear", Linear)
    arch, weights, payload, manifest, _ = _fixture(tmp_path, base_model=base_model)
    embedder = h3.MiniMaxH3TimeEmbedder(arch, prefix="time_embedder")
    runtime = MiniMaxH3RuntimeAdalnCache()
    proj = h3.MiniMaxH3AdalnProj(
        arch,
        18 * arch.hidden_size,
        None,
        expand_ratio=6,
        modality_num=3,
        prefix="blocks.0.adaln_proj",
        adaln_cache=runtime,
    )
    embedder.load_state_dict(
        {
            name.removeprefix("time_embedder."): value
            for name, value in weights.items()
            if name.startswith("time_embedder.")
        }
    )
    proj.load_state_dict(
        {
            name.removeprefix("blocks.0.adaln_proj."): value
            for name, value in weights.items()
            if name.startswith("blocks.0.adaln_proj.")
        }
    )
    calls = mocker.spy(proj.linear, "forward")
    with torch.no_grad():
        for _ in range(2):
            for i, plan in enumerate(timestep_plans(manifest["schedule"])):
                embedding = embedder(plan)
                assert torch.equal(embedding, payload["time_embeddings"][i, : len(plan)])
                runtime.prepare(embedding)
                actual = torch.stack(proj(embedding), dim=1).reshape(len(plan), -1)
                assert torch.equal(actual, payload["block_params"][i, : len(plan), 0])
    plan_count = len(timestep_plans(manifest["schedule"]))
    assert calls.call_count == plan_count and runtime.hits == plan_count


@pytest.mark.parametrize("enforce_eager", [False, True], ids=["compiled", "eager"])
@pytest.mark.parametrize(
    "key,mode",
    [("minimax_h3_adaln_cache_path", "t2va"), ("minimax_h3_ref_adaln_cache_path", "ref2va-mixed")],
)
def test_optional_sidecar_supports_eager_and_compiled_execution(tmp_path, mocker, enforce_eager, key, mode):
    import vllm_omni.diffusion.models.minimax_h3.adaln_cache as cache_module

    arch, _, _, manifest, path = _fixture(tmp_path, mode=mode)
    open_sidecar = mocker.spy(cache_module, "safe_open")
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(cache_config={key: str(path)}, enforce_eager=enforce_eager)
    transformer = SimpleNamespace(arch=arch, adaln_cache=MiniMaxH3RuntimeAdalnCache())

    pipeline._configure_adaln_sidecar(transformer, key, manifest["model_variant"], None, eligible=True)

    open_sidecar.assert_called_once()
    assert isinstance(transformer._adaln_sidecar_candidate, MiniMaxH3AdalnCache)
    assert transformer._adaln_sidecar_candidate.path == str(path)
    assert transformer.adaln_cache.max_bytes > 0


def test_corrupt_optional_sidecar_falls_back_without_rejecting_model(tmp_path):
    path = tmp_path / "truncated.safetensors"
    path.write_bytes(b"truncated")
    pipeline = object.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(cache_config={"cache": str(path)}, enforce_eager=True)
    transformer = SimpleNamespace(arch=h3.MiniMaxH3DiTArchConfig(), adaln_cache=MiniMaxH3RuntimeAdalnCache())
    pipeline._configure_adaln_sidecar(transformer, "cache", "fl2va", None, eligible=True)
    assert not hasattr(transformer, "_adaln_sidecar_candidate")
