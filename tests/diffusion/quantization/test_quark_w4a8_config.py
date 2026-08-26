# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the W4A8 (MXFP4 weight / MXFP8 activation) quantization config.

Numerics live in test_flydsl_w4a8_rocm.py, which needs a gfx950 device. Here we
cover config parsing, factory registration, layer routing and the capability
gate — all of which must behave sanely on a machine with no AMD GPU at all.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _patch_tp_state(monkeypatch):
    """Patch TP rank/world_size so ModelWeightParameter can be instantiated on CPU
    without an initialized distributed group.  Returns TP=1 rank=0 for all tests."""
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture(autouse=True)
def _clear_backend_caches():
    """supports() and _provider() are lru_cached; a stale entry would leak the
    real machine's capability answer into tests that patch it."""
    from vllm_omni.quantization import flydsl_w4a8

    flydsl_w4a8.supports.cache_clear()
    flydsl_w4a8._provider.cache_clear()
    yield
    flydsl_w4a8.supports.cache_clear()
    flydsl_w4a8._provider.cache_clear()


def _fake_linear(in_features: int, out_features: int):
    """A LinearBase instance without running __init__.

    LinearBase.__init__ calls get_quant_method() itself, so it cannot be used to
    build the input to a get_quant_method() test. Only input_size/output_size are
    read during routing.
    """
    from vllm.model_executor.layers.linear import ReplicatedLinear

    layer = object.__new__(ReplicatedLinear)
    layer.input_size = in_features
    layer.output_size = out_features
    return layer


@pytest.fixture
def _supported(monkeypatch):
    """Pretend the machine is a gfx950 with a working kernel provider."""
    from vllm_omni.quantization import flydsl_w4a8

    monkeypatch.setattr(flydsl_w4a8, "supports", lambda: (True, ""))


# ---------------------------------------------------------------------------
# DiffusionQuarkW4A8Config parsing
# ---------------------------------------------------------------------------


def test_get_name():
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    assert DiffusionQuarkW4A8Config.get_name() == "quark_w4a8"


def test_from_config_defaults():
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    cfg = DiffusionQuarkW4A8Config.from_config({})
    assert cfg.svd_rank is None
    assert cfg.ignored_layers == []
    assert cfg.is_checkpoint_w4a8_serialized is False


@pytest.mark.parametrize("key", ["svd_rank", "rank"])
def test_from_config_svd_rank_keys(key):
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    assert DiffusionQuarkW4A8Config.from_config({key: 16}).svd_rank == 16


@pytest.mark.parametrize("key", ["ignored_layers", "modules_to_not_convert"])
def test_from_config_ignored_layer_keys(key):
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    assert DiffusionQuarkW4A8Config.from_config({key: ["proj_out"]}).ignored_layers == ["proj_out"]


def test_from_config_ignores_unknown_keys():
    """Checkpoint config.json carries keys this scheme does not consume."""
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    cfg = DiffusionQuarkW4A8Config.from_config({"quant_algo": "MXFP4", "producer": {"name": "quark"}})
    assert cfg.svd_rank is None


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_svd_rank_rejected(bad):
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    with pytest.raises(ValueError, match="positive integer"):
        DiffusionQuarkW4A8Config(svd_rank=bad)


def test_act_dtypes_bf16_only():
    """The kernel emits bf16 and quantizes its own activations; accepting fp16
    would silently round-trip through bf16."""
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    assert DiffusionQuarkW4A8Config.get_supported_act_dtypes() == [torch.bfloat16]


def test_apply_vllm_mapper_rewrites_ignored_layers():
    from vllm.model_executor.models.utils import WeightsMapper

    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    cfg = DiffusionQuarkW4A8Config(ignored_layers=["blocks.0.proj_out"])
    cfg.apply_vllm_mapper(WeightsMapper(orig_to_new_prefix={"blocks.": "layers."}))
    assert cfg.ignored_layers == ["layers.0.proj_out"]


# ---------------------------------------------------------------------------
# Factory registration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "expected_rank"),
    [("quark_w4a8", None), ("quark-w4a8", None), ("quark_svdquant", 32)],
)
def test_build_quant_config(method, expected_rank):
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    cfg = build_quant_config(method)
    assert isinstance(cfg, DiffusionQuarkW4A8Config)
    assert cfg.svd_rank == expected_rank


def test_quark_svdquant_respects_explicit_rank():
    from vllm_omni.quantization import build_quant_config

    cfg = build_quant_config({"method": "quark_svdquant", "svd_rank": 16})
    assert cfg.svd_rank == 16


def test_methods_registered():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "quark_w4a8" in SUPPORTED_QUANTIZATION_METHODS
    assert "quark_svdquant" in SUPPORTED_QUANTIZATION_METHODS


def test_bare_svdquant_name_not_claimed():
    """Upstream vLLM PR #3830 uses "svdquant" for Nunchaku; taking it here would
    shadow that method once it lands."""
    from vllm_omni.quantization.factory import _OVERRIDES

    assert "svdquant" not in _OVERRIDES


def test_importing_quantization_package_does_not_import_quark():
    """Quark is an optional, heavy dependency. It must only load on gfx950 when a
    W4A8 layer is actually constructed."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import sys, vllm_omni.quantization; print('quark' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip().endswith("False"), result.stdout


# ---------------------------------------------------------------------------
# Capability gate
# ---------------------------------------------------------------------------


def test_supports_returns_reason_off_rocm(monkeypatch):
    from vllm_omni.quantization import flydsl_w4a8

    monkeypatch.setattr(flydsl_w4a8, "current_omni_platform", None, raising=False)
    monkeypatch.setattr("vllm_omni.platforms.current_omni_platform.is_rocm", lambda: False)

    usable, reason = flydsl_w4a8.supports()
    assert usable is False
    assert "ROCm" in reason


def test_unknown_provider_is_a_config_error(monkeypatch):
    """A typo in VLLM_OMNI_SVDQUANT_PROVIDER is misconfiguration, not a missing
    capability, so it must raise rather than silently disable the backend."""
    from vllm_omni.quantization import flydsl_w4a8

    monkeypatch.setenv("VLLM_OMNI_SVDQUANT_PROVIDER", "nope")
    with pytest.raises(ValueError, match="VLLM_OMNI_SVDQUANT_PROVIDER"):
        flydsl_w4a8._provider()


def test_flydsl_provider_not_available_yet(monkeypatch):
    from vllm_omni.quantization import flydsl_w4a8

    monkeypatch.setenv("VLLM_OMNI_SVDQUANT_PROVIDER", "flydsl")
    with pytest.raises(ImportError, match="does not package its kernels"):
        flydsl_w4a8._provider()


@pytest.mark.parametrize(
    ("in_f", "out_f", "plain", "svd"),
    [
        (5120, 5120, True, True),
        (3072, 3072, True, True),
        (3072, 192, True, False),  # Wan proj_out: tileable at 32, refused by SVD
        (5120, 13824, True, True),  # A14B ffn up: 13824 == 54 * 256
        (3072, 100, False, False),
        (100, 3072, False, False),
        # K carries the packed-scale constraint, not just a tiling one: Quark's
        # _validate_a8w4_inputs raises below 256, and _pack_weight_asm quietly
        # emits an unshuffled layout the preshuffle kernel cannot read.
        (128, 3072, False, False),
        (1152, 3072, False, False),  # 1152 % 32 == 0 but 1152 % 256 == 128
        (256, 3072, True, True),
    ],
)
def test_shape_gates(in_f, out_f, plain, svd):
    from vllm_omni.quantization import flydsl_w4a8

    assert flydsl_w4a8.supports_shape(in_f, out_f) is plain
    assert flydsl_w4a8.supports_svd_shape(in_f, out_f) is svd


# ---------------------------------------------------------------------------
# get_quant_method routing
# ---------------------------------------------------------------------------


def test_non_linear_layer_returns_none():
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    assert DiffusionQuarkW4A8Config().get_quant_method(torch.nn.LayerNorm(8), "norm") is None


def test_plain_routes_to_bf16_storage(_supported):
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8LinearMethod,
    )

    method = DiffusionQuarkW4A8Config().get_quant_method(_fake_linear(5120, 5120), "to_q")
    assert type(method) is QuarkW4A8LinearMethod
    assert method.storage.name == "bf16"


def test_svd_routes_to_svd_method(_supported):
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    method = DiffusionQuarkW4A8Config(svd_rank=32).get_quant_method(_fake_linear(5120, 5120), "to_q")
    assert isinstance(method, QuarkW4A8SVDLinearMethod)
    assert method.storage.name == "bf16"
    assert method.derive_factors is True


def test_ignored_layer_routes_to_bf16(_supported):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    cfg = DiffusionQuarkW4A8Config(ignored_layers=["blocks.0.to_q"])
    assert isinstance(cfg.get_quant_method(_fake_linear(5120, 5120), "blocks.0.to_q"), UnquantizedLinearMethod)


def test_svd_rejects_wan_proj_out_shape_loudly(_supported, caplog):
    """out_features=192 is below the SVD epilogue's 256 floor. Quark produces
    garbage rather than failing, so routing must fall back to BF16 and say so."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    cfg = DiffusionQuarkW4A8Config(svd_rank=32)
    with caplog.at_level("WARNING"):
        method = cfg.get_quant_method(_fake_linear(3072, 192), "proj_out")
    assert isinstance(method, UnquantizedLinearMethod)
    assert "proj_out" in caplog.text


def test_untileable_shape_falls_back_to_bf16(_supported):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    method = DiffusionQuarkW4A8Config().get_quant_method(_fake_linear(3072, 100), "odd")
    assert isinstance(method, UnquantizedLinearMethod)


def test_serialized_svd_routes_to_svd_bf16_loaded(_supported):
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(svd_rank=32, is_checkpoint_w4a8_serialized=True)
    method = cfg.get_quant_method(_fake_linear(5120, 5120), "to_q")
    assert isinstance(method, QuarkW4A8SVDLinearMethod)
    assert method.storage.name == "bf16"
    assert method.derive_factors is False


def test_serialized_plain_routes_to_bf16(_supported):
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8LinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(is_checkpoint_w4a8_serialized=True)
    method = cfg.get_quant_method(_fake_linear(5120, 5120), "to_q")
    assert type(method) is QuarkW4A8LinearMethod
    assert method.storage.name == "bf16"


def test_serialized_svd_shape_not_svd_falls_back_to_plain(_supported):
    """A serialized SVD checkpoint folds the correction back into any layer the
    SVD gate rejects, so such a layer must route to the plain method (not BF16 --
    the checkpoint carries a plain 4-bit weight there)."""
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8LinearMethod,
        QuarkW4A8SVDLinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(svd_rank=32, is_checkpoint_w4a8_serialized=True)
    # 1152 % 32 == 0 (plain-tileable) but 1152 % 256 != 0 (not SVD-tileable).
    method = cfg.get_quant_method(_fake_linear(3072, 1152), "ffn")
    assert type(method) is QuarkW4A8LinearMethod
    assert not isinstance(method, QuarkW4A8SVDLinearMethod)
    assert method.storage.name == "bf16"


def test_online_svd_shape_not_svd_falls_back_to_bf16(_supported):
    """Online svdquant (no serialized checkpoint) on an SVD-rejected shape has no
    folded plain weight, so it must go to BF16 -- unlike the serialized case."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    method = DiffusionQuarkW4A8Config(svd_rank=32).get_quant_method(_fake_linear(3072, 1152), "ffn")
    assert isinstance(method, UnquantizedLinearMethod)


def test_serialized_untileable_falls_back_to_bf16(_supported):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    cfg = DiffusionQuarkW4A8Config(svd_rank=32, is_checkpoint_w4a8_serialized=True)
    method = cfg.get_quant_method(_fake_linear(3072, 100), "odd")
    assert isinstance(method, UnquantizedLinearMethod)


@pytest.mark.parametrize(
    ("output_partition_sizes", "expected_rank"),
    [([1024, 1024, 1024], 96), ([5120], 32)],
)
def test_svd_serialized_create_weights_rank_from_shards(_supported, output_partition_sizes, expected_rank):
    """rank_eff = svd_rank * len(output_partition_sizes): 3*R for a fused to_qkv
    (three shards), R for a single-shard linear."""
    from vllm_omni.quantization.quark_w4a8_config import (
        _STORAGES,
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(svd_rank=32, is_checkpoint_w4a8_serialized=True)
    method = QuarkW4A8SVDLinearMethod(cfg, _STORAGES["bf16"], derive_factors=False)
    layer = torch.nn.Module()
    out = sum(output_partition_sizes)
    method.create_weights(
        layer,
        input_size_per_partition=3072,
        output_partition_sizes=output_partition_sizes,
        input_size=3072,
        output_size=out,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    assert layer.weight.shape == (out, 3072)
    assert layer.proj_down.shape == (expected_rank, 3072)
    assert layer.proj_up.shape == (out, expected_rank)


# ---------------------------------------------------------------------------
# Pre-packed serialized checkpoints (mxfp4_packed)
# ---------------------------------------------------------------------------


def test_from_config_parses_export_format():
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    cfg = DiffusionQuarkW4A8Config.from_config({"quark_export_format": "mxfp4_packed"})
    assert cfg.quark_export_format == "mxfp4_packed"
    assert DiffusionQuarkW4A8Config.from_config({}).quark_export_format is None


def test_serialized_packed_svd_routes(_supported):
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(svd_rank=32, is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_packed")
    method = cfg.get_quant_method(_fake_linear(5120, 5120), "to_q")
    assert isinstance(method, QuarkW4A8SVDLinearMethod)
    assert method.storage.name == "mxfp4_packed"


def test_serialized_packed_plain_routes(_supported):
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8LinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_packed")
    method = cfg.get_quant_method(_fake_linear(5120, 5120), "to_q")
    assert type(method) is QuarkW4A8LinearMethod
    assert method.storage.name == "mxfp4_packed"


def test_packed_create_weights_registers_uint8_shapes(_supported):
    """weight_shuffle is (N, K/2) and weight_scale (N, K/32), both uint8; K%256==0
    guarantees the scale needs no padding."""
    from vllm_omni.quantization.quark_w4a8_config import (
        _STORAGES,
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(svd_rank=32, is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_packed")
    method = QuarkW4A8SVDLinearMethod(cfg, _STORAGES["mxfp4_packed"], derive_factors=False)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=3072,
        output_partition_sizes=[1024, 1024, 1024],
        input_size=3072,
        output_size=3072,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    assert layer.weight_shuffle.shape == (3072, 1536) and layer.weight_shuffle.dtype == torch.uint8
    assert layer.weight_scale.shape == (3072, 96) and layer.weight_scale.dtype == torch.uint8
    assert layer.proj_down.shape == (96, 3072)  # 3 * svd_rank on the fused to_qkv
    assert layer.proj_up.shape == (3072, 96)


# ---------------------------------------------------------------------------
# Unshuffled serialized checkpoints (mxfp4_unshuffled, TP>1)
# ---------------------------------------------------------------------------


def test_serialized_unshuffled_plain_routes(_supported):
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8LinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_unshuffled")
    method = cfg.get_quant_method(_fake_linear(5120, 5120), "to_q")
    assert type(method) is QuarkW4A8LinearMethod
    assert method.storage.name == "mxfp4_unshuffled"


def test_serialized_unshuffled_svd_routes(_supported):
    from vllm_omni.quantization.quark_w4a8_config import (
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(
        svd_rank=32, is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_unshuffled"
    )
    method = cfg.get_quant_method(_fake_linear(5120, 5120), "to_q")
    assert isinstance(method, QuarkW4A8SVDLinearMethod)
    assert method.storage.name == "mxfp4_unshuffled"


def test_unshuffled_plain_create_weights_shardable_params(_supported):
    """weight_packed is a PackedvLLMParameter (shards K via packed_dim) and
    weight_scale a GroupQuantScaleParameter — the classes that let vLLM shard for TP."""
    from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter

    from vllm_omni.quantization.quark_w4a8_config import (
        _STORAGES,
        DiffusionQuarkW4A8Config,
        QuarkW4A8LinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_unshuffled")
    method = QuarkW4A8LinearMethod(cfg, _STORAGES["mxfp4_unshuffled"])
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=3072,
        output_partition_sizes=[5120],
        input_size=3072,
        output_size=5120,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    assert isinstance(layer.weight_packed, PackedvLLMParameter)
    assert isinstance(layer.weight_scale, GroupQuantScaleParameter)
    assert layer.weight_packed.shape == (5120, 1536) and layer.weight_packed.dtype == torch.uint8
    assert layer.weight_scale.shape == (5120, 96) and layer.weight_scale.dtype == torch.uint8


def test_unshuffled_svd_factor_param_axes(_supported):
    """proj_up on the output axis (_ColumnvLLMParameter, shards N column-parallel);
    proj_down on the input axis (RowvLLMParameter, shards K row-parallel). Each
    replicates on the other parallelism, so both directions work with no branch."""
    from vllm.model_executor.parameter import RowvLLMParameter, _ColumnvLLMParameter

    from vllm_omni.quantization.quark_w4a8_config import (
        _STORAGES,
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(
        svd_rank=32, is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_unshuffled"
    )
    method = QuarkW4A8SVDLinearMethod(cfg, _STORAGES["mxfp4_unshuffled"], derive_factors=False)
    layer = torch.nn.Module()
    method.create_weights(
        layer,
        input_size_per_partition=3072,
        output_partition_sizes=[1024, 1024, 1024],
        input_size=3072,
        output_size=3072,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    assert layer.proj_up.shape == (3072, 96) and isinstance(layer.proj_up, _ColumnvLLMParameter)
    assert layer.proj_down.shape == (96, 3072) and isinstance(layer.proj_down, RowvLLMParameter)


def test_unshuffled_svd_accepts_row_parallel_tp(_supported):
    """Row-parallel SVD is supported on the unshuffled storage: proj_down shards
    on K (input dim), proj_up replicates, and the correction rides the layer's
    output all-reduce -- so create_weights must not raise."""
    from vllm_omni.quantization.quark_w4a8_config import (
        _STORAGES,
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(
        svd_rank=32, is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_unshuffled"
    )
    method = QuarkW4A8SVDLinearMethod(cfg, _STORAGES["mxfp4_unshuffled"], derive_factors=False)
    layer = torch.nn.Module()
    layer.tp_size = 2
    method.create_weights(
        layer,
        input_size_per_partition=1536,  # < input_size -> input (K) sharded = row-parallel
        output_partition_sizes=[5120],
        input_size=3072,
        output_size=5120,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *a, **k: None,
    )
    assert layer.proj_down.shape == (32, 1536)  # sharded on K
    assert layer.proj_up.shape == (5120, 32)  # replicated


def test_bf16_svd_still_rejects_tp(_supported):
    """A non-shardable storage (bf16/packed) still refuses TP>1 in either direction."""
    from vllm_omni.quantization.quark_w4a8_config import (
        _STORAGES,
        DiffusionQuarkW4A8Config,
        QuarkW4A8SVDLinearMethod,
    )

    cfg = DiffusionQuarkW4A8Config(svd_rank=32, is_checkpoint_w4a8_serialized=True)
    method = QuarkW4A8SVDLinearMethod(cfg, _STORAGES["bf16"], derive_factors=False)
    layer = torch.nn.Module()
    layer.tp_size = 2
    with pytest.raises(NotImplementedError, match="cannot be sharded"):
        method.create_weights(
            layer,
            input_size_per_partition=5120,
            output_partition_sizes=[5120],
            input_size=5120,
            output_size=5120,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *a, **k: None,
        )


def test_unsupported_hardware_raises(monkeypatch):
    """Silently falling back to BF16 for the whole model would look like a
    successful quantized run, so an unusable backend must be an error."""
    from vllm_omni.quantization import flydsl_w4a8
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    monkeypatch.setattr(flydsl_w4a8, "supports", lambda: (False, "requires gfx950, detected gfx942"))
    with pytest.raises(NotImplementedError, match="gfx942"):
        DiffusionQuarkW4A8Config().get_quant_method(_fake_linear(5120, 5120), "to_q")
