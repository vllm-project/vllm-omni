# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""W4A8 quantization for diffusion transformers on ROCm gfx950 (MI355X).

MXFP4 weights (groups of 32 K-elements sharing one ``float8_e8m0fnu`` exponent)
multiplied against MXFP8 activations that are quantized *dynamically* inside the
kernel. Three accuracy tiers, selected by ``svd_rank`` and the checkpoint:

  **plain W4A8 (RTN)**       ``svd_rank`` absent   ``y = Q(x) @ Q(W).T + bias``
  **online W4A8 SVD**        ``svd_rank`` present, stock BF16 checkpoint
  **calibrated W4A8 SVD**    ``svd_rank`` present, serialized checkpoint

  SVD tiers: ``y = Q(x) @ Q(Wr).T + (x @ L1.T) @ L2.T + bias``

where ``Wr = W - L2 @ L1`` is the 4-bit residual and the low-rank up-projection
is fused into the GEMM epilogue, so every tier is one kernel launch.

Checkpoint contract
-------------------
Two load modes, chosen by ``is_checkpoint_w4a8_serialized``:

**Online (default, stock BF16 checkpoint)** -- *plain W4A8 (RTN)* and *online W4A8
SVD*. Both read a stock BF16 checkpoint and do all their work at load time -- no
export step, nothing extra in the state dict. ``plain`` packs each weight to MXFP4
(RTN) as it loads; the SVD tier additionally derives ``proj_down`` (R, K) /
``proj_up`` (N, R) from the weight with ``torch.svd_lowrank`` and quantizes only
the residual, keeping the factors as non-persistent buffers. This online SVD is a
*weight* SVD, not the activation-aware smoothing the paper describes (see
``_low_rank_split``).

**Serialized (calibrated checkpoint)** -- *calibrated W4A8 SVD*. A checkpoint
produced offline by ``examples/quantization/export_quark_svdquant_w4a8.py``
(Quark's ``SVDQuantProcessor``: SmoothQuant smoothing + exact SVD on the smoothed
weight) carries the residual under ``weight`` and the calibrated factors under
``proj_down`` / ``proj_up``. Self-attention QKV is pre-fused in the exporter, so a
fused ``to_qkv`` layer's factors have rank ``3 * svd_rank``.

Implementation: two linear methods -- :class:`QuarkW4A8LinearMethod` (plain) and
:class:`QuarkW4A8SVDLinearMethod` (adds the fused low-rank branch) -- delegate the
residual's storage to a ``_Storage`` strategy, selected by ``quark_export_format``
(a bad value is a ``KeyError`` at lookup, not a silent fallback):

* ``bf16`` (default) -- residual is BF16 on disk (or a stock weight), packed to
  MXFP4 at load. **TP=1 only.**
* ``mxfp4_packed`` -- residual *preshuffled* into the kernel layout on disk
  (``weight_shuffle``/``weight_scale``), fastest load, **TP=1 only** (the shuffle
  bakes in K/N).
* ``mxfp4_unshuffled`` -- residual in *natural* order (``weight_packed``/
  ``weight_scale``), shardable for **TP>1**; each rank shuffles its shard at load
  via ``flydsl_w4a8.shuffle_for_kernel``.

TP: ``_check_parallelism`` allows TP>1 only on the shardable (unshuffled) storage.
Plain and SVD both support column- and row-parallel; the SVD low-rank term needs
no new collective -- its per-rank partial ``d_p @ proj_up.T`` rides the layer's
existing output all-reduce. TP>1 is wired but unvalidated on multi-GPU (dev box has
1 GPU); TP=1 is bit-identical to the packed path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.nn import Module
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped
from vllm.model_executor.model_loader.weight_utils import initialize_single_dummy_weight
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
    _ColumnvLLMParameter,
)

from vllm_omni.quantization import flydsl_w4a8
from vllm_omni.quantization._copy_missing_attrs import (
    copy_missing_attrs as _copy_missing_attrs,
)
from vllm_omni.quantization.mxfp8_config import (
    MXFPLinearMethodBase,
    _LazyWeightMixin,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class DiffusionQuarkW4A8Config(QuantizationConfig):
    """MXFP4-weight / MXFP8-activation config, with or without SVDQuant.

    Args:
        svd_rank: rank of the low-rank correction branch. ``None`` selects the
            plain variant.
        ignored_layers: layer name patterns to leave in BF16.
        is_checkpoint_w4a8_serialized: load a calibrated serialized checkpoint
            (BF16 residual + on-disk ``proj_down``/``proj_up`` factors) instead of
            quantizing a stock BF16 checkpoint at load. See the module docstring.
        quark_export_format: on-disk residual format of a serialized checkpoint.
            ``None``/``"bf16"`` = unpacked BF16, packed at load; ``"mxfp4_packed"``
            = preshuffled MXFP4 (TP=1); ``"mxfp4_unshuffled"`` = natural-order
            MXFP4, shardable for TP>1 (shuffled per shard at load).
    """

    def __init__(
        self,
        svd_rank: int | None = None,
        ignored_layers: list[str] | None = None,
        is_checkpoint_w4a8_serialized: bool = False,
        quark_export_format: str | None = None,
    ) -> None:
        super().__init__()
        if svd_rank is not None and svd_rank <= 0:
            raise ValueError(f"svd_rank must be a positive integer, got {svd_rank!r}")
        self.svd_rank = svd_rank
        self.ignored_layers = ignored_layers or []
        self.is_checkpoint_w4a8_serialized = is_checkpoint_w4a8_serialized
        self.quark_export_format = quark_export_format

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        # Not a member of vllm's QuantizationMethods Literal; vllm-omni registers
        # out-of-tree method names through the quantization factory, exactly as
        # DiffusionMXFP4DualScaleMixedConfig does for "mxfp4_dualscale".
        return "quark_w4a8"  # type: ignore[return-value]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        # The kernel returns bf16 and quantizes its own activations; fp16 in
        # would silently round-trip through bf16.
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Only consulted on CUDA. ROCm gating happens in flydsl_w4a8.supports().
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DiffusionQuarkW4A8Config:
        svd_rank = cls.get_from_keys_or(config, ["svd_rank", "rank"], None)
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(config, ["modules_to_not_convert"], None)
        is_serialized = cls.get_from_keys_or(config, ["is_checkpoint_w4a8_serialized"], False)
        export_format = cls.get_from_keys_or(config, ["quark_export_format"], None)
        return cls(
            svd_rank=svd_rank,
            ignored_layers=ignored_layers,
            is_checkpoint_w4a8_serialized=is_serialized,
            quark_export_format=export_format,
        )

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if not isinstance(layer, LinearBase):
            return None

        if is_layer_skipped(
            prefix=prefix,
            ignored_layers=self.ignored_layers,
            fused_mapping=self.packed_modules_mapping,
        ):
            return UnquantizedLinearMethod()

        usable, reason = flydsl_w4a8.supports()
        if not usable:
            raise NotImplementedError(f"quantization='quark_w4a8' is unavailable: {reason}")

        in_features, out_features = layer.input_size, layer.output_size

        # Residual storage: BF16 (packed at load) online, or the serialized
        # checkpoint's on-disk format. An unknown quark_export_format is a clear
        # KeyError here rather than a silent bf16 fallback.
        storage = (
            _STORAGES[self.quark_export_format or "bf16"] if self.is_checkpoint_w4a8_serialized else _STORAGES["bf16"]
        )

        if self.svd_rank is not None and flydsl_w4a8.supports_svd_shape(in_features, out_features):
            # Derive factors online (stock BF16); load them from a serialized checkpoint.
            return QuarkW4A8SVDLinearMethod(self, storage, derive_factors=not self.is_checkpoint_w4a8_serialized)

        if self.svd_rank is not None and not self.is_checkpoint_w4a8_serialized:
            # Online svdquant on an SVD-rejected shape stays BF16 (the fused epilogue
            # needs both dims >= 256 and a multiple of 256). A *serialized* checkpoint
            # instead carries a plain 4-bit weight here -- the exporter folds the
            # factors back into it -- so it falls through to the plain method below.
            logger.warning(
                "quark_w4a8(svd_rank=%d): %s has shape (out=%d, in=%d); the fused SVD epilogue "
                "requires both >= 256 and a multiple of 256. Keeping this layer in BF16.",
                self.svd_rank,
                prefix,
                out_features,
                in_features,
            )
            return UnquantizedLinearMethod()

        if not flydsl_w4a8.supports_shape(in_features, out_features):
            logger.warning(
                "quark_w4a8: %s has shape (out=%d, in=%d), which the W4A8 kernel cannot "
                "tile; keeping this layer in BF16.",
                prefix,
                out_features,
                in_features,
            )
            return UnquantizedLinearMethod()
        return QuarkW4A8LinearMethod(self, storage)


# ---------------------------------------------------------------------------
# Linear methods
#
# Three orthogonal axes -- residual storage (bf16 / packed / unshuffled MXFP4),
# factor provenance (none / derived / loaded), forward op (plain / svd) --
# expressed as two linear-method classes delegating storage to a ``_Storage``
# strategy, rather than one class per combination. Mirrors upstream vLLM's
# CompressedTensorsLinearMethod + CompressedTensorsScheme split.
# ---------------------------------------------------------------------------


def _init_layer_meta(
    layer: Module,
    input_size_per_partition: int,
    output_partition_sizes: list[int],
    params_dtype: torch.dtype,
) -> None:
    layer.logical_widths = output_partition_sizes
    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = sum(output_partition_sizes)
    layer.orig_dtype = params_dtype
    layer.weight_block_size = None


def _materialize_meta_weight(layer: Module) -> None:
    """Meta -> real device for the lazy bf16 path when no real weight loaded
    (dummy-weight init). A no-op once a checkpoint chunk has materialized it."""
    if layer.weight is not None and layer.weight.device == torch.device("meta"):
        weight = ModelWeightParameter(
            data=torch.empty_like(layer.weight, device=layer._load_device),
            input_dim=1,
            output_dim=0,
            weight_loader=layer.weight.weight_loader,
        )
        _copy_missing_attrs(layer.weight, weight)
        layer.register_parameter("weight", weight)
        initialize_single_dummy_weight(layer.weight)


def _drop_bf16_weight(layer: Module) -> None:
    # A14B holds both experts resident, so free the BF16 copy at load.
    if isinstance(getattr(layer, "weight", None), torch.nn.Parameter):
        delattr(layer, "weight")
        layer.register_parameter("weight", None)


def _swap_param_to_buffer(layer: Module, name: str, tensor: torch.Tensor) -> None:
    if name in layer._parameters:
        del layer._parameters[name]
    layer.register_buffer(name, tensor.contiguous(), persistent=False)


def _low_rank_split(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``W`` into ``L2 @ L1 + residual`` using its top-``rank`` subspace.

    Returns ``(residual, proj_up, proj_down)`` with ``proj_up`` (N, R) and
    ``proj_down`` (R, K), both carrying ``sqrt(s)`` so neither factor has a
    wildly different dynamic range from the other.

    ``svd_lowrank`` (randomized range finding) rather than a full SVD: only the
    leading R directions are wanted, and a full decomposition of a 13824x5120
    weight per layer would dominate model load time.

    Note this is a *plain* weight SVD, not the activation-aware smoothing that
    the SVDQuant paper describes -- that needs calibration data this path does
    not have. It captures the low-rank term of the method and none of the
    outlier migration, so treat its accuracy as a floor, not the published
    result.
    """
    w = weight.float()
    q = min(rank + 8, *w.shape)
    u, s, v = torch.svd_lowrank(w, q=q, niter=4)
    root = s[:rank].sqrt()
    proj_up = u[:, :rank] * root
    proj_down = root.unsqueeze(1) * v[:, :rank].T
    residual = w - proj_up @ proj_down
    return (
        residual.to(weight.dtype),
        proj_up.to(weight.dtype).contiguous(),
        proj_down.to(weight.dtype).contiguous(),
    )


# ---------------------------------------------------------------------------
# Storage strategies: where the 4-bit operand comes from, how it reaches the
# kernel buffers, and whether the on-disk tensors may be sharded for TP.
# ---------------------------------------------------------------------------


class _Storage:
    name: str
    shardable: bool

    def register(self, owner, layer, in_part, out_parts, in_size, out_size, dtype, extra) -> None:
        raise NotImplementedError

    def materialize(self, layer: Module) -> None:
        pass

    def install(self, layer: Module) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Bf16Storage(_Storage):
    """BF16 residual (stock weight or serialized bf16), quantized to MXFP4 at load.

    Lazy (meta device + per-layer pack) when factors are derived or absent;
    non-lazy when a serialized checkpoint also carries proj_down/proj_up, because
    their load order relative to the weight is not guaranteed. Not shardable in
    this build -- the per-rank pack is only validated at TP=1.
    """

    name = "bf16"
    shardable = False

    def register(self, owner, layer, in_part, out_parts, in_size, out_size, dtype, extra) -> None:
        if getattr(owner, "derive_factors", True):
            _LazyWeightMixin.create_weights(owner, layer, in_part, out_parts, in_size, out_size, dtype, **extra)
        else:
            layer.register_parameter(
                "weight",
                ModelWeightParameter(
                    data=torch.empty(sum(out_parts), in_part, dtype=dtype),
                    input_dim=1,
                    output_dim=0,
                    weight_loader=extra.get("weight_loader"),
                ),
            )

    def materialize(self, layer: Module) -> None:
        _materialize_meta_weight(layer)

    def install(self, layer: Module) -> tuple[torch.Tensor, torch.Tensor]:
        packed, scale = flydsl_w4a8.pack_weight(layer.weight.data)
        _drop_bf16_weight(layer)
        return packed, scale


class Mxfp4Storage(_Storage):
    """4-bit residual on disk: (N, K/2) e2m1 pairs + (N, K/32) E8M0 scales.

    ``shuffled=True`` is already in the GEMM's 16x16 tile layout: zero work at
    load, but a logical slice is not a byte slice, so it is unshardable.
    ``False`` is natural (N, K) order that each rank shards then shuffles -- what
    enables TP>1. The two use different checkpoint keys so pointing the wrong
    format at a checkpoint fails as a missing key instead of loading a permuted
    tensor as natural-order and silently emitting garbage.
    """

    def __init__(self, shuffled: bool) -> None:
        self.shuffled = shuffled
        self.name = "mxfp4_packed" if shuffled else "mxfp4_unshuffled"
        self.weight_key = "weight_shuffle" if shuffled else "weight_packed"
        self.shardable = not shuffled

    def register(self, owner, layer, in_part, out_parts, in_size, out_size, dtype, extra) -> None:
        n, k, loader = sum(out_parts), in_part, extra.get("weight_loader")
        # K is a multiple of 256 for any quantized layer (shape gate), so K/32 is
        # a multiple of 8 and the E8M0 scale needs no padding -- shapes are exact.
        if self.shuffled:
            weight: ModelWeightParameter | PackedvLLMParameter = ModelWeightParameter(
                data=torch.empty(n, k // 2, dtype=torch.uint8),
                input_dim=None,
                output_dim=0,
                weight_loader=loader,
            )
            scale: ModelWeightParameter | GroupQuantScaleParameter = ModelWeightParameter(
                data=torch.empty(n, k // 32, dtype=torch.uint8),
                input_dim=None,
                output_dim=0,
                weight_loader=loader,
            )
        else:
            weight = PackedvLLMParameter(
                data=torch.empty(n, k // 2, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=2,
                weight_loader=loader,
            )
            scale = GroupQuantScaleParameter(
                data=torch.empty(n, k // 32, dtype=torch.uint8),
                input_dim=1,
                output_dim=0,
                weight_loader=loader,
            )
        layer.register_parameter(self.weight_key, weight)
        layer.register_parameter("weight_scale", scale)

    def install(self, layer: Module) -> tuple[torch.Tensor, torch.Tensor]:
        weight = getattr(layer, self.weight_key).data
        scale = layer.weight_scale.data
        for name in (self.weight_key, "weight_scale"):
            layer._parameters.pop(name, None)
        if self.shuffled:
            # The GEMM op consumes the uint8 views directly.
            return weight, scale
        # Shuffle this rank's local shard into the kernel layout (TP=1 -> identical
        # bytes to packing the whole weight; verified equal on gfx950).
        return flydsl_w4a8.shuffle_for_kernel(weight, scale)


_STORAGES: dict[str, _Storage] = {
    "bf16": Bf16Storage(),
    "mxfp4_packed": Mxfp4Storage(shuffled=True),
    "mxfp4_unshuffled": Mxfp4Storage(shuffled=False),
}


# ---------------------------------------------------------------------------
# Linear methods: lifecycle skeleton; storage + factor hooks fill in the rest.
# ---------------------------------------------------------------------------


class QuarkW4A8LinearMethod(MXFPLinearMethodBase):
    """Plain W4A8: MXFP4 weight, dynamically quantized MXFP8 activation.

    After ``process_weights_after_loading`` the layer holds ``_kernel_weight`` /
    ``_kernel_scale`` (non-persistent uint8 buffers in the FlyDSL GEMM layout);
    where those bytes come from is the ``storage``'s business.
    ``_quantize_activation`` passes through -- activation quantization is inside
    the custom op, so torch.compile sees one opaque node.
    """

    row_parallel_ok = True

    def __init__(self, quant_config: DiffusionQuarkW4A8Config, storage: _Storage) -> None:
        self.quant_config = quant_config
        self.storage = storage
        self.out_dtype = torch.get_default_dtype()
        flydsl_w4a8.register_ops()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        self._check_parallelism(layer, input_size_per_partition, input_size)
        _init_layer_meta(layer, input_size_per_partition, output_partition_sizes, params_dtype)
        self.storage.register(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            extra_weight_attrs,
        )
        self._register_factors(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            params_dtype,
            extra_weight_attrs.get("weight_loader"),
        )

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return
        self.storage.materialize(layer)
        # Before install: may rewrite layer.weight (online SVD) or swap loaded
        # factor params to buffers.
        self._prepare_factors(layer)
        kernel_weight, kernel_scale = self.storage.install(layer)
        layer.register_buffer("_kernel_weight", kernel_weight, persistent=False)
        layer.register_buffer("_kernel_scale", kernel_scale, persistent=False)
        layer._already_called_process_weights_after_loading = True

    def _register_factors(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        params_dtype: torch.dtype,
        weight_loader,
    ) -> None:
        pass

    def _prepare_factors(self, layer: Module) -> None:
        pass

    def _check_parallelism(self, layer: Module, input_size_per_partition: int, input_size: int) -> None:
        tp_size = getattr(layer, "tp_size", 1)
        if tp_size > 1 and not self.storage.shardable:
            raise NotImplementedError(
                f"quark_w4a8: the {self.storage.name} residual format cannot be sharded; got tp_size={tp_size}."
            )
        # Row-parallel shards the input dim, so input_size_per_partition < input_size.
        if tp_size > 1 and input_size_per_partition != input_size and not self.row_parallel_ok:
            raise NotImplementedError(
                "quark_w4a8 (svd): row-parallel TP needs an all-reduce of the low-rank term "
                f"(not yet implemented). Got input sharded to {input_size_per_partition} of {input_size}."
            )

    def _quantize_activation(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        # Activation quantization to MXFP8 is inside the custom op called by
        # _quant_matmul, so pass the raw activation through here.
        return x, None

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor | None,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.ops.vllm_omni.flydsl_w4a8_gemm(
            x_q,
            layer._kernel_weight,
            layer._kernel_scale,
            bias,
            layer.output_size_per_partition,
        )
        if output.dtype != ori_dtype:
            output = output.to(ori_dtype)
        return output


class QuarkW4A8SVDLinearMethod(QuarkW4A8LinearMethod):
    """W4A8 plus a rank-R low-rank correction fused into the GEMM epilogue.

    The 4-bit operand is the residual ``Wr = W - L2 @ L1``; ``proj_down`` (R, K)
    and ``proj_up`` (N, R) stay BF16. ``derive_factors`` picks the provenance:
    online (``svd_lowrank`` at load) vs loaded from a serialized checkpoint.

    Column- and row-parallel both work on a shardable storage: each factor is
    registered on the layer axis it shares (``proj_up`` on N, ``proj_down`` on K),
    so vLLM shards it for the matching parallelism and replicates it otherwise.
    Row-parallel needs no new collective -- by linearity the per-rank partial
    ``d_p @ proj_up.T`` rides the layer's existing output all-reduce.
    """

    row_parallel_ok = True

    def __init__(self, quant_config: DiffusionQuarkW4A8Config, storage: _Storage, derive_factors: bool) -> None:
        super().__init__(quant_config, storage)
        self.derive_factors = derive_factors

    def _register_factors(
        self,
        layer: Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        params_dtype: torch.dtype,
        weight_loader,
    ) -> None:
        if self.derive_factors:
            return  # derived in _prepare_factors and registered as buffers
        # rank_eff is 3*svd_rank on a fused to_qkv (three output shards) and
        # svd_rank on a single-shard linear -- matching the exporter's fusion.
        rank = self.quant_config.svd_rank
        assert rank is not None
        rank_eff = rank * len(output_partition_sizes)
        # Each factor shares exactly one of the layer's axes, so register it on
        # that axis with a single-axis parameter class: vLLM shards it for the
        # matching parallelism and replicates it for the other. No row/column
        # branch, and row-parallel needs no new collective.
        #   proj_up   (N, rank_eff) -> output axis N (column shards, row replicates)
        #   proj_down (rank_eff, K) -> input axis K  (row shards, column replicates)
        layer.register_parameter(
            "proj_up",
            _ColumnvLLMParameter(
                data=torch.empty(sum(output_partition_sizes), rank_eff, dtype=params_dtype),
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "proj_down",
            RowvLLMParameter(
                data=torch.empty(rank_eff, input_size_per_partition, dtype=params_dtype),
                input_dim=1,
                weight_loader=weight_loader,
            ),
        )

    def _prepare_factors(self, layer: Module) -> None:
        if self.derive_factors:
            assert self.quant_config.svd_rank is not None
            residual, proj_up, proj_down = _low_rank_split(layer.weight.data, self.quant_config.svd_rank)
            layer.weight.data = residual
            layer.register_buffer("proj_down", proj_down, persistent=False)
            layer.register_buffer("proj_up", proj_up, persistent=False)
        else:
            _swap_param_to_buffer(layer, "proj_down", layer.proj_down.data)
            _swap_param_to_buffer(layer, "proj_up", layer.proj_up.data)

    def _quant_matmul(
        self,
        x_q: torch.Tensor,
        x_scale: torch.Tensor | None,
        layer: torch.nn.Module,
        bias: torch.Tensor | None,
        ori_dtype: torch.dtype,
    ) -> torch.Tensor:
        output = torch.ops.vllm_omni.flydsl_w4a8_svd_gemm(
            x_q,
            layer._kernel_weight,
            layer._kernel_scale,
            layer.proj_down,
            layer.proj_up,
            bias,
            layer.output_size_per_partition,
        )
        if output.dtype != ori_dtype:
            output = output.to(ori_dtype)
        return output
