# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the SPARSE_ATTN dispatcher backend.

The dispatcher is hardware-agnostic and is exercised directly (no GPU platform
needed). CPU-runnable tests carry the file-level markers below; the on-CUDA
variant is gated with skipif and runs only where a GPU is present.

Covers:
- SparseAttentionBackend enum slot + inherited capability defaults
- plugin resolution via entry points + param merge + forward dispatch
- per-role plugin shorthand routing + flat/nested config parsing
- on-CUDA dispatch (skipif no GPU)
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionImpl,
    AttentionMetadata,
    PerForwardState,
)
from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.attention.backends.sparse_attn import (
    SparseAttentionBackend,
    _apply_default_geometry,
    _row0,
    _SparseAttentionImpl,
    merge_sparse_params,
    resolve_sparse_attn_plugin,
)
from vllm_omni.diffusion.attention.selector import _route_sparse_plugin_shorthand
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _fake_ep(name, fn):
    ep = MagicMock()
    ep.name = name
    ep.value = f"test_pkg:{name}"
    ep.load.return_value = fn
    return ep


def _recording_plugin():
    captured = {}

    def fn(q, k, v, params, is_causal, softmax_scale):
        captured["params"] = params
        captured["is_causal"] = is_causal
        captured["softmax_scale"] = softmax_scale
        return q  # identity -> lets us assert the dispatcher returned its output

    return fn, captured


def _capable_plugin(capabilities):
    fn, captured = _recording_plugin()
    fn.capabilities = capabilities
    return fn, captured


class TestSparseAttentionBackend:
    """In-tree slot inheriting the conservative ABC defaults."""

    def test_registered_in_enum(self):
        assert DiffusionAttentionBackendEnum.SPARSE_ATTN.get_class() is SparseAttentionBackend

    def test_capability_defaults(self):
        assert SparseAttentionBackend.get_name() == "SPARSE_ATTN"
        assert SparseAttentionBackend.get_impl_cls() is _SparseAttentionImpl
        # [] == "any" head size: the external kernel owns its own support.
        assert SparseAttentionBackend.get_supported_head_sizes() == []
        assert SparseAttentionBackend.supports_head_size(123) is True
        # Conservative inherited defaults match the maskless/no-buffer contract.
        assert SparseAttentionBackend.accept_output_buffer is False
        assert SparseAttentionBackend.supports_attention_mask() is False


class TestResolveSparseAttnPlugin:
    """Entry-point resolution; never silently degrade to dense."""

    def test_requires_name(self):
        with pytest.raises(ValueError, match="requires a kernel name"):
            resolve_sparse_attn_plugin(None)

    def test_unknown_name_raises(self):
        with patch("importlib.metadata.entry_points", return_value=[]):
            with pytest.raises(ValueError, match="No sparse attention plugin"):
                resolve_sparse_attn_plugin("nope")

    def test_found_case_insensitive(self):
        fn, _ = _recording_plugin()
        with patch("importlib.metadata.entry_points", return_value=[_fake_ep("Dummy_Sparse", fn)]):
            assert resolve_sparse_attn_plugin("dummy_sparse") is fn

    def test_load_failure_reraised_with_context(self):
        ep = MagicMock()
        ep.name = "broken"
        ep.value = "broken_pkg:fn"
        ep.load.side_effect = ImportError("boom")
        with patch("importlib.metadata.entry_points", return_value=[ep]):
            with pytest.raises(ValueError, match="failed to import"):
                resolve_sparse_attn_plugin("broken")


class TestMergeSparseParams:
    """Policy 1: static < per-forward < dynamic; None dropped."""

    def test_precedence_and_none_drop(self):
        out = merge_sparse_params(
            {"topk": 0.3, "block": None},  # static (a None to drop)
            {"denoise_step_idx": 5},  # per-forward
            {"topk": 0.7},  # dynamic overrides static
        )
        assert out == {"topk": 0.7, "denoise_step_idx": 5}

    def test_empty_tiers(self):
        assert merge_sparse_params(None, None, None) == {}


class TestSparseDispatch:
    """End-to-end dispatch on CPU via a mocked function plugin."""

    def _make_impl(self, fn, *, params=None):
        cfg = {"backend": "dummy_sparse", "params": params or {}}
        with patch("importlib.metadata.entry_points", return_value=[_fake_ep("dummy_sparse", fn)]):
            return _SparseAttentionImpl(
                num_heads=4,
                head_size=16,
                softmax_scale=0.125,  # distinctive, != head_size**-0.5, to prove it's forwarded
                causal=False,
                backend_kwargs=cfg,
            )

    def test_forward_merges_params_and_dispatches(self):
        fn, captured = _recording_plugin()
        impl = self._make_impl(fn, params={"topk": 0.5})
        q = torch.randn(2, 8, 4, 16)
        md = AttentionMetadata(
            per_forward=PerForwardState(denoise_step_idx=torch.tensor([3, 3])),
            extra={"block_mask": "M"},
        )
        out = impl.forward(q, q, q, md)

        assert out is q  # identity plugin -> dispatcher returned the kernel output
        params = captured["params"]
        assert params["topk"] == 0.5  # static config
        assert params["denoise_step_idx"].tolist() == [3, 3]  # per-forward (canonical)
        assert params["block_mask"] == "M"  # dynamic opaque
        assert captured["is_causal"] is False
        assert captured["softmax_scale"] == 0.125  # configured scale forwarded, not recomputed

    def test_forward_without_metadata(self):
        fn, captured = _recording_plugin()
        impl = self._make_impl(fn, params={"topk": 0.3})
        q = torch.randn(1, 4, 4, 16)
        impl.forward(q, q, q, None)
        assert captured["params"] == {"topk": 0.3}


class TestDefaultGeometryResolver:
    """The dispatcher derives frame/patch counts from raw geometry primitives,
    so a plugin gets them for free with NO per-model edits."""

    def _impl(self, fn, *, params=None):
        cfg = {"backend": "dummy_sparse", "params": params or {}}
        with patch("importlib.metadata.entry_points", return_value=[_fake_ep("dummy_sparse", fn)]):
            return _SparseAttentionImpl(num_heads=4, head_size=16, softmax_scale=1.0, backend_kwargs=cfg)

    @staticmethod
    def _md_with_geometry():
        # latent_shape (T,H,W)=(4,4,4), patch_size (1,2,2) -> 4 frames, 4 patches
        # per frame -> seq_len 16 (the resolver checks the product against the
        # actual query sequence length).
        return AttentionMetadata(
            per_forward=PerForwardState(
                latent_shape=torch.tensor([[4, 4, 4], [4, 4, 4]]),
                patch_size=torch.tensor([[1, 2, 2], [1, 2, 2]]),
            )
        )

    def test_row0_reads_first_sample(self):
        assert _row0(torch.tensor([[21, 60, 104], [21, 60, 104]])) == (21, 60, 104)
        assert _row0((1, 2, 2)) == (1, 2, 2)  # tolerate plain tuples
        assert _row0([[1, 2, 2]]) == (1, 2, 2)  # and list-of-rows

    def test_apply_default_geometry_pure(self):
        params = {"latent_shape": (21, 60, 104), "patch_size": (1, 2, 2)}
        _apply_default_geometry(params)
        assert params["total_latent_frames"] == 21
        assert params["patches_per_frame"] == 1560

    def test_apply_default_geometry_noop_without_primitives(self):
        params = {"topk": 0.3}
        _apply_default_geometry(params)
        assert "total_latent_frames" not in params

    def test_apply_default_geometry_seq_len_match_fills(self):
        params = {"latent_shape": (21, 60, 104), "patch_size": (1, 2, 2)}
        _apply_default_geometry(params, seq_len=21 * 1560)
        assert params["total_latent_frames"] == 21

    def test_apply_default_geometry_seq_len_mismatch_drops_derived(self):
        # SP auto-padding / non-video tokens make the derived (frame, patch)
        # mapping wrong for the sequence the kernel sees -> derived fields are
        # dropped, raw primitives remain for a plugin geometry_fn.
        params = {"latent_shape": (21, 60, 104), "patch_size": (1, 2, 2)}
        _apply_default_geometry(params, seq_len=21 * 1560 + 8)
        assert "total_latent_frames" not in params
        assert "patches_per_frame" not in params
        assert params["latent_shape"] == (21, 60, 104)

    def test_default_resolver_fills_derived_geometry(self):
        fn, captured = _recording_plugin()
        q = torch.randn(2, 16, 4, 16)  # seq_len 16 == 4 frames * 4 patches
        self._impl(fn).forward(q, q, q, self._md_with_geometry())
        params = captured["params"]
        assert _row0(params["latent_shape"]) == (4, 4, 4)  # raw primitives forwarded
        assert params["total_latent_frames"] == 4  # ... and derived counts resolved
        assert params["patches_per_frame"] == 4

    def test_mismatched_seq_len_drops_derived_via_forward(self):
        fn, captured = _recording_plugin()
        q = torch.randn(2, 8, 4, 16)  # seq_len 8 != 4 * 4 -> derived dropped
        self._impl(fn).forward(q, q, q, self._md_with_geometry())
        params = captured["params"]
        assert "total_latent_frames" not in params
        assert "patches_per_frame" not in params
        assert _row0(params["latent_shape"]) == (4, 4, 4)  # raw still flows

    def test_explicit_derived_value_wins_over_resolver(self):
        # A dynamic `extra` override must beat the default resolver (later wins).
        fn, captured = _recording_plugin()
        q = torch.randn(2, 8, 4, 16)
        md = self._md_with_geometry()
        md.extra = {"total_latent_frames": 7}
        self._impl(fn).forward(q, q, q, md)
        assert captured["params"]["total_latent_frames"] == 7

    def test_no_geometry_no_derived_fields(self):
        fn, captured = _recording_plugin()
        q = torch.randn(1, 8, 4, 16)
        self._impl(fn).forward(q, q, q, AttentionMetadata())
        assert "total_latent_frames" not in captured["params"]
        assert "patches_per_frame" not in captured["params"]


class TestGeometryFn:
    """An optional plugin `geometry_fn` runs after the default resolver and may
    override/extend the geometry with a bespoke representation."""

    def _impl(self, fn):
        with patch("importlib.metadata.entry_points", return_value=[_fake_ep("dummy_sparse", fn)]):
            return _SparseAttentionImpl(
                num_heads=4,
                head_size=16,
                softmax_scale=1.0,
                backend_kwargs={"backend": "dummy_sparse", "params": {}},
            )

    def test_geometry_fn_overrides_and_extends(self):
        fn, captured = _recording_plugin()
        seen = {}

        def geometry_fn(query_shape, params):
            seen["query_shape"] = query_shape
            seen["frames_in"] = params.get("total_latent_frames")  # sees default-resolved value
            return {"total_latent_frames": 99, "block_layout": "csr", "ignored": None}

        fn.geometry_fn = geometry_fn
        q = torch.randn(2, 16, 4, 16)  # seq_len 16 == 4 frames * 4 patches
        md = AttentionMetadata(
            per_forward=PerForwardState(
                latent_shape=torch.tensor([[4, 4, 4]] * 2),
                patch_size=torch.tensor([[1, 2, 2]] * 2),
            )
        )
        self._impl(fn).forward(q, q, q, md)
        assert seen["frames_in"] == 4  # default resolver ran first
        assert seen["query_shape"] == q.shape
        assert captured["params"]["total_latent_frames"] == 99  # plugin override won
        assert captured["params"]["block_layout"] == "csr"  # bespoke extension merged
        assert "ignored" not in captured["params"]  # None returns dropped

    def test_non_callable_geometry_fn_ignored(self):
        fn, _ = _recording_plugin()
        fn.geometry_fn = "not callable"  # malformed -> ignored, no crash
        q = torch.randn(1, 8, 4, 16)
        self._impl(fn).forward(q, q, q, AttentionMetadata())


class TestConsumesPerForward:
    """The Attention layer builds per_forward only for backends that read it.

    Dense backends inherit ``consumes_per_forward = False`` so the layer skips
    constructing step / geometry tensors per layer per step on the default path;
    the sparse dispatcher opts in.
    """

    def test_base_dense_default_is_false(self):
        # The ABC default: a backend is assumed NOT to consume per_forward.
        assert AttentionImpl.consumes_per_forward is False

    def test_sparse_dispatcher_opts_in(self):
        # Class-level flag, readable without instantiating (gate is computed in
        # Attention.__init__ from the impl class).
        assert _SparseAttentionImpl.consumes_per_forward is True


class TestGatedMetadata:
    """attn_mask / full_attn_spans reach the kernel only when the plugin
    advertises support; an unadvertised gated input fails loudly."""

    def _impl(self, fn):
        with patch("importlib.metadata.entry_points", return_value=[_fake_ep("dummy_sparse", fn)]):
            return _SparseAttentionImpl(
                num_heads=4,
                head_size=16,
                softmax_scale=1.0,
                backend_kwargs={"backend": "dummy_sparse", "params": {}},
            )

    def test_attn_mask_forwarded_when_advertised(self):
        fn, captured = _capable_plugin({"supports_attention_mask": True})
        q = torch.randn(1, 8, 4, 16)
        mask = torch.ones(1, 8, dtype=torch.bool)
        self._impl(fn).forward(q, q, q, AttentionMetadata(attn_mask=mask))
        assert captured["params"]["attn_mask"] is mask

    def test_full_attn_spans_forwarded_when_advertised(self):
        fn, captured = _capable_plugin({"supports_full_attn_spans": True})
        q = torch.randn(1, 8, 4, 16)
        spans = [[(0, 4)]]
        self._impl(fn).forward(q, q, q, AttentionMetadata(full_attn_spans=spans))
        assert captured["params"]["full_attn_spans"] is spans

    def test_unadvertised_attn_mask_raises(self):
        fn, _ = _recording_plugin()  # no capabilities attribute
        q = torch.randn(1, 8, 4, 16)
        md = AttentionMetadata(attn_mask=torch.ones(1, 8, dtype=torch.bool))
        with pytest.raises(ValueError, match="supports_attention_mask"):
            self._impl(fn).forward(q, q, q, md)

    def test_unadvertised_full_attn_spans_raises(self):
        fn, _ = _recording_plugin()
        q = torch.randn(1, 8, 4, 16)
        md = AttentionMetadata(full_attn_spans=[[(0, 4)]])
        with pytest.raises(ValueError, match="supports_full_attn_spans"):
            self._impl(fn).forward(q, q, q, md)

    def test_malformed_capabilities_refuses(self):
        fn, _ = _recording_plugin()
        fn.capabilities = ["not", "a", "dict"]  # not a dict -> {} -> refuse
        q = torch.randn(1, 8, 4, 16)
        md = AttentionMetadata(attn_mask=torch.ones(1, 8, dtype=torch.bool))
        with pytest.raises(ValueError, match="supports_attention_mask"):
            self._impl(fn).forward(q, q, q, md)

    def test_absent_gated_fields_not_in_params(self):
        fn, captured = _recording_plugin()  # no capabilities, no mask provided
        q = torch.randn(1, 8, 4, 16)
        self._impl(fn).forward(q, q, q, AttentionMetadata())
        assert "attn_mask" not in captured["params"]
        assert "full_attn_spans" not in captured["params"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSparseDispatchCuda:
    """The device-agnostic dispatcher also runs on CUDA.

    A real sparse kernel ships as a separate external package, so this uses a
    reference (dense-equivalent) function plugin and checks tensors stay on
    device.
    """

    def test_dispatch_runs_on_cuda(self):
        captured = {}

        def kernel(q, k, v, params, is_causal, softmax_scale):
            captured["params"] = params
            captured["step_device"] = params["denoise_step_idx"].device
            return q

        with patch("importlib.metadata.entry_points", return_value=[_fake_ep("ref", kernel)]):
            impl = _SparseAttentionImpl(
                num_heads=4,
                head_size=64,
                softmax_scale=1.0,
                backend_kwargs={"backend": "ref", "params": {"topk": 0.5}},
            )

        dev = torch.device("cuda")
        q = torch.randn(2, 128, 4, 64, dtype=torch.bfloat16, device=dev)
        md = AttentionMetadata(
            per_forward=PerForwardState(
                denoise_step_idx=torch.tensor([3, 3], device=dev),
                total_denoise_steps=torch.tensor([40, 40], device=dev),
            )
        )
        out = impl.forward(q, q, q, md)

        assert out.device.type == "cuda"
        assert captured["params"]["topk"] == 0.5  # static config
        assert captured["params"]["denoise_step_idx"].tolist() == [3, 3]  # per-forward
        assert captured["step_device"].type == "cuda"  # tensors stay on-device


class TestQkvLayoutGuard:
    """The plugin contract mandates BSND; a model-declared non-BSND layout would
    hand the kernel mislabeled axes, so the impl refuses it at init."""

    def _make(self, qkv_layout):
        fn, _ = _recording_plugin()
        with patch("importlib.metadata.entry_points", return_value=[_fake_ep("dummy_sparse", fn)]):
            return _SparseAttentionImpl(
                num_heads=4,
                head_size=16,
                softmax_scale=1.0,
                qkv_layout=qkv_layout,
                backend_kwargs={"backend": "dummy_sparse"},
            )

    def test_non_bsnd_layout_raises(self):
        with pytest.raises(ValueError, match="BSND"):
            self._make("BNSD")

    def test_bsnd_and_unset_accepted(self):
        self._make("BSND")
        self._make(None)


class TestPluginShorthandRouting:
    """backend=<plugin> routes to the SPARSE_ATTN dispatcher."""

    def test_shorthand_routed_to_dispatcher(self):
        out = _route_sparse_plugin_shorthand(AttentionSpec(backend="dummy_sparse", extra={"topk": 0.3}))
        assert out.backend == "SPARSE_ATTN"
        assert out.extra == {"backend": "dummy_sparse", "params": {"topk": 0.3}}

    def test_shorthand_lone_nested_params_hoisted(self):
        # A user combining the shorthand with an explicit-form nested params dict
        # must not get double-wrapped params={"params": {...}}.
        out = _route_sparse_plugin_shorthand(AttentionSpec(backend="dummy_sparse", extra={"params": {"topk": 0.3}}))
        assert out.extra == {"backend": "dummy_sparse", "params": {"topk": 0.3}}

    def test_builtin_backend_unchanged(self):
        spec = AttentionSpec(backend="FLASH_ATTN")
        assert _route_sparse_plugin_shorthand(spec).backend == "FLASH_ATTN"

    def test_explicit_sparse_attn_unchanged(self):
        spec = AttentionSpec(backend="SPARSE_ATTN", extra={"backend": "dummy_sparse", "params": {"topk": 0.5}})
        out = _route_sparse_plugin_shorthand(spec)
        assert out.backend == "SPARSE_ATTN"
        assert out.extra == {"backend": "dummy_sparse", "params": {"topk": 0.5}}


class TestConfigParsing:
    """Impl accepts both nested and flat sparse config forms."""

    def _params_for(self, cfg):
        fn, _ = _recording_plugin()
        with patch("importlib.metadata.entry_points", return_value=[_fake_ep("dummy_sparse", fn)]):
            impl = _SparseAttentionImpl(num_heads=4, head_size=16, softmax_scale=1.0, backend_kwargs=cfg)
        return impl._static_params

    def test_nested_params(self):
        assert self._params_for({"backend": "dummy_sparse", "params": {"topk": 0.3}}) == {"topk": 0.3}

    def test_flat_params(self):
        # unknown keys (everything but "backend") become params
        assert self._params_for({"backend": "dummy_sparse", "topk": 0.3, "block": 128}) == {
            "topk": 0.3,
            "block": 128,
        }


class TestCrossAttnProtection:
    """Sparse on role='self' must not leak to role='cross'.

    Wan2.2 constructs self-attn with role='self' and cross-attn with role='cross',
    so a per-role sparse config targeting 'self' leaves cross-attn on the dense
    default. This is the per-role mechanism, not a per-model hook.
    """

    def test_self_only_config_leaves_cross_dense(self):
        cfg = AttentionConfig(per_role={"self": AttentionSpec(backend="dummy_sparse", extra={"topk": 0.3})})
        self_spec, _ = cfg.resolve_with_source(role="self")
        cross_spec, _ = cfg.resolve_with_source(role="cross")
        assert self_spec is not None and self_spec.backend == "dummy_sparse"
        assert cross_spec is None  # cross-attn untouched -> dense platform default
