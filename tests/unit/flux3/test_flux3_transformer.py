# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Flux3Transformer2DModel.

Tests run on CUDA.  All vllm tensor-parallel operations are mocked to
world_size=1 so no distributed init is required.  Tests that need model
construction or a forward pass are skipped automatically on machines
without a GPU.

Shape invariants (channel counts and 4-axis IDs) are checked against the
model's own contracts.
"""

from unittest import mock

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.flux3.flux3_transformer import (
    Flux3Transformer2DModel,
    _build_lora_weights_mapper,
    _build_qkv_weights_mapper,
    _build_split_qkv_checkpoint_mapper,
    _resolve_fuse_qkv,
)

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

require_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

DEVICE = torch.device("cuda")

# ---------------------------------------------------------------------------
# Shape constants
# ---------------------------------------------------------------------------

# Per-stream DiT input channel counts (_MODALITY_CHANNELS).
IMAGE_C = 128


# ---------------------------------------------------------------------------
# Module-scoped TP mock — needed on any device, no dist group in unit tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def mock_tp():
    """Patch vllm TP helpers to world_size=1 so the DiT builds and runs
    without an initialized distributed process group.

    Module-scoped so it is active before class-scoped fixtures set up their
    model instances.  Must be patched at every import namespace (linear.py
    and parameter.py both bind these names directly into their own namespace).
    """
    from vllm_omni.diffusion.attention import selector as _attn_selector

    _attn_selector._cached_get_backend_cls.cache_clear()
    _attn_selector._log_backend_resolution.cache_clear()

    patches = [
        mock.patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        mock.patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        mock.patch(
            "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        mock.patch(
            "vllm.model_executor.layers.linear.tensor_model_parallel_all_reduce",
            side_effect=lambda x: x,
        ),
        mock.patch(
            "vllm.model_executor.layers.linear.tensor_model_parallel_all_gather",
            side_effect=lambda x, dim=-1: x,
        ),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()
    _attn_selector._cached_get_backend_cls.cache_clear()
    _attn_selector._log_backend_resolution.cache_clear()


# ---------------------------------------------------------------------------
# Tiny model factory (fast; keeps tests under a second on GPU)
# ---------------------------------------------------------------------------

_TINY_KWARGS = dict(
    hidden_size=128,
    num_heads=4,  # head_dim = 128/4 = 32
    depth=1,
    depth_single_blocks=2,
    depth_late_blocks=0,
    context_in_dim=32,
    vec_in_dim=16,
    mlp_ratio=2.0,
    axes_dim=(8, 8, 8, 8),  # sum=32 == head_dim ✓
    theta=100,
    dtype=torch.bfloat16,
)


def _make_tiny_dit(**overrides) -> Flux3Transformer2DModel:
    kwargs = {**_TINY_KWARGS, **overrides}
    dtype = kwargs.pop("dtype", None)
    model = Flux3Transformer2DModel(**kwargs).to(DEVICE)
    return model.to(dtype=dtype) if dtype is not None else model


# ---------------------------------------------------------------------------
# Synthetic inputs — always on DEVICE
# ---------------------------------------------------------------------------


def _make_inputs(
    batch_size: int = 1,
    ctx_len: int = 4,
    context_in_dim: int = 32,
    vec_in_dim: int = 16,
    active_mods: list[str] | None = None,
    token_counts: dict[str, int] | None = None,
    device: torch.device = DEVICE,
    dtype: torch.dtype = torch.bfloat16,
):
    """Build minimal valid inputs for Flux3Transformer2DModel.forward."""
    if active_mods is None:
        active_mods = ["x_image"]

    # Stream token counts (small but >= 2).
    S = {"x_image": 4, "x_image_cond": 4}
    if token_counts is not None:
        S.update(token_counts)
    C = {
        "x_image": IMAGE_C,
        "x_image_cond": IMAGE_C,
    }

    ctx = torch.randn(batch_size, ctx_len, context_in_dim, device=device, dtype=dtype)
    ctx_ids = torch.zeros(batch_size, ctx_len, 4, device=device, dtype=torch.long)
    ctx_ids[..., 3] = torch.arange(ctx_len, device=device)
    vector = torch.randn(batch_size, vec_in_dim, device=device, dtype=dtype)
    timesteps_ctx = torch.zeros(batch_size, ctx_len, device=device, dtype=dtype)

    kwargs: dict = {}
    for mod in active_mods:
        s = S[mod]
        c = C[mod]
        kwargs[mod] = torch.randn(batch_size, s, c, device=device, dtype=dtype)
        kwargs[f"{mod}_ids"] = torch.zeros(batch_size, s, 4, device=device, dtype=torch.long)
        kwargs[f"{mod}_timesteps"] = torch.full((batch_size, s), 0.5, device=device, dtype=dtype)

    return ctx, ctx_ids, vector, timesteps_ctx, kwargs


# ---------------------------------------------------------------------------
# Shape invariants — pure arithmetic checks (no model construction, no GPU).
# ---------------------------------------------------------------------------


class TestShapeInvariants:
    """Shape / count invariants for the pipeline contract."""

    def test_channel_counts(self):
        """Image stream channel counts."""
        from vllm_omni.diffusion.models.flux3.flux3_transformer import (
            Flux3Transformer2DModel,
        )

        assert Flux3Transformer2DModel._MODALITY_CHANNELS == {
            "image": IMAGE_C,
            "image_cond": IMAGE_C,
        }

    def test_nvfp4_qkv_layout_fuses_only_prims_ts_targets(self):
        """Only PrimsTS targets can fuse NVFP4 Q/K/V without losing scales."""

        class NvFp4Config:
            def get_name(self):
                return "modelopt_fp4"

        config = NvFp4Config()
        assert _resolve_fuse_qkv(config) is False
        assert _resolve_fuse_qkv(config, "prims_ts") is False
        assert _resolve_fuse_qkv(config, "prims_ts", prims_ts_target=True) is True
        assert _resolve_fuse_qkv(None) is True

    def test_mixed_qkv_layout_maps_checkpoint_to_each_block(self):
        mapper, stacked = _build_qkv_weights_mapper(
            {"content_mode_blocks.image.0": True, "content_mode_blocks.image_cond.0": False},
        )
        weights = [
            (f"transformer.content_mode_blocks.{mode}.0.{projection}.{suffix}", torch.ones(1))
            for mode in ("image", "image_cond")
            for projection in ("q_proj", "k_proj", "v_proj")
            for suffix in ("weight", "weight_scale_2")
        ]
        mapped = list(mapper.apply(weights))

        assert [name for name, _ in mapped[:6]] == [
            f"transformer.content_mode_blocks.image.0.qkv.{suffix}"
            for _projection in ("q_proj", "k_proj", "v_proj")
            for suffix in ("weight", "weight_scale_2")
        ]
        assert [tensor.shard_id for _, tensor in mapped[:6]] == ["q", "q", "k", "k", "v", "v"]
        assert [name for name, _ in mapped[6:]] == [
            f"transformer.content_mode_blocks.image_cond.0.qkv.{projection}.{suffix}"
            for projection in ("q_proj", "k_proj", "v_proj")
            for suffix in ("weight", "weight_scale_2")
        ]
        assert all(not hasattr(tensor, "shard_id") for _, tensor in mapped[6:])
        assert stacked == [
            (".qkv.", f".{projection}.", shard)
            for projection, shard in (("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v"))
        ]

    def test_split_qkv_checkpoint_mapper_resolves_nested_runtime_names(self):
        mapper = _build_split_qkv_checkpoint_mapper(
            {
                "content_mode_blocks.image.0": True,
                "content_mode_blocks.image_cond.0": False,
                "txt_mode_blocks.0": False,
            }
        )
        model = object.__new__(Flux3Transformer2DModel)
        model._split_qkv_checkpoint_mapper = mapper

        assert mapper is not None
        assert mapper.apply_list(
            [
                "txt_mode_blocks.0.q_proj",
                "transformer.content_mode_blocks.image.0.q_proj.input_scale",
                "transformer.content_mode_blocks.image_cond.0.q_proj.input_scale",
            ]
        ) == [
            "txt_mode_blocks.0.qkv.q_proj",
            "transformer.content_mode_blocks.image.0.q_proj.input_scale",
            "transformer.content_mode_blocks.image_cond.0.qkv.q_proj.input_scale",
        ]
        assert (
            model.remap_checkpoint_key("transformer.content_mode_blocks.image_cond.0.q_proj.input_scale")
            == "transformer.content_mode_blocks.image_cond.0.qkv.q_proj.input_scale"
        )
        assert model.remap_checkpoint_key("transformer.content_mode_blocks.image.0.q_proj.input_scale") is None

    def test_lora_mapper_follows_split_and_fused_qkv_layouts(self):
        mapper = _build_lora_weights_mapper(
            {
                "content_mode_blocks.image.0": False,
                "single_blocks.0": True,
            }
        )

        assert mapper.apply_list(
            [
                "diffusion_model.content_mode_blocks.image.0.q_proj.lora_A.weight",
                "dit.content_mode_blocks.image.0.k_proj.lora_B.weight",
                "single_blocks.0.q_proj.lora_A.weight",
            ]
        ) == [
            "content_mode_blocks.image.0.qkv.q_proj.lora_A.weight",
            "content_mode_blocks.image.0.qkv.k_proj.lora_B.weight",
            "single_blocks.0.q_proj.lora_A.weight",
        ]

    def test_split_qkv_load_mapper_does_not_remap_runtime_names_twice(self):
        mapper, _ = _build_qkv_weights_mapper({"content_mode_blocks.image_cond.0": False})
        name = "content_mode_blocks.image_cond.0.qkv.q_proj.input_scale"

        assert mapper.apply_list([name]) == [name]

    @pytest.mark.parametrize(
        ("quant_name", "expected_packed_mapping"),
        [
            ("modelopt", {"qkv": ["q_proj", "k_proj", "v_proj"]}),
            ("modelopt_fp4", {}),
        ],
    )
    def test_quant_config_sees_flux3_runtime_names(self, quant_name, expected_packed_mapping):
        class QuantConfig:
            def __init__(self, name):
                self.name = name
                self.packed_modules_mapping = {}
                self.exclude_modules = [
                    "txt_mode_blocks.0.q_proj",
                    "txt_mode_blocks.0.k_proj",
                    "txt_mode_blocks.0.v_proj",
                    "final_layer.image.adaLN_modulation.1",
                ]

            def get_name(self):
                return self.name

            def apply_vllm_mapper(self, mapper):
                self.exclude_modules = mapper.apply_list(self.exclude_modules)

        quant_config = QuantConfig(quant_name)

        model = object.__new__(Flux3Transformer2DModel)
        model.quant_config = quant_config
        model._maybe_apply_model_mapping()

        assert quant_config.packed_modules_mapping == expected_packed_mapping
        assert quant_config.exclude_modules == [
            "txt_mode_blocks.0.q_proj",
            "txt_mode_blocks.0.k_proj",
            "txt_mode_blocks.0.v_proj",
            "final_layer.image.adaLN_modulation",
        ]
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            is_layer_skipped,
        )

        assert is_layer_skipped(
            "txt_mode_blocks.0.qkv",
            quant_config.exclude_modules,
            quant_config.packed_modules_mapping,
        ) is bool(expected_packed_mapping)
        assert is_layer_skipped(
            "final_layer.image.adaLN_modulation",
            quant_config.exclude_modules,
            quant_config.packed_modules_mapping,
        )

    def test_4_axis_position_ids(self):
        """Position IDs are always 4-axis (t, h, w, l) for image streams."""
        _, ctx_ids, _, _, kwargs = _make_inputs(
            active_mods=["x_image", "x_image_cond"],
            device=torch.device("cpu"),
        )
        assert ctx_ids.shape[-1] == 4
        assert kwargs["x_image_ids"].shape[-1] == 4
        assert kwargs["x_image_cond_ids"].shape[-1] == 4


# ---------------------------------------------------------------------------
# Construction (GPU required — Attention selects a CUDA backend at init time)
# ---------------------------------------------------------------------------


@require_gpu
class TestConstruction:
    def test_default_construction(self):
        dit = _make_tiny_dit()
        assert isinstance(dit, nn.Module)

    def test_axes_dim_mismatch_raises(self):
        with pytest.raises(AssertionError, match="axes_dim"):
            _make_tiny_dit(axes_dim=(4, 4, 4, 4))  # sum=16 ≠ head_dim=32

    def test_config_attributes(self):
        dit = _make_tiny_dit()
        assert dit.config.hidden_size == 128
        assert dit.config.num_heads == 4
        assert dit.config.depth == 1
        assert dit.config.depth_single_blocks == 2
        assert dit.config.context_in_dim == 32
        assert dit.config.vec_in_dim == 16
        assert dit.config.out_modes == ("image",)

    def test_flow_map_adds_reference_time_embedder(self):
        dit = _make_tiny_dit(flow_map=True)

        assert dit.config.flow_map is True
        assert dit.ref_time_in is not None
        assert "ref_time_in.in_layer.weight" in dict(dit.named_parameters())
        assert "ref_time_in.out_layer.weight" in dict(dit.named_parameters())

    def test_sequence_map_covers_all_modalities(self):
        from vllm_omni.diffusion.models.flux3.flux3_transformer import (
            Flux3Transformer2DModel,
        )

        expected = {"x_image", "x_image_cond"}
        assert set(Flux3Transformer2DModel._SEQUENCE_MAP.keys()) == expected

    def test_per_modality_embedders_exist(self):
        dit = _make_tiny_dit()
        for mod_key in dit._MODALITY_CHANNELS:
            assert mod_key in dit.emb_in
            assert mod_key in dit.content_mode_blocks
        assert set(dit.final_layer) == {"image"}

    def test_parameters_on_cuda(self):
        dit = _make_tiny_dit()
        for name, param in dit.named_parameters():
            assert param.is_cuda, f"parameter {name!r} is not on CUDA"


# ---------------------------------------------------------------------------
# Forward pass — output shapes (GPU required)
# ---------------------------------------------------------------------------


@require_gpu
class TestForwardShape:
    @pytest.fixture(scope="class")
    @classmethod
    def dit(cls):
        return _make_tiny_dit()

    def _run(self, dit, active_mods):
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=active_mods)
        return dit(ctx, ctx_ids, vector, ts_ctx, **kwargs), kwargs

    def test_image_output_shape(self, dit):
        out, kwargs = self._run(dit, ["x_image"])
        assert "x_image" in out
        assert out["x_image"].shape == kwargs["x_image"].shape

    def test_image_with_image_cond(self, dit):
        out, kwargs = self._run(dit, ["x_image", "x_image_cond"])
        assert "x_image" in out
        assert out["x_image"].shape == kwargs["x_image"].shape
        assert "x_image_cond" not in out

    def test_output_dtype_matches_input(self, dit):
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=["x_image"], dtype=torch.bfloat16)
        out = dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)
        assert out["x_image"].dtype == torch.bfloat16

    def test_output_on_cuda(self, dit):
        out, _ = self._run(dit, ["x_image"])
        assert out["x_image"].is_cuda

    def test_flow_map_forward_requires_reference_timestep(self):
        dit = _make_tiny_dit(flow_map=True).eval()
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=["x_image"])

        with pytest.raises(ValueError, match="timesteps_ref is required"):
            dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)

        out = dit(ctx, ctx_ids, vector, ts_ctx, timesteps_ref=torch.tensor([0.25], device=DEVICE), **kwargs)
        assert out["x_image"].shape == kwargs["x_image"].shape

    def test_phase2_uses_one_joint_norm_per_block(self):
        dit = _make_tiny_dit(dtype=torch.bfloat16)
        calls = [0 for _ in dit.single_blocks]
        handles = []
        for index, block in enumerate(dit.single_blocks):

            def count_call(_module, _args, _output, *, index=index):
                calls[index] += 1

            handles.append(block.norm1.register_forward_hook(count_call))
        try:
            ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(
                active_mods=["x_image"],
                dtype=torch.bfloat16,
            )
            dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

        assert calls == [1] * len(dit.single_blocks)

    def test_joint_norm_matches_separate_stream_norms(self, dit):
        block = dit.single_blocks[0]
        batch, ctx_len, content_len, hidden = 2, 4, 6, dit.config.hidden_size
        ctx = torch.randn(batch, ctx_len, hidden, device=DEVICE)
        content = torch.randn(batch, content_len, hidden, device=DEVICE)
        ctx_scale = torch.randn_like(ctx)
        ctx_shift = torch.randn_like(ctx)
        content_scale = torch.randn_like(content)
        content_shift = torch.randn_like(content)

        separate = torch.cat(
            [block.norm1(content, content_scale, content_shift), block.norm1(ctx, ctx_scale, ctx_shift)],
            dim=1,
        )
        joint = block.norm1(
            torch.cat([content, ctx], dim=1),
            torch.cat([content_scale, ctx_scale], dim=1),
            torch.cat([content_shift, ctx_shift], dim=1),
        )

        torch.testing.assert_close(joint, separate)

    def test_joint_phase2_matches_cache_dit_two_stream_contract(self):
        dit = _make_tiny_dit(dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in dit.parameters():
                parameter.uniform_(-0.02, 0.02)
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(
            active_mods=["x_image", "x_image_cond"],
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            optimized = dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)
            dit._is_cached = True
            try:
                two_stream = dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)
            finally:
                del dit._is_cached

        assert optimized.keys() == two_stream.keys()
        for key in optimized:
            torch.testing.assert_close(optimized[key], two_stream[key])

    @pytest.mark.parametrize(
        ("generated_mod", "condition_mod"),
        [("x_image", "x_image_cond")],
    )
    def test_static_condition_phase1_matches_uncached(self, generated_mod, condition_mod):
        dit = _make_tiny_dit(dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in dit.parameters():
                parameter.uniform_(-0.02, 0.02)
        active_mods = [generated_mod, condition_mod]
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(
            batch_size=2,
            active_mods=active_mods,
            dtype=torch.bfloat16,
        )
        ids_by_mod = {key: kwargs[f"{key}_ids"] for key in active_mods}
        static_content = {
            condition_mod: (
                kwargs[condition_mod],
                kwargs[f"{condition_mod}_timesteps"],
            )
        }

        with torch.no_grad():
            uncached = dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)
            precomputed = dit.precompute_static_conditioning(
                ctx,
                ctx_ids,
                vector,
                ts_ctx,
                ids_by_mod=ids_by_mod,
                static_content=static_content,
            )

        calls = 0

        def count_call(_module, _args, _output):
            nonlocal calls
            calls += 1

        condition_key = dit._SEQUENCE_MAP[condition_mod]
        handles = [block.register_forward_hook(count_call) for block in dit.content_mode_blocks[condition_key]]
        try:
            with torch.no_grad():
                cached = dit(ctx, ctx_ids, vector, ts_ctx, precomputed=precomputed, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

        assert calls == 0
        assert cached.keys() == uncached.keys()
        for key in cached:
            torch.testing.assert_close(cached[key], uncached[key])

    def test_padded_cfg_text_matches_separate_forwards(self):
        """B=2 over text padded to the longer branch equals two B=1 forwards."""
        from vllm_omni.diffusion.models.flux3.pipeline_flux3 import _merge_padded_cfg, _pad_seq

        dit = _make_tiny_dit(dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in dit.parameters():
                parameter.uniform_(-0.02, 0.02)
        branches = [_make_inputs(ctx_len=n, dtype=torch.bfloat16) for n in (4, 8)]
        x = branches[0][4]  # same image stream for both branches, as in CFG
        ids_by_mod = {"x_image": x["x_image_ids"]}
        with torch.no_grad():
            separate, precomputed = [], []
            for ctx, ctx_ids, vector, ts_ctx, _ in branches:
                pre = dit.precompute_static_conditioning(ctx, ctx_ids, vector, ts_ctx, ids_by_mod=ids_by_mod)
                precomputed.append(pre)
                separate.append(dit(ctx, ctx_ids, vector, ts_ctx, precomputed=pre, **x)["x_image"])
            merged = _merge_padded_cfg(*precomputed, 8)
            merged["text_lens"] = (4, 8)
            fused_inputs = [torch.cat([_pad_seq(a[i], 8) if i != 2 else a[i] for a in branches]) for i in range(4)]
            fused_x = {k: torch.cat([v, v]) for k, v in x.items()}
            fused = dit(*fused_inputs, precomputed=merged, **fused_x)["x_image"]

        torch.testing.assert_close(fused, torch.cat(separate), atol=2e-2, rtol=2e-2)

    @pytest.mark.parametrize("active_mods", [["x_image"], ["x_image", "x_image_cond"]])
    def test_broadcast_modulation_matches_per_token(self, active_mods, monkeypatch):
        """One modulation row per stream, gathered per token, equals per-token modulation."""
        dit = _make_tiny_dit(dtype=torch.bfloat16).eval()
        with torch.no_grad():
            for parameter in dit.parameters():
                parameter.uniform_(-0.02, 0.02)
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(batch_size=2, active_mods=active_mods, dtype=torch.bfloat16)
        static = {k: (kwargs[k], kwargs[f"{k}_timesteps"]) for k in active_mods if k.endswith("_cond")}
        ids_by_mod = {k: kwargs[f"{k}_ids"] for k in active_mods}
        outputs = []
        with torch.no_grad():
            for broadcast in (False, True):
                if not broadcast:  # per-token reference: every timestep row is embedded
                    monkeypatch.setattr(dit, "_timestep_rows", lambda timesteps: timesteps)
                else:
                    monkeypatch.undo()
                pre = dit.precompute_static_conditioning(
                    ctx, ctx_ids, vector, ts_ctx, ids_by_mod=ids_by_mod, static_content=static
                )
                outputs.append(dit(ctx, ctx_ids, vector, ts_ctx, precomputed=pre, **kwargs)["x_image"])
                outputs.append(dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)["x_image"])

        torch.testing.assert_close(outputs[2], outputs[0], atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(outputs[3], outputs[1], atol=2e-2, rtol=2e-2)

    def test_fp8_attention_real_sm100_forward(self):
        """Run actual FlashInfer/cuDNN FP8 attention on a supported SM100 GPU.

        Selects FP8 through the standard per-role attention config
        (FLASHINFER_ATTN + quant dtype_qk=fp8_e4m3), no model-side flag.
        """
        from vllm.utils.flashinfer import is_flashinfer_cudnn_fp8_prefill_attn_supported

        # The general capability check admits SM120, where this graph has no cuDNN engine.
        if torch.cuda.get_device_capability() != (10, 0):
            pytest.skip("This cuDNN FP8 attention graph requires SM100")

        if not is_flashinfer_cudnn_fp8_prefill_attn_supported():
            pytest.skip("Requires SM100 with FlashInfer cuDNN FP8 prefill attention support")

        from vllm_omni.diffusion.attention.backends.flashinfer_attn import FlashInferAttentionImpl
        from vllm_omni.diffusion.config import set_current_diffusion_config
        from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec, OmniDiffusionConfig

        od_config = OmniDiffusionConfig(
            model="",
            diffusion_attention_config=AttentionConfig(
                default=AttentionSpec(backend="FLASHINFER_ATTN", quant={"dtype_qk": "fp8_e4m3"}),
            ),
        )
        with set_current_diffusion_config(od_config):
            dit = _make_tiny_dit(
                hidden_size=1024,
                num_heads=8,
                axes_dim=(32, 32, 32, 32),
                dtype=torch.bfloat16,
            ).eval()
        # vllm Linear weights are uninitialized until checkpoint load; give them
        # real values so the fp8 path is exercised on non-trivial activations.
        with torch.no_grad():
            for p in dit.parameters():
                p.normal_(0, 0.02) if p.ndim >= 2 else p.fill_(1.0)
        blocks = [*dit.txt_mode_blocks, *dit.single_blocks]
        blocks.extend(block for stream in dit.content_mode_blocks.values() for block in stream)
        assert blocks and all(
            isinstance(block.attn.attention, FlashInferAttentionImpl) and block.attn.attention._fp8_cudnn
            for block in blocks
        )

        ctx, ctx_ids, vector, timesteps_ctx, kwargs = _make_inputs(
            ctx_len=256,
            active_mods=["x_image"],
            token_counts={"x_image": 256},
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            out = dit(ctx, ctx_ids, vector, timesteps_ctx, **kwargs)

        assert out["x_image"].shape == kwargs["x_image"].shape
        assert out["x_image"].dtype == torch.bfloat16
        assert torch.isfinite(out["x_image"].float()).all()


# ---------------------------------------------------------------------------
# Forward pass — active/inactive modality contract (GPU required)
# ---------------------------------------------------------------------------


@require_gpu
class TestActiveInactiveModalities:
    @pytest.fixture(scope="class")
    @classmethod
    def dit(cls):
        return _make_tiny_dit()

    def test_non_input_mods_absent_from_output(self, dit):
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=["x_image"])
        out = dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)
        # Streams not passed as inputs must not appear in the output.
        not_passed = {"x_image_cond"}
        assert not (set(out.keys()) & not_passed), f"non-input mods leaked into output: {set(out.keys()) & not_passed}"

    def test_all_image_streams_active(self, dit):
        image_mods = ["x_image", "x_image_cond"]
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=image_mods)
        out = dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)
        assert set(out.keys()) == {"x_image"}

    def test_output_keys_match_active_mods(self, dit):
        active = ["x_image", "x_image_cond"]
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=active)
        out = dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)
        assert set(out.keys()) == {"x_image"}

    def test_unknown_inputs_are_rejected(self, dit):
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=["x_image"])
        kwargs.update(
            {
                "x_unknown": torch.zeros(1, 2, 128, device=DEVICE, dtype=torch.bfloat16),
                "x_unknown_ids": torch.zeros(1, 2, 4, device=DEVICE, dtype=torch.long),
                "x_unknown_timesteps": torch.zeros(1, 2, device=DEVICE, dtype=torch.bfloat16),
            }
        )

        with pytest.raises(ValueError, match="Flux3 transformer"):
            dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)

    @pytest.mark.parametrize(
        "active",
        [
            ["x_image"],
            ["x_image", "x_image_cond"],
        ],
    )
    def test_late_blocks_process_only_active_modalities(self, active):
        dit = _make_tiny_dit(depth_late_blocks=1)
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=active)

        out = dit(ctx, ctx_ids, vector, ts_ctx, **kwargs)

        expected = {"x_image"} if "x_image" in active else set()
        assert set(out) == expected
        for mod in expected:
            assert out[mod].shape == kwargs[mod].shape


# ---------------------------------------------------------------------------
# PAG (perturbed-attention guidance) (GPU required)
# ---------------------------------------------------------------------------


@require_gpu
class TestPAG:
    @pytest.fixture(scope="class")
    @classmethod
    def dit(cls):
        return _make_tiny_dit()

    def test_pag_none_does_not_crash(self, dit):
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=["x_image"])
        out = dit(ctx, ctx_ids, vector, ts_ctx, pag_single_layers=None, **kwargs)
        assert "x_image" in out

    def test_pag_layer_zero_does_not_crash(self, dit):
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=["x_image"])
        out = dit(ctx, ctx_ids, vector, ts_ctx, pag_single_layers=[0], pag_strength=0.5, **kwargs)
        assert "x_image" in out

    def test_pag_empty_list_does_not_crash(self, dit):
        ctx, ctx_ids, vector, ts_ctx, kwargs = _make_inputs(active_mods=["x_image"])
        out = dit(ctx, ctx_ids, vector, ts_ctx, pag_single_layers=[], **kwargs)
        assert "x_image" in out


# ---------------------------------------------------------------------------
# load_weights (GPU required — model construction needs CUDA backend)
# ---------------------------------------------------------------------------


@require_gpu
class TestLoadWeights:
    def test_load_weights_empty_returns_empty_set(self):
        dit = _make_tiny_dit()
        loaded = dit.load_weights(iter([]))
        assert isinstance(loaded, set)
        assert len(loaded) == 0

    def test_load_weights_direct_param(self):
        dit = _make_tiny_dit()
        # Grab any parameter and reload its current value
        name, param = next(iter(dit.named_parameters()))
        loaded = dit.load_weights(iter([(name, param.data.clone())]))
        assert name in loaded
