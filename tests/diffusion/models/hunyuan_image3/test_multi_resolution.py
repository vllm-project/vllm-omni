# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for multi-resolution support in HunyuanImage3.

Covers three areas:

1. ``ResolutionGroup._calc_by_step`` — training-bucket generation,
   verifying that each resolution has correct width×height and that
   indices match expected ratio ordering.

2. ``HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS`` — extra resolution matching
   via ``Resolution.append`` and new-entry insertion in
   ``ResolutionGroup.__init__``.

3. ``get_cached_resolution_group`` and ``ar2diffusion`` — caching
   behaviour and correct ``ResolutionGroup`` construction for the
   AR→DiT stage transition.
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS,
    Resolution,
    ResolutionGroup,
    get_cached_resolution_group,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_reso(desc: str | tuple[int, int]) -> Resolution:
    """Build a ``Resolution`` from a ``"HxW"`` string or ``(h, w)`` tuple."""
    if isinstance(desc, str):
        h, w = desc.split("x")
        return Resolution(int(h), int(w))
    return Resolution(desc[0], desc[1])


def _reso_set(rg: ResolutionGroup) -> set[tuple[int, int]]:
    """Return all ``(h, w)`` tuples reachable through a ``ResolutionGroup``.

    For entries that carry ``extra_res`` this includes every alternative
    resolution in that set.
    """
    out: set[tuple[int, int]] = set()
    for r in rg.data:
        out.add((r.h, r.w))
        out |= r.extra_res
    return out


# ---------------------------------------------------------------------------
# 1.  _calc_by_step — training-bucket resolution parsing
# ---------------------------------------------------------------------------

_BASE = 1024
_STEP = _BASE // 16  # 64
_ALIGN = 16
_MIN_SIDE = _BASE // 2  # 512
_MAX_SIDE = _BASE * 2  # 2048


def _build_plain_group(
    base_size: int = _BASE,
    step: int | None = None,
    align: int = _ALIGN,
) -> ResolutionGroup:
    """Return a ``ResolutionGroup`` with *no* extra resolutions."""
    return ResolutionGroup(base_size=base_size, step=step, align=align, extra_resolutions=None)


class TestCalcByStep:
    """Verify ``_calc_by_step`` generates the expected training buckets."""

    # -- count & range -------------------------------------------------------

    def test_count_for_base_1024(self):
        """33 buckets: 1 centre + 2×16 from the stepping loops."""
        rg = _build_plain_group()
        # 1 centre + 16 portrait + 16 landscape
        assert len(rg.data) == 33, f"expected 33 training buckets, got {len(rg.data)}"

    def test_all_sizes_within_bounds(self):
        rg = _build_plain_group()
        for r in rg.data:
            assert _MIN_SIDE <= r.h <= _MAX_SIDE, f"height {r.h} outside [{_MIN_SIDE}, {_MAX_SIDE}]"
            assert _MIN_SIDE <= r.w <= _MAX_SIDE, f"width {r.w} outside [{_MIN_SIDE}, {_MAX_SIDE}]"

    def test_all_sizes_aligned(self):
        rg = _build_plain_group()
        for r in rg.data:
            assert r.h % _ALIGN == 0, f"height {r.h} not {_ALIGN}-aligned"
            assert r.w % _ALIGN == 0, f"width {r.w} not {_ALIGN}-aligned"

    # -- ratio ordering ------------------------------------------------------

    def test_sorted_by_ratio_ascending(self):
        rg = _build_plain_group()
        ratios = [r.ratio for r in rg.data]
        for i in range(1, len(ratios)):
            assert ratios[i - 1] <= ratios[i], f"ratio[{i - 1}]={ratios[i - 1]:.4f} > ratio[{i}]={ratios[i]:.4f}"

    def test_ratio_index_matches_numpy_array(self):
        rg = _build_plain_group()
        for i, r in enumerate(rg.data):
            assert math.isclose(rg.ratio[i], r.ratio, rel_tol=1e-9), (
                f"rg.ratio[{i}]={rg.ratio[i]:.6f} != data[{i}].ratio={r.ratio:.6f}"
            )

    # -- key index positions -------------------------------------------------

    @pytest.mark.parametrize(
        "idx,expected_hw",
        [
            # First (most portrait) → 1:4 ratio bucket
            (0, (512, 2048)),
            # Centre square bucket
            (16, (1024, 1024)),
            # Most landscape bucket → 4:1 ratio
            (32, (2048, 512)),
        ],
    )
    def test_landmark_index(self, idx, expected_hw):
        rg = _build_plain_group()
        r = rg.data[idx]
        assert (r.h, r.w) == expected_hw, f"idx {idx}: expected {expected_hw}, got ({r.h}, {r.w})"

    def test_center_is_square(self):
        rg = _build_plain_group()
        r = rg.data[16]
        assert r.h == r.w == _BASE

    def test_first_and_last_are_extremes(self):
        rg = _build_plain_group()
        # first  → 512×2048  (ratio 0.25)
        assert rg.data[0].ratio < 0.3
        # last   → 2048×512  (ratio 4.0)
        assert rg.data[-1].ratio > 3.9

    # -- width × height matching ---------------------------------------------

    def test_every_resolution_has_same_area_product(self):
        """Symmetry: stepping loop produces the same product set in both
        directions, so (h × w) tuples should appear symmetrically."""
        rg = _build_plain_group()
        products = {(r.h, r.w) for r in rg.data}
        # For every (h, w) there should exist (w, h) or a nearby equivalent
        for h, w in products:
            if h == w:
                continue  # square — trivially symmetric
            assert (w, h) in products or any(abs(rr.h - w) <= _STEP and abs(rr.w - h) <= _STEP for rr in rg.data), (
                f"no symmetric counterpart for {h}×{w}"
            )

    def test_product_of_wh_divisible_by_256(self):
        """Each bucket's area should be a multiple of 256 (16×16),
        which is the VAE down-sample grid."""
        rg = _build_plain_group()
        for r in rg.data:
            assert (r.h * r.w) % 256 == 0, f"{r.h}×{r.w} = {r.h * r.w} not divisible by 256"

    # -- custom base_size / step / align -------------------------------------

    @pytest.mark.parametrize("base_size", [512, 768, 1280])
    def test_count_scales_with_base_size(self, base_size):
        step = base_size // 16
        rg = ResolutionGroup(base_size=base_size, step=step, align=16, extra_resolutions=None)
        expected = 1 + 2 * (base_size // step)  # centre + 2×phases
        assert len(rg.data) == expected, f"base_size={base_size}: expected {expected}, got {len(rg.data)}"

    def test_custom_step(self):
        """A smaller step produces more buckets."""
        rg_small = ResolutionGroup(base_size=1024, step=32, align=16, extra_resolutions=None)
        rg_default = _build_plain_group()
        assert len(rg_small.data) > len(rg_default.data), "smaller step should produce more buckets"


# ---------------------------------------------------------------------------
# 2.  HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS matching
# ---------------------------------------------------------------------------

_EXTRA_SQUARES = {"512x512", "640x640", "768x768", "896x896"}
_EXTRA_NON_SQUARES = {"1024x768", "1280x720", "768x1024", "720x1280"}


def _build_extra_group() -> ResolutionGroup:
    return ResolutionGroup(
        base_size=_BASE,
        extra_resolutions=[_make_reso(s) for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS],
    )


class TestExtraResolutionMatching:
    """Verify extra resolutions are correctly integrated into
    the ``ResolutionGroup``."""

    # -- square extras (ratio = 1.0) -----------------------------------------

    def test_square_extras_appended_to_base_square(self):
        """512², 640², 768², 896² all have ratio 1.0 and should be
        appended to the (1024, 1024) Resolution via ``append()``."""
        rg = _build_extra_group()
        # Find the 1024×1024 entry
        square_reso = next(r for r in rg.data if r.h == 1024 and r.w == 1024)
        expected_extra = {(512, 512), (640, 640), (768, 768), (896, 896)}
        assert square_reso.extra_res == expected_extra, f"square extra_res: {square_reso.extra_res}"

    def test_base_1024_unchanged(self):
        """The 1024×1024 entry itself still reports h=1024, w=1024."""
        rg = _build_extra_group()
        r = next(r for r in rg.data if r.h == 1024 and r.w == 1024)
        assert r.h == 1024 and r.w == 1024

    def test_squares_not_standalone_entries(self):
        """Square extras must NOT appear as top-level ``Resolution`` entries;
        they live inside ``extra_res`` only."""
        rg = _build_extra_group()
        standalone = {(r.h, r.w) for r in rg.data}
        for square in [(512, 512), (640, 640), (768, 768), (896, 896)]:
            assert square not in standalone, f"{square} should be in extra_res, not a standalone entry"

    # -- non-square extras (ratio ≠ 1.0) ------------------------------------

    @pytest.mark.parametrize("reso_str", sorted(_EXTRA_NON_SQUARES))
    def test_non_square_extra_is_standalone_entry(self, reso_str):
        """1024×768, 1280×720, 768×1024, 720×1280 are appended as
        new ``Resolution`` entries because no training bucket shares
        their exact ratio."""
        r = _make_reso(reso_str)
        rg = _build_extra_group()
        matching = [d for d in rg.data if d.h == r.h and d.w == r.w]
        assert len(matching) == 1, f"{reso_str} should be in rg.data exactly once"
        assert matching[0].extra_res == set(), f"{reso_str} should have empty extra_res, got {matching[0].extra_res}"

    def test_non_square_extras_empty_extra_res(self):
        """Non-square extras have no own extra_res."""
        rg = _build_extra_group()
        for r in rg.data:
            if (r.h, r.w) in {(1024, 768), (1280, 720), (768, 1024), (720, 1280)}:
                assert r.extra_res == set()

    # -- total count ---------------------------------------------------------

    def test_total_count(self):
        """33 training + 4 non-square extras = 37."""
        rg = _build_extra_group()
        assert len(rg.data) == 37, f"expected 37, got {len(rg.data)}"

    def test_extra_count(self):
        """Exactly 8 extra resolutions (4 square + 4 non-square)."""
        rg = _build_extra_group()
        reso_set = _reso_set(rg)
        extra_in_set = reso_set - {(r.h, r.w) for r in _build_plain_group().data}
        assert len(extra_in_set) == 8, f"expected 8 extra, got {len(extra_in_set)}: {extra_in_set}"

    # -- index stability -----------------------------------------------------

    def test_indices_0_32_unchanged(self):
        """Indices 0–32 remain the training buckets (sorted by ratio)."""
        plain = _build_plain_group()
        rg = _build_extra_group()
        for i in range(33):
            assert (rg.data[i].h, rg.data[i].w) == (plain.data[i].h, plain.data[i].w), (
                f"idx {i}: extra group has ({rg.data[i].h},{rg.data[i].w}), "
                f"expected ({plain.data[i].h},{plain.data[i].w})"
            )

    @pytest.mark.parametrize(
        "reso_str,expected_idx",
        [
            ("1024x768", 33),
            ("1280x720", 34),
            ("768x1024", 35),
            ("720x1280", 36),
        ],
    )
    def test_non_square_extra_indices(self, reso_str, expected_idx):
        """Non-square extras are appended in order at indices 33-36."""
        r = _make_reso(reso_str)
        rg = _build_extra_group()
        idx = next(i for i, d in enumerate(rg.data) if d.h == r.h and d.w == r.w)
        assert idx == expected_idx, f"{reso_str}: expected idx {expected_idx}, got {idx}"

    # -- ratio lookup via get_base_size_and_ratio_index ----------------------

    @pytest.mark.parametrize(
        "h,w,expected_idx",
        [
            (1024, 768, 33),
            (1280, 720, 34),
            (768, 1024, 35),
            (720, 1280, 36),
            (1024, 1024, 16),
            (512, 512, 16),
            (640, 640, 16),
            (768, 768, 16),
            (896, 896, 16),
        ],
    )
    def test_ratio_index_for_resolutions(self, h, w, expected_idx):
        """``get_base_size_and_ratio_index`` must return the correct
        ratio index for both training-bucket and extra resolutions."""
        rg = _build_extra_group()
        _, idx = rg.get_base_size_and_ratio_index(w, h)
        assert idx == expected_idx, f"({w},{h}): expected idx {expected_idx}, got {idx}"


# ---------------------------------------------------------------------------
# Resolution.match — area-based matching within a ratio group
# ---------------------------------------------------------------------------


class TestResolutionMatch:
    """Verify ``Resolution.match`` picks the best-fit alternative from
    ``extra_res`` based on area proximity."""

    def test_no_extra_returns_self(self):
        r = _make_reso("1024x768")
        assert r.match(800, 600) == (768, 1024)

    @pytest.mark.parametrize(
        "target_w,target_h,expected",
        [
            # Exact area match in extra_res
            (512, 512, (512, 512)),
            (640, 640, (640, 640)),
            (768, 768, (768, 768)),
            (896, 896, (896, 896)),
            # Closer to 1024² than to any extra square
            (1024, 1024, (1024, 1024)),
            (1000, 1000, (1024, 1024)),
            # Small square picks 512²
            (256, 256, (512, 512)),
            (400, 400, (512, 512)),
            # 700² → closest to 640² (diff 129600) vs 768² (diff 99856)
            # 768*768 - 700*700 = 589824 - 490000 = 99824
            # 640*640 - 700*700 = 409600 - 490000 = -80400 → abs=80400
            # So 640² is closer in area
            (700, 700, (640, 640)),
            # 800² → closest to 768² (diff 49984) vs 896² (diff 162944)
            (800, 800, (768, 768)),
        ],
    )
    def test_square_extra_match(self, target_w, target_h, expected):
        """Build a Resolution like the 1024² entry with square extras,
        then verify area-based matching."""
        r = Resolution(1024, 1024)
        for s in [(512, 512), (640, 640), (768, 768), (896, 896)]:
            r.extra_res.add(s)
        result = r.match(target_w, target_h)
        assert result == expected, f"match({target_w},{target_h}) = {result}, expected {expected}"

    def test_match_returns_tuple(self):
        r = Resolution(1024, 1024)
        r.extra_res.add((512, 512))
        result = r.match(512, 512)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(v, int) for v in result)

    def test_match_does_not_mutate_resolution(self):
        r = Resolution(1024, 1024)
        r.extra_res.add((512, 512))
        extra_before = set(r.extra_res)
        r.match(300, 300)
        assert r.extra_res == extra_before


# ---------------------------------------------------------------------------
# ResolutionGroup.get_target_size — end-to-end resolution selection
# ---------------------------------------------------------------------------


class TestGetTargetSize:
    """Verify ``get_target_size`` uses ratio-then-area logic correctly."""

    @pytest.fixture(scope="class")
    def rg(self) -> ResolutionGroup:
        return _build_extra_group()

    @pytest.mark.parametrize(
        "w,h,expected_min_area,expected_max_area",
        [
            # Square targets should land on square variants
            (1024, 1024, 1024 * 1024, 1024 * 1024),
            (512, 512, 512 * 512, 512 * 512),
            (640, 640, 640 * 640, 640 * 640),
            (768, 768, 768 * 768, 768 * 768),
            (896, 896, 896 * 896, 896 * 896),
        ],
    )
    def test_square_target_returns_square(self, rg, w, h, expected_min_area, expected_max_area):
        rw, rh = rg.get_target_size(w, h)
        area = rw * rh
        assert expected_min_area <= area <= expected_max_area, f"({w},{h}) → ({rw},{rh}) area={area}"

    def test_training_bucket_self_maps(self, rg):
        """A training bucket should map to itself (or an extra_res variant
        with the same ratio)."""
        for r in rg.data:
            rw, rh = rg.get_target_size(r.w, r.h)
            # Either exact match or the ratio must be preserved
            assert abs(rh / rw - r.ratio) < 0.01, f"({r.w},{r.h}) → ({rw},{rh}) ratio drifted from {r.ratio:.4f}"

    def test_wild_ratio_still_finds_bucket(self, rg):
        """Even an unusual aspect ratio should map to some valid bucket."""
        for w, h in [(800, 600), (1200, 900), (1600, 400), (400, 1600)]:
            rw, rh = rg.get_target_size(w, h)
            assert rw > 0 and rh > 0
            assert rw % 16 == 0 and rh % 16 == 0


# ---------------------------------------------------------------------------
# 3.  get_cached_resolution_group & ar2diffusion integration
# ---------------------------------------------------------------------------


class TestGetCachedResolutionGroup:
    """Verify ``get_cached_resolution_group`` returns a correctly
    configured ``ResolutionGroup`` and respects the LRU cache."""

    def test_returns_resolution_group(self):
        rg = get_cached_resolution_group(base_size=1024)
        assert isinstance(rg, ResolutionGroup)

    def test_base_size_propagated(self):
        rg = get_cached_resolution_group(base_size=1024)
        assert rg.base_size == 1024

    def test_has_extra_resolutions(self):
        rg = get_cached_resolution_group(base_size=1024)
        reso_set = _reso_set(rg)
        for s in HUNYUAN_IMAGE3_EXTRA_RESOLUTIONS:
            r = _make_reso(s)
            assert (r.h, r.w) in reso_set, f"{s} missing from cached group"

    def test_cache_same_base_size_same_object(self):
        """LRU cache: same base_size → same object identity."""
        # Clear any prior cache entries
        get_cached_resolution_group.cache_clear()
        rg1 = get_cached_resolution_group(base_size=1024)
        rg2 = get_cached_resolution_group(base_size=1024)
        assert rg1 is rg2, "same base_size must return cached object"

    def test_cache_different_base_size_different_object(self):
        get_cached_resolution_group.cache_clear()
        rg1 = get_cached_resolution_group(base_size=1024)
        rg2 = get_cached_resolution_group(base_size=768)
        assert rg1 is not rg2, "different base_size must return different objects"

    def test_cache_size_limit(self):
        """Cache is bounded at 4 entries (``maxsize=4``)."""
        get_cached_resolution_group.cache_clear()
        for bs in [512, 768, 1024, 1280, 1536]:
            get_cached_resolution_group(base_size=bs)
        info = get_cached_resolution_group.cache_info()
        assert info.currsize <= 4, f"cache grew to {info.currsize}, max is 4"
        # All 5 unique base_size values caused cache misses
        assert info.misses >= 5, f"expected >=5 misses for 5 unique base_size values, got {info.misses}"

    def test_resolution_indexing_works(self):
        """Indices 0..len(rg)-1 must be accessible."""
        rg = get_cached_resolution_group(base_size=1024)
        for i in range(len(rg)):
            r = rg[i]
            assert isinstance(r, Resolution)
            assert r.h > 0 and r.w > 0

    @pytest.mark.parametrize("base_size", [512, 1024])
    def test_ratio_index_roundtrip_via_group_length(self, base_size):
        """Every ratio index in [0, len(rg)) must resolve to a valid
        Resolution with height/width within bounds."""
        rg = get_cached_resolution_group(base_size=base_size)
        for i in range(len(rg)):
            r = rg[i]
            assert isinstance(r.height, int) and r.height > 0
            assert isinstance(r.width, int) and r.width > 0


# ---------------------------------------------------------------------------
# ar2diffusion — integration tests
# ---------------------------------------------------------------------------


# Token IDs matching the real prompt_utils values
_RATIO_0 = 128044
_RATIO_32 = 128076
_RATIO_33 = 130103
_RATIO_36 = 130106


def _fake_ar_output(
    *,
    cumulative_token_ids: list[int] | None = None,
    text: str = "",
) -> MagicMock:
    """Minimal stub for a vLLM ``RequestOutput`` with one completion."""
    choice = MagicMock()
    choice.cumulative_token_ids = cumulative_token_ids or []
    choice.text = text
    output = MagicMock()
    output.outputs = [choice]
    return output


class TestAr2DiffusionIntegration:
    """Verify that ``ar2diffusion`` calls ``get_cached_resolution_group``
    and obtains a correctly configured ``ResolutionGroup`` whose indices map
    to the expected training-bucket or extra-resolution sizes."""

    def _call_ar2diffusion(
        self,
        token_ids: list[int],
        prompt_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Invoke ``ar2diffusion`` with a single fake AR output.

        Returns the diffusion request payload so callers can assert on the
        resolved height/width.
        """
        from vllm_omni.model_executor.stage_input_processors.hunyuan_image3 import (
            ar2diffusion,
        )

        pk = dict(prompt_kwargs or {})
        pk.setdefault("prompt", "a beautiful sunset")
        pk.setdefault("height", 1024)
        pk.setdefault("width", 1024)
        pk.setdefault("image_base_size", 1024)

        # Build a prompt-like object that behaves like the real OmniTokensPrompt
        class _FakePrompt:
            def __init__(self, d):
                self._d = d

            def __getitem__(self, key):
                return self._d[key]

            def get(self, key, default=None):
                return self._d.get(key, default)

            def _asdict(self):
                return dict(self._d)

        prompt = _FakePrompt(pk)
        ar_output = _fake_ar_output(cumulative_token_ids=token_ids)
        result = ar2diffusion(
            source_outputs=[ar_output],
            prompt=[prompt],
            requires_multimodal_data=False,
        )
        return result

    # -- ratio token ↔ resolution resolution ---------------------------------

    @pytest.mark.parametrize(
        "ratio_idx,expected_hw",
        [
            # Training buckets
            (0, (512, 2048)),
            (16, (1024, 1024)),
            (32, (2048, 512)),
            # Non-square extras
            (33, (1024, 768)),
            (34, (1280, 720)),
            (35, (768, 1024)),
            (36, (720, 1280)),
        ],
    )
    def test_ratio_index_resolves_to_correct_size(self, ratio_idx, expected_hw):
        """When the AR emits ``<img_ratio_N>``, the DiT input must carry
        the training-bucket width/height for that index."""
        rg = get_cached_resolution_group(base_size=1024)
        reso = rg[ratio_idx]
        assert (reso.height, reso.width) == expected_hw, (
            f"ratio_idx {ratio_idx}: expected {expected_hw}, got ({reso.height}, {reso.width})"
        )

    @pytest.mark.parametrize("ratio_idx", [0, 16, 32, 33, 34, 35, 36])
    def test_ar2diffusion_extracts_height_width(self, ratio_idx):
        """End-to-end: simulate AR output containing ``<img_ratio_{idx}>``
        and assert the DiT inputs carry the expected size."""
        rg = get_cached_resolution_group(base_size=1024)
        expected_h = rg[ratio_idx].height
        expected_w = rg[ratio_idx].width

        token_id = _RATIO_0 + ratio_idx if ratio_idx <= 32 else _RATIO_33 + (ratio_idx - 33)
        result = self._call_ar2diffusion(
            token_ids=[150000, 150001, token_id],
            prompt_kwargs={"height": 1024, "width": 1024, "image_base_size": 1024},
        )
        assert result is not None
        diffusion_input = result
        assert diffusion_input["height"] == expected_h, (
            f"ratio_idx={ratio_idx}: expected height {expected_h}, got {diffusion_input['height']}"
        )
        assert diffusion_input["width"] == expected_w, (
            f"ratio_idx={ratio_idx}: expected width {expected_w}, got {diffusion_input['width']}"
        )

    def test_no_ratio_token_falls_back_to_prompt_size(self):
        """If no ratio token is present, height/width from the prompt
        are kept unchanged."""
        result = self._call_ar2diffusion(
            token_ids=[150000, 150001],  # no ratio tokens
            prompt_kwargs={"height": 768, "width": 1024, "image_base_size": 1024},
        )
        assert result is not None
        diffusion_input = result
        assert diffusion_input["height"] == 768
        assert diffusion_input["width"] == 1024

    def test_out_of_range_ratio_index_falls_back(self):
        """If ratio_idx ≥ len(reso_group), ar2diffusion must keep the
        prompt size (fallback path)."""
        with patch(
            "vllm_omni.model_executor.stage_input_processors.hunyuan_image3._extract_ratio_index",
            return_value=999,  # out of range
        ):
            result = self._call_ar2diffusion(
                token_ids=[150000],
                prompt_kwargs={"height": 512, "width": 512, "image_base_size": 1024},
            )
        assert result is not None
        assert result["height"] == 512
        assert result["width"] == 512

    # -- cache behaviour -----------------------------------------------------

    def test_ar2diffusion_uses_cached_group(self):
        """Each invocation of ``ar2diffusion`` calls
        ``get_cached_resolution_group``, which returns a shared cached object."""
        get_cached_resolution_group.cache_clear()
        rg_before = get_cached_resolution_group(base_size=1024)

        token_id = _RATIO_0 + 16  # <img_ratio_16> = 1024×1024
        self._call_ar2diffusion(
            token_ids=[token_id],
            prompt_kwargs={"height": 1024, "width": 1024, "image_base_size": 1024},
        )

        rg_after = get_cached_resolution_group(base_size=1024)
        assert rg_before is rg_after, "get_cached_resolution_group should return the same cached object"

    def test_correct_reso_group_used_for_base_size(self):
        """Verify ``ar2diffusion`` constructs or retrieves the
        ResolutionGroup for the correct ``image_base_size``."""
        get_cached_resolution_group.cache_clear()

        # First request with a non-default base_size
        token_id = _RATIO_0  # <img_ratio_0>
        self._call_ar2diffusion(
            token_ids=[token_id],
            prompt_kwargs={"height": 512, "width": 512, "image_base_size": 512},
        )

        # The cached group for base_size=512 should exist
        rg512 = get_cached_resolution_group(base_size=512)
        assert rg512.base_size == 512

        # And base_size=1024 should also be separately accessible
        rg1024 = get_cached_resolution_group(base_size=1024)
        assert rg1024.base_size == 1024
        assert rg512 is not rg1024


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_resolution_string_parsing(self):
        """Resolution accepts both "HxW" strings and (h, w) tuples."""
        r1 = Resolution("512x2048")
        assert r1.h == 512 and r1.w == 2048

        r2 = Resolution(768, 1024)
        assert r2.h == 768 and r2.w == 1024

        r3 = Resolution(1024)  # single int → square
        assert r3.h == 1024 and r3.w == 1024

    def test_resolution_extra_res_initially_empty(self):
        r = Resolution(1024, 1024)
        assert r.extra_res == set()
        assert r.match(512, 512) == (1024, 1024)

    def test_resolution_append_deduplicates(self):
        """Appending the same resolution twice is a no-op on the set."""
        r = Resolution(1024, 1024)
        r.append(Resolution(512, 512))
        r.append(Resolution(512, 512))
        assert len(r.extra_res) == 1

    def test_all_final_resolutions_are_16_aligned(self):
        """Every reachable resolution (including extras) must be
        divisible by the VAE down-sample factor of 16."""
        rg = _build_extra_group()
        for h, w in _reso_set(rg):
            assert h % 16 == 0, f"height {h} not 16-aligned in {h}×{w}"
            assert w % 16 == 0, f"width {w} not 16-aligned in {h}×{w}"

    def test_ratio_array_length_matches_data(self):
        rg = _build_extra_group()
        assert len(rg.ratio) == len(rg.data)

    def test_resolution_repr(self):
        r = Resolution(1024, 1024)
        rep = repr(r)
        assert "1024x1024" in rep

        r2 = Resolution(1024, 1024)
        r2.extra_res.add((512, 512))
        rep2 = repr(r2)
        assert "512x512" in rep2
