# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pytest marks and decorators for hardware / resource selection (CUDA, ROCm, …).

Module import is stdlib-only so ``tools/pre_commit/check_test_marks.py`` can load
this file without pytest or vllm. Decorator helpers import pytest lazily.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

    # CUDA (and only CUDA) may list several SKUs so one item matches H100 nightly
    # and B200 mirror collection: ``res={"cuda": ["H100", "B200"]}``.
    # Kept under TYPE_CHECKING so Python 3.9 agents do not evaluate ``str | list``.
    SkuSpec = str | list[str] | tuple[str, ...]

# Marker description tags in ``pyproject.toml`` ``tool.pytest.ini_options.markers``.
# Example: ``"H100: [hardware-resource] [cuda] Tests that require H100 GPU"``.
_HARDWARE_RESOURCE_MARKER_TAG = "[hardware-resource]"
_HARDWARE_CARDS_MARKER_TAG = "[hardware-cards]"
_HARDWARE_PLATFORM_MARKER_TAG = "[hardware-platform]"
_CI_LEVEL_MARKER_TAG = "[ci-level]"
# On ``[hardware-platform]`` lines (not SKUs): also attach ``pytest.mark.gpu``.
_GPU_DEVICE_CLASS_TAG = "[gpu]"
_CARDS_MARK_NAME_RE = re.compile(r"^cards_(\d+)$")
# Indirect fixture in ``tests.helpers.fixtures.runtime``: mixed-count ``@hardware_test``
# parametrizes it so each collected item carries one platform's SKU and ``cards_{n}``.
HARDWARE_MARK_NORMALIZATION_FIXTURE = "_normalized_hardware_marks"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_pyproject() -> dict:
    path = _repo_root() / "pyproject.toml"
    try:
        try:
            import tomllib
        except ModuleNotFoundError:  # Python < 3.11
            import tomli as tomllib  # type: ignore[no-redef]

        with path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


@lru_cache(maxsize=16)
def _marker_names_with_tag(tag: str) -> frozenset[str]:
    """Return pytest marker names whose description contains ``tag``."""
    data = _load_pyproject()
    entries = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("markers", [])
    if not entries:
        text = (_repo_root() / "pyproject.toml").read_text(encoding="utf-8")
        return frozenset(
            re.findall(
                rf'^\s*"([A-Za-z0-9_]+)\s*:.*{re.escape(tag)}',
                text,
                flags=re.M,
            )
        )

    names: set[str] = set()
    for entry in entries:
        text = str(entry)
        if tag not in text:
            continue
        name = text.split(":", 1)[0].strip()
        if name:
            names.add(name)
    return frozenset(names)


@lru_cache(maxsize=1)
def get_hardware_mark_list() -> frozenset[str]:
    """Return hardware SKU marker names tagged ``[hardware-resource]``."""
    return _marker_names_with_tag(_HARDWARE_RESOURCE_MARKER_TAG)


@lru_cache(maxsize=1)
def get_supported_card_counts() -> frozenset[int]:
    """Return ``n`` for each ``cards_{n}`` marker tagged ``[hardware-cards]``."""
    counts: set[int] = set()
    for name in _marker_names_with_tag(_HARDWARE_CARDS_MARKER_TAG):
        matched = _CARDS_MARK_NAME_RE.fullmatch(name)
        if matched:
            counts.add(int(matched.group(1)))
    return frozenset(counts)


@lru_cache(maxsize=1)
def get_supported_platforms() -> frozenset[str]:
    """Return platform marker names tagged ``[hardware-platform]``.

    Includes ``cpu`` / ``gpu`` (device-class marks) and accelerator keys.
    ``hardware_marks`` / ``hardware_test`` ``res`` dicts reject ``cpu`` / ``gpu``.
    """
    return _marker_names_with_tag(_HARDWARE_PLATFORM_MARKER_TAG)


@lru_cache(maxsize=1)
def get_level_markers() -> frozenset[str]:
    """Return CI level marker names tagged ``[ci-level]`` (``core_model``, ``slow``, …)."""
    return _marker_names_with_tag(_CI_LEVEL_MARKER_TAG)


@lru_cache(maxsize=8)
def get_skus_for_platform(platform: str) -> frozenset[str]:
    """SKU names tagged ``[hardware-resource]`` and ``[platform]`` (e.g. ``[cuda]``)."""
    return get_hardware_mark_list() & _marker_names_with_tag(f"[{platform}]")


def _require_sku(platform: str, res: str) -> None:
    allowed = get_skus_for_platform(platform)
    if res not in allowed:
        supported = ", ".join(sorted(allowed)) or f"(none tagged [hardware-resource] [{platform}])"
        raise ValueError(f"Invalid {platform} resource type: {res}. Supported: {supported}")


def _as_sku_list(res: SkuSpec) -> list[str]:
    skus = [res] if isinstance(res, str) else list(res)
    if not skus:
        raise ValueError("resource list must not be empty")
    return skus


def _cards_mark(num_cards: int) -> pytest.MarkDecorator:
    """Return ``cards_{n}`` so collection can filter with ``-m cards_2`` (etc.).

    Multi-card selection is ``not cards_1`` (or an explicit ``cards_2`` / ``cards_4``).
    """
    import pytest

    if not isinstance(num_cards, int) or isinstance(num_cards, bool) or num_cards < 1:
        raise ValueError(f"num_cards must be a positive int, got {num_cards!r}")
    if num_cards not in get_supported_card_counts():
        counts = get_supported_card_counts()
        supported = ", ".join(f"cards_{n}" for n in sorted(counts))
        supported = supported or "(none tagged [hardware-cards])"
        raise ValueError(f"num_cards={num_cards} has no registered marker. Supported: {supported}.")
    return getattr(pytest.mark, f"cards_{num_cards}")


def _cuda_marks(*, res: SkuSpec, num_cards: int):
    import pytest
    from vllm.platforms import current_platform

    skus = _as_sku_list(res)
    for sku in skus:
        _require_sku("cuda", sku)
    marks = [getattr(pytest.mark, sku) for sku in skus]
    marks.extend([pytest.mark.cuda, _cards_mark(num_cards)])
    if num_cards == 1 or not current_platform.is_cuda():
        return marks
    return marks + [
        pytest.mark.skipif(
            current_platform.device_count() < num_cards,
            reason=f"Need at least {num_cards} CUDA GPUs to run the test.",
        )
    ]


def _rocm_marks(*, res: str, num_cards: int):
    import pytest

    _require_sku("rocm", res)
    return [getattr(pytest.mark, res), pytest.mark.rocm, _cards_mark(num_cards)]


def _xpu_marks(*, res: str, num_cards: int):
    import pytest
    from vllm.platforms import current_platform

    _require_sku("xpu", res)
    marks = [getattr(pytest.mark, res), pytest.mark.xpu, _cards_mark(num_cards)]
    if num_cards == 1 or not current_platform.is_xpu():
        return marks
    return marks + [
        pytest.mark.skipif(
            current_platform.device_count() < num_cards,
            reason=f"Need at least {num_cards} XPUs to run the test.",
        )
    ]


def _musa_marks(*, res: str, num_cards: int):
    import pytest

    _require_sku("musa", res)
    return [getattr(pytest.mark, res), pytest.mark.musa, _cards_mark(num_cards)]


@lru_cache(maxsize=1)
def _gpu_res_platforms() -> frozenset[str]:
    """Accelerator platforms tagged ``[hardware-platform] [gpu]`` (not the ``gpu`` mark)."""
    return (get_supported_platforms() & _marker_names_with_tag(_GPU_DEVICE_CLASS_TAG)) - {"gpu"}


def _npu_marks(*, res: str, num_cards: int):
    import pytest

    _require_sku("npu", res)
    return [pytest.mark.npu, getattr(pytest.mark, res), _cards_mark(num_cards)]


def _res_platforms() -> frozenset[str]:
    """Accelerator keys allowed in ``hardware_test(res=...)`` / ``hardware_marks(res=...)``.

    A ``[hardware-platform]`` name is a ``res`` key only if some SKU is tagged
    with that platform (``[cuda]``, ``[npu]``, …). ``cpu`` / ``gpu`` have no SKUs.
    """
    return frozenset(p for p in get_supported_platforms() if get_skus_for_platform(p))


def _normalize_num_cards(res: dict[str, SkuSpec], num_cards: int | dict[str, int]) -> dict[str, int]:
    allowed = _res_platforms()
    device_class = get_supported_platforms() - allowed
    for platform in res:
        if platform in device_class:
            raise ValueError(
                f"{platform!r} is not a res dict key; use pytest.mark.{platform} "
                f"or an accelerator key ({', '.join(sorted(allowed))})."
            )
        if platform not in allowed:
            supported = ", ".join(sorted(allowed)) or "(none)"
            raise ValueError(f"Unsupported platform: {platform}. res keys: {supported}.")
    if isinstance(num_cards, int) and not isinstance(num_cards, bool):
        return {platform: num_cards for platform in res}
    if not isinstance(num_cards, dict):
        raise ValueError(f"num_cards must be a positive int or dict, got {num_cards!r}")
    num_cards_dict = dict(num_cards)
    for platform in num_cards_dict:
        if platform not in res:
            raise ValueError(f"Platform '{platform}' in num_cards but not in res.")
    for platform in res:
        num_cards_dict.setdefault(platform, 1)
    return num_cards_dict


def _marks_for_platform(platform: str, resource: SkuSpec, num_cards: int) -> list[pytest.MarkDecorator]:
    import pytest

    if platform == "cuda":
        marks = _cuda_marks(res=resource, num_cards=num_cards)
    else:
        if not isinstance(resource, str):
            raise ValueError(f"{platform} resource must be a string, got {resource!r}")
        builders = {
            "rocm": _rocm_marks,
            "xpu": _xpu_marks,
            "musa": _musa_marks,
            "npu": _npu_marks,
        }
        builder = builders.get(platform)
        if builder is None:
            raise ValueError(f"Unsupported platform: {platform}")
        marks = builder(res=resource, num_cards=num_cards)
    if platform in _gpu_res_platforms():
        return [pytest.mark.gpu] + marks
    return marks


def _skipif_not_platform(platform: str) -> pytest.MarkDecorator:
    import pytest

    from vllm_omni.platforms import current_omni_platform

    checkers = {
        "cuda": current_omni_platform.is_cuda,
        "rocm": current_omni_platform.is_rocm,
        "xpu": current_omni_platform.is_xpu,
        "npu": current_omni_platform.is_npu,
        "musa": current_omni_platform.is_musa,
    }
    return pytest.mark.skipif(not checkers[platform](), reason=f"Requires {platform} platform")


def _apply_marks(func, marks: list[pytest.MarkDecorator]):
    for mark in reversed(marks):
        func = mark(func)
    return func


def hardware_marks(*, res: dict[str, SkuSpec], num_cards: int | dict[str, int] = 1):
    """Return marks for a **single** pytest item.

    Different per-platform ``num_cards`` cannot share one item: ``cards_2`` and
    ``cards_4`` would both match ``-m cards_2`` / ``-m cards_4``. Use
    ``@hardware_test`` (one collected variant per platform) or call this once
    per platform and attach each list to its own ``pytest.param``.
    """
    num_cards_dict = _normalize_num_cards(res, num_cards)
    unique_counts = set(num_cards_dict.values())
    if len(res) > 1 and len(unique_counts) > 1:
        raise ValueError(
            "hardware_marks() cannot attach different per-platform num_cards to one pytest item "
            f"(got {num_cards_dict}). Use @hardware_test, which collects one variant per platform, "
            "or call hardware_marks() once per platform."
        )

    all_marks: list[pytest.MarkDecorator] = []
    for platform, resource in res.items():
        all_marks.extend(_marks_for_platform(platform, resource, num_cards_dict[platform]))
    return all_marks


def hardware_test(*, res: dict[str, SkuSpec], num_cards: int | dict[str, int] = 1):
    """Apply hardware marks; split collected items when card counts differ by platform.

    A single item cannot carry both ``cards_4`` (CUDA) and ``cards_2`` (ROCm):
    ``-m "H100 and cards_4 and cuda"`` would otherwise also match
    ``-m "H100 and cards_2 and cuda"``. When ``num_cards`` differs across
    platforms, this decorator parametrizes one variant per platform so each
    item only has that platform's SKU and ``cards_{n}``.
    """
    import pytest

    num_cards_dict = _normalize_num_cards(res, num_cards)
    mixed_counts = len(res) > 1 and len(set(num_cards_dict.values())) > 1
    if not mixed_counts:
        all_marks = hardware_marks(res=res, num_cards=num_cards)

        def apply_union_marks(f):
            return _apply_marks(f, all_marks)

        return apply_union_marks

    params = []
    for platform, resource in res.items():
        marks = _marks_for_platform(platform, resource, num_cards_dict[platform])
        marks.append(_skipif_not_platform(platform))
        params.append(pytest.param(platform, marks=marks, id=platform))

    def apply_split_params(f):
        func = pytest.mark.usefixtures(HARDWARE_MARK_NORMALIZATION_FIXTURE)(f)
        return pytest.mark.parametrize(HARDWARE_MARK_NORMALIZATION_FIXTURE, params, indirect=True)(func)

    return apply_split_params
