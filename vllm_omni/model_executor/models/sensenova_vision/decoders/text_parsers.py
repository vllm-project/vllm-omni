"""Structured-text parsers for SenseNova-Vision detection/keypoint outputs.

Ports the detection parsing helpers from the official SenseNova-Vision
``utils/parsing_output.py``. Coordinates are kept in normalized space and
clipped to ``[0, 0.999]`` (matching ``clip_normalized_coords``); labels are
normalized with ``normalize_category`` for benchmark matching.
"""

from __future__ import annotations

from typing import Any

__all__ = ["parse_bbox", "parse_points", "parse_keypoints"]


def _normalize_category(category: str) -> str:
    """Normalize a category name for benchmark matching."""
    return (
        category.replace("-merged", "")
        .replace("-other", "")
        .replace("-stuff", "")
        .replace("-negative", "")
        .replace("-", " ")
        .lower()
        .strip()
    )


def _clip_normalized_coords(coords) -> list[float]:
    """Clip normalized coordinates into the half-open visual range."""
    return [max(0.0, min(0.999, float(coord))) for coord in coords]


def parse_bbox(text: str) -> dict[str, list[list[float]]]:
    """Parse detection text in ``<p>label</p><bbox>[x0,y0,x1,y1]</bbox>`` form.

    Returns ``{label: [[x0, y0, x1, y1], ...]}`` with coordinates normalized
    and clipped to ``[0, 0.999]``.
    """
    results: dict[str, list[list[float]]] = {}
    source = str(text or "").strip().rstrip(" .\n")
    for part in source.split("<p>")[1:]:
        if "</p>" not in part:
            continue
        cat_end = part.find("</p>")
        category = part[:cat_end].strip()
        rest = part[cat_end + len("</p>") :]

        bboxes = []
        while "<bbox>[" in rest:
            start = rest.find("<bbox>[")
            end = rest.find("]</bbox>")
            if start == -1 or end == -1:
                break
            coord_str = rest[start + len("<bbox>[") : end]
            rest = rest[end + len("]</bbox>") :]
            coords = [coord.strip() for coord in coord_str.split(",")]
            if len(coords) == 4:
                try:
                    bboxes.append(_clip_normalized_coords(coords))
                except ValueError:
                    continue

        if bboxes:
            results[_normalize_category(category)] = bboxes
    return results


def parse_points(text: str) -> dict[str, list[list[float]]]:
    """Parse point detection text in ``<p>label</p><point>[x,y]</point>`` form.

    Returns ``{label: [[x, y], ...]}`` with coordinates normalized and clipped
    to ``[0, 0.999]``.
    """
    results: dict[str, list[list[float]]] = {}
    source = str(text or "").strip().rstrip(" .\n")
    for part in source.split("<p>")[1:]:
        if "</p>" not in part:
            continue
        cat_end = part.find("</p>")
        category = part[:cat_end].strip()
        rest = part[cat_end + len("</p>") :]

        points = []
        while "<point>[" in rest:
            start = rest.find("<point>[")
            end = rest.find("]</point>")
            if start == -1 or end == -1:
                break
            coord_str = rest[start + len("<point>[") : end]
            rest = rest[end + len("]</point>") :]
            coords = [coord.strip() for coord in coord_str.split(",")]
            if len(coords) == 2:
                try:
                    points.append(_clip_normalized_coords(coords))
                except ValueError:
                    continue

        if points:
            results[_normalize_category(category)] = points
    return results


def parse_keypoints(text: str) -> dict[str, list[dict[str, Any]]]:
    """Parse keypoint output grouped by category and instance.

    Expected structure:
    ``<p>person</p><bbox>[...]</bbox>left shoulder<kpt>[x,y]</kpt>...``.
    Optional ``<ins>...</ins>`` tags are ignored. Invisible keypoints encoded
    as ``<kpt>unvisible</kpt>`` are stored as ``[-1, -1]``.

    Returns ``{category: [{bbox?, keypoints: {name: [x, y]}}]}``.
    """
    results: dict[str, list[dict[str, Any]]] = {}
    source = str(text or "").strip().rstrip(" .\n")
    for part in source.split("<p>")[1:]:
        if "</p>" not in part:
            continue

        cat_end = part.find("</p>")
        category = part[:cat_end].strip()
        rest = part[cat_end + len("</p>") :].replace("<ins>", "").replace("</ins>", "")

        bbox = None
        if "<bbox>[" in rest:
            start = rest.find("<bbox>[")
            end = rest.find("]</bbox>")
            if start != -1 and end != -1:
                coord_str = rest[start + len("<bbox>[") : end]
                coords = [coord.strip() for coord in coord_str.split(",")]
                if len(coords) == 4:
                    try:
                        bbox = _clip_normalized_coords(coords)
                    except ValueError:
                        pass
                rest = rest[:start] + rest[end + len("]</bbox>") :]

        keypoints: dict[str, list[float]] = {}
        while "<kpt>" in rest:
            start = rest.find("<kpt>")
            end = rest.find("</kpt>", start)
            if start == -1 or end == -1:
                break

            keypoint_name = rest[:start].strip()
            content = rest[start + len("<kpt>") : end].strip()
            rest = rest[end + len("</kpt>") :]

            if not keypoint_name:
                continue
            if content == "unvisible":
                keypoints[keypoint_name] = [-1.0, -1.0]
                continue
            try:
                coords = [coord.strip() for coord in content.strip("[]").split(",")]
                if len(coords) == 2:
                    keypoints[keypoint_name] = _clip_normalized_coords(coords)
            except ValueError:
                continue

        if category:
            cat_clean = _normalize_category(category)
            instance: dict[str, Any] = {"keypoints": keypoints}
            if bbox is not None:
                instance["bbox"] = bbox
            results.setdefault(cat_clean, []).append(instance)
    return results
