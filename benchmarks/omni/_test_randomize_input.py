# SPDX-License-Identifier: Apache-2.0
"""Isolated tests for the --randomize-input feature in omni_dataset.

The local Python lacks PIL / aiohttp / tqdm, so we stub them before importing
the benchmark modules. These tests exercise pure dataset-construction logic:
prompt / input-resolution / image-content randomization, the input_image_size
bookkeeping field, reproducibility via seed, and the no-op behavior when the
flag is off (so legacy runs keep the exact same bot_task + shuffle sequence).
"""

from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


# --- Stub PIL before any benchmark import ----------------------------------
class _FakeImage:
    def __init__(self, mode: str, size: tuple[int, int], color: tuple[int, int, int]):
        self.mode = mode
        self.size = size
        self.color = color

    def save(self, path: str) -> None:
        # Write a tiny sentinel file so os.path.exists passes and the path can
        # be re-opened. Encode (w, h, color) so tests can verify per-request
        # uniqueness without a real PNG decoder.
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self.size[0]}x{self.size[1]}|{self.color[0]},{self.color[1]},{self.color[2]}")


class _FakeImageDraw:
    def __init__(self, img: _FakeImage):
        self.img = img

    def ellipse(self, bbox, fill=None):
        pass

    def rectangle(self, bbox, fill=None):
        pass

    def line(self, coords, fill=None, width=1):
        pass


# PIL.Image is a module exposing new() / open(); PIL.ImageDraw exposes Draw().
_pil = types.ModuleType("PIL")
_pil_image = types.ModuleType("PIL.Image")
_pil_image.new = staticmethod(lambda mode, size, color: _FakeImage(mode, size, color))
_pil_image.open = staticmethod(lambda path: _FakeImage("RGB", (0, 0), (0, 0, 0)))
_pil_image.Image = _FakeImage
_pil.Image = _pil_image
_pil_image_draw = types.ModuleType("PIL.ImageDraw")
_pil_image_draw.Draw = staticmethod(lambda img: _FakeImageDraw(img))
sys.modules["PIL"] = _pil
sys.modules["PIL.Image"] = _pil_image
sys.modules["PIL.ImageDraw"] = _pil_image_draw


# --- Stub aiohttp + tqdm for omni_backends import -------------------------
for _mod in ("aiohttp", "tqdm", "tqdm.asyncio", "numpy"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["aiohttp"].FormData = object  # type: ignore[attr-defined]
sys.modules["aiohttp"].ClientSession = object  # type: ignore[attr-defined]
sys.modules["aiohttp"].ClientResponse = object  # type: ignore[attr-defined]
sys.modules["tqdm"].tqdm = object  # type: ignore[attr-defined]
# omni_benchmark_serving imports numpy as np and tqdm.asyncio.tqdm; not needed
# here since we only test omni_dataset directly.

# Make the sibling diffusion dir importable (omni_backends imports backends).
_DIFFUSION = os.path.join(os.path.dirname(HERE), "diffusion")
if _DIFFUSION not in sys.path:
    sys.path.insert(0, _DIFFUSION)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import random  # noqa: E402

from omni_dataset import (  # noqa: E402  # noqa: E402
    _INPUT_IMAGE_COUNTER,
    DEFAULT_I2T_PROMPT,
    DEFAULT_IT2I_PROMPT,
    DEFAULT_T2I_PROMPT,
    INPUT_RESOLUTIONS,
    OmniConfig,
    TaskCounts,
    _draw_input_resolution,
    _draw_prompt,
    _generate_diverse_image,
    build_requests,
)


def _base_kwargs(**over):
    kw = dict(
        counts=TaskCounts(i2t=7, t2i=2, it2i=1),
        config=OmniConfig(),
        dataset="random",
        dataset_path=None,
        bot_task_weights={"recaption": 1.0, "think": 1.0, "think_recaption": 1.0},
        input_image=None,
        prompts={},
        chat_url="http://x/v1/chat/completions",
        images_edits_url="http://x/v1/images/edits",
        return_stage_metrics=False,
        it2i_endpoint="chat",
        seed=7,
        shuffle=True,
    )
    kw.update(over)
    return kw


def test_draw_prompt_picks_from_pool():
    rng = random.Random(7)
    seen_i2t = {_draw_prompt("i2t", rng) for _ in range(200)}
    seen_t2i = {_draw_prompt("t2i", rng) for _ in range(200)}
    seen_it2i = {_draw_prompt("it2i", rng) for _ in range(200)}
    # Drawing 200 times from a 10-item pool should hit most of them.
    assert len(seen_i2t) >= 8, seen_i2t
    assert len(seen_t2i) >= 8, seen_t2i
    assert len(seen_it2i) >= 8, seen_it2i
    # Prompts should vary in length (the pool is ordered short -> long).
    lens = sorted(len(p.split()) for p in seen_i2t)
    assert lens[0] < lens[-1], lens
    print("test_draw_prompt_picks_from_pool OK")


def test_draw_input_resolution_valid():
    rng = random.Random(11)
    for _ in range(100):
        w, h = _draw_input_resolution(rng)
        assert (w, h) in INPUT_RESOLUTIONS.values(), (w, h)
    # Over many draws, all three sizes should appear.
    draws = {_draw_input_resolution(rng) for _ in range(300)}
    assert draws == set(INPUT_RESOLUTIONS.values()), draws
    print("test_draw_input_resolution_valid OK")


def test_generate_diverse_image_unique_per_call():
    rng = random.Random(3)
    paths = [_generate_diverse_image(rng, 512, 512) for _ in range(5)]
    # Each call returns a distinct temp-file path.
    assert len(set(paths)) == 5, paths
    # Files exist on disk (the fake image writes a sentinel).
    for p in paths:
        assert os.path.exists(p), p
    # Contents differ (different random palette -> different sentinel string).
    contents = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            contents.append(f.read())
    assert len(set(contents)) == 5, contents
    # Clean up.
    for p in paths:
        os.remove(p)
    print("test_generate_diverse_image_unique_per_call OK")


def test_build_randomize_on_varies_inputs():
    reqs = build_requests(**_base_kwargs(randomize_input=True))
    assert len(reqs) == 10, len(reqs)
    # i2t (7) + it2i (1) carry input images; t2i (2) do not.
    with_img = [r for r in reqs if r.task_type in ("i2t", "it2i")]
    without_img = [r for r in reqs if r.task_type == "t2i"]
    assert len(with_img) == 8 and len(without_img) == 2

    # Input-image paths are unique per i2t/it2i request (no shared placeholder).
    img_paths = [r.image_paths[0] for r in with_img]
    assert all(p and os.path.exists(p) for p in img_paths), img_paths
    assert len(set(img_paths)) == 8, "input images should be unique per request"

    # input_image_size recorded for every i2t/it2i, in the allowed set.
    for r in with_img:
        assert r.input_image_size in INPUT_RESOLUTIONS, r.input_image_size
    for r in without_img:
        assert r.input_image_size is None

    # Prompts vary across all requests (length diversity across the whole list).
    prompts = [r.prompt for r in reqs]
    assert len(set(prompts)) >= 6, f"expected diverse prompts, got {len(set(prompts))} unique"
    # And they come from the random pools, not the fixed defaults.
    assert all(p != DEFAULT_I2T_PROMPT for p in [r.prompt for r in reqs if r.task_type == "i2t"])
    assert all(p != DEFAULT_T2I_PROMPT for p in [r.prompt for r in reqs if r.task_type == "t2i"])

    # Clean up generated temp images.
    for p in img_paths:
        if os.path.exists(p):
            os.remove(p)
    print("test_build_randomize_on_varies_inputs OK")


def test_build_randomize_off_legacy_behavior():
    """With randomize_input=False, no input_image_size, single placeholder, no
    generated temp images, and prompts are the fixed defaults."""
    _INPUT_IMAGE_COUNTER["n"] = 0
    reqs = build_requests(**_base_kwargs(randomize_input=False))
    # No input_image_size recorded.
    assert all(r.input_image_size is None for r in reqs)
    # All i2t/it2i share the SAME placeholder path.
    with_img = [r for r in reqs if r.task_type in ("i2t", "it2i")]
    paths = {r.image_paths[0] for r in with_img}
    assert len(paths) == 1, f"expected single shared placeholder, got {paths}"
    # No diverse input images were generated (counter untouched).
    assert _INPUT_IMAGE_COUNTER["n"] == 0, _INPUT_IMAGE_COUNTER["n"]
    # Prompts are the fixed defaults.
    assert all(r.prompt == DEFAULT_I2T_PROMPT for r in reqs if r.task_type == "i2t")
    assert all(r.prompt == DEFAULT_T2I_PROMPT for r in reqs if r.task_type == "t2i")
    assert all(r.prompt == DEFAULT_IT2I_PROMPT for r in reqs if r.task_type == "it2i")
    print("test_build_randomize_off_legacy_behavior OK")


def test_reproducible_same_seed():
    def build():
        # Reset the per-process image-id counter so two in-process builds
        # produce identical temp-file paths (mirrors fresh-process runs).
        _INPUT_IMAGE_COUNTER["n"] = 0
        return build_requests(**_base_kwargs(randomize_input=True))

    a = build()
    b = build()
    assert [r.prompt for r in a] == [r.prompt for r in b], "prompts differ across runs"
    assert [r.input_image_size for r in a] == [r.input_image_size for r in b], "sizes differ"
    a_imgs = [r.image_paths[0] for r in a if r.image_paths]
    b_imgs = [r.image_paths[0] for r in b if r.image_paths]
    assert a_imgs == b_imgs, "paths differ"
    # Clean up.
    for r in a:
        if r.image_paths and os.path.exists(r.image_paths[0]):
            os.remove(r.image_paths[0])
    # The second build created its own temp files; clean those too.
    for r in b:
        if r.image_paths and os.path.exists(r.image_paths[0]):
            os.remove(r.image_paths[0])
    print("test_reproducible_same_seed OK")


def test_randomize_off_preserves_legacy_sequence():
    """The bot_task + shuffle sequence when randomize_input=False must match a
    build that never knew the flag existed (no extra rng draws). We approximate
    that by comparing two off-builds with different seeds and confirming the
    it2i bot_task assignment + shuffle order are seed-determined and stable
    across re-runs (reproducibility), which is the property we must not break."""
    r1 = build_requests(**_base_kwargs(seed=7, randomize_input=False))
    r2 = build_requests(**_base_kwargs(seed=7, randomize_input=False))
    assert [r.bot_task for r in r1 if r.task_type == "it2i"] == [r.bot_task for r in r2 if r.task_type == "it2i"]
    assert [r.task_type for r in r1] == [r.task_type for r in r2]  # shuffle order stable
    # A different seed changes the shuffle order (sanity: shuffle actually runs).
    r3 = build_requests(**_base_kwargs(seed=99, randomize_input=False))
    assert [r.task_type for r in r1] != [r.task_type for r in r3] or len(r1) == 1
    print("test_randomize_off_preserves_legacy_sequence OK")


def test_randomize_skipped_for_custom_dataset():
    """randomize_input is a no-op for dataset='custom' (rows win)."""
    import json
    import tempfile

    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for t in ("i2t", "t2i", "it2i"):
        tmp.write(json.dumps({"task": t, "prompt": f"custom-{t}-prompt"}) + "\n")
    tmp.close()
    try:
        reqs = build_requests(**_base_kwargs(dataset="custom", dataset_path=tmp.name, randomize_input=True))
        # Custom prompts win (not from the random pools).
        assert all(r.prompt.startswith("custom-") for r in reqs), [r.prompt for r in reqs]
        # No input_image_size recorded (custom path doesn't set it).
        assert all(r.input_image_size is None for r in reqs)
    finally:
        os.remove(tmp.name)
    print("test_randomize_skipped_for_custom_dataset OK")


def test_no_shuffle_group_order_with_randomize():
    """randomize_input + --no-shuffle still respects group_order grouping."""
    reqs = build_requests(**_base_kwargs(shuffle=False, group_order=("t2i", "i2t", "it2i"), randomize_input=True))
    types_seq = [r.task_type for r in reqs]
    # All t2i first, then all i2t, then it2i.
    assert types_seq == ["t2i"] * 2 + ["i2t"] * 7 + ["it2i"] * 1, types_seq
    # Clean up.
    for r in reqs:
        if r.image_paths and os.path.exists(r.image_paths[0]):
            os.remove(r.image_paths[0])
    print("test_no_shuffle_group_order_with_randomize OK")


def main() -> None:
    test_draw_prompt_picks_from_pool()
    test_draw_input_resolution_valid()
    test_generate_diverse_image_unique_per_call()
    test_build_randomize_on_varies_inputs()
    test_build_randomize_off_legacy_behavior()
    test_reproducible_same_seed()
    test_randomize_off_preserves_legacy_sequence()
    test_randomize_skipped_for_custom_dataset()
    test_no_shuffle_group_order_with_randomize()
    print("\nAll randomize-input tests passed.")


if __name__ == "__main__":
    main()
