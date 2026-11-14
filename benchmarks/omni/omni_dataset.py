# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: E501
"""Dataset / request-list construction for the omni serving benchmark.

Builds a single shuffled list of ``OmniRequest`` spanning i2t / t2i / it2i in
the counts the caller asked for, with it2i ``bot_task`` sampled (weighted)
across recaption / think / think_recaption. Two sources:

* ``random`` (default): synthetic prompts + a placeholder image for i2t/it2i.
  Zero external dependencies — enough to drive the scheduler and collect
  latency / throughput stats.
* ``custom``: a JSONL file (``--dataset-path``) where each line declares its
  own ``task`` / ``prompt`` / ``image_path`` / ``bot_task`` / generation knobs.
  The per-task pools are sampled (cycling) up to the requested counts.

The shuffle is a true ``random.shuffle`` over the merged list (controlled by
``--seed``), so the three task types are interleaved rather than sent in
blocks — which is what exercises the DTPS scheduling path.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Any

from omni_backends import IT2I_BOT_TASKS, OmniRequest
from PIL import Image

# Default prompts for the random dataset. Overridable via CLI.
DEFAULT_I2T_PROMPT = "Describe this image in detail."
DEFAULT_T2I_PROMPT = "A cat sitting on a wooden bench, photorealistic, soft natural light."
DEFAULT_IT2I_PROMPT = "Add warm sunset lighting to this image and make the sky more dramatic."


@dataclass
class OmniConfig:
    """Generation knobs shared across t2i / it2i requests."""

    width: int | None = 1024
    height: int | None = 1024
    num_inference_steps: int | None = 50
    seed: int | None = None
    model: str = "default"


@dataclass
class TaskCounts:
    i2t: int = 0
    t2i: int = 0
    it2i: int = 0

    @property
    def total(self) -> int:
        return self.i2t + self.t2i + self.it2i


def parse_bot_task_weights(raw: str | None) -> dict[str, float]:
    """Parse ``recaption=2,think=1,think_recaption=1`` into a weight dict.

    Defaults to equal weight across the three thinking-intensity modes. Unknown
    task names raise so a typo doesn't silently drop coverage.
    """
    if not raw:
        return {t: 1.0 for t in IT2I_BOT_TASKS}

    weights: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid bot_task weight entry '{part}'; expected name=weight (e.g. think=2).")
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name not in IT2I_BOT_TASKS:
            raise ValueError(f"Unknown bot_task '{name}'; valid: {sorted(IT2I_BOT_TASKS)}.")
        try:
            w = float(value)
        except ValueError as e:
            raise ValueError(f"Invalid weight '{value}' for bot_task '{name}'.") from e
        if w < 0:
            raise ValueError(f"bot_task weight for '{name}' must be >= 0, got {w}.")
        weights[name] = w

    if not weights:
        return {t: 1.0 for t in IT2I_BOT_TASKS}
    # Fill any unmentioned task with 0 so it can be opted out explicitly.
    for t in IT2I_BOT_TASKS:
        weights.setdefault(t, 0.0)
    if sum(weights.values()) <= 0:
        raise ValueError("bot_task weights sum to 0; at least one must be > 0.")
    return weights


def allocate_bot_tasks(n: int, weights: dict[str, float], rng: random.Random) -> list[str]:
    """Distribute ``n`` bot_task slots *exactly* proportional to ``weights``.

    Uses the largest-remainder method (:func:`_allocate_weighted`) so the counts
    match the weights as closely as integers allow (e.g. ``n=10``, weights
    ``2:2:1`` -> ``4/4/2``, not a random ``6/3/1`` draw). The resulting list is
    shuffled with ``rng`` so the assignment to it2i positions is randomized —
    the global shuffle later re-interleaves everything, but this also keeps
    ``--no-shuffle`` runs from grouping all of one bot_task together.
    """
    return _allocate_weighted(n, weights, rng)


def sample_bot_tasks(n: int, weights: dict[str, float], rng: random.Random) -> list[str]:
    """Sample ``n`` bot_task names by weight *with replacement* (stochastic).

    Each of the ``n`` slots is drawn independently, so the realized counts only
    match the weights in expectation (e.g. ``n=10`` weights ``2:2:1`` may yield
    ``6/3/1``). Use :func:`allocate_bot_tasks` for exact proportional counts.
    """
    tasks = list(weights.keys())
    ws = [weights[t] for t in tasks]
    return rng.choices(tasks, weights=ws, k=n)


def draw_bot_tasks(n: int, weights: dict[str, float], rng: random.Random, sampling: str) -> list[str]:
    """Dispatch to :func:`allocate_bot_tasks` (``proportional``) or
    :func:`sample_bot_tasks` (``random``)."""
    if sampling == "random":
        return sample_bot_tasks(n, weights, rng)
    return allocate_bot_tasks(n, weights, rng)


# Allowed generation resolutions for t2i / it2i and their (width, height) pairs.
GEN_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    "1280x720": (1280, 720),
}


def parse_gen_resolution_weights(raw: str | None) -> dict[str, float] | None:
    """Parse ``512x512=2,1024x1024=2,1280x720=1`` into a weight dict.

    Returns ``None`` when ``raw`` is empty so the caller keeps the global
    ``--width`` / ``--height``. Unknown resolution strings raise so a typo
    doesn't silently fall back. Weights must be ``>= 0`` and sum ``> 0``;
    unmentioned resolutions are filled with ``0.0`` so they can be opted out.
    """
    if not raw:
        return None
    weights: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid resolution weight entry '{part}'; expected WxH=weight (e.g. 1024x1024=2).")
        name, value = part.split("=", 1)
        name = name.strip()
        value = value.strip()
        if name not in GEN_RESOLUTIONS:
            raise ValueError(f"Unknown resolution '{name}'; valid: {sorted(GEN_RESOLUTIONS)}.")
        try:
            w = float(value)
        except ValueError as e:
            raise ValueError(f"Invalid weight '{value}' for resolution '{name}'.") from e
        if w < 0:
            raise ValueError(f"resolution weight for '{name}' must be >= 0, got {w}.")
        weights[name] = w
    if not weights:
        return None
    for r in GEN_RESOLUTIONS:
        weights.setdefault(r, 0.0)
    if sum(weights.values()) <= 0:
        raise ValueError("resolution weights sum to 0; at least one must be > 0.")
    return weights


def _allocate_weighted(n: int, weights: dict[str, float], rng: random.Random) -> list[str]:
    """Distribute ``n`` slots exactly proportional to ``weights`` (largest-
    remainder), then shuffle the result with ``rng``.

    Shared by bot_task and resolution allocation: the counts match the weights
    as closely as integers allow (e.g. ``n=10``, weights ``2:2:1`` -> ``4/4/2``).
    The shuffle randomizes the assignment to positions — the global shuffle
    later re-interleaves everything, but this also keeps ``--no-shuffle`` runs
    from grouping all of one key together.
    """
    keys = list(weights.keys())
    ws = [max(0.0, float(weights[k])) for k in keys]
    total = sum(ws)
    if total <= 0:
        raise ValueError("weights sum to 0; at least one must be > 0.")
    raw = [w * n / total for w in ws]
    floors = [int(r) for r in raw]
    remainder = n - sum(floors)
    if remainder > 0:
        # Break ties by largest fractional remainder; equal fractions keep the
        # dict order (stable sort) so the allocation is deterministic per seed.
        order = sorted(range(len(keys)), key=lambda i: raw[i] - floors[i], reverse=True)
        for i in range(remainder):
            floors[order[i]] += 1
    result: list[str] = []
    for key, count in zip(keys, floors):
        result.extend([key] * count)
    rng.shuffle(result)
    return result


def allocate_resolutions(n: int, weights: dict[str, float], rng: random.Random) -> list[tuple[int, int]]:
    """Exact-proportional ``(width, height)`` list of length ``n`` per ``weights``."""
    names = _allocate_weighted(n, weights, rng)
    return [GEN_RESOLUTIONS[name] for name in names]


def sample_resolutions(n: int, weights: dict[str, float], rng: random.Random) -> list[tuple[int, int]]:
    """Independent weighted draws (with replacement) of ``(width, height)``."""
    names = list(weights.keys())
    ws = [weights[name] for name in names]
    drawn = rng.choices(names, weights=ws, k=n)
    return [GEN_RESOLUTIONS[name] for name in drawn]


def draw_resolutions(n: int, weights: dict[str, float], rng: random.Random, sampling: str) -> list[tuple[int, int]]:
    """Dispatch to :func:`allocate_resolutions` (``proportional``) or
    :func:`sample_resolutions` (``random``)."""
    if sampling == "random":
        return sample_resolutions(n, weights, rng)
    return allocate_resolutions(n, weights, rng)


def _placeholder_image_path() -> str:
    """A 512x512 RGB placeholder used when no input image is supplied."""
    path = os.path.join(tempfile.gettempdir(), "omni_benchmark_placeholder.png")
    if not os.path.exists(path):
        Image.new("RGB", (512, 512), (128, 160, 200)).save(path)
    return path


# Allowed input-image resolutions for i2t / it2i when --randomize-input is on
# (mirrors GEN_RESOLUTIONS so the input side spans the same size range).
INPUT_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "512x512": (512, 512),
    "1024x1024": (1024, 1024),
    "1280x720": (1280, 720),
}

# Diverse prompt pools per task type, ordered short -> long so that a uniform
# draw across a pool yields a length distribution. Used only when
# --randomize-input is on (the --i2t-prompt / --t2i-prompt / --it2i-prompt
# overrides and the custom dataset rows still win when randomization is off).
_RANDOM_I2T_PROMPTS: tuple[str, ...] = (
    "Describe this image.",
    "What is in this picture?",
    "List the main objects you see and their colors.",
    "Describe the scene, the lighting, and the overall mood of this image.",
    "What is happening here? Describe the subjects, the background, and any text visible in the image.",
    "Give a detailed description of this image: the subjects, their poses and expressions, the setting, the time of day, and the composition.",
    "Analyze this image thoroughly. Cover the foreground and background, the colors and textures, the style (photo or illustration), and any notable details a caption model should capture.",
    "Write a long, fine-grained caption for this image. Describe every region left to right, the objects and their attributes, the scene semantics, the lighting direction, and any artifacts or text. Then summarize the image in one sentence.",
    "Pretend you are describing this image to someone who cannot see it. Walk through the layout, the people or animals present and what they are doing, the environment and weather, the camera framing, and your best guess of where and when it was taken.",
    "Provide an exhaustive description: enumerate all objects with counts and colors, describe the spatial relations between them, note the style and quality of the image, identify any text or logos, comment on the lighting and shadows, and suggest a plausible caption and three relevant tags.",
)

_RANDOM_T2I_PROMPTS: tuple[str, ...] = (
    "A cat.",
    "A red apple on a table.",
    "A city street at night, neon signs.",
    "A futuristic robot standing in a rainy alley, cinematic lighting.",
    "A cozy cabin in a snowy forest at dusk, warm light from the windows, smoke rising from the chimney.",
    "A majestic dragon perched on a mountain peak at sunrise, dramatic clouds, highly detailed digital painting.",
    "A photorealistic portrait of an elderly fisherman with a weathered face, sitting by the harbor, soft golden-hour light, shallow depth of field.",
    "A surreal floating island with waterfalls cascading into the sky, lush vegetation, ancient ruins, fantasy concept art, volumetric lighting, intricate detail.",
    "A bustling medieval marketplace filled with vendors and townspeople, colorful tents, cobblestone streets, a castle in the background, painterly illustration, rich color palette, fine detail.",
    "An astronaut exploring a bioluminescent alien jungle at night, strange glowing plants, a crashed spaceship in the distance, two moons in the sky, cinematic sci-fi concept art, ultra-detailed, atmospheric perspective.",
)

_RANDOM_IT2I_PROMPTS: tuple[str, ...] = (
    "Make it night.",
    "Add a sunset glow.",
    "Convert this photo to a watercolor painting.",
    "Change the background to a beach and keep the subject unchanged.",
    "Add warm sunset lighting and make the sky more dramatic with orange and purple clouds.",
    "Turn this image into a Studio Ghibli style illustration while preserving the composition and subjects.",
    "Remove the person in the foreground and replace the background with a snowy mountain landscape, keep the lighting consistent.",
    "Make this image look like a 1980s film photograph with film grain, slightly faded colors, and a soft vignette around the edges.",
    "Extend the canvas to the left and right with more of the same room, keeping the perspective and lighting consistent with the original image.",
    "Transform this daytime street scene into a rainy evening: wet reflective pavement, puddles, neon reflections, fog, and people holding umbrellas, while keeping the buildings and signage identical.",
)


def _draw_prompt(task_type: str, rng: random.Random) -> str:
    """Uniformly draw a diverse prompt for ``task_type`` from the random pools.

    The pools are ordered short -> long, so a uniform draw realizes a spread of
    prompt lengths. Reproducible via the shared ``rng`` (seeded by --seed).
    """
    pool = {
        "i2t": _RANDOM_I2T_PROMPTS,
        "t2i": _RANDOM_T2I_PROMPTS,
        "it2i": _RANDOM_IT2I_PROMPTS,
    }[task_type]
    return rng.choice(pool)


def _draw_input_resolution(rng: random.Random) -> tuple[int, int]:
    """Uniformly draw an input-image (width, height) from INPUT_RESOLUTIONS."""
    name = rng.choice(list(INPUT_RESOLUTIONS.keys()))
    return INPUT_RESOLUTIONS[name]


def _generate_diverse_image(rng: random.Random, width: int, height: int) -> str:
    """Generate a unique varied-content RGB image and return its file path.

    Each call draws a random palette + a random set of shapes from ``rng``, so
    the content differs per request (reproducible via --seed) instead of every
    request reusing one solid-color placeholder. The image is written to a temp
    file keyed by an internal counter so repeated builds don't collide. Used
    only for i2t / it2i under --randomize-input.
    """
    from PIL import ImageDraw

    # Random but harmonized palette: a base color + a few accent colors.
    base = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))

    def _accent() -> tuple[int, int, int]:
        # Pick a color that contrasts with base so shapes stay visible.
        return (
            (base[0] + rng.randint(80, 180)) % 256,
            (base[1] + rng.randint(80, 180)) % 256,
            (base[2] + rng.randint(80, 180)) % 256,
        )

    img = Image.new("RGB", (width, height), base)
    draw = ImageDraw.Draw(img)

    # A handful of large filled shapes for broad structure.
    n_ellipses = rng.randint(3, 7)
    for _ in range(n_ellipses):
        x0 = rng.randint(0, width - 1)
        y0 = rng.randint(0, height - 1)
        x1 = rng.randint(x0, width)
        y1 = rng.randint(y0, height)
        draw.ellipse([x0, y0, max(x0 + 2, x1), max(y0 + 2, y1)], fill=_accent())

    n_rects = rng.randint(2, 6)
    for _ in range(n_rects):
        x0 = rng.randint(0, width - 1)
        y0 = rng.randint(0, height - 1)
        x1 = rng.randint(x0, width)
        y1 = rng.randint(y0, height)
        draw.rectangle([x0, y0, max(x0 + 2, x1), max(y0 + 2, y1)], fill=_accent())

    # A few thin lines for extra texture.
    n_lines = rng.randint(2, 5)
    for _ in range(n_lines):
        x0 = rng.randint(0, width - 1)
        y0 = rng.randint(0, height - 1)
        x1 = rng.randint(0, width - 1)
        y1 = rng.randint(0, height - 1)
        draw.line([x0, y0, x1, y1], fill=_accent(), width=max(1, min(width, height) // 200))

    path = os.path.join(tempfile.gettempdir(), f"omni_bench_input_{_next_input_image_id()}.png")
    img.save(path)
    return path


_INPUT_IMAGE_COUNTER = {"n": 0}


def _next_input_image_id() -> int:
    """Monotonic id for generated input-image temp filenames (build-local)."""
    n = _INPUT_IMAGE_COUNTER["n"]
    _INPUT_IMAGE_COUNTER["n"] = n + 1
    return n


def _resolve_image_path(item: dict[str, Any], fallback: str | None) -> str | None:
    """Pick the input image for an i2t/it2i item: explicit path, else fallback."""
    if item.get("image_path"):
        p = str(item["image_path"])
        if not os.path.exists(p):
            raise ValueError(f"Image file not found: {p}")
        return p
    if item.get("image_paths"):
        ps = item["image_paths"]
        if isinstance(ps, str):
            ps = [ps]
        p = str(ps[0])
        if not os.path.exists(p):
            raise ValueError(f"Image file not found: {p}")
        return p
    return fallback


def _load_custom_pools(path: str) -> dict[str, list[dict[str, Any]]]:
    """Load a custom JSONL file, bucketing rows by their ``task`` field."""
    if not path.endswith(".jsonl"):
        raise ValueError("Custom dataset must be a JSONL file (--dataset-path).")
    if not os.path.exists(path):
        raise ValueError(f"Dataset file not found: {path}")

    pools: dict[str, list[dict[str, Any]]] = {"i2t": [], "t2i": [], "it2i": []}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e}") from e
            task = str(row.get("task", "")).strip().lower()
            if task not in pools:
                raise ValueError(f"Line {lineno}: task must be one of i2t/t2i/it2i, got '{task}'.")
            if "prompt" not in row:
                raise ValueError(f"Line {lineno}: missing required 'prompt' field.")
            pools[task].append(row)
    return pools


def _cycle(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    """Take ``n`` items from ``pool`` by cycling when the pool is smaller."""
    if not pool:
        return []
    if n <= len(pool):
        return pool[:n]
    return [pool[i % len(pool)] for i in range(n)]


def build_requests(
    *,
    counts: TaskCounts,
    config: OmniConfig,
    dataset: str,
    dataset_path: str | None,
    bot_task_weights: dict[str, float],
    input_image: str | None,
    prompts: dict[str, str] | None,
    chat_url: str,
    images_edits_url: str,
    return_stage_metrics: bool,
    it2i_endpoint: str,
    seed: int,
    shuffle: bool,
    bot_task_sampling: str = "proportional",
    group_order: tuple[str, ...] = ("i2t", "t2i", "it2i"),
    gen_resolution_weights: dict[str, float] | None = None,
    gen_resolution_sampling: str = "proportional",
    randomize_input: bool = False,
) -> list[OmniRequest]:
    """Construct the full request list for the run.

    Order of operations: allocate it2i bot_tasks + per-type generation
    resolutions → build per-task lists → merge in ``group_order`` → shuffle
    (or not). The RNG is single and seeded so the whole sequence (bot_task
    allocation + resolution allocation + shuffle) is reproducible. RNG draws
    that are disabled by default (``gen_resolution_weights is None``) are
    skipped entirely, so a run that doesn't use the resolution feature gets the
    exact same bot_task + shuffle sequence as before.

    ``bot_task_sampling`` / ``gen_resolution_sampling`` are ``proportional``
    (exact counts via largest remainder, default) or ``random`` (independent
    weighted draws, matches weights only in expectation). ``group_order`` gives
    the concatenation order of the i2t/t2i/it2i groups when ``shuffle`` is
    False (it is ignored when ``shuffle`` is True since the merge is then
    randomized). ``gen_resolution_weights`` overrides ``config.width`` /
    ``config.height`` per t2i/it2i request — each of t2i and it2i is allocated
    resolutions independently per the ratio; a custom row's explicit
    ``width``/``height`` still wins.

    ``randomize_input`` (``--randomize-input``, ``random`` dataset only) varies
    the *input* side per request to approximate real-world diversity: a prompt
    drawn from a short→long pool per task type (i2t / t2i / it2i); for i2t /
    it2i a unique varied-content image generated at a resolution drawn from
    512x512 / 1024x1024 / 1280x720. All draws come from the same seeded ``rng``
    *after* the bot_task + resolution allocations, so a run with
    ``randomize_input=False`` keeps the exact same bot_task + resolution +
    shuffle sequence as before. ``--input-image`` and the ``--*-prompt``
    overrides are ignored while randomization is on; custom dataset rows always
    win (randomization is skipped for ``dataset="custom"``).
    """
    rng = random.Random(seed)
    use_random = randomize_input and dataset == "random"

    prompts = prompts or {}
    i2t_prompt = prompts.get("i2t") or DEFAULT_I2T_PROMPT
    t2i_prompt = prompts.get("t2i") or DEFAULT_T2I_PROMPT
    it2i_prompt = prompts.get("it2i") or DEFAULT_IT2I_PROMPT

    # The shared placeholder / --input-image is only needed when input images
    # are NOT being generated per-request by randomization.
    placeholder = None
    if (counts.i2t > 0 or counts.it2i > 0) and not use_random:
        placeholder = input_image if input_image else _placeholder_image_path()
        if input_image and not os.path.exists(input_image):
            raise ValueError(f"--input-image not found: {input_image}")

    custom_pools: dict[str, list[dict[str, Any]]] | None = None
    if dataset == "custom":
        if not dataset_path:
            raise ValueError("--dataset-path is required when --dataset custom.")
        custom_pools = _load_custom_pools(dataset_path)

    def gen_kwargs_from_row(row: dict[str, Any]) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        for k in ("width", "height", "num_inference_steps", "seed"):
            if k in row and row[k] is not None:
                kw[k] = row[k]
        return kw

    # Bot-task draw stays the first RNG use so a run without the resolution
    # feature reproduces the legacy bot_task + shuffle sequence exactly.
    it2i_bot_tasks = draw_bot_tasks(counts.it2i, bot_task_weights, rng, bot_task_sampling) if counts.it2i else []
    # Per-type generation resolutions for the image-output tasks. Each of t2i /
    # it2i gets its own exact-proportional (or random) draw over the allowed
    # resolutions; a custom row's explicit width/height still wins per-request.
    t2i_resolutions = (
        draw_resolutions(counts.t2i, gen_resolution_weights, rng, gen_resolution_sampling)
        if gen_resolution_weights and counts.t2i
        else None
    )
    it2i_resolutions = (
        draw_resolutions(counts.it2i, gen_resolution_weights, rng, gen_resolution_sampling)
        if gen_resolution_weights and counts.it2i
        else None
    )

    i2t_reqs: list[OmniRequest] = []
    t2i_reqs: list[OmniRequest] = []
    it2i_reqs: list[OmniRequest] = []

    # --- i2t (image understanding; modalities=text; finishes at AR stage) ---
    for i in range(counts.i2t):
        row = None
        if custom_pools and custom_pools["i2t"]:
            row = custom_pools["i2t"][i % len(custom_pools["i2t"])]
        if row:
            prompt = str(row.get("prompt", i2t_prompt))
            img = _resolve_image_path(row, placeholder)
            in_size = None
        elif use_random:
            prompt = _draw_prompt("i2t", rng)
            iw, ih = _draw_input_resolution(rng)
            img = _generate_diverse_image(rng, iw, ih)
            in_size = f"{iw}x{ih}"
        else:
            prompt = i2t_prompt
            img = placeholder
            in_size = None
        extra_body: dict[str, Any] = {"modalities": ["text"], "temperature": 0}
        if return_stage_metrics:
            extra_body["return_stage_metrics"] = True
        i2t_reqs.append(
            OmniRequest(
                task_type="i2t",
                prompt=prompt,
                api_url=chat_url,
                model=config.model,
                image_paths=[img] if img else None,
                extra_body=extra_body,
                seed=config.seed,
                input_image_size=in_size,
            )
        )

    # --- t2i (text-to-image; modalities=image; no reference image) ---
    for i in range(counts.t2i):
        row = None
        if custom_pools and custom_pools["t2i"]:
            row = custom_pools["t2i"][i % len(custom_pools["t2i"])]
        if row:
            prompt = str(row.get("prompt", t2i_prompt))
            gkw = gen_kwargs_from_row(row)
        elif use_random:
            prompt = _draw_prompt("t2i", rng)
            gkw = {}
        else:
            prompt = t2i_prompt
            gkw = {}
        if t2i_resolutions is not None and "width" not in gkw and "height" not in gkw:
            rw, rh = t2i_resolutions[i]
            gkw.setdefault("width", rw)
            gkw.setdefault("height", rh)
        extra_body = {"modalities": ["image"], "temperature": 0}
        if return_stage_metrics:
            extra_body["return_stage_metrics"] = True
        t2i_reqs.append(
            OmniRequest(
                task_type="t2i",
                prompt=prompt,
                api_url=chat_url,
                model=config.model,
                width=gkw.get("width", config.width),
                height=gkw.get("height", config.height),
                num_inference_steps=gkw.get("num_inference_steps", config.num_inference_steps),
                seed=gkw.get("seed", config.seed),
                extra_body=extra_body,
            )
        )

    # --- it2i (image editing; modalities=image + reference image + bot_task) ---
    for i in range(counts.it2i):
        row = None
        if custom_pools and custom_pools["it2i"]:
            row = custom_pools["it2i"][i % len(custom_pools["it2i"])]
        bot_task = it2i_bot_tasks[i]
        if row:
            prompt = str(row.get("prompt", it2i_prompt))
            img = _resolve_image_path(row, placeholder)
            if row.get("bot_task"):
                bot_task = str(row["bot_task"])
            gkw = gen_kwargs_from_row(row)
            in_size = None
        elif use_random:
            prompt = _draw_prompt("it2i", rng)
            iw, ih = _draw_input_resolution(rng)
            img = _generate_diverse_image(rng, iw, ih)
            gkw = {}
            in_size = f"{iw}x{ih}"
        else:
            prompt = it2i_prompt
            img = placeholder
            gkw = {}
            in_size = None
        if it2i_resolutions is not None and "width" not in gkw and "height" not in gkw:
            rw, rh = it2i_resolutions[i]
            gkw.setdefault("width", rw)
            gkw.setdefault("height", rh)
        extra_body: dict[str, Any] = {"modalities": ["image"], "temperature": 0}
        # On the chat endpoint bot_task rides in extra_body; on /v1/images/edits
        # it is a form field handled by async_request_image_edits via
        # default_bot_task / extra_body.
        if it2i_endpoint == "chat":
            extra_body["bot_task"] = bot_task
        if return_stage_metrics:
            extra_body["return_stage_metrics"] = True
        it2i_url = images_edits_url if it2i_endpoint == "images-edits" else chat_url
        it2i_reqs.append(
            OmniRequest(
                task_type="it2i",
                prompt=prompt,
                api_url=it2i_url,
                model=config.model,
                image_paths=[img] if img else None,
                width=gkw.get("width", config.width),
                height=gkw.get("height", config.height),
                num_inference_steps=gkw.get("num_inference_steps", config.num_inference_steps),
                seed=gkw.get("seed", config.seed),
                extra_body=extra_body,
                default_bot_task=bot_task,
                bot_task=bot_task,
                input_image_size=in_size,
            )
        )

    # Merge per-type lists in group_order (any task type missing from
    # group_order is appended after, defensively, so counts are preserved).
    by_type = {"i2t": i2t_reqs, "t2i": t2i_reqs, "it2i": it2i_reqs}
    requests: list[OmniRequest] = []
    for task in group_order:
        requests.extend(by_type.get(task, []))
    for task in ("i2t", "t2i", "it2i"):
        if task not in group_order:
            requests.extend(by_type.get(task, []))

    if shuffle:
        rng.shuffle(requests)
    return requests


__all__ = [
    "DEFAULT_I2T_PROMPT",
    "DEFAULT_T2I_PROMPT",
    "DEFAULT_IT2I_PROMPT",
    "OmniConfig",
    "TaskCounts",
    "GEN_RESOLUTIONS",
    "INPUT_RESOLUTIONS",
    "parse_bot_task_weights",
    "allocate_bot_tasks",
    "sample_bot_tasks",
    "draw_bot_tasks",
    "parse_gen_resolution_weights",
    "allocate_resolutions",
    "sample_resolutions",
    "draw_resolutions",
    "build_requests",
]
