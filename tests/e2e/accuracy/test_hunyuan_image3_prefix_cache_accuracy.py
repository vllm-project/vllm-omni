# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Online DiT-only regression for actual cross-request paged prefix reuse.

Single diffusion stage (no staged deployment), 50 denoising steps, CFG=2.5
(two rows), multiple seeds and repeated warm hits. Outputs are compared
against the input reference image as a sanity gate, and prefix/repeat outputs
are compared directly against the paged no-cache output. Actual cache hits
and repeated-output determinism are checked separately.
"""

import base64
import copy
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import requests
import yaml
from PIL import Image

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

_MODEL = "tencent/HunyuanImage-3.0-Instruct"
_STEPS = 50
HUNYUAN_IMAGE_REF_PATH = get_asset_path("hunyuan/hunyuan_image_ref.png")

_PROMPT = (
    "Watercolor painting with soft blue and green washes. Retain the subject from the reference image, "
    "its proportions, pose, expression, and placement in the frame. Render the background as a quiet "
    "garden with delicate brushwork, subtle paper texture, and natural daylight. Keep the subject "
    "recognizable and clearly separated from the background. Do not add text, logos, borders, additional "
    "subjects, or a collage. Produce a single coherent image with fine details and a balanced composition."
)
_WARMUP_PROMPT = (
    "Restore the reference image with neutral studio lighting and a plain background. Preserve the "
    "subject, its proportions, expression, and pose. Keep natural colors and detailed textures, without "
    "text or additional subjects."
)

_DEPLOY_CONFIG = {
    "pipeline": "hunyuan_image3_dit",
    "async_chunk": False,
    "trust_remote_code": True,
    "stages": [
        {
            "stage_id": 0,
            "devices": "0,1,2,3",
            "max_num_seqs": 1,
            "max_model_len": 22800,
            "max_num_batched_tokens": 32768,
            "diffusion_kv_max_rows_per_request": 2,
            "gpu_memory_utilization": 0.9,
            "enforce_eager": True,
            "vae_use_tiling": True,
            "parallel_config": {
                "tensor_parallel_size": 4,
                "enable_expert_parallel": True,
                "sequence_parallel_size": 1,
                "cfg_parallel_size": 1,
                "vae_patch_parallel_size": 4,
            },
            "default_sampling_params": {"seed": 42},
        }
    ],
}

# Mode -> (diffusion_kv_mode, enable_prefix_caching); everything else identical.
_MODE_KV_CONFIGS = {
    "dense": ("dense_legacy", False),
    "paged_no_cache": ("paged_scheduler", False),
    "paged_prefix": ("paged_scheduler", True),
}

# Loose sanity gate against the input reference image. The edit prompt is a
# heavy style transfer and the reference is 1280x720 while outputs are
# 1024x1024, so these only check that the result is still a recognizable edit
# of the subject. Same spirit as test_hunyuan_image3.py's reference-image
# criteria; do not import that module: it sets backends and inspects topology.
_REFERENCE_THRESHOLDS = {"clip_score": 78.0, "ssim": 0.15, "psnr": 8.0}
# Prefix reuse must reproduce the uncached paged output up to small numerical
# drift from the trimmed first-step query; observed drift is far above these.
_CONSISTENCY_THRESHOLDS = {"clip_score": 95.0, "ssim": 0.90, "psnr": 25.0}

_PREFILL_RE = re.compile(
    r"Diffusion prefix prefill: request_id=(\S+) sequence_id=(\d+) "
    r"cached_prefix_len=(\d+) prefix_len=(\d+) query_len=(\d+)"
)
_SLICE_RE = re.compile(r"Hunyuan prefix slice: cached_prefix_len=(\d+) query_len=(\d+)")
_REFERENCE_RE = re.compile(r"Hunyuan reference span: sequence_id=(\d+) start=(\d+) end=(\d+)")


@dataclass(frozen=True)
class PrefixPrefill:
    request_id: str
    sequence_id: int
    cached_prefix_len: int
    prefix_len: int
    query_len: int


def _parse_prefix_prefills(log: str) -> list[PrefixPrefill]:
    return [PrefixPrefill(request_id, *map(int, values)) for request_id, *values in _PREFILL_RE.findall(log)]


def _assert_prefix_was_sliced(events: list[PrefixPrefill], log: str) -> None:
    """Cross-check worker boundaries with actual model-side query slicing.

    TP ranks execute identical slices, so compare sets rather than log counts.
    Call with logs from one completed, sequential request only (CFGP=1).
    """
    expected = {(event.cached_prefix_len, event.query_len) for event in events if event.cached_prefix_len > 0}
    actual = {tuple(map(int, match)) for match in _SLICE_RE.findall(log)}
    assert actual == expected, f"Hunyuan did not execute the scheduled prefix slices: {expected=}, {actual=}"


def _reference_prefix_starts(log: str) -> dict[int, int]:
    """Read the first reference's actual token boundary, deduplicating TP logs."""
    starts: dict[int, int] = {}
    for row, start, end in _REFERENCE_RE.findall(log):
        row, start, end = int(row), int(start), int(end)
        assert 0 <= start < end, f"Invalid reference span: {row=}, {start=}, {end=}"
        starts[row] = min(starts.get(row, start), start)
    return starts


def _assert_reference_prefix_hit(events: list[PrefixPrefill], *, num_branches: int, log: str) -> None:
    """Require reuse within the actual image span, not just system/text tokens."""
    starts = _reference_prefix_starts(log)
    assert set(starts) == set(range(num_branches)), f"Missing reference spans: {starts}"
    assert len(events) == num_branches, f"Expected {num_branches} prefill rows, got {events}"
    assert len({event.request_id for event in events}) == 1, f"Mixed requests: {events}"
    assert {event.sequence_id for event in events} == set(range(num_branches)), events
    assert len({event.cached_prefix_len for event in events}) == 1, f"CFG boundaries differ: {events}"
    for event in events:
        assert starts[event.sequence_id] < event.cached_prefix_len <= event.prefix_len, (
            f"No reference-image prefix hit: {event}, reference_start={starts[event.sequence_id]}"
        )
        assert event.query_len > 0, f"No suffix queries remain: {event}"


def _make_config(mode: str, path: Path) -> None:
    config = copy.deepcopy(_DEPLOY_CONFIG)
    kv_mode, prefix = _MODE_KV_CONFIGS[mode]
    config["stages"][0]["diffusion_kv_mode"] = kv_mode
    config["stages"][0]["enable_prefix_caching"] = prefix
    path.write_text(yaml.safe_dump(config))


def _assert_image_quality(metrics, *, label, thresholds):
    failures = []
    for name, threshold in thresholds.items():
        value = metrics[name]
        print(f"{label} {name}: value={value:.6f}, threshold>={threshold:.6f}")
        if not value >= threshold:
            failures.append(f"{label} {name} below threshold: got {value:.6f}, expected >= {threshold:.6f}")
    assert not failures, "\n".join(failures)


def _image_metrics(clip_scorer, prediction, reference):
    """CLIP/SSIM/PSNR between two images, plus MAE/P99 as diagnostics."""
    from tests.e2e.accuracy.helpers import compute_image_ssim_psnr

    if prediction.size != reference.size:
        prediction = prediction.resize(reference.size)
    ssim, psnr = compute_image_ssim_psnr(prediction=prediction, reference=reference)
    difference = np.abs(
        np.asarray(prediction, dtype=np.float32) / 255 - np.asarray(reference, dtype=np.float32) / 255
    )
    return {
        "clip_score": clip_scorer.image_image_score(prediction, reference),
        "ssim": ssim,
        "psnr": psnr,
        "mae": float(difference.mean()),
        "p99": float(np.quantile(difference, 0.99)),
    }


def _generate_edit(server, *, prompt, seed, guidance_scale, image_path, output_path, capfd):
    # Requests are sequential. All traces between these two snapshots belong
    # to this call, including both CFG rows on TP rank zero (CFGP=1).
    before = capfd.readouterr()
    output_path.with_suffix(".before.log").write_text(before.out + before.err)
    try:
        with image_path.open("rb") as reference:
            response = requests.post(
                f"http://{server.host}:{server.port}/v1/images/edits",
                data={
                    "model": server.model,
                    "prompt": prompt,
                    "size": "1024x1024",
                    "n": 1,
                    "response_format": "b64_json",
                    "num_inference_steps": _STEPS,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "bot_task": "vanilla",
                    "sys_type": "en_unified",
                },
                files=[("image", (image_path.name, reference, "image/png"))],
                timeout=600,
            )
        assert response.ok, f"HTTP {response.status_code}: {response.text}"
        payload = response.json()
        assert len(payload["data"]) == 1
        image = Image.open(io.BytesIO(base64.b64decode(payload["data"][0]["b64_json"]))).convert("RGB")
        image.load()
        assert image.size == (1024, 1024), f"Unexpected output size: {image.size}"
        image.save(output_path)
    finally:
        captured = capfd.readouterr()
        output_path.with_suffix(".log").write_text(captured.out + captured.err)
    log = captured.out + captured.err
    events = _parse_prefix_prefills(log)
    _assert_prefix_was_sliced(events, log)
    return image, events


@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("seed_base", [42, 44])
def test_hunyuan_image3_warm_prefix_accuracy(tmp_path, capfd, seed_base):
    from tests.e2e.accuracy.helpers import CLIPScorer, model_output_dir
    from tests.helpers.runtime import OmniServer

    model = os.environ.get("HUNYUAN_IMAGE3_MODEL", _MODEL)
    output_dir = model_output_dir(tmp_path, model)
    image_path = HUNYUAN_IMAGE_REF_PATH
    assert _WARMUP_PROMPT != _PROMPT

    # CFG=2.5 keeps two rows per request so both CFG branches are exercised.
    # The seed bases give two independent image-edit samples while avoiding
    # the CFG=1 path, which currently builds an inconsistent one/two-row batch.
    cases = [(2.5, seed_base + 1)]
    with Image.open(image_path) as reference_image:
        reference_image = reference_image.convert("RGB")
    results = {}
    traces = {}
    include_dense = os.environ.get("HUNYUAN_IMAGE3_INCLUDE_DENSE") == "1"
    for mode in _MODE_KV_CONFIGS:
        if mode == "dense" and not include_dense:
            # Dense is useful for local diagnostics, but the consistency
            # baseline is the uncached paged output.
            continue
        deploy_path = output_dir / f"{mode}.yaml"
        _make_config(mode, deploy_path)
        # DEBUG provides per-request consumed prefill boundaries. No metrics
        # API or server-wide hit counters are needed for this regression.
        with OmniServer(
            model,
            ["--deploy-config", str(deploy_path), "--trust-remote-code", "--stage-init-timeout", "900"],
            env_dict={
                "VLLM_LOGGING_LEVEL": "DEBUG",
                "DIFFUSION_ATTENTION_BACKEND": os.environ.get("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN"),
            },
        ) as server:
            for guidance, seed in cases:
                common = dict(server=server, seed=seed, guidance_scale=guidance, image_path=image_path, capfd=capfd)
                num_branches = 1 if guidance == 1.0 else 2
                label = f"{mode}_cfg{guidance}"
                if mode == "paged_prefix":
                    _, warmup = _generate_edit(
                        **common,
                        prompt=_WARMUP_PROMPT,
                        output_path=output_dir / f"{label}_warmup.png",
                    )
                    assert len(warmup) == num_branches, f"Missing worker traces: {warmup}"
                    # The second CFG case may reuse initial system-text blocks
                    # from the first, but must not reuse its sampled image KV.
                    warmup_log = (output_dir / f"{label}_warmup.log").read_text()
                    reference_starts = _reference_prefix_starts(warmup_log)
                    assert set(reference_starts) == set(range(num_branches)), reference_starts
                    assert all(event.cached_prefix_len <= reference_starts[event.sequence_id] for event in warmup), (
                        warmup,
                        reference_starts,
                    )
                    traces[f"{label}_warmup"] = [vars(event) for event in warmup]

                image, events = _generate_edit(
                    **common, prompt=_PROMPT, output_path=output_dir / f"{label}.png"
                )
                results[mode, guidance] = image
                traces[label] = [vars(event) for event in events]
                if mode == "dense":
                    assert not events, f"Dense unexpectedly emitted paged traces: {events}"
                elif mode == "paged_no_cache":
                    assert len(events) == num_branches, f"Missing worker traces: {events}"
                    assert all(event.cached_prefix_len == 0 for event in events), events
                else:
                    _assert_reference_prefix_hit(
                        events, num_branches=num_branches, log=(output_dir / f"{label}.log").read_text()
                    )
                    assert all(event.cached_prefix_len < event.prefix_len for event in events), events
                    repeated, repeat_events = _generate_edit(
                        **common, prompt=_PROMPT, output_path=output_dir / f"{label}_repeat.png"
                    )
                    _assert_reference_prefix_hit(
                        repeat_events, num_branches=num_branches, log=(output_dir / f"{label}_repeat.log").read_text()
                    )
                    # The long target prompt adds complete cacheable blocks.
                    # Repetition must hit beyond the partial reference boundary.
                    assert repeat_events[0].cached_prefix_len > events[0].cached_prefix_len, (
                        f"Expected a longer hit on repetition: partial={events}, repeat={repeat_events}"
                    )
                    results["paged_repeat", guidance] = repeated
                    traces[f"{label}_repeat"] = [vars(event) for event in repeat_events]
                    for repeat_index in (2, 3):
                        repeat_label = f"{label}_repeat{repeat_index}"
                        next_image, next_events = _generate_edit(
                            **common, prompt=_PROMPT, output_path=output_dir / f"{repeat_label}.png"
                        )
                        _assert_reference_prefix_hit(
                            next_events,
                            num_branches=num_branches,
                            log=(output_dir / f"{repeat_label}.log").read_text(),
                        )
                        assert [(event.sequence_id, event.cached_prefix_len) for event in next_events] == [
                            (event.sequence_id, event.cached_prefix_len) for event in repeat_events
                        ], (repeat_events, next_events)
                        traces[repeat_label] = [vars(event) for event in next_events]
                        # Equal request, seed and hit boundary must be repeatable.
                        assert next_image.tobytes() == repeated.tobytes(), (
                            f"Repeated warm hit is not deterministic: CFG={guidance}, seed={seed}, "
                            f"repeat={repeat_index}"
                        )
                (output_dir / "prefix_traces.json").write_text(json.dumps(traces, indent=2))

    # Use the same image-image CLIP scorer as the AR-to-DiT test, after all
    # generation servers have exited. MAE/P99 remain diagnostics, not gates.
    clip_scorer = CLIPScorer()
    quality_metrics = {}
    failures = []
    for guidance, seed in cases:
        predictions = ["paged_no_cache", "paged_prefix", "paged_repeat"]
        if include_dense:
            predictions = ["dense"] + predictions
        comparisons = [
            (prediction, reference_image, _REFERENCE_THRESHOLDS, "vs input reference")
            for prediction in predictions
        ]
        no_cache = results["paged_no_cache", guidance]
        comparisons += [
            (prediction, no_cache, _CONSISTENCY_THRESHOLDS, "vs paged_no_cache")
            for prediction in predictions
            if prediction != "paged_no_cache"
        ]
        for prediction, reference, thresholds, suffix in comparisons:
            label = f"{_MODEL} CFG={guidance} seed={seed} {prediction} {suffix}"
            metrics = _image_metrics(clip_scorer, results[prediction, guidance], reference)
            quality_metrics[label] = metrics
            # Report every lane/metric even if an earlier comparison fails.
            try:
                _assert_image_quality(metrics, label=label, thresholds=thresholds)
            except AssertionError as error:
                failures.append(str(error))
    (output_dir / "quality_metrics.json").write_text(json.dumps(quality_metrics, indent=2))
    assert not failures, "\n".join(failures)
