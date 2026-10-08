# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Single-DiT IT2I accuracy against the existing official Instruct reference.

Reuse the IT2I inputs and quality thresholds for cold, partial and repeated prefix
hits. The reference comes from AR->DiT; this is a reference-quality check, not a
claim of identical conditioning or bitwise equivalence between the pipelines.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from benchmarks.accuracy.common import VllmOmniImageClient
from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARK_CONFIG = _REPO_ROOT / "tests/dfx/perf/tests/test_hunyuan_image3_prefix_caching.json"
_IT2I_INPUT = json.loads(get_asset_path("hunyuan_image3/it2i.jsonl").read_text(encoding="utf-8"))
_REFERENCE_PATH = get_asset_path("hunyuan_image3/hunyuan_image_instruct_ref.png")
_PROMPT = _IT2I_INPUT["prompt"]
# Warm up with a different prompt, then score only the original IT2I prompt.
# This exercises partial image-prefix hits without changing the golden's input.
_EDIT_PROMPT = (
    f"请制作一张清晰的产品展示图：{_PROMPT}。保留标志的形状、文字与颜色，"
    "呈现真实的材质纹理，使用柔和的自然光和简洁背景，不添加其他物体。"
)
# Match the official Instruct reference thresholds in test_hunyuan_image3.py.
_ACCURACY_THRESHOLDS = {"clip_score": 90.0, "ssim": 0.26, "psnr": 12.5}
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


def _assert_prefix_execution(log: str) -> list[PrefixPrefill]:
    events = [PrefixPrefill(request_id, *map(int, values)) for request_id, *values in _PREFILL_RE.findall(log)]
    # TP ranks execute identical slices; compare sets rather than log counts.
    expected = {(event.cached_prefix_len, event.query_len) for event in events if event.cached_prefix_len > 0}
    actual = {tuple(map(int, match)) for match in _SLICE_RE.findall(log)}
    assert actual == expected, f"Scheduled/model prefix slices differ: {expected=}, {actual=}"
    if events:
        assert len(events) == 2 and {event.sequence_id for event in events} == {0, 1}, events
        assert len({event.request_id for event in events}) == 1, events
        assert len({event.cached_prefix_len for event in events}) == 1, f"CFG boundaries differ: {events}"
    return events


def _assert_reference_prefix_hit(events: list[PrefixPrefill], log: str) -> None:
    # Require a hit into BOTH reference images, not only the system prompt.
    spans = {tuple(map(int, match)) for match in _REFERENCE_RE.findall(log)}
    for event in events:
        row_spans = [(start, end) for row, start, end in spans if row == event.sequence_id]
        assert len(row_spans) == len(_IT2I_INPUT["image_files"]), spans
        assert all(0 <= start < end for start, end in row_spans), spans
        assert max(start for start, _ in row_spans) < event.cached_prefix_len <= event.prefix_len, event
        assert event.query_len > 0, event


def _generate_edit(client, *, model, images, prompt, output_path, capfd, **generation):
    # Preserve startup/previous diagnostics and capture one sequential request.
    before = capfd.readouterr()
    output_path.with_suffix(".before.log").write_text(before.out + before.err)
    try:
        image = client.generate_image_edit(model=model, prompt=prompt, images=images, **generation)
        assert image.size == (generation["width"], generation["height"]), image.size
        image.save(output_path)
    finally:
        captured = capfd.readouterr()
        output_path.with_suffix(".log").write_text(captured.out + captured.err)
    log = captured.out + captured.err
    return image, _assert_prefix_execution(log), log


@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("tp_size,sp_size", [(4, 1), (2, 2)], ids=["tp4-sp1", "tp2-sp2"])
def test_hunyuan_image3_warm_prefix_accuracy(tmp_path, capfd, tp_size, sp_size):
    from tests.e2e.accuracy.helpers import CLIPScorer, compute_image_ssim_psnr, download_images, model_output_dir
    from tests.helpers.runtime import OmniServer

    configs = json.loads(_BENCHMARK_CONFIG.read_text())
    model = os.environ.get("HUNYUAN_IMAGE3_MODEL", configs[0]["server_params"]["model"])
    output_dir = model_output_dir(tmp_path, model)
    images = download_images(_IT2I_INPUT["image_files"])
    reference = Image.open(_REFERENCE_PATH).convert("RGB")
    # Pin the official reference's settings independently of the shorter perf run.
    generation = dict(
        seed=42,
        num_inference_steps=50,
        guidance_scale=2.5,
        width=1280,
        height=720,
        bot_task="think_recaption",
        sys_type="en_unified",
    )
    results, traces = {}, {}
    for config in configs:
        mode = config["test_name"].removeprefix("test_hunyuan_image3_prefix_")
        server_params = config["server_params"]
        # Keep four workers; exercise trimmed queries with strict Ulysses,
        # not the T2I/no-hit AllGather-KV path covered by other SP tests.
        server_params["stage_overrides"]["0"]["parallel_config"].update(
            tensor_parallel_size=tp_size,
            sequence_parallel_size=sp_size,
            ulysses_degree=sp_size,
            ulysses_mode="strict",
            ring_degree=1,
            allgather_degree=1,
            cfg_parallel_size=1,
        )
        server_args = [
            "--deploy-config",
            str(_REPO_ROOT / server_params["serve_args"]["deploy-config"]),
            "--stage-overrides",
            json.dumps(server_params["stage_overrides"]),
            "--trust-remote-code",
            "--stage-init-timeout",
            "900",
        ]
        requests = [("original", _PROMPT)]
        if mode == "paged_prefix":
            requests = [("warmup", _EDIT_PROMPT), ("partial", _PROMPT)]
            requests += [(f"repeat{i}", _PROMPT) for i in range(1, 4)]
        with OmniServer(
            model,
            server_args,
            env_dict={
                "VLLM_LOGGING_LEVEL": "DEBUG",
                "DIFFUSION_ATTENTION_BACKEND": os.environ.get("DIFFUSION_ATTENTION_BACKEND", "FLASH_ATTN"),
            },
        ) as server:
            client = VllmOmniImageClient(f"http://{server.host}:{server.port}")
            for case, prompt in requests:
                label = f"{mode}_{case}"
                image, events, log = _generate_edit(
                    client,
                    model=server.model,
                    images=images,
                    prompt=prompt,
                    output_path=output_dir / f"{label}.png",
                    capfd=capfd,
                    **generation,
                )
                if case != "warmup":
                    results[mode, case] = image
                traces[label] = [vars(event) for event in events]
                (output_dir / "prefix_traces.json").write_text(json.dumps(traces, indent=2))
                if mode == "dense":
                    assert not events, events
                    continue
                assert len(events) == 2, f"Missing CFG prefill traces: {events}"
                if mode == "paged_no_cache" or case == "warmup":
                    assert all(event.cached_prefix_len == 0 for event in events), events
                    continue
                _assert_reference_prefix_hit(events, log)
                if case == "partial":
                    assert all(event.cached_prefix_len < event.prefix_len for event in events), events
                else:
                    partial_len = traces["paged_prefix_partial"][0]["cached_prefix_len"]
                    assert events[0].cached_prefix_len > partial_len, events
                    if case != "repeat1":
                        first = traces["paged_prefix_repeat1"]
                        assert [event.cached_prefix_len for event in events] == [
                            event["cached_prefix_len"] for event in first
                        ], events
                        assert image.tobytes() == results["paged_prefix", "repeat1"].tobytes(), label

    # Score every mode against the official output, not an uncached Omni output.
    clip_scorer = CLIPScorer()
    metrics, failures = {}, []
    for (mode, case), image in results.items():
        ssim, psnr = compute_image_ssim_psnr(prediction=image, reference=reference)
        label = f"{mode}_{case} vs official_ref"
        metrics[label] = {"clip_score": clip_scorer.image_image_score(image, reference), "ssim": ssim, "psnr": psnr}
        for name, threshold in _ACCURACY_THRESHOLDS.items():
            value = metrics[label][name]
            print(f"{label} {name}: {value:.6f}, required >= {threshold}")
            if not value >= threshold:
                failures.append(f"{label}: {name}={value:.6f} < {threshold}")
    (output_dir / "quality_metrics.json").write_text(json.dumps(metrics, indent=2))
    assert not failures, "\n".join(failures)
