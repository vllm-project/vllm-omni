# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in M1 acceptance: real MRv2 turn serving with restored Talker inputs.

Set MOONCAKE_CONFIG_PATH to a running Store's JSON configuration. Optional
MOONCAKE_TEST_MODEL selects a local copy of the pinned checkpoint. The test
uses two H100-class GPUs with the standard Qwen3 MRv2 turn deployment.
It writes tensor/restore evidence under pytest's temporary directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

import pytest
import torch
from safetensors.torch import load_file

from tests.engine import kv_offload_helpers as helpers
from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, *hardware_marks(res={"cuda": "H100"}, num_cards=2)]

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
BLOCK_SIZE = 16
AUDIT_ENV = "VLLM_OMNI_MOONCAKE_TEST_AUDIT_DIR"
WORKER = "tests.engine.mooncake_helpers.AuditedGPUARWorker"


def _observations(directory: Path, case: str, kind: str) -> list[dict]:
    return [json.loads(path.read_text()) for path in directory.glob(f"{case}-{kind}-*.json")]


def _multimodal_cases(prompt: dict) -> list[tuple[str, dict]]:
    import numpy as np
    from PIL import Image

    cases: list[tuple[str, dict]] = []
    samples = np.arange(16000, dtype=np.float32) / 16000
    for modality, tag, original, changed in (
        (
            "image",
            "<|vision_start|><|image_pad|><|vision_end|>",
            Image.new("RGB", (112, 112), "red"),
            Image.new("RGB", (112, 112), "blue"),
        ),
        (
            "audio",
            "<|audio_start|><|audio_pad|><|audio_end|>",
            (0.1 * np.sin(2 * np.pi * 440 * samples), 16000),
            (0.1 * np.sin(2 * np.pi * 660 * samples), 16000),
        ),
    ):
        # Media begins before the first complete block. Changing only its
        # contents must invalidate every block, despite identical prompt text.
        text = prompt["prompt"].replace("user\n", f"user\n{tag}", 1)
        first = {"prompt": text, "multi_modal_data": {modality: original}}
        different = {"prompt": text, "multi_modal_data": {modality: changed}}
        cases.extend(((f"{modality}_cold", first), (f"{modality}_full", first), (f"{modality}_changed", different)))
    return cases


def _run_arm(model: str, directory: Path, cases: list[tuple[str, dict]], mooncake: dict | None, monkeypatch) -> dict:
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    directory.mkdir()
    (directory / "case.txt").write_text("startup")
    monkeypatch.setenv(AUDIT_ENV, str(directory))
    thinker = {
        "use_v2_model_runner": True,
        "worker_cls": WORKER,
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "max_model_len": 2048,
        "max_num_batched_tokens": 256,
        "enable_chunked_prefill": True,
        "enable_prefix_caching": False,
        "block_size": BLOCK_SIZE,
        "gpu_memory_utilization": 0.85,
        "revision": REVISION,
    }
    if mooncake is not None:
        thinker["omni_kv_config"] = {"kv_store_config": {"mooncake_config": mooncake}}
    overrides = helpers.stage_overrides(
        lmcache=False,
        thinker_extra=thinker,
        downstream_extra={
            "1": {
                "enable_prefix_caching": False,
                "gpu_memory_utilization": 0.3,
                "max_num_seqs": 1,
                "revision": REVISION,
                "subtalker_sampling_params": {"do_sample": False},
            },
            "2": {
                "enable_prefix_caching": False,
                "gpu_memory_utilization": 0.2,
                "max_num_seqs": 1,
                "revision": REVISION,
            },
        },
    )
    omni = Omni(
        model=model,
        revision=REVISION,
        deploy_config="qwen3_omni_moe_mrv2.yaml",
        stage_overrides=overrides,
        trust_remote_code=True,
        stage_init_timeout=1800,
        init_timeout=1800,
        batch_timeout=5,
    )
    # The shared collector passes this list explicitly to every request.
    omni.default_sampling_params_list = [SamplingParams(**helpers.GREEDY) for _ in range(omni.num_stages)]
    outputs = {}
    try:
        for case, prompt in cases:
            (directory / "case.txt").write_text(case)
            outputs[case] = helpers.collect(omni, [prompt])
    finally:
        omni.close()
    return outputs


def _assert_outputs_equal(expected: dict, actual: dict) -> None:
    assert set(actual) == set(expected)
    for case in expected:
        assert actual[case] and set(actual[case]) == set(expected[case])
        assert all(helpers.audio_len(entry) > 0 for entry in actual[case].values()), f"{case}: no output audio"
        for prompt, entry in actual[case].items():
            assert entry["token_ids"] == expected[case][prompt]["token_ids"], f"{case}: generated token IDs differ"
        problems = helpers.compare(expected[case], actual[case])
        assert not problems, f"{case}: " + "\n".join(problems)


def _assert_handoffs_equal(expected: Path, actual: Path, cases: list[tuple[str, dict]]) -> None:
    for case, _ in cases:
        reference = _observations(expected, case, "handoff")
        observed = _observations(actual, case, "handoff")
        assert len(reference) == len(observed) == 1, f"{case}: prompt must reach Talker exactly once"
        assert observed[0]["prompt_token_ids"] == reference[0]["prompt_token_ids"]
        before = load_file(expected / reference[0]["tensor_file"])
        after = load_file(actual / observed[0]["tensor_file"])
        for key in ("embedding", "hidden", "talker"):
            assert after[key].shape[0] == observed[0]["prompt_len"]
            torch.testing.assert_close(after[key], before[key], rtol=0, atol=0, msg=f"{case}: {key} rows differ")


def test_mooncake_turn_cold_full_and_partial_hits(tmp_path, monkeypatch):
    # Exact fresh-row comparisons require invariance across the shorter
    # prefill shapes produced by cache hits.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config_path = os.environ.get("MOONCAKE_CONFIG_PATH")
    if not config_path:
        pytest.skip("set MOONCAKE_CONFIG_PATH to run native Mooncake turn acceptance")
    assert config_path is not None
    assert Path(config_path).is_file(), "MOONCAKE_CONFIG_PATH must name a Store JSON configuration"
    pytest.importorskip("mooncake", reason="Mooncake is required for native acceptance")
    from transformers import AutoTokenizer

    model = os.environ.get("MOONCAKE_TEST_MODEL", MODEL)
    tokenizer = AutoTokenizer.from_pretrained(model, revision=REVISION)
    first, partial = helpers.prompts(2)
    # Avoid the final-token replay corner here: separately covered by the
    # scheduler tests. This makes expected full-hit length unambiguous.
    while len(tokenizer.encode(first["prompt"])) % BLOCK_SIZE == 0:
        first = {"prompt": first["prompt"] + " "}
    # Change an earlier block: changing only the final question can leave
    # every complete block shared and accidentally exercise another full hit.
    partial = {"prompt": partial["prompt"].replace("Fact 12:", "Additional facts follow. Fact 12:", 1)}
    first_tokens = tokenizer.encode(first["prompt"])
    partial_tokens = tokenizer.encode(partial["prompt"])
    common_prefix = next(
        (index for index, (a, b) in enumerate(zip(first_tokens, partial_tokens)) if a != b),
        min(len(first_tokens), len(partial_tokens)),
    )
    assert common_prefix >= BLOCK_SIZE
    assert common_prefix // BLOCK_SIZE * BLOCK_SIZE < len(partial_tokens) // BLOCK_SIZE * BLOCK_SIZE
    cases = [("cold", first), ("full", first), ("partial", partial), *_multimodal_cases(first)]
    config = {
        "config_file": str(Path(config_path).resolve()),
        "namespace": f"m1-acceptance-{uuid4().hex}",
        "model_revision": REVISION,
    }
    baseline = _run_arm(model, tmp_path / "baseline", cases, None, monkeypatch)
    repeat = _run_arm(model, tmp_path / "repeat", cases, None, monkeypatch)
    # Establish exact reproducibility before interpreting cached-run equality.
    # If this fails, define independent baseline-derived tolerances first;
    # never silently loosen tolerances after seeing cached results.
    _assert_outputs_equal(baseline, repeat)
    _assert_handoffs_equal(tmp_path / "baseline", tmp_path / "repeat", cases)
    cached = _run_arm(model, tmp_path / "cached", cases, config, monkeypatch)
    _assert_outputs_equal(baseline, cached)
    _assert_handoffs_equal(tmp_path / "baseline", tmp_path / "cached", cases)

    for case, _ in cases:
        captures = _observations(tmp_path / "cached", case, "capture")
        assert captures, f"{case}: production capture integration was not exercised"
        inserted = [entry for entry in captures if entry["inserted_rows"] > 0]
        if case == "cold" or case.endswith(("_cold", "_changed")):
            assert not inserted, f"{case}: unseen text or changed media must not restore a prefix"
            continue
        assert len(inserted) == 1, f"{case}: restore must be observed exactly once"
        entry = inserted[0]
        assert entry["inserted_rows"] == entry["start_token"]
        full_prefix = min(entry["prompt_len"] // BLOCK_SIZE * BLOCK_SIZE, entry["prompt_len"] - 1)
        if case == "full" or case.endswith("_full"):
            assert entry["inserted_rows"] == full_prefix
        else:
            assert 0 < entry["inserted_rows"] < full_prefix
