# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regression for NPU metadata gathering and MammothModa2 token constraints.

Execute the source methods without importing the hardware-dependent runners.
The inherited GPU gather method and the model's logit mask run unchanged.
"""

from __future__ import annotations

import ast
import json
import logging
import runpy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_ROOT = Path(__file__).resolve().parents[3]


def _source_class(path: str, name: str, method: str, bases: tuple[str, ...] = ()) -> ast.ClassDef:
    tree = ast.parse((_ROOT / path).read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method]
    assert len(cls.body) == 1
    cls.bases = [ast.Name(id=base, ctx=ast.Load()) for base in bases]
    cls.decorator_list = []
    return cls


@pytest.fixture
def source_methods():
    gather = "_gather_runtime_additional_information"
    model_path = "vllm_omni/model_executor/models/mammoth_moda2/mammoth_moda2.py"
    model_tree = ast.parse((_ROOT / model_path).read_text(encoding="utf-8"))
    nodes = [
        node
        for node in model_tree.body
        if isinstance(node, ast.FunctionDef) and node.name in ("_runtime_meta", "_first_int")
    ]
    nodes.extend(
        [
            _source_class("vllm_omni/worker/gpu_model_runner.py", "OmniGPUModelRunner", gather),
            _source_class(
                "vllm_omni/platforms/npu/worker/npu_model_runner.py",
                "OmniNPUModelRunner",
                gather,
                ("OmniGPUModelRunner",),
            ),
            _source_class(model_path, "MammothModa2ARForConditionalGeneration", "_apply_t2i_token_constraints"),
        ]
    )
    namespace = {"torch": torch, "Any": Any, "logger": logging.getLogger(__name__)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), __file__, "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("generated_len", [0, 63, 64, 65, 4159])
def test_npu_preserves_mammoth_t2i_grid_constraints(source_methods, nested, generated_len):
    extras = runpy.run_path(str(_ROOT / "vllm_omni/model_extras/mammothmodal2_preview.py"))
    metadata = extras["build_text_to_image_prompt"]("A cat", None, 1024, 1024)["additional_information"]
    payload = {"meta": metadata} if nested else metadata
    runner = source_methods.OmniNPUModelRunner()
    runner.input_batch = SimpleNamespace(req_ids=["image-request"])
    runner.requests = {"image-request": SimpleNamespace(output_token_ids=[0] * generated_len)}
    runner.model_intermediate_buffer = {"image-request": payload}

    infos = runner._gather_runtime_additional_information()
    assert source_methods._runtime_meta(infos[0])["omni_task"] == ["t2i"]
    assert infos[0]["generated_len"] == generated_len
    assert infos[0]["request_id"] == "image-request"
    assert ("meta" in infos[0]) == nested

    model = source_methods.MammothModa2ARForConditionalGeneration()
    model.language_model = SimpleNamespace(base_vocab_size=152064)
    model._last_runtime_additional_information = infos
    logits = model._apply_t2i_token_constraints(torch.zeros(1, 168512))
    allowed = torch.where(torch.isfinite(logits[0]))[0]
    expected = torch.tensor([152064]) if generated_len % 65 == 64 else torch.arange(152072, 168457)
    assert torch.equal(allowed, expected)

    # Gathering again during decoding must not change the metadata layout.
    runner.requests["image-request"].output_token_ids.append(0)
    assert runner._gather_runtime_additional_information()[0]["generated_len"] == generated_len + 1
    assert source_methods._runtime_meta(infos[0])["omni_task"] == ["t2i"]


def test_npu_preserves_request_ids_and_batch_order(source_methods):
    payloads = {
        "flat": {"omni_task": ["chat"]},
        "nested": {"meta": {"last_chunk": True}},
        "handoff": {"request_id": "upstream", "meta": {"request_id": "upstream"}},
    }
    runner = source_methods.OmniNPUModelRunner()
    runner.input_batch = SimpleNamespace(req_ids=["nested", "flat", "handoff", "empty"])
    runner.requests = {}
    runner.model_intermediate_buffer = payloads

    infos = runner._gather_runtime_additional_information()
    assert [info["request_id"] for info in infos] == ["nested", "flat", "upstream", "empty"]
    assert infos[0]["meta"] == {"last_chunk": True, "request_id": "nested"}
    assert "meta" not in infos[1]
    assert infos[2]["meta"]["request_id"] == "upstream"
    assert infos[3] == {"request_id": "empty"}

    model = source_methods.MammothModa2ARForConditionalGeneration()
    model.language_model = SimpleNamespace(base_vocab_size=152064)
    model._last_runtime_additional_information = [infos[1]]
    logits = model._apply_t2i_token_constraints(torch.zeros(1, 168512))
    assert torch.isfinite(logits[0, :152064]).all()
    assert torch.isneginf(logits[0, 152064:]).all()


@pytest.fixture
def code2wav_parser():
    path = "vllm_omni/model_executor/models/minicpmo_4_5/minicpmo_4_5_code2wav.py"
    tree = ast.parse((_ROOT / path).read_text(encoding="utf-8"))
    functions = {"_scalar", "_codec_tensor", "_carries_stage_payload", "_batch_error"}
    constants = {"_RUNNER_STAMPED_KEYS", "_PRODUCER_META_KEYS"}
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in functions)
        or (isinstance(node, ast.ClassDef) and node.name in {"_WorkItem", "_RequestState"})
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets)
        )
    ]
    nodes.append(_source_class(path, "MiniCPMO45Code2Wav", "_parse_item"))
    device_copy = runpy.run_path(str(_ROOT / "vllm_omni/utils/device_copy.py"))
    namespace = {
        "__name__": __name__,
        "torch": torch,
        "json": json,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "dataclass": dataclass,
        "to_device_nonblocking": device_copy["to_device_nonblocking"],
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), path, "exec"), namespace)
    parser = namespace["MiniCPMO45Code2Wav"]()
    parser._states = {}
    parser._default_prompt_id = "default"
    parser._default_prompt_wav = "reference.wav"
    # Audio reference preparation is outside this request-metadata regression.
    parser._resolve_prompt = lambda state_id, info, meta, previous: ("default", "reference.wav", None)
    return parser, namespace["_RequestState"]


@pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
@pytest.mark.parametrize("handoff", [False, True], ids=["new-id", "existing-id"])
def test_code2wav_consumes_npu_metadata_across_chunks(source_methods, code2wav_parser, nested, handoff):
    parser, state_type = code2wav_parser
    runner = source_methods.OmniNPUModelRunner()
    runner.input_batch = SimpleNamespace(req_ids=["engine-request"])
    runner.requests = {}
    for sequence in (0, 1):
        meta = {"cache_epoch": 7, "chunk_seq": sequence, "last_chunk": sequence == 1, "code_flat_numel": 2}
        info = {"meta": meta} if nested else meta
        info["codes"] = {"audio": torch.tensor([10, 11])}
        if handoff:
            info["request_id"] = "upstream-request"
            if nested:
                info["meta"]["request_id"] = "upstream-request"
        runner.model_intermediate_buffer = {"engine-request": info}
        gathered = runner._gather_runtime_additional_information()[0]
        item = parser._parse_item(0, "engine-request", torch.tensor([0]), gathered)
        assert item.request_id == ("upstream-request" if handoff else "engine-request")
        assert item.has_payload
        assert item.cache_epoch == 7
        assert item.chunk_seq == sequence
        assert item.last_chunk == (sequence == 1)
        assert torch.equal(item.tokens, torch.tensor([10, 11]))
        parser._states["engine-request"] = state_type(7, sequence, "default", "reference.wav", None)


def test_code2wav_npu_prewarm_remains_payload_free(source_methods, code2wav_parser):
    parser, _ = code2wav_parser
    runner = source_methods.OmniNPUModelRunner()
    runner.input_batch = SimpleNamespace(req_ids=["prewarm-request"])
    runner.requests = {}
    runner.model_intermediate_buffer = {}
    gathered = runner._gather_runtime_additional_information()[0]
    item = parser._parse_item(0, "prewarm-request", torch.tensor([0]), gathered)
    assert item.request_id == "prewarm-request"
    assert not item.has_payload
    assert item.tokens.numel() == 0
