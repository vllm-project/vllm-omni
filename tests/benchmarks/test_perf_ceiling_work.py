# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Check deployment precision from the counting CLI through emitted report flags."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModel, LlamaConfig, LlamaForCausalLM

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SCRIPT = Path(__file__).resolve().parents[2] / ".claude/skills/omni-perf-ceiling/scripts/count_reference_work.py"


@pytest.mark.parametrize(
    "reference_dtype,options,weight_bytes,kv_bytes,compute_mode",
    [
        (torch.bfloat16, ["--dtype", "thinker=fp32"], 4, 4, "fp32"),
        (torch.float32, ["--dtype", "thinker=bf16"], 2, 2, "bf16"),
        (torch.bfloat16, ["--dtype", "thinker=tf32"], 4, 4, "tf32"),
        (torch.bfloat16, ["--dtype", "thinker=hf32"], 4, 4, "hf32"),
        (torch.bfloat16, [], 2, 2, "bf16"),
        (
            torch.bfloat16,
            ["--dtype", "thinker=fp32", "--weight-dtype", "thinker=bf16", "--kv-dtype", "thinker=fp16"],
            2,
            2,
            "fp32",
        ),
        (torch.bfloat16, ["--kv-dtype", "thinker=fp32"], 2, 4, "bf16"),
    ],
)
def test_deployment_dtype_counts_and_flags(
    monkeypatch, tmp_path, capsys, reference_dtype, options, weight_bytes, kv_bytes, compute_mode
):
    spec = importlib.util.spec_from_file_location("count_reference_work", SCRIPT)
    assert spec is not None and spec.loader is not None
    counter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(counter)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        dtype=reference_dtype,
    )
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = torch.nn.Module()
        model.llm = LlamaForCausalLM(config).to(dtype=reference_dtype)
    # The untied input embedding is gathered; all other parameters are streamed.
    streamed_elements = sum(p.numel() for p in model.parameters()) - model.llm.get_input_embeddings().weight.numel()
    kv_elements_per_token = 2 * config.num_hidden_layers * config.num_key_value_heads * 8
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr(AutoModel, "from_config", lambda *a, **kw: model)
    output = tmp_path / "work.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--model",
            "local-tiny-model",
            "--decoder",
            "thinker@0=llm",
            "--prompt-tokens",
            "4",
            "--context",
            "thinker=4",
            "--tokens-per-audio-s",
            "thinker=2",
            "--uncached-tokens",
            "thinker=4",
            "--fill-steps",
            "thinker@1=2",
            "--json",
            str(output),
            *options,
        ],
    )
    assert counter.main() == 0
    counts = json.loads(output.read_text())["decoders"]["thinker"]
    expected_gib = streamed_elements * weight_bytes / 2**30
    assert counts["streamed_gib_per_step"] == pytest.approx(expected_gib)
    assert counts["kv_bytes_per_token"] == kv_elements_per_token * kv_bytes
    flags = {
        line.split()[0]: line.split("=", 1)[1].rstrip(" \\").split(",")
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("--")
    }
    decode = flags["--decode-bound"]
    assert float(decode[0]) == pytest.approx(expected_gib, rel=5e-4)
    assert float(decode[2]) == pytest.approx(kv_elements_per_token * kv_bytes * 4 / 2**30, rel=5e-4)
    assert decode[4] == compute_mode
    for flag, steps in (("--fixed-bound", 1), ("--fill-bound", 2)):
        assert flags[flag][1] == compute_mode
        assert float(flags[flag][2]) == pytest.approx(steps * expected_gib, rel=5e-4)
