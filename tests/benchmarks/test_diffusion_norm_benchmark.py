# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def _load_benchmark_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "benchmarks" / "kernels" / "bench_diffusion_norm_impls.py"
    spec = importlib.util.spec_from_file_location("bench_diffusion_norm_impls", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_diffusion_actual_shapes_exclude_multi_gpu_sources_by_default():
    bench = _load_benchmark_module()

    cases = bench.filter_cases(
        bench.actual_diffusion_shapes(),
        ops={"rmsnorm", "fused_add_rmsnorm", "layernorm"},
        models=None,
        shape_ids=None,
        include_multi_gpu_source_shapes=False,
        limit_cases=None,
    )

    shape_ids = {case.shape_id for case in cases}
    assert "qwen_rms_26x3584" in shape_ids
    assert "hunyuan_fused_add_140x4096" in shape_ids
    assert "mova_rms_44100x5120" not in shape_ids
    assert all(case.source_gpu_config == "1 GPU" for case in cases)


def test_provider_filter_avoids_unrequested_optional_providers():
    bench = _load_benchmark_module()

    case = bench.make_smoke_cases(["rmsnorm"])[0]
    tensors = bench.make_case_tensors(case, bench.dtype_from_name("fp32"), bench.torch.device("cpu"), seed=0)
    providers = bench.build_providers(case, tensors, bench.EPS, {"vllm_omni_native"})

    assert [provider.name for provider in providers] == ["vllm_omni_native"]


def test_cpu_smoke_writes_result_files(tmp_path):
    bench = _load_benchmark_module()
    args = argparse.Namespace(
        shape_preset="smoke",
        hidden_sizes="128",
        batch_sizes="4",
        ops="rmsnorm",
        dtypes="fp32",
        providers="torch_builtin",
        models="",
        shape_ids="",
        include_multi_gpu_source_shapes=False,
        limit_cases=1,
        device="cpu",
        eps=bench.EPS,
        warmup=0,
        iters=1,
        seed=0,
        atol=None,
        rtol=None,
        skip_correctness=True,
        benchmark_failed_correctness=False,
    )

    rows = bench.run_suite(args)
    assert len(rows) == 1
    assert rows[0].status == "ok"

    csv_path = tmp_path / "diffusion_norm_impls.csv"
    json_path = tmp_path / "diffusion_norm_impls.json"
    md_path = tmp_path / "diffusion_norm_impls_summary.md"

    bench.write_csv(rows, csv_path)
    bench.write_json(rows, json_path)
    bench.write_markdown(rows, md_path)

    assert csv_path.read_text(encoding="utf-8").startswith("op,provider,dtype")
    assert '"rows"' in json_path.read_text(encoding="utf-8")
    assert "Diffusion Norm Provider Benchmark" in md_path.read_text(encoding="utf-8")
