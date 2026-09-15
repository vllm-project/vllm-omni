# SPDX-License-Identifier: Apache-2.0
"""CPU-only budget arithmetic and pre-launch wiring tests.

Run directly with Python when the GPU/vLLM test dependencies are unavailable.
Only logging/formatting imports are stubbed; the production ledger is executed.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace as Ns
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


budget = load("_static_hbm_test_budget", "vllm_omni/config/static_budget.py")
with patch.dict(sys.modules, {
    "vllm.logger": Ns(init_logger=logging.getLogger),
    "vllm.utils.mem_utils": Ns(format_gib=lambda n: n / 1024**3),
}):
    admission = load("_static_hbm_test_admission", "vllm_omni/engine/stage_admission.py")
G = 1024**3


def replica(stage, limit, devices, diffusion=False):
    return Ns(
        metadata=Ns(stage_id=stage), replica_id=0, devices=devices,
        stage_cfg=Ns(engine_args={"hbm_limit_gb": limit, "hbm_reserved_gb": 2, "diffusion_kv_mode": "paged_scheduler"}),
        stage_vllm_config=None if diffusion else Ns(
            model_config=Ns(hbm_limit_gb=limit, hbm_reserved_gb=2),
            cache_config=Ns(gpu_memory_utilization=0.9),
        ),
    )


class StaticBudgetTests(unittest.TestCase):
    def check(self, replicas):
        with patch.dict(sys.modules, {"vllm_omni.config.static_budget": budget}):
            return admission.check_admission(
                [Ns(replicas=replicas)],
                resolve_physical_devices=lambda r: r.devices,
                device_total_memory=lambda _: 80 * G,
            )

    def test_derive_total_not_kv(self):
        self.assertEqual(budget.derive_kv_budget(24, 2, 14 * G), 8 * G)

    def test_explicit_kv_cannot_bypass_total(self):
        self.assertEqual(budget.derive_kv_budget(24, 2, 14 * G, 20 * G), 8 * G)
        self.assertEqual(budget.derive_kv_budget(24, 2, 14 * G, 4 * G), 4 * G)

    def test_profile_exhaustion(self):
        for usage in (22 * G, 25 * G):
            with self.assertRaisesRegex(ValueError, "exhausted"):
                budget.derive_kv_budget(24, 2, usage)

    def test_invalid_limits(self):
        for value in (0, -1, True, float("nan"), float("inf"), "24"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget.budget_bytes(value)

    def test_invalid_reserve(self):
        for value in (-1, True, float("inf"), 24):
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget.budget_bytes(24, value)

    def test_legacy_opt_out(self):
        self.assertIsNone(budget.budget_bytes(None))

    def test_initial_free_memory(self):
        self.assertEqual(budget.initial_budget(24, 2, 24 * G), 24 * G)
        with self.assertRaises(ValueError):
            budget.initial_budget(24, 2, 23 * G)

    def test_shared_ar_diffusion(self):
        ledger = self.check([replica(0, 24, [0]), replica(1, 48, [0], True)])[0]
        self.assertEqual(ledger.required_bytes, 74 * G)  # includes 2 GiB public slack
        self.assertEqual(ledger.graph_reserve_bytes, 0)  # included in stage reserve

    def test_overcommitted_before_launch(self):
        with self.assertRaises(admission.StageAdmissionError):
            self.check([replica(0, 48, [0]), replica(1, 48, [0], True)])

    def test_distinct_devices(self):
        ledgers = self.check([replica(0, 48, [0]), replica(1, 48, [1])])
        self.assertEqual(set(ledgers), {0, 1})

    def test_tp_budget_is_per_device(self):
        ledgers = self.check([replica(0, 24, [0, 1]), replica(1, 48, [1])])
        self.assertEqual(ledgers[0].required_bytes, 26 * G)
        self.assertEqual(ledgers[1].required_bytes, 74 * G)

    def test_replicas_accumulate(self):
        a = replica(0, 40, [0])
        b = replica(0, 40, [0])
        b.replica_id = 1
        with self.assertRaises(admission.StageAdmissionError):
            self.check([a, b])

    def test_unbudgeted_stage_rejected(self):
        with self.assertRaisesRegex(admission.StageAdmissionError, "explicit budgets"):
            self.check([replica(0, 24, [0]), replica(1, None, [0])])

    def test_unknown_or_remote_device_rejected(self):
        for devices in (None, admission.ADMISSION_EXEMPT):
            with self.subTest(devices=devices), self.assertRaises(admission.StageAdmissionError):
                self.check([replica(0, 24, devices)])

    def test_dense_diffusion_is_not_silently_admitted(self):
        d = replica(1, 24, [0], True)
        d.stage_cfg.engine_args["diffusion_kv_mode"] = "dense_legacy"
        with self.assertRaisesRegex(admission.StageAdmissionError, "paged_scheduler"):
            self.check([d])

    def test_generation_worker_is_not_silently_admitted(self):
        r = replica(0, 24, [0])
        r.stage_vllm_config.model_config.worker_type = "generation"
        with self.assertRaisesRegex(admission.StageAdmissionError, "generation"):
            self.check([r])

    def test_legacy_utilization_unchanged(self):
        d = admission.StageDemand(0, 0, [0], 0.5, 2 * G)
        ledger = admission.evaluate([d], {0: 80 * G})[0]
        self.assertEqual(ledger.required_bytes, 44 * G)


class WorkerBudgetTests(unittest.TestCase):
    """Execute the production profiling method with a fake runner/profile.

    Extract only this method to avoid importing CUDA-dependent worker classes.
    This does not test vLLM construction or GPU allocations.
    """
    def run_profile(self, total, explicit=None):
        tree = ast.parse((ROOT / "vllm_omni/worker/base.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniGPUWorkerBase")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "determine_available_memory")
        method.decorator_list = []
        namespace = {
            "logger": Ns(debug=lambda *a, **k: None, info=lambda *a, **k: None,
                         info_once=lambda *a, **k: None),
            "os": Ns(getpid=lambda: 1), "format_gib": lambda n: n / G,
            "current_omni_platform": Ns(is_rocm=lambda: False),
            "memory_profiling": lambda *a, **k: nullcontext(Ns(
                non_torch_increase=G, torch_peak_increase=3 * G, total_consumed=14 * G,
            )),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "worker/base.py", "exec"), namespace)
        calls = []
        worker = Ns(
            vllm_config=Ns(model_config=Ns(hbm_limit_gb=total, hbm_reserved_gb=2, stage_id=0)),
            cache_config=Ns(kv_cache_memory_bytes=explicit),
            model_runner=Ns(model_memory_usage=10 * G, profile_run=lambda: calls.append("profile")),
            init_snapshot=object(), requested_memory=24 * G, rank=0, local_rank=0,
        )
        with patch.dict(sys.modules, {"vllm_omni.config.static_budget": budget}):
            result = namespace["determine_available_memory"](worker)
        self.assertEqual(calls, ["profile"])
        return result

    def test_static_worker_profiles_even_with_override(self):
        self.assertEqual(self.run_profile(24, 20 * G), 8 * G)

    def test_static_worker_honors_smaller_override(self):
        self.assertEqual(self.run_profile(24, 4 * G), 4 * G)

    def test_legacy_explicit_budget_preserved(self):
        self.assertEqual(self.run_profile(None, 20 * G), 20 * G)

    def test_legacy_profile_budget_preserved(self):
        self.assertEqual(self.run_profile(None), 10 * G)


if __name__ == "__main__":
    unittest.main()
