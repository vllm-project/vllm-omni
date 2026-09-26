# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""conftest.py for benchmarks unit tests.

Installs lightweight mock stubs for ``vllm`` (and sub-packages) so the
data-module unit tests can run without a full vLLM installation.  Only the
symbols actually imported by
``vllm_omni.benchmarks.data_modules.seed_tts_dataset`` are emulated.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any


def _install_vllm_stubs() -> None:
    """Register minimal vllm stubs in sys.modules.

    Only installs when real vllm is unavailable.  We actively probe the
    import because an empty or partial vllm may not yet have imported
    the submodules we rely on, and unconditionally registering stubs
    would shadow the real package for sibling tests (e.g.
    ``tests/benchmarks/metrics/test_metrics.py`` needs the real
    ``vllm.benchmarks.serve``).
    """
    try:
        import vllm.benchmarks.datasets  # noqa: F401
        import vllm.tokenizers  # noqa: F401
    except ImportError:
        pass
    else:
        return  # real vllm available — do not shadow it
    if "vllm.benchmarks.datasets" in sys.modules:
        return

    # ------------------------------------------------------------------ #
    # vllm.benchmarks.datasets                                            #
    # ------------------------------------------------------------------ #
    @dataclass
    class SampleRequest:
        prompt: str = ""
        prompt_len: int = 0
        expected_output_len: int = 0
        multi_modal_data: Any = None
        request_id: str = ""

    class BenchmarkDataset:
        def __init__(
            self,
            dataset_path: str = "",
            random_seed: int = 0,
            disable_shuffle: bool = False,
            **kwargs: Any,
        ) -> None:
            self.dataset_path = dataset_path
            self.random_seed = random_seed
            self.disable_shuffle = disable_shuffle

        def maybe_oversample_requests(
            self,
            out: list,
            num_requests: int,
            request_id_prefix: str,
            no_oversample: bool,
        ) -> None:
            pass

    # ------------------------------------------------------------------ #
    # vllm.tokenizers / vllm.tokenizers.hf                               #
    # ------------------------------------------------------------------ #
    class TokenizerLike:
        pass

    def get_cached_tokenizer(t: Any) -> Any:
        return t

    # ------------------------------------------------------------------ #
    # Wire up sys.modules                                                 #
    # ------------------------------------------------------------------ #
    vllm_mod = types.ModuleType("vllm")
    vllm_benchmarks = types.ModuleType("vllm.benchmarks")
    vllm_benchmarks_datasets = types.ModuleType("vllm.benchmarks.datasets")
    vllm_tokenizers = types.ModuleType("vllm.tokenizers")
    vllm_tokenizers_hf = types.ModuleType("vllm.tokenizers.hf")

    vllm_benchmarks_datasets.BenchmarkDataset = BenchmarkDataset  # type: ignore[attr-defined]
    vllm_benchmarks_datasets.SampleRequest = SampleRequest  # type: ignore[attr-defined]
    vllm_tokenizers.TokenizerLike = TokenizerLike  # type: ignore[attr-defined]
    vllm_tokenizers_hf.get_cached_tokenizer = get_cached_tokenizer  # type: ignore[attr-defined]

    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.benchmarks"] = vllm_benchmarks
    sys.modules["vllm.benchmarks.datasets"] = vllm_benchmarks_datasets
    sys.modules["vllm.tokenizers"] = vllm_tokenizers
    sys.modules["vllm.tokenizers.hf"] = vllm_tokenizers_hf


# Install stubs immediately at collection time (before any test import).
_install_vllm_stubs()
