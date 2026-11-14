# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT_DIR = Path(__file__).parents[2]
HOOK_PATH = ROOT_DIR / "docs/mkdocs/hooks/generate_api_readme.py"
SPEC = importlib.util.spec_from_file_location("generate_api_readme", HOOK_PATH)
assert SPEC and SPEC.loader
generate_api_readme = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_api_readme)


def test_block_table_summary_uses_qualified_nested_class_names():
    module_path = "vllm_omni.worker.v1_compat.block_table"
    source = ROOT_DIR / "vllm_omni/worker/v1_compat/block_table.py"

    classes, functions = generate_api_readme.parse_file_for_symbols(source, module_path)

    assert classes == [
        f"{module_path}.SlotMappingMode",
        f"{module_path}.BlockTable",
        f"{module_path}.MultiGroupBlockTable",
        f"{module_path}.ComputeSlotMappingKernel",
        f"{module_path}.ComputeSlotMappingKernel.CompileKey",
    ]
    assert functions == [f"{module_path}.get_block_table_width"]
    readme = generate_api_readme.generate_readme({"Workers": classes})
    assert f"- [{module_path}.ComputeSlotMappingKernel.CompileKey][]" in readme
    assert f"- [{module_path}.CompileKey][]" not in readme
