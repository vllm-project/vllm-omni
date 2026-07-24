#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from dataclasses import dataclass, field
from pathlib import PurePath

import regex as re


@dataclass
class ForbiddenImport:
    pattern: str
    tip: str
    allowed_pattern: re.Pattern = re.compile(r"^$")  # matches nothing by default
    allowed_files: set[str] = field(default_factory=set)
    allowed_dirs: set[str] = field(default_factory=set)


CHECK_IMPORTS = {
    # STOP AND READ BEFORE YOU ADD ANYTHING ELSE TO THIS LIST:
    #  The pickle and cloudpickle modules are known to be unsafe when
    #  deserializing data from potentially untrusted parties. They have resulted
    #  in multiple CVEs for vLLM and numerous vulnerabilities in the Python
    #  ecosystem more broadly. Before adding new uses of pickle/cloudpickle,
    #  please consider safer alternatives like msgpack or pydantic that are
    #  already in use in vLLM. Only add to this list if absolutely necessary and
    #  after careful security review.
    "pickle/cloudpickle": ForbiddenImport(
        pattern=(
            r"^\s*(import\s+(pickle|cloudpickle)(\s|$|\sas)"
            r"|from\s+(pickle|cloudpickle)\s+import\b)"
        ),
        tip=("Avoid using pickle or cloudpickle or add this file to tools/pre_commit/check_forbidden_imports.py."),
        allowed_files={
            "tests/helpers/process.py",
            "vllm_omni/diffusion/distributed/group_coordinator.py",
            "tests/diffusion/attention/test_attention_sp.py",
        },
    ),
    "huggingface_hub": ForbiddenImport(
        pattern=(
            r"^\s*from\s+huggingface_hub\s+import\b[^\n]*"
            r"\b(?:snapshot_download|hf_hub_download|HfApi|HfFileSystem"
            r"|get_safetensors_metadata)\b"
        ),
        tip=(
            "Use 'hf_api()' / 'hf_fs()' from "
            "'vllm_omni.transformers_utils.repo_utils' instead, so requests are "
            "tagged with vLLM-Omni's library info."
        ),
        allowed_files={"vllm_omni/transformers_utils/repo_utils.py"},
        allowed_dirs={"examples", "benchmarks"},
    ),
}


def check_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return_code = 0
    # Check all patterns in the whole file
    for import_name, forbidden_import in CHECK_IMPORTS.items():
        # Skip files that are allowed for this import
        if path in forbidden_import.allowed_files:
            continue
        # Skip files whose top-level directory is exempt from this import
        parts = PurePath(path).parts
        if parts and parts[0] in forbidden_import.allowed_dirs:
            continue
        # Search for forbidden imports
        for match in re.finditer(forbidden_import.pattern, content, re.MULTILINE):
            # Check if it's allowed
            if forbidden_import.allowed_pattern.match(match.group()):
                continue
            # Calculate line number from match position
            line_num = content[: match.start() + 1].count("\n") + 1
            print(
                f"{path}:{line_num}: "
                "\033[91merror:\033[0m "  # red color
                f"Found forbidden import: {import_name}. {forbidden_import.tip}"
            )
            return_code = 1
    return return_code


def main():
    returncode = 0
    for path in sys.argv[1:]:
        returncode |= check_file(path)
    return returncode


def test_regex():
    pickle_cases = [
        # Should match
        ("import pickle", True),
        ("import cloudpickle", True),
        ("import pickle as pkl", True),
        ("import cloudpickle as cpkl", True),
        ("from pickle import *", True),
        ("from cloudpickle import dumps", True),
        ("from pickle import dumps, loads", True),
        ("from cloudpickle import (dumps, loads)", True),
        ("    import pickle", True),
        ("\timport cloudpickle", True),
        ("from   pickle   import   loads", True),
        # Should not match
        ("import somethingelse", False),
        ("from somethingelse import pickle", False),
        ("# import pickle", False),
        ("print('import pickle')", False),
        ("import pickleas as asdf", False),
    ]
    pickle_pattern = re.compile(CHECK_IMPORTS["pickle/cloudpickle"].pattern)
    for i, (line, should_match) in enumerate(pickle_cases):
        result = bool(pickle_pattern.match(line))
        assert result == should_match, f"pickle case {i} failed: '{line}' (expected {should_match}, got {result})"

    hf_cases = [
        # Should match
        ("from huggingface_hub import snapshot_download", True),
        ("from huggingface_hub import hf_hub_download", True),
        ("from huggingface_hub import HfApi", True),
        ("from huggingface_hub import HfFileSystem", True),
        ("from huggingface_hub import get_safetensors_metadata", True),
        ("    from huggingface_hub import snapshot_download", True),
        ("from huggingface_hub import PyTorchModelHubMixin, hf_hub_download", True),
        # Should not match
        ("import huggingface_hub", False),
        ("import huggingface_hub as hf", False),
        ("from huggingface_hub import PyTorchModelHubMixin", False),
        ("from huggingface_hub import try_to_load_from_cache", False),
        ("from huggingface_hub.constants import HF_HUB_CACHE", False),
        ("from huggingface_hub.utils import EntryNotFoundError", False),
        ("from vllm_omni.transformers_utils.repo_utils import hf_api", False),
        ("# resolves via ``huggingface_hub.snapshot_download``", False),
        ('    """Falls back to snapshot_download for remote repos."""', False),
    ]
    hf_pattern = re.compile(CHECK_IMPORTS["huggingface_hub"].pattern, re.MULTILINE)
    for i, (line, should_match) in enumerate(hf_cases):
        result = bool(hf_pattern.search(line))
        assert result == should_match, (
            f"huggingface_hub case {i} failed: '{line}' (expected {should_match}, got {result})"
        )

    print("All regex tests passed.")


if __name__ == "__main__":
    if "--test-regex" in sys.argv:
        test_regex()
    else:
        sys.exit(main())
