#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from dataclasses import dataclass, field
from pathlib import PurePath

import regex as re

# Hub entry points that must go through the library-tagged ``hf_api()`` helper.
_HF_NAMES = r"snapshot_download|hf_hub_download|HfApi|HfFileSystem|get_safetensors_metadata"


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
            r"^\s*from\s+huggingface_hub\s+import\s*\([^)]*\b(?:" + _HF_NAMES + r")\b"
            r"|"
            r"^\s*from\s+huggingface_hub\s+import\b[^\n]*\b(?:" + _HF_NAMES + r")\b"
        ),
        tip=(
            "Use 'hf_api()' from 'vllm_omni.transformers_utils.repo_utils' (or "
            "add a tagged helper there) instead, so requests are tagged with "
            "vLLM-Omni's library info."
        ),
        allowed_files={"vllm_omni/transformers_utils/repo_utils.py"},
        allowed_dirs={"examples", "benchmarks"},
    ),
}


def check_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return_code = 0
    parts = PurePath(path).parts
    top_dir = parts[0] if parts else None
    # Check all patterns in the whole file
    for import_name, forbidden_import in CHECK_IMPORTS.items():
        # Skip files/directories that are allowed for this import
        if path in forbidden_import.allowed_files or top_dir in forbidden_import.allowed_dirs:
            continue
        # Search for forbidden imports
        for match in re.finditer(forbidden_import.pattern, content, re.MULTILINE):
            # Check if it's allowed
            if forbidden_import.allowed_pattern.match(match.group()):
                continue
            # Skip matches inside a comment
            line_start = content.rfind("\n", 0, match.start()) + 1
            if "#" in content[line_start : match.start()]:
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
    def matches(rule: str, content: str) -> bool:
        return bool(re.search(CHECK_IMPORTS[rule].pattern, content, re.MULTILINE))

    pickle_cases = [
        # Should match
        ("import pickle", True),
        ("import cloudpickle", True),
        ("import pickle as pkl", True),
        ("import cloudpickle as cpkl", True),
        ("from pickle import loads", True),
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
    for i, (content, should_match) in enumerate(pickle_cases):
        result = matches("pickle/cloudpickle", content)
        assert result == should_match, f"pickle case {i} failed: {content!r} (expected {should_match}, got {result})"

    hf_cases = [
        # Should match
        ("from huggingface_hub import snapshot_download", True),
        ("from huggingface_hub import hf_hub_download", True),
        ("from huggingface_hub import HfApi", True),
        ("from huggingface_hub import HfFileSystem", True),
        ("from huggingface_hub import get_safetensors_metadata", True),
        ("    from huggingface_hub import snapshot_download", True),
        ("from huggingface_hub import PyTorchModelHubMixin, hf_hub_download", True),
        ("from huggingface_hub import (snapshot_download)", True),
        # Parenthesized multi-line import must not bypass the hook
        ("from huggingface_hub import (\n    snapshot_download,\n)", True),
        ("from huggingface_hub import (\n    PyTorchModelHubMixin,\n    HfApi,\n)", True),
        # Should not match
        ("import huggingface_hub", False),
        ("import huggingface_hub as hf", False),
        ("from huggingface_hub import PyTorchModelHubMixin", False),
        ("from huggingface_hub import try_to_load_from_cache", False),
        ("from huggingface_hub.constants import HF_HUB_CACHE", False),
        ("from huggingface_hub.utils import EntryNotFoundError", False),
        ("from vllm_omni.transformers_utils.repo_utils import hf_api", False),
        ("from huggingface_hub import (\n    PyTorchModelHubMixin,\n)", False),
        ("# from huggingface_hub import snapshot_download", False),
    ]
    for i, (content, should_match) in enumerate(hf_cases):
        result = matches("huggingface_hub", content)
        assert result == should_match, (
            f"huggingface_hub case {i} failed: {content!r} (expected {should_match}, got {result})"
        )

    print("All regex tests passed.")


if __name__ == "__main__":
    if "--test-regex" in sys.argv:
        test_regex()
    else:
        sys.exit(main())
