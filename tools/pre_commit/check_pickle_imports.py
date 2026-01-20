#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys

import regex as re

# List of files (relative to repo root) that are allowed to import pickle or
# cloudpickle
#
# STOP AND READ BEFORE YOU ADD ANYTHING ELSE TO THIS LIST:
#  The pickle and cloudpickle modules are known to be unsafe when deserializing
#  data from potentially untrusted parties. They have resulted in multiple CVEs
#  for vLLM and numerous vulnerabilities in the Python ecosystem more broadly.
#  Before adding new uses of pickle/cloudpickle, please consider safer
#  alternatives like msgpack or pydantic that are already in use in vLLM. Only
#  add to this list if absolutely necessary and after careful security review.
ALLOWED_FILES = {
    "vllm_omni/entrypoints/omni_llm.py",
    "tests/e2e/offline_inference/utils.py",
    "tests/utils.py",
    "vllm_omni/diffusion/distributed/group_coordinator.py",
    "tests/diffusion/attention/test_sequence_parallel.py",
}

PICKLE_RE = re.compile(
    r"^\s*(import\s+(pickle|cloudpickle)(\s|$|\sas)"
    r"|from\s+(pickle|cloudpickle)\s+import\b)"
)


def scan_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if PICKLE_RE.match(line):
                print(
                    f"{path}:{i}: "
                    "\033[91merror:\033[0m "  # red color
                    "Found pickle/cloudpickle import"
                )
                return 1
    return 0


def main():
    returncode = 0
    for filename in sys.argv[1:]:
        if filename in ALLOWED_FILES:
            continue
        returncode |= scan_file(filename)
    return returncode


def test_regex():
    test_cases = [
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
    for i, (line, should_match) in enumerate(test_cases):
        result = bool(PICKLE_RE.match(line))
        assert result == should_match, f"Test case {i} failed: '{line}' (expected {should_match}, got {result})"
    print("All regex tests passed.")


if __name__ == "__main__":
    if "--test-regex" in sys.argv:
        test_regex()
    else:
        sys.exit(main())
