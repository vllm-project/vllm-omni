# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Apply comparison patches only when needed, then prove each CPU contract."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

PATCHES = (
    ("sglang_omni", "sglang-omni-ignore-eos.patch", "test_sglang_sampling.py"),
    ("sglang", "sglang-diffusion-config-only-guider.patch", "test_sglang_diffusion_loader.py"),
    ("sglang", "sglang-diffusers-revision.patch", "test_sglang_diffusers_revision.py"),
    ("sglang", "sglang-video-image-reference.patch", "test_sglang_video_api.py"),
)


def main() -> None:
    patch_dir = Path(__file__).resolve().parent
    repository = patch_dir.parent.parent
    results = []
    for package, filename, test in PATCHES:
        spec = importlib.util.find_spec(package)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Comparison backend package is missing: {package}")
        source_root = Path(spec.origin).resolve().parent.parent
        patch = patch_dir / filename
        # A native SGLang installation can be editable and outside site-packages.
        command = ["git", "-C", "/tmp", "apply", "--unsafe-paths", f"--directory={source_root}"]
        check = subprocess.run([*command, "--check", str(patch)], check=False)
        mode = "upstream"
        if check.returncode == 0:
            subprocess.run([*command, str(patch)], check=True)
            mode = "applied"

        # A changed/already-fixed upstream is accepted only through the same
        # behavioral contract. Missing dependencies and skipped tests fail closed.
        with tempfile.TemporaryDirectory(prefix="sglang-contract-") as temporary:
            report = Path(temporary) / "results.xml"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--noconftest",
                    "-o",
                    "addopts=",
                    "-q",
                    f"--junitxml={report}",
                    str(repository / "tests/dfx/perf/tests" / test),
                ],
                cwd=repository,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1"},
                check=True,
            )
            suites = list(ET.parse(report).getroot().iter("testsuite"))
            if not suites or sum(int(suite.attrib["tests"]) for suite in suites) == 0:
                raise RuntimeError(f"No CPU contract tests ran for {filename}")
            if any(int(suite.attrib.get("skipped", "0")) for suite in suites):
                raise RuntimeError(f"CPU contract tests skipped for {filename}")
        results.append(
            {"file": filename, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(), "mode": mode, "test": test}
        )

    Path("/opt/sglang-omni-comparison-patches.json").write_text(
        json.dumps({"schema_version": 1, "patches": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
