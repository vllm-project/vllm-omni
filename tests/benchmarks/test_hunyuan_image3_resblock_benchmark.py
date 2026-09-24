import subprocess
from pathlib import Path

import pytest

import vllm_omni
from benchmarks.kernels import hunyuan_image3_resblock_benchmarks as benchmark

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("alternate_install", (False, True), ids=("checkout", "alternate-install"))
def test_source_provenance_tracks_imported_package(
    alternate_install: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    expected_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    expected_package_file = Path(vllm_omni.__file__).resolve()
    if alternate_install:
        expected_package_file = tmp_path / "installed" / "vllm_omni" / "__init__.py"
        expected_package_file.parent.mkdir(parents=True)
        expected_package_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(vllm_omni, "__file__", str(expected_package_file))
    monkeypatch.chdir(tmp_path)

    metadata = benchmark._source_provenance()

    assert metadata == {
        "commit": expected_commit,
        "benchmark_file": str(repo_root / "benchmarks" / "kernels" / "hunyuan_image3_resblock_benchmarks.py"),
        "vllm_omni_file": str(expected_package_file.resolve()),
    }
