"""Exercise public-install guards without downloading or using accelerator devices."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]


class PublicInstallGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "checkout"
        self.scripts = self.root / "examples/cosmos3"
        (self.scripts / "environment").mkdir(parents=True)
        for name in ("install_public.sh", "env_public.sh", "env_native.sh"):
            shutil.copyfile(SCRIPTS / name, self.scripts / name)
        shutil.copyfile(
            SCRIPTS / "environment/public-versions.env",
            self.scripts / "environment/public-versions.env",
        )
        self.environment = os.environ.copy()
        self.environment["COSMOS3_BUILD_JOBS"] = "2"

    def fake_arm(self):
        tools = self.root / "test-bin"
        tools.mkdir()
        uname = tools / "uname"
        uname.write_text("#!/usr/bin/env bash\nprintf 'aarch64\\n'\n")
        uname.chmod(0o755)
        self.environment["PATH"] = str(tools) + os.pathsep + self.environment["PATH"]
        return tools

    def run_stage(self, stage):
        return subprocess.run(
            ["bash", str(self.scripts / "install_public.sh"), stage],
            env=self.environment,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_plan_does_not_create_runtime_or_venv(self):
        result = self.run_stage("--plan")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("COMPILE_CUSTOM_KERNELS=0", result.stdout)
        self.assertFalse((self.root / ".runtime").exists())
        self.assertFalse((self.root / ".venv-public").exists())

    def test_existing_venv_is_preserved(self):
        self.fake_arm()
        venv = self.root / ".venv-public"
        venv.mkdir()
        sentinel = venv / "keep.txt"
        sentinel.write_text("existing environment")
        result = self.run_stage("python")
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertEqual(sentinel.read_text(), "existing environment")
        self.assertFalse((self.root / ".runtime/public-builds/python").exists())

    def test_bad_python_source_hash_stops_before_build(self):
        tools = self.fake_arm()
        curl = tools / "curl"
        curl.write_text('#!/usr/bin/env bash\nprintf bad-source > "${!#}"\n')
        curl.chmod(0o755)
        result = self.run_stage("python")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("FAILED", result.stdout)
        self.assertFalse((self.root / ".runtime/public-builds/python").exists())
        self.assertFalse((self.root / ".venv-public").exists())

    def test_failing_cann_environment_keeps_nounset_and_returns_failure(self):
        runtime = self.root / ".runtime"
        runtime.mkdir()
        cann = runtime / "cann-env.sh"
        cann.write_text("return 17\n")
        (runtime / "public-runtime.env").write_text(f"export COSMOS3_CANN_ENV='{cann}'\n")
        result = subprocess.run(
            [
                "bash",
                "-uc",
                'source "$1"; status=$?; [[ $- == *u* ]] || exit 99; exit "$status"',
                "_",
                str(self.scripts / "env_public.sh"),
            ],
            env=self.environment,
            text=True,
            capture_output=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 17, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
