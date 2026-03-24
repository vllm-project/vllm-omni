"""
Shared fixtures, helpers, and path constants for tests/examples/.
"""

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, cast

import mistune
import pytest
import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Path constants and fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "examples"

# Use Python tempfile instead of pytest's tmp_path_factory because
# OUTPUT_DIR is needed in test collection time, but tmp_path_factory is only available in test running time.
# It is needed during test collection because extract_readme_snippets replaces LoRA path with a generated one under OUTPUT_DIR,
# and extract_readme_snippets is called at collection time to generate separate test cases for each README code block.
OUTPUT_DIR = (
    REPO_ROOT / prefix
    if (prefix := os.environ.get("OUTPUT_DIR"))
    else Path(tempfile.mkdtemp(prefix="vllm_omni_test_examples_"))
)


# ---------------------------------------------------------------------------
# Code snippet extraction and asset file helpers
# ---------------------------------------------------------------------------

# parameters: language, code, h2_title
ReadmeSnippetExtractionSkipPredicate = Callable[[str, str, str], tuple[bool, str]]


class ReadmeSnippet(NamedTuple):
    language: str
    code: str
    h2_title: str
    index_in_section: int
    output_file_path: Path | None = None
    skip: tuple[bool, str] = (False, "")

    @property
    def test_id(self) -> str:
        return f"{ReadmeSnippet._slug(self.h2_title)}_{self.index_in_section:03d}"

    @staticmethod
    def extract_readme_snippets(
        readme_path: Path,
        skipif: ReadmeSnippetExtractionSkipPredicate | None = None,
    ) -> list["ReadmeSnippet"]:
        markdown = mistune.create_markdown(renderer="ast")
        tokens = markdown(readme_path.read_text(encoding="utf-8"))
        tokens = cast(list[dict[str, Any]], tokens)  # mistune's AST renderer always produces a list, not a str

        h2_title = ""
        section_counts: defaultdict[str, int] = defaultdict(int)
        snippets: list[ReadmeSnippet] = []

        for token in tokens:
            token_type = token.get("type")

            if token_type == "heading":
                level = (token.get("attrs") or {}).get("level")
                title = ReadmeSnippet._heading_text(token)
                if level == 2:
                    h2_title = title
                continue

            if token_type != "block_code":
                continue

            try:
                info = token.get("attrs").get("info")  # type: ignore[reportOptionalMemberAccess]
                language = info.strip().split()[0].lower()  # type: ignore[reportOptionalMemberAccess]

                # Common shell aliases to "bash" in several markdown renderers.
                if language in {"shell", "sh", "ksh", "zsh"}:
                    language = "bash"

                if language not in {"bash", "python"}:
                    continue
            except AttributeError:
                # The fence is missing explicit language info; skip it.
                continue

            key = h2_title
            section_counts[key] += 1
            code = token.get("raw", "")
            output_file_path = None
            if language == "bash":
                argv = ReadmeSnippet._normalize_bash_command(code, Path(readme_path.parent))
                code = shlex.join(argv)
                output_file_path = ReadmeSnippet._output_file_path_from_argv(argv)
            if skipif is not None:
                skip_config = skipif(language, code, h2_title)
            else:
                skip_config = (False, "")
            snippet = ReadmeSnippet(
                language=language,
                code=code,
                h2_title=h2_title,
                index_in_section=section_counts[key],
                output_file_path=output_file_path,
                skip=skip_config,
            )
            snippets.append(snippet)

        return snippets

    @staticmethod
    def _normalize_bash_command(command: str, readme_dir: Path) -> list[str]:
        line_joined_command = re.sub(r"\\\s*\n", " ", command).strip()
        argv = shlex.split(line_joined_command, comments=True)
        assert argv, "README bash fence produced an empty command"

        # Normalize python directory and example script location
        if argv[0] in {"python", "python3"}:
            argv[0] = sys.executable
            if len(argv) > 1 and argv[1].endswith(".py"):
                script_arg = argv[1]
                script_path = Path(script_arg)
                if script_path.is_absolute():
                    resolved_script = script_path
                else:
                    # Take the file name only, and append script_dir to its front
                    resolved_script = readme_dir / script_path.name
                assert resolved_script.exists(), (
                    f"README bash snippet references a script that does not exist: {script_arg} (resolved to {resolved_script})"
                )
                argv[1] = str(resolved_script)

        # Normalize LoRA adapter path and ensure README LoRA assets exist.
        try:
            lora_arg_idx = argv.index("--lora-path")  # Raise ValueError if not found
            assert len(argv) > lora_arg_idx + 1, "README bash snippet uses --lora-path without a following value"

            lora_dir = OUTPUT_DIR / "lora"
            adapter_model = lora_dir / "adapter_model.safetensors"
            adapter_config = lora_dir / "adapter_config.json"
            if not adapter_model.exists() or not adapter_config.exists():
                write_zimage_lora(lora_dir, v_scale=8.0)

            argv[lora_arg_idx + 1] = str(lora_dir)
        except ValueError:
            pass

        return argv

    @staticmethod
    def _output_file_path_from_argv(argv: list[str]) -> Path | None:
        if "--output" not in argv:
            return None
        output_param_idx = argv.index("--output")
        assert len(argv) > output_param_idx + 1, "README bash snippet uses --output without a following value"
        output_arg = argv[output_param_idx + 1]
        return Path(output_arg)

    @staticmethod
    def _slug(text: str) -> str:
        return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")

    @staticmethod
    def _heading_text(token: dict) -> str:
        return "".join(child.get("raw", "") for child in token.get("children", [])).strip()


# [TODO] Duplicate `_write_zimage_lora` in tests/e2e/online_serving/test_images_generations_lora.py. Combine these helpers and tests/e2e/offline_inference/test_diffusion_lora.py to test/utils later
def write_zimage_lora(adapter_dir: Path, *, q_scale: float = 0.0, k_scale: float = 0.0, v_scale: float = 0.0):
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Z-Image transformer uses dim=3840 by default.
    dim = 3840
    module_name = "transformer.layers.0.attention.to_qkv"
    rank = 1

    lora_a = torch.zeros((rank, dim), dtype=torch.float32)
    lora_a[0, 0] = 1.0

    # QKVParallelLinear packs (Q, K, V) => out dim is 3 * dim (tp=1).
    lora_b = torch.zeros((3 * dim, rank), dtype=torch.float32)
    if q_scale:
        lora_b[:dim, 0] = q_scale
    if k_scale:
        lora_b[dim : 2 * dim, 0] = k_scale
    if v_scale:
        lora_b[2 * dim :, 0] = v_scale

    save_file(
        {
            f"base_model.model.{module_name}.lora_A.weight": lora_a,
            f"base_model.model.{module_name}.lora_B.weight": lora_b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": rank,
                "target_modules": [module_name],
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Code runner and subprocess helpers
# ---------------------------------------------------------------------------


class ExampleRunResult(NamedTuple):
    run_dir: Path
    assets: list[Path]


class ExampleRunner:
    """Run extracted README snippets and return generated assets.

    The output materials are organized in a three-level directory structure:
    - Set at init: `self.output_root` for all tests (from env OUTPUT_DIR)
    - Set at `self.run(...)`: `output_subfolder` for a specific example page (e.g., `example_offline_t2i`)
    - Generated by `extract_readme_snippets`: `snippet.test_id` for a specific code block (matching H2 titles, e.g., `basic_usage_001`)
    """

    IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    def run(
        self, snippet: ReadmeSnippet, *, output_subfolder: Path = Path("."), env: dict[str, str] | None = None
    ) -> ExampleRunResult:
        run_dir = self.output_root / output_subfolder / snippet.test_id
        run_dir.mkdir(parents=True, exist_ok=True)

        if snippet.language == "python":
            assets = self._run_python_snippet(snippet, run_dir, env)
            return ExampleRunResult(run_dir=run_dir, assets=assets)

        if snippet.language == "bash":
            asset = self._run_bash_snippet(snippet, run_dir, env)
            return ExampleRunResult(run_dir=run_dir, assets=[asset])

        raise AssertionError(f"Unsupported snippet language: {snippet.language}")

    def _run_python_snippet(
        self, snippet: ReadmeSnippet, run_dir: Path, env: dict[str, str] | None = None
    ) -> list[Path]:
        # Saving the script to a temporary file and `run_cmd` it.
        # Not using `exec(snippet.code)` because the output is lost.
        script_path = run_dir / "snippet.py"
        script_path.write_text(snippet.code, encoding="utf-8")

        before = self._collect_images(run_dir)
        run_cmd([sys.executable, str(script_path)], cwd=run_dir, env=env)
        after = self._collect_images(run_dir)

        assets = sorted(after - before)
        return assets

    def _run_bash_snippet(self, snippet: ReadmeSnippet, run_dir: Path, env: dict[str, str] | None = None) -> Path:
        run_cmd(snippet.code, shell=True, cwd=run_dir, env=env)

        assert snippet.output_file_path is not None, (
            f"README bash snippet is missing --output argument: {snippet.test_id}. "
            "The test script cannot guess the output file path."
        )

        # If the code snippet declares a relative path for the output file, append this path to the parent output collection directory.
        # If the code snippet declares an absolute path (not likely but just in case), the return value resolution removes `run_dir`, also correctly pointing to this file.
        return run_dir / snippet.output_file_path

    def _collect_images(self, root: Path) -> set[Path]:
        return {path for path in root.rglob("*") if path.suffix.lower() in self.IMAGE_SUFFIXES}


@pytest.fixture
def example_runner() -> ExampleRunner:
    return ExampleRunner(output_root=OUTPUT_DIR)


def run_cmd(
    command: list[str] | str,
    *,
    shell: bool = False,
    env: dict[str, str] | None = None,
    cwd: Path | str | None = None,
) -> str:
    """Run a command as a subprocess; assert zero exit code and return stdout.

    Output is fully captured and returned as a string so callers can parse it
    (e.g. with :func:`extract_content_after_keyword`).
    Use this for scripts whose printed output is part of the test assertion.
    """
    if env is not None:
        env = {**os.environ.copy(), **env}
    result = subprocess.run(command, capture_output=True, text=True, shell=shell, env=env, cwd=cwd)

    if result.returncode != 0:
        print(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, command)

    all_output = result.stdout
    print(f"All output:\n{all_output}")
    return all_output


# ---------------------------------------------------------------------------
# Output validation helpers
# ---------------------------------------------------------------------------


def extract_content_after_keyword(keywords: str, text: str) -> str:
    """Return the text that follows *keywords* in *text* (regex match).

    Raises ``AssertionError`` if the keyword is not found, so test failures
    produce a clear message pointing at the missing keyword.
    """
    matches = re.findall(rf"{keywords}\s*(.+)", text, re.DOTALL)

    if not matches:
        raise AssertionError(f"Keywords {keywords} not found in provided text output")
    return matches[0]
