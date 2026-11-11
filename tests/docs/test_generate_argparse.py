# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT_DIR = Path(__file__).parents[2]
HOOK_PATH = ROOT_DIR / "docs/mkdocs/hooks/generate_argparse.py"
SPEC = importlib.util.spec_from_file_location("generate_argparse", HOOK_PATH)
assert SPEC and SPEC.loader
generate_argparse = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generate_argparse)


def test_static_serve_parser_supports_video_output_transport() -> None:
    parser = generate_argparse.create_parser_subparser_init(generate_argparse.OmniServeCommand)

    args = parser.parse_args(["--omni", "--video-output-transport", '{"enable_device_postprocess": true}'])

    assert args.video_output_transport == {"enable_device_postprocess": True}


@pytest.mark.parametrize("value", ["1", "2", "8"])
def test_static_serve_parser_supports_vae_tile_batch_size(value: str) -> None:
    parser = generate_argparse.create_parser_subparser_init(generate_argparse.OmniServeCommand)

    args = parser.parse_args(["--omni", "--vae-tile-batch-size", value])

    assert args.vae_tile_batch_size == int(value)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid"])
def test_static_serve_parser_rejects_invalid_vae_tile_batch_size(value: str) -> None:
    parser = generate_argparse.create_parser_subparser_init(generate_argparse.OmniServeCommand)

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--omni", f"--vae-tile-batch-size={value}"])

    assert exc_info.value.code == 2


def test_on_startup_generates_vae_tile_batch_size_docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "generated"
    monkeypatch.setattr(generate_argparse, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(generate_argparse, "ARGPARSE_DOC_DIR", output_dir)

    generate_argparse.on_startup("build", False)

    content = (output_dir / "omni_serve.inc.md").read_text(encoding="utf-8")
    assert "--vae-tile-batch-size" in content
    assert "Maximum decoder spatial tiles per rank per forward" in content
