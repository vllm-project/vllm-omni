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
