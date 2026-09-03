# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

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


def test_static_serve_parser_supports_cfg_companion_timeout() -> None:
    """The docs AST extractor must provide every CLI type helper it executes."""
    parser = generate_argparse.create_parser_subparser_init(generate_argparse.OmniServeCommand)
    action = next(action for action in parser._actions if action.dest == "cfg_companion_timeout")

    assert action.type("2.5") == 2.5
