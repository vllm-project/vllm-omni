# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import logging
import re
import sys
from argparse import SUPPRESS, Action, ArgumentParser, HelpFormatter, _ArgumentGroup
from collections.abc import Iterable
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

from pydantic_core import core_schema


class _FlexibleArgumentParser(ArgumentParser):
    """Fallback parser for docs when vllm is unavailable.

    Accepts the 'deprecated' kwarg used by vllm CLI and emits warnings
    if a deprecated argument is actually provided.
    """

    _deprecated: set[Action] = set()
    _deprecated_warned: set[str] = set()

    if sys.version_info < (3, 13):

        def parse_known_args(self, args=None, namespace=None):
            namespace, args = super().parse_known_args(args, namespace)
            for action in _FlexibleArgumentParser._deprecated:
                if (
                    hasattr(namespace, dest := action.dest)
                    and getattr(namespace, dest) != action.default
                    and dest not in _FlexibleArgumentParser._deprecated_warned
                ):
                    _FlexibleArgumentParser._deprecated_warned.add(dest)
                    logger.warning("argument '%s' is deprecated", dest)
            return namespace, args

    def add_argument(self, *args, **kwargs):
        deprecated = kwargs.pop("deprecated", False)
        action = super().add_argument(*args, **kwargs)
        if deprecated:
            _FlexibleArgumentParser._deprecated.add(action)
        return action

    class _FlexibleArgumentGroup(_ArgumentGroup):
        def add_argument(self, *args, **kwargs):
            deprecated = kwargs.pop("deprecated", False)
            action = super().add_argument(*args, **kwargs)
            if deprecated:
                _FlexibleArgumentParser._deprecated.add(action)
            return action

    def add_argument_group(self, *args, **kwargs):
        group = self._FlexibleArgumentGroup(self, *args, **kwargs)
        self._action_groups.append(group)
        return group


logger = logging.getLogger("mkdocs")

# Define root and doc output directories
ROOT_DIR = Path(__file__).parent.parent.parent.parent
ARGPARSE_DOC_DIR = ROOT_DIR / "docs/generated/argparse_omni"

# Ensure the repo root is in sys.path for dynamic imports
sys.path.insert(0, str(ROOT_DIR))


class PydanticMagicMock(MagicMock):
    """`MagicMock` that's able to generate pydantic-core schemas and avoids _mock_name=None errors."""

    def __init__(self, *args, **kwargs):
        name = kwargs.pop("name", None)
        super().__init__(*args, **kwargs)
        self.__spec__ = ModuleSpec(name, None)
        self._mock_name = name or "mock"

    def __get_pydantic_core_schema__(self, source_type, handler):
        return core_schema.any_schema()


# --- Static extraction for CLI argument docs ---


def extract_omni_serve_subparser_init():
    """
    Statically parse vllm_omni/entrypoints/cli/serve.py to extract the subparser_init method
    and return a callable that adds arguments to a parser. This avoids import and mock issues.
    """
    serve_path = ROOT_DIR / "vllm_omni" / "entrypoints" / "cli" / "serve.py"
    with open(serve_path, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source, filename=str(serve_path))
    # Find class OmniServeCommand
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "OmniServeCommand":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "subparser_init":
                    # We'll exec this function body in a dummy context
                    func_code = ast.Module(body=[item], type_ignores=[])
                    code = compile(func_code, filename=str(serve_path), mode="exec")
                    # Prepare dummy context
                    local_vars = {}

                    # Provide a dummy subparsers with add_parser returning our parser
                    class DummySubparsers:
                        def add_parser(self, name, **kwargs):
                            return _FlexibleArgumentParser(prog=name)

                    dummy_subparsers = DummySubparsers()
                    # Provide globals for exec
                    # Extract DESCRIPTION from the source file (assume it's a triple-quoted string at the top)
                    m = re.search(r'DESCRIPTION\s*=\s*([ru]?""".*?""")', source, re.DOTALL)
                    DESCRIPTION = m.group(1)[3:-3] if m else ""
                    exec_globals = {
                        "_FlexibleArgumentParser": _FlexibleArgumentParser,
                        "FlexibleArgumentParser": _FlexibleArgumentParser,
                        "make_arg_parser": lambda parser: parser,  # no-op for doc
                        "VLLM_SUBCMD_PARSER_EPILOG": "",
                        "logger": logger,
                        "DummySubparsers": DummySubparsers,
                        "argparse": __import__("argparse"),
                        "DESCRIPTION": DESCRIPTION,
                    }
                    exec(code, exec_globals, local_vars)
                    # Get the function
                    subparser_init = local_vars["subparser_init"]

                    # Return a callable that mimics OmniServeCommand().subparser_init
                    def parser_factory():
                        class DummySelf:
                            name = "serve"

                        return subparser_init(DummySelf(), dummy_subparsers)

                    return parser_factory
    raise RuntimeError("Could not statically extract OmniServeCommand.subparser_init")


OmniServeCommand = type("OmniServeCommand", (), {"subparser_init": staticmethod(extract_omni_serve_subparser_init())})


class MarkdownFormatter(HelpFormatter):
    """Custom formatter that generates markdown for argument groups."""

    def __init__(self, prog: str, starting_heading_level: int = 3):
        super().__init__(prog, max_help_position=sys.maxsize, width=sys.maxsize)

        self._section_heading_prefix = "#" * starting_heading_level
        self._argument_heading_prefix = "#" * (starting_heading_level + 1)
        self._markdown_output = []

    def start_section(self, heading: str):
        if heading not in {"positional arguments", "options"}:
            heading_md = f"\n{self._section_heading_prefix} {heading}\n\n"
            self._markdown_output.append(heading_md)

    def end_section(self):
        pass

    def add_text(self, text: str):
        if text:
            self._markdown_output.append(f"{text.strip()}\n\n")

    def add_usage(self, usage, actions, groups, prefix=None):
        pass

    def add_arguments(self, actions: Iterable[Action]):
        for action in actions:
            if len(action.option_strings) == 0 or "--help" in action.option_strings:
                continue

            option_strings = f"`{'`, `'.join(action.option_strings)}`"
            heading_md = f"{self._argument_heading_prefix} {option_strings}\n\n"
            self._markdown_output.append(heading_md)

            if choices := action.choices:
                choices = f"`{'`, `'.join(str(c) for c in choices)}`"
                self._markdown_output.append(f"Possible choices: {choices}\n\n")
            elif (metavar := action.metavar) and isinstance(metavar, (list, tuple)):
                metavar = f"`{'`, `'.join(str(m) for m in metavar)}`"
                self._markdown_output.append(f"Possible choices: {metavar}\n\n")

            if action.help:
                self._markdown_output.append(f"{action.help}\n\n")

            if (default := action.default) != SUPPRESS:
                # Make empty string defaults visible
                if default == "":
                    default = '""'
                self._markdown_output.append(f"Default: `{default}`\n\n")

    def format_help(self):
        """Return the formatted help as markdown."""
        return "".join(self._markdown_output)


# Function to create parser using subparser_init style CLI class


def create_parser_subparser_init(subcmd_class):
    """
    Create an argparse parser using subparser_init style CLI class, with MarkdownFormatter.
    Compatible with both AST-extracted static parser_factory and real class instance method.
    """

    class DummySubparsers:
        def add_parser(self, name, **kwargs):
            return _FlexibleArgumentParser(prog=name)

    dummy_subparsers = DummySubparsers()
    # If subparser_init is a zero-arg static function (AST-extracted), call it directly
    subparser_init = getattr(subcmd_class, "subparser_init", None)
    if subparser_init is not None:
        import inspect

        sig = inspect.signature(subparser_init)
        if len(sig.parameters) == 0:
            parser = subparser_init()
        else:
            parser = subcmd_class().subparser_init(dummy_subparsers)
    else:
        parser = subcmd_class().subparser_init(dummy_subparsers)
    parser.formatter_class = MarkdownFormatter
    return parser


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    """
    Entry point for doc generation. Builds doc directory and outputs markdown for each CLI command.
    """
    logger.info("Generating vllm-omni argparse documentation")
    logger.debug("Root directory: %s", ROOT_DIR.resolve())
    logger.debug("Output directory: %s", ARGPARSE_DOC_DIR.resolve())
    if not ARGPARSE_DOC_DIR.exists():
        ARGPARSE_DOC_DIR.mkdir(parents=True)

    # Register all CLI parsers; you can easily add more commands here
    parsers = {
        "omni_serve": create_parser_subparser_init(OmniServeCommand),
        # "another_cmd": create_parser_subparser_init(AnotherCommandClass),
    }

    for stem, parser in parsers.items():
        doc_path = ARGPARSE_DOC_DIR / f"{stem}.inc.md"
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(super(type(parser), parser).format_help())
        logger.info("Argparse generated: %s", doc_path.relative_to(ROOT_DIR))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    on_startup("build", False)
