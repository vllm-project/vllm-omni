# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Hook to automatically generate docs/api/README.md from the codebase.

This script scans the vllm_omni module for public classes and functions,
categorizes them, and generates a summary README file.
"""

import ast
import logging
from pathlib import Path

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).parent.parent.parent.parent
API_README_PATH = ROOT_DIR / "docs" / "api" / "README.md"

# Category mappings: module prefix -> category name and description
CATEGORIES = {
    "entrypoints": {
        "name": "Entry Points",
        "description": "Main entry points for vLLM-Omni inference and serving.",
    },
    "inputs": {
        "name": "Inputs",
        "description": "Input data structures for multi-modal inputs.",
    },
    "engine": {
        "name": "Engine",
        "description": "Engine classes for offline and online inference.",
    },
    "core": {
        "name": "Core",
        "description": "Core scheduling and caching components.",
    },
    # "model_executor": {
    #     "name": "Model Executor",
    #     "description": "Model execution components.",
    # },
    "config": {
        "name": "Configuration",
        "description": "Configuration classes.",
    },
    "worker": {
        "name": "Workers",
        "description": "Worker classes and model runners for distributed inference.",
    },
}


class APIVisitor(ast.NodeVisitor):
    """AST visitor to extract public classes and module-level functions."""

    def __init__(self, module_path: str):
        self.module_path = module_path
        self.classes: list[str] = []
        self.functions: list[str] = []
        self._class_stack: list[str] = []  # Track nested class definitions

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        if not node.name.startswith("_"):
            self.classes.append(f"{self.module_path}.{node.name}")
        # Track that we're entering a class
        self._class_stack.append(node.name)
        self.generic_visit(node)
        # Remove from stack when done visiting
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions - only collect module-level functions."""
        # Only collect if we're not inside a class (stack is empty)
        if not self._class_stack and not node.name.startswith("_"):
            self.functions.append(f"{self.module_path}.{node.name}")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions - only collect module-level functions."""
        # Only collect if we're not inside a class (stack is empty)
        if not self._class_stack and not node.name.startswith("_"):
            self.functions.append(f"{self.module_path}.{node.name}")
        self.generic_visit(node)


def parse_file_for_symbols(file_path: Path, module_path: str) -> tuple[list[str], list[str]]:
    """
    Parse a Python file and extract public classes and functions.

    Returns:
        Tuple of (classes, functions)
    """
    try:
        # If this is __init__.py, use parent module path
        if file_path.name == "__init__.py":
            # Remove .__init__ from module path
            if module_path.endswith(".__init__"):
                module_path = module_path[:-9]

        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        visitor = APIVisitor(module_path)
        visitor.visit(tree)

        return visitor.classes, visitor.functions
    except Exception as e:
        logger.debug(f"Could not parse {file_path}: {e}")
        return [], []


def scan_package(package_name: str = "vllm_omni") -> dict[str, list[str]]:
    """
    Scan the vllm_omni package and categorize public symbols.

    Returns:
        Dict mapping category names to lists of symbol full names
    """
    categorized: dict[str, list[str]] = {cat["name"]: [] for cat in CATEGORIES.values()}

    try:
        # Find the package directory
        package_path = ROOT_DIR / package_name
        if not package_path.exists():
            logger.warning(f"Package path not found: {package_path}")
            return categorized

        # Walk through all Python files
        for py_file in package_path.rglob("*.py"):
            # Skip __init__.py and private modules
            if py_file.name.startswith("_") and py_file.name != "__init__.py":
                continue

            # Get module path
            relative_path = py_file.relative_to(ROOT_DIR)
            module_path = str(relative_path.with_suffix("")).replace("/", ".").replace("\\", ".")

            # Skip excluded modules (avoid importing vllm during docs build)
            excluded_prefixes = [
                "vllm_omni.diffusion.models.qwen_image",
                "vllm_omni.entrypoints.async_diffusion",
                "vllm_omni.entrypoints.openai",
            ]
            if any(module_path.startswith(prefix) for prefix in excluded_prefixes):
                continue

            # Handle __init__.py - use parent module path
            if py_file.name == "__init__.py":
                # Remove .__init__ from module path
                if module_path.endswith(".__init__"):
                    module_path = module_path[:-9]

            # Determine category from module path
            category = None
            for prefix, cat_info in CATEGORIES.items():
                if prefix in module_path:
                    category = cat_info["name"]
                    break

            if not category:
                continue

            # Parse file for symbols
            classes, functions = parse_file_for_symbols(py_file, module_path)

            # Filter out internal implementation classes
            # Skip classes that look like internal components (DiT layers, etc.)
            internal_patterns = [
                "Block",
                "Layer",
                "Net",
                "Embedding",
                "Norm",
                "Activation",
                "Solver",
                "Pooling",
                "Attention",
                "MLP",
                "DecoderLayer",
                "InputEmbedding",
                "TimestepEmbedding",
                "CodecEmbedding",
                "DownSample",
                "UpSample",
                "Res2Net",
                "SqueezeExcitation",
                "TimeDelay",
                "TorchActivation",
                "SnakeBeta",
                "SinusPosition",
                "RungeKutta",
                "AMPBlock",
                "AdaLayerNorm",
            ]

            # Add classes (filter out internal ones)
            for class_name in classes:
                class_short_name = class_name.split(".")[-1]
                # Skip if it matches internal patterns (unless it's a main model class)
                if any(pattern in class_short_name for pattern in internal_patterns):
                    # But include main model classes
                    if not any(
                        main_class in class_short_name
                        for main_class in [
                            "ForConditionalGeneration",
                            "Model",
                            "Registry",
                            "Worker",
                            "Runner",
                            "Scheduler",
                            "Manager",
                            "Processor",
                            "Config",
                        ]
                    ):
                        continue
                categorized[category].append(class_name)

            # Add important functions (parse, preprocess, etc.)
            for func_name in functions:
                # Include functions that match certain patterns
                if any(keyword in func_name.lower() for keyword in ["parse", "preprocess"]):
                    categorized[category].append(func_name)

        # Sort symbols within each category
        for category in categorized:
            categorized[category].sort()

    except Exception as e:
        logger.error(f"Error scanning package: {e}", exc_info=True)

    return categorized


def generate_readme(categorized: dict[str, list[str]]) -> str:
    """Generate the API README markdown content."""
    lines = ["# Summary", ""]

    # Generate sections for each category
    for prefix, cat_info in CATEGORIES.items():
        category_name = cat_info["name"]
        description = cat_info["description"]
        symbols = categorized.get(category_name, [])

        if not symbols:
            continue

        lines.append(f"## {category_name}")
        lines.append("")
        lines.append(description)
        lines.append("")

        for symbol in symbols:
            lines.append(f"- [{symbol}][]")

        lines.append("")

    return "\n".join(lines)


def on_startup(command, dirty: bool):
    """MkDocs hook entry point."""
    logger.info("Generating API README documentation")

    # Scan the package
    categorized = scan_package()

    # Generate README content
    content = generate_readme(categorized)

    # Write to file
    API_README_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(API_README_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"API README generated: {API_README_PATH.relative_to(ROOT_DIR)}")
