# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model scope for GPU documentation tests; CPU example tests are independent."""

import ast
import shlex
from fnmatch import fnmatchcase

DOC_TEST_MODELS = (
    "openbmb/MiniCPM-o-4_5",
    "Qwen/Qwen3-Omni-*",
    "Qwen/Qwen3-TTS-*",
    "Qwen/Qwen-Image-2.1",
    "Qwen/Qwen-Image-2.1-Turbo",
    "tencent/HunyuanImage-3.0",
    "tencent/HunyuanImage-3.0-*",
    "hunyuanvideo-community/HunyuanVideo-1.5-*",
    "tencent/HunyuanVideo-1.5",
    "tencent/HunyuanVideo-1.5-*",
    "Wan-AI/Wan2.2-*",
    "MiniMaxAI/MiniMax-H3",
)


def models_in_snippet(language: str, code: str) -> tuple[str, ...]:
    """Read explicit model arguments, never execute code or infer script defaults.

    Unknown/dynamic model arguments fail closed. All model arguments in a block
    must be supported, so a supported model cannot hide a second unsupported one.
    """
    models: list[str] = []
    if language == "bash":
        try:
            argv = shlex.split(code.replace("\\\n", " "), comments=True)
        except ValueError:
            return ()
        for index, arg in enumerate(argv):
            if arg == "--model":
                if index + 1 == len(argv):
                    return ()
                models.append(argv[index + 1])
            elif arg.startswith("--model="):
                models.append(arg.partition("=")[2])
            elif arg.startswith("MODEL="):
                models.append(arg.partition("=")[2])
            elif arg == "serve" and index > 0 and argv[index - 1] == "vllm":
                if index + 1 == len(argv):
                    return ()
                models.append(argv[index + 1])
    elif language == "python":
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ()
        bindings: dict[str, str | None] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        value = node.value
                        bindings[target.id] = (
                            value.value
                            if target.id not in bindings
                            and isinstance(value, ast.Constant)
                            and isinstance(value.value, str)
                            else None
                        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            values = [kw.value for kw in node.keywords if kw.arg in {"model", "model_name", "model_id"}]
            if isinstance(node.func, ast.Name) and node.func.id in {"Omni", "AsyncOmni"} and node.args:
                values.append(node.args[0])
            for value in values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    models.append(value.value)
                elif isinstance(value, ast.Name) and isinstance(resolved := bindings.get(value.id), str):
                    models.append(resolved)
                else:
                    return ()
    return tuple(dict.fromkeys(models))


def doc_models_allowed(models: tuple[str, ...]) -> bool:
    return bool(models) and all(any(fnmatchcase(model, pattern) for pattern in DOC_TEST_MODELS) for model in models)


def doc_item_models(params: dict) -> tuple[str, ...]:
    """Use actual snippet/server parameters rather than module names or headings."""
    models: list[str] = []
    for value in params.values():
        if hasattr(value, "language") and hasattr(value, "code"):
            snippet_models = models_in_snippet(value.language, value.code)
            if not snippet_models:
                return ()
            models.extend(snippet_models)
        elif isinstance(getattr(value, "model", None), str):
            models.append(value.model)
    return tuple(dict.fromkeys(models))
