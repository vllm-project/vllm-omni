# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline AST dependency declaration gate; see tools/pre_commit/README.md."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "tools/pre_commit/import_dependencies.json"
BASELINE = "tools/pre_commit/import_dependencies_baseline.json"
# A fixed target, never the hook host's installed packages or platform. Profiles
# may override it for platform-specific declarations. This is not an ABI check.
ENVIRONMENT = {
    "implementation_name": "cpython",
    "implementation_version": "3.12.0",
    "os_name": "posix",
    "platform_machine": "x86_64",
    "platform_python_implementation": "CPython",
    "platform_release": "",
    "platform_system": "Linux",
    "platform_version": "",
    "python_full_version": "3.12.0",
    "python_version": "3.12",
    "sys_platform": "linux",
    "extra": "",
}


def requirements(path: Path, stack: tuple[Path, ...] = ()) -> list[Requirement]:
    """Read install requirements, including recursive -r but never constraints."""
    path = path.resolve()
    if path in stack:
        raise ValueError(f"Recursive requirements include: {path}")
    result = []
    source = re.sub(r"\\\r?\n", " ", path.read_text(encoding="utf-8"))
    for raw in source.splitlines():
        line = re.split(r"\s+#", raw, maxsplit=1)[0].strip()
        if not line or line.startswith("#"):
            continue
        include = re.match(r"^(?:-r\s*|--requirement(?:=|\s+))(.+)$", line)
        if include:
            result.extend(requirements(path.parent / include[1].strip(), (*stack, path)))
        elif re.match(r"^(?:-c|--constraint(?:=|\s))", line):
            continue
        elif line.startswith(("--index-url", "--extra-index-url", "--find-links", "--trusted-host", "--no-index")):
            continue
        else:
            # Unknown pip directives and unnamed VCS URLs fail closed. Named
            # PEP 508 direct references and extras are handled by packaging.
            result.append(Requirement(line))
    return result


class Imports(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: list[str] = []
        self.found: list[tuple[str, str, str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # noqa: N815 - ast visitor API
    visit_ClassDef = visit_FunctionDef  # noqa: N815 - ast visitor API

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.found.append((alias.name, ".".join(self.scope), ast.unparse(node), node.lineno))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level == 0 and node.module:
            for alias in node.names:
                self.found.append((f"{node.module}.{alias.name}", ".".join(self.scope), ast.unparse(node), node.lineno))


def tracked_files(root: Path) -> list[str]:
    return (
        subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
        .decode()
        .split("\0")
    )


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = json.loads((root / CONFIG).read_text(encoding="utf-8"))
        self.project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        self.cache: dict[str, set[str]] = {}
        self.declaration_files = {CONFIG, BASELINE, "pyproject.toml", "setup.py", ".pre-commit-config.yaml"}
        self.declaration_files.update(self.config.get("rescan_on", []))
        for group in self.config["groups"].values():
            self.declaration_files.update(group.get("files", []))
        self.baseline = json.loads((root / BASELINE).read_text(encoding="utf-8"))
        self.allowances = {}
        for entry in self.baseline:
            if not entry.get("reason", "").strip() or entry["count"] < 1:
                raise ValueError("Every baseline entry needs a reason and positive count")
            key = tuple(entry[k] for k in ("path", "module", "scope", "statement"))
            if key in self.allowances:
                raise ValueError(f"Duplicate baseline entry: {key}")
            self.allowances[key] = entry["count"]

    def group_requirements(self, name: str, stack: tuple[str, ...] = ()) -> list[Requirement]:
        if name in stack:
            raise ValueError(f"Recursive group: {name}")
        group = self.config["groups"][name]
        result = [Requirement(r) for r in group.get("prerequisites", [])]
        if group.get("prerequisites") and not group.get("reason"):
            raise ValueError(f"Prerequisites need provenance: {name}")
        for parent in group.get("extends", []):
            result.extend(self.group_requirements(parent, (*stack, name)))
        for path in group.get("files", []):
            result.extend(requirements(self.root / path))
        for extra in group.get("extras", []):
            result.extend(Requirement(r) for r in self.project["project"]["optional-dependencies"][extra])
        if group.get("build_system"):
            result.extend(Requirement(r) for r in self.project["build-system"]["requires"])
        return result

    def profile(self, path: str) -> str:
        for rule in self.config["rules"]:
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule["paths"]):
                return rule["group"]
        return self.config["default_group"]

    def packages(self, profile: str) -> set[str]:
        if profile not in self.cache:
            env = ENVIRONMENT | self.config["groups"][profile].get("environment", {})
            # tomli's guarded import is required on supported Python 3.10 even
            # when the pre-commit host uses Python 3.12. Check declaration names
            # across the supported Python minor versions, not installed metadata.
            self.cache[profile] = {
                canonicalize_name(r.name)
                for r in self.group_requirements(profile)
                if r.marker is None
                or any(
                    r.marker.evaluate(env | {"python_version": v, "python_full_version": v + ".0"})
                    for v in ("3.10", "3.11", "3.12", "3.13")
                )
            }
        return self.cache[profile]

    def local(self, path: str, module: str) -> bool:
        top = module.split(".")[0]
        if top in self.config["internal_modules"]:
            return True
        # Bare sibling imports in scripts/examples. Never consult sys.path or
        # site-packages, and never exempt arbitrary names found elsewhere.
        directories = [(self.root / path).parent]
        for rule in self.config.get("local_roots", []):
            if not rule.get("reason"):
                raise ValueError("Local import roots need a reason")
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule["paths"]):
                directories.extend(self.root / directory for directory in rule["roots"])
        return any(
            (directory / f"{top}.py").is_file() or (directory / top / "__init__.py").is_file()
            for directory in directories
        )

    def distributions(self, module: str) -> set[str]:
        # Longest prefix wins so namespace packages can have distinct owners.
        parts = module.split(".")
        for n in range(len(parts), 0, -1):
            key = ".".join(parts[:n])
            if key in self.config["import_names"]:
                return {canonicalize_name(p) for p in self.config["import_names"][key]}
        return {canonicalize_name(parts[0])}

    def scan(self, paths: list[str]) -> tuple[list[dict], list[str]]:
        missing: Counter = Counter()
        lines = {}
        errors = []
        stdlib = sys.stdlib_module_names | {"tomllib"}
        for path in sorted(set(paths)):
            if not path.endswith((".py", ".pyi")) or not (self.root / path).is_file():
                continue
            try:
                visitor = Imports()
                visitor.visit(ast.parse((self.root / path).read_bytes(), filename=path))
                packages = self.packages(self.profile(path))
                for module, scope, statement, line in visitor.found:
                    if module.split(".")[0] in stdlib or self.local(path, module):
                        continue
                    if self.distributions(module) & packages:
                        continue
                    key = (path, module, scope, statement)
                    missing[key] += 1
                    lines[key] = line
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"{path}: cannot scan: {exc}")
        entries = [
            dict(zip(("path", "module", "scope", "statement"), key), count=count)
            for key, count in sorted(missing.items())
        ]
        for key, count in missing.items():
            if count > self.allowances.get(key, 0):
                path, module, _, _ = key
                errors.append(
                    f"{path}:{lines[key]}: {module!r} requires {' or '.join(sorted(self.distributions(module)))} "
                    f"in dependency group {self.profile(path)!r}; declare it in that group's install requirements "
                    "or add a narrowly scoped, explained exception."
                )
        return entries, errors

    def must_rescan(self, paths: list[str]) -> bool:
        # Included requirements and deleted manifests must trigger a full scan,
        # even if they are no longer referenced by their former parent file.
        return any(
            p
            and (p in self.declaration_files or not p.endswith((".py", ".pyi")))
            or p.startswith("tools/pre_commit/check_import_dependencies")
            for p in paths
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filenames", nargs="*")
    parser.add_argument("--all-files", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, help="Write outstanding imports for review; never updates the baseline")
    args = parser.parse_args(argv)
    try:
        checker = Checker(args.root.resolve())
        paths = args.filenames
        deleted = (
            subprocess.check_output(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=D", "-z"], cwd=checker.root
            )
            .decode()
            .split("\0")
        )
        full_scan = args.all_files or not paths or checker.must_rescan(paths + deleted)
        if full_scan:
            paths = tracked_files(checker.root)
        entries, errors = checker.scan(paths)
        if full_scan:
            outstanding = {tuple(e[k] for k in ("path", "module", "scope", "statement")): e["count"] for e in entries}
            for key, count in checker.allowances.items():
                if outstanding.get(key, 0) < count:
                    errors.append(
                        f"{key[0]}: remove or reduce stale baseline entry for {key[1]} in {key[2] or '<module>'}"
                    )
        if args.report:
            args.report.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("Import dependency declarations passed.")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Dependency checker configuration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
