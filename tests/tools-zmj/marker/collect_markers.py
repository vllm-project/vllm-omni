#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
静态扫描 tests/ 下仅 test_*.py 文件，收集出现的 pytest marker 名称。

- 识别 @pytest.mark.xxx、@pytest.mark.xxx(...)、pytestmark = [...] 等形式。
- 可选：从 hardware_marks(...) / @hardware_test(...) 中字面量 res= / num_cards= 推断附加硬件类 marker
  （与 tests.utils 逻辑尽量一致；动态 res 无法推断）。

用法（在仓库根目录）:
  python tests/tools-zmj/marker/collect_markers.py
  python tests/tools-zmj/marker/collect_markers.py --by-file --infer-hardware
  python tests/tools-zmj/marker/collect_markers.py --json markers.json
  python tests/tools-zmj/marker/collect_markers.py --csv markers.csv

CSV 第三列「修饰器」：
  - 包含文件头/类上的 ``pytestmark = ...``（标注为 [模块 pytestmark] / [类 pytestmark]）；
  - 默认不包含 ``@pytest.mark.parametrize``（可用 ``--csv-include-parametrize`` 保留）。
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

# pytest 内置/常用装饰器名，单独归类（仍计入「全部」集合）
PYTEST_BUILTIN_MARK_NAMES = frozenset(
    {
        "filterwarnings",
        "parametrize",
        "skip",
        "skipif",
        "usefixtures",
        "xfail",
    }
)

HARDWARE_FUNCS = frozenset({"hardware_marks", "hardware_test"})


def _is_pytest_mark_attr(node: ast.AST) -> str | None:
    """若为 pytest.mark.<name> 或 pytest.mark.<name>(...) 则返回 name。"""
    if isinstance(node, ast.Call):
        node = node.func
    cur: ast.AST = node
    if not isinstance(cur, ast.Attribute):
        return None
    name = cur.attr
    cur = cur.value
    if not isinstance(cur, ast.Attribute):
        return None
    if cur.attr != "mark":
        return None
    if not isinstance(cur.value, ast.Name) or cur.value.id != "pytest":
        return None
    return name


def _const_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _const_int(node: ast.AST | None) -> int | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return None


def _parse_string_dict(node: ast.AST) -> dict[str, str] | None:
    """仅支持 dict 字面量且 key/value 均为字符串常量。"""
    if not isinstance(node, ast.Dict):
        return None
    out: dict[str, str] = {}
    for k, v in zip(node.keys, node.values):
        if k is None:  # **spread
            return None
        ks = _const_str(k)
        vs = _const_str(v)
        if ks is None or vs is None:
            return None
        out[ks] = vs
    return out


def _parse_num_cards(node: ast.AST | None) -> int | dict[str, int] | None:
    if node is None:
        return None
    n = _const_int(node)
    if n is not None:
        return n
    d = _parse_int_dict(node)
    return d


def _parse_int_dict(node: ast.AST) -> dict[str, int] | None:
    if not isinstance(node, ast.Dict):
        return None
    out: dict[str, int] = {}
    for k, v in zip(node.keys, node.values):
        if k is None:
            return None
        ks = _const_str(k)
        vi = _const_int(v)
        if ks is None or vi is None:
            return None
        out[ks] = vi
    return out


def infer_hardware_marker_names(
    res: dict[str, str], num_cards: int | dict[str, int]
) -> list[str]:
    """
    与 tests.utils.hardware_marks 行为对齐的近似推断（仅静态场景）。
    xpu_marks 在源码中对 distributed 使用 distributed_rocm，此处照抄。
    """
    names: list[str] = []
    for platform, resource in res.items():
        if isinstance(num_cards, dict):
            n = num_cards.get(platform, 1)
        else:
            n = num_cards

        if platform in ("cuda", "rocm", "xpu"):
            names.append("gpu")
        if platform == "cuda":
            names.extend(["cuda", resource])
            if n > 1:
                names.extend(["distributed_cuda", "skipif_cuda"])
        elif platform == "rocm":
            names.extend(["rocm", resource])
            if n > 1:
                names.append("distributed_rocm")
        elif platform == "xpu":
            names.extend(["xpu", resource])
            if n > 1:
                names.append("distributed_rocm")
        elif platform == "npu":
            names.append("npu")
            if resource in ("A2", "A3"):
                names.append(resource)
            if n > 1:
                names.append("distributed_npu")
    return names


def _hardware_call_func_name(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _extract_hardware_from_call(call: ast.Call) -> list[str] | None:
    if _hardware_call_func_name(call) not in HARDWARE_FUNCS:
        return None
    res_node = None
    num_node = None
    for kw in call.keywords:
        if kw.arg == "res":
            res_node = kw.value
        elif kw.arg == "num_cards":
            num_node = kw.value
    res_d = _parse_string_dict(res_node) if res_node else None
    if not res_d:
        return None
    nc: int | dict[str, int] = 1
    parsed = _parse_num_cards(num_node)
    if parsed is not None:
        nc = parsed
    return infer_hardware_marker_names(res_d, nc)


def iter_test_py_files(tests_root: Path) -> Iterator[Path]:
    for p in tests_root.rglob("*.py"):
        if p.name.startswith("test_") and p.name.endswith(".py"):
            yield p


def format_decorator_list(
    decorators: list[ast.expr],
    *,
    exclude_parametrize: bool = True,
) -> str:
    """将函数上的装饰器列表转为可读字符串（多行，每行一个 @...）。"""
    if not decorators:
        return ""
    parts: list[str] = []
    for d in decorators:
        if exclude_parametrize and _is_pytest_mark_attr(d) == "parametrize":
            continue
        try:
            parts.append("@" + ast.unparse(d))
        except (AttributeError, TypeError, ValueError):
            parts.append("@<unparseable>")
    return "\n".join(parts)


def extract_pytestmark_rhs_from_stmts(body: list[ast.stmt]) -> str | None:
    """从语句块顶层取第一个 ``pytestmark = ...`` / 注解赋值的右侧表达式源码。"""
    for node in body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "pytestmark":
                    try:
                        return ast.unparse(node.value)
                    except (AttributeError, TypeError, ValueError):
                        return None
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "pytestmark" and node.value:
                try:
                    return ast.unparse(node.value)
                except (AttributeError, TypeError, ValueError):
                    return None
    return None


def collect_test_function_rows(
    body: list[ast.stmt],
    filepath: str,
    class_prefix: str | None,
    out: list[tuple[str, str, str]],
    module_pytestmark_line: str,
    enclosing_pytestmark_lines: list[str],
    *,
    csv_exclude_parametrize: bool,
) -> None:
    """递归扫描类嵌套，收集 test_* 函数；合并模块/类 pytestmark 与函数装饰器。"""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test_"):
                continue
            qual = f"{class_prefix}.{node.name}" if class_prefix else node.name
            func_dec = format_decorator_list(
                node.decorator_list,
                exclude_parametrize=csv_exclude_parametrize,
            )
            parts: list[str] = []
            if module_pytestmark_line:
                parts.append(module_pytestmark_line)
            parts.extend(enclosing_pytestmark_lines)
            if func_dec:
                parts.append(func_dec)
            dec_str = "\n".join(parts)
            out.append((filepath, qual, dec_str))
        elif isinstance(node, ast.ClassDef):
            prefix = f"{class_prefix}.{node.name}" if class_prefix else node.name
            cls_rhs = extract_pytestmark_rhs_from_stmts(node.body)
            cls_line = (
                f"[类 pytestmark] {prefix}: pytestmark = {cls_rhs}"
                if cls_rhs
                else ""
            )
            new_enclosing = enclosing_pytestmark_lines + ([cls_line] if cls_line else [])
            collect_test_function_rows(
                node.body,
                filepath,
                prefix,
                out,
                module_pytestmark_line,
                new_enclosing,
                csv_exclude_parametrize=csv_exclude_parametrize,
            )


def _collect_marks_from_expr(expr: ast.AST, out: set[str]) -> None:
    if isinstance(expr, (ast.List, ast.Tuple)):
        for elt in expr.elts:
            if isinstance(elt, ast.Starred):
                _collect_marks_from_expr(elt.value, out)
            else:
                m = _is_pytest_mark_attr(elt)
                if m:
                    out.add(m)
                elif isinstance(elt, ast.Call):
                    m2 = _is_pytest_mark_attr(elt)
                    if m2:
                        out.add(m2)
    else:
        m = _is_pytest_mark_attr(expr)
        if m:
            out.add(m)
        elif isinstance(expr, ast.Call):
            m2 = _is_pytest_mark_attr(expr)
            if m2:
                out.add(m2)


def collect_from_tree(tree: ast.AST, out: set[str], infer_hardware: bool, hw_out: set[str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "pytestmark":
                    _collect_marks_from_expr(node.value, out)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "pytestmark" and node.value:
                _collect_marks_from_expr(node.value, out)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                m = _is_pytest_mark_attr(dec)
                if m:
                    out.add(m)
                if infer_hardware and isinstance(dec, ast.Call):
                    inferred = _extract_hardware_from_call(dec)
                    if inferred:
                        hw_out.update(inferred)
        elif isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                m = _is_pytest_mark_attr(dec)
                if m:
                    out.add(m)
                if infer_hardware and isinstance(dec, ast.Call):
                    inferred = _extract_hardware_from_call(dec)
                    if inferred:
                        hw_out.update(inferred)
        elif isinstance(node, ast.Call):
            if infer_hardware:
                inferred = _extract_hardware_from_call(node)
                if inferred:
                    hw_out.update(inferred)
            m = _is_pytest_mark_attr(node)
            if m:
                out.add(m)


def resolve_tests_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    here = Path(__file__).resolve()
    # tests/tools-zmj/marker/collect_markers.py -> tests
    return here.parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="收集 tests 下 test_*.py 用例使用的 pytest markers（静态扫描）"
    )
    parser.add_argument(
        "--tests-root",
        type=Path,
        default=None,
        help="tests 目录路径（默认：本脚本上级的 tests/）",
    )
    parser.add_argument(
        "--by-file",
        action="store_true",
        help="按文件打印 marker 集合",
    )
    parser.add_argument(
        "--infer-hardware",
        action="store_true",
        help="从 hardware_marks / hardware_test 的字面量 res= 推断硬件相关 marker",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="PATH",
        help="将结果写入 JSON（含 all、by_file、builtin、project、inferred_hardware）",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="将每个 test_* 函数一行写入 CSV：文件名、函数名、修饰器",
    )
    parser.add_argument(
        "--csv-include-parametrize",
        action="store_true",
        help="CSV 第三列保留 @pytest.mark.parametrize（默认剔除）",
    )
    args = parser.parse_args()
    tests_root = resolve_tests_root(args.tests_root)
    if not tests_root.is_dir():
        print(f"tests 目录不存在: {tests_root}", file=sys.stderr)
        return 1

    by_file: dict[str, set[str]] = defaultdict(set)
    by_file_hw: dict[str, set[str]] = defaultdict(set)
    all_marks: set[str] = set()
    all_hw: set[str] = set()
    csv_rows: list[tuple[str, str, str]] = []

    for py in sorted(iter_test_py_files(tests_root)):
        try:
            src = py.read_text(encoding="utf-8")
        except OSError as e:
            print(f"跳过（无法读取）: {py}: {e}", file=sys.stderr)
            continue
        try:
            tree = ast.parse(src, filename=str(py))
        except SyntaxError as e:
            print(f"跳过（语法错误）: {py}: {e}", file=sys.stderr)
            continue
        local: set[str] = set()
        local_hw: set[str] = set()
        collect_from_tree(tree, local, args.infer_hardware, local_hw)
        try:
            key = py.relative_to(tests_root.parent).as_posix()
        except ValueError:
            key = str(py.resolve())
        by_file[key].update(local)
        if args.infer_hardware:
            by_file_hw[key].update(local_hw)
        all_marks.update(local)
        all_hw.update(local_hw)
        if args.csv is not None:
            mod_rhs = (
                extract_pytestmark_rhs_from_stmts(tree.body)
                if isinstance(tree, ast.Module)
                else None
            )
            module_line = (
                f"[模块 pytestmark] pytestmark = {mod_rhs}" if mod_rhs else ""
            )
            collect_test_function_rows(
                tree.body,
                key,
                None,
                csv_rows,
                module_line,
                [],
                csv_exclude_parametrize=not args.csv_include_parametrize,
            )

    builtin_hits = sorted(n for n in all_marks if n in PYTEST_BUILTIN_MARK_NAMES)
    project_hits = sorted(n for n in all_marks if n not in PYTEST_BUILTIN_MARK_NAMES)
    combined = sorted(all_marks | all_hw)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["文件名", "函数名", "修饰器"])
            for row in sorted(csv_rows, key=lambda r: (r[0], r[1])):
                w.writerow(row)
        print(f"已写入 CSV：{args.csv}（test_* 函数行数 {len(csv_rows)}）")

    if args.json:
        payload = {
            "tests_root": str(tests_root),
            "all": sorted(all_marks),
            "all_with_inferred": combined,
            "builtin": builtin_hits,
            "project_explicit": project_hits,
            "inferred_hardware": sorted(all_hw) if args.infer_hardware else [],
            "by_file": {k: sorted(v) for k, v in sorted(by_file.items())},
            "by_file_inferred_hardware": {k: sorted(v) for k, v in sorted(by_file_hw.items())},
        }
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"已写入 {args.json}")

    print(f"扫描目录: {tests_root}")
    print(f"文件数: {len(by_file)}")
    print(f"显式 pytest.mark 种类数: {len(all_marks)}")
    if args.infer_hardware:
        print(f"推断硬件 marker 种类数: {len(all_hw)}")
    print("--- 全部显式 marker（排序） ---")
    for m in sorted(all_marks):
        tag = "(builtin)" if m in PYTEST_BUILTIN_MARK_NAMES else ""
        print(f"  {m} {tag}".rstrip())
    if args.infer_hardware and all_hw:
        print("--- 推断的硬件相关 marker ---")
        for m in sorted(all_hw):
            print(f"  {m}")

    if args.by_file:
        print("--- 按文件 ---")
        for path in sorted(by_file.keys()):
            marks = sorted(by_file[path])
            extra = ""
            if args.infer_hardware and by_file_hw.get(path):
                extra = f"  [推断] {sorted(by_file_hw[path])}"
            print(f"{path}: {marks}{extra}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
