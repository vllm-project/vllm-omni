"""
统计 tests 目录下所有 test_xxx 用例及其 pytest 修饰器，排除 e2e、examples、perf。
输出：用例（文件::函数名） | 所有修饰器
"""
from pathlib import Path
import re
import csv
import ast

TESTS_ROOT = Path(__file__).resolve().parent
EXCLUDE_DIRS = {"e2e", "examples", "perf"}


def should_skip(rel_path: Path) -> bool:
    """rel_path 为相对于 TESTS_ROOT 的路径，如 entrypoints/test_foo.py 或 e2e/xxx/test.py"""
    return any(part in EXCLUDE_DIRS for part in rel_path.parts)


def collect_decorators_and_tests(file_path: Path) -> list[tuple[str, list[str]]]:
    """返回 [(test_func_name, [decorator1_src, decorator2_src, ...]), ...]，使用 AST 支持多行修饰器。"""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(file_path))
    except SyntaxError as e:
        raise RuntimeError(f"AST 解析失败: {e}") from e

    results: list[tuple[str, list[str]]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            decorators: list[str] = []
            for dec in node.decorator_list:
                src = ast.get_source_segment(text, dec)
                if src is None:
                    # 回退：尽量给一个可读表示
                    try:
                        src = "@(" + ast.unparse(dec) + ")"  # type: ignore[attr-defined]
                    except Exception:
                        src = "@<unknown_decorator>"
                # 确保带上前导 @，有些情况下 get_source_segment 可能只返回表达式
                src = src.strip()
                if not src.startswith("@"):
                    src = "@" + src
                decorators.append(src)
            results.append((node.name, decorators))

    return results


def main():
    rows = []
    for py in sorted(TESTS_ROOT.rglob("*.py")):
        rel = py.relative_to(TESTS_ROOT)
        if should_skip(rel):
            continue
        try:
            for func_name, decorators in collect_decorators_and_tests(py):
                case_id = f"{rel.as_posix()}::{func_name}"
                # Markdown 用竖线分隔，CSV 用分号分隔
                decorator_str_md = " | ".join(decorators) if decorators else "(无)"
                decorator_str_csv = "; ".join(decorators) if decorators else "(无)"
                rows.append((case_id, decorator_str_md, decorator_str_csv))
        except Exception as e:
            rows.append((str(rel), f"(解析错误: {e})", ""))

    # 输出 CSV，便于 Excel 打开
    out_csv = TESTS_ROOT / "test_decorators_report.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["用例 (文件::函数名)", "修饰器 (同一单元格用 ; 分隔)"])
        for case_id, _, dec_csv in rows:
            w.writerow([case_id, dec_csv])

    # 同时输出 Markdown 表格到文件
    out_md = TESTS_ROOT / "test_decorators_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# tests 用例与修饰器统计（已排除 e2e、examples、perf）\n\n")
        f.write("| 用例 (文件::函数名) | 修饰器 |\n")
        f.write("| --- | --- |\n")
        for case_id, dec_str, _ in rows:
            # Markdown 表格内 | 转义为 \| 或放在代码块里避免破坏列
            dec_esc = dec_str.replace("|", "\\|").replace("\n", " ")
            f.write(f"| {case_id} | {dec_esc} |\n")

    print(f"共 {len(rows)} 个用例，已写入:")
    print(f"  - {out_csv}")
    print(f"  - {out_md}")
    return rows


if __name__ == "__main__":
    main()
