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

# 暂不统计的修饰器（按前缀匹配）
SKIP_DECORATOR_PREFIXES = ("@pytest.mark.parametrize", "@pytest.mark.asyncio")


def _get_pytestmark_from_module(tree: ast.Module, text: str) -> list[str]:
    """从模块 AST 中提取 pytestmark 的标记列表，返回带 @ 的字符串列表。"""
    marks: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "pytestmark":
            continue
        value = node.value
        if isinstance(value, ast.List):
            for elt in value.elts:
                src = ast.get_source_segment(text, elt)
                if src:
                    s = src.strip()
                    if not s.startswith("@"):
                        s = "@" + s
                    marks.append(s)
        else:
            src = ast.get_source_segment(text, value)
            if src:
                s = src.strip()
                if not s.startswith("@"):
                    s = "@" + s
                marks.append(s)
        break
    return marks


def _should_skip_decorator(dec_src: str) -> bool:
    """是否跳过该修饰器（不纳入统计）。"""
    s = dec_src.strip()
    return any(s.startswith(prefix) for prefix in SKIP_DECORATOR_PREFIXES)


def should_skip(rel_path: Path) -> bool:
    """rel_path 为相对于 TESTS_ROOT 的路径，如 entrypoints/test_foo.py 或 e2e/xxx/test.py"""
    return any(part in EXCLUDE_DIRS for part in rel_path.parts)


def collect_decorators_and_tests(file_path: Path) -> list[tuple[str, list[str]]]:
    """返回 [(test_func_name, [decorator1_src, ...]), ...]，使用 AST 支持多行修饰器；含模块级 pytestmark，排除 parametrize/asyncio。"""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(file_path))
    except SyntaxError as e:
        raise RuntimeError(f"AST 解析失败: {e}") from e

    file_marks = _get_pytestmark_from_module(tree, text)
    results: list[tuple[str, list[str]]] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            decorators: list[str] = list(file_marks)  # 先加入文件级 pytestmark
            for dec in node.decorator_list:
                src = ast.get_source_segment(text, dec)
                if src is None:
                    try:
                        src = "@(" + ast.unparse(dec) + ")"  # type: ignore[attr-defined]
                    except Exception:
                        src = "@<unknown_decorator>"
                src = src.strip()
                if not src.startswith("@"):
                    src = "@" + src
                if _should_skip_decorator(src):
                    continue
                decorators.append(src)
            results.append((node.name, decorators))

    return results


def main():
    # rows: (file, func_name, decorators_md, decorators_csv)
    rows: list[tuple[str, str, str, str]] = []
    for py in sorted(TESTS_ROOT.rglob("*.py")):
        rel = py.relative_to(TESTS_ROOT)
        if should_skip(rel):
            continue
        try:
            for func_name, decorators in collect_decorators_and_tests(py):
                file_str = rel.as_posix()
                # Markdown 用竖线分隔，CSV 用分号分隔
                decorator_str_md = " | ".join(decorators) if decorators else "(无)"
                decorator_str_csv = "; ".join(decorators) if decorators else "(无)"
                rows.append((file_str, func_name, decorator_str_md, decorator_str_csv))
        except Exception as e:
            file_str = rel.as_posix()
            rows.append((file_str, "<解析错误>", f"(解析错误: {e})", ""))

    # 按文件名、函数名排序，方便查看和“合并”相同文件
    rows.sort(key=lambda x: (x[0], x[1]))

    # 输出 CSV，便于 Excel 打开
    out_csv = TESTS_ROOT / "test_decorators_report.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["文件", "函数名", "修饰器 (同一单元格用 ; 分隔)"])
        for file_str, func_name, _, dec_csv in rows:
            w.writerow([file_str, func_name, dec_csv])

    # 同时输出 Markdown 表格到文件
    out_md = TESTS_ROOT / "test_decorators_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# tests 用例与修饰器统计（已排除 e2e、examples、perf）\n\n")
        f.write("| 文件 | 函数名 | 修饰器 |\n")
        f.write("| --- | --- | --- |\n")

        last_file: str | None = None
        for file_str, func_name, dec_str, _ in rows:
            # 同一个文件的多行，只在第一行显示文件名，后续行留空，达到“合并”视觉效果
            file_cell = file_str if file_str != last_file else ""
            last_file = file_str
            # Markdown 表格内 | 转义为 \| 或放在代码块里避免破坏列
            dec_esc = dec_str.replace("|", "\\|").replace("\n", " ")
            f.write(f"| {file_cell} | {func_name} | {dec_esc} |\n")

    print(f"共 {len(rows)} 个用例，已写入:")
    print(f"  - {out_csv}")
    print(f"  - {out_md}")
    return rows


if __name__ == "__main__":
    main()
