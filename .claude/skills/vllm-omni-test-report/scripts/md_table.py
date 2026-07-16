"""Plain GitHub-flavored Markdown tables (no cell padding)."""

from __future__ import annotations

import html


def render_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """
    Render a pipe table with a ``---`` separator row. Cells are not padded for column alignment.

    Each row must have len(row) == len(headers). Cells should not contain unescaped ``|``.
    """
    cols = len(headers)
    if not headers:
        return ""
    for row in rows:
        if len(row) != cols:
            raise ValueError(f"row has {len(row)} cells, expected {cols}: {row!r}")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * cols) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_html_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    table_class: str = "",
) -> str:
    """HTML table with escaped cell text (no inline Markdown)."""
    cols = len(headers)
    for row in rows:
        if len(row) != cols:
            raise ValueError(f"row has {len(row)} cells, expected {cols}: {row!r}")
    cls = f' class="{html.escape(table_class)}"' if table_class else ""
    lines = [f"<table{cls}>", "<thead><tr>"]
    for h in headers:
        lines.append(f"<th>{html.escape(h)}</th>")
    lines.extend(["</tr></thead>", "<tbody>"])
    for row in rows:
        lines.append("<tr>")
        for c in row:
            lines.append(f"<td>{html.escape(c)}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines)
