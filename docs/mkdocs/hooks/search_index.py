# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduce browser search-index memory for generated API reference pages."""

from material.plugins.search.plugin import SearchIndex

_add_entry_from_context = SearchIndex.add_entry_from_context


def add_entry_from_context(self: SearchIndex, page) -> None:
    search = page.meta.get("search") or {}
    if search.get("exclude"):
        return

    if str(page.url).startswith("api/"):
        entry = {
            "location": page.url,
            "title": str(page.meta.get("title", page.title)),
            "text": "",
        }
        if "boost" in search:
            entry["boost"] = search["boost"]
        self.entries.append(entry)
        return

    _add_entry_from_context(self, page)


SearchIndex.add_entry_from_context = add_entry_from_context
