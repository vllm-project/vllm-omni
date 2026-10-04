# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resolve checked-in inventories before mkdocstrings starts downloading them."""

from pathlib import Path
from urllib.parse import urlsplit

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority


@event_priority(100)
def on_config(config: MkDocsConfig) -> None:
    base_dir = Path(config.config_file_path).parent
    handler = config.plugins["mkdocstrings"].config.handlers["python"]
    for inventory in handler.get("inventories", []):
        if isinstance(inventory, dict) and not urlsplit(inventory["url"]).scheme:
            path = (base_dir / inventory["url"]).resolve(strict=True)
            inventory["url"] = path.as_uri()
