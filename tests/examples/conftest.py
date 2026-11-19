# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pytest fixtures for tests/examples."""

from tests.examples.helpers import example_runner  # noqa: F401
from tests.examples.model_scope import doc_item_models, doc_models_allowed


def pytest_collection_modifyitems(config, items):
    """Limit GPU doc tests without excluding CPU tests of other example code."""
    selected, excluded = [], []
    for item in items:
        is_doc_test = item.get_closest_marker("full_model") and not item.get_closest_marker("cpu")
        params = getattr(getattr(item, "callspec", None), "params", {})
        if is_doc_test and not doc_models_allowed(doc_item_models(params)):
            excluded.append(item)
        else:
            selected.append(item)
    if excluded:
        config.hook.pytest_deselected(items=excluded)
        items[:] = selected
