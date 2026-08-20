"""Fixtures config creation & out of tree registry management."""

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES


@pytest.fixture
def clean_pipeline_registry():
    """Ensure the OMNI_PIPELINES are in a clean state for a test that mutates it."""
    snapshot = dict(OMNI_PIPELINES)
    yield
    OMNI_PIPELINES.clear()
    OMNI_PIPELINES.update(snapshot)
