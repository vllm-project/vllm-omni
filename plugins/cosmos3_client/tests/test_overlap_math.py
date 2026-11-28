"""Nested timing spans must not double-count overlap or elapsed time."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "examples/cosmos3/compare_overlap.py"
spec = importlib.util.spec_from_file_location("compare_overlap", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_nested_and_adjacent_intervals_are_counted_once():
    assert module.union([(0, 4), (1, 2), (4, 5)]) == [[0, 5]]
    assert module.duration([(0, 4), (1, 2), (4, 5)]) == 5


def test_overlap_measures_parallel_work_instead_of_adding_durations():
    assert module.overlap([(0, 4)], [(2, 6)]) == 2
    assert module.overlap([(0, 4), (1, 3)], [(2, 6), (2.5, 3)]) == 2
    assert module.overlap([(0, 2)], [(2, 4)]) == 0
    assert module.overlap([], [(2, 4)]) == 0


def test_invalid_interval_is_rejected():
    with pytest.raises(ValueError, match="Negative"):
        module.duration([(2, 1)])
