"""Focused edge-case tests for test environment helpers."""

from __future__ import annotations

import pytest

from tests.helpers.env import get_physical_device_indices


class TestGetPhysicalDeviceIndices:
    def test_no_visible_devices_returns_identity(self) -> None:
        assert get_physical_device_indices([0, 1, 2]) == [0, 1, 2]

    def test_empty_list_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert get_physical_device_indices([]) == []

    def test_maps_logical_to_physical(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        assert get_physical_device_indices([0, 1]) == [2, 3]

    def test_rejects_out_of_range_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        with pytest.raises(ValueError, match="Device index 5 is not in visible devices"):
            get_physical_device_indices([0, 5])

    def test_rejects_negative_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        with pytest.raises(ValueError, match="Device index -1 is not in visible devices"):
            get_physical_device_indices([-1])
