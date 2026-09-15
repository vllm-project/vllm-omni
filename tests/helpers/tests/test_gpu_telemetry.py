# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.helpers import gpu_telemetry as gt
from tests.helpers.gpu_telemetry import (
    GpuTelemetrySampler,
    decode_throttle_reasons,
    format_summary_line,
    interval_from_env,
    parse_sample_line,
    summarize_samples,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# index, sm, sm_max, mem, power, temp, util, throttle
_ROW = "0, 1980, 1980, 2619, 412.50, 61, 96, 0x0000000000000000"


def test_parse_sample_line_with_throttle():
    s = parse_sample_line(_ROW, has_throttle=True)
    assert s is not None
    assert (s.index, s.sm_mhz, s.sm_max_mhz, s.mem_mhz) == (0, 1980.0, 1980.0, 2619.0)
    assert (s.power_w, s.temp_c, s.util_pct, s.throttle_mask) == (412.5, 61.0, 96.0, 0)


def test_parse_sample_line_without_throttle():
    s = parse_sample_line("1, 1500, 1980, 2619, 300, 55, 40", has_throttle=False)
    assert s is not None
    assert s.index == 1 and s.sm_mhz == 1500.0 and s.throttle_mask is None


def test_parse_sample_line_rejects_wrong_arity_and_na():
    assert parse_sample_line("0, 1980, 1980", has_throttle=True) is None
    assert parse_sample_line("", has_throttle=False) is None
    s = parse_sample_line("0, [N/A], 1980, [N/A], [N/A], 61, 96, 0x4", has_throttle=True)
    assert s is not None
    assert s.sm_mhz is None and s.util_pct == 96.0 and s.throttle_mask == 0x4


def test_decode_throttle_reasons_drops_idle_bit():
    assert decode_throttle_reasons(0x0) == []
    assert decode_throttle_reasons(0x1) == []
    assert decode_throttle_reasons(0x4 | 0x40) == ["sw_power_cap", "hw_thermal_slowdown"]
    assert decode_throttle_reasons(0x1 | 0x8) == ["hw_slowdown"]


def _s(index=0, sm=1980.0, util=95.0, mask=0, sm_max=1980.0, power=400.0, temp=60.0):
    return gt.GpuSample(index, sm, sm_max, 2619.0, power, temp, util, mask)


def test_summarize_samples_busy_only_for_clocks():
    samples = [
        _s(util=5, sm=345),  # idle between requests: must not drag clock stats down
        _s(util=95, sm=1980),
        _s(util=90, sm=1755, mask=0x4),
        _s(util=60, sm=1980, mask=0x1),
    ]
    out = summarize_samples(samples)
    g = out["0"]
    assert g["samples"] == 4 and g["busy_samples"] == 3
    assert g["sm_mhz_busy"] == {"min": 1755.0, "mean": pytest.approx((1980 + 1755 + 1980) / 3), "max": 1980.0}
    assert g["sm_ratio_busy_min"] == pytest.approx(1755 / 1980)
    assert g["util_pct"]["min"] == 5.0 and g["util_pct"]["max"] == 95.0
    assert g["util_pct_busy_mean"] == pytest.approx((95 + 90 + 60) / 3)
    assert g["mem_mhz_busy"] == {"min": 2619.0, "mean": 2619.0, "max": 2619.0}
    assert g["throttle_reasons"] == {"sw_power_cap": 1}


def test_summarize_samples_multi_gpu_and_empty():
    out = summarize_samples([_s(index=0), _s(index=1, util=10)])
    assert set(out) == {"0", "1"}
    assert out["1"]["busy_samples"] == 0 and out["1"]["sm_mhz_busy"] is None and out["1"]["mem_mhz_busy"] is None
    assert summarize_samples([]) == {}


def test_format_summary_line():
    assert format_summary_line({"available": False, "reason": "nvidia-smi not found"}).endswith(
        "(nvidia-smi not found)"
    )
    line = format_summary_line({"available": True, "gpus": summarize_samples([_s(), _s(sm=1755, mask=0x4)])})
    assert "gpu0: sm 1755/1868/1980MHz (0.89x max) util_busy 95% 2/2 busy throttle=sw_power_cap:1" in line


def test_interval_from_env(monkeypatch):
    monkeypatch.delenv(gt.ENV_INTERVAL, raising=False)
    assert interval_from_env() == gt.DEFAULT_INTERVAL_S
    monkeypatch.setenv(gt.ENV_INTERVAL, "0.5")
    assert interval_from_env() == 0.5
    monkeypatch.setenv(gt.ENV_INTERVAL, "0")
    assert interval_from_env() == 0.0
    monkeypatch.setenv(gt.ENV_INTERVAL, "garbage")
    assert interval_from_env() == gt.DEFAULT_INTERVAL_S
    for bad in ("nan", "inf", "-inf", "-1"):
        monkeypatch.setenv(gt.ENV_INTERVAL, bad)
        assert interval_from_env() == gt.DEFAULT_INTERVAL_S


def test_sampler_disabled_by_env(monkeypatch):
    monkeypatch.setenv(gt.ENV_INTERVAL, "0")
    with GpuTelemetrySampler() as t:
        pass
    assert t.summary() == {"available": False, "reason": f"disabled ({gt.ENV_INTERVAL}=0)"}


def test_sampler_unavailable_without_nvidia_smi(monkeypatch):
    monkeypatch.delenv(gt.ENV_INTERVAL, raising=False)
    monkeypatch.setattr(gt.shutil, "which", lambda _: None)
    with GpuTelemetrySampler() as t:
        pass
    assert t.summary() == {"available": False, "reason": "nvidia-smi not found"}


def test_sampler_falls_back_to_legacy_throttle_field(monkeypatch):
    monkeypatch.delenv(gt.ENV_INTERVAL, raising=False)
    monkeypatch.setattr(gt.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    seen = []

    def fake_query(self, fields):
        seen.append(fields)
        if "clocks_event_reasons.active" in fields:
            return None  # old driver rejects the new name
        if "clocks_throttle_reasons.active" in fields:
            return _ROW + "\n"
        return "0, 1980, 1980, 2619, 412.50, 61, 96\n"

    monkeypatch.setattr(GpuTelemetrySampler, "_run_query", fake_query)
    t = GpuTelemetrySampler(interval_s=0.01)
    t.start()
    t.stop()
    assert t._throttle_field == "clocks_throttle_reasons.active"
    s = t.summary()
    assert s["available"] is True and s["throttle_field"] == "clocks_throttle_reasons.active"
    assert s["gpus"]["0"]["samples"] >= 1


def test_sampler_collects_samples_from_fake_nvidia_smi(monkeypatch):
    monkeypatch.delenv(gt.ENV_INTERVAL, raising=False)
    monkeypatch.setattr(gt.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    rows = iter(
        [
            "0, 345, 1980, 2619, 70, 40, 0, 0x0000000000000001\n",
            "0, 1980, 1980, 2619, 410, 62, 97, 0x0000000000000000\n",
            "0, 1620, 1980, 2619, 700, 78, 94, 0x0000000000000004\n",
        ]
    )
    last = "0, 1620, 1980, 2619, 700, 78, 94, 0x0000000000000004\n"
    monkeypatch.setattr(GpuTelemetrySampler, "_run_query", lambda self, fields: next(rows, last))

    with GpuTelemetrySampler(interval_s=0.005) as t:
        import time

        time.sleep(0.1)
    s = t.summary()
    assert s["available"] is True and s["interval_s"] == 0.005
    g = s["gpus"]["0"]
    assert g["samples"] >= 3
    assert g["sm_mhz_busy"]["min"] == 1620.0 and g["sm_max_mhz"] == 1980.0
    assert g["throttle_reasons"].get("sw_power_cap", 0) >= 1
    assert "gpu0:" in format_summary_line(s)


def test_visible_device_arg(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert gt._visible_device_arg() == ["-i", "2,3"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc,MIG-def")
    assert gt._visible_device_arg() == []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert gt._visible_device_arg() == []
