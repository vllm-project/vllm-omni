# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _load_module():
    script = Path(__file__).parents[2] / "examples/offline_inference/minimax_h3/analyze_nsys.py"
    spec = importlib.util.spec_from_file_location("analyze_minimax_h3_nsys", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aggregate_categories_and_gpu_balance(tmp_path):
    analyzer = _load_module()
    assert analyzer.classify_kernel("cudnn_generated_fort_native_sdpa_sm100_flash_fprop_f16") == "Dense Attention"
    assert analyzer.classify_kernel("ncclDevKernel_AllReduce_Sum_bf16_RING_LL") == "NCCL AllReduce"
    assert analyzer.classify_kernel("ncclKernel-ReduceScatter") == "NCCL ReduceScatter"
    assert analyzer.classify_kernel("ncclKernel_Broadcast") == "NCCL Broadcast"
    database = tmp_path / "report.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE NVTX_EVENTS (
                start INTEGER, end INTEGER, text TEXT, textId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
                start INTEGER, end INTEGER, deviceId INTEGER,
                demangledName INTEGER, shortName INTEGER
            );
            """
        )
        strings = [
            (1, "ncclKernel_AllGather"),
            (2, "ncclDevKernel_SendRecv"),
            (3, "ncclDevKernel_AllReduce_Sum_bf16_RING_LL"),
            (4, "ncclKernel_ReduceScatter"),
            (5, "ncclKernel_Broadcast"),
            (6, "ncclMysteryKernel"),
            (7, "fmhaSm120Kernel"),
            (8, "gemm_kernel"),
        ]
        connection.executemany("INSERT INTO StringIds VALUES (?, ?)", strings)
        connection.executemany(
            "INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, NULL)",
            [
                (0, 1100, "minimax_h3_task:t2va"),
                (2000, 3100, "minimax_h3_task:fl2va_first_frame"),
            ],
        )
        kernels = []
        for base in (0, 2000):
            for device in range(4):
                offset = base + device * 2
                kernels.extend(
                    [
                        (offset + 10, offset + 110, device, 1, 1),
                        (offset + 120, offset + 170, device, 2, 2),
                        (offset + 180, offset + 260, device, 3, 3),
                        (offset + 270, offset + 310, device, 4, 4),
                        (offset + 320, offset + 350, device, 5, 5),
                        (offset + 360, offset + 380, device, 6, 6),
                        (offset + 390, offset + 590, device, 7, 7),
                        (offset + 600, offset + 1000, device, 8, 8),
                    ]
                )
        connection.executemany(
            "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)",
            kernels,
        )

    results = analyzer.analyze(database)
    assert set(results) == {"T2V", "I2V"}
    t2v = results["T2V"]
    assert t2v["categories"]["NCCL AllGather"]["percent"] == 100 / 920 * 100
    assert t2v["categories"]["NCCL SendRecv"]["percent"] == 50 / 920 * 100
    assert t2v["categories"]["NCCL AllReduce"]["percent"] == 80 / 920 * 100
    assert t2v["categories"]["NCCL ReduceScatter"]["percent"] == 40 / 920 * 100
    assert t2v["categories"]["NCCL Broadcast"]["percent"] == 30 / 920 * 100
    assert t2v["categories"]["NCCL Other"]["percent"] == 20 / 920 * 100
    assert t2v["categories"]["Dense Attention"]["percent"] == 200 / 920 * 100
    assert t2v["nccl_total"]["percent"] == 320 / 920 * 100
    assert t2v["top_nccl_kernels"][0]["name"] == "ncclKernel_AllGather"
    assert t2v["load_balance"]["max_deviation_percent"] == 0
    markdown = analyzer.render_markdown(results, top_nccl_kernels=3)
    assert "### T2V" in markdown and "### I2V" in markdown
    assert "NCCL AllReduce：8.70%" in markdown
    assert "NCCL 未分类：2.17%" in markdown
    assert "NCCL 总计：34.78%" in markdown
    assert "耗时最高的 NCCL kernels（前 3 个）" in markdown
