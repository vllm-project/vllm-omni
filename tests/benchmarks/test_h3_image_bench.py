# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Integration checks against a local fake image API; no GPU requests."""

import asyncio
import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
from aiohttp import web
from PIL import Image

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


class ClientIntegration(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, mode):
        counts = {"posts": 0, "active": 0, "peak": 0}
        image = Image.new("RGB", (8, 8))
        image.putpixel((3, 2), (255, 60, 20))
        buf = io.BytesIO()
        image.save(buf, format="PNG")

        async def generate(request):
            counts["posts"] += 1
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
            data = await request.json()
            self.assertEqual(data["size"], "8x8")
            self.assertEqual(data["num_inference_steps"], 6)
            await asyncio.sleep(0.05)
            counts["active"] -= 1
            if mode == "http_error":
                return web.json_response({"error": "test failure"}, status=500)
            payload = base64.b64encode(buf.getvalue()).decode() if mode == "success" else "not-base64!"
            return web.json_response({"created": 1, "data": [{"b64_json": payload}], "timings": {"generation": 1}})

        app = web.Application()
        app.router.add_post("/v1/images/generations", generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            with tempfile.TemporaryDirectory() as directory:
                out = Path(directory) / "run"
                script = Path(__file__).resolve().parents[2] / "benchmarks/diffusion/h3_image_bench.py"
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    str(script),
                    "--output-dir",
                    str(out),
                    "--base-url",
                    f"http://127.0.0.1:{port}",
                    "--phases",
                    "c1-b1",
                    "c2",
                    "--num-prompts",
                    "2",
                    "--warmups",
                    "1",
                    "--width",
                    "8",
                    "--height",
                    "8",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()
                report = json.loads((out / "report.json").read_text())
                if mode == "success":
                    self.assertEqual(process.returncode, 0, stderr.decode())
                    self.assertEqual(counts["posts"], 6)
                    self.assertEqual(counts["peak"], 2)
                    self.assertEqual(report["valid_outputs"], 6)
                    self.assertEqual(report["measured_successes"], 4)
                    self.assertEqual(len(list((out / "images").glob("*.png"))), 6)
                    records = [json.loads(line) for line in (out / "requests.jsonl").read_text().splitlines()]
                    self.assertNotIn("b64_json", json.dumps([r["response_metadata"] for r in records]))
                    self.assertTrue(
                        all(r["validated_delivery_latency_s"] >= r["delivery_latency_s"] > 0 for r in records)
                    )
                    self.assertEqual([g["concurrency"] for g in report["groups"]], [1, 2])
                else:
                    self.assertEqual(process.returncode, 1)
                    self.assertEqual(counts["posts"], 1, "Failed POST must not be retried")
                    self.assertEqual(report["status"], "failed")
                    self.assertEqual(report["valid_outputs"], 0)
                    state_file = next((out / "request-state").glob("*.json"))
                    self.assertEqual(json.loads(state_file.read_text())["status"], "failed")
        finally:
            await runner.cleanup()

    async def test_success_and_closed_loop_concurrency(self):
        await self.exercise("success")

    async def test_http_error_stops_without_retry(self):
        await self.exercise("http_error")

    async def test_invalid_image_stops_without_retry(self):
        await self.exercise("bad_image")

    async def test_duplicate_phases_are_rejected_before_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            script = Path(__file__).resolve().parents[2] / "benchmarks/diffusion/h3_image_bench.py"
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script),
                "--output-dir",
                str(out),
                "--base-url",
                "http://127.0.0.1:1",
                "--timeout",
                "0.1",
                "--phases",
                "c1-b1",
                "c1-b1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            self.assertEqual(process.returncode, 2)
            self.assertIn("Phases must be unique", stderr.decode())
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
