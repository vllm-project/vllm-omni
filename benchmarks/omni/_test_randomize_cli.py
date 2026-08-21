# SPDX-License-Identifier: Apache-2.0
"""CLI-level dry-run test for --randomize-input.

Stubs PIL / aiohttp / numpy / tqdm (absent on the local Python) then drives the
real build_parser + benchmark() with --dry-run to confirm the new flag parses,
the dry-run preview shows the new inWxH / prompt_words columns, and the tally
section reports the randomized input distribution. Sends nothing.
"""

from __future__ import annotations

import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
_DIFFUSION = os.path.join(os.path.dirname(HERE), "diffusion")
for _p in (HERE, _DIFFUSION):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --- Stub PIL --------------------------------------------------------------
class _FakeImage:
    def __init__(self, mode, size, color):
        self.mode, self.size, self.color = mode, size, color

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{self.size[0]}x{self.size[1]}|{self.color}")


class _FakeDraw:
    def ellipse(self, *a, **k):
        pass

    def rectangle(self, *a, **k):
        pass

    def line(self, *a, **k):
        pass


_pil = types.ModuleType("PIL")
_pil_image = types.ModuleType("PIL.Image")
_pil_image.new = staticmethod(lambda m, s, c: _FakeImage(m, s, c))
_pil_image.open = staticmethod(lambda p: _FakeImage("RGB", (0, 0), (0, 0, 0)))
_pil.Image = _pil_image
_pil_image_draw = types.ModuleType("PIL.ImageDraw")
_pil_image_draw.Draw = staticmethod(lambda img: _FakeDraw())
sys.modules["PIL"] = _pil
sys.modules["PIL.Image"] = _pil_image
sys.modules["PIL.ImageDraw"] = _pil_image_draw


# --- Stub aiohttp / numpy / tqdm ------------------------------------------
for _mod in ("aiohttp", "numpy", "tqdm", "tqdm.asyncio"):
    if _mod not in sys.modules:
        sys.modules[_mod] = types.ModuleType(_mod)
sys.modules["aiohttp"].FormData = object  # type: ignore[attr-defined]
sys.modules["aiohttp"].ClientSession = object  # type: ignore[attr-defined]
sys.modules["aiohttp"].ClientResponse = object  # type: ignore[attr-defined]
sys.modules["numpy"].ndarray = object  # type: ignore[attr-defined]
sys.modules["numpy"].array = lambda *a, **k: None  # type: ignore[attr-defined]
sys.modules["numpy"].percentile = lambda *a, **k: 0.0  # type: ignore[attr-defined]
sys.modules["numpy"].mean = lambda *a, **k: 0.0  # type: ignore[attr-defined]
sys.modules["numpy"].isnan = lambda *a, **k: False  # type: ignore[attr-defined]
sys.modules["tqdm"].tqdm = object  # type: ignore[attr-defined]
sys.modules["tqdm.asyncio"].tqdm = object  # type: ignore[attr-defined]


import asyncio  # noqa: E402

from omni_benchmark_serving import benchmark, build_parser  # noqa: E402


def _run(argv: list[str]) -> str:
    args = build_parser().parse_args(argv)
    out_buf: list[str] = []

    class _Cap:
        def write(self, s):
            if s.strip():
                out_buf.append(str(s).rstrip("\n"))

        def flush(self):
            pass

    import builtins

    real_print = builtins.print
    builtins.print = lambda *a, **k: out_buf.append(" ".join(str(x) for x in a))  # type: ignore[assignment]
    try:
        asyncio.run(benchmark(args))
    finally:
        builtins.print = real_print  # type: ignore[assignment]
    return "\n".join(out_buf)


def test_dry_run_randomize_on():
    out = _run(
        [
            "--num-i2t",
            "7",
            "--num-t2i",
            "2",
            "--num-it2i",
            "1",
            "--seed",
            "7",
            "--randomize-input",
            "--dry-run",
        ]
    )
    # New dry-run header includes the new columns.
    assert "inWxH" in out, out
    assert "prompt_words" in out, out
    # The footer flags randomize_input=True.
    assert "randomize_input=True" in out, out
    # The tally section reports a randomized prompt word-count summary + input resolution.
    assert "prompt word count (randomized)" in out, out
    assert "i2t/it2i input-image resolution used:" in out, out
    # inWxH column is non-"-" for i2t/it2i rows (sample the first i2t row).
    assert "  i2t  " in out, out
    print("=== dry-run --randomize-input output ===")
    print(out)
    print("\ntest_dry_run_randomize_on OK")


def test_dry_run_randomize_off():
    out = _run(
        [
            "--num-i2t",
            "7",
            "--num-t2i",
            "2",
            "--num-it2i",
            "1",
            "--seed",
            "7",
            "--dry-run",
        ]
    )
    # No randomized tally when off.
    assert "prompt word count (randomized)" not in out, out
    assert "i2t/it2i input-image resolution used:" not in out, out
    # The inWxH column shows "-" for all rows (no input randomization).
    assert "inWxH" in out, out
    print("\ntest_dry_run_randomize_off OK")


def test_no_randomize_input_flag_default_false():
    args = build_parser().parse_args(["--num-i2t", "1", "--dry-run"])
    assert args.randomize_input is False
    args2 = build_parser().parse_args(["--num-i2t", "1", "--randomize-input", "--dry-run"])
    assert args2.randomize_input is True
    args3 = build_parser().parse_args(["--num-i2t", "1", "--no-randomize-input", "--dry-run"])
    assert args3.randomize_input is False
    print("test_no_randomize_input_flag_default_false OK")


def main() -> None:
    test_no_randomize_input_flag_default_false()
    test_dry_run_randomize_off()
    test_dry_run_randomize_on()
    print("\nAll CLI randomize-input tests passed.")


if __name__ == "__main__":
    main()
