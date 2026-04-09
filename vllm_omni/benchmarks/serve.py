import argparse
import asyncio
from typing import Any

from vllm.benchmarks.serve import main_async


def main(args: argparse.Namespace) -> dict[str, Any]:
    return asyncio.run(main_async(args))
