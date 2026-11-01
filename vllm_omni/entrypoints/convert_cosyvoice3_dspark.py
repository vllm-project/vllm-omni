# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Convert a compatible CosyVoice3 DSpark checkpoint to the speech vocabulary."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()

    from vllm_omni.model_executor.models.cosyvoice3.dspark import convert_checkpoint

    convert_checkpoint(arguments.source, arguments.destination)


if __name__ == "__main__":
    main()
