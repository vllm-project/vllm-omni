# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Do not bypass the existing request-state graph guard in an optimization probe."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    from vllm_omni.model_executor.models.zonos2.zonos2_talker import Zonos2TalkerForConditionalGeneration

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(enforce_eager=False),
    )
    try:
        Zonos2TalkerForConditionalGeneration(vllm_config=config)
    except ValueError as error:
        assert "enforce_eager=True" in str(error)
        args.out.write_text(
            json.dumps(
                {
                    "full_runtime_ar_graph": "unsupported_by_existing_guard",
                    "reason": str(error),
                    "guard_bypassed": False,
                    "production_changed": False,
                    "dac_component_graph_probe": "independent supported-shape experiment, not AR graph enablement",
                },
                indent=2,
            )
        )
    else:
        raise AssertionError("Unexpected change in the existing graph guard")


if __name__ == "__main__":
    main()
