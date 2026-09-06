# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Smoke test for the YAML config surface, not for accuracy.

Asserts only that a deploy config carrying ``omni_kv_config`` reaches
``KVTransferConfig`` and the engine produces output -- the path a user actually
writes, which the accuracy tests bypass by patching stages in Python. What the
restore produces is checked by ``test_kv_offload_hit_vs_cold`` and
``test_kv_offload_consistency``.
"""

import tempfile

import pytest
import yaml

pytestmark = [pytest.mark.advanced_model, pytest.mark.omni, pytest.mark.cuda]

DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-3B"

MODES = {
    "lmcache": {
        "kv_store_config": {
            "lmcache_config": {
                "config_file": "",  # uses default LMCache config
            }
        }
    },
}


def build_stage_config(model: str, mode: str) -> str:
    """Build a temp stage config YAML with the specified KV config mode."""
    omni_kv_config = MODES[mode]

    config = {
        # main now validates async_chunk producers; Qwen2.5-Omni has no
        # async-chunk stage processor, so pin it off for this offline config.
        "async_chunk": False,
        "stages": [
            {
                "stage_id": 0,
                "max_model_len": 512,
                "max_num_batched_tokens": 512,
                "max_num_seqs": 4,
                "gpu_memory_utilization": 0.8,
                "skip_mm_profiling": True,
                "enforce_eager": True,
                "trust_remote_code": True,
                "enable_prefix_caching": False,
                "devices": "0",
                "omni_kv_config": omni_kv_config,
                "default_sampling_params": {
                    "temperature": 0.0,
                    "max_tokens": 64,
                    "seed": 42,
                },
            }
        ],
    }

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.flush()
    return tmp.name


def _run(model: str, mode: str, num_prompts: int = 3) -> bool:
    from vllm_omni.entrypoints.omni import Omni

    config_path = build_stage_config(model, mode)
    omni = Omni(
        model=model,
        deploy_config=config_path,
        stage_init_timeout=300,
        batch_timeout=5,
        init_timeout=300,
        log_stats=True,
    )

    prompts = [
        {"prompt": f"<|im_start|>user\nCount from 1 to {i + 5}.<|im_end|>\n<|im_start|>assistant\n"}
        for i in range(num_prompts)
    ]
    sampling_params_list = omni.default_sampling_params_list
    outputs = omni.generate(prompts, sampling_params_list)
    omni.close()

    return any(
        out.request_output and out.request_output.outputs and out.request_output.outputs[0].text.strip()
        for out in outputs
        if out.final_output_type == "text"
    )


@pytest.mark.parametrize("mode", list(MODES.keys()))
def test_kv_offload_modes(mode):
    pytest.importorskip("lmcache", reason="lmcache not installed")
    assert _run(DEFAULT_MODEL, mode), (
        f"No text output for mode={mode}; the YAML config surface did not reach the engine"
    )
