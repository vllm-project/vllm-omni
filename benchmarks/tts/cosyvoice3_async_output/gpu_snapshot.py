# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise the actual CUDA snapshot and background output lifecycle."""

import argparse
import json
from pathlib import Path

import torch

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.gpu_ar_model_runner import (
    OmniAsyncGPUModelRunnerOutput,
    _snapshot_tensor_payload_to_cpu_async,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    copy_stream = torch.cuda.Stream()
    token_stream = torch.cuda.Stream()
    for step in range(64):
        source = torch.full((1, 128), float(step), device="cuda")
        snapshot = _snapshot_tensor_payload_to_cpu_async(
            {"hidden_states": source, "embed.embedding": [source]},
            copy_stream=copy_stream,
            pin_memory=True,
        )

        def builder(snapshot=snapshot):
            snapshot.wait()
            return OmniModelRunnerOutput(
                req_ids=["request"],
                req_id_to_index={"request": 0},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=None,
                inter_stage_outputs=[
                    {
                        "hidden": snapshot.payload["hidden_states"],
                        "embed.embedding": snapshot.payload["embed.embedding"][0],
                    }
                ],
            )

        pending = OmniAsyncGPUModelRunnerOutput(
            model_runner_output_builder=builder,
            cuda_device=0,
            sampled_token_ids=torch.tensor([[step]], device="cuda", dtype=torch.int32),
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=token_stream,
            vocab_size=6564,
        )
        # Reuse the source before consuming the background output.
        source.fill_(-1)
        result = pending.get_output()
        torch.testing.assert_close(result.inter_stage_outputs[0]["hidden"], torch.full((1, 128), float(step)))
        torch.testing.assert_close(result.inter_stage_outputs[0]["embed.embedding"], torch.full((1, 128), float(step)))
        assert result.sampled_token_ids == [[step]]
    current_omni_platform.synchronize()
    result = {"iterations": 64, "cuda_snapshot": "passed", "background_output": "passed", "sampled_tokens": "passed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
