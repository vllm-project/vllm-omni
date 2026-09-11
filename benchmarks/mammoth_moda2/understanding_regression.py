# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qualification harness, not a user-facing model example.

Runs the shared Omni API and shared prompt builder against frozen source trees.
Records exact completion tokens so baseline/current and A/B/A comparisons are
independent of human judgments about wording. No model arithmetic is patched.
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from vllm_omni import Omni
from vllm_omni.model_extras import build_x_to_text_prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mode", choices=["text-to-text", "image-to-text"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    has_image = args.mode == "image-to-text"
    prompts = (
        [
            "What color is the teapot? Answer with one color word.",
            "Name the two main objects on the table and their colors in one short sentence.",
        ]
        if has_image
        else [
            "What is 17 + 25? Answer with digits only.",
            "Translate the English word 'cat' into Chinese. Answer with the Chinese word only.",
        ]
    )
    image = Image.open(args.image).convert("RGB") if has_image else None
    omni = Omni(
        model=args.model,
        mode=args.mode,
        deploy_config=args.deploy_config,
        trust_remote_code=True,
        enforce_eager=True,
    )
    records = []
    try:
        sampling_params = list(omni.default_sampling_params_list)
        assert len(sampling_params) == 1, "Understanding must remain AR-only"
        params = sampling_params[0]
        params.max_tokens = 64
        params.temperature = 0.0
        params.top_p = 1.0
        params.seed = 42
        for index in (0, 1, 0):
            prompt, stop_ids = build_x_to_text_prompt(
                model_family="mammoth_moda2",
                model=args.model,
                prompt=prompts[index],
                has_image=has_image,
            )
            if image is not None:
                prompt["multi_modal_data"] = {"image": image}
            if stop_ids is not None:
                params.stop_token_ids = stop_ids
            start = time.perf_counter()
            outputs = list(omni.generate([prompt], sampling_params_list=sampling_params))
            completions = []
            for output in outputs:
                request_output = getattr(output, "request_output", output)
                for completion in request_output.outputs:
                    completions.append(
                        {
                            "text": completion.text,
                            "token_ids": list(completion.token_ids),
                            "finish_reason": completion.finish_reason,
                            "stop_reason": completion.stop_reason,
                        }
                    )
            record = {
                "prompt_index": index,
                "prompt": prompts[index],
                "elapsed_seconds": time.perf_counter() - start,
                "completions": completions,
            }
            records.append(record)
            Path(args.output).write_text(
                json.dumps(
                    {"model": args.model, "mode": args.mode, "records": records},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print(
                "QUALIFICATION_RESULT " + json.dumps(record, ensure_ascii=False),
                flush=True,
            )
            assert len(completions) == 1
            assert completions[0]["text"].strip() and completions[0]["token_ids"]
            assert completions[0]["finish_reason"] == "stop", "Unexpected truncation or failure"
        assert records[0]["completions"] == records[2]["completions"], "A/B/A replay changed"
    finally:
        omni.close()


if __name__ == "__main__":
    main()
