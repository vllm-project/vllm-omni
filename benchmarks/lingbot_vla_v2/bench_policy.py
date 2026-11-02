# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare direct policy latency with fixed noise and interleaved ABBA order."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2 import LingbotVLAV2Config
from lerobot.policies.lingbot_vla_v2.modeling_lingbot_vla_v2 import LingbotVLAV2Policy
from safetensors.torch import load_file

import vllm_omni
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.lingbot_vla_v2.pipeline_lingbot_vla_v2 import LingBotVLAV2Pipeline
from vllm_omni.platforms import current_omni_platform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("lingbot-policy-benchmark.json"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.use_deterministic_algorithms(False)
    model = args.model
    cfg = LingbotVLAV2Config.from_pretrained(model)
    cfg.tokenizer_path = str(args.backbone_config)
    cfg.device = "cuda"
    cfg.dtype = torch.float32
    cfg.num_steps = 10
    reference = LingbotVLAV2Policy(cfg)
    reference.load_state_dict(load_file(model / "model.safetensors"), strict=True)
    reference.float().cuda().eval()
    pre, post = make_pre_post_processors(
        cfg, pretrained_path=model, preprocessor_overrides={"device_processor": {"device": "cuda"}}
    )
    native = LingBotVLAV2Pipeline(
        od_config=OmniDiffusionConfig(
            model=str(model),
            model_class_name="LingBotVLAV2Pipeline",
            dtype=torch.float32,
            num_gpus=1,
            model_config={"tokenizer_path": str(args.backbone_config)},
        )
    )
    state = native.processor.stats["observation.state"]["mean"].numpy().copy()
    y, x = np.mgrid[:256, :256]
    obs = {"state": state, "prompt": "pick up the red block and place it in the bin"}
    for i, key in enumerate(native.config.camera_keys):
        obs[key] = np.stack([(x + i * 71) % 256, (y + i * 31) % 256, (x // 16 + y // 16) % 2 * 255], -1).astype(
            np.uint8
        )

    def inputs_for_reference(policy, preprocessor, observation):
        batch = {
            "observation.state": torch.from_numpy(observation["state"]).unsqueeze(0),
            "task": [observation["prompt"]],
        }
        for key in native.config.camera_keys:
            batch[key] = torch.from_numpy(observation[key]).permute(2, 0, 1).float().unsqueeze(0) / 255
        names = ["images", "img_masks", "lang_tokens", "lang_masks", "state", "image_grid_thw"]
        return dict(zip(names, policy._extract_model_inputs(preprocessor(batch)), strict=True))

    noise = torch.randn((1, 50, 55), device="cuda", generator=torch.Generator(device="cuda").manual_seed(123))
    with torch.inference_mode():
        ref_inputs = inputs_for_reference(reference, pre, obs)
        native_inputs = native.processor.build_model_inputs(obs)
        for key in ref_inputs:
            torch.testing.assert_close(ref_inputs[key], native_inputs[key], atol=0, rtol=0)

        def hf_model():
            return reference.model.sample_actions(**ref_inputs, noise=noise.clone())

        def omni_model():
            return native.model.sample_actions(**native_inputs, noise=noise)

        def hf_policy():
            return (
                post(reference.model.sample_actions(**inputs_for_reference(reference, pre, obs), noise=noise.clone()))[
                    0
                ]
                .cpu()
                .numpy()
            )

        def omni_policy():
            return native.processor.build_model_outputs(
                native.model.sample_actions(**native.processor.build_model_inputs(obs), noise=noise)
            )

        expected = hf_policy()
        actual = omni_policy()
        np.testing.assert_allclose(actual, expected, atol=0.0001, rtol=1e-05)
        rows = []
        for name, functions in [
            ("model_only", {"HF": hf_model, "Omni": omni_model}),
            ("processor_and_model", {"HF": hf_policy, "Omni": omni_policy}),
        ]:
            for _ in range(3):
                for fn in functions.values():
                    fn()
            torch.accelerator.synchronize()
            samples = {"HF": [], "Omni": []}
            blocks = []
            for block in range(10):
                current = {"HF": [], "Omni": []}
                for label in ["HF", "Omni", "Omni", "HF"]:
                    torch.accelerator.synchronize()
                    start = time.perf_counter()
                    functions[label]()
                    torch.accelerator.synchronize()
                    ms = (time.perf_counter() - start) * 1000
                    samples[label].append(ms)
                    current[label].append(ms)
                blocks.append(statistics.mean(current["HF"]) / statistics.mean(current["Omni"]))
            row = {
                "scope": name,
                "samples_ms": samples,
                "median_ms": {k: statistics.median(v) for k, v in samples.items()},
                "paired_ratio_median": statistics.median(blocks),
                "paired_ratio_range": [min(blocks), max(blocks)],
            }
            rows.append(row)
            print("BENCHMARK", row["scope"], row["median_ms"], row["paired_ratio_median"], flush=True)
        result = {
            "rows": rows,
            "parity_max_abs": float(np.max(np.abs(actual - expected))),
            "warmup_per_path": 3,
            "abba_blocks": 10,
            "batch_size": 1,
            "views": 3,
            "flow_steps": 10,
            "horizon": 50,
            "dtype": "float32",
            "matmul_tf32": False,
            "cudnn_tf32": True,
            "device": current_omni_platform.get_device_name(),
            "model": str(model),
            "vllm_omni": vllm_omni.__version__,
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "scope": "Direct inference on the same device/runtime; excludes loading, HTTP and simulation.",
        }
        args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
