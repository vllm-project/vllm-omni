# Physis Same-Step video models

Experimental text-to-video pipelines for five post-trained Physis checkpoints.
These use the Same-Step schedule with the topology contract from PR #9169 and native Omni transformer,
encoder and decoder components. Full output quality equivalence to the model
reference is not established; this is not a general replacement for the base models.

## Checkpoints and fixed workloads

Download a complete Diffusers-format snapshot, including its transformer, text
encoder, tokenizer, connectors where applicable, and VAE. Pass an absolute local
snapshot directory to Omni. Model access requires the account's Hugging Face access.

| Repository under Physis-AI | Revision | Class | Wan profile | Chunks | Frames | Playback FPS |
| --- | --- | --- | --- | ---: | ---: | ---: |
| physisserve-wan2p1_1p3b-same_step-sdv2 | 8c0b1ff906dfb77951c08acbd9235430beec62fa | PhysisWanPipeline | wan21_1p3b | 19 | 81 | 16 |
| physisserve-wan2p1_14b-same_step-sdv2 | 41182358232c5579b1e27233a79d9c4d3cedd137 | PhysisWanPipeline | wan21_14b | 19 | 81 | 16 |
| physisserve-wan2p2_5b-same_step-ls-dev | 69cdd6cbb041a459daca6dfb3861a85cada65e6c | PhysisWanPipeline | wan22_5b | 29 | 121 | 24 |
| physisserve-ltx2p5_22b-same_step-ls-dev | 877fa76caa4e77bd92ec2ae00ee69849d1bfcf34 | PhysisLTXPipeline | — | 14 | 121 | 24 |
| physisserve-minimaxh3_33b-same_step-ls-dev | d8006165af6c3ebe021e3f973bb2d1ab0f8ec53c | PhysisH3Pipeline | — | 48 | 162 | 24 |

For example, with the Hugging Face CLI authenticated:

```bash
hf download Physis-AI/physisserve-wan2p1_1p3b-same_step-sdv2 \
  --revision 8c0b1ff906dfb77951c08acbd9235430beec62fa \
  --local-dir /path/to/physis-wan21-1p3b
```

All profiles currently require 832x480, four trained denoising steps,
guidance scale 1, one output and 18 history chunks plus a permanent clean sink.
The supplied deployment configs leave num_frames unset, so omitted frame counts
resolve from the selected profile and num_chunks before validation. When passing
OmniDiffusionSamplingParams directly, set num_frames explicitly or use None to
select that model default. Different explicit
resolution, step count or frame count raises ValueError. LTX uses a fixed model
time rate of 24; playback FPS is a separate output setting.

An omitted seed uses 0 for reproducible output. The same prompt, configuration
and seed produce the same video; set seed to another integer for a different
sample. If supplied, extra_args.seed takes precedence over the standard seed.

## Generate one video

The default deployment requires four GPUs, one replica per trained denoising
step. This example uses Wan 2.1 1.3B:

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

omni = Omni(
    model="/path/to/physis-wan21-1p3b",
    model_class_name="PhysisWanPipeline",
    deploy_config="vllm_omni/deploy/physis_wan.yaml",
)
try:
    outputs = omni.generate(
        {"prompt": "A corgi running along the beach at sunset, waves rolling in, golden light."},
        OmniDiffusionSamplingParams(
            height=480, width=832, num_frames=81, fps=16,
            num_inference_steps=4, guidance_scale=1.0, seed=0,
            output_type="np",
            extra_args={"num_chunks": 19, "cell_schedule": "same_step",
                        "num_denoise_steps": 4, "kv_history_chunks": 18},
        ),
    )
    print(outputs[0])
finally:
    omni.close()
```

For the other Wan checkpoints set model_config.physis_profile in a deployment
copy to the matching table value. LTX and H3 use physis_ltx.yaml and
physis_h3.yaml with their respective classes and request shapes.

## Parallel layouts and measurement

TP and SP remain 1. Four trained denoising steps require step_replicas S=4;
the deployment's pipeline_parallel_size is S times G:

| GPUs | step_replicas S | Layer groups G | Meaning |
| ---: | ---: | ---: | --- |
| 4 | 4 | 1 | One weight replica per denoising step |
| 8 | 4 | 2 | Each step replica splits its transformer layers between two GPUs |

Update stages[0].devices to logical IDs within the scheduler's visible GPU
allocation and stages[0].parallel_config.pipeline_parallel_size to the GPU
count. Set model_config.ar_diffusion_stage_config.step_replicas to 4.
Do not overwrite the scheduler's CUDA_VISIBLE_DEVICES.

One- and two-GPU results from before PR #9169 are historical evidence for the
earlier serial schedule. They do not validate a current supported deployment.

Text encoders run on CPU; rank 0 decodes the whole latent sequence on GPU.
H3 requires vae_use_tiling: true as supplied in its deploy config. LTX uses
FP32 latent/KV storage with BF16 transformer compute; it is not an all-BF16 run.
Streaming video delivery, audio, TP/SP combinations and NPU are not validated.

The E2E entry in benchmark PR #8728 measures Omni.generate including text
encoding, DiT, VAE and postprocessing, excluding model initialization and file
encoding. Run one warmup and three measured requests with the fixed workload
above. Keep DiT timing, E2E wall time and playback FPS separate. Scaling between
the layouts above changes both GPU count and the parallel layout.

For a four-GPU benchmark, use the selected Physis deployment config and keep
these values in its deployment copy; retain the model-specific fields:

```yaml
stages:
  - stage_id: 0
    devices: "0,1,2,3"
    parallel_config:
      pipeline_parallel_size: 4
      tensor_parallel_size: 1
      ulysses_degree: 1
      ring_degree: 1
      cfg_parallel_size: 1
    model_config:
      ar_diffusion_stage_config:
        step_replicas: 4
```

The four-step Same-Step schedule has no clean-history step. The permanent
clean sink is computed separately and remains part of each model's attention
history.
