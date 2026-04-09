# CFG-Parallel

This section describes how to add CFG-Parallel (Classifier-Free Guidance Parallel) to a diffusion pipeline. We use the Qwen-Image pipeline as the reference implementation.

---

## Table of Contents

- [Overview](#overview)
- [Step-by-Step Implementation](#step-by-step-implementation)
- [Customization](#customization)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Reference Implementations](#reference-implementations)
- [Summary](#summary)

---

## Overview

### What is CFG-Parallel?

In standard Classifier-Free Guidance, each diffusion step requires two forward passes through the transformer:

1. **Positive/Conditional**: Guided by the text prompt
2. **Negative/Unconditional**: Typically using empty or negative prompt

CFG-Parallel eliminates this bottleneck by distributing the two forward passes across different GPU ranks, allowing them to execute simultaneously rather than sequentially.

### Architecture

vLLM-omni provides `CFGParallelMixin` that encapsulates all CFG parallel logic. Pipelines inherit from this mixin and implement a `diffuse()` method that orchestrates the denoising loop.

| Method | Purpose | Automatic Behavior |
|--------|---------|-------------------|
| [`predict_noise_maybe_with_cfg()`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/distributed/cfg_parallel/) | Predict noise with CFG | Detects parallel mode, distributes computation, gathers results |
| [`scheduler_step_maybe_with_cfg()`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/distributed/cfg_parallel/) | Step scheduler | All ranks step locally (no broadcast needed) |
| [`combine_cfg_noise()`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/distributed/cfg_parallel/) | Combine positive/negative | Applies CFG formula with optional normalization |
| [`predict_noise()`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/distributed/cfg_parallel/) | Forward pass wrapper | Override for custom transformer calls |
| [`cfg_normalize_function()`](https://docs.vllm.ai/projects/vllm-omni/en/latest/api/vllm_omni/diffusion/distributed/cfg_parallel/) | Normalize CFG output | Override for custom normalization |

### How It Works

`predict_noise_maybe_with_cfg()` automatically detects and switches between two execution modes:

- **CFG-Parallel mode** (when `cfg_world_size > 1`):
  - Rank 0 computes positive prompt prediction
  - Rank 1 computes negative prompt prediction
  - Results are gathered via `all_gather()`
  - All ranks compute CFG combine locally (deterministic, identical results)

- **Sequential mode** (when `cfg_world_size == 1`):
  - Single rank computes both positive and negative predictions
  - Directly combines them with CFG formula

`scheduler_step_maybe_with_cfg()` ensures consistent latent states across all ranks:

- All ranks compute the scheduler step locally — no broadcast needed because `predict_noise_maybe_with_cfg` already ensures all ranks have identical noise predictions after `all_gather` + local combine.

---

## Step-by-Step Implementation

### Step 1: Inherit `CFGParallelMixin`

Allow your pipeline to inherit from `CFGParallelMixin` and implements the `diffuse()` method for your specific model.

**Example (Qwen-Image):**

```python
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
import torch.nn as nn
class YourModelPipeline(nn.Module, CFGParallelMixin):
    def diffuse(self, ...) -> torch.Tensor:
        for i, t in enumerate(timesteps):
            # Prepare positive_kwargs (conditional) and negative_kwargs (unconditional)
            positive_kwargs = {...}  # hidden_states, encoder_hidden_states, etc.
            negative_kwargs = {...} if do_true_cfg else None

            # Key method 1: Predict noise with automatic CFG parallel handling
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=true_cfg_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
            )

            # Key method 2: Step scheduler with automatic CFG synchronization
            latents = self.scheduler_step_maybe_with_cfg(
                noise_pred, t, latents, do_true_cfg
            )

        return latents
```

**Key Points:**

- `positive_kwargs`: transformer arguments for conditional (text-guided) prediction
- `negative_kwargs`: transformer arguments for unconditional prediction (set to `None` if CFG disabled)
- For image editing pipelines, add `output_slice=image_seq_len` to extract the generative image portion

### Step 2: Call `diffuse`

Call `self.diffuse` in your pipeline's forward function:

```python
import torch.nn as nn
class YourModelPipeline(nn.Module, CFGParallelMixin):
    def forward(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        **kwargs,
    ):
        # Encode prompts, Initialize latents, Get timesteps
        ...
        # Run diffusion loop (calls the mixin's diffuse method)
        latents = self.diffuse(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_embeds,
            negative_prompt_embeds_mask=negative_mask,
            latents=latents,
            timesteps=timesteps,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=guidance_scale,
            ...
        )
```

---

## Customization

### Override `predict_noise()` for Custom Transformer Calls

If your transformer requires custom prediction function, you can rewrite `predict_noise` function. Taking Wan2.2 as an example, which has two transformer models. The actual transformer to be called is determined by `self.transformer`.

```python
class Wan22Pipeline(nn.Module, CFGParallelMixin):
    def predict_noise(self, current_model: nn.Module | None = None, **kwargs: Any) -> torch.Tensor:
        if current_model is None:
            current_model = self.transformer
        return current_model(**kwargs)[0]
```


### Override `cfg_normalize_function()` for Custom Normalization

Some models have their own normalization function. Taking LongCat Image model as an example:

```python
class LongCatImagePipeline(nn.Module, CFGParallelMixin):
    def cfg_normalize_function(self, noise_pred, comb_pred, cfg_renorm_min=0.0):
        """
        Normalize the combined noise prediction.
        """
        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
        noise_pred = comb_pred * scale
        return noise_pred

        # The original cfg_normalize_function function in CFGParallelMixin
        # cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
        # noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
        # noise_pred = comb_pred * (cond_norm / noise_norm)
        # return noise_pred
```


### Override `combine_cfg_noise()` for Multi-Output Models

When `predict_noise()` returns a tuple (e.g., video + audio), the default `combine_cfg_noise()` applies CFG to every element. Override it to apply different logic per element — for example, CFG on video but positive-only on audio:

```python
class MyVideoAudioPipeline(nn.Module, CFGParallelMixin):
    def combine_cfg_noise(self, positive_noise_pred, negative_noise_pred, scale, normalize):
        (video_pos, audio_pos) = positive_noise_pred
        (video_neg, audio_neg) = negative_noise_pred
        video_combined = super().combine_cfg_noise(video_pos, video_neg, scale, normalize)
        return (video_combined, audio_pos)  # audio: positive only, no CFG
```

This also requires `predict_noise()` to return a tuple (see [Override predict_noise](#override-predict_noise-for-custom-transformer-calls) above).

### Implement a Composite Scheduler for Multi-Output Models

When each output has its own denoising schedule, implement a composite scheduler that dispatches to per-output schedulers. Assign it to `self.scheduler` so the default `scheduler_step()` works without override.

**Complete example (video + audio with separate schedulers and diffuse loop):**

```python
class VideoAudioScheduler:
    """Composite scheduler dispatching to video and audio schedulers."""
    def __init__(self, video_scheduler, audio_scheduler):
        self.video_scheduler = video_scheduler
        self.audio_scheduler = audio_scheduler

    def step(self, noise_pred, t, latents, return_dict=False, generator=None):
        video_out = self.video_scheduler.step(noise_pred[0], t[0], latents[0], return_dict=False, generator=generator)[0]
        audio_out = self.audio_scheduler.step(noise_pred[1], t[1], latents[1], return_dict=False, generator=generator)[0]
        return ((video_out, audio_out),)

class MyVideoAudioPipeline(nn.Module, CFGParallelMixin):
    def __init__(self, ...):
        self.scheduler = VideoAudioScheduler(video_sched, audio_sched)

    def predict_noise(self, **kwargs):
        video_pred, audio_pred = self.transformer(**kwargs)
        return (video_pred, audio_pred)

    def combine_cfg_noise(self, positive_noise_pred, negative_noise_pred, scale, normalize):
        # ... (as above)

    def diffuse(self, video_latents, audio_latents, timesteps_video, timesteps_audio, ...):
        for t_v, t_a in zip(timesteps_video, timesteps_audio):
            positive_kwargs = {...}  
            negative_kwargs = {...} if do_true_cfg else None

            video_pred, audio_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg, true_cfg_scale=self.guidance_scale,
                positive_kwargs=positive_kwargs, negative_kwargs=negative_kwargs,
            )
            video_latents, audio_latents = self.scheduler_step_maybe_with_cfg(
                (video_pred, audio_pred), (t_v, t_a),
                (video_latents, audio_latents), do_true_cfg=do_true_cfg,
                generator=generator,
            )
        return video_latents, audio_latents
```

> **Note:** If you use a non-deterministic scheduler, e.g., DDPM, please set `self.scheduler_step_maybe_with_cfg(..., generator=torch.Generator(device).manual_seed(seed))` explicitly to control the randomness of scheduler step among ranks.

---

## Testing

After adding CFG-Parallel support, test with:

```bash
cd examples/offline_inference/text_to_image
python text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --negative-prompt "ugly, unclear" \
    --cfg-scale 4.0 \
    --num-inference-steps 50 \
    --output "cfg_enabled.png" \
    --cfg-parallel-size 2
```

**Verify:**

1. Check logs for CFG parallel being activated
2. Record the `e2e_time_ms` in the log and compare with CFG-Parallel disabled
3. Compare the generated result quality with baseline
4. Record comparison results in your PR

---

## Troubleshooting

### Issue: CFG parallel not activating

**Symptoms:** Generation still slow, logs don't show CFG parallel being used.

**Causes & Solutions:**

- **CFG is not enabled:**

**Problem:** Guidance scale too low or negative prompt not provided.

**Solution:** Ensure `guidance_scale > 1.0` and negative prompt is provided:
```python
images = pipeline(
    prompt="a cat",
    negative_prompt="",  # Must provide (even if empty)
    guidance_scale=3.5,   # Must be > 1.0
)
```

---

## Reference Implementations

Complete examples in the codebase:

| Model | Path | Pattern | Notes |
|-------|------|---------|-------|
| **Qwen-Image** | `vllm_omni/diffusion/models/qwen_image/cfg_parallel.py` | Mixin | Dual-stream transformer |
| **Qwen-Image-Edit** | `vllm_omni/diffusion/models/qwen_image/pipeline_qwen_image_edit.py` | Mixin | Image editing with `output_slice` |
| **Wan2.2** | `vllm_omni/diffusion/models/wan2_2/pipeline_wan2_2.py` | Mixin | Dual-transformer architecture |
| **CFGParallelMixin** | `vllm_omni/diffusion/distributed/cfg_parallel.py` | Base implementation | Core mixin class |

---

## Summary

Adding CFG-Parallel support:

1. ✅ **Create mixin** - Inherit from `CFGParallelMixin` and implement `diffuse()` method
2. ✅ **(Optional) Customize** - Override `predict_noise()` or `cfg_normalize_function()` for custom behavior
3. ✅ **Test** - Verify with `--cfg-parallel-size 2` and compare performance
