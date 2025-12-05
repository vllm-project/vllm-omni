import os
import time
import json
import functools

import torch

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import (  # type: ignore[attr-defined]
            QwenImagePipeline,
        )
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (  # type: ignore[attr-defined]
            QwenImageTransformer2DModel,
        )
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (  # type: ignore[attr-defined]
            AutoencoderKLQwenImage,
        )


_ENABLE_PROFILING = os.getenv("QWEN_PROFILE_ENABLE", "").lower() in (
    "1", "true", "yes"
)

print(f"[sitecustomize] loaded, pid={os.getpid()}, profiling={_ENABLE_PROFILING}")

# GPU Synchronization Timing
def sync_time() -> float:
    if not _ENABLE_PROFILING:
        return time.perf_counter()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def log_record(record: dict) -> None:
    """Appending profiling records to a JSONL file"""
    if not _ENABLE_PROFILING:
        return

    path = os.getenv("QWEN_PROFILE_JSON", "")
    if not path:
        return
    dir_name = os.path.dirname(path)
    if dir_name:
            os.makedirs(dir_name, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")



#  patch helper
def patch_method(cls, method_name: str, wrapper_fn):
    if not _ENABLE_PROFILING:
        return

    if cls is None or not hasattr(cls, method_name):
        return

    orig = getattr(cls, method_name)
    if hasattr(orig, "__patched__"):
        return

    wrapped = wrapper_fn(orig)
    wrapped.__patched__ = True
    setattr(cls, method_name, wrapped)



def _patch():
    if not _ENABLE_PROFILING:
        print("[sitecustomize] profiling disabled, skip patch")
        return


    print("[sitecustomize] start patching QwenImagePipeline / transformer / VAE")

    # 1. text encoding: encode_prompt 
    def wrap_encode_prompt(orig_fn):
        @functools.wraps(orig_fn)
        def wrapper(self, *args, **kwargs):
            t0 = sync_time()
            out = orig_fn(self, *args, **kwargs)
            dt = sync_time() - t0

            log_record(
                {
                    "stage": "text_encoding",
                    "time": dt,
                    "pid": os.getpid(),
                }
            )
            return out

        return wrapper

    patch_method(QwenImagePipeline, "encode_prompt", wrap_encode_prompt)

    # 2. Diffusion main loop: Record the time taken for each step of transformer.forward.
    _denoise_state = {"steps": 0, "total_time": 0.0}

    def wrap_transformer_forward(orig_fn):
        @functools.wraps(orig_fn)
        def wrapper(self, *args, **kwargs):
            t0 = sync_time()
            out = orig_fn(self, *args, **kwargs)
            dt = sync_time() - t0

            _denoise_state["steps"] += 1
            _denoise_state["total_time"] += dt
            return out

        return wrapper

    patch_method(QwenImageTransformer2DModel, "forward", wrap_transformer_forward)

    # 3. pipeline.forward 
    def wrap_pipeline_forward(orig_fn):
        @functools.wraps(orig_fn)
        def wrapper(self, req):
            _denoise_state["steps"] = 0
            _denoise_state["total_time"] = 0.0

            t0 = sync_time()
            out = orig_fn(self, req)
            total_forward_time = sync_time() - t0

            steps = _denoise_state["steps"]
            total_denoise = _denoise_state["total_time"]
            avg_step = total_denoise / steps if steps > 0 else 0.0

            log_record(
                {
                    "stage": "diffusion",
                    "num_inference_steps": steps,
                    "denoise_total": total_denoise,
                    "denoise_step_avg": avg_step,
                    "pipeline_total_forward": total_forward_time,
                    "pid": os.getpid(),
                }
            )
            return out

        return wrapper

    patch_method(QwenImagePipeline, "forward", wrap_pipeline_forward)

    # 4. VAE decode 
    def wrap_vae_decode(orig_fn):
        @functools.wraps(orig_fn)
        def wrapper(self, *args, **kwargs):
            t0 = sync_time()
            out = orig_fn(self, *args, **kwargs)
            dt = sync_time() - t0

            log_record(
                {
                    "stage": "vae_decode",
                    "time": dt,
                    "pid": os.getpid(),
                }
            )
            return out

        return wrapper

    patch_method(AutoencoderKLQwenImage, "decode", wrap_vae_decode)

    print("[sitecustomize] patch done")


_patch()