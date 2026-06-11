import argparse
import os
import shutil
import time

from huggingface_hub import snapshot_download


def timed_download(repo_id: str, local_dir: str, allow_patterns: list | None = None):
    """Download files from HF repo and log time + destination."""
    if os.path.exists(local_dir):
        print(f"Directory {local_dir} already exists. Skipping download.")
        return
    print(f"Starting download from {repo_id} into {local_dir}")
    start_time = time.time()

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )

    elapsed = time.time() - start_time
    print(f"✅ Finished downloading {repo_id} in {elapsed:.2f} seconds. Files saved at: {local_dir}")


def main(output_dir: str):
    wan_dir = os.path.join(output_dir, "wan_models", "Wan2.1-T2V-1.3B")

    # W2an Base Model
    timed_download(
        repo_id="Wan-AI/Wan2.1-T2V-1.3B",
        local_dir=wan_dir,
        allow_patterns=["google/*", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "config.json"],
    )

    # Copy WAN config.json to where it will be looked for by vllm omni
    shutil.copyfile(os.path.join(wan_dir, "config.json"), os.path.join(output_dir, "config.json"))

    # CausVid Weights
    timed_download(
        repo_id="tianweiy/CausVid",
        local_dir=os.path.join(output_dir, "causvid"),
        allow_patterns=["autoregressive_checkpoint/model.pt"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument("--output-dir", type=str, default="./causvid", help="Base directory to save downloaded models")
    args = parser.parse_args()
    main(args.output_dir)
