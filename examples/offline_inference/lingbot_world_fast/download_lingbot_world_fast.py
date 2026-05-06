import argparse
import fcntl
import json
import os
import site
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import snapshot_download

DEPENDENCY_REPO = "https://github.com/Robbyant/lingbot-world"
DEPENDENCY_BRANCH = "main"
CACHE_DIR = Path(tempfile.gettempdir()) / "vllm-omni-dependency"
LOCK_FILE = CACHE_DIR / ".install.lock"
DEPENDENCY_DIR = CACHE_DIR / "Lingbot-World"


def download_dependency():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if not DEPENDENCY_DIR.exists():
            print(f"Downloading Lingbot World Fast to {DEPENDENCY_DIR} ...")
            subprocess.run(
                ["git", "clone", "--depth", "1", DEPENDENCY_REPO, "--branch", DEPENDENCY_BRANCH, str(DEPENDENCY_DIR)],
                check=True,
            )
            print("Download finished.")
        fcntl.flock(f, fcntl.LOCK_UN)

    # write .pth to site-packages
    site_packages = Path(site.getsitepackages()[0])
    pth_file = site_packages / "vllm_omni_dependency.pth"
    pth_file.write_text(str(DEPENDENCY_DIR))
    print(f"Added {DEPENDENCY_DIR} to site-packages via {pth_file}")


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
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )

    elapsed = time.time() - start_time
    print(f"✅ Finished downloading {repo_id} in {elapsed:.2f} seconds. Files saved at: {local_dir}")


def main(output_dir: str):
    lingbot_base_dir = os.path.join(output_dir, "lingbot-world-base-cam")

    # Base Model
    timed_download(
        repo_id="robbyant/lingbot-world-base-cam",
        local_dir=lingbot_base_dir,
        allow_patterns=["google/*", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"],
    )

    lingbot_fast_dir = os.path.join(lingbot_base_dir, "Lingbot-World-Fast")

    timed_download(repo_id="robbyant/lingbot-world-fast", local_dir=lingbot_fast_dir)

    # Lingbot World does not come with config.json which is required by diffusers
    config = {
        "_class_name": "WanModel",
        "_diffusers_version": "0.33.0",
        "dim": 5120,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_dim": 36,
        "model_type": "i2v",
        "num_heads": 40,
        "num_layers": 40,
        "out_dim": 16,
        "text_len": 512,
    }

    with open(
        os.path.join(output_dir, "lingbot-world-base-cam", "Lingbot-World-Fast", "config.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(config, f, indent=2)

    print(f"model_index.json created at {os.path.join(output_dir, 'model_index.json')}")

    download_dependency()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument(
        "--output-dir", type=str, default="./lingbot_world", help="Base directory to save downloaded models"
    )
    args = parser.parse_args()
    main(args.output_dir)
