import argparse
import fcntl
import os
import site
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import snapshot_download

DEPENDENCY_REPO = "https://github.com/bytedance/DreamID-V.git"
DEPENDENCY_BRANCH = "omni"
CACHE_DIR = Path(tempfile.gettempdir()) / "vllm-omni-dependency"
LOCK_FILE = CACHE_DIR / ".install.lock"
DEPENDENCY_DIR = CACHE_DIR / "DreamID-Omni"


def download_dependency():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOCK_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        if not DEPENDENCY_DIR.exists():
            print(f"Downloading DreamID-Omni to {DEPENDENCY_DIR} ...")
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
    # Wan2.2
    wan_dir = os.path.join(output_dir, "Wan2.2-TI2V-5B")
    timed_download(
        repo_id="Wan-AI/Wan2.2-TI2V-5B",
        local_dir=wan_dir,
        allow_patterns=["google/*", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.2_VAE.pth"],
    )

    # MMAudio
    mm_audio_dir = os.path.join(output_dir, "MMAudio")
    timed_download(
        repo_id="hkchengrex/MMAudio",
        local_dir=mm_audio_dir,
        allow_patterns=["ext_weights/best_netG.pt", "ext_weights/v1-16.pth"],
    )

    dreamid_dir = os.path.join(output_dir, "DreamID-Omni")

    timed_download(repo_id="XuGuo699/DreamID-Omni", local_dir=dreamid_dir)

    # Now we construct the config file
    import json

    data = {
        "_class_name": "DreamIDOmniPipeline",
    }

    with open(os.path.join(output_dir, "model_index.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"model_index.json created at {os.path.join(output_dir, 'model_index.json')}")

    transformer_dir = os.path.join(output_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    with open(os.path.join(transformer_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"fusion": "DreamID-Omni/dreamid_omni.safetensors"}, f)
    print(f"transformer/config.json created at {os.path.join(transformer_dir, 'config.json')}")

    # now we download the dependency code
    download_dependency()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument(
        "--output-dir", type=str, default="./dreamid_omni", help="Base directory to save downloaded models"
    )
    args = parser.parse_args()
    main(args.output_dir)
