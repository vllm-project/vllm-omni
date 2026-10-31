# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit, pinned asset preparation before offline ZONOS2 GPU CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

MODEL_REVISION = "65f1e80f94b599d474bb6af9094a803dc52f60bd"
MODEL_HASHES = {
    "model.pth": "5f6aa0fff9036ee44ccbc625d40aa6bdd8ea223480a5447e9f6aad70c38b6ecd",
    "params.json": "3514236d14470d6fe44a658d02b05d1b05383ab32c6015fb59ae1bee7b31a930",
}
SPEAKER_REVISION = "7577f61c42737fc8064bba773e2a18602df92803"
SPEAKER_HASHES = {
    "config.json": "297ac64afea59191e5aa446cb8acfdfdecfcd34771edda47d6948d1be5834ae9",
    "configuration_ecapa_tdnn.py": "6e187fd0adb8245829c855614e880551e2ec14c2372b2e5ad3c7e6565726d860",
    "modeling_ecapa_tdnn.py": "88281ea40ad4792943d598d416476faa4b311acedf153350a556ceaa6b55805a",
    "model.safetensors": "df60a638e7f4a29331c0af2bd2984ee5b992fee9d5923c776f7e4bdc3dedea48",
}
DAC_URL = "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth"
DAC_SHA256 = "a88eed82a7024ccc1facdb1e605c4c2f99281c8118c22c9895ffa846d8fb61aa"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(directory: Path, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        path = directory / name
        if not path.is_file() or file_digest(path) != digest:
            raise ValueError(f"Missing or incorrect pinned asset: {path}")


def verify_model(directory: Path) -> None:
    import torch
    from safetensors import safe_open

    manifest = json.loads((directory / "manifest.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if manifest.get("source_sha256") != MODEL_HASHES or manifest.get("tensor_count") != 507:
        raise ValueError("ZONOS2 converted manifest is not from the pinned source checkpoint")
    if config.get("model_type") != "zonos2" or config.get("architectures") != ["Zonos2ForConditionalGeneration"]:
        raise ValueError("ZONOS2 converted config does not describe the native pipeline")
    entries = {row["key"]: row for row in manifest["tensors"]}
    if len(entries) != 507:
        raise ValueError("ZONOS2 tensor manifest must contain 507 unique entries")
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as source:
        if set(source.keys()) != set(entries):
            raise ValueError("Converted ZONOS2 tensor inventory does not match its manifest")
        for key, entry in entries.items():
            tensor = source.get_tensor(key)
            raw = tensor.contiguous().view(torch.uint8).numpy()
            if list(tensor.shape) != entry["shape"] or str(tensor.dtype).removeprefix("torch.") != entry["dtype"]:
                raise ValueError(f"Converted ZONOS2 tensor shape/dtype mismatch: {key}")
            if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
                raise ValueError(f"Converted ZONOS2 tensor SHA256 mismatch: {key}")


def download_snapshot(repo: str, revision: str, files: dict[str, str], directory: Path, *, offline: bool) -> None:
    missing = [name for name in files if not (directory / name).is_file()]
    if missing:
        if offline:
            raise FileNotFoundError(f"Offline preparation is missing {repo}: {missing}")
        # Downloads are explicit setup work. Inference remains offline.
        env = dict(os.environ, HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0", CUDA_VISIBLE_DEVICES="")
        subprocess.run(
            ["hf", "download", repo, *missing, "--revision", revision, "--local-dir", str(directory)],
            check=True,
            env=env,
        )
    verify_files(directory, files)


def prepare(root: Path, *, offline: bool = False) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    model_override = os.environ.get("ZONOS2_TEST_MODEL_PATH")
    model = Path(model_override) if model_override else root / "converted"
    if not model.exists():
        if model_override:
            raise FileNotFoundError(f"Preprovisioned model path does not exist: {model}")
        raw = root / "original"
        download_snapshot("Zyphra/ZONOS2", MODEL_REVISION, MODEL_HASHES, raw, offline=offline)
        # Hash verification precedes the converter's trusted pickle boundary.
        temporary = Path(tempfile.mkdtemp(prefix="conversion-", dir=root))
        converter = Path(__file__).with_name("convert_zonos2_to_safetensors.py")
        subprocess.run(
            [sys.executable, str(converter), "--input", str(raw), "--output-dir", str(temporary)],
            check=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        )
        verify_model(temporary)
        temporary.rename(model)
    else:
        verify_model(model)
    speaker_override = os.environ.get("VLLM_ZONOS2_SPEAKER_PATH")
    speaker = Path(speaker_override) if speaker_override else root / "speaker"
    if speaker_override:
        verify_files(speaker, SPEAKER_HASHES)
    else:
        download_snapshot(
            "marksverdhei/Qwen3-Voice-Embedding-12Hz-1.7B", SPEAKER_REVISION, SPEAKER_HASHES, speaker, offline=offline
        )
    dac_override = os.environ.get("VLLM_ZONOS2_DAC_PATH")
    dac = Path(dac_override) if dac_override else root / "dac-44khz.pth"
    if not dac.is_file():
        if offline or dac_override:
            raise FileNotFoundError(f"Missing local DAC asset: {dac}")
        with tempfile.TemporaryDirectory(dir=root, prefix="dac-") as folder:
            temporary = Path(folder) / "weights.pth"
            with urllib.request.urlopen(DAC_URL, timeout=120) as response, temporary.open("wb") as target:
                while chunk := response.read(8 << 20):
                    target.write(chunk)
            if file_digest(temporary) != DAC_SHA256:
                raise ValueError(f"Downloaded DAC SHA256 mismatch: {temporary}")
            temporary.rename(dac)
    if file_digest(dac) != DAC_SHA256:
        raise ValueError(f"Pinned DAC SHA256 mismatch: {dac}")
    return {
        "ZONOS2_TEST_MODEL_PATH": str(model.resolve()),
        "VLLM_ZONOS2_DAC_PATH": str(dac.resolve()),
        "VLLM_ZONOS2_SPEAKER_PATH": str(speaker.resolve()),
        "VLLM_ZONOS2_TN_CACHE_DIR": str(root.resolve() / "tn-cache"),
        "ZONOS2_TEST_OUTPUT_DIR": str(root.resolve() / "results"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Verify provisioned assets without network access")
    args = parser.parse_args()
    values = prepare(args.asset_root, offline=args.offline)
    args.env_file.parent.mkdir(parents=True, exist_ok=True)
    args.env_file.write_text("".join(f"export {key}={shlex.quote(value)}\n" for key, value in values.items()))
    manifest = {"model_revision": MODEL_REVISION, "speaker_revision": SPEAKER_REVISION, "paths": values}
    (args.asset_root / "prepared.json").write_text(json.dumps(manifest, indent=2))
    print(f"Verified pinned ZONOS2 assets; environment: {args.env_file}", flush=True)


if __name__ == "__main__":
    main()
