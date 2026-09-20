# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Gander Unit8 dialogue grammar and release composition.

The public release contains a complete Thinker and a separate complete Talker.
Compose an HF directory without copying tensors before using the MiniCPM pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONTROL_TOKENS = {
    "interrupt_token_id": "<|interrupt|>",
    "backchannel_token_id": "<|backchannel|>",
    "tool_call_token_id": "<tool_call>",
    "tool_call_end_token_id": "</tool_call>",
    "tool_response_token_id": "<tool_response>",
    "tool_response_end_token_id": "</tool_response>",
}


def control_token_ids(tokenizer) -> dict[str, int]:
    result = {}
    for key, token in CONTROL_TOKENS.items():
        value = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(value, int) or value < 0 or value == tokenizer.unk_token_id:
            raise ValueError(f"Gander tokenizer is missing control token: {token}")
        result[key] = value
    return result


def dialogue_constraint(tokens: list[int], ids: dict[str, int]) -> tuple[bool, set[int]]:
    """Return (allowlist, ids), matching upstream's tool-free dialogue grammar."""
    actions = {ids[k] for k in ("listen_token_id", "speak_token_id", "interrupt_token_id", "backchannel_token_id")}
    protocol = {ids[k] for k in CONTROL_TOKENS}
    if not tokens:
        return True, actions
    if ids["turn_eos_token_id"] in tokens:
        return True, {ids["chunk_eos_token_id"]}
    if len(tokens) - 1 >= 8:
        return True, {ids["chunk_eos_token_id"], ids["turn_eos_token_id"]}
    return False, actions | protocol


def compose_release(source: Path, destination: Path) -> Path:
    """Make a local, symlink-backed model; never modify the downloaded snapshot."""
    from safetensors import safe_open

    source = source.resolve()
    thinker, talker = source / "thinker", source / "talker"
    manifest = json.loads((source / "release_manifest.json").read_text())
    contract = manifest["unit_contract"]
    if contract["text_tokens_per_speak_unit"] != 8 or contract["speech_tokens_per_speak_unit"] != 50:
        raise ValueError("Only the Gander Unit8/50 release is supported")
    config = json.loads((thinker / "config.json").read_text())
    talker_config = json.loads((talker / "talker_config.json").read_text())
    if not talker_config.get("weights_are_complete"):
        raise ValueError("Gander requires the materialized Talker release")
    index = json.loads((thinker / "model.safetensors.index.json").read_text())
    weights = dict(index["weight_map"])
    for filename in set(weights.values()):
        if not (thinker / filename).is_file():
            raise FileNotFoundError(thinker / filename)
    with safe_open(talker / "model.safetensors", framework="pt") as handle:
        for key in handle.keys():
            if not key.startswith("tts.") or key in weights:
                raise ValueError(f"Unexpected or duplicate Talker tensor: {key}")
            weights[key] = "gander-talker.safetensors"
    config.update(init_tts=True, gander_unit8=True, tts_config=talker_config["tts_config"])
    config["tts_config"]["gander_unit8"] = True
    destination.mkdir(parents=True, exist_ok=False)
    for file in thinker.iterdir():
        if file.name not in {"config.json", "model.safetensors.index.json"}:
            (destination / file.name).symlink_to(file)
    (destination / "gander-talker.safetensors").symlink_to(talker / "model.safetensors")
    (destination / "assets").mkdir()
    for asset in (talker / "assets").iterdir():
        (destination / "assets" / asset.name).symlink_to(asset)
    (destination / "assets" / "HT_ref_audio.wav").symlink_to(talker / "assets" / "ref_audio.wav")
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    index["weight_map"] = weights
    index["metadata"] = {"total_size": sum(c["parameter_bytes"] for c in manifest["components"].values())}
    (destination / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Downloaded Gander-Omni/Gander snapshot")
    parser.add_argument("destination", type=Path, help="New composed HF model directory")
    args = parser.parse_args()
    print(compose_release(args.source, args.destination))
