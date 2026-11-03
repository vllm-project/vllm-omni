#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Assemble an AuK checkpoint directory that ``--model <path>`` can load.

The released AuK repos ship ``config.yaml`` plus loose safetensors and depend
on a separate Qwen2.5-Omni-3B snapshot for the frozen text encoder. This tool
writes the ``config.json`` the vLLM-Omni pipeline expects and links every
artifact both stages need into one flat directory:

    <out>/
      config.json          the Qwen2.5-Omni-3B config with model_type "auk",
                           AuK's architecture, and the dit / vae / variant /
                           flash_t_grid / defaults sections added at the top
                           level (see transformers_utils/configs/auk.py)
      auk.safetensors      -> auk_base.safetensors or auk_flash.safetensors
      vae.safetensors      -> the AuK VAE weights
      thinker/             -> the Qwen2.5-Omni-3B snapshot, for reference only
      model-0000*.safetensors, model.safetensors.index.json,
      tokenizer*, vocab.json, merges.txt, added_tokens.json,
      special_tokens_map.json, chat_template.json,
      preprocessor_config.json, generation_config.json, spk_dict.pt
                           -> the same snapshot's entries

The Qwen entries are linked at the top level rather than left under
``thinker/`` because both stages load from this one directory: the encoder
reads the thinker shards, the tokenizer and the audio processor from here,
and takes the layer-fusion tensors out of ``auk.safetensors``. The weight
index keeps ``auk.safetensors`` and ``vae.safetensors`` out of the thinker's
own weight load, and the DiT and VAE weights are read by model code directly.

Usage:
    python tools/prepare_auk_checkpoint.py \
        --auk-dir  /path/to/AuK \
        --qwen-dir /path/to/Qwen2.5-Omni-3B \
        --out      /path/to/auk-omni-base

Re-running against an existing directory is safe: config.json is rewritten
and the links are repointed. Pass ``--copy`` on filesystems without symlinks.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

# The distilled student's 4-step grid. Duplicated from
# vllm_omni/transformers_utils/configs/auk.py so this tool stays runnable
# without importing the package (and its torch dependency).
FLASH_T_GRID = [
    0.0,
    0.07612049579620361,
    0.2928932309150696,
    0.6173166036605835,
    1.0,
]

# Names the assembled directory owns; everything else in it is a Qwen entry.
WEIGHTS_LINK = "auk.safetensors"
VAE_LINK = "vae.safetensors"
THINKER_LINK = "thinker"
CONFIG_NAME = "config.json"

# Snapshot entries that are not reused: the config is rewritten, and the repo
# paperwork would misdescribe the assembled directory.
_SKIP_QWEN_ENTRIES = frozenset({CONFIG_NAME, "README.md", "LICENSE"})

DIT_KEYS = ("dim", "heads", "ff_mult", "text_hidden_dim", "num_layers", "num_single_layers")
VAE_KEYS = ("latent_dim", "downsample_rate", "target_sample_rate", "model_init_kwargs")

_INTERPOLATION = re.compile(r"^\$\{([A-Za-z0-9_.]+)\}$")


def _strip_comment(line: str) -> str:
    """Drop a trailing ``#`` comment that is not inside a quoted scalar."""
    out: list[str] = []
    quote: str | None = None
    for ch in line:
        if quote is not None:
            if ch == quote:
                quote = None
            out.append(ch)
            continue
        if ch in "\"'":
            quote = ch
            out.append(ch)
            continue
        if ch == "#":
            break
        out.append(ch)
    return "".join(out).rstrip()


def _parse_scalar(text: str) -> Any:
    """Parse one YAML scalar or inline sequence."""
    text = text.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("null", "~"):
        return None
    if text[0] in "[{":
        normalized = re.sub(r"\b(True|true|yes)\b", "True", text)
        normalized = re.sub(r"\b(False|false|no)\b", "False", normalized)
        normalized = re.sub(r"\b(null|~)\b", "None", normalized)
        try:
            return ast.literal_eval(normalized)
        except (SyntaxError, ValueError):
            return text
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text.strip("\"'")


def _parse_yaml_minimal(text: str) -> dict[str, Any]:
    """Parse the small subset of YAML the AuK configs use.

    Nested mappings by indentation, scalars, and inline sequences. Used only
    when PyYAML is unavailable.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if ":" not in body:
            raise ValueError(f"unsupported YAML line (no key): {raw!r}")
        key, _, rest = body.partition(":")
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"unsupported YAML indentation at: {raw!r}")
        parent = stack[-1][1]
        if rest.strip():
            parent[key] = _parse_scalar(rest)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _resolve_interpolations(node: Any, root: dict[str, Any]) -> Any:
    """Replace ``${a.b.c}`` scalars with the value at that dotted path."""
    if isinstance(node, dict):
        return {k: _resolve_interpolations(v, root) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_interpolations(v, root) for v in node]
    if isinstance(node, str):
        match = _INTERPOLATION.match(node.strip())
        if match is None:
            return node
        target: Any = root
        for part in match.group(1).split("."):
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"cannot resolve interpolation ${{{match.group(1)}}}")
            target = target[part]
        return target
    return node


def load_auk_yaml(path: Path) -> dict[str, Any]:
    """Load the AuK ``config.yaml`` with interpolations resolved."""
    text = path.read_text()
    try:
        import yaml

        parsed = yaml.safe_load(text)
    except ImportError:
        parsed = _parse_yaml_minimal(text)
    if not isinstance(parsed, dict) or "model" not in parsed:
        raise ValueError(f"{path} has no top-level 'model' section")
    return _resolve_interpolations(parsed, parsed)


def detect_variant(auk_dir: Path, model_section: dict[str, Any]) -> str:
    """Infer base vs flash from the weight filename, then the model name."""
    if (auk_dir / "auk_flash.safetensors").exists():
        return "flash"
    if (auk_dir / "auk_base.safetensors").exists():
        return "base"
    name = str(model_section.get("name", "")).lower()
    return "flash" if "flash" in name else "base"


def weights_name(variant: str) -> str:
    return "auk_flash.safetensors" if variant == "flash" else "auk_base.safetensors"


def build_config(
    auk_yaml: dict[str, Any],
    qwen_config: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Assemble the ``config.json`` body for the output directory.

    The Qwen config is kept whole, including ``talker_config`` and
    ``token2wav_config``: ``AuKConfig`` subclasses ``Qwen2_5OmniConfig`` and
    the encoder stage reuses the in-tree thinker processor, which asserts that
    type. AuK's own sections are added at the top level.
    """
    model = auk_yaml["model"]
    arch = model.get("arch", {})
    vae = model.get("vae", {})

    missing = [key for key in DIT_KEYS if key not in arch]
    if missing:
        raise ValueError(f"AuK config.yaml arch section is missing {missing}")
    missing = [key for key in VAE_KEYS if key not in vae]
    if missing:
        raise ValueError(f"AuK config.yaml vae section is missing {missing}")
    if not isinstance(qwen_config.get("thinker_config"), dict):
        raise ValueError("Qwen snapshot config.json has no 'thinker_config' mapping")

    dit = {key: arch[key] for key in DIT_KEYS}
    # dim_head is implicit upstream (dim // heads) but the port takes it
    # explicitly, so it is recorded rather than recomputed at load time.
    dit["dim_head"] = int(arch["dim"]) // int(arch["heads"])
    # Carried from the released config: the DiT builds an attention mask over
    # the padded [audio | text] sequence and parity depends on it.
    dit["attn_mask_enabled"] = bool(arch.get("attn_mask_enabled", True))

    config = dict(qwen_config)
    config["model_type"] = "auk"
    config["architectures"] = ["AuKForConditionalGeneration"]
    config["dit"] = dit
    config["vae"] = {key: vae[key] for key in VAE_KEYS}
    config["variant"] = variant
    config["flash_t_grid"] = FLASH_T_GRID
    config["defaults"] = {"nfe": 32, "cfg": 2.0, "sway": -1.0}
    return config


def _replace_path(target: Path) -> None:
    """Remove an existing file, directory, or link at ``target``."""
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)


def link_or_copy(source: Path, target: Path, *, copy: bool) -> None:
    """Point ``target`` at ``source``, replacing whatever is already there."""
    if not source.exists():
        raise FileNotFoundError(f"missing source artifact: {source}")
    if target.is_symlink() and not copy and target.readlink() == source.resolve():
        return
    _replace_path(target)
    if copy:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


MANIFEST_NAME = ".auk-assembled.json"


def _reject_overlap(out: Path, *inputs: Path) -> None:
    """Refuse an output directory that is, contains, or lives inside an input."""
    out_r = out.resolve()
    for src in inputs:
        src_r = src.resolve()
        if out_r == src_r or out_r in src_r.parents or src_r in out_r.parents:
            raise ValueError(f"--out {out} overlaps input directory {src}; choose a separate directory")


def assemble(auk_dir: Path, qwen_dir: Path, out: Path, variant: str, *, copy: bool) -> None:
    """Link the AuK weights and the Qwen snapshot entries into ``out``."""
    _reject_overlap(out, auk_dir, qwen_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / MANIFEST_NAME
    previous: set[str] = set()
    if manifest_path.is_file():
        previous = set(json.loads(manifest_path.read_text()).get("entries", []))
    weights = auk_dir / weights_name(variant)

    entries: dict[str, Path] = {
        WEIGHTS_LINK: weights,
        VAE_LINK: auk_dir / VAE_LINK,
        THINKER_LINK: qwen_dir,
    }
    # The assembled config replaces the Qwen one; every other entry is reused.
    for entry in qwen_dir.iterdir():
        if entry.name.startswith(".") or entry.name in _SKIP_QWEN_ENTRIES:
            continue
        entries[entry.name] = entry

    for name, source in entries.items():
        link_or_copy(source, out / name, copy=copy)

    # Drop only entries this tool created in an earlier run and no longer needs;
    # anything else in the directory belongs to the user.
    for name in previous - set(entries) - {CONFIG_NAME}:
        stale = out / name
        if stale.exists() or stale.is_symlink():
            _replace_path(stale)
    manifest_path.write_text(json.dumps({"entries": sorted(entries)}, indent=1) + "\n")


def print_tree(out: Path) -> None:
    """Print the assembled directory, one entry per line."""
    print(f"\n{out}/")
    for entry in sorted(out.iterdir()):
        if entry.is_symlink():
            print(f"  {entry.name} -> {entry.readlink()}")
        elif entry.is_dir():
            print(f"  {entry.name}/")
        else:
            print(f"  {entry.name}  ({entry.stat().st_size / 1e9:.2f} GB)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--auk-dir", required=True, help="HF snapshot of the AuK or AuK-Flash repo")
    parser.add_argument("--qwen-dir", required=True, help="HF snapshot of Qwen2.5-Omni-3B")
    parser.add_argument("--out", required=True, help="Directory to assemble (created if absent)")
    parser.add_argument(
        "--variant",
        choices=("base", "flash"),
        help="Override the variant; detected from the AuK snapshot by default",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy the weights and the snapshot entries instead of symlinking",
    )
    args = parser.parse_args()

    auk_dir = Path(args.auk_dir).expanduser().resolve()
    qwen_dir = Path(args.qwen_dir).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    _reject_overlap(out, auk_dir, qwen_dir)

    auk_yaml = load_auk_yaml(auk_dir / "config.yaml")
    variant = args.variant or detect_variant(auk_dir, auk_yaml["model"])
    qwen_config = json.loads((qwen_dir / CONFIG_NAME).read_text())

    config = build_config(auk_yaml, qwen_config, variant)
    assemble(auk_dir, qwen_dir, out, variant, copy=args.copy)
    (out / CONFIG_NAME).write_text(json.dumps(config, indent=2) + "\n")

    print(f"variant: {variant}  (weights {weights_name(variant)})")
    print_tree(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
