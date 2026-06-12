"""Required one-shot setup: produce an Alpamayo-extended tokenizer dir.

Alpamayo-1.5-10B ships NO tokenizer files. The production setup borrows the
Qwen3-VL-8B-Instruct tokenizer and adds 4000 discrete trajectory tokens
(`<i0>..<i3999>`) + ~28 special tokens (`<|traj_history|>`,
`<|traj_future_start|>`, `<|cot_start|>`, ...).

Doing this at processor-construction time only patches the multimodal-
processor tokenizer instance — the SEPARATE tokenizer that the chat-
completion endpoint uses for rendering chat templates remains unaware of
these tokens, so any `<|traj_history|>` literal in a chat prompt gets
tokenized as a bag of ordinary subword tokens rather than the intended
single id, and server-side history fusion finds 0 placeholders.

A runtime hook that walked into vllm's TokenizerPool closure to refresh
the deepcopies was prototyped (commit 4c9c3694 / reverted) but rejected
as too fragile across vllm versions. The disk-baked approach below is
the supported path.

Run this script ONCE per host. Then point `--tokenizer` at the output.

Usage::

    python3 scripts/build_alpamayo_tokenizer.py \\
        --src /data/models/Qwen3-VL-8B-Instruct \\
        --dst /tmp/alpamayo-tokenizer

Then::

    vllm-omni serve /data/models/Alpamayo-1.5-10B --omni \\
        --tokenizer /tmp/alpamayo-tokenizer \\
        ...
"""

import argparse
import os
import shutil
import sys

from transformers import AutoProcessor

# Use add_alpamayo_tokens from the model module so the token order matches
# the released config.traj_token_ids exactly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm_omni.model_executor.models.alpamayo.processing import (  # noqa: E402
    add_alpamayo_tokens,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/data/models/Qwen3-VL-8B-Instruct", help="Base tokenizer/processor dir.")
    p.add_argument("--dst", required=True, help="Output dir (will be overwritten).")
    p.add_argument("--traj-vocab-size", type=int, default=4000)
    args = p.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)
    os.makedirs(args.dst, exist_ok=True)

    # Load processor (tokenizer + image processor) from the base.
    processor = AutoProcessor.from_pretrained(args.src, trust_remote_code=True)
    tokenizer = processor.tokenizer

    info = add_alpamayo_tokens(tokenizer, traj_vocab_size=args.traj_vocab_size)
    print(f"added Alpamayo tokens; vocab is now {len(tokenizer)}")
    print(f"  future_start id: {info['traj_token_ids'].get('future_start')}")
    print(f"  history    id:   {info['traj_token_ids'].get('history')}")
    print(f"  traj_token_start_idx: {info['traj_token_start_idx']}")

    # Save processor (includes both image processor JSON and the EXTENDED
    # tokenizer files — added_tokens.json etc. get serialized).
    processor.save_pretrained(args.dst)
    print(f"\nwrote extended tokenizer/processor to {args.dst}")
    print("now relaunch the server with --tokenizer", args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
