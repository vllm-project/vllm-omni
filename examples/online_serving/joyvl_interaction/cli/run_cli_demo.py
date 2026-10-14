# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

from stream_client import stream

ICON = {"response": "SPEAK   ", "silence": "silence ", "delegate": "DELEGATE"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--server", default="http://127.0.0.1:8070")
    parser.add_argument("--query", default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--session-id", default="cli")
    args = parser.parse_args()

    for tick in stream(args.video, args.server, args.session_id, args.query, args.fps):
        if tick.action == "silence":
            continue
        line = f"[{tick.t:6.1f}s] {ICON.get(tick.action)} {tick.text}"
        if tick.delegation:
            line += f"  (-> {tick.delegation.get('question', '')})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
