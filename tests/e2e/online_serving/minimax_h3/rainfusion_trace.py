# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Check streamed Ref2VA logs for a sparse RainFusion execution without saving them."""

import sys


class Ref2VASparseTrace:
    def __init__(self) -> None:
        self.request_started = False
        self.multi_video = False
        self.sparse_kernel = False

    def observe(self, line: str) -> None:
        if "MINIMAX_H3_REF2VA_REQUEST_BEGIN" in line:
            self.request_started = True
            self.multi_video = False
            self.sparse_kernel = False
            return
        if not self.request_started:
            return
        if "RAINFUSION_ATTN multi-video active:" in line:
            self.multi_video = True
        if "RainFusion uses MindIE-SD sparse attention" in line:
            self.sparse_kernel = True
        if "RAINFUSION_ATTN hybrid active:" in line and "sparse_q_rows=" in line:
            rows = line.split("sparse_q_rows=", 1)[1].split(",", 1)[0].strip()
            if rows.isdigit() and int(rows) > 0:
                self.sparse_kernel = True

    def assert_sparse_execution(self) -> None:
        assert self.multi_video and self.sparse_kernel, "Ref2VA RainFusion sparse execution was not observed"


def main() -> int:
    trace = Ref2VASparseTrace()
    for line in sys.stdin:
        sys.stdout.write(line)
        sys.stdout.flush()
        trace.observe(line)
    try:
        trace.assert_sparse_execution()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
