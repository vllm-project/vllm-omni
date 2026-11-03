# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Backward-compatible MiniCPM entry point for the shared realtime UI."""

from examples.online_serving.realtime_web.server import (
    APP_DIR,
    STATIC_DIR,
    _join_ws_url,
    build_app,
)
from examples.online_serving.realtime_web.server import (
    main as _main,
)

__all__ = ["APP_DIR", "STATIC_DIR", "_join_ws_url", "build_app", "main"]


def main() -> None:
    _main(default_profile="minicpm-native")


if __name__ == "__main__":
    main()
