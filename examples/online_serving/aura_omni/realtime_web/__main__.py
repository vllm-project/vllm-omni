# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

from examples.online_serving.realtime_web.server import main
from examples.online_serving.realtime_web.tools import RealtimeToolExecutor, VisionToolExecutor

if __name__ == "__main__":
    # AURA_DEMO_TOOLS: "vision" (default), "safe" for the allowlisted set, or 0 for no tools.
    tool_set = os.environ.get("AURA_DEMO_TOOLS", "vision").strip().lower()
    if tool_set in {"0", "false", "off", "no"}:
        tool_executor = None
    elif tool_set == "safe":
        tool_executor = RealtimeToolExecutor()
    else:
        tool_executor = VisionToolExecutor()
    main(default_profile="aura-ptt", tool_executor=tool_executor)
