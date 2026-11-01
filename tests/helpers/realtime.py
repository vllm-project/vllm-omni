# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared scenario fixtures for Realtime API tool-calling tests."""

from typing import Any

WEATHER_TOOL_PROMPT = "What is the weather in Paris right now? Use the get_weather tool."
WEATHER_TOOL_OUTPUT = "Sunny, 22C."
WEATHER_TOOL_ANSWER = "22"
WEATHER_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "name": "get_weather",
    "description": "Return the current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
}
