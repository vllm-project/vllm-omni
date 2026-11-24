# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o 4.5 vocoder assets, batching and graph execution.

Keep imports lazy: model registration and optional backends load their own
components without importing the entire vocoder stack.
"""
