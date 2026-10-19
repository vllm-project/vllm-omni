# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""OpenAI Live API (``/v1/live/sessions``) for vLLM-Omni.

One ``LiveSessionHandler`` per WebSocket drives one resumable request per
session through ``AsyncOmni``. Only native full-duplex models are served: the
model decides itself when to listen and speak. See
``docs/serving/live_sessions_api.md``.
"""
