# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Engine runtime error types."""


class NativeKVHandoffError(RuntimeError):
    """One request lost its producer binding or transfer metadata."""
