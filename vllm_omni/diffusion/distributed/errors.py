# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


class DistributedExecutionError(BaseException):
    """A rank failed while peers may still be executing device communication.

    Bypass request-level ``except Exception`` handlers: the worker must exit
    and let its executor terminate the group, rather than reuse it or attempt
    distributed cleanup that may itself block on a peer.
    """
