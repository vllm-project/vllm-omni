# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test worker extension that mirrors
``verl_omni.workers.rollout.vllm_rollout.utils.vLLMOmniColocateWorkerExtension``

Fidelity goals:

* ``__new__`` calls ``set_death_signal()`` (Linux ``PR_SET_PDEATHSIG``) so
  spawned vLLM workers die with the parent.
* ``__new__`` calls ``VLLMOmniHijackForTest.hijack()`` which monkey-patches
  ``vllm_omni.diffusion.lora.manager.DiffusionLoRAManager._load_adapter``
  to accept in-memory LoRA tensors (verbatim port of
  ``verl_omni.utils.vllm_omni.utils.VLLMOmniHijack.hijack``).

The ``verl.utils.vllm.VLLMHijack.hijack()`` call (which patches vLLM's
``LRUCacheWorkerLoRAManager._load_adapter`` for AR/text mode) is
intentionally skipped because (a) ``verl`` is not importable from this
repo and (b) the diffusion test path never reaches that manager.
"""

from __future__ import annotations

import ctypes
import logging
import os
import platform
import signal

from msgspec import field

from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension
from vllm_omni.lora.request import LoRARequest as OmniLoRARequest

logger = logging.getLogger(__name__)


class OmniTensorLoRARequestForTest(OmniLoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


def set_death_signal() -> None:
    """Verbatim port of ``verl.workers.rollout.vllm_rollout.utils.set_death_signal``."""
    if platform.system() != "Linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(1, signal.SIGKILL)
        if os.getppid() == 1:
            os.kill(os.getpid(), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        # libc.so.6 may not be present (musl, alpine) — best-effort only.
        pass


# ---------------------------------------------------------------------
#  The worker extension itself
# ---------------------------------------------------------------------


class vLLMOmniColocateWorkerExtensionForTest(CustomPipelineWorkerExtension):
    """Mirror of ``vLLMOmniColocateWorkerExtension`` (verl-omni).

    The production ``__new__`` runs ``set_death_signal`` + ``VLLMOmniHijack.hijack``
    on every vLLM worker process. Replicating both ensures this test
    reproduces the same monkey-patched environment.
    """

    def __new__(cls, **kwargs):
        set_death_signal()
        return super().__new__(cls)

    @staticmethod
    def test_extension_name() -> str:
        """Return a stable identifier for assertions in unit tests."""
        return "vllm-omni-colocate-worker-extension-for-test"
