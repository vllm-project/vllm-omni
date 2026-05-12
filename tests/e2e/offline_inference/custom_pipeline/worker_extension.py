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

try:
    from vllm.lora.lora_model import LoRAModel
except ImportError:  # vLLM versions before the rename keep it under .models
    from vllm.lora.models import LoRAModel  # type: ignore[no-redef]

from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import get_adapter_absolute_path
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension
from vllm_omni.lora.request import LoRARequest as OmniLoRARequest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
#  In-memory LoRA request (mirror of verl_omni's OmniTensorLoRARequest)
# ---------------------------------------------------------------------


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
#  VLLMOmniHijackForTest — verbatim port of
#  verl_omni.utils.vllm_omni.utils.VLLMOmniHijack (minus verl's VLLMHijack
#  cascade, which targets the AR/text LRU LoRA manager not exercised here).
# ---------------------------------------------------------------------


class VLLMOmniHijackForTest:
    """Monkey-patches vllm-omni internals to support in-memory LoRA tensors."""

    _applied: bool = False

    @staticmethod
    def hijack() -> None:
        if VLLMOmniHijackForTest._applied:
            return

        def hijack__load_adapter(
            self, lora_request: OmniTensorLoRARequestForTest
        ) -> tuple[LoRAModel, PEFTHelper]:
            if not self._expected_lora_modules:
                raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

            logger.debug("Supported LoRA modules: %s", self._expected_lora_modules)

            lora_tensors = None
            if isinstance(lora_request, OmniTensorLoRARequestForTest):
                peft_config = lora_request.peft_config
                lora_tensors = lora_request.lora_tensors
                peft_helper = PEFTHelper.from_dict(peft_config)
            else:
                lora_path = get_adapter_absolute_path(lora_request.lora_path)
                logger.debug("Resolved LoRA path: %s", lora_path)
                peft_helper = PEFTHelper.from_local_dir(
                    lora_path,
                    max_position_embeddings=None,
                    tensorizer_config_dict=lora_request.tensorizer_config_dict,
                )

            if isinstance(lora_request, OmniTensorLoRARequestForTest):
                lora_model = LoRAModel.from_lora_tensors(
                    tensors=lora_tensors,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",
                    dtype=self.dtype,
                    model_vocab_size=None,
                    weights_mapper=None,
                )
            else:
                lora_model = LoRAModel.from_local_checkpoint(
                    lora_path,
                    expected_lora_modules=self._expected_lora_modules,
                    peft_helper=peft_helper,
                    lora_model_id=lora_request.lora_int_id,
                    device="cpu",
                    dtype=self.dtype,
                    model_vocab_size=None,
                    tensorizer_config_dict=lora_request.tensorizer_config_dict,
                    weights_mapper=None,
                )

            for lora in lora_model.loras.values():
                lora.optimize()

            return lora_model, peft_helper

        DiffusionLoRAManager._load_adapter = hijack__load_adapter
        VLLMOmniHijackForTest._applied = True


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
        VLLMOmniHijackForTest.hijack()
        return super().__new__(cls)

    @staticmethod
    def test_extension_name() -> str:
        """Return a stable identifier for assertions in unit tests."""
        return "vllm-omni-colocate-worker-extension-for-test"