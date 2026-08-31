# SPDX-License-Identifier: Apache-2.0
# minicpm-challenge: profile-skip (Slice B2). Installed from the worker
# runner modules (imported inside engine processes after vllm_ascend is up;
# importing vllm_ascend earlier trips a circular import).
#
# vllm-ascend Worker.profile_memory queries torch.npu.memory_reserved()/
# memory_allocated() at the top of every execute_model step (~4.7% of the
# stage1 frame cadence). Its outputs are write-only (consumed only by two
# DEBUG log lines inside itself), so replace it with a no-op unless
# MINICPMO_PROFILE_SKIP=0.

_INSTALLED = {"done": False}


def install_profile_skip() -> None:
    if _INSTALLED["done"]:
        return
    _INSTALLED["done"] = True
    import logging
    import os

    if os.environ.get("MINICPMO_PROFILE_SKIP", "1") == "0":
        return
    try:
        from vllm_ascend.worker import worker as _wmod
    except Exception:
        _INSTALLED["done"] = False
        return

    cls = None
    for name in dir(_wmod):
        obj = getattr(_wmod, name)
        if (isinstance(obj, type)
                and obj.__module__.startswith("vllm_ascend")
                and "profile_memory" in vars(obj)):
            cls = obj
            break
    if cls is None:
        logging.getLogger("vllm.omni.profile_skip").info(
            "profile-skip: no candidate class; not patching")
        return

    def _noop(self) -> None:
        return None

    cls.profile_memory = _noop
    logging.getLogger("vllm.omni.profile_skip").info(
        "profile-skip installed on %s.%s", cls.__module__, cls.__name__)
