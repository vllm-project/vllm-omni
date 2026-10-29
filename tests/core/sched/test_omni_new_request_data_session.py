# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session-id carriage in OmniNewRequestData conversions.

Runnable without a vllm install: when vllm is absent, stub modules stand
in for vllm.v1.core.sched.output / vllm.v1.request and for vllm_omni.engine
(output.py only needs AdditionalInformationPayload from it), mirroring the
namespace-parent shim used by the prefix-cache tests. The stub
NewRequestData carries a session_id field (the upstream-likely shape);
from_base pops it before passing explicitly, so the no-field shape is
safe by construction (pop default None).
"""

import logging
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import ModuleType

import pytest

try:  # pragma: no cover - real vllm present on CI/GPU machines
    import vllm  # noqa: F401
    from vllm.v1.core.sched.output import NewRequestData  # noqa: F401

    _HAS_VLLM = True
except ModuleNotFoundError:  # pragma: no cover - vllm-less dev machines
    _HAS_VLLM = False
    # One level deeper than the prefix-cache tests (tests/core/sched),
    # so the repo root is parents[3], not parents[2].
    _root = Path(__file__).resolve().parents[3]
    for _pkg in ("vllm_omni", "vllm_omni.core", "vllm_omni.core.sched"):
        if _pkg not in sys.modules:
            _m = ModuleType(_pkg)
            _m.__path__ = [str(_root / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = _m
    _engine = ModuleType("vllm_omni.engine")

    class AdditionalInformationPayload:  # noqa: D101 - stub only
        pass

    _engine.AdditionalInformationPayload = AdditionalInformationPayload
    sys.modules["vllm_omni.engine"] = _engine

    @dataclass
    class NewRequestData:  # noqa: D101 - stub only
        req_id: str = ""
        prompt_token_ids: list = field(default_factory=list)
        mm_features: object = None
        sampling_params: object = None
        pooling_params: object = None
        block_ids: object = ()
        num_computed_tokens: int = 0
        lora_request: object = None
        prompt_embeds: object = None
        prompt_is_token_ids: object = None
        prefill_token_ids: object = None
        session_id: str | None = None

    @dataclass
    class CachedRequestData:  # noqa: D101 - stub only
        pass

    @dataclass
    class SchedulerOutput:  # noqa: D101 - stub only
        pass

    def _mod(name: str) -> ModuleType:
        mod = ModuleType(name)
        sys.modules[name] = mod
        return mod

    _vllm = _mod("vllm")
    _vllm_logger = _mod("vllm.logger")
    _vllm_logger.init_logger = logging.getLogger
    _vllm.logger = _vllm_logger
    _mod("vllm.v1")
    _mod("vllm.v1.core")
    _mod("vllm.v1.core.sched")
    _sched_out = _mod("vllm.v1.core.sched.output")
    _sched_out.NewRequestData = NewRequestData
    _sched_out.CachedRequestData = CachedRequestData
    _sched_out.SchedulerOutput = SchedulerOutput
    _req_mod = _mod("vllm.v1.request")

    @dataclass
    class Request:  # noqa: D101 - stub only
        request_id: str = ""

    _req_mod.Request = Request

from vllm_omni.core.sched.output import OmniNewRequestData

if not _HAS_VLLM:
    # Withdraw the stubs again: the module under test keeps its own
    # references to the stub classes, but later importorskip checks
    # (e.g. test_real_vllm_scheduler_contract) must still see vllm as
    # absent instead of being fooled by the stubs. The namespace parents
    # and the imported output module stay cached.
    for _name in (
        "vllm_omni.engine",
        "vllm",
        "vllm.logger",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.output",
        "vllm.v1.request",
    ):
        sys.modules.pop(_name, None)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _Req:
    request_id: str = "r"
    prompt_token_ids: list = field(default_factory=list)
    mm_features: object = None
    sampling_params: object = None
    pooling_params: object = None
    num_computed_tokens: int = 0
    lora_request: object = None
    session_id: str | None = None


def test_from_request_carries_session_id():
    data = OmniNewRequestData.from_request(_Req(session_id="S"), ([0],))
    assert data.session_id == "S"


def test_from_request_empty_session_is_none():
    data = OmniNewRequestData.from_request(_Req(session_id=""), ([0],))
    assert data.session_id is None


_NO_PAYLOAD_SESSION = pytest.mark.skipif(
    "session_id" not in {f.name for f in fields(NewRequestData)},
    reason="upstream NewRequestData has no session field to carry",
)


@_NO_PAYLOAD_SESSION
def test_from_base_prefers_request_session_over_payload_session():
    base = NewRequestData(req_id="r", session_id="OLD")
    data = OmniNewRequestData.from_base(base, _Req(session_id="NEW"))
    assert data.session_id == "NEW"


@_NO_PAYLOAD_SESSION
def test_from_base_falls_back_to_payload_session():
    base = NewRequestData(req_id="r", session_id="OLD")
    data = OmniNewRequestData.from_base(base, _Req(session_id=None))
    assert data.session_id == "OLD"
    data = OmniNewRequestData.from_base(base, None)
    assert data.session_id == "OLD"
