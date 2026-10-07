# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Typed runner contracts for models that plug into the omni worker loops.

A model that a runner drives through a non-standard step (multi-frame decode
in the Talker's case) has to answer a handful of attributes and methods. The
runners used to read them with bare ``getattr(model, "x", False)``, which
turns a missing forward into a silent False: the loop never engages, nothing
is logged, and on the NPU the failure surfaces much later as an acl 507035
vector-core fault or an engine that dies with empty audio. The helpers below
make the miss loud in the unit tests instead, and the Protocol gives the
supported surface one name.
"""

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "SupportsMultiFrameDecode",
    "requires_request_sample_eligibility",
    "supports_multi_frame_decode",
]


@runtime_checkable
class SupportsMultiFrameDecode(Protocol):
    """The runner contract for Talker-style multi-frame (K-step) decode.

    ``batch_stop_logits`` is public because the runners read it across module
    boundaries: it is the per-request stop-row holder the collapsed head
    fills, and a private name on the model made every reader carry its own
    ``getattr`` guard.
    """

    supports_multi_frame_decode: bool

    @property
    def batch_stop_logits(self) -> Any: ...

    def take_batch_stop_logits(self) -> Any: ...

    def set_batch_stop_logits(self, logits: Any) -> None: ...

    def merge_frame_outputs(self, frame_outputs: list[Any], frame_stop_logits: list[Any]) -> Any: ...


def _missing_members(model: Any, names: tuple[str, ...]) -> list[str]:
    missing = [name for name in names if not hasattr(model, name)]
    # Protocol members declared as annotations are not real attributes, so a
    # data-only member is satisfied by the flag itself; the methods below are
    # the ones a missing forward would actually hide.
    return [name for name in missing if name != "supports_multi_frame_decode"]


def supports_multi_frame_decode(model: Any) -> bool:
    """Whether ``model`` runs the multi-frame loop, verified when it says yes.

    Returns False when the flag is off or absent (that is the common case:
    Thinker stages and every model without the loop). When the flag is on but
    a member of the contract is missing, raises TypeError naming what is
    absent -- a dropped forward fails in CI rather than as empty audio on a
    device.
    """
    flag = getattr(model, "supports_multi_frame_decode", False)
    if not flag:
        return False
    missing = _missing_members(model, SupportsMultiFrameDecode.__protocol_attrs__)
    if missing:
        raise TypeError(
            f"{type(model).__name__} advertises supports_multi_frame_decode but does not provide "
            f"{sorted(missing)}; the multi-frame runner would silently read defaults"
        )
    return True


def requires_request_sample_eligibility(model: Any) -> bool:
    """Whether ``model`` wants the runner's per-request sample-eligibility rows.

    Off/absent means the runner does not build the rows at all, which is the
    upstream behaviour; the flag is read plainly (no cross-member contract to
    verify).
    """
    return bool(getattr(model, "requires_request_sample_eligibility", False))
