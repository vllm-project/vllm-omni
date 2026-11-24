# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What the weights a MiniMax-H3 request samples with imply for the pipeline.

H3 is served from many weight sets: the released checkpoint, full students
(FastH3 V2, FastH3 Omni), adapters fused at load (FastH3 Preview) and
request-scoped adapters (LightX2V Turbo, native distilled LoRAs, ordinary
LoRAs). They differ in how a request is *sampled*: the sigma positions, the
step count, the flow shifts, and whether the output heads are parallel-decoding
(PDD) heads. Those contracts stay with the spec that loads each artifact.

They do not differ in how a request is *conditioned*. ControlNet hints,
latent-tail continuation, latent-mask editing and timeline guides are rows and
masks packed around the target latent; none of them reads the schedule. A
feature is therefore available on every variant unless the variant names a
reason, and the pipeline asks the variant rather than testing for one release.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace

from vllm_omni.errors import OmniClientError


class H3Feature(enum.Enum):
    """Conditioning features a request can ask for on top of plain generation."""

    CONTROL = "control"
    CONTINUATION = "continuation"
    LATENT_EDIT = "latent-mask editing"
    TIMELINE_GUIDES = "timeline guides"


@dataclass(frozen=True)
class H3Variant:
    """The weight set behind a request, as far as the pipeline needs to know.

    ``pdd`` selects the parallel-decoding sampler: sigma nodes on the release's
    own grid and one fused output head per step. ``distilled_adapters`` says
    whether a few-step adapter (Turbo, native distilled LoRA) may be activated
    on top; a full student already pins its schedule, so it takes ordinary
    adapters only. ``unsupported`` maps a feature to the message a request for
    it receives.
    """

    name: str
    pdd: bool = False
    distilled_adapters: bool = True
    unsupported: tuple[tuple[H3Feature, str], ...] = field(default_factory=tuple)

    def reason(self, feature: H3Feature) -> str | None:
        """Why ``feature`` is unavailable on these weights, or ``None`` when it is available."""
        return dict(self.unsupported).get(feature)

    def supports(self, feature: H3Feature) -> bool:
        return self.reason(feature) is None

    def require(self, feature: H3Feature) -> None:
        """Reject the request when these weights cannot serve ``feature``."""
        reason = self.reason(feature)
        if reason is not None:
            raise OmniClientError(reason)

    def with_adapter(self, name: str) -> H3Variant:
        """These weights with a request-scoped adapter active on top.

        An adapter changes the sampling contract, which its own spec enforces;
        it adds no conditioning restriction, so only the name moves.
        """
        return replace(self, name=name)


_GUIDES_NEED_REFERENCE_STUDENT = "MiniMax H3 timeline guides require base H3 weights or FastH3 Omni"

BASE_H3 = H3Variant(name="MiniMax H3")
# Ref2VA student sampled on its PDD grid. Its eight forwards are the whole
# schedule, so few-step adapters have nothing to bind to.
FASTH3_OMNI = H3Variant(name="FastH3 Omni", pdd=True, distilled_adapters=False)
# T2VA-only students trained with VSA compression gates. Guide rows change the
# token layout those gates were trained on, and no guided output of either has
# been checked, so guides stay off until one is.
FASTH3_V2 = H3Variant(
    name="FastH3 V2",
    unsupported=((H3Feature.TIMELINE_GUIDES, _GUIDES_NEED_REFERENCE_STUDENT),),
)
FASTH3_PREVIEW = H3Variant(
    name="FastH3 Preview",
    unsupported=((H3Feature.TIMELINE_GUIDES, _GUIDES_NEED_REFERENCE_STUDENT),),
)


def control_task(partition: str) -> str | None:
    """The task a control request must use on a server that serves ``partition``.

    The control branch is bound to the primary DiT. On an FL2VA DiT it hints a
    text-only generation; on a Ref2VA DiT it hints the target rows of a
    reference-conditioned one, and reference rows receive no hint. A combined
    server has two DiTs and one branch, so it serves no control.
    """
    if partition == "ref2va":
        return "ref2va"
    if partition == "fl2va":
        return "t2va"
    return None
