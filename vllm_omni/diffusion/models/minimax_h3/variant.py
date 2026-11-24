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
masks packed around the target latent; none of them reads the schedule, so the
pipeline serves them on every weight set without asking the variant. The one
conditioning restriction a weight set carries is why it refuses timeline guides.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class H3Variant:
    """The weight set behind a request, as far as the pipeline needs to know.

    ``pdd`` selects the parallel-decoding sampler: sigma nodes on the release's
    own grid and one fused output head per step. ``distilled_adapters`` says
    whether a few-step adapter (Turbo, native distilled LoRA) may be activated
    on top; a full student already pins its schedule, so it takes ordinary
    adapters only. ``guides_unsupported`` is the message a timeline-guide
    request receives, or ``None`` where guides are served.
    """

    name: str
    pdd: bool = False
    distilled_adapters: bool = True
    guides_unsupported: str | None = None


_GUIDES_NEED_REFERENCE_STUDENT = "MiniMax H3 timeline guides require base H3 weights or FastH3 Omni"

BASE_H3 = H3Variant(name="MiniMax H3")
# Ref2VA student sampled on its PDD grid. Its eight forwards are the whole
# schedule, so few-step adapters have nothing to bind to.
FASTH3_OMNI = H3Variant(name="FastH3 Omni", pdd=True, distilled_adapters=False)
# T2VA-only students trained with VSA compression gates. Guide rows change the
# token layout those gates were trained on, and no guided output of either has
# been checked, so guides stay off until one is.
FASTH3_V2 = H3Variant(name="FastH3 V2", guides_unsupported=_GUIDES_NEED_REFERENCE_STUDENT)
FASTH3_PREVIEW = H3Variant(name="FastH3 Preview", guides_unsupported=_GUIDES_NEED_REFERENCE_STUDENT)


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
