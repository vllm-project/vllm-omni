# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kandinsky 6 cache adapters.

Generic MagCache / TeaCache hooks assume a single residual stream named
``hidden_states``. K6's fused visual blocks return ``(video, audio)``, so
the denoise loop consults :class:`K6StepCache` instead and reuses the
previous velocity when the schedule says the step can be skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

# Pro geometry ratios from k6_video
# ``k6_pro_125_480_864_mCache_mOffload.yaml`` (cache.magcache.mag_ratios).
K6_PRO_MAG_RATIOS: tuple[float, ...] = (
    1.0,
    1.0,
    0.94526286,
    0.94622554,
    1.13130749,
    1.11243134,
    1.02926022,
    1.02675114,
    1.011996,
    1.01235353,
    1.03039566,
    1.02848639,
    1.00101278,
    1.00307236,
    1.02238594,
    1.02108294,
    1.0248162,
    1.02491556,
    0.99771836,
    0.99882495,
    1.01694329,
    1.01367628,
    1.01096435,
    1.00959183,
    1.01537554,
    1.01504827,
    0.99237553,
    0.99345238,
    1.02368983,
    1.02088727,
    1.01396087,
    1.01278108,
    0.99773024,
    0.99927632,
    1.01573298,
    1.01489232,
    1.00129728,
    1.00120319,
    1.01759963,
    1.01691798,
    0.99335877,
    0.99548372,
    1.02925427,
    1.02299559,
    0.99749375,
    0.99682709,
    0.99694342,
    1.00064403,
    1.01101129,
    1.0117984,
    1.04420642,
    1.03624643,
    0.9599572,
    0.97176547,
    1.00452036,
    1.0046107,
    1.01447715,
    1.01351055,
    0.9999695,
    1.00270806,
    1.00770242,
    1.00271775,
    0.98795461,
    0.98880837,
    1.01113352,
    1.00860993,
    1.00116034,
    1.00251894,
    1.05147006,
    1.04040677,
    0.99651072,
    0.99212138,
    0.99040673,
    0.98881456,
    1.01265486,
    1.01241698,
    0.97343865,
    0.97302804,
    1.02539269,
    1.02113221,
    1.02117306,
    1.02987275,
    0.98607016,
    0.98262938,
    0.98404577,
    0.98453141,
    1.01065119,
    1.00807871,
    0.99986077,
    0.99668861,
    0.98597074,
    0.98133325,
    0.98239303,
    0.98370113,
    0.93155548,
    0.93534231,
    0.94231988,
    0.93388087,
    0.84200656,
    0.83657581,
)

# Uncalibrated. Borrowed from the Qwen-Image TeaCache polynomial so the flag
# skips steps instead of raising. Quality will not match a fitted K6 curve.
K6_TEACACHE_COEFFICIENTS: tuple[float, ...] = (
    -4.50000000e02,
    2.80000000e02,
    -4.50000000e01,
    3.20000000e00,
    -2.00000000e-02,
)


@dataclass
class K6StepCache:
    """Step-level skip state shared by MagCache and TeaCache."""

    kind: str
    threshold: float
    max_skip_steps: int
    retention_ratio: float
    mag_ratios: tuple[float, ...] | None = None
    coefficients: tuple[float, ...] | None = None
    rel_l1_thresh: float = 0.2
    last_cond: object = None
    last_uncond: object = None
    accumulated_err: float = 0.0
    accumulated_ratio: float = 1.0
    accumulated_steps: int = 0

    def should_skip(self, step_index: int) -> bool:
        if self.last_cond is None or step_index <= 0:
            return False
        if self.kind == "mag":
            ratios = self.mag_ratios or ()
            scale = float(ratios[step_index]) if step_index < len(ratios) else 1.0
            self.accumulated_ratio *= scale
            self.accumulated_steps += 1
            self.accumulated_err += abs(1.0 - self.accumulated_ratio)
            if (
                self.accumulated_err <= self.threshold
                and self.accumulated_steps <= self.max_skip_steps
                and step_index >= max(1, int(self.retention_ratio * max(step_index, 1)))
            ):
                return True
            self.accumulated_ratio = 1.0
            self.accumulated_steps = 0
            self.accumulated_err = 0.0
            return False
        # TeaCache stand-in: uncalibrated polynomial of the step index.
        # The real TeaCache distance is the modulated input; this only makes
        # the flag skip steps instead of raising.
        coeff = self.coefficients or K6_TEACACHE_COEFFICIENTS
        x = float(step_index)
        predicted = sum(weight * (x ** (len(coeff) - 1 - i)) for i, weight in enumerate(coeff))
        return abs(predicted) < self.rel_l1_thresh * 1.0e4

    def store(self, cond: object, uncond: object) -> None:
        self.last_cond = cond
        self.last_uncond = uncond


def attach_mag_cache(transformer, *, threshold: float, max_skip_steps: int, retention_ratio: float) -> None:
    transformer._k6_step_cache = K6StepCache(
        kind="mag",
        threshold=threshold,
        max_skip_steps=max_skip_steps,
        retention_ratio=retention_ratio,
        mag_ratios=K6_PRO_MAG_RATIOS,
    )


def attach_tea_cache(transformer, *, rel_l1_thresh: float, coefficients: tuple[float, ...] | None) -> None:
    transformer._k6_step_cache = K6StepCache(
        kind="tea",
        threshold=rel_l1_thresh if rel_l1_thresh is not None else 0.2,
        max_skip_steps=3,
        retention_ratio=0.0,
        coefficients=coefficients or K6_TEACACHE_COEFFICIENTS,
        rel_l1_thresh=rel_l1_thresh if rel_l1_thresh is not None else 0.2,
    )
