# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections.abc import Callable
from dataclasses import dataclass, field

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["sources", "destinations"])
def _copy_rectangles(
    sources,
    destinations,
    widths: tl.constexpr,
    heights: tl.constexpr,
    src_pitches: tl.constexpr,
    dst_pitches: tl.constexpr,
    ends: tl.constexpr,
    block: tl.constexpr,
):
    tile = tl.program_id(0)
    for i in tl.static_range(len(widths)):
        if (tile >= ends[i]) & (tile < ends[i + 1]):
            # All threads in a CTA copy the same rectangle. Only the last
            # tile of each rectangle can cross its logical extent.
            offset = (tile - ends[i]) * block + tl.arange(0, block)
            row, col = offset // widths[i], offset % widths[i]
            valid = row < heights[i]
            source = sources[i].to(tl.pointer_type(tl.uint16))
            dest = destinations[i].to(tl.pointer_type(tl.uint16))
            v = tl.load(source + row * src_pitches[i] + col, valid, other=0)
            tl.store(dest + row * dst_pitches[i] + col, v, valid)


@dataclass
class LocalCopyBatch:
    ops: tuple
    widths: tuple
    heights: tuple
    src_pitches: tuple
    dst_pitches: tuple
    ends: tuple
    block: int = 1024
    launch: Callable = field(init=False, repr=False)

    @classmethod
    def prepare(cls, ops):
        # The original plan already knows sequence permutation, fused-V
        # pitch and joint front/rear offsets. Use its exact byte geometry.
        if any(op[j] % 2 for op in ops for j in (0, 1, 2, 3, 4)):
            raise ValueError("The 16-bit staging copy requires aligned rectangles")
        widths = tuple(op[4] // 2 for op in ops)
        heights = tuple(op[5] for op in ops)
        ends = [0]
        for width, height in zip(widths, heights, strict=True):
            ends.append(ends[-1] + triton.cdiv(width * height, cls.block))
        return cls(
            tuple(ops),
            widths,
            heights,
            tuple(op[3] // 2 for op in ops),
            tuple(op[1] // 2 for op in ops),
            tuple(ends),
        )

    def arguments(self):
        # bind() changes pointer values between layers. Geometry stays fixed.
        return (
            tuple(op[2] for op in self.ops),
            tuple(op[0] for op in self.ops),
            self.widths,
            self.heights,
            self.src_pitches,
            self.dst_pitches,
            self.ends,
            self.block,
        )

    def load(self):
        # Compiling alone leaves the device module and launcher lazy.
        # Binding the 3-D grid loads both without executing a copy.
        grid = (self.ends[-1], 1, 1)
        kernel = _copy_rectangles.warmup(*self.arguments(), grid=grid, num_warps=4)
        self.launch = kernel[grid]

    def submit(self):
        self.launch(*self.arguments())
