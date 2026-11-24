# MiniMax-H3 BF16 attention on gfx950

`wide_first_gfx950.s` contains two wave64 entry points. Both run with 512
threads per workgroup. Their order on a stream is V packing, queue reset,
then dense attention.

| Entry point | Grid | Explicit kernel arguments |
| --- | --- | --- |
| `h3_pack_v` | 593 tiles × 14 heads | K, V, unused packed-K pointer, packed-V pointer, long segment length |
| `h3_wide_attention` | 256 workgroups | 224-byte attention arguments by value, 32-bit descriptor count, pointer to the 32-bit queue counter |

Q and K are read from the strided input views. Packing puts V into the
operand layout for the second matrix multiply. Each descriptor identifies
one query tile and its sequence's valid K/V range. The last 14 descriptors
cover the short sequence, one per head. The two sequences have independent
softmax denominators.

The source labels `.Lwide_stage_a`, `.Lwide_stage_b` and `.Lwide_drain`
mark the three hot regions for 384-query tiles. The corresponding
`.Lnarrow_*` regions implement 256-query tiles. Placement controls the
instruction-fetch layout. The work table controls the order in which the
workgroups claim tiles.

## Rebuilding the code object

From this directory, assemble and link with ROCm LLVM. Compilation can run
on a host without a GPU.

```bash
/opt/rocm/llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx950 \
    -x assembler -nostdlib wide_first_gfx950.s -o wide_first_gfx950.co
```

The executable symbol bytes produced by ROCm 7.2 match the two kernels
used in the MI350X serving benchmark. The hashes cover each symbol's
instruction bytes, including any padding inside the symbol.

| Symbol | Bytes | SHA-256 |
| --- | ---: | --- |
| `h3_pack_v` | 536 | `546fcce842a2a63f57cf1aafbdb021728e86f500c2e4cd291ded7b484cb1591a` |
| `h3_wide_attention` | 29,656 | `353ff0a19a937155b7c9241abd96c41fbb892b891b47e4969888978dbfca1111` |

To compare against ordinary AITER, run the replay from the repository root.
It checks every output row across all nine layouts and includes Q/K/V
staging and both RCCL all-to-all operations. For each timed call it takes the maximum
across the four ranks, then reports the median across 54 calls per path.

```bash
torchrun --nproc-per-node=4 benchmarks/kernels/bench_minimax_h3_wide_fmha.py /tmp/h3-kernels/wide_first.hsaco
```

The HIP implementation from which this specialization was developed is in
[MoonMath's gfx950 attention sources](https://github.com/moonmath-ai/amd-kernels/tree/73961f8ccebf057b9a223a46528a65fc35ba9273/csrc/gfx950).
The local [third-party notice](NOTICE) includes the MoonMath and AMD Opus license.
