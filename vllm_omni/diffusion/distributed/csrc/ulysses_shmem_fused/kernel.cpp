// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_mte.h"
#include "device/gm2gm/shmem_device_cc.h"
#include "compat_signals.h"
using namespace AscendC;

__aicore__ inline void pull_body(TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1>& queue,
    GM_ADDR peers, GM_ADDR output, uint32_t rank, uint32_t world, uint32_t batch,
    uint32_t seq, uint32_t row_bytes, uint32_t reverse, uint64_t offset) {
    uint32_t rows = 32768 / row_bytes;
    if (rows > 16) rows = 16;
    uint32_t tiles = (seq + rows - 1) / rows;
    uint32_t cores = GetBlockNum() * GetTaskRation();
    for (uint32_t job = GetBlockIdx(); job < batch * world * tiles; job += cores) {
        // Interleave peers within each tile so workers spread remote reads.
        uint32_t tile = (job / world) % tiles;
        uint32_t peer = job % world;
        uint32_t b = job / tiles / world;
        uint32_t s = tile * rows;
        uint32_t count = (seq - s < rows) ? seq - s : rows;
        auto source = (__gm__ uint8_t*)((__gm__ uint64_t*)peers)[peer] + offset;
        uint64_t src = reverse ? ((uint64_t(b) * world + rank) * seq + s) * row_bytes
                               : ((uint64_t(b) * seq + s) * world + rank) * row_bytes;
        uint64_t dst = reverse ? ((uint64_t(b) * seq + s) * world + peer) * row_bytes
                               : ((uint64_t(b) * world + peer) * seq + s) * row_bytes;
        GlobalTensor<uint8_t> input_gm, output_gm;
        input_gm.SetGlobalBuffer(source + src);
        output_gm.SetGlobalBuffer(output + dst);
        auto local = queue.AllocTensor<uint8_t>();
        DataCopyExtParams read{(uint16_t)count, row_bytes,
            reverse ? 0 : (world - 1) * row_bytes, 0, 0};
        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
        DataCopyPad(local, input_gm, read, pad);
        queue.EnQue(local);
        local = queue.DeQue<uint8_t>();
        DataCopyExtParams write{(uint16_t)count, row_bytes, 0,
            reverse ? (world - 1) * row_bytes : 0, 0};
        DataCopyPad(output_gm, local, write);
        queue.FreeTensor(local);
    }
}

extern "C" __global__ __aicore__ void ulysses_fused_exchange_kernel(
    uint64_t ffts, GM_ADDR peers, GM_ADDR q, GM_ADDR k, GM_ADDR v,
    uint32_t rank, uint32_t world, uint32_t batch, uint32_t seq,
    uint32_t qr, uint32_t kr, uint32_t vr, uint32_t reverse, uint32_t count) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    util_set_ffts_config(ffts);
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queue;
    if ASCEND_IS_AIV { pipe.InitBuffer(queue, 1, 32768); }
    aclshmem_barrier_all();
    if ASCEND_IS_AIV {
        uint64_t stride = uint64_t(batch) * seq * world;
        pull_body(queue, peers, q, rank, world, batch, seq, qr, reverse, 0);
        if (count == 3) {
            pull_body(queue, peers, k, rank, world, batch, seq, kr, 0, stride * qr);
            pull_body(queue, peers, v, rank, world, batch, seq, vr, 0, stride * (qr + kr));
        }
        PipeBarrier<PIPE_ALL>();
    }
    aclshmem_barrier_all();
}

extern "C" void launch_ulysses_fused_exchange(void* stream, uint64_t ffts,
    void* peers, void* q, void* k, void* v, uint32_t rank, uint32_t world,
    uint32_t batch, uint32_t seq, uint32_t qr, uint32_t kr, uint32_t vr,
    uint32_t reverse, uint32_t count) {
    // MIX 1:2 provides two AIV workers per block. Launching world blocks
    // gives each peer two workers: the 2*world job stride advances tiles
    // without changing job % world, for any supported group size.
    ulysses_fused_exchange_kernel<<<world, nullptr, stream>>>(ffts, (uint8_t*)peers,
        (uint8_t*)q, (uint8_t*)k, (uint8_t*)v, rank, world, batch, seq, qr, kr, vr, reverse, count);
}
