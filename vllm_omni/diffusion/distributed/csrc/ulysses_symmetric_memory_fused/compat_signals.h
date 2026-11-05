/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Exact HCCS signal helpers from the installed CANN SHMEM 1.6.0 header.
// Avoid its unrelated UDMA include, whose internal header is absent in the wheel.

ACLSHMEM_DEVICE void aclshmemi_signal_set(__gm__ int32_t *addr, int32_t val)
{
    MSTX_FUSE_SCOPE_START();
    aclshmemi_store(addr, val);

    // flush data cache to GM after signal to ensure it is visible to other ranks
    dcci_cacheline((__gm__ uint8_t *)addr);
    MSTX_FUSE_SCOPE_END();
    MSTX_SIGNAL_SET_REPORT(addr, val);
}

ACLSHMEM_DEVICE int32_t aclshmemi_signal_wait_until_eq_for_barrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
{
    do {
        dcci_cacheline((__gm__ uint8_t *)sig_addr);

        if (*sig_addr == cmp_val) {
            return *sig_addr;
        }

        // in case when peer pe enters next barrier
        if (*sig_addr == cmp_val + 1) {
            return *sig_addr;
        }
    } while (true);

    // never reach
    return -1;
}
