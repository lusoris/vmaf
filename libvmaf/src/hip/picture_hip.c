/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Picture allocation / lifecycle for the HIP backend (ADR-0212 / T7-10).
 *  Stub only — replace with a real `hipMalloc` / `hipMallocAsync` (ROCm
 *  6+) allocator backed by an arena when the kernels need it. Mirrors
 *  libvmaf/src/vulkan/picture_vulkan.c.
 */

#include <errno.h>
#include <stddef.h>

#include "picture_hip.h"

int vmaf_hip_picture_alloc(VmafHipContext *ctx, void **out, size_t size)
{
    (void)ctx;
    (void)out;
    (void)size;
    /* TODO: allocate a hipDeviceptr_t via hipMallocAsync. No malloc on
     * the hot path; route every allocation through a pre-sized arena. */
    return -ENOSYS;
}

void vmaf_hip_picture_free(VmafHipContext *ctx, void *buf)
{
    (void)ctx;
    (void)buf;
    /* TODO: hipFreeAsync. */
}
