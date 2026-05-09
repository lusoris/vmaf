/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Picture allocation / lifecycle for the Metal backend (ADR-0361 / T8-1).
 *  Stub only — replace with a real `[id<MTLDevice> newBufferWithLength:options:]`
 *  allocator backed by `MTLResourceStorageModeShared` (zero-copy
 *  unified memory on Apple Silicon) when the kernels need it. Mirrors
 *  libvmaf/src/hip/picture_hip.c (ADR-0212).
 *
 *  Apple-Silicon-specific posture: unified memory means there is no
 *  H2D / D2H staging — the runtime PR will allocate one MTLBuffer per
 *  picture and let CPU and GPU read/write the same DRAM. This is the
 *  load-bearing perf story for this backend; see ADR-0361 §Context.
 */

#include <errno.h>
#include <stddef.h>

#include "picture_metal.h"

int vmaf_metal_picture_alloc(VmafMetalContext *ctx, void **out, size_t size)
{
    (void)ctx;
    (void)out;
    (void)size;
    /* TODO (T8-1b runtime): allocate an MTLBuffer with
     * MTLResourceStorageModeShared on Apple Silicon — no H2D copy
     * needed thanks to unified memory. Cache the buffer's [contents]
     * pointer for the host-side write; the GPU sees the same memory
     * via the buffer handle on dispatch. */
    return -ENOSYS;
}

void vmaf_metal_picture_free(VmafMetalContext *ctx, void *buf)
{
    (void)ctx;
    (void)buf;
    /* TODO (T8-1b runtime): release the MTLBuffer via MetalCpp's
     * NS::SharedPtr destructor (auto-release on scope exit). */
}
