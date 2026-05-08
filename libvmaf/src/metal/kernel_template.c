/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal kernel-template helper bodies — scaffold only (T8-1 /
 *  ADR-0361).
 *
 *  Every helper currently returns -ENOSYS. The runtime PR (T8-1b)
 *  replaces the bodies with real Metal calls (`[id<MTLDevice>
 *  newCommandQueue]`, `[id<MTLDevice> newSharedEvent]`,
 *  `[id<MTLBlitCommandEncoder> fillBuffer:range:value:]`,
 *  `[id<MTLCommandBuffer> waitUntilCompleted]`, ...) via the
 *  MetalCpp C++ wrapper. The header `kernel_template.h` documents
 *  the lifecycle every consumer expects.
 *
 *  Mirrors `libvmaf/src/hip/kernel_template.c` (ADR-0241). The Metal
 *  variant ships out-of-line definitions for the same reason the HIP
 *  variant does — see `kernel_template.h` for the rationale.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include "common.h"
#include "kernel_template.h"

int vmaf_metal_kernel_lifecycle_init(VmafMetalKernelLifecycle *lc, VmafMetalContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return -EINVAL;
    }
    lc->cmd_queue = 0;
    lc->submit = 0;
    lc->finished = 0;
    /* TODO (T8-1b runtime): [id<MTLDevice> newCommandQueue] +
     * [id<MTLDevice> newSharedEvent] ×2. Roll back on partial
     * failure. */
    return -ENOSYS;
}

int vmaf_metal_kernel_buffer_alloc(VmafMetalKernelBuffer *buf, VmafMetalContext *ctx, size_t bytes)
{
    (void)ctx;
    if (buf == NULL) {
        return -EINVAL;
    }
    buf->buffer = 0;
    buf->host_view = NULL;
    buf->bytes = bytes;
    /* TODO (T8-1b runtime): [id<MTLDevice>
     * newBufferWithLength:options:MTLResourceStorageModeShared] +
     * cache `[buffer contents]` in host_view. Unified memory means
     * the same allocation backs both host and device. */
    return -ENOSYS;
}

int vmaf_metal_kernel_submit_pre_launch(VmafMetalKernelLifecycle *lc, VmafMetalContext *ctx,
                                        VmafMetalKernelBuffer *buf,
                                        uintptr_t picture_command_buffer,
                                        uintptr_t dist_ready_event)
{
    (void)ctx;
    (void)buf;
    (void)picture_command_buffer;
    (void)dist_ready_event;
    if (lc == NULL) {
        return -EINVAL;
    }
    /* TODO (T8-1b runtime): blit-encoder fillBuffer to zero the
     * accumulator on lc->cmd_queue + encodeWaitForEvent on the
     * picture command buffer for the dist-ready event. */
    return -ENOSYS;
}

int vmaf_metal_kernel_collect_wait(VmafMetalKernelLifecycle *lc, VmafMetalContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return -EINVAL;
    }
    /* TODO (T8-1b runtime): drain the private command queue via
     * [id<MTLCommandBuffer> waitUntilCompleted] on the latest
     * submitted command buffer. */
    return -ENOSYS;
}

int vmaf_metal_kernel_lifecycle_close(VmafMetalKernelLifecycle *lc, VmafMetalContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return 0;
    }
    /* Best-effort: every handle is zero in the scaffold; the runtime
     * PR sequences waitUntilCompleted → release command queue →
     * release events and aggregates the first error. */
    lc->cmd_queue = 0;
    lc->submit = 0;
    lc->finished = 0;
    return 0;
}

int vmaf_metal_kernel_buffer_free(VmafMetalKernelBuffer *buf, VmafMetalContext *ctx)
{
    (void)ctx;
    if (buf == NULL) {
        return 0;
    }
    /* Scaffold leaves both pointer slots zero; runtime PR releases
     * the MTLBuffer (NS::SharedPtr destructor) and clears
     * host_view. */
    buf->buffer = 0;
    buf->host_view = NULL;
    buf->bytes = 0;
    return 0;
}
