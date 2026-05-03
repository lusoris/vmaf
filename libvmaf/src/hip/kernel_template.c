/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP kernel-template helper bodies — scaffold only (T7-10 first
 *  consumer / ADR-0241).
 *
 *  Every helper currently returns -ENOSYS. The runtime PR (T7-10b)
 *  replaces the bodies with real HIP calls (`hipStreamCreate`,
 *  `hipEventCreate`, `hipMemsetAsync`, `hipStreamSynchronize`, ...).
 *  The header `kernel_template.h` documents the lifecycle every
 *  consumer expects.
 *
 *  Mirrors `libvmaf/src/cuda/kernel_template.h`'s inline helpers but
 *  as out-of-line definitions; see `kernel_template.h` for why the
 *  HIP variant cannot be `static inline` while the runtime is absent.
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include "common.h"
#include "kernel_template.h"

int vmaf_hip_kernel_lifecycle_init(VmafHipKernelLifecycle *lc, VmafHipContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return -EINVAL;
    }
    lc->str = 0;
    lc->submit = 0;
    lc->finished = 0;
    /* TODO (T7-10b runtime): hipStreamCreateWithFlags +
     * hipEventCreateWithFlags ×2. Roll back on partial failure. */
    return -ENOSYS;
}

int vmaf_hip_kernel_readback_alloc(VmafHipKernelReadback *rb, VmafHipContext *ctx, size_t bytes)
{
    (void)ctx;
    if (rb == NULL) {
        return -EINVAL;
    }
    rb->device = NULL;
    rb->host_pinned = NULL;
    rb->bytes = bytes;
    /* TODO (T7-10b runtime): hipMallocAsync(device, bytes) +
     * hipHostMalloc(host_pinned, bytes, hipHostMallocDefault). */
    return -ENOSYS;
}

int vmaf_hip_kernel_submit_pre_launch(VmafHipKernelLifecycle *lc, VmafHipContext *ctx,
                                      VmafHipKernelReadback *rb, uintptr_t picture_stream,
                                      uintptr_t dist_ready_event)
{
    (void)ctx;
    (void)rb;
    (void)picture_stream;
    (void)dist_ready_event;
    if (lc == NULL) {
        return -EINVAL;
    }
    /* TODO (T7-10b runtime): hipMemsetAsync(rb->device, 0, rb->bytes,
     * (hipStream_t)lc->str) + hipStreamWaitEvent(picture_stream,
     * dist_ready_event, 0). */
    return -ENOSYS;
}

int vmaf_hip_kernel_collect_wait(VmafHipKernelLifecycle *lc, VmafHipContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return -EINVAL;
    }
    /* TODO (T7-10b runtime): hipStreamSynchronize((hipStream_t)lc->str). */
    return -ENOSYS;
}

int vmaf_hip_kernel_lifecycle_close(VmafHipKernelLifecycle *lc, VmafHipContext *ctx)
{
    (void)ctx;
    if (lc == NULL) {
        return 0;
    }
    /* Best-effort: every handle is zero in the scaffold; the runtime
     * PR sequences hipStreamSynchronize → hipStreamDestroy →
     * hipEventDestroy ×2 and aggregates the first error. */
    lc->str = 0;
    lc->submit = 0;
    lc->finished = 0;
    return 0;
}

int vmaf_hip_kernel_readback_free(VmafHipKernelReadback *rb, VmafHipContext *ctx)
{
    (void)ctx;
    if (rb == NULL) {
        return 0;
    }
    /* Scaffold leaves both pointers NULL; runtime PR issues hipFreeAsync
     * + hipHostFree and aggregates errors. */
    rb->device = NULL;
    rb->host_pinned = NULL;
    rb->bytes = 0;
    return 0;
}
