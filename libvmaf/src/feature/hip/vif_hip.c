/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Stub VIF kernel for the HIP backend. Scaffolded by ADR-0212 /
 *  T7-10 — replace with a real HIP implementation when the runtime
 *  PR lands.
 */

#include <errno.h>

#include "feature_hip.h"

int vmaf_hip_vif_init(VmafHipContext *ctx)
{
    (void)ctx;
    /* TODO: allocate device-side scratch buffers via hipMallocAsync. */
    return 0;
}

int vmaf_hip_vif_run(VmafHipContext *ctx, const void *ref, const void *dis, int width, int height,
                     int stride)
{
    (void)ctx;
    (void)ref;
    (void)dis;
    (void)width;
    (void)height;
    (void)stride;
    /* TODO: launch VIF kernel on HIP. */
    return -ENOSYS;
}

void vmaf_hip_vif_destroy(VmafHipContext *ctx)
{
    (void)ctx;
    /* TODO: free device-side scratch buffers. */
}
