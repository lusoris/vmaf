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
    /* Scaffold stub — return -ENOSYS to signal the feature engine that
     * VIF-HIP is not yet available, matching adm_hip.c and motion_hip.c
     * posture (ADR-0241).  A return of 0 would mislead the engine into
     * thinking initialisation succeeded and then fail silently at run time.
     * TODO: allocate device-side scratch buffers via hipMallocAsync. */
    return -ENOSYS;
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
