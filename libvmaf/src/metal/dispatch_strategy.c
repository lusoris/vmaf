/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Dispatch strategy stub for the Metal backend (ADR-0361 / T8-1).
 *  Mirrors libvmaf/src/hip/dispatch_strategy.c. The runtime PR will
 *  replace this with a feature-name → metallib-kernel routing table.
 */

#include "dispatch_strategy.h"

int vmaf_metal_dispatch_supports(const VmafMetalContext *ctx, const char *feature)
{
    (void)ctx;
    (void)feature;
    /* TODO (T8-1b runtime): walk a feature-name → metal-kernel
     * registry once kernels exist (initially: motion_v2_metal). */
    return 0;
}
