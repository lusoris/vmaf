/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#ifndef LIBVMAF_METAL_DISPATCH_STRATEGY_H_
#define LIBVMAF_METAL_DISPATCH_STRATEGY_H_

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Returns 1 if the Metal backend can dispatch the named feature on
 * the given context, 0 otherwise. Stub: returns 0 unconditionally
 * until the kernels land. Same shape as the Vulkan / SYCL / CUDA /
 * HIP dispatch predicates.
 */
int vmaf_metal_dispatch_supports(const VmafMetalContext *ctx, const char *feature);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_METAL_DISPATCH_STRATEGY_H_ */
