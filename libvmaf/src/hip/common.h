/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#ifndef LIBVMAF_HIP_COMMON_H_
#define LIBVMAF_HIP_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scaffolded by ADR-0212 / T7-10 (mirrors the Vulkan scaffold ADR-0175).
 * Replace the stubs in common.c, picture_hip.c, dispatch_strategy.c,
 * and feature/hip/<feature>_hip.c with real HIP runtime implementations.
 */

typedef struct VmafHipContext VmafHipContext;

int vmaf_hip_context_new(VmafHipContext **ctx, int device_index);
void vmaf_hip_context_destroy(VmafHipContext *ctx);
int vmaf_hip_device_count(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_HIP_COMMON_H_ */
