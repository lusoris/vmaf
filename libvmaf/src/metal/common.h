/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#ifndef LIBVMAF_METAL_COMMON_H_
#define LIBVMAF_METAL_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Scaffolded by ADR-0361 / T8-1 (mirrors the HIP scaffold ADR-0212 and
 * the Vulkan scaffold ADR-0175). Replace the stubs in common.c,
 * picture_metal.c, dispatch_strategy.c, kernel_template.c, and
 * feature/metal/<feature>_metal.c with real Metal runtime
 * implementations (MetalCpp wrapper — see ADR-0361).
 */

typedef struct VmafMetalContext VmafMetalContext;

int vmaf_metal_context_new(VmafMetalContext **ctx, int device_index);
void vmaf_metal_context_destroy(VmafMetalContext *ctx);
int vmaf_metal_device_count(void);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_METAL_COMMON_H_ */
