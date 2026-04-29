/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Forward declarations for the HIP feature kernel stubs scaffolded
 *  under ADR-0212 / T7-10. Each kernel exposes the standard
 *  `init` / `run` / `destroy` triplet against an opaque
 *  `VmafHipContext`. The runtime PR (T7-10b) wires these into the
 *  feature registry; the scaffold only declares them so the entry
 *  points are externally linkable (otherwise clang-tidy's
 *  `misc-use-internal-linkage` flags every stub).
 */

#ifndef LIBVMAF_FEATURE_HIP_H_
#define LIBVMAF_FEATURE_HIP_H_

#include "../../hip/common.h"

#ifdef __cplusplus
extern "C" {
#endif

int vmaf_hip_adm_init(VmafHipContext *ctx);
int vmaf_hip_adm_run(VmafHipContext *ctx, const void *ref, const void *dis, int width, int height,
                     int stride);
void vmaf_hip_adm_destroy(VmafHipContext *ctx);

int vmaf_hip_vif_init(VmafHipContext *ctx);
int vmaf_hip_vif_run(VmafHipContext *ctx, const void *ref, const void *dis, int width, int height,
                     int stride);
void vmaf_hip_vif_destroy(VmafHipContext *ctx);

int vmaf_hip_motion_init(VmafHipContext *ctx);
int vmaf_hip_motion_run(VmafHipContext *ctx, const void *ref, const void *dis, int width,
                        int height, int stride);
void vmaf_hip_motion_destroy(VmafHipContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_FEATURE_HIP_H_ */
