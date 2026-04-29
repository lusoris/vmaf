/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP backend common surface — scaffold only (ADR-0209 / T7-10).
 *  Mirrors libvmaf/src/vulkan/common.c. Replace the stubs with a real
 *  HIP runtime probe (`hipInit` / `hipGetDeviceCount` /
 *  `hipDeviceGetName`) and stream creation when the kernels land.
 */

#include <errno.h>
#include <stdlib.h>

#include "common.h"

#include "libvmaf/libvmaf_hip.h"

struct VmafHipContext {
    int device_index;
    /* TODO: add hipDevice_t / hipStream_t handles, allocator, etc. */
};

/* Public-API thin shim: the opaque public type aliases the internal
 * struct in the scaffold. The runtime PR may insert an extra wrapper
 * if its lifetime story diverges from the internal context. */
struct VmafHipState {
    struct VmafHipContext ctx;
};

int vmaf_hip_context_new(VmafHipContext **out, int device_index)
{
    if (out == NULL) {
        return -EINVAL;
    }
    VmafHipContext *ctx = calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return -ENOMEM;
    }
    ctx->device_index = device_index;
    /* TODO: hipSetDevice + hipStreamCreate */
    *out = ctx;
    return 0;
}

void vmaf_hip_context_destroy(VmafHipContext *ctx)
{
    if (ctx == NULL) {
        return;
    }
    /* TODO: hipStreamDestroy */
    free(ctx);
}

int vmaf_hip_device_count(void)
{
    /* TODO: probe HIP runtime via hipGetDeviceCount. */
    return 0;
}

/* ---- Public C-API entry points (libvmaf_hip.h) ---- */

int vmaf_hip_available(void)
{
#ifdef HAVE_HIP
    return 1;
#else
    return 0;
#endif
}

int vmaf_hip_state_init(VmafHipState **out, VmafHipConfiguration cfg)
{
    (void)out;
    (void)cfg;
    /* TODO (T7-10b runtime PR): allocate VmafHipState, init HIP device
     * + stream, return 0 on success / -ENODEV when no device. */
    return -ENOSYS;
}

int vmaf_hip_import_state(VmafContext *ctx, VmafHipState *state)
{
    (void)ctx;
    (void)state;
    /* TODO: stash the HIP state on the VmafContext so the dispatch
     * strategy can route HIP-capable feature extractors. */
    return -ENOSYS;
}

void vmaf_hip_state_free(VmafHipState **state)
{
    if (state == NULL || *state == NULL) {
        return;
    }
    /* TODO: tear down HIP stream / device handles. */
    *state = NULL;
}

int vmaf_hip_list_devices(void)
{
    /* TODO: hipGetDeviceCount + hipDeviceGetName; print one line per
     * device. Returns the count for parity with vmaf_cuda_list_devices. */
    return -ENOSYS;
}
