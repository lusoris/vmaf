/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Metal backend common surface — scaffold only (ADR-0361 / T8-1).
 *  Mirrors libvmaf/src/hip/common.c (ADR-0212). Replace the stubs with
 *  a real Metal runtime probe (`MTLCreateSystemDefaultDevice` /
 *  `[id<MTLDevice> name]` / `[id<MTLDevice> supportsFamily:]`) and
 *  command-queue creation when the kernels land.
 *
 *  The runtime PR (T8-1b) will rewrite this TU as Objective-C++
 *  (`.mm`) so it can include `<Metal/Metal.hpp>`; until then a plain
 *  `.c` keeps the scaffold compiling on every build host (Linux dev
 *  sessions, Windows runners) without a Metal SDK.
 */

#include <errno.h>
#include <stdlib.h>

#include "common.h"

#include "libvmaf/libvmaf_metal.h"

struct VmafMetalContext {
    int device_index;
    /* TODO (T8-1b runtime): MetalCpp NS::SharedPtr<MTL::Device> +
     * NS::SharedPtr<MTL::CommandQueue> handles; allocator; metallib
     * cache. Tracked as `uintptr_t` slots in the runtime PR to match
     * the public-header purity convention. */
};

/* Public-API thin shim: the opaque public type aliases the internal
 * struct in the scaffold. The runtime PR may insert an extra wrapper
 * if its lifetime story diverges from the internal context. Same
 * convention as the HIP scaffold. */
struct VmafMetalState {
    struct VmafMetalContext ctx;
};

int vmaf_metal_context_new(VmafMetalContext **out, int device_index)
{
    if (out == NULL) {
        return -EINVAL;
    }
    VmafMetalContext *ctx = calloc(1, sizeof(*ctx));
    if (ctx == NULL) {
        return -ENOMEM;
    }
    ctx->device_index = device_index;
    /* TODO (T8-1b runtime): MTLCreateSystemDefaultDevice +
     * [id<MTLDevice> newCommandQueue]. */
    *out = ctx;
    return 0;
}

void vmaf_metal_context_destroy(VmafMetalContext *ctx)
{
    if (ctx == NULL) {
        return;
    }
    /* TODO (T8-1b runtime): release MTLCommandQueue + MTLDevice via
     * MetalCpp's NS::SharedPtr destructors. */
    free(ctx);
}

int vmaf_metal_device_count(void)
{
    /* TODO (T8-1b runtime): on macOS this is 1 for the system default
     * device on Apple Silicon (no multi-GPU fan-out); on non-macOS
     * hosts always 0. */
    return 0;
}

/* ---- Public C-API entry points (libvmaf_metal.h) ---- */

int vmaf_metal_available(void)
{
#ifdef HAVE_METAL
    return 1;
#else
    return 0;
#endif
}

int vmaf_metal_state_init(VmafMetalState **out, VmafMetalConfiguration cfg)
{
    (void)out;
    (void)cfg;
    /* TODO (T8-1b runtime PR): allocate VmafMetalState, init Metal
     * device + command queue, return 0 on success / -ENODEV when no
     * Apple-Family-7+ device is available (Intel Mac, non-macOS, or
     * GPU below M1). */
    return -ENOSYS;
}

int vmaf_metal_import_state(VmafContext *ctx, VmafMetalState *state)
{
    (void)ctx;
    (void)state;
    /* TODO (T8-1b runtime): stash the Metal state on the VmafContext
     * so the dispatch strategy can route Metal-capable feature
     * extractors. */
    return -ENOSYS;
}

void vmaf_metal_state_free(VmafMetalState **state)
{
    if (state == NULL || *state == NULL) {
        return;
    }
    /* TODO (T8-1b runtime): tear down MTLCommandQueue / MTLDevice
     * handles via MetalCpp. */
    *state = NULL;
}

int vmaf_metal_list_devices(void)
{
    /* TODO (T8-1b runtime): MTLCopyAllDevices on macOS, filter by
     * Apple-Family-7+, print one line per device. Returns the count
     * for parity with vmaf_cuda_list_devices / vmaf_hip_list_devices. */
    return -ENOSYS;
}
