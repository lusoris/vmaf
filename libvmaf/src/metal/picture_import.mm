/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  IOSurface zero-copy import — audit-first scaffold (ADR-0423 /
 *  T8-IOS). Mirrors the Vulkan import surface from ADR-0184 /
 *  ADR-0186. Every entry point currently returns -ENOSYS; the
 *  implementation PR (T8-IOS-b) replaces the stubs with
 *  `[id<MTLDevice> newTextureWithDescriptor:iosurface:plane:]` /
 *  `CVMetalTextureCacheCreateTextureFromImage` wiring against the
 *  same MTLDevice the source IOSurface was rendered on.
 *
 *  Why a dedicated TU: keeps the audit-first contract isolated from
 *  the runtime TUs (common.mm, picture_metal.mm) so the flip in
 *  T8-IOS-b is a single-file change reviewers can read end-to-end.
 *  Same pattern Vulkan used between ADR-0184 (image-import scaffold)
 *  and ADR-0186 (image-import impl).
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#import <Foundation/Foundation.h>

extern "C" {
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_metal.h"
}

int vmaf_metal_state_init_external(VmafMetalState **out,
                                   VmafMetalExternalHandles handles)
{
    (void)handles;
    if (out == NULL) {
        return -EINVAL;
    }
    *out = NULL;
    /* T8-IOS scaffold contract: every entry point reports -ENOSYS
     * until T8-IOS-b lands. See libvmaf/include/libvmaf/libvmaf_metal.h
     * and docs/adr/0423-metal-iosurface-import-scaffold.md. */
    return -ENOSYS;
}

int vmaf_metal_picture_import(VmafMetalState *state, uintptr_t iosurface,
                              unsigned plane, unsigned w, unsigned h,
                              unsigned bpc, int is_ref, unsigned index)
{
    (void)state;
    (void)iosurface;
    (void)plane;
    (void)w;
    (void)h;
    (void)bpc;
    (void)is_ref;
    (void)index;
    return -ENOSYS;
}

int vmaf_metal_wait_compute(VmafMetalState *state)
{
    (void)state;
    return -ENOSYS;
}

int vmaf_metal_read_imported_pictures(VmafContext *ctx, unsigned index)
{
    (void)ctx;
    (void)index;
    return -ENOSYS;
}
