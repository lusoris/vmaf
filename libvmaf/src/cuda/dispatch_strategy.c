/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  CUDA dispatch_strategy stub. Today every extractor returns
 *  DIRECT; this TU exists to expose the registry-aware decision
 *  surface so future graph-capture work for ADM (16 dispatches/
 *  frame) can land without touching the registration sites.
 *  See ADR-0181.
 */
#include "dispatch_strategy.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static int parse_per_feature_override(const char *env_value, const char *feature_name,
                                      VmafCudaDispatchStrategy *out)
{
    if (!env_value || !feature_name)
        return 0;
    const size_t name_len = strlen(feature_name);
    const char *p = env_value;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == ',')
            ++p;
        if (!*p)
            break;
        const char *colon = strchr(p, ':');
        if (!colon)
            break;
        const size_t tok_name_len = (size_t)(colon - p);
        if (tok_name_len == name_len && memcmp(p, feature_name, name_len) == 0) {
            const char *v = colon + 1;
            if (strncmp(v, "graph", 5) == 0) {
                *out = VMAF_CUDA_DISPATCH_GRAPH_CAPTURE;
                return 1;
            }
            if (strncmp(v, "direct", 6) == 0) {
                *out = VMAF_CUDA_DISPATCH_DIRECT;
                return 1;
            }
        }
        const char *next = strchr(colon, ',');
        if (!next)
            break;
        p = next + 1;
    }
    return 0;
}

VmafCudaDispatchStrategy vmaf_cuda_select_strategy(const char *feature_name,
                                                   const VmafFeatureCharacteristics *chars,
                                                   unsigned frame_w, unsigned frame_h)
{
    (void)chars;
    (void)frame_w;
    (void)frame_h;

    /* Per-feature env override is honoured today (lets users opt-in
     * to graph capture once a backend impl lands). */
    const char *env_disp = getenv("VMAF_CUDA_DISPATCH");
    VmafCudaDispatchStrategy out = VMAF_CUDA_DISPATCH_DIRECT;
    if (parse_per_feature_override(env_disp, feature_name, &out))
        return out;

    /* Stub default — DIRECT for every feature. CUDA graph capture
     * is a follow-up PR (see ADR-0181 § Consequences / follow-ups). */
    return VMAF_CUDA_DISPATCH_DIRECT;
}
