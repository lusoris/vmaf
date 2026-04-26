/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  SYCL dispatch_strategy implementation. See dispatch_strategy.h
 *  and ADR-0181.
 */
#include "dispatch_strategy.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Backend default: SYCL graph replay wins above 720p frame area on
 * Intel Arc A380 / oneAPI 2025.3 (empirical sweep documented in
 * libvmaf/src/sycl/common.cpp § "Resolution-aware default").
 * Smaller frames pay more in graph setup than they save in
 * dispatch overhead. */
#define VMAF_SYCL_DEFAULT_AREA_THRESHOLD (1280U * 720U)

/* Parse `env_value` looking for a `feature_name:strategy` token. Returns:
 *   1 + GRAPH_REPLAY  → token found, value is "graph"
 *   1 + DIRECT        → token found, value is "direct"
 *   0                 → token not found
 *
 * Token format: comma-separated `name:strategy` pairs. Whitespace
 * around tokens is ignored; case-sensitive feature-name match. */
static int parse_per_feature_override(const char *env_value, const char *feature_name,
                                      VmafSyclDispatchStrategy *out)
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
                *out = VMAF_SYCL_DISPATCH_GRAPH_REPLAY;
                return 1;
            }
            if (strncmp(v, "direct", 6) == 0) {
                *out = VMAF_SYCL_DISPATCH_DIRECT;
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

VmafSyclDispatchStrategy vmaf_sycl_select_strategy(const char *feature_name,
                                                   const VmafFeatureCharacteristics *chars,
                                                   unsigned frame_w, unsigned frame_h)
{
    /* Per-feature env override has highest precedence. */
    const char *env_disp = getenv("VMAF_SYCL_DISPATCH");
    VmafSyclDispatchStrategy out = VMAF_SYCL_DISPATCH_DIRECT;
    if (parse_per_feature_override(env_disp, feature_name, &out))
        return out;

    /* Legacy global env knobs. USE wins over NO when both are set
     * (matches the existing libvmaf/src/sycl/common.cpp semantics). */
    const char *env_use_graph = getenv("VMAF_SYCL_USE_GRAPH");
    const char *env_no_graph = getenv("VMAF_SYCL_NO_GRAPH");
    if (env_use_graph && env_use_graph[0] == '1')
        return VMAF_SYCL_DISPATCH_GRAPH_REPLAY;
    if (env_no_graph && env_no_graph[0] == '1')
        return VMAF_SYCL_DISPATCH_DIRECT;

    /* Descriptor-driven decision. AUTO falls through to the
     * resolution-area default. */
    if (chars) {
        if (chars->dispatch_hint == VMAF_FEATURE_DISPATCH_BATCHED)
            return VMAF_SYCL_DISPATCH_GRAPH_REPLAY;
        if (chars->dispatch_hint == VMAF_FEATURE_DISPATCH_DIRECT)
            return VMAF_SYCL_DISPATCH_DIRECT;
    }

    /* Backend default — area-threshold heuristic preserved from the
     * pre-T7-26 inline logic so behaviour at ≥720p / <720p is
     * unchanged for AUTO descriptors. */
    const unsigned threshold = (chars && chars->min_useful_frame_area) ?
                                   chars->min_useful_frame_area :
                                   VMAF_SYCL_DEFAULT_AREA_THRESHOLD;
    const unsigned long area = (unsigned long)frame_w * (unsigned long)frame_h;
    return (area >= threshold) ? VMAF_SYCL_DISPATCH_GRAPH_REPLAY : VMAF_SYCL_DISPATCH_DIRECT;
}
