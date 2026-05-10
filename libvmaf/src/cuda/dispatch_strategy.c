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

#include <pthread.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/*
 * Cache the VMAF_CUDA_DISPATCH environment variable once, protected by
 * pthread_once, to avoid calling getenv() from multiple threads
 * simultaneously (concurrency-mt-unsafe per POSIX.1-2008 §2.2.2 if another
 * thread calls setenv/putenv/unsetenv concurrently).
 *
 * Contract: callers must set VMAF_CUDA_DISPATCH before the first CUDA frame
 * is submitted (i.e., before any call to vmaf_read_pictures on a CUDA
 * context). The snapshot is permanent; later setenv calls are not observed.
 *
 * NULL means the variable was unset at snapshot time.
 */
static pthread_once_t g_env_once = PTHREAD_ONCE_INIT;
static const char *g_env_disp = NULL;

static void cache_env_dispatch(void)
{
    /* Serialised by pthread_once — only one thread reaches this function.
     * The remaining concurrency-mt-unsafe risk (concurrent setenv from another
     * thread racing the single getenv here) is caller-contract per
     * POSIX.1-2008 §2.2.2: "Conforming multi-threaded applications shall not
     * use setenv, unsetenv, or putenv while a call to getenv is in progress."
     * NOLINT(concurrency-mt-unsafe) — call is serialised by pthread_once;
     * simultaneous setenv is a caller-contract violation, not a library bug. */
    const char *val = getenv("VMAF_CUDA_DISPATCH"); // NOLINT(concurrency-mt-unsafe)
    if (val)
        g_env_disp = strdup(val); /* stable copy; outlives any subsequent setenv */
}

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

    /* Read VMAF_CUDA_DISPATCH at most once across all threads (pthread_once
     * provides the acquire fence that makes g_env_disp visible here). */
    (void)pthread_once(&g_env_once, cache_env_dispatch);

    VmafCudaDispatchStrategy out = VMAF_CUDA_DISPATCH_DIRECT;
    if (parse_per_feature_override(g_env_disp, feature_name, &out))
        return out;

    /* Stub default — DIRECT for every feature. CUDA graph capture
     * is a follow-up PR (see ADR-0181 § Consequences / follow-ups). */
    return VMAF_CUDA_DISPATCH_DIRECT;
}
