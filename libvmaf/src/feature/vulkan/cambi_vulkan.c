/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  cambi feature kernel on the Vulkan backend — **feasibility spike
 *  scaffold** (ADR-0205). This file ships as an architectural
 *  reference for the cambi GPU port; it is *not yet wired into the
 *  build* and *not yet registered* with `feature_extractor.c`.
 *
 *  This mirrors the precedent of `ssimulacra2_vulkan.c` (ADR-0201),
 *  which landed as a dormant in-tree reference before its build
 *  integration PR. The follow-up integration PR will:
 *
 *    1. Add this TU to `vulkan_sources` in
 *       `libvmaf/src/vulkan/meson.build`.
 *    2. Add the four `cambi_*.comp` shaders to the same file's
 *       `shaders` list.
 *    3. Add `extern VmafFeatureExtractor vmaf_fex_cambi_vulkan;` and
 *       `&vmaf_fex_cambi_vulkan` registration in
 *       `libvmaf/src/feature/feature_extractor.c`.
 *    4. Implement the host residual call path that runs the existing
 *       CPU `calculate_c_values` + `spatial_pooling` against the
 *       GPU-produced image + mask buffers.
 *    5. Add a Lavapipe lane to `tests-and-quality-gates.yml` and a
 *       `FEATURE_METRICS` row in `cross_backend_vif_diff.py`.
 *    6. Empirically validate the precision contract (places=2 per
 *       ADR-0192; expected bit-exact since the GPU phases are
 *       integer-only).
 *
 *  Architecture sketch (from ADR-0205 §Decision):
 *
 *      ┌────────────────────────────────────────────────────────┐
 *      │ Host: cambi_preprocessing (decimate + bit-shift)       │
 *      │       — runs on CPU; GPU upload after.                 │
 *      ├────────────────────────────────────────────────────────┤
 *      │ GPU dispatch chain (per scale 0..4):                   │
 *      │   1. cambi_derivative.comp   → derivative buffer       │
 *      │   2. cambi_mask_dp.comp      → spatial mask            │
 *      │      (separable summed-area table; two passes)         │
 *      │   3. cambi_decimate.comp     → image  / 2 (skip s=0)   │
 *      │   3b. cambi_decimate.comp    → mask   / 2 (skip s=0)   │
 *      │   4. cambi_filter_mode.comp  → mode-filter image       │
 *      │      (horizontal then vertical; AXIS spec const)       │
 *      ├────────────────────────────────────────────────────────┤
 *      │ Readback boundary: HOST_VISIBLE map of image + mask.   │
 *      ├────────────────────────────────────────────────────────┤
 *      │ Host residual:                                         │
 *      │   - calculate_c_values (existing CPU path)             │
 *      │   - spatial_pooling   (top-K mean)                     │
 *      │   - scale-weighted final score                         │
 *      └────────────────────────────────────────────────────────┘
 *
 *  Precision contract: places=2 per ADR-0192. The hybrid satisfies
 *  this trivially because the GPU phases are integer + bit-exact;
 *  the precision-sensitive c-values + pooling phase stays on the
 *  host. See ADR-0205 §Precision investigation.
 *
 *  Out of scope for v1 (deferred to a future ADR per the digest
 *  §"Follow-up work for v2"):
 *    - Fully-on-GPU calculate_c_values via strategy III (direct
 *      per-pixel histogram). Tracked as long-tail batch 4.
 *    - Heatmap dump on GPU (CPU-only feature for now).
 *    - CUDA + SYCL twins (ship after Vulkan v1 lands).
 */

#include <errno.h>
#include <stdint.h>
#include <string.h>

#include "feature_extractor.h"
#include "feature_name.h"

/* Forward declaration of the eventual Vulkan glue struct. The full
 * definition lands in the integration PR; this stub documents the
 * fields the integration is expected to carry so reviewers of the
 * spike PR can sanity-check the design. */
typedef struct CambiVulkanState {
    /* Geometry / configuration (mirrors CambiState in cambi.c). */
    unsigned enc_width;
    unsigned enc_height;
    unsigned enc_bitdepth;
    unsigned src_width;
    unsigned src_height;
    uint16_t window_size;
    uint16_t src_window_size;
    double topk;
    double cambi_topk;
    double tvi_threshold;
    double cambi_max_val;
    double cambi_vis_lum_threshold;
    uint16_t vlt_luma;
    uint16_t max_log_contrast;

    /* Vulkan plumbing (filled in by the integration PR; declared
     * here as opaque pointers so the public ABI is stable). */
    void *ctx; /* VmafVulkanContext * */
    int owns_ctx;
    void *dsl_derivative; /* VkDescriptorSetLayout */
    void *dsl_decimate;
    void *dsl_filter_mode;
    void *dsl_mask_dp;
    void *pipeline_layout_derivative;
    void *pipeline_layout_decimate;
    void *pipeline_layout_filter_mode;
    void *pipeline_layout_mask_dp;
    void *shader_derivative;
    void *shader_decimate;
    void *shader_filter_mode;
    void *shader_mask_dp;
    void *pipeline_derivative;
    void *pipeline_decimate;
    void *pipeline_filter_mode_h;
    void *pipeline_filter_mode_v;
    void *pipeline_mask_dp_row;
    void *pipeline_mask_dp_col;
    void *desc_pool;

    /* GPU buffers (one per scale: 5 scales × 2 buffers).
     * HOST_VISIBLE so the residual can read back without staging. */
    void *image_buf; /* VmafVulkanBuffer * */
    void *mask_buf;
    void *scratch_buf; /* derivative + filter_mode scratch  */

    /* Host-side reuse of the existing CPU helpers. */
    void *cambi_buffers; /* CambiBuffers — same as CPU path  */

    VmafDictionary *feature_name_dict;
} CambiVulkanState;

/* The integration PR replaces this `init_stub`/`extract_stub`/
 * `close_stub` triple with the real Vulkan-aware lifecycle. Keeping
 * them as visible no-ops here documents the API surface and ensures
 * the file compiles in isolation if a future contributor flips the
 * meson wiring on by accident. */

static int cambi_vulkan_init_stub(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                                  unsigned bpc, unsigned w, unsigned h)
{
    (void)fex;
    (void)pix_fmt;
    (void)bpc;
    (void)w;
    (void)h;
    /* Spike scaffold: returning -ENOSYS makes it explicit that this
     * extractor is not yet usable. The integration PR replaces this
     * with the full Vulkan init (descriptor pool, pipeline build,
     * GPU buffer allocation, host CambiBuffers reuse).
     *
     * NB: do NOT register this extractor in feature_extractor.c
     * until the integration PR; otherwise selecting `cambi_vulkan`
     * will fail at init time with -ENOSYS, confusing users. */
    return -ENOSYS;
}

static int cambi_vulkan_extract_stub(VmafFeatureExtractor *fex, VmafPicture *ref_pic,
                                     VmafPicture *ref_pic_90, VmafPicture *dist_pic,
                                     VmafPicture *dist_pic_90, unsigned index,
                                     VmafFeatureCollector *fc)
{
    (void)fex;
    (void)ref_pic;
    (void)ref_pic_90;
    (void)dist_pic;
    (void)dist_pic_90;
    (void)index;
    (void)fc;
    /* Integration PR fills this in:
     *   1. Upload dist_pic luma plane to image_buf[scale=0].
     *   2. Dispatch chain per ADR-0205 §Decision (derivative →
     *      mask_dp → decimate → filter_mode for each scale).
     *   3. Wait for the queue, flush HOST_VISIBLE caches.
     *   4. Call calculate_c_values + spatial_pooling on the mapped
     *      image_buf + mask_buf for each scale.
     *   5. Apply scale-weighted final score per CPU path.
     *   6. Append "Cambi_feature_cambi_score" to the collector. */
    return -ENOSYS;
}

static int cambi_vulkan_close_stub(VmafFeatureExtractor *fex)
{
    (void)fex;
    return 0;
}

static const char *cambi_vulkan_provided_features[] = {
    "Cambi_feature_cambi_score",
    NULL,
};

/* The integration PR replaces this dormant entry point with the
 * full extractor structure and adds it to feature_extractor.c. */
VmafFeatureExtractor vmaf_fex_cambi_vulkan_scaffold = {
    .name = "cambi_vulkan",
    .init = cambi_vulkan_init_stub,
    .extract = cambi_vulkan_extract_stub,
    .options = NULL,
    .close = cambi_vulkan_close_stub,
    .priv_size = sizeof(CambiVulkanState),
    .provided_features = cambi_vulkan_provided_features,
};
