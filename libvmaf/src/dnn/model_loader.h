/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 */

#ifndef LIBVMAF_DNN_MODEL_LOADER_H_
#define LIBVMAF_DNN_MODEL_LOADER_H_

#include <stdbool.h>
#include <stddef.h>

#include "libvmaf/model.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Compile-time cap for ONNX file size (50 MB).
 *
 *  The historical `VMAF_MAX_MODEL_BYTES` env override was removed in
 *  T7-12 once two release cycles passed without a shipped model
 *  approaching the cap. The constant is now the single source of
 *  truth — bump it here (and re-run the size-cap tests) if a future
 *  use case genuinely needs a larger envelope. */
#define VMAF_DNN_DEFAULT_MAX_BYTES ((size_t)50u * 1024u * 1024u)

/** Post-training quantisation mode (ADR-0129 / ADR-0173). Tracks the
 *  per-model registry field of the same name; FP32 means the loader
 *  uses the .onnx file as shipped, the other three modes mean the
 *  loader prefers a sibling `<basename>.int8.onnx` produced by
 *  `ai/scripts/ptq_*.py` / `qat_train.py`. */
typedef enum VmafModelQuantMode {
    VMAF_QUANT_FP32 = 0,
    VMAF_QUANT_DYNAMIC = 1,
    VMAF_QUANT_STATIC = 2,
    VMAF_QUANT_QAT = 3,
} VmafModelQuantMode;

typedef struct VmafModelSidecar {
    VmafModelKind kind; /**< mirrors sidecar "kind" field */
    int opset;
    char *name;        /**< owned */
    char *input_name;  /**< owned */
    char *output_name; /**< owned */
    float norm_mean;
    float norm_std;
    bool has_norm;
    float expected_min;
    float expected_max;
    bool has_range;
    VmafModelQuantMode quant_mode; /**< default FP32; mirrors sidecar/registry */
} VmafModelSidecar;

/** Byte-identical magic check. Returns VMAF_MODEL_KIND_SVM for libsvm json/pkl,
 *  DNN_FR/DNN_NR for ONNX (kind refined from sidecar), or -1 on unknown. */
int vmaf_dnn_sniff_kind(const char *path);

/** Loads `<path>.json` next to the ONNX file. Returns 0 on success. */
int vmaf_dnn_sidecar_load(const char *onnx_path, VmafModelSidecar *out);
void vmaf_dnn_sidecar_free(VmafModelSidecar *s);

/** Validate an ONNX file on disk: size cap + operator allowlist. */
int vmaf_dnn_validate_onnx(const char *path, size_t max_bytes);

#ifdef __cplusplus
}
#endif

#endif /* LIBVMAF_DNN_MODEL_LOADER_H_ */
