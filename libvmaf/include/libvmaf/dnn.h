/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Licensed under the BSD+Patent License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      https://opensource.org/licenses/BSDplusPatent
 */

/**
 * @file dnn.h
 * @brief Public DNN surface — load/execute tiny ONNX models alongside SVM models.
 *
 * All functions return 0 on success and a negative errno on failure. When
 * libvmaf was built with `-Denable_dnn=false`, `vmaf_dnn_available()`
 * returns 0 and every other entry point returns -ENOSYS.
 */

#ifndef __VMAF_DNN_H__
#define __VMAF_DNN_H__

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "libvmaf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum VmafDnnDevice {
    VMAF_DNN_DEVICE_AUTO = 0,
    VMAF_DNN_DEVICE_CPU = 1,
    VMAF_DNN_DEVICE_CUDA = 2,
    VMAF_DNN_DEVICE_OPENVINO = 3, /**< covers SYCL / oneAPI / Intel GPU */
    VMAF_DNN_DEVICE_ROCM = 4,
} VmafDnnDevice;

typedef struct VmafDnnConfig {
    VmafDnnDevice device;
    int device_index; /**< multi-GPU index; 0 for single-GPU/CPU */
    int threads;      /**< CPU EP intra-op threads; 0 = ORT default */
    bool fp16_io;     /**< request fp16 tensors when supported */
} VmafDnnConfig;

/**
 * Returns 1 if libvmaf was built with DNN support (-Denable_dnn=true) and
 * ONNX Runtime is linked, 0 otherwise.
 */
int vmaf_dnn_available(void);

/**
 * Attach a tiny ONNX model (C1 / C2) to @p ctx. The model is registered
 * alongside any SVM models and participates in the same per-frame pipeline.
 *
 * @param ctx        live VmafContext (from vmaf_init())
 * @param onnx_path  filesystem path to a .onnx file; must be a regular file
 * @param cfg        optional device config; NULL uses VMAF_DNN_DEVICE_AUTO
 *
 * @return 0 on success, -ENOSYS if built without DNN support, -EINVAL on bad
 *         args, -ENOENT if the path does not exist, -E2BIG if the file is
 *         larger than the compile-time 50 MB cap (VMAF_DNN_DEFAULT_MAX_BYTES).
 */
int vmaf_use_tiny_model(VmafContext *ctx, const char *onnx_path, const VmafDnnConfig *cfg);

/**
 * Standalone DNN session for filter-style inference (learned pre-processing,
 * C3). Unlike vmaf_use_tiny_model() this path does NOT need a VmafContext —
 * intended for consumers that want luma-in / luma-out without scoring.
 */
typedef struct VmafDnnSession VmafDnnSession;

/**
 * Open a session against @p onnx_path. Applies the same size-cap + allowlist
 * validation as vmaf_use_tiny_model().
 */
int vmaf_dnn_session_open(VmafDnnSession **out, const char *onnx_path, const VmafDnnConfig *cfg);

/**
 * Run one luma-in / luma-out pass. The model's input must be NCHW
 * [1, 1, H, W] float32. Input luma is normalised to [0,1] (and mean/std
 * from the sidecar if available); output is denormalised, rounded, and
 * clamped to [0, 255].
 *
 * @return 0 on success, -ENOTSUP if the model shape is not luma-only,
 *         -ERANGE if @p w/@p h don't match the model's static input shape.
 */
int vmaf_dnn_session_run_luma8(VmafDnnSession *sess, const uint8_t *in, size_t in_stride, int w,
                               int h, uint8_t *out, size_t out_stride);

/**
 * 10-/12-/16-bit variant of @ref vmaf_dnn_session_run_luma8 — accepts a
 * packed uint16 little-endian plane and writes one back. The same model
 * trained on normalized float [0,1] works for any bit depth because the
 * loader simply divides by `(1 << bpc) - 1` on the way in and multiplies
 * on the way out. Used by the ffmpeg `vmaf_pre` filter for
 * yuv420p10le / yuv422p10le / yuv444p10le (and 12le) formats, and — on
 * any bit depth — to filter chroma planes with their own dimensions.
 * ADR-0170 / T6-4.
 *
 * @param sess        open session from @ref vmaf_dnn_session_open.
 * @param in          packed uint16 LE input plane.
 * @param in_stride   source row stride in **bytes** (>= w * 2).
 * @param w, h        plane dimensions. Must match the model's static
 *                    input shape or the chroma-plane dimensions if the
 *                    model was re-opened at chroma resolution.
 * @param bpc         bits per component in [9, 16].
 * @param out         packed uint16 LE output plane.
 * @param out_stride  destination row stride in **bytes**.
 *
 * @return 0 on success; -ENOTSUP if the model shape is not plane-only
 *         single-channel; -ERANGE if @p w/@p h don't match; -EINVAL on
 *         a bad @p bpc.
 */
int vmaf_dnn_session_run_plane16(VmafDnnSession *sess, const uint16_t *in, size_t in_stride, int w,
                                 int h, int bpc, uint16_t *out, size_t out_stride);

/**
 * One input tensor passed to vmaf_dnn_session_run(). @p name binds by
 * ONNX graph input name when non-NULL; when NULL, the tensor is bound
 * positionally at the descriptor's array index. Tensors are float32,
 * row-major, with @p rank dimensions.
 */
typedef struct VmafDnnInput {
    const char *name;
    const float *data;
    const int64_t *shape;
    size_t rank;
} VmafDnnInput;

/**
 * One output tensor for vmaf_dnn_session_run(). @p data / @p capacity
 * are caller-owned; @p written is populated with the element count
 * actually produced. @p name binds by ONNX graph output name when
 * non-NULL, else positionally.
 */
typedef struct VmafDnnOutput {
    const char *name;
    float *data;
    size_t capacity;
    size_t written;
} VmafDnnOutput;

/**
 * Run one inference pass with arbitrary named inputs and outputs. All
 * tensors are float32. The session's ONNX graph must declare exactly
 * @p n_inputs inputs and @p n_outputs outputs; mismatched arity returns
 * -EINVAL. Output buffers that are smaller than the produced tensor
 * return -ENOSPC; on -ENOSPC the @p written field is still populated
 * with the required element count so the caller can resize and retry.
 *
 * @return 0 on success; -ENOSYS if built without DNN support;
 *         -EINVAL on bad arity / null pointers; -ENOSPC if any output
 *         buffer is too small; -EIO on ORT failure.
 */
int vmaf_dnn_session_run(VmafDnnSession *sess, const VmafDnnInput *inputs, size_t n_inputs,
                         VmafDnnOutput *outputs, size_t n_outputs);

void vmaf_dnn_session_close(VmafDnnSession *sess);

/**
 * Name of the ONNX Runtime execution provider that actually bound to the
 * session. Useful for diagnostics and for asserting AUTO-chain behaviour
 * in tests. Stable strings: "CPU", "CUDA", "OpenVINO:GPU", "OpenVINO:CPU",
 * "ROCm". Returns NULL if @p sess is NULL or libvmaf was built without
 * DNN support. Lifetime: owned by @p sess.
 */
const char *vmaf_dnn_session_attached_ep(VmafDnnSession *sess);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_DNN_H__ */
