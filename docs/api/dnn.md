# DNN session API — `libvmaf/dnn.h`

The DNN surface in [`libvmaf/include/libvmaf/dnn.h`](../../libvmaf/include/libvmaf/dnn.h)
lets callers load and run tiny ONNX models from C, either attached to a
`VmafContext` (so DNN scores show up next to SVM scores in the normal VMAF
report) or as a standalone session (luma-in / luma-out filter-style
inference, no VmafContext required).

This is the runtime half of the tiny-AI surface; training lives in `ai/`.
See [ADR-0022](../adr/0022-inference-runtime-onnx.md) (ORT as the inference
runtime) and [ADR-0023](../adr/0023-tinyai-user-surfaces.md) (the four user
surfaces: CLI, C API, ffmpeg, training).

## Availability check

```c
int vmaf_dnn_available(void);
```

Returns `1` if libvmaf was built with `-Denable_dnn=true` and ONNX Runtime is
linked, `0` otherwise. When `0`, every other entry point in `dnn.h` returns
`-ENOSYS`. This is the cheap way to branch between DNN and classic-only
build configs at runtime without wrapping every call in its own check.

## Device config — `VmafDnnConfig`

```c
typedef enum VmafDnnDevice {
    VMAF_DNN_DEVICE_AUTO     = 0,
    VMAF_DNN_DEVICE_CPU      = 1,
    VMAF_DNN_DEVICE_CUDA     = 2,
    VMAF_DNN_DEVICE_OPENVINO = 3,  /* OpenVINO GPU with CPU fallback */
    VMAF_DNN_DEVICE_ROCM     = 4,
    VMAF_DNN_DEVICE_COREML   = 5,
    VMAF_DNN_DEVICE_COREML_ANE = 6,
    VMAF_DNN_DEVICE_COREML_GPU = 7,
    VMAF_DNN_DEVICE_COREML_CPU = 8,
    VMAF_DNN_DEVICE_OPENVINO_NPU = 9,
    VMAF_DNN_DEVICE_OPENVINO_CPU = 10,
    VMAF_DNN_DEVICE_OPENVINO_GPU = 11,
} VmafDnnDevice;

typedef struct VmafDnnConfig {
    VmafDnnDevice device;
    int  device_index;  /* multi-GPU index; 0 for single-GPU/CPU */
    int  threads;       /* CPU EP intra-op threads; 0 = ORT default */
    bool fp16_io;       /* request fp16 tensors when supported */
} VmafDnnConfig;
```

- `device = AUTO` — tries CUDA, OpenVINO GPU, ROCm, CoreML, then CPU.
  OpenVINO NPU is intentionally explicit-only because small graphs can pay a
  noticeable NPU power-state latency floor.
- `device = CPU` — forces ORT's CPU execution provider.
- `device = CUDA` — tries `CUDAExecutionProvider` when the linked ORT build
  exports it. If CUDA EP append fails, the session falls back to CPU and
  still opens.
- `device = OPENVINO` — tries OpenVINO `device_type=GPU`, then
  `device_type=CPU`.
- `device = OPENVINO_NPU` / `_CPU` / `_GPU` — pins the OpenVINO EP to a
  single `device_type` (`NPU`, `CPU`, or `GPU`). Missing EP support or
  absent silicon still degrades to CPU through the common fallback.
- `device = ROCM` — tries `ROCMExecutionProvider`, then falls back to CPU.
- `device = COREML` / `_ANE` / `_GPU` / `_CPU` — tries
  `CoreMLExecutionProvider`. The base selector lets CoreML choose compute
  units; the variants set `MLComputeUnits` to `CPUAndNeuralEngine`,
  `CPUAndGPU`, or `CPUOnly`. Non-Apple ORT builds fall back to CPU.
- `fp16_io` — enables fp32-to-fp16 staging for model slots declared as
  `FLOAT16`. OpenVINO also receives the `precision=FP16` EP option.
- `threads = 0` — lets ORT pick. Set explicitly when pinning affinity or
  benchmarking.

Pass `NULL` for `cfg` in any function that accepts one to use
`VMAF_DNN_DEVICE_AUTO` with zero device index, zero threads, and no fp16 I/O.

## Attached mode — `vmaf_use_tiny_model`

```c
int vmaf_use_tiny_model(VmafContext *ctx,
                        const char *onnx_path,
                        const VmafDnnConfig *cfg);
```

Register a tiny ONNX model on a live `VmafContext`. The model participates in
the per-frame pipeline; its outputs appear in the report alongside SVM
scores. Use this when you want "VMAF + tiny AI score" in the same run.

Returns:

- `0` — success.
- `-ENOSYS` — built without DNN support.
- `-EINVAL` — bad args (null `ctx` or `onnx_path`).
- `-ENOENT` — `onnx_path` does not exist or is not a regular file.
- `-E2BIG` — file exceeds the compile-time 50 MB cap
  (`VMAF_DNN_DEFAULT_MAX_BYTES` — defence against adversarial bloat;
  see [ADR-0039](../adr/0039-onnx-runtime-op-walk-registry.md)). The
  historical `VMAF_MAX_MODEL_BYTES` env override was retired in T7-12.
- `-ENOMEM` — session allocation failed (ORT env, session options, or
  internal buffer allocation).
- Negative `errno` from the operator-allowlist walk if the model contains a
  banned op.

Equivalent CLI flag: `--tiny-model <path>`
([usage/cli.md](../usage/cli.md#tiny-ai-flags-fork-added)).

## Standalone sessions — `VmafDnnSession`

Standalone mode is for filter-style inference that does not need a
`VmafContext` — e.g. a learned de-banding preprocessor that mutates a luma
plane before downstream processing.

```c
typedef struct VmafDnnSession VmafDnnSession;

int  vmaf_dnn_session_open (VmafDnnSession **out, const char *onnx_path, const VmafDnnConfig *cfg);
void vmaf_dnn_session_close(VmafDnnSession *sess);
```

Both `*_session_open` and `vmaf_use_tiny_model` apply the same
size-cap + operator-allowlist walk. See
[ADR-0039](../adr/0039-onnx-runtime-op-walk-registry.md) for the allowlist
and [ADR-0041](../adr/0041-lpips-sq-extractor.md) for an example of an
extractor that uses a session under the hood.

### Luma-only convenience call

```c
int vmaf_dnn_session_run_luma8(VmafDnnSession *sess,
                               const uint8_t *in,  size_t in_stride,
                               int w, int h,
                               uint8_t *out,       size_t out_stride);
```

Runs one luma-in / luma-out pass. Only works when the ONNX graph has:

- exactly one float32 input of static shape `[1, 1, H, W]`,
- exactly one output of the same shape.

The implementation:

1. Reads `in` (uint8 luma), normalises to `[0, 1]` (applies mean/std from the
   sidecar JSON if present).
2. Runs ORT.
3. De-normalises, rounds, clamps to `[0, 255]`, writes `out`.

Errors:

- `-EINVAL` — `sess`, `in`, or `out` is NULL.
- `-ENOTSUP` — graph shape isn't the supported NCHW `[1,1,H,W]` luma layout,
  or ORT returned fewer output elements than `w*h`.
- `-ERANGE` — `w`/`h` don't match the graph's static input shape. Use
  `vmaf_dnn_session_run()` for dynamic shapes.
- `-ENOSYS` — libvmaf was built without ONNX Runtime support
  (`enable_onnx=false`); every DNN entry point returns `-ENOSYS` in that
  configuration.

### 10/12/16-bit convenience call

```c
int vmaf_dnn_session_run_plane16(VmafDnnSession *sess,
                                 const uint16_t *in,  size_t in_stride,
                                 int w, int h, int bpc,
                                 uint16_t *out,        size_t out_stride);
```

The bit-depth-extended sibling of `_luma8`. Used by the ffmpeg
`vmaf_pre` filter for `yuv420p10le` / `yuv422p10le` / `yuv444p10le`
(and the 12-bit LE counterparts), and — at any supported bit depth —
to run the same session on chroma planes at their sub-sampled
dimensions. Added in
[ADR-0170](../adr/0170-vmaf-pre-10bit-chroma.md) (T6-4).

- `in` / `out` are packed `uint16` little-endian single-plane
  buffers.
- `in_stride` / `out_stride` are in **bytes** (not samples) — same
  convention as `_luma8`, so a 10-bit 1920×1080 plane has
  `stride ≥ 1920 * 2`.
- `bpc` in range 9..16 selects the normalisation divisor
  `(1 << bpc) - 1`. Passing `bpc=8` returns `-EINVAL` — use
  `_luma8` for 8-bit input.

The model must still declare `[1, 1, H, W]` static shape; the only
new freedom is the bit depth of the host-side buffer the loader
normalises from. A single `learned_filter_v1` session works for
both luma and chroma — re-call with chroma W/H (the shape is
declared dynamic, see the open() comment).

Errors match `_luma8`, plus `-EINVAL` for a `bpc` outside `[9, 16]`.

### General named-binding call

For models with multiple inputs / outputs or non-luma shapes, use the
general call:

```c
typedef struct VmafDnnInput {
    const char    *name;   /* bind by graph name; NULL = positional */
    const float   *data;   /* row-major float32 */
    const int64_t *shape;  /* rank dims */
    size_t         rank;
} VmafDnnInput;

typedef struct VmafDnnOutput {
    const char *name;      /* bind by graph name; NULL = positional */
    float      *data;      /* caller-owned */
    size_t      capacity;  /* element count allocated */
    size_t      written;   /* OUT: element count produced */
} VmafDnnOutput;

int vmaf_dnn_session_run(VmafDnnSession *sess,
                         const VmafDnnInput *inputs,   size_t n_inputs,
                         VmafDnnOutput     *outputs,   size_t n_outputs);
```

Name-binding (`name != NULL`) resolves by the ONNX graph's declared input /
output names. Positional binding (`name == NULL`) uses the tensor's array
index. Mix is allowed but discouraged — pick one style per session.

Errors:

- `-ENOSYS` — built without DNN support.
- `-EINVAL` — mismatched arity, null pointers, rank zero.
- `-ENOMEM` — allocation failure (per-input staging buffer, or tensor
  creation).
- `-ENOSPC` — some `outputs[i].capacity` is smaller than the produced tensor.
  On this return, `outputs[i].written` is populated with the required
  element count (the code sets `written = produced` *before* the capacity
  check), so the caller can resize and retry with the same bindings.
- `-EIO` — ORT failure (bad graph, EP crash, OOM on device). The diagnostic
  is logged via the `VmafContext` log callback if one is configured (for
  sessions opened without a `VmafContext`, logging goes through the
  library's global log sink).

See [ADR-0040](../adr/0040-dnn-session-multi-input-api.md) for the rationale
behind multi-input/output + named binding.

## Thread-safety

- A single `VmafDnnSession` is **not** re-entrant. Driving inference from two
  threads requires either per-thread sessions or external locking.
- Opening multiple sessions concurrently is safe; they do not share state
  beyond process-global ORT singletons.
- Attaching a tiny model via `vmaf_use_tiny_model()` is subject to the same
  single-driver rule as the rest of the `VmafContext` API — see
  [index.md](index.md#thread-safety).

## Runnable example — standalone luma filter

```c
#include <errno.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <libvmaf/dnn.h>

int main(int argc, char **argv)
{
    if (!vmaf_dnn_available()) {
        fprintf(stderr, "libvmaf was built without DNN support\n");
        return 2;
    }

    VmafDnnSession *sess = NULL;
    VmafDnnConfig cfg = { .device = VMAF_DNN_DEVICE_AUTO };

    int err = vmaf_dnn_session_open(&sess, argv[1], &cfg);
    if (err < 0) {
        fprintf(stderr, "open failed: %d (%s)\n", err, strerror(-err));
        return 1;
    }

    const int W = 1920, H = 1080;
    uint8_t *in  = malloc((size_t)W * H);
    uint8_t *out = malloc((size_t)W * H);
    /* ...fill `in` with luma from your pipeline... */

    err = vmaf_dnn_session_run_luma8(sess, in, W, W, H, out, W);
    if (err < 0) fprintf(stderr, "run failed: %d\n", err);

    free(in); free(out);
    vmaf_dnn_session_close(sess);
    return err < 0 ? 1 : 0;
}
```

Build:

```
cc filter.c -o filter $(pkg-config --cflags --libs libvmaf)
```

Only works when libvmaf was built with `-Denable_dnn=true`.

## Known limitations

- Operator allowlist covers the set required by tiny FR / NR / filter models
  shipped in `model/tiny/`; untrusted models with new op types will be
  rejected at `_open`. Extend the allowlist via the registry — see
  [ADR-0039](../adr/0039-onnx-runtime-op-walk-registry.md).
- EP selection is a preference, not a hard requirement: when a requested
  provider is missing from the linked ORT build, session open falls back to
  CPU. Call `vmaf_dnn_session_attached_ep()` and assert on the returned string
  if your application needs to fail on missing CUDA / OpenVINO / CoreML /
  ROCm.
- `VmafDnnConfig.fp16_io` only changes slots whose ONNX element type is
  `FLOAT16`; float32 model inputs and outputs stay float32. It is therefore
  harmless but not a speed switch for fp32-only graphs.
- There is no callback / progress hook; inference is synchronous per call.
- Sessions are heap-only; no stack-allocated variant.

## Related

- [ADR-0022](../adr/0022-inference-runtime-onnx.md) — choice of ONNX Runtime.
- [ADR-0023](../adr/0023-tinyai-user-surfaces.md) — where the CLI / C API / ffmpeg / training surfaces intersect.
- [ADR-0036](../adr/0036-tinyai-wave1-scope-expansion.md) — Wave 1 scope
  (LPIPS, saliency, per-shot CRF, `vmaf_post`, allowlist `Loop`/`If`, MCP VLM).
- [ADR-0039](../adr/0039-onnx-runtime-op-walk-registry.md) — operator
  allowlist + model registry.
- [ADR-0040](../adr/0040-dnn-session-multi-input-api.md) — multi-input/output
  named-binding API.
- [ADR-0041](../adr/0041-lpips-sq-extractor.md) — LPIPS-SqueezeNet extractor
  (consumer of this API).
- [ADR-0042](../adr/0042-tinyai-docs-required-per-pr.md) — tiny-AI doc
  specialisation.
- [../ai/inference.md](../ai/inference.md) — CLI-side tiny-AI walkthrough.
