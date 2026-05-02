# Tiny AI — inference

Three consumer surfaces share one runtime: `vmaf` CLI, libvmaf C API, and
ffmpeg filters. All three funnel through
[`libvmaf/src/dnn/ort_backend.c`](../../libvmaf/src/dnn/ort_backend.c).

## Prerequisites

- libvmaf built with `-Denable_dnn=enabled` (or `auto` with ONNX Runtime
  discoverable via `pkg-config`).
- ONNX Runtime ≥ 1.20 available at build time. ONNX Runtime isn't in the
  distro setup scripts under [scripts/setup/](../../scripts/setup/) yet —
  install the prebuilt release tarball from
  <https://github.com/microsoft/onnxruntime/releases> or a distro package
  if available, and make sure its `libonnxruntime.so` + headers are on
  `PKG_CONFIG_PATH` before running `meson setup`.
- A `.onnx` model + sidecar `.json` pair under `model/tiny/` or anywhere
  else — the CLI flag accepts an absolute path.

Verify at runtime:

```bash
vmaf --help | grep -- '--tiny-model'   # must list the flag
vmaf --tiny-model /missing.onnx 2>&1   # should print a clear error,
                                       # not "option not found"
```

## Surface 1 — the `vmaf` CLI

```bash
# C1 — drop-in augmentation of the classic SVM. Default tiny FR model
# is now vmaf_tiny_v2 (ADR-0216) — supersedes the prior vmaf_tiny_v1.
vmaf -r ref.yuv -d dis.yuv -w 1920 -h 1080 -p 420 -b 8 \
     -m version=vmaf_v0.6.1 \
     --tiny-model model/tiny/vmaf_tiny_v2.onnx \
     --tiny-device cuda

# C2 — no-reference.
vmaf -d dis.yuv -w 1920 -h 1080 -p 420 -b 8 \
     --tiny-model model/tiny/vmaf_nr_mobilenet_v1.onnx \
     --no-reference
```

> **Default flip (2026-04-29).** `vmaf_tiny_v2` replaces
> `vmaf_tiny_v1` as the recommended tiny FR fusion model. Same input
> contract (canonical-6 features), same output range (0–100 VMAF),
> +0.005–0.018 PLCC across the Phase-3 validation chain. The v1 file
> stays on disk as a regression baseline. See
> [`models/vmaf_tiny_v2.md`](models/vmaf_tiny_v2.md) for the full
> model card.
>
> **`vmaf_tiny_v3` available alongside v2 (2026-05-02, ADR-0241).**
> A wider/deeper variant (`mlp_medium` 6 → 32 → 16 → 1, ~769 params)
> trained on the same 4-corpus parquet, same recipe. Netflix LOSO
> mean PLCC 0.9986 ± 0.0015 vs v2's 0.9978 ± 0.0021 (+0.0008 mean,
> -30 % std). v2 remains the production default; pick v3 for
> lowest-variance estimates. See [`models/vmaf_tiny_v3.md`](models/vmaf_tiny_v3.md).

**Architecture ladder (2026-05-02).** The tiny VMAF fusion family
now spans three rungs sharing the canonical-6 input contract:

| Model | Arch | Params | ONNX | NF LOSO PLCC | Status |
| --- | --- | ---: | ---: | ---: | --- |
| [`vmaf_tiny_v2`](models/vmaf_tiny_v2.md) | mlp_small | 257 | 2.5 KB | 0.9978 ± 0.0021 | **Production default** |
| [`vmaf_tiny_v3`](models/vmaf_tiny_v3.md) | mlp_medium | 769 | 4.5 KB | 0.9986 ± 0.0015 | Opt-in (recommended higher tier) |
| [`vmaf_tiny_v4`](models/vmaf_tiny_v4.md) | mlp_large | 3073 | 14.0 KB | 0.9987 ± 0.0015 | Opt-in (top of measured ladder) |

v4's PLCC win over v3 is +0.0001 (below 1 std) — the ladder
saturates on the canonical-6 + 4-corpus regime. ADR-0242 records
"the arch ladder stops here". Pick v3 unless you specifically want
the absolute top of the measured ladder; pick v2 for the smallest
bundle.

New flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--tiny-model PATH` | — | ONNX model path; sidecar JSON at `${PATH%.onnx}.json`. |
| `--tiny-device STR` | `auto` | `auto` \| `cpu` \| `cuda` \| `openvino` \| `rocm`. |
| `--tiny-threads N` | `0` | CPU EP intra-op threads; 0 = ORT default. |
| `--tiny-fp16` | off | Request fp16 I/O when the EP supports it. |
| `--tiny-model-verify` | off | Require Sigstore-bundle verification (`cosign verify-blob`) before model load. Refuses to load on missing bundle, missing `cosign`, or non-zero exit. See [model-registry.md](model-registry.md) and [security.md](security.md). |
| `--no-reference` | off | Skip reference loading; only valid with an NR tiny model. |

Output JSON gains a `tiny_model` block alongside `pooled_metrics`:

```json
{
  "pooled_metrics": { "vmaf": { "mean": 91.23... } },
  "tiny_model": {
    "name": "vmaf_tiny_fr_v1",
    "kind": "fr",
    "device": "cuda",
    "mean": 90.8...,
    "per_frame": [...]
  }
}
```

## Surface 2 — the libvmaf C API

```c
#include <libvmaf/libvmaf.h>
#include <libvmaf/dnn.h>

VmafContext *ctx;
vmaf_init(&ctx, (VmafConfiguration){ /* ... */ });

if (!vmaf_dnn_available()) {
    fprintf(stderr, "libvmaf built without --enable_dnn; rebuild.\n");
    return 1;
}

VmafDnnConfig dnn_cfg = {
    .device       = VMAF_DNN_DEVICE_CUDA,
    .device_index = 0,
    .threads      = 0,
    .fp16_io      = false,
};
int err = vmaf_use_tiny_model(ctx, "/models/vmaf_tiny_fr_v1.onnx", &dnn_cfg);
if (err < 0) { /* handle -errno */ }

/* … feed frames as usual; tiny-model scores appear in the same
     per-frame collector the built-in SVM uses. */
```

The sidecar JSON is discovered automatically at
`${onnx_path%.onnx}.json`. Its `kind` field (`fr` / `nr`) tells libvmaf
whether to expect a reference.

## Surface 3 — ffmpeg filters

Apply `ffmpeg-patches/*.patch` against a pinned FFmpeg SHA (see
[`ffmpeg-patches/test/build-and-run.sh`](../../ffmpeg-patches/test/build-and-run.sh))
then:

```bash
# C1 / C2 scoring through vf_libvmaf.
ffmpeg -i ref.mp4 -i dis.mp4 \
    -lavfi "[0:v][1:v]libvmaf=tiny_model=/models/vmaf_tiny_fr_v1.onnx:tiny_device=cuda" \
    -f null -

# C3 learned pre-filter.
ffmpeg -i in.mp4 \
    -vf "vmaf_pre=model=/models/filter_denoise_residual_v1.onnx:device=cuda" \
    out.mp4
```

## Execution-provider matrix

| Backend flag | ORT EP | Notes |
| --- | --- | --- |
| `--tiny-device cpu` | CPUExecutionProvider | Always available. |
| `--tiny-device cuda` | CUDAExecutionProvider | Requires CUDA-enabled ORT; shares context with libvmaf-cuda. |
| `--tiny-device openvino` | OpenVINOExecutionProvider | Covers Intel GPU / SYCL / oneAPI. Tries GPU device type first, falls back to CPU device type. Also covers the integrated Xe / Xe2 GPU on Intel AI-PC platforms (Meteor / Lunar / Arrow Lake) for free; AI-PC *NPU* support is intentionally deferred — see [Research-0031](../research/0031-intel-ai-pc-applicability.md). |
| `--tiny-device rocm` | ROCmExecutionProvider | Requires ROCm-enabled ORT. |
| `--tiny-device auto` | best available | Ordered try-chain: CUDA → OpenVINO (GPU then CPU) → ROCm → CPU. |

### Graceful EP fallback

If the requested EP isn't compiled into the linked ORT build (for
example, you ask for `cuda` on a CPU-only ORT), the session still
opens — it silently degrades to the CPU EP rather than failing. This
matches `VmafDnnConfig.device` being documented as a *hint*, not a
requirement: a laptop and a workstation running the same binary get
the best EP each one has.

To see which EP actually bound, call
`vmaf_dnn_session_attached_ep()` on the session:

```c
VmafDnnSession *sess;
vmaf_dnn_session_open(&sess, "/models/m.onnx",
                      &(VmafDnnConfig){.device = VMAF_DNN_DEVICE_AUTO});
printf("bound EP: %s\n", vmaf_dnn_session_attached_ep(sess));
/* One of: "CPU", "CUDA", "OpenVINO:GPU", "OpenVINO:CPU", "ROCm" */
```

Consumers that need a hard failure on missing EP should assert on the
returned string at the call site (for example
`strcmp(ep, "CUDA") == 0`).

### fp16 I/O (`VmafDnnConfig.fp16_io`)

Setting `.fp16_io = true` enables a host-side fp32 ↔ fp16 round-trip
at the I/O boundary, triggered per input/output slot when the model's
graph declares that slot as `FLOAT16`. The public API always takes
fp32; libvmaf performs the cast internally. When the model declares
`FLOAT32` on a slot, `fp16_io = true` is a no-op at that slot. When
the EP is OpenVINO, the precision hint `FP16` is additionally passed
to the EP so intermediate compute also runs at half precision.

Example — running a FLOAT16-typed model:

```c
VmafDnnConfig cfg = {.device = VMAF_DNN_DEVICE_AUTO, .fp16_io = true};
VmafDnnSession *sess;
vmaf_dnn_session_open(&sess, "/models/m_fp16.onnx", &cfg);

float in[H*W] = { /* fp32 input */ };
float out[H*W];
VmafDnnInput  din = {.data = in,  .shape = (int64_t[4]){1,1,H,W}, .rank = 4};
VmafDnnOutput dout = {.data = out, .capacity = H*W};
vmaf_dnn_session_run(sess, &din, 1, &dout, 1);
```

## Expected cross-device variance

Running the same `.onnx` on two different EPs produces near-identical
scores:

- CPU vs CUDA (FP32): within **1e-4**.
- CPU vs CUDA (FP16 via `--tiny-fp16`): within **1e-2**.

CI exercises CPU-only; GPU parity is checked manually on the dev workstation
for now (planned: self-hosted runner).
