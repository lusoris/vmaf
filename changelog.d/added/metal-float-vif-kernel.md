## Metal float_vif kernel (T8-1k / ADR-0445)

Added `float_vif_metal` — a 4-scale VIF feature extractor on the Metal
(Apple Silicon) backend. The kernel emits four per-scale scores
(`VMAF_feature_vif_scale{0,1,2,3}_score`) consumed by the VMAF v2 model,
closing the largest remaining gap in the Metal pipeline.

Implementation: two MSL kernel functions in `float_vif.metal`:

- `float_vif_compute` — separable Gaussian filter (V→H) + VIF statistic
  + per-threadgroup (num, den) reduction. One dispatch per scale.
- `float_vif_decimate` — downsamples the current scale to produce the
  half-resolution input for the next scale. Three dispatches per frame.

Total: 7 Metal dispatches per frame (4 compute + 3 decimate).
Border handling: mirror padding (`VIF_OPT_HANDLE_BORDERS`), matching the
CPU, CUDA, and Vulkan reference paths. kernelscale=1.0 (v1).

Cross-backend ULP gate: `places=4` vs CPU reference (ADR-0214), validated
by macOS CI lane `Build — macOS Metal (T8-1 scaffold)`.

macOS smoke test:

```bash
meson setup build -Denable_metal=enabled
ninja -C build
meson test -C build test_metal_smoke
```

Cross-backend diff (requires macOS with Apple Silicon):

```bash
python3 scripts/ci/cross_backend_diff.py \
  --ref testdata/yuv/src01_hrc00_576x324.yuv \
  --dis testdata/yuv/src01_hrc01_576x324.yuv \
  --width 576 --height 324 --fps 25 \
  --model model/vmaf_v0.6.1.json \
  --feature float_vif_metal --places 4
```

Local validation skipped: Linux host cannot compile or run Metal.
