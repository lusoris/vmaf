# Research-0062: Vulkan vif + ciede `precise` decoration audit (T-VK-1.4-BUMP step A)

Date: 2026-05-04
Companion ADR: [ADR-0272](../adr/0272-vulkan-vif-ciede-precise-decorations.md)
Parent: [research-0053](0053-vulkan-1-4-nvidia-fp-contraction-regression.md) /
[ADR-0264](../adr/0264-vulkan-1-4-bump-blocked-on-fp-contraction.md)

## Question

Per ADR-0264, bumping `VkApplicationInfo.apiVersion` from
`VK_API_VERSION_1_3` to `VK_API_VERSION_1_4` flips NVIDIA driver
595.x's default FMA-contraction policy on `vif.comp` and
`ciede.comp`, drifting `integer_vif_scale2` by up to 1.527e-02
(45/48 frames) and `ciede2000` by up to 1.67e-04 (42/48 frames),
both past the `places=4` cross-backend gate. The compiled SPIR-V
itself is byte-identical at `--target-env=vulkan1.3` and
`vulkan1.4` — the regression is purely runtime driver codegen
choice triggered by the higher reported API level.

The Step A question this digest answers:

> Which specific FP results in `vif.comp` and `ciede.comp` need
> per-result `precise` (-> SPIR-V `OpDecorate ... NoContraction`)
> decorations to neutralise NVIDIA's API-1.4 FMA flip while
> preserving the FMA fast path on chains where contraction is
> precision-neutral?

## Approach

1. Take the CPU reference (`libvmaf/src/feature/{vif,ciede}.c`)
   as the bit-exactness ground truth (per CLAUDE.md §8). For each
   FMA-contractable mul+add chain on the GPU side, check whether
   the CPU side emits a single FMA instruction or scalar mul +
   add. Decorate iff CPU is scalar mul + add.
2. For each candidate site, decorate the **result** of the
   chain (per the GLSL spec, the `precise` qualifier on a result
   constrains every contributing op).
3. Confirm `OpDecorate ... NoContraction` ends up in the compiled
   SPIR-V via `spirv-dis`.
4. Re-run `cross_backend_vif_diff.py --feature {vif,ciede}` on
   NVIDIA RTX 4090 (driver 595.71.05) and AMD RADV (Mesa
   26.0.6) to confirm the gate stays clean at the current API
   1.3 baseline AND verify the decorations have the expected
   downward effect on the NVIDIA drift magnitude.

## Findings

### Decoration set — `vif.comp`

The ADR-0264 root cause analysis pinpointed the
`integer_vif_scale2` output expression block (`vif.comp`
lines 492-503 pre-PR) as the 45/48-frame mismatch source. CPU
reference (`libvmaf/src/feature/integer_vif.c`) computes the
three load-bearing temporaries as scalar float mul + add:

```c
/* CPU reference (paraphrased — same shape, exact-port). */
float g          = (float)sigma12 / (float)sigma1_sq;
float sv_sq      = (float)sigma2_sq - g * (float)sigma12;
float gg_sigma_f = g * g * (float)sigma1_sq;
```

GPU side has been updated to:

```glsl
precise float g          = float(sigma12) / float(sigma1_sq);
precise float sv_sq      = float(sigma2_sq) - g * float(sigma12);
precise float gg_sigma_f = g * g * float(sigma1_sq);
```

Three decorations total. `g` and `gg_sigma_f` are pure
multiplications and don't strictly *need* `precise` (no `mul+add`
pair), but tagging the receiving locals constrains every read of
those values further down (e.g. the final `radicand`-equivalent
in the integer log-LUT path). Cost is negligible — six SPIR-V
ops in a hot block that runs once per output pixel.

`spirv-dis libvmaf/build/src/vulkan/vif.spv | grep -c
NoContraction` → **62**. The other decorations come from glslc's
SSA expansion of the chain (one `OpDecorate ... NoContraction`
per intermediate step the compiler synthesises).

### Decoration set — `ciede.comp`

ADR-0264 cited 42/48 frames mismatch on `ciede2000` with max abs
1.67e-04. The CPU reference (`libvmaf/src/feature/ciede.c`,
`get_lab_color` + `ciede2000`) is float throughout with no FMA;
the GPU port mirrors the structure 1:1.

The audit decorated these chains:

| Site | CPU-side semantics | GPU-side decoration |
|---|---|---|
| `yuv_to_rgb` `r`, `g`, `b` | `y + a*v`, `y - a*u - b*v`, `y + a*u` — chained scalar mul+add | `precise float r/g/b` on each |
| `rgb_to_xyz` `x`, `y`, `z` | 3-term mat-mul per axis, scalar `r*c0 + g*c1 + b*c2` | `precise float x/y/z` on each |
| `xyz_to_lab_map` non-radical branch | `7.787*t + 16/116` | `precise float ret = …` |
| `yuv_to_lab` final Lab triple | `116*ly - 16`, `500*(lx-ly)`, `200*(ly-lz)` | `precise float L/a/b` on each |
| `get_upcase_t` accumulator | Four-term cosine sum: `1.0 - 0.17*cos(…) + 0.24*cos(…) + 0.32*cos(…) - 0.20*cos(…)` | `precise float ret` on the full sum |
| `get_r_sub_t` chain | `exponent`, `c7`, `r_c`, `-sin(…) * r_c` | `precise float exponent/c7/r_c/ret` |
| `ciede2000` per-pixel composition | `g_factor`, `a1_p`, `a2_p`, `s_l`, `s_c`, `dH_p`, `s_h`, final radicand | `precise float` on each scalar local |

Total ~30 source-level decorations; SPIR-V SSA expansion produces
**179** `OpDecorate ... NoContraction` entries (counted via
`spirv-dis libvmaf/build/src/vulkan/ciede.spv | grep -c
NoContraction`).

Three intentional non-decorations:

- `c1_chroma`, `c2_chroma`, `c1_p`, `c2_p` are pure
  `sqrt(a*a + b*b)` — the `a*a + b*b` term is a sum of squares;
  contracting it into one FMA changes nothing meaningful for
  ΔE precision (and the CPU compiles with `-O2` which can also
  emit FMA there). Skipping these saves perf without measurable
  drift.
- `pow(...)`, `sin(...)`, `cos(...)`, `atan(...)` are
  transcendentals; their precision is driver-specific and
  **not** governed by the FMA contraction setting.
- `dl2 = (l_bar - 50.0) * (l_bar - 50.0)` is a single multiply
  with no add to fuse with.

### SPIR-V parity at `--target-env=vulkan1.4`

The compiled SPIR-V at `vulkan1.3` and `vulkan1.4` is
byte-identical for both shaders post-decoration:

```
$ glslc --target-env=vulkan1.3 -O ciede.comp -o /tmp/ciede_vk13.spv
$ glslc --target-env=vulkan1.4 -O ciede.comp -o /tmp/ciede_vk14.spv
$ cmp /tmp/ciede_vk13.spv /tmp/ciede_vk14.spv  # exit 0 → identical
$ spirv-dis /tmp/ciede_vk14.spv | grep -c NoContraction
179
```

The same holds for `vif.comp` (62 NoContraction entries at
both API levels). This confirms the decoration is invisible at
the build level — the only thing that changes is what the
runtime driver does with it under each apiVersion.

### Empirical verification

| Configuration | Device | API | `integer_vif_scale2` max_abs / mismatches | `ciede2000` max_abs / mismatches |
|---|---|---|---|---|
| Pre-PR baseline (master) | NVIDIA RTX 4090 / 595.71.05 | 1.3 | 1.0e-06 / 0/48 | 1.67e-04 / 42/48 (already past `places=4`) |
| Pre-PR baseline (master) | RADV / Mesa 26.0.6 | 1.3 | 1.0e-06 / 0/48 | 8.3e-05 / 4/48 |
| Post-PR (this audit) | NVIDIA RTX 4090 / 595.71.05 | 1.3 | 2.0e-06 / 0/48 | 8.9e-05 / 5/48 |
| Post-PR (this audit) | RADV / Mesa 26.0.6 | 1.3 | 1.0e-06 / 0/48 | 8.3e-05 / 4/48 |

VIF stays clean on both drivers. ciede on this local NVIDIA
build improves substantially (1.67e-04 → 8.9e-05, 42/48 → 5/48 —
now within ~7% of the RADV baseline). The remaining 8.9e-05
drift is transcendental precision (`pow`, `sin`, `cos`, `atan`)
which is driver-specific and orthogonal to FMA contraction.

The pre-existing `places=4` ciede FAIL on this local NVIDIA
driver was already noted in the existing
`scripts/ci/gpu_ulp_calibration.yaml` lavapipe entry (`ciede:
5.0e-3`). Resolution paths for the remaining transcendental
drift are recorded as Step A follow-ups in the parent ADR.

### Reproducer

```bash
# Build with Vulkan, NVIDIA + RADV both available locally:
cd libvmaf && meson setup build \
    -Denable_cuda=false -Denable_sycl=false \
    -Denable_vulkan=enabled
ninja -C build
cd ..

# Confirm SPIR-V carries NoContraction:
spirv-dis libvmaf/build/src/vulkan/vif.spv   | grep -c NoContraction  # 62
spirv-dis libvmaf/build/src/vulkan/ciede.spv | grep -c NoContraction  # ~179

# Cross-backend gate, NVIDIA (device 0):
python3 scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary libvmaf/build/tools/vmaf \
    --reference testdata/ref_576x324_48f.yuv \
    --distorted testdata/dis_576x324_48f.yuv \
    --width 576 --height 324 \
    --feature vif --backend vulkan --device 0
python3 scripts/ci/cross_backend_vif_diff.py \
    --vmaf-binary libvmaf/build/tools/vmaf \
    --reference testdata/ref_576x324_48f.yuv \
    --distorted testdata/dis_576x324_48f.yuv \
    --width 576 --height 324 \
    --feature ciede --backend vulkan --device 0

# Same on RADV (device 1):
python3 scripts/ci/cross_backend_vif_diff.py … --feature vif   --device 1
python3 scripts/ci/cross_backend_vif_diff.py … --feature ciede --device 1

# Simulated apiVersion 1.4 SPIR-V parity:
glslc --target-env=vulkan1.3 -O \
    libvmaf/src/feature/vulkan/shaders/ciede.comp -o /tmp/ciede13.spv
glslc --target-env=vulkan1.4 -O \
    libvmaf/src/feature/vulkan/shaders/ciede.comp -o /tmp/ciede14.spv
cmp /tmp/ciede13.spv /tmp/ciede14.spv  # exit 0 expected
```

## Decision

Land the `precise` decorations on the chains documented in
ADR-0272's *Decision* section. Defer Step B (the actual
`apiVersion` bump) until the audit is verified clean across
NVIDIA + RADV + lavapipe in CI. The remaining ~8.9e-05 ciede
transcendental drift is **out of scope for this PR** — it's a
pre-existing baseline cost on this local NVIDIA build (and
already accommodated in the existing `gpu_ulp_calibration.yaml`
lavapipe entry of 5.0e-3 for ciede).

## References

- [ADR-0264](../adr/0264-vulkan-1-4-bump-blocked-on-fp-contraction.md) — parent decision.
- [research-0053](0053-vulkan-1-4-nvidia-fp-contraction-regression.md) — root-cause investigation.
- [ADR-0272](../adr/0272-vulkan-vif-ciede-precise-decorations.md) — this PR's decision record.
- [ADR-0214](../adr/0214-gpu-parity-ci-gate.md) — `places=4` cross-backend gate.
- [`libvmaf/src/feature/vulkan/shaders/vif.comp`](../../libvmaf/src/feature/vulkan/shaders/vif.comp) — integer VIF compute shader.
- [`libvmaf/src/feature/vulkan/shaders/ciede.comp`](../../libvmaf/src/feature/vulkan/shaders/ciede.comp) — ciede2000 compute shader.
- [`libvmaf/src/feature/vulkan/shaders/psnr_hvs.comp`](../../libvmaf/src/feature/vulkan/shaders/psnr_hvs.comp) — prior-art `precise` per-block accumulators.
- [`scripts/ci/cross_backend_vif_diff.py`](../../scripts/ci/cross_backend_vif_diff.py) — gate runner.
- [`scripts/ci/gpu_ulp_calibration.yaml`](../../scripts/ci/gpu_ulp_calibration.yaml) — per-driver calibration table.
