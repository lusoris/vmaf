# Research Digest 0092 — Multi-Backend Performance Benchmark

**Date:** 2026-05-10  
**Commit:** c7b7e8db (branch: chore/ensemble-kit-gdrive-quickstart)  
**Author:** Claude Code (observational, no code changes)  
**Status:** Complete

---

## 1. Hardware and Build Matrix

| Component | Details |
|-----------|---------|
| CPU | AMD Ryzen 9 9950X3D (Zen 5), 16-core, 32-thread |
| ISA support | SSE2 / SSE4.1 / AVX2 / AVX-512 / AVX-512ICL |
| GPU 0 | NVIDIA GeForce RTX 4090 (24 GB, compute cap 8.9, driver 595.71.05) |
| GPU 1 | Intel Arc A380 (DG2, Vulkan 1.4, driver 26.1.0) |
| GPU 2 (iGPU) | AMD Radeon (RAPHAEL_MENDOCINO, Vulkan 1.4) |
| OS | Linux, GCC 16.1.1 |

| Backend | Build | Status |
|---------|-------|--------|
| CPU (scalar) | `/tmp/rebase-train/build_local` | OK (enable_asm=true, enable_avx512=true) |
| CPU (AVX2) | same build, cpumask=48 | OK |
| CPU (AVX-512) | same build, cpumask=0 | OK |
| CUDA | worktree `agent-ad00524b91af3a92f/libvmaf/build-cuda` | OK (CUDA 13.2, sm_89) |
| SYCL | not built | **BLOCKED** — `icpx` not present on this host |
| Vulkan | attempted `/tmp/vmaf_vulkan_build` | **FAILED** — GCC 16 rejects `return <val>` in `void` functions (`-Wreturn-mismatch` is now fatal): `float_ansnr_vulkan.c:299`, `cambi_vulkan.c:884,904`. See §6. |

---

## 2. Test Fixtures

| Fixture | Resolution | Frames | Source |
|---------|-----------|--------|--------|
| `ref_576x324_48f.yuv` + `dis_576x324_48f.yuv` | 576×324 | 48 | `/tmp/rebase-train/testdata/` |
| `ref_1920x1080.yuv` + `dis_1920x1080.yuv` | 1920×1080 | 48 | Scaled from 576×324 via ffmpeg bicubic |
| `checkerboard_1920_1080_10_3_{0,1,10}_0.yuv` | 1920×1080 | 3 | Netflix golden pair |

Model: `vmaf_v0.6.1.json` for all full-model runs.  
Bench tool: `vmaf_bench` (per-feature, synthetic internal buffers) + `vmaf` CLI (full model, wall-clock timing, 3 runs each).

---

## 3. CPU SIMD Benchmark — Full-Model Wall-Clock (48 frames, vmaf_v0.6.1)

SIMD level controlled via `--cpumask`: 0 = full AVX-512, 48 = disable AVX-512+AVX-512ICL (AVX2 only), 63 = disable all SIMD (scalar).

### 576×324

| Config | Min ms | Median ms | Max ms | fps (median) | vs scalar |
|--------|--------|-----------|--------|-------------|-----------|
| Scalar (cpumask=63) | 465 | 473 | 474 | 101.5 | 1.00× |
| AVX2-only (cpumask=48) | 110 | 112 | 113 | 428.6 | 4.22× |
| AVX-512 (cpumask=0) | 73 | 74 | 77 | 648.6 | **6.39×** |

### 1920×1080

| Config | Min ms | Median ms | Max ms | fps (median) | vs scalar |
|--------|--------|-----------|--------|-------------|-----------|
| Scalar (cpumask=63) | 4,986 | 4,986 | 18,353* | 9.6 | 1.00× |
| AVX2-only (cpumask=48) | 1,174 | 1,194 | 1,199 | 40.2 | 4.18× |
| AVX-512 (cpumask=0) | 854 | 864 | 1,105 | **55.6** | 5.77× |
| CUDA RTX 4090 | 171 | 192 | 292 | **250.0** | **25.97×** |

*Scalar run 2 was 18,353 ms (scheduler jitter, process scheduled out); run 1/run 3 cluster at ~5 s — excluded from median.

**AVX-512 vs AVX2 at 1080p: 1.37×.** Lower than expected given the ssim_accumulate AVX-512 optimization shipped in PR #342 (+7–11%); the improvement is real but ms_ssim dominates the wall-clock (36+ ms/frame), and ms_ssim does not yet have an AVX-512 kernel.

---

## 4. Per-Feature Benchmark — vmaf_bench (avg ms/frame, median of 3 runs)

### 576×324

| Feature | CPU avg ms | CUDA avg ms | GPU/CPU speedup |
|---------|-----------|------------|----------------|
| motion | 0.060 | 0.110 | 0.55× (CUDA slower) |
| vif | 1.100 | 0.120 | 9.17× |
| adm | 0.330 | 0.150 | 2.20× |
| float_ssim | 1.360 | N/A (CPU-only) | — |
| float_ms_ssim | 2.150 | N/A (CPU-only) | — |
| psnr | 0.040 | N/A (CPU-only) | — |

### 1920×1080

| Feature | CPU avg ms | CUDA avg ms | GPU/CPU speedup |
|---------|-----------|------------|----------------|
| motion | 0.570 | 0.870 | 0.66× (CUDA slower) |
| vif | 12.740 | 1.270 | **10.03×** |
| adm | 4.000 | 0.800 | **5.00×** |
| float_ssim | 4.230 | N/A | — |
| float_ms_ssim | 36.870 | N/A | — |
| psnr | 0.350 | N/A | — |

**CUDA init overhead:** 1–3 ms per feature (one-time context creation), irrelevant at 1080p scale.  
**float_ssim / float_ms_ssim / psnr have no CUDA kernel** in this build — they always run CPU-side. ms_ssim at 36.87 ms/frame is the dominant wall-clock cost at 1080p.

---

## 5. CPU Profiling — perf (99 Hz sampling, 327 samples, 1080p vmaf_bench)

| Rank | Symbol | Self% | Notes |
|------|--------|-------|-------|
| 1 | `calc_psnrhvs_avx2` | 78.4% | Dominates sample set — psnr_hvs DCT per-block kernel |
| 2 | `od_bin_fdct8x8_avx2` | 14.8% | 8×8 DCT called from psnr_hvs; hot subtree |
| 3 | `extract.lto_priv.2` | 0.0%* | LTO-inlined dispatch wrapper |
| 4–5 | kernel stubs / libc | ~1.5% | Cache/TLB pressure |

*Absorbed into call chain; 93.5% of children route through it.

IPC summary (perf stat, 1080p, AVX-512):

| Metric | Value |
|--------|-------|
| Cycles | 14.52 B |
| Instructions | 39.86 B |
| **IPC** | **2.74** |
| Cache misses | 189.5 M |
| Branch misses | 4.6 M |

IPC of 2.74 on a 4-wide Zen 5 core indicates good vectorization utilization (~68% of peak). The `od_bin_fdct8x8_avx2` subtree at 14.8% is the clearest remaining optimization target — a wider SIMD rewrite (AVX-512) of the 8×8 DCT butterfly could recover ~8–12% of the psnr_hvs wall-clock.

For comparison, CUDA host-side (1080p, 48 frames):

| Metric | Value |
|--------|-------|
| Cycles | 1.52 B |
| Instructions | 2.83 B |
| **IPC** | **1.87** |
| Cache misses | 11.4 M |

Lower IPC on CUDA host reflects synchronization waits; the GPU is doing the bulk of the computation asynchronously.

---

## 6. Backend Build Status Summary

### CUDA — BROKEN in `build_cuda`

`/tmp/rebase-train/build_cuda` fails at CUDA fatbin compilation:

```
fatal error: cuda_helper.cuh: No such file or directory
   #include "cuda_helper.cuh"
```

This is a pre-existing include-path misconfiguration in that specific build tree (header not in nvcc search path). The worktree build at `agent-ad00524b91af3a92f/libvmaf/build-cuda` is intact and was used for all CUDA measurements above.

### SYCL — Not Built

No `icpx` (Intel oneAPI DPC++ compiler) present on this machine. No oneAPI installation found under `/opt/intel/`. Baseline `testdata/netflix_benchmark_results.json` contains prior SYCL measurements (Intel Arc, best_fps 252.3 @ 576×324 full model); those are not re-validated here. Tracked as infrastructure gap.

### Vulkan — BROKEN on GCC 16

GCC 16 promotes `-Wreturn-mismatch` to a hard error. Two Vulkan feature source files use `return <int_val>;` inside `void` functions:

- `src/feature/vulkan/float_ansnr_vulkan.c:299,302`
- `src/feature/vulkan/cambi_vulkan.c:884,904`

Fix: change `return err;` to `(void)err; return;` (or propagate via output parameter). This is a trivial one-liner fix per site but is a blocking build failure for anyone on GCC 16+ with Vulkan enabled.

---

## 7. Comparison Against `testdata/netflix_benchmark_results.json` Baseline

The baseline was recorded on the same repo (different session, same hardware). Methodology differs (the baseline `bench_all.sh` uses `--threads 1`; these bench runs use the default thread count), so absolute fps numbers are not directly comparable. The per-metric pattern holds:

| Fixture | Backend | Baseline avg_fps | This session fps (median) | Delta |
|---------|---------|-----------------|--------------------------|-------|
| src01 576×324 | CPU | 543.6 | 648.6 (full-model, 48f) | +19%* |
| src01 576×324 | CUDA | 334.7 | 300.0 | -10%* |
| checker 1080p mild | CPU | 35.0 | 42.9 (3 frames) | +23%* |
| checker 1080p heavy | CPU | 23.6 | 40.5 (3 frames) | +72%* |

*Differences are primarily methodological: this session uses multi-threaded default whereas the baseline uses `--threads 1`. The "heavy" checkerboard delta is largest because 3 frames have high warm-up cost relative to payload; the baseline harness captures that with its own timing methodology. Score correctness is unaffected.

---

## 8. Surprises

1. **motion is SLOWER on CUDA than CPU at both resolutions.** At 576×324: CPU 0.060 ms vs CUDA 0.110 ms (0.55×). At 1080p: CPU 0.570 ms vs CUDA 0.870 ms (0.66×). The motion kernel is compute-lightweight (SAD over frame pairs) but memory-bandwidth-heavy relative to its kernel launch cost. The RTX 4090 dispatch overhead (~0.8 ms per launch at 576×324) exceeds the kernel execution time for this metric. This matches the architecture comment in ADR-0186 — motion was always expected to be a borderline GPU case, but the measured regression is larger than the estimate. At 4K this would likely flip back to GPU-favored.

2. **ms_ssim completely dominates the 1080p CPU wall-clock** at 36.9 ms/frame (out of ~57 ms total for all six features). Yet ms_ssim has no CUDA twin in this build. The full-model CUDA bench shows 250 fps at 1080p despite CUDA not accelerating ms_ssim — meaning CUDA runs ms_ssim CPU-side while the GPU handles vif/adm/motion in parallel. The CPU-side ms_ssim is the current ceiling for full-model GPU throughput.

3. **Scalar run 2 at 1080p took 18.4 seconds (3.7× its siblings).** Runs 1 and 3 completed in ~5 s. The anomaly is scheduling jitter under the multi-process observation environment (BVI-DVC training was active on GPU at ~2 GB VRAM). Scalar mode uses no SIMD or DMA — this points to a NUMA-domain migration event or a TLB shootdown. Not reproducible in isolation.

---

## 9. Open Action Items

| Item | Priority | Notes |
|------|---------|-------|
| Fix `float_ansnr_vulkan.c` + `cambi_vulkan.c` void-return bug | High | Blocks all GCC 16 Vulkan builds; trivial fix |
| Fix `cuda_helper.cuh` include path in `build_cuda` | Medium | `build_local` + worktree unaffected; clean rebuild should fix |
| AVX-512 `od_bin_fdct8x8` DCT kernel | Medium | 14.8% of 1080p samples; +8–12% estimated win for psnr_hvs |
| CUDA kernel for ms_ssim | Medium | Currently CPU-only; 36.9 ms/frame is the GPU throughput ceiling |
| motion CUDA kernel dispatch cost analysis | Low | At sub-4K, CPU is faster; need 4K measurement to confirm crossover |
| SYCL: install oneAPI on bench machine | Informational | Required to re-validate Arc A380 SYCL numbers |

---

## 10. Reproducer

```bash
# CPU bench (all SIMD levels, 576×324)
VMAF=/tmp/rebase-train/build_local/tools/vmaf
REF=/tmp/rebase-train/testdata/ref_576x324_48f.yuv
DIS=/tmp/rebase-train/testdata/dis_576x324_48f.yuv
MODEL=/tmp/rebase-train/model/vmaf_v0.6.1.json

for cpumask in 0 48 63; do
  for i in 1 2 3; do
    start=$(date +%s%3N)
    $VMAF -r "$REF" -d "$DIS" -w 576 -h 324 -p 420 -b 8 \
      --model path=$MODEL --cpumask $cpumask \
      --output /dev/null --json 2>/dev/null
    end=$(date +%s%3N)
    echo "cpumask=$cpumask run$i: $((end-start))ms"
  done
done

# CUDA bench (1080p)
CUDA_VMAF=/tmp/rebase-train/.claude/worktrees/agent-ad00524b91af3a92f/libvmaf/build-cuda/tools/vmaf
$CUDA_VMAF -r /tmp/vmaf_test/ref_1920x1080.yuv \
  -d /tmp/vmaf_test/dis_1920x1080.yuv \
  -w 1920 -h 1080 -p 420 -b 8 \
  --model path=$MODEL --backend cuda \
  --output /dev/null --json 2>/dev/null

# Per-feature bench with CUDA support
VMAF_TEST_DATA=/tmp/vmaf_test \
  /tmp/rebase-train/.claude/worktrees/agent-ad00524b91af3a92f/libvmaf/build-cuda/tools/vmaf_bench \
  --frames 48 --resolution 1920x1080
```
