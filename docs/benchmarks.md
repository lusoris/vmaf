# Benchmarks

> **Scope:** this file tracks *fork-added* benchmarks (GPU backends, SIMD
> paths, `--precision` overhead). Netflix's upstream correctness numbers
> are the Netflix golden CPU pools — see [CLAUDE.md §8](../CLAUDE.md).

Runs below are produced by `make bench` (drives `testdata/bench_all.sh`) on
a fixed hardware profile and pinned commit. Contribute new numbers via a
PR that updates this file alongside the commit that motivates the rerun.

## Hardware profiles

| Profile | CPU | GPU | Memory | OS |
|---------|-----|-----|--------|-----|
| `ryzen-4090` | AMD Ryzen 9 7950X (16c/32t) | NVIDIA RTX 4090 | 64 GB DDR5-6000 | Linux 6.19 (CachyOS) |
| `xeon-arc` | Intel Xeon w9-3475X | Intel Arc A770 | 128 GB DDR5-4800 | Ubuntu 24.04 |
| `m4-pro` | Apple M4 Pro | (integrated) | 48 GB unified | macOS 15 |

## Backend comparison (Netflix normal pair, 576×324)

Source: `python/test/resource/yuv/src01_hrc00_576x324.yuv` vs `…hrc01…`.
Precision: `--precision=max` (IEEE-754 round-trip lossless `%.17g`; the
CLI default is `%.6f` per
[ADR-0119](adr/0119-cli-precision-default-revert.md), so cross-backend
diffs must opt in explicitly).

| Backend | fps (higher better) | vmaf pool | delta vs CPU |
|---------|---------------------|-----------|--------------|
| `cpu` (scalar) | `TBD` | `76.66890482443686` | 0 |
| `cpu` + AVX2 | `TBD` | `76.66890482443686` | 0 ULP |
| `cpu` + AVX-512 | `TBD` | `76.66890482443686` | 0 ULP |
| `cuda` (RTX 4090) | `TBD` | `TBD` | ≤ 2 ULP |
| `sycl` (Arc A770) | `TBD` | `TBD` | ≤ 2 ULP |

Fill in `TBD` via a benchmark rerun on the `ryzen-4090` profile and
commit a PR with the updated table.

## `--precision` overhead

String formatting is not on the hot path; switching from `%.6f` (default)
to `%.17g` (`--precision=max`) changes only the output stage.

| `--precision` | output stage ns/frame | output size (JSON) |
|---------------|-----------------------|--------------------|
| no flag (`%.6f`) | `TBD` | baseline |
| `6` | `TBD` | ~baseline |
| `max` (`%.17g`) | `TBD` | +~20 % chars |

## How to reproduce

```bash
make bench PROFILE=ryzen-4090
# or manually:
build/tools/vmaf -r ref.yuv -d dis.yuv \
    -w 576 -h 324 -p 420 -b 8 \
    --precision=max --output=bench.json
```

Then append a row with hardware + commit SHA + measured pool to the
table above.
