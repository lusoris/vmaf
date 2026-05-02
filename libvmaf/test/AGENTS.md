# AGENTS.md — libvmaf/test

Orientation for agents working on the C unit test suite. Parent:
[../AGENTS.md](../AGENTS.md).

## Scope

C unit tests for the libvmaf engine. Runs on every build via
`meson test -C build`. A separate suite under
[dnn/](dnn/) covers the ONNX Runtime integration.

## Test style

All tests follow a trivial µnit-style pattern declared in
[test.h](test.h):

```c
static char *test_some_invariant(void)
{
    mu_assert("description", predicate);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_some_invariant);
    return NULL;
}
```

Each `test_*.c` compiles into its own binary. `meson.build` registers them
with `meson test`. No fixtures, no shared state — each test owns its setup
and teardown.

## Ground rules

- **Parent rules** apply (see [../AGENTS.md](../AGENTS.md)).
- **POSIX-only APIs in tests** must be shimmed for MINGW. See
  [test_lpips.c](test_lpips.c) for the `_putenv_s`-based shim for
  `setenv`/`unsetenv` — MinGW's mingw.org / MSYS2 headers do not expose
  those functions under `-std=c11 -pedantic`. The CI MINGW build will catch
  this but running `meson test` locally on Linux won't.
- **Never modify Netflix golden assertions**: those are Python-side, not
  here — see [../../python/test/](../../python/test/) and
  [ADR-0024](../../docs/adr/0024-netflix-golden-preserved.md).
- **New extractor → new test file** following the `test_lpips.c` pattern:
  (a) registered by name, (b) registered by provided feature name,
  (c) options table well-formed, (d) init rejects missing required input.
- **New SIMD parity test → use [`simd_bitexact_test.h`](simd_bitexact_test.h)**
  (ADR-0221). The shared harness centralises the `xorshift32` PRNG,
  the portable POSIX/MinGW/MSVC aligned allocator, the x86 AVX2 CPUID
  gate, and the `SIMD_BITEXACT_ASSERT_MEMCMP` /
  `SIMD_BITEXACT_ASSERT_RELATIVE` assertion macros. Do not re-implement
  these inline. **Include-order invariant**: `#include "test.h"` MUST
  precede `#include "simd_bitexact_test.h"` because `test.h` has no
  header guard and would redefine the `mu_report` `static inline` on a
  double include. Existing migrated tests
  (`test_psnr_hvs_avx2.c`, `test_psnr_hvs_neon.c`, `test_moment_simd.c`,
  `test_motion_v2_simd.c`) are reference templates;
  `test_ssimulacra2_simd.c` is an intentional non-migrated example
  (its `fill_random` FP rounding order is load-bearing for input bit
  patterns).

## Governing ADRs

- [ADR-0015](../../docs/adr/0015-ci-matrix-asan-ubsan-tsan.md) — sanitizer
  matrix (tests run under ASan + UBSan).
- [ADR-0024](../../docs/adr/0024-netflix-golden-preserved.md) — Netflix
  goldens (Python-side) never change.
- [ADR-0221](../../docs/adr/0221-simd-bitexact-test-harness.md) — SIMD
  bit-exact test harness shared header (`simd_bitexact_test.h`).
