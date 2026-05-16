# ADR-0453: Race-safe static globals and `getenv` caching for log, cpu, GPU pool, and DNN modules

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: concurrency, security, threading, dnn, cuda, sycl

## Context

A threading audit (`.workingdir/audit-memory-safety-threading-2026-05-16.md`) identified
six data-race sites in the same class as the PR #864 `dispatch_strategy.c` fix:

1. `libvmaf/src/log.c` — `vmaf_log_level` and `istty` plain `static` globals written by
   `vmaf_set_log_level()` and read by `vmaf_log()` from arbitrary threads.
2. `libvmaf/src/cpu.c` — `flags` and `flags_mask` plain `static` globals written by
   `vmaf_init_cpu()` / `vmaf_set_cpu_flags_mask()` and read by `vmaf_get_cpu_flags()`.
3. `libvmaf/src/gpu_picture_pool.c` — NVTX diagnostic counter (`static unsigned glob`)
   incremented non-atomically from concurrent `vmaf_gpu_picture_pool_fetch()` calls.
4. `libvmaf/src/dnn/model_loader.c:335` — `getenv("VMAF_TINY_MODEL_DIR")` called
   directly in `vmaf_dnn_validate_onnx()`.
5. `libvmaf/src/dnn/model_loader.c:633` — `getenv("PATH")` called directly in
   `vmaf_dnn_verify_signature()`.
6. `libvmaf/src/sycl/common.cpp:228,249` — `getenv("VMAF_SYCL_PROFILE")` and
   `getenv("VMAF_SYCL_TIMING")` called directly in `vmaf_sycl_state_init()`.

`getenv()` is not required to be thread-safe by C99 / POSIX.1-2008 §2.2.2 when another
thread concurrently calls `setenv`/`putenv`/`unsetenv`. Non-atomic reads/writes to plain
`static` module globals across threads are C11 data races (undefined behaviour).

The pattern is identical to PR #864's fix in `cuda/dispatch_strategy.c`.

## Decision

Apply two complementary fixes, each zero-overhead on x86-64:

- **Module-level mutable static globals** (`vmaf_log_level`, `istty`, `flags`,
  `flags_mask`): convert to `_Atomic` with `atomic_store_explicit` on writes and
  `atomic_load_explicit` on reads, both with `memory_order_relaxed`. The relaxed
  ordering is correct: these are independent hint fields with no cross-memory
  synchronisation requirement.

- **NVTX diagnostic counter** (`glob`): convert to `_Atomic(unsigned)` with
  `atomic_fetch_add_explicit(..., memory_order_relaxed)`.

- **`getenv()` calls in `model_loader.c`** (`VMAF_TINY_MODEL_DIR`, `PATH`): cache via
  `pthread_once`-protected static, matching the PR #864 pattern exactly. The cached
  value is a `strdup`'d copy so it remains stable even if the caller later calls
  `unsetenv`. A Windows-compatible `INIT_ONCE` path is also provided.

- **`getenv()` calls in `sycl/common.cpp`** (`VMAF_SYCL_PROFILE`, `VMAF_SYCL_TIMING`):
  cache via `std::once_flag` / `std::call_once` (the C++ equivalent of `pthread_once`),
  matching the same intent.

- **`getenv()` in `tiny_extractor_template.h`** (`vmaf_tiny_ai_resolve_model_path`):
  the variable name is dynamic (passed as a parameter), making a per-variable static
  cache inapplicable. This function is called only during extractor init, which the
  VMAF init sequence (`vmaf_use_features_from_model` / `vmaf_init`) serialises to a
  single thread. The thread-safety contract is documented in the function's doc-comment
  and file header, and the `getenv()` call is annotated with `NOLINT(concurrency-mt-unsafe)`
  citing this ADR.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| `pthread_mutex_t` around each global | Portable, explicit | Runtime overhead in hot `vmaf_log()` path; contention on every log call | `_Atomic` relaxed is strictly cheaper and sufficient |
| `pthread_rwlock_t` for read-heavy globals | Allows multiple concurrent readers | More complex; still overhead; overkill for int-sized scalars | `_Atomic` relaxed is sufficient and portable via C11 |
| Do nothing (caller contract) | No code change | C11 UB; TSan detects; future callers will race | Unacceptable; bug is real |

## Consequences

- **Positive**: C11 data races eliminated in `log.c`, `cpu.c`, `gpu_picture_pool.c`,
  `model_loader.c`, and `sycl/common.cpp`. ThreadSanitizer (TSan) will no longer report
  these sites. Zero observable behaviour change on single-threaded paths. Zero overhead
  on x86-64 for the `_Atomic` paths.
- **Negative**: `<stdatomic.h>` is now included in `log.c`, `cpu.c`, and
  `gpu_picture_pool.c`. `<pthread.h>` is now included in `model_loader.c` on POSIX.
  All are standard headers already available on every supported platform.
- **Neutral / follow-ups**: The `tiny_extractor_template.h` `getenv()` remains with a
  documented single-threaded-init contract. If tiny-AI extractors are ever registered
  concurrently, the site must be revisited.

## References

- PR #864 — `cuda/dispatch_strategy.c` `pthread_once` pattern (the reference fix).
- POSIX.1-2008 §2.2.2 — Thread-safety of `getenv` / `setenv`.
- SEI CERT C ENV34-C — do not store pointers returned by `getenv`.
- SEI CERT C CON43-C — do not assume `pthread_create` establishes happens-before on
  data that was not synchronised.
- `.workingdir/audit-memory-safety-threading-2026-05-16.md` — source audit.
