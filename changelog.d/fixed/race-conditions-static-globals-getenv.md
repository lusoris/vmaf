Fixed six C11 data races in `log.c`, `cpu.c`, `gpu_picture_pool.c`, `model_loader.c`,
and `sycl/common.cpp`:

- `vmaf_log_level` and `istty` in `log.c` converted to `_Atomic(int)` with
  `memory_order_relaxed` stores/loads — eliminates the TSan race between
  `vmaf_set_log_level()` and `vmaf_log()`.
- `flags` and `flags_mask` in `cpu.c` converted to `_Atomic(unsigned)` — eliminates
  the race between `vmaf_init_cpu()` / `vmaf_set_cpu_flags_mask()` and
  `vmaf_get_cpu_flags()`.
- NVTX diagnostic counter `glob` in `gpu_picture_pool.c` converted to
  `_Atomic(unsigned)` with `atomic_fetch_add_explicit` — eliminates the race in
  concurrent `vmaf_gpu_picture_pool_fetch()` calls.
- `getenv("VMAF_TINY_MODEL_DIR")` and `getenv("PATH")` in `model_loader.c` cached
  via `pthread_once`-protected statics (matching the PR #864 pattern in
  `cuda/dispatch_strategy.c`).
- `getenv("VMAF_SYCL_PROFILE")` and `getenv("VMAF_SYCL_TIMING")` in
  `sycl/common.cpp` cached via `std::once_flag` / `std::call_once`.

Zero observable behaviour change on single-threaded paths. See ADR-0453.
