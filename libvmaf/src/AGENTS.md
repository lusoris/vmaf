# AGENTS.md — libvmaf/src

Scoped orientation for any coding agent working directly inside `libvmaf/src/`.
Parent scope: [`../AGENTS.md`](../AGENTS.md) (libvmaf) and
[`../../AGENTS.md`](../../AGENTS.md) (root).

## Mandatory safety invariants

The following invariants were established during the 2026-05-16 memory-safety
audit (findings #7, #8, #10). Every PR that touches the affected files — or
adds new code in the same category — must preserve them.

### 1. Every `pthread_*_init` return value must be checked (finding #7)

`pthread_mutex_init`, `pthread_cond_init`, and `pthread_rwlock_init` return
non-zero on `ENOMEM` on some POSIX implementations (embedded, musl-based
systems). Ignoring the return value leaves the pool or lock object in an
undefined state; the next `pthread_mutex_lock` call is undefined behaviour.

Pattern to follow: staged init with teardown of already-initialised
primitives on failure (see `vmaf_thread_pool_create` in
[`thread_pool.c`](thread_pool.c)).

### 2. Every `aligned_malloc` / `malloc` call must NULL-check before use (finding #8)

A missing NULL check after `aligned_malloc` causes a null-pointer dereference
on OOM (ASan-detected). In hot-path functions such as `adm_dwt2_*` in
[`feature/adm_tools.c`](feature/adm_tools.c), the allocation must either be
NULL-checked and the function must return an error, or the buffer must be
pre-allocated in the extractor `init` callback so the per-frame path stays
allocation-free. See Power of 10 rule 3 and CERT MEM30-C.

### 3. Size-computing functions that accept external `unsigned w` / `h` must bound-check first (finding #10)

When `w` is large enough that `(w + ALIGN - 1u)` wraps on `unsigned`
arithmetic, the resulting aligned size is 0. The allocator succeeds, and any
pixel read is OOB. Add an early-exit `if (w == 0 || w > 32768u || ...)
return -EINVAL;` guard at the public entry point before any arithmetic.
Pattern: see `vmaf_picture_alloc` in [`picture.c`](picture.c). CERT INT30-C.

## Bootstrap score-name invariant (ADR-0480)

The four bootstrap score-name suffixes (`_bagging`, `_stddev`, `_ci_p95_lo`,
`_ci_p95_hi`) are defined once in [`bootstrap_names.h`](bootstrap_names.h).
Both `predict.c` (`bootstrap_append_named_scores`) and `libvmaf.c`
(`vmaf_score_pooled_model_collection`) consume this header.

**Do not** add or rename suffixes in either file without updating
`bootstrap_names.h`.  The `BOOTSTRAP_NAME_BUF_SZ()` macro sizes the name
buffer based on the longest suffix (`_ci_p95_lo`, 10 chars + NUL); adding a
longer suffix without updating the macro will silently truncate the name.
