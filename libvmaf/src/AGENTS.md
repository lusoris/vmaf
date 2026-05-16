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

## Vendored-code scope rule (ADR-0455)

**Vendored third-party code is in scope for the banned-function ban.**
Files under `src/mcp/3rdparty/`, `src/svm.cpp`, `src/svm.h`,
and `src/feature/third_party/` receive the same scrutiny as
fork-original code. Banned functions in `svm.cpp`, `cJSON.c`,
and `pdjson.c` (see docs/principles.md §1.2 rule 30) must be
replaced — not suppressed with `// NOLINT vendored`.

Current status:
- `svm.cpp` — `rand()` replaced with `rand_r(&svm_rand_state)` +
  `svm_set_rand_seed()` API (ADR-0455). Remaining suppression covers
  function-size, nesting, and null-analyzer warnings only.
- `mcp/3rdparty/cJSON/cJSON.c` — `sprintf`/`strcpy` replaced with
  `snprintf`/`memmove`/`memcpy` (ADR-0455).
- `pdjson.c` — not yet audited for banned functions; tracked in backlog.

## Rebase-sensitive invariants

- **svm.cpp PRNG state** (ADR-0455): `svm_rand_state` is a `__thread`
  (thread-local) `unsigned`. Any upstream libsvm sync must preserve the
  thread-local declaration and the `svm_set_rand_seed(unsigned)` public
  API. Do NOT reintroduce `rand()` — the CI lint gate will reject it.
- **cJSON.c `snprintf` / `memcpy` calls** (ADR-0455): when refreshing
  the cJSON vendor from upstream, ensure the safe replacements survive.
  See `docs/rebase-notes.md` §cJSON-vendored-fork-diff for the diff list.
