# Research-0136: Vendored Banned-Function Fix — cJSON.c and svm.cpp

**Date**: 2026-05-16
**Author**: Claude (Anthropic) + lusoris
**Status**: Accepted (implements ADR-0455)

## Problem

The memory-safety audit (`.workingdir/audit-memory-safety-threading-2026-05-16.md`)
identified two vendored files with banned-function violations under the project's
coding standards (docs/principles.md §1.2 rule 30). A prior policy classified these
as out-of-scope because they are third-party upstream code. That policy was reversed
("vendored is in scope"), requiring fixes.

**Finding #12 — cJSON.c**

11 banned-function calls across 5 sites:

| Location | Call | Risk |
|---|---|---|
| `cJSON_Version` (line 125) | `sprintf(version, "%i.%i.%i", ...)` | Stack buffer `version[15]`; format-string bound overflow possible if version components have > 3 digits — unlikely but lint-illegal |
| `cJSON_SetValuestring` (line 383) | `strcpy(object->valuestring, valuestring)` | Caller already bounds-checks length, but strcpy is banned regardless |
| `print_number` (lines 515, 517, 520, 526) | 4x `sprintf(number_buffer, ...)` | Stack buffer `number_buffer[26]`; `%1.17g` can produce up to 24 chars — tight but fits; still banned |
| `print_string_ptr` (line 827) | `strcpy((char *)output, "\"\"")` | Constant 3-byte write into `ensure()`'d buffer |
| `print_string_ptr` (line 904) | `sprintf(output_pointer, "u%04x", ...)` | Pointer into heap; `ensure()` pre-sized it correctly |
| `print_value` (lines 1254, 1262, 1270) | 3x `strcpy(output, "null"/"false"/"true")` | Constant writes into `ensure()`'d buffer |

**Finding #11 — svm.cpp**

3 calls to `rand()` — the global PRNG — in training-time code:

| Location | Call | Effect |
|---|---|---|
| `svm_binary_svc_probability` (line 1823) | `rand() % (prob->l - i)` | Shuffle 5-fold split of binary SVC probability estimate data |
| `svm_cross_validation` stratified path (line 2257) | `rand() % (count[c] - i)` | Shuffle within each class before stratified fold assignment |
| `svm_cross_validation` non-stratified path (line 2289) | `rand() % (l - i)` | Shuffle all samples before fold assignment |

The global `rand()` state is not thread-safe and is not seeded deterministically,
making cross-validation results non-reproducible across runs.

## Option Analysis

### cJSON.c: per-call fix vs. upstream re-download

The upstream cJSON 1.7.18 release (2023-08-20) already uses `snprintf` throughout.
An upstream re-download would be semantically equivalent but introduces two risks:

1. Any other local patches (there are none for cJSON currently) would be lost.
2. The diff becomes opaque — a full file replacement is harder to review than
   8 targeted replacements.

Per-call fixes chosen. Each replacement is minimal and faithful to upstream intent:

- `sprintf(buf, fmt, ...)` → `snprintf(buf, sizeof(buf), fmt, ...)` where `buf`
  is a stack-allocated array. `sizeof` gives the exact array size at compile time.
- `strcpy(dst, src)` where length is already validated → `memmove(dst, src, len + 1)`
  to handle any alias case and avoid the banned symbol.
- `strcpy(dst, "literal")` into `ensure()`'d buffers → `memcpy(dst, "literal", sizeof("literal"))`
  which copies `strlen + 1` bytes in one instruction.
- `sprintf(output_pointer, "u%04x", ch)` → `snprintf(output_pointer, 6, "u%04x", ch)`
  with the hard constant 6 (1 `u` + 4 hex digits + NUL). The surrounding
  `ensure()` call pre-allocates the buffer; the bound is load-bearing.

### svm.cpp: arc4random() vs. rand_r + svm_set_rand_seed

**Option A (arc4random)**: cryptographically-seeded, thread-safe by construction,
no external state. Rejected because:
- Not available on all POSIX targets without a compatibility shim.
- Non-deterministic — cannot reproduce a failing test by seed.

**Option B (rand_r + svm_set_rand_seed)**: POSIX.1-2008, thread-local state,
caller-controlled seed. Chosen because:
- Deterministic under a fixed seed — test failures can be reproduced.
- Thread-safe — each thread has its own `svm_rand_state`.
- Minimal API: one `void svm_set_rand_seed(unsigned seed)` function.
- Default (lazy) seed from `time(NULL) ^ getpid()` preserves the old behaviour
  (different results per run) when the caller does not set a seed.

The integer cast `(int)(svm_rand_next() % (unsigned)(count - i))` is deliberate:
`rand_r` returns `int` in [0, RAND_MAX]; the modulo argument is promoted to
`unsigned` to avoid signed-unsigned UB, then the result is cast back to `int` for
the Fisher-Yates swap index. No signed overflow is possible because the result is
in `[0, count - i - 1]` and `count <= prob->l <= INT_MAX`.

## Bit-Exactness of the Predict Path

`svm_predict`, `svm_predict_values`, and `svm_predict_probability` contain no
randomness and are unchanged. The predict path output is bit-identical before and
after this change for any fixed model. Verified by the existing golden gate:
`python/test/vmafexec_test.py`.

## New Test: test_svm_rand_seed.c

A 30-point synthetic regression dataset (2 features, 3-class labels) drives
`svm_cross_validation` (nr_fold=3, LINEAR SVC). Two properties are asserted:

1. **Same seed → same targets**: two calls with `seed=42` must produce identical
   fold-target vectors. This catches any accidental global-state residue.
2. **Different seeds → different targets**: calls with `seed=1` and `seed=9999`
   must differ in at least one fold target. For 30 points over 3 folds the
   probability that both seeds produce identical shuffles is O(1/30!) ≈ 0.

## References

- `.workingdir/audit-memory-safety-threading-2026-05-16.md` findings #11 and #12
- [ADR-0455](../adr/0452-vendored-banned-functions-fix.md)
- [ADR-0278](../adr/0278-nolint-citation-closeout.md) — NOLINT citation rule
- cJSON upstream: https://github.com/DaveGamble/cJSON (MIT license)
- libsvm upstream: https://www.csie.ntu.edu.tw/~cjlin/libsvm/ (BSD-3-Clause)
