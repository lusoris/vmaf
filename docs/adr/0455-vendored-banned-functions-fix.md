# ADR-0455: Fix Banned Functions in Vendored Code (cJSON, libsvm)

- **Status**: Accepted
- **Date**: 2026-05-16
- **Deciders**: lusoris, Claude (Anthropic)
- **Tags**: security, coding-standards, vendored, cjson, libsvm, fork-local

## Context

The project-wide coding standards (docs/principles.md §1.2 rule 30) ban `sprintf`,
`strcpy`, `strcat`, `rand`, and other unsafe functions. A prior policy treated
vendored third-party files as out-of-scope for this ban. The policy was reversed
(per the "vendored is in scope" rule) following a memory-safety audit
(`.workingdir/audit-memory-safety-threading-2026-05-16.md` findings #11 and #12).

Two vendored files contained violations:

1. `libvmaf/src/mcp/3rdparty/cJSON/cJSON.c` — 11 calls to `sprintf` and `strcpy`.
   The file is nominally cJSON v1.7.18 but predates that release's own `snprintf`
   cleanup (the upstream 1.7.18 release date was 2023-08-20; the vendored copy was
   not refreshed at that time). The `libvmaf/src/mcp/AGENTS.md` invariant #5
   ("do NOT patch it locally") was written before the vendored-is-in-scope policy
   reversal and is now superseded by this ADR.

2. `libvmaf/src/svm.cpp` — 3 calls to `rand()` in `svm_binary_svc_probability`
   and `svm_cross_validation`. The whole file was under a blanket
   `// NOLINTBEGIN ... // NOLINTEND` citing "upstream libsvm parity" as
   justification for suppressing the rand() warning.

## Decision

### cJSON.c — per-call fixes

Replace each banned call in-place rather than re-downloading upstream v1.7.18.
The upstream v1.7.18 source uses `snprintf` throughout, so the replacements are
semantically identical to what an upstream refresh would produce.

- `sprintf(buf, fmt, ...)` → `snprintf(buf, sizeof(buf), fmt, ...)` at all 7
  `sprintf` call sites. The target buffers are all stack-allocated arrays whose
  size is statically known (`version[15]`, `number_buffer[26]`), so
  `sizeof(buf)` is correct.
- `strcpy(object->valuestring, valuestring)` → `memmove(object->valuestring,
  valuestring, strlen(valuestring) + 1u)`. At that call site the caller has
  already verified `strlen(valuestring) <= strlen(object->valuestring)`, so the
  destination has sufficient capacity. `memmove` is used (over `memcpy`) to
  correctly handle any alias scenario, though aliasing is not possible in
  practice here.
- `strcpy((char *)output, "null" / "false" / "true" / "\"\"")` → `memcpy`
  with `sizeof(literal)` as the count. The `ensure()` call above each site has
  already confirmed the output buffer has at least that many bytes.
- `sprintf((char *)output_pointer, "u%04x", ...)` → `snprintf` with a literal
  bound of 6 (5 data bytes + NUL). The surrounding code advances `output_pointer`
  by 4 after the write and `ensure()` has reserved enough space.

The `// NOLINTBEGIN` cover is NOT added to cJSON.c; the file is now clean.
AGENTS.md invariant #5 in `libvmaf/src/mcp/AGENTS.md` is updated to note that
local patches are now permitted for safety fixes.

### svm.cpp — seeded PRNG (Option B)

Replace `rand()` with a thread-local `rand_r(&svm_rand_state)` throughout the
training path (`svm_binary_svc_probability`, `svm_cross_validation`). A public
API `svm_set_rand_seed(unsigned seed)` is added so callers can seed the PRNG for
deterministic test reproduction. The default (unset) seed is derived from
`time(NULL) ^ getpid()` on first use, so independent processes diverge.

The predict path (`svm_predict`, `svm_predict_values`, `svm_predict_probability`)
is entirely unaffected — it contains no randomness.

The blanket `// NOLINTBEGIN ... // NOLINTEND` comment is updated to reflect that
the rand() violation has been resolved; the suppression remains for all other
upstream libsvm warnings (function size, branch depth, null-analyzer paths, etc.)
which would require a full re-flow of the vendored file to resolve.

A new C test `libvmaf/test/test_svm_rand_seed.c` verifies:
1. Two calls with the same seed produce identical fold-target vectors.
2. Two calls with different seeds produce at least one differing target.

## Alternatives considered

| Option | Pros | Cons | Verdict |
|---|---|---|---|
| cJSON: re-download upstream v1.7.18 | Refreshes all changes at once | Requires network, may introduce unrelated delta, violates the "reviewable diff" principle | Rejected — per-call fixes are smaller and reviewable |
| cJSON: NOLINT the whole file | Zero diff to the vendored text | Perpetuates the ban violation; NOLINT without refactoring is itself a lint violation per ADR-0278 | Rejected |
| svm.cpp Option A: `arc4random()` | No PRNG state to manage | Not on POSIX < macOS 10.12 without additional shim; non-deterministic across runs — breaks any seed-based reproducer | Rejected |
| svm.cpp Option B: `rand_r` + `svm_set_rand_seed` | POSIX-standard; deterministic; caller-controllable | Requires thread-local storage (widely supported) | Accepted |

## Consequences

### Positive
- `sprintf` and `strcpy` calls in cJSON.c are eliminated; buffer-overrun risk
  from future format-string accidents is closed.
- SVM cross-validation fold assignments are now reproducible under a fixed seed;
  test failures can be reproduced exactly.
- The blanket NOLINT comment in svm.cpp is narrowed in its claimed scope (rand()
  is no longer a reason for the suppression).

### Negative
- cJSON.c now diverges from the upstream source. The fork must carry its own diff.
  See `docs/rebase-notes.md` for the guidance on refreshing it.
- `svm.cpp`'s default (time-based) seed means the fold assignment changes across
  runs unless the caller seeds explicitly. This is the same behaviour as before
  (implicit global `rand()` state), but is now explicit.

### Neutral follow-ups
- The `libvmaf/src/mcp/AGENTS.md` invariant #5 is updated in this PR.
- `docs/rebase-notes.md` gains an entry for the cJSON and libsvm divergence.

## References

- req: "fix banned-function uses in vendored code — two files: cJSON.c and svm.cpp"
- `.workingdir/audit-memory-safety-threading-2026-05-16.md` findings #11 (rand) and #12 (sprintf/strcpy)
- docs/principles.md §1.2 rule 30 — banned-function list
- ADR-0278 — NOLINT citation-closeout rule
- ADR-0141 — touched-file cleanup rule
- ADR-0209 — MCP embedded scaffold (cJSON origin)
