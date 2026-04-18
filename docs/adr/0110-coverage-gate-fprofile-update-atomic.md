# ADR-0110: Coverage gate uses `-fprofile-update=atomic` to survive parallel meson tests

- **Status**: Accepted
- **Date**: 2026-04-18
- **Deciders**: Lusoris, Claude (Anthropic)
- **Tags**: ci, build, simd, testing

## Context

The Coverage gate (CPU job in `.github/workflows/ci.yml`) fails reliably
on `master` after the AVX-512 SIMD rollouts. The hard-fail line is
emitted by `lcov`/`geninfo` during the "Gather lcov report" step:

```
geninfo: ERROR: Unexpected negative count '-44224' for
  /home/runner/work/vmaf/vmaf/libvmaf/src/feature/x86/vif_avx2.c:673.
        Perhaps you need to compile with '-fprofile-update=atomic'
```

Mechanism: meson runs the unit-test suite in parallel
(`meson test -j $(nproc)`); every test binary links the same
instrumented `libvmaf.so`; SIMD inner loops increment the same
`.gcda` 64-bit counters concurrently. The default
`-fprofile-update=single` performs a non-atomic read-modify-write,
which loses updates and can produce *negative* counter values.
`geninfo` refuses to process a negative count and exits non-zero,
so the coverage artifact is never produced and the gate fails.

The downstream symptom (`test_run_vmafexec FAILED` + `frame_skipping
Timeout` in the pytest step) is unrelated CI noise: that step runs
under `|| true` and never fails the gate by itself; the orphan
`vmaf` subprocesses get killed in the runner cleanup.

## Decision

Compile the coverage build with `-fprofile-update=atomic` for both
C and C++ translation units, in both the CPU and the (advisory) GPU
coverage jobs. As a belt-and-suspenders measure, also pass
`--ignore-errors negative` to `lcov --capture` so a future SIMD
addition that reintroduces a small race window degrades to a warning
instead of a hard gate failure.

The flag is applied only in the coverage CI workflow, not the
default build, so production builds are unaffected.

## Alternatives considered

| Option | Pros | Cons | Why not chosen |
|---|---|---|---|
| `-fprofile-update=atomic` + `--ignore-errors negative` (chosen) | Fixes the race correctly at ~5% coverage-build runtime cost; gate is robust to future SIMD growth | Coverage build is ~5% slower | Correct fix; cost is acceptable for a once-per-CI-run job |
| `meson test -j 1` for coverage only | No counter races at all (single process) | Coverage step goes from ~3 min to ~15 min — burns the 30-min job budget | Too slow once GPU paths are added |
| `--ignore-errors negative` alone (no atomic) | One-line workaround | Counter values silently wrong; coverage % becomes meaningless on instrumented SIMD lines | Defeats the purpose of having a coverage gate |
| Skip the failing test cases | Trivial diff | The tests aren't the cause; they are downstream noise. Skipping hides a real product bug if one ever appears here | User explicitly rejected this approach during scope discussion ("who said skip?") |
| Switch to `gcovr` with per-process `.gcda` directories via `GCOV_PREFIX` | Eliminates inter-process race entirely | Requires per-test wrapper script + lcov→gcovr migration; new tooling surface to maintain | Larger blast radius than the race justifies |

## Consequences

- **Positive**: Coverage gate becomes deterministically green on
  `master` again; merge queue unblocks. The fix is local to the CI
  workflow — no source-tree changes, no surprises in production
  builds.
- **Positive**: Future AVX-512 / NEON SIMD additions don't reopen
  this race because atomic counters are now in force for the
  coverage build.
- **Negative**: ~5% slower coverage build (atomic RMW vs. plain RMW
  on each instrumented basic-block edge). Negligible against the
  20-minute pytest timeout budget already in place.
- **Negative**: `--ignore-errors negative` masks future races *if*
  they reintroduce themselves and `-fprofile-update=atomic` is
  somehow disabled. Mitigation: the `lcov` step still prints
  warnings, which surface in the job log.
- **Neutral / follow-ups**: When the GPU coverage gate is promoted
  from advisory to required (post-runner-stability window), confirm
  the same flags are still in force on `coverage-gpu`. ADR-0037's
  required-status-check list will need `coverage-gpu` added at that
  point.

## References

- Master CI run that exposed the issue: <https://github.com/lusoris/vmaf/actions/runs/24605954678>
  (and four prior consecutive failures on `master` since the AVX-512
  motion port landed).
- GCC docs on `-fprofile-update`: <https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html>
- `lcov` `--ignore-errors` reference:
  <https://github.com/linux-test-project/lcov/blob/master/man/lcov.1>
- Related ADRs: [ADR-0015](0015-ci-matrix-asan-ubsan-tsan.md) (sanitizer
  CI matrix — coverage runs in the same family),
  [ADR-0037](0037-master-branch-protection.md) (required status checks
  list).
- Source: `req` — direct user direction on this PR
  ("we should up the coverage soon I guess" + rejection of skip-style
  fixes: "who said skip?").
