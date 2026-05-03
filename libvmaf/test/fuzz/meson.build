# libFuzzer harness for the Y4M parser.
#
# Opt-in: only built when `-Dfuzz=true` is passed to `meson setup`. Requires
# clang as the compiler — gcc does not ship libFuzzer. Pairs with
# AddressSanitizer (`-Db_sanitize=address`) for heap-error coverage.
#
# See ADR-0270 (libFuzzer scaffold for OSSF Scorecard remediation) and
# docs/development/fuzzing.md for the design rationale and operator runbook.

if not get_option('fuzz')
    subdir_done()
endif

if cc.get_id() != 'clang'
    error('-Dfuzz=true requires clang (libFuzzer is a clang feature). ' +
          'Re-run meson setup with CC=clang.')
endif

# `-fsanitize=fuzzer` ships both libFuzzer's runtime (the `main`
# replacement that calls LLVMFuzzerTestOneInput) and the SanitizerCoverage
# instrumentation libFuzzer needs. The library under test is built with
# `-fsanitize=fuzzer-no-link` (instrumentation only, no `main`); we apply
# that via meson's `b_sanitize` only on the harness target's c_args /
# link_args so it doesn't infect the main libvmaf build.
fuzz_flags = ['-fsanitize=fuzzer']

fuzz_y4m_input = executable('fuzz_y4m_input',
    ['fuzz_y4m_input.c', '../../tools/y4m_input.c', '../../tools/yuv_input.c',
     '../../tools/vidinput.c'],
    include_directories : [libvmaf_inc, test_inc, include_directories('../../tools/')],
    c_args : fuzz_flags + compat_cflags,
    link_args : fuzz_flags,
)
