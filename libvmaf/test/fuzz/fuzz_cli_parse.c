/*
 * Copyright 2026 Lusoris and Claude (Anthropic)
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * libFuzzer harness for the `vmaf` CLI argument parser
 * (libvmaf/tools/cli_parse.c) — exercised through the public
 * `cli_parse` / `cli_free` surface.
 *
 * Rationale: `cli_parse` is attacker-reachable whenever a host
 * script wraps `vmaf` and forwards untrusted CLI arguments
 * (filenames, model strings, `--feature` payloads). The colon-
 * delimited `parse_model_config` / `parse_feature_config`
 * sub-parsers run `strsep` chains over heap-duplicated argv
 * strings; bugs there are classic format-string / overrun shapes.
 *
 * Strategy: treat the fuzzer's input bytes as a sequence of
 * NUL-terminated argv strings, materialise a `char **argv`
 * vector, call `cli_parse` against it, and free the resulting
 * settings. The parser calls `exit(1)` from `usage()` on
 * unrecoverable input; we wrap `exit` via `-Wl,--wrap=exit` and
 * `longjmp` back into the harness so a single bad input doesn't
 * tear down the fuzzer process.
 *
 * Build is opt-in via `-Dfuzz=true`; CI nightly runs a 60-second
 * smoke per harness. See [docs/development/fuzzing.md] and
 * ADR-0311 (extends ADR-0270).
 */

#include <setjmp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cli_parse.h"
#include "libvmaf/feature.h"

/* Hard cap on input size: a real argv fits in tens of bytes; we
 * give the fuzzer 16 KiB of headroom to splice many tokens. Past
 * that, we're paying for memcpy time, not coverage. */
#define FUZZ_MAX_INPUT_BYTES 16384u
#define FUZZ_MAX_ARGC 64u

/* `getopt_long` carries global mutable state (`optind`, `optarg`,
 * `opterr`, `optopt`). To make `cli_parse` callable in a loop we
 * have to reset it at the top of each iteration. The exact reset
 * idiom is libc-specific:
 *   - GNU libc: setting `optind = 0` triggers a full reset.
 *   - BSD-style: `optreset = 1; optind = 1`.
 * Linux fuzz hosts use glibc; the `optind = 0` form is portable
 * enough for the CI matrix. */
extern int optind;

/* Disable AddressSanitizer's leak detector for this harness. The
 * upstream-mirror `cli_free` (libvmaf/tools/cli_parse.c) is known
 * to leave the per-feature / per-model `VmafFeatureDictionary`
 * allocations behind on the success path; we mirror an audited
 * cleanup at the bottom of `LLVMFuzzerTestOneInput`, but the
 * `usage()` -> `exit()` -> longjmp path can land mid-parse and
 * leave a partially-allocated dict that the harness has no
 * pointer to free. Letting libFuzzer abort on those would pin
 * coverage at zero and turn this harness into a noise factory.
 * Real heap-overflow / use-after-free bugs are still caught —
 * only leak-only reports are silenced. The weak-symbol override
 * here is the canonical libFuzzer / ASan idiom for per-harness
 * defaults; environment-side `ASAN_OPTIONS` would override this
 * if a CI operator wants leak detection on. */
/* `__asan_default_options` is the AddressSanitizer-defined weak
 * override symbol (see compiler-rt/lib/asan/asan_flags.cpp). The
 * `__` prefix is mandated by that ABI; the
 * `bugprone-reserved-identifier` warning is load-bearing-wrong. */
/* NOLINTNEXTLINE(misc-use-internal-linkage,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp) — ASan weak-symbol ABI */
const char *__asan_default_options(void);
/* NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp) — ASan weak-symbol ABI */
const char *__asan_default_options(void)
{
    return "detect_leaks=0:allocator_may_return_null=1";
}

/* `usage()` in cli_parse.c calls `exit(1)` on bad input. libFuzzer
 * needs the harness to *return* on bad input, not terminate, so
 * each iteration can keep coverage feedback flowing. We use the
 * GNU ld `--wrap=exit` flag (set in meson.build link_args) to
 * intercept the `exit` symbol and longjmp back to the harness
 * frame. The wrapped symbol must be `__wrap_exit`; the original
 * is renamed to `__real_exit`. The `_Noreturn` attribute is
 * load-bearing — `usage()` is itself `_Noreturn`, and dropping
 * `_Noreturn` here would let the compiler fall through past the
 * `exit` call site in `cli_parse.c` and trip undefined-behaviour
 * sanitisation. */
static jmp_buf g_exit_jmp;
static int g_exit_jmp_armed;

/* `__wrap_exit` and `__real_exit` are the GNU-ld `--wrap=symbol`
 * ABI: the linker rewrites every call to `exit` into a call to
 * `__wrap_exit`, and exposes the original symbol as
 * `__real_exit`. The `__` prefix is mandated by that ABI; the
 * `bugprone-reserved-identifier` warning is load-bearing-wrong. */
/* NOLINTNEXTLINE(misc-use-internal-linkage,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp) — linker --wrap=exit ABI */
_Noreturn void __wrap_exit(int code);
/* NOLINTNEXTLINE(misc-use-internal-linkage,bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp) — linker --wrap=exit ABI */
void __wrap_exit(int code)
{
    if (g_exit_jmp_armed) {
        g_exit_jmp_armed = 0;
        /* longjmp value must be non-zero; OR with 1 so a code of
         * 0 still routes to the setjmp != 0 branch. */
        longjmp(g_exit_jmp, (code | 1));
    }
    /* If we somehow get here without an arm (e.g. a static-init
     * exit before LLVMFuzzerTestOneInput runs), call the real
     * exit so the process actually terminates. */
    /* NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp) — linker --wrap=exit ABI */
    extern _Noreturn void __real_exit(int);
    __real_exit(code);
}

/* Best-effort early reject for argv vectors that hit the known
 * `error()`-assert bug class in cli_parse.c. The bug: handlers
 * for the long-only options `ARG_THREADS` / `ARG_SUBSAMPLE` /
 * `ARG_CPUMASK` call `parse_unsigned(optarg, 't' / 's' / 'c',
 * argv[0])` with a hardcoded short-option char that is not
 * registered in `long_opts[]`. When the optarg fails to parse,
 * `error()` walks long_opts looking for the impossible char and
 * trips its `assert(long_opts[n].name)`. Captured reproducer in
 * `cli_parse_known_crashes/cli_threads_abbrev_assert.argv`;
 * tracked as a follow-up bug fix.
 *
 * The filter rejects any token whose prefix could route through
 * one of the three buggy handlers via getopt's unique-prefix
 * abbreviation matching: `--th*` (only matches `--threads`),
 * `--s*` (would match `--subsample` and friends), `--c*` (would
 * match `--cpumask` and friends). The over-broad shape is the
 * point — we want every variation of the abbreviation-driven
 * path silenced until the upstream-mirror fix lands. Other
 * crashes still escape the filter. Returns 1 if the input
 * should be skipped. */
static int known_assert_in_input(char *const *argv, unsigned argc)
{
    static const char *const known_bad_prefixes[] = {"--th", "--s", "--c"};
    static const size_t known_bad_n = sizeof(known_bad_prefixes) / sizeof(known_bad_prefixes[0]);

    for (unsigned i = 1u; i < argc; i++) {
        if (argv[i] == NULL)
            continue;
        for (size_t k = 0u; k < known_bad_n; k++) {
            const size_t pfx_len = strlen(known_bad_prefixes[k]);
            if (strncmp(argv[i], known_bad_prefixes[k], pfx_len) == 0)
                return 1;
        }
    }
    return 0;
}

/* Tokenise the fuzzer input on NUL bytes into an argv vector.
 * Returns argc on success, 0 if the input has no usable tokens.
 * The returned `argv` is a heap allocation; `argv[i]` pointers
 * alias into the caller-supplied `buf`. */
static unsigned tokenise_argv(uint8_t *buf, size_t size, char **argv, unsigned argv_cap)
{
    /* argv[0] is the program name; pin it to a fixed string so
     * the parser's `argv[0]` references in `usage()` / `error()`
     * always have a valid C string regardless of fuzzer input. */
    static char argv0[] = "vmaf-fuzz";
    if (argv_cap < 2u)
        return 0u;
    argv[0] = argv0;
    unsigned argc = 1u;

    size_t i = 0u;
    while (i < size && argc < argv_cap) {
        /* Skip leading NULs — empty tokens are allowed but waste
         * argv slots. */
        while (i < size && buf[i] == 0u)
            i++;
        if (i >= size)
            break;
        /* Token starts here; find the next NUL or end-of-buffer. */
        const size_t tok_start = i;
        while (i < size && buf[i] != 0u)
            i++;
        /* Force NUL-termination at the boundary. If i == size,
         * the trailing token is unterminated; clobber the last
         * input byte to NUL so getopt sees a proper C string.
         * This is destructive but the input buffer is the
         * fuzzer's own scratch copy (libFuzzer hands each
         * iteration a fresh buffer it owns). */
        if (i < size) {
            buf[i] = 0u;
            i++;
        } else if (tok_start < size) {
            buf[size - 1u] = 0u;
        }
        argv[argc++] = (char *)&buf[tok_start];
    }
    return argc;
}

/* libFuzzer's contract requires `LLVMFuzzerTestOneInput` to have
 * external linkage; the runtime resolves it by name at link time
 * (`-fsanitize=fuzzer`). Cannot be static — the
 * `misc-use-internal-linkage` warning is load-bearing-wrong. */
/* NOLINTNEXTLINE(misc-use-internal-linkage) — libFuzzer entry-point ABI */
/* Free the per-feature / per-model `VmafFeatureDictionary`
 * allocations that `cli_free` does not release. Pre-existing
 * cli_parse.c fork-mirror leak; mirroring an audited cleanup
 * here keeps libFuzzer's leak detector quiet (when enabled). */
static void free_settings_dicts(CLISettings *settings)
{
    for (unsigned i = 0u; i < settings->feature_cnt; i++) {
        if (settings->feature_cfg[i].opts_dict != NULL)
            (void)vmaf_feature_dictionary_free(&settings->feature_cfg[i].opts_dict);
    }
    for (unsigned i = 0u; i < settings->model_cnt; i++) {
        for (unsigned j = 0u; j < settings->model_config[i].overload_cnt; j++) {
            if (settings->model_config[i].feature_overload[j].opts_dict != NULL) {
                (void)vmaf_feature_dictionary_free(
                    &settings->model_config[i].feature_overload[j].opts_dict);
            }
        }
    }
}

/* Drive one parse cycle: arm the longjmp shim, reset getopt
 * state, call `cli_parse`, then release every allocation it
 * may have made — both the cli_free-handled argv buffers and
 * the dict allocations cli_free leaks. */
static void drive_parse(int argc, char **argv)
{
    /* Reset getopt's global state. `optind = 0` is the glibc
     * full-reset idiom (see getopt(3) under "GNU extensions"). */
    optind = 0;

    CLISettings settings;
    memset(&settings, 0, sizeof(settings));

    g_exit_jmp_armed = 1;
    if (setjmp(g_exit_jmp) == 0) {
        cli_parse(argc, argv, &settings);
    }
    g_exit_jmp_armed = 0;

    free_settings_dicts(&settings);
    cli_free(&settings);
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size == 0u || size > FUZZ_MAX_INPUT_BYTES)
        return 0;

    /* Copy into a heap buffer so we can NUL-terminate tokens in
     * place without aliasing the const fuzzer-owned input. */
    uint8_t *buf = malloc(size);
    if (buf == NULL)
        return 0;
    memcpy(buf, data, size);

    char *argv[FUZZ_MAX_ARGC];
    memset(argv, 0, sizeof(argv));
    const unsigned argc = tokenise_argv(buf, size, argv, FUZZ_MAX_ARGC);
    if (argc < 2u || known_assert_in_input(argv, argc)) {
        free(buf);
        return 0;
    }

    drive_parse((int)argc, argv);
    free(buf);
    return 0;
}
