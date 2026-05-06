/*
 * Copyright 2026 Lusoris and Claude (Anthropic)
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Regression test for the `cli_parse.c` long-only-error-fix
 * (ADR-0316, follow-up to ADR-0311 / PR #408 fuzzer-parked
 * crash).
 *
 * Bug: invalid `optarg` for `--threads` / `--subsample` /
 * `--cpumask` (e.g. `--threads abc`, or the `--th=foosoxe`
 * abbreviation captured at
 * `libvmaf/test/fuzz/cli_parse_known_crashes/`) used to trip
 * `error()`'s `assert(long_opts[n].name)` because the
 * call-site passed a synthesised short-option char (`'t'` /
 * `'s'` / `'c'`) that does not appear in `long_opts[]`. Fix
 * passes the long-only enum value (e.g. `ARG_THREADS`)
 * instead, so `error()` finds the matching entry and emits a
 * clean usage() line + `exit(1)` rather than `SIGABRT`.
 *
 * Test strategy: fork(); the child invokes `cli_parse` with
 * the abbreviated bad input, captures stderr to a pipe, and
 * the parent asserts the child exited with status 1 (clean
 * usage error) and printed an "Invalid argument" line — NOT
 * died from SIGABRT. POSIX fork/waitpid is required; the test
 * is wired Windows-off in meson.build alongside
 * `test_y4m_411_oob`.
 */

#include <getopt.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "test.h"

#include "cli_parse.h"

/* Capture-and-replay test: fork(), in the child run
 * cli_parse(argv) with stderr redirected into a pipe, in the
 * parent read the captured bytes and waitpid() for the exit
 * status. Returns 0 on a clean usage() error (status == 1 +
 * stderr contains `needle`); returns -1 on assert/abort
 * (signal exit) or any other unexpected outcome. */
static int run_parse_expect_usage_error(int argc, char **argv, const char *needle)
{
    int pipefd[2];
    if (pipe(pipefd) != 0)
        return -1;

    const pid_t pid = fork();
    if (pid < 0) {
        (void)close(pipefd[0]);
        (void)close(pipefd[1]);
        return -1;
    }

    if (pid == 0) {
        /* Child: redirect stderr into the pipe write end so
         * the parent can scan for the "Invalid argument"
         * line emitted by usage(). */
        (void)close(pipefd[0]);
        (void)dup2(pipefd[1], 2);
        (void)close(pipefd[1]);

        CLISettings settings;
        memset(&settings, 0, sizeof(settings));
        optind = 1;
        cli_parse(argc, argv, &settings);
        /* Should be unreachable — cli_parse should have
         * called usage() -> exit(1) on the bad optarg. If we
         * reach here, the parser silently accepted invalid
         * input; flag it as a distinct failure shape. */
        _exit(2);
    }

    /* Parent: drain stderr and reap the child. */
    (void)close(pipefd[1]);
    char buf[4096];
    size_t total = 0;
    for (;;) {
        if (total >= sizeof(buf) - 1u)
            break;
        const ssize_t n = read(pipefd[0], buf + total, sizeof(buf) - 1u - total);
        if (n <= 0)
            break;
        total += (size_t)n;
    }
    buf[total] = '\0';
    (void)close(pipefd[0]);

    int status = 0;
    if (waitpid(pid, &status, 0) < 0)
        return -1;

    /* The bug shape we're guarding against is SIGABRT from
     * the assert; reject any signal-termination outcome. */
    if (!WIFEXITED(status))
        return -1;
    if (WEXITSTATUS(status) != 1)
        return -1;
    if (strstr(buf, needle) == NULL)
        return -1;
    return 0;
}

static char *test_threads_invalid_optarg_does_not_assert()
{
    char *argv[] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--threads", "abc"};
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));
    const int rc = run_parse_expect_usage_error(argc, argv, "Invalid argument");
    mu_assert("cli_parse: --threads abc must exit(1) with usage error, not SIGABRT", rc == 0);
    return NULL;
}

static char *test_subsample_invalid_optarg_does_not_assert()
{
    char *argv[] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--subsample", "xyz"};
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));
    const int rc = run_parse_expect_usage_error(argc, argv, "Invalid argument");
    mu_assert("cli_parse: --subsample xyz must exit(1) with usage error, not SIGABRT", rc == 0);
    return NULL;
}

static char *test_cpumask_invalid_optarg_does_not_assert()
{
    char *argv[] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--cpumask", "qqq"};
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));
    const int rc = run_parse_expect_usage_error(argc, argv, "Invalid argument");
    mu_assert("cli_parse: --cpumask qqq must exit(1) with usage error, not SIGABRT", rc == 0);
    return NULL;
}

/* Mirrors the parked fuzzer reproducer at
 * libvmaf/test/fuzz/cli_parse_corpus/cli_threads_abbrev_assert.argv:
 * `--th=foosoxe` (getopt unique-prefix abbreviation of
 * `--threads`). This is the exact shape PR #408's fuzzer
 * surfaced; promoting the file to the corpus protects the
 * fuzzer path, this case protects the C unit-test path. */
static char *test_threads_abbrev_does_not_assert()
{
    char *argv[] = {"vmaf", "-r", "ref.y4m", "-d", "dis.y4m", "--th=foosoxe"};
    const int argc = (int)(sizeof(argv) / sizeof(argv[0]));
    const int rc = run_parse_expect_usage_error(argc, argv, "Invalid argument");
    mu_assert("cli_parse: --th=foosoxe abbrev must exit(1) with usage error, not SIGABRT", rc == 0);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_threads_invalid_optarg_does_not_assert);
    mu_run_test(test_subsample_invalid_optarg_does_not_assert);
    mu_run_test(test_cpumask_invalid_optarg_does_not_assert);
    mu_run_test(test_threads_abbrev_does_not_assert);
    return NULL;
}
