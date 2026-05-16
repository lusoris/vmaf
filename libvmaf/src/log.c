/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include "libvmaf/libvmaf.h"
#include "log.h"

#include <stdarg.h>
#include <stdatomic.h>
#include <stdio.h>
#ifdef _WIN32
/* MSVC provides isatty / fileno via <io.h> (named with leading underscores;
 * the non-underscored aliases stay available for POSIX source portability). */
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

/* _Atomic on the two module-level globals avoids data races when
 * vmaf_set_log_level() is called from one thread while vmaf_log() is
 * running on another.  Relaxed ordering is sufficient: both fields are
 * independently readable hints; no synchronisation with other memory
 * operations is required.  Zero overhead on x86-64 (MOV is already
 * atomic for aligned int-sized loads/stores). */
static _Atomic(int) vmaf_log_level_atomic = VMAF_LOG_LEVEL_INFO;
static _Atomic(int) istty_atomic = 0;

void vmaf_set_log_level(enum VmafLogLevel level)
{
    level = level < VMAF_LOG_LEVEL_NONE ? VMAF_LOG_LEVEL_NONE : level;
    level = level > VMAF_LOG_LEVEL_DEBUG ? VMAF_LOG_LEVEL_DEBUG : level;
    atomic_store_explicit(&vmaf_log_level_atomic, (int)level, memory_order_relaxed);
    atomic_store_explicit(&istty_atomic, isatty(fileno(stderr)), memory_order_relaxed);
}

static const char *level_str[] = {
    [VMAF_LOG_LEVEL_ERROR] = "ERROR",
    [VMAF_LOG_LEVEL_WARNING] = "WARNING",
    [VMAF_LOG_LEVEL_INFO] = "INFO",
    [VMAF_LOG_LEVEL_DEBUG] = "DEBUG",
};

static const char *level_str_color[] = {
    [VMAF_LOG_LEVEL_ERROR] = "\x1B[31m",
    [VMAF_LOG_LEVEL_WARNING] = "\x1B[33m",
    [VMAF_LOG_LEVEL_INFO] = "\x1B[32m",
    [VMAF_LOG_LEVEL_DEBUG] = "\x1B[34m",
};

void vmaf_log(enum VmafLogLevel level, const char *fmt, ...)
{
    if (level <= VMAF_LOG_LEVEL_NONE)
        return;
    const int cur_level = atomic_load_explicit(&vmaf_log_level_atomic, memory_order_relaxed);
    if (level > (enum VmafLogLevel)cur_level)
        return;

    const int tty = atomic_load_explicit(&istty_atomic, memory_order_relaxed);
    va_list args;
    (void)fprintf(stderr, "%slibvmaf%s %s%s%s ", tty ? "\x1B[35m" : "", tty ? "\x1B[0m" : "",
                  tty ? level_str_color[level] : "", level_str[level], tty ? "\x1B[0m" : "");
    va_start(args, fmt);
    (void)vfprintf(stderr, fmt, args);
    va_end(args);
}
