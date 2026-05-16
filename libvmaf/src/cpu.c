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

#include <stdatomic.h>

#include "config.h"
#include "cpu.h"

/* _Atomic on flags and flags_mask avoids data races when vmaf_init_cpu() or
 * vmaf_set_cpu_flags_mask() is called from one thread while vmaf_get_cpu_flags()
 * is called from another.  Relaxed ordering is sufficient: both fields are
 * independent read hints with no cross-memory synchronisation requirement.
 * Zero overhead on x86-64 (aligned int-sized MOV is already atomic). */
static _Atomic(unsigned) flags = 0u;
static _Atomic(unsigned) flags_mask = (unsigned)-1;

void vmaf_init_cpu(void)
{
#if ARCH_X86
    const unsigned detected = vmaf_get_cpu_flags_x86();
    atomic_store_explicit(&flags, detected, memory_order_relaxed);
#if HAVE_AVX512
    if (detected & VMAF_X86_CPU_FLAG_AVX512) {
        /* Warm up AVX-512 execution units. On Intel CPUs, the 512-bit
         * units power down after idle and take 10-20µs to reactivate.
         * Issuing a dummy instruction here avoids that latency penalty
         * on the first frame of actual computation.
         * GCC/clang inline asm only — MSVC dropped inline asm on x64.
         * On MSVC the warmup is skipped (micro-opt, not correctness). */
#if defined(__GNUC__) || defined(__clang__)
        __asm__ volatile("vpxord %%zmm0, %%zmm0, %%zmm0" ::: "zmm0");
#endif
    }
#endif
#elif ARCH_AARCH64
    atomic_store_explicit(&flags, vmaf_get_cpu_flags_arm(), memory_order_relaxed);
#endif
}

void vmaf_set_cpu_flags_mask(const unsigned mask)
{
    atomic_store_explicit(&flags_mask, mask, memory_order_relaxed);
}

unsigned vmaf_get_cpu_flags(void)
{
    const unsigned f = atomic_load_explicit(&flags, memory_order_relaxed);
    const unsigned m = atomic_load_explicit(&flags_mask, memory_order_relaxed);
    return f & m;
}
