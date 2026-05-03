/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP host glue for the motion_v2 feature extractor — sixth
 *  kernel-template consumer (T7-10b follow-up / ADR-0267).
 *
 *  Mirrors libvmaf/src/feature/cuda/integer_motion_v2_cuda.h. The HIP
 *  kernel artefact (a `motion_v2_score.hip` device blob equivalent of
 *  the CUDA PTX) is a follow-up — the runtime PR (T7-10b) ships it
 *  alongside the live `kernel_template.c` bodies. Until then the
 *  consumer compiles host-only and `init()` surfaces `-ENOSYS`
 *  through the kernel-template helpers.
 */
#ifndef FEATURE_INTEGER_MOTION_V2_HIP_H_
#define FEATURE_INTEGER_MOTION_V2_HIP_H_

/* Placeholder — once the runtime PR adds the device-side blob, this
 * header will declare its symbol the same way `motion_v2_score_ptx`
 * is declared in the CUDA twin. */

#endif /* FEATURE_INTEGER_MOTION_V2_HIP_H_ */
