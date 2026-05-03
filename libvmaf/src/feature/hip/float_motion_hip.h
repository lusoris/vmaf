/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP host glue for the float_motion feature extractor — seventh
 *  kernel-template consumer (T7-10b follow-up / ADR-0273).
 *
 *  Mirrors libvmaf/src/feature/cuda/float_motion_cuda.h. The HIP
 *  kernel artefact (a `float_motion_score.hip` device blob
 *  equivalent of the CUDA PTX) is a follow-up — the runtime PR
 *  (T7-10b) ships it alongside the live `kernel_template.c` bodies.
 *  Until then the consumer compiles host-only and `init()` surfaces
 *  `-ENOSYS` through the kernel-template helpers.
 */
#ifndef FEATURE_FLOAT_MOTION_HIP_H_
#define FEATURE_FLOAT_MOTION_HIP_H_

/* Placeholder — once the runtime PR adds the device-side blob, this
 * header will declare its symbol the same way `float_motion_score_ptx`
 * is declared in the CUDA twin. */

#endif /* FEATURE_FLOAT_MOTION_HIP_H_ */
