/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  HIP host glue for the PSNR feature extractor (T7-10 first
 *  consumer / ADR-0241).
 *
 *  Mirrors libvmaf/src/feature/cuda/integer_psnr_cuda.h. The HIP
 *  kernel artefact (a `psnr_score.hip` device-blob equivalent of the
 *  CUDA PTX) is a follow-up — the runtime PR (T7-10b) ships it
 *  alongside the live `kernel_template.c` bodies.
 */
#ifndef FEATURE_PSNR_HIP_H_
#define FEATURE_PSNR_HIP_H_

/* Placeholder — once the runtime PR adds the device-side blob, this
 * header will declare its symbol the same way `psnr_score_ptx` is
 * declared in the CUDA twin. */

#endif /* FEATURE_PSNR_HIP_H_ */
