/**
 *
 *  Copyright 2026 Lusoris and Claude (Anthropic)
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

#ifndef ARM64_NEON_SSIMULACRA2_H_
#define ARM64_NEON_SSIMULACRA2_H_

#include <stddef.h>
#include <stdint.h>

/*
 * aarch64 NEON variants of the SSIMULACRA 2 SIMD kernels. Same
 * bit-exact contract as the AVX2 / AVX-512 TUs (ADR-0161) — 4-wide
 * float lanes. `cbrtf` applied per-lane via scalar libm.
 */

void ssimulacra2_multiply_3plane_neon(const float *a, const float *b, float *mul, unsigned w,
                                      unsigned h);
void ssimulacra2_linear_rgb_to_xyb_neon(const float *lin, float *xyb, unsigned w, unsigned h);
void ssimulacra2_downsample_2x2_neon(const float *in, unsigned iw, unsigned ih, float *out,
                                     unsigned *ow_out, unsigned *oh_out);
void ssimulacra2_ssim_map_neon(const float *m1, const float *m2, const float *s11, const float *s22,
                               const float *s12, unsigned w, unsigned h, double plane_averages[6]);
void ssimulacra2_edge_diff_map_neon(const float *img1, const float *mu1, const float *img2,
                                    const float *mu2, unsigned w, unsigned h,
                                    double plane_averages[12]);

#endif /* ARM64_NEON_SSIMULACRA2_H_ */
