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

/* Per-scale noise-weight arrays (length 4, one entry per DWT scale).
 * adm_f1s: DLM noise-floor weight per scale; NULL defaults to DEFAULT_ADM_NOISE_WEIGHT.
 * adm_f2s: AIM noise-floor weight per scale; NULL defaults to 0.0 (no AIM noise floor).
 * adm_skip_scale0: when true, scale-0 contributes 0 num/aim_num and 1e-10 den (matches
 *                  integer_adm.c adm_skip_scale0 convention).
 * adm_csf_scale / adm_csf_diag_scale: CSF band-scale multipliers used when
 *                  adm_csf_mode == ADM_CSF_MODE_BARTEN (default 1.0 each). */
int compute_adm(const float *ref, const float *dis, int w, int h, int ref_stride, int dis_stride,
                double *score, double *score_num, double *score_den, double *scores,
                double border_factor, double adm_enhn_gain_limit, double adm_norm_view_dist,
                int adm_ref_display_height, int adm_csf_mode, double *score_aim,
                const double *adm_f1s, const double *adm_f2s, bool adm_skip_scale0,
                double adm_csf_scale, double adm_csf_diag_scale);
