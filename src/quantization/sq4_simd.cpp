
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sq4_simd.h"

namespace vsag {

float (*SQ4ComputeIPAlign64)(const uint8_t* query,
                             const uint8_t* codes,
                             const float* lowerBound,
                             const float* diff,
                             uint64_t dim);
float (*SQ4ComputeIPAlign32)(const uint8_t* query,
                             const uint8_t* codes,
                             const float* lowerBound,
                             const float* diff,
                             uint64_t dim);
float (*SQ4ComputeIPAlign16)(const uint8_t* query,
                             const uint8_t* codes,
                             const float* lowerBound,
                             const float* diff,
                             uint64_t dim);
float (*SQ4ComputeIPAlign1)(const uint8_t* query,
                            const uint8_t* codes,
                            const float* lowerBound,
                            const float* diff,
                            uint64_t dim);

float (*SQ4ComputeL2Align64)(const uint8_t* query,
                             const uint8_t* codes,
                             const float* lowerBound,
                             const float* diff,
                             uint64_t dim);
float (*SQ4ComputeL2Align32)(const uint8_t* query,
                             const uint8_t* codes,
                             const float* lowerBound,
                             const float* diff,
                             uint64_t dim);
float (*SQ4ComputeL2Align16)(const uint8_t* query,
                             const uint8_t* codes,
                             const float* lowerBound,
                             const float* diff,
                             uint64_t dim);
float (*SQ4ComputeL2Align1)(const uint8_t* query,
                            const uint8_t* codes,
                            const float* lowerBound,
                            const float* diff,
                            uint64_t dim);

float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim) {
    return Generic::SQ4ComputeIP(query, codes, lowerBound, diff, dim);
}

float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim) {
    return Generic::SQ4ComputeL2(query, codes, lowerBound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    int32_t result = 0;
    uint32_t cur_pos = 0;

    std::vector<uint32_t> align_bits = {7, 6, 5, 0};  // 2 ^ align_bit, e.g. 2 ^ 7 = 128 dims

    for (auto align_bit : align_bits) {
        uint32_t dim_segments = (dim - cur_pos) >> align_bit << align_bit;
        if (dim_segments > 0) {
            switch (align_bit) {
                case 7:  // 64 * 8bits = 128 dims
                    result += SQ4ComputeIPAlign64(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, lowerBound, diff, dim_segments);
                    break;
                case 6:  // 32 * 8bits = 64 dims
                    result += SQ4ComputeIPAlign32(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, lowerBound, diff, dim_segments);
                    break;
                case 5:  // 16 * 8bits = 32 dims
                    result += SQ4ComputeIPAlign16(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, lowerBound, diff, dim_segments);
                    break;
                default:  //  1 * 8bits = 2dims
                    result += SQ4ComputeIPAlign1(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, lowerBound, diff, dim_segments);
                    break;
            }

            cur_pos += dim_segments;
        }
        if (cur_pos == dim) {
            return result;
        }
    }

    return result;
}

float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    int32_t result = 0;
    uint32_t cur_pos = 0;

    std::vector<uint32_t> align_bits = {7, 6, 5, 0};  // 2 ^ align_bit, e.g. 2 ^ 7 = 128 dims

    for (auto align_bit : align_bits) {
        uint32_t dim_segments = (dim - cur_pos) >> align_bit << align_bit;
        if (dim_segments > 0) {
            switch (align_bit) {
                case 7:  // 64 * 8bits = 128 dims
                    result += SQ4ComputeL2Align64(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, lowerBound, diff, dim_segments);
                    break;
                case 6:  // 32 * 8bits = 64 dims
                    result += SQ4ComputeL2Align32(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, lowerBound, diff, dim_segments);
                    break;
                case 5:  // 16 * 8bits = 32 dims
                    result += SQ4ComputeL2Align16(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, lowerBound, diff, dim_segments);
                    break;
                default:  //  1 * 8bits = 2dims
                    result += SQ4ComputeL2Align1(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, lowerBound, diff, dim_segments);
                    break;
            }

            cur_pos += dim_segments;
        }
        if (cur_pos == dim) {
            return result;
        }
    }

    return result;
}

namespace Generic {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim) {
    int32_t result = 0;

    for (uint32_t d = 0; d < dim; d += 2) {
        float x_lo = query[d];
        float x_hi = query[d + 1];
        float y_lo = (codes[d / 2] & 0x0f) * 15.0 / diff[d] + lowerBound[d];
        float y_hi = ((codes[d / 2] & 0xf0) >> 4) * 15.0 / diff[d + 1] + lowerBound[d + 1];

        result += (x_lo * y_lo + x_hi * y_hi);
    }

    return result;
}

float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim) {
    int32_t result = 0;

    for (uint32_t d = 0; d < dim; d += 2) {
        float x_lo = query[d];
        float x_hi = query[d + 1];
        float y_lo = (codes[d / 2] & 0x0f) * 15.0 / diff[d] + lowerBound[d];
        float y_hi = ((codes[d / 2] & 0xf0) >> 4) * 15.0 / diff[d + 1] + lowerBound[d + 1];

        result += (x_lo - y_lo) * (x_lo - y_lo) + (x_hi - y_hi) * (x_hi - y_hi);
    }

    return result;
}

float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    int32_t result = 0;

    for (uint32_t d = 0; d < dim; d += 2) {
        float x_lo = (codes1[d / 2] & 0x0f) * 15.0 / diff[d] + lowerBound[d];
        float x_hi = ((codes1[d / 2] & 0xf0) >> 4) * 15.0 / diff[d + 1] + lowerBound[d + 1];
        float y_lo = (codes2[d / 2] & 0x0f) * 15.0 / diff[d] + lowerBound[d];
        float y_hi = ((codes2[d / 2] & 0xf0) >> 4) * 15.0 / diff[d + 1] + lowerBound[d + 1];

        result += (x_lo * y_lo + x_hi * y_hi);
    }

    return result;
}

float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    int32_t result = 0;

    for (uint32_t d = 0; d < dim; d += 2) {
        float x_lo = (codes1[d / 2] & 0x0f) * 15.0 / diff[d] + lowerBound[d];
        float x_hi = ((codes1[d / 2] & 0xf0) >> 4) * 15.0 / diff[d + 1] + lowerBound[d + 1];
        float y_lo = (codes2[d / 2] & 0x0f) * 15.0 / diff[d] + lowerBound[d];
        float y_hi = ((codes2[d / 2] & 0xf0) >> 4) * 15.0 / diff[d + 1] + lowerBound[d + 1];

        result += (x_lo - y_lo) * (x_lo - y_lo) + (x_hi - y_hi) * (x_hi - y_hi);
    }

    return result;
}
}  // namespace Generic

namespace SSE {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim) {
    return Generic::SQ4ComputeIP(query, codes, lowerBound, diff, dim);
}

float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim) {
    return Generic::SQ4ComputeL2(query, codes, lowerBound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    return Generic::SQ4ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}

float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    return Generic::SQ4ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}
}  // namespace SSE

namespace AVX2 {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim) {
    return Generic::SQ4ComputeIP(query, codes, lowerBound, diff, dim);
}

float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim) {
    return Generic::SQ4ComputeL2(query, codes, lowerBound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    return Generic::SQ4ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}

float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    return Generic::SQ4ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}
}  // namespace AVX2

namespace AVX512 {
float
SQ4ComputeIP(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             uint64_t dim) {
    return Generic::SQ4ComputeIP(query, codes, lowerBound, diff, dim);
}

float
SQ4ComputeL2(const float* query,
             const uint8_t* codes,
             const float* lowerBound,
             const float* diff,
             const uint64_t dim) {
    return Generic::SQ4ComputeL2(query, codes, lowerBound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    return Generic::SQ4ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}

float
SQ4ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const float* lowerBound,
                  const float* diff,
                  const uint64_t dim) {
    return Generic::SQ4ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}
}  // namespace AVX512

void
SetSIMD() {
    {  // set IP distance
        SQ4ComputeIPAlign1 = Generic::SQ4ComputeCodesIP;
#ifdef ENABLE_SSE
        SQ4ComputeIPAlign16 = SSE::SQ4ComputeCodesIP;
        SQ4ComputeIPAlign32 = SQ4ComputeIPAlign16;
        SQ4ComputeIPAlign64 = SQ4ComputeIPAlign16;
#else
        SQ4ComputeIPAlign16 = SQ4ComputeIPAlign1;
        SQ4ComputeIPAlign32 = SQ4ComputeIPAlign1;
        SQ4ComputeIPAlign64 = SQ4ComputeIPAlign1;
#endif

#ifdef ENABLE_AVX2
        SQ4ComputeIPAlign32 = AVX2::SQ4ComputeCodesIP;
        SQ4ComputeIPAlign64 = SQ4ComputeIPAlign32;
#endif

#ifdef ENABLE_AVX512
        SQ4ComputeIPAlign64 = AVX512::SQ4ComputeCodesIP;
#endif
    }

    {
        SQ4ComputeL2Align1 = Generic::SQ4ComputeCodesL2;
#ifdef ENABLE_SSE
        SQ4ComputeL2Align16 = SSE::SQ4ComputeCodesL2;
        SQ4ComputeL2Align32 = SQ4ComputeL2Align16;
        SQ4ComputeL2Align64 = SQ4ComputeL2Align16;
#else
        SQ4ComputeL2Align16 = SQ4ComputeL2Align1;
        SQ4ComputeL2Align32 = SQ4ComputeL2Align1;
        SQ4ComputeL2Align64 = SQ4ComputeL2Align1;
#endif

#ifdef ENABLE_AVX2
        SQ4ComputeL2Align32 = AVX2::SQ4ComputeCodesL2;
        SQ4ComputeL2Align64 = SQ4ComputeL2Align32;
#endif

#ifdef ENABLE_AVX512
        SQ4ComputeL2Align64 = AVX512::SQ4ComputeCodesL2;
#endif
    }
}

}  // namespace vsag