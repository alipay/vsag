
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

#include "sq4_uniform_simd.h"

namespace vsag {

float (*SQ4UniformComputeIPAlign64)(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float (*SQ4UniformComputeIPAlign32)(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float (*SQ4UniformComputeIPAlign16)(const uint8_t* query, const uint8_t* codes, uint64_t dim);
float (*SQ4UniformComputeIPAlign1)(const uint8_t* query, const uint8_t* codes, uint64_t dim);

float
SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, const uint64_t dim) {
    int32_t result = 0;
    uint32_t cur_pos = 0;

    std::vector<uint32_t> align_bits = {7, 6, 5, 0};  // 2 ^ align_bit, e.g. 2 ^ 7 = 128 dims

    for (auto align_bit : align_bits) {
        uint32_t dim_segments = (dim - cur_pos) >> align_bit << align_bit;
        if (dim_segments > 0) {
            switch (align_bit) {
                case 7:  // 64 * 8bits = 128 dims
                    result += SQ4UniformComputeIPAlign64(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, dim_segments);
                    break;
                case 6:  // 32 * 8bits = 64 dims
                    result += SQ4UniformComputeIPAlign32(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, dim_segments);
                    break;
                case 5:  // 16 * 8bits = 32 dims
                    result += SQ4UniformComputeIPAlign16(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, dim_segments);
                    break;
                default:  //  1 * 8bits = 2dims
                    result += SQ4UniformComputeIPAlign1(
                        codes1 + cur_pos / 2, codes2 + cur_pos / 2, dim_segments);
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
SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, const uint64_t dim) {
    int32_t result = 0;

    for (uint32_t d = 0; d < dim; d += 2) {
        float x_lo = codes1[d / 2] & 0x0f;
        float x_hi = (codes1[d / 2] & 0xf0) >> 4;
        float y_lo = codes2[d / 2] & 0x0f;
        float y_hi = (codes2[d / 2] & 0xf0) >> 4;

        result += (x_lo * y_lo + x_hi * y_hi);
    }

    return result;
}
}  // namespace Generic

namespace SSE {
float
SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, const uint64_t dim) {
    alignas(128) int16_t temp[8];
    int32_t result = 0;
    __m128i sum = _mm_setzero_si128();
    __m128i mask = _mm_set1_epi8(0xf);
    for (uint32_t d = 0; d < (dim + 1) / 2; d += 32) {
        auto xx = _mm_loadu_si128((__m128i*)(codes1 + d));
        auto yy = _mm_loadu_si128((__m128i*)(codes2 + d));
        auto xx1 = _mm_and_si128(xx, mask);                     // 16 * 8bits
        auto xx2 = _mm_and_si128(_mm_srli_epi16(xx, 4), mask);  // 16 * 8bits
        auto yy1 = _mm_and_si128(yy, mask);
        auto yy2 = _mm_and_si128(_mm_srli_epi16(yy, 4), mask);

        sum = _mm_add_epi16(sum, _mm_maddubs_epi16(xx1, yy1));
        sum = _mm_add_epi16(sum, _mm_maddubs_epi16(xx2, yy2));
    }
    _mm_store_si128((__m128i*)temp, sum);
    for (int i = 0; i < 8; ++i) {
        result += temp[i];
    }
    return result;
}
}  // namespace SSE

namespace AVX2 {
float
SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, const uint64_t dim) {
    alignas(256) int16_t temp[16];
    int32_t result = 0;
    __m256i sum = _mm256_setzero_si256();
    __m256i mask = _mm256_set1_epi8(0xf);
    for (uint32_t d = 0; d < (dim + 1) / 2; d += 32) {
        auto xx = _mm256_loadu_si256((__m256i*)(codes1 + d));
        auto yy = _mm256_loadu_si256((__m256i*)(codes2 + d));
        auto xx1 = _mm256_and_si256(xx, mask);                        // 32 * 8bits
        auto xx2 = _mm256_and_si256(_mm256_srli_epi16(xx, 4), mask);  // 32 * 8bits
        auto yy1 = _mm256_and_si256(yy, mask);
        auto yy2 = _mm256_and_si256(_mm256_srli_epi16(yy, 4), mask);

        sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(xx1, yy1));
        sum = _mm256_add_epi16(sum, _mm256_maddubs_epi16(xx2, yy2));
    }
    _mm256_store_si256((__m256i*)temp, sum);
    for (int i = 0; i < 16; ++i) {
        result += temp[i];
    }
    return result;
}
}  // namespace AVX2

namespace AVX512 {
float
SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, const uint64_t dim) {
    alignas(512) int16_t temp[32];
    int32_t result = 0;
    __m512i sum = _mm512_setzero_si512();
    __m512i mask = _mm512_set1_epi8(0xf);
    for (uint32_t d = 0; d < (dim + 1) / 2; d += 64) {
        auto xx = _mm512_loadu_si512((__m512i*)(codes1 + d));
        auto yy = _mm512_loadu_si512((__m512i*)(codes2 + d));
        auto xx1 = _mm512_and_si512(xx, mask);                        // 64 * 8bits
        auto xx2 = _mm512_and_si512(_mm512_srli_epi16(xx, 4), mask);  // 64 * 8bits
        auto yy1 = _mm512_and_si512(yy, mask);
        auto yy2 = _mm512_and_si512(_mm512_srli_epi16(yy, 4), mask);

        sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(xx1, yy1));
        sum = _mm512_add_epi16(sum, _mm512_maddubs_epi16(xx2, yy2));
    }
    _mm512_store_si512((__m512i*)temp, sum);
    for (int i = 0; i < 32; ++i) {
        result += temp[i];
    }
    return result;
}
}  // namespace AVX512

void
SQ4UniformSetSIMD() {
    SQ4UniformComputeIPAlign1 = Generic::SQ4UniformComputeCodesIP;
#ifdef ENABLE_SSE
    SQ4UniformComputeIPAlign16 = SSE::SQ4UniformComputeCodesIP;
    SQ4UniformComputeIPAlign32 = SQ4UniformComputeIPAlign16;
    SQ4UniformComputeIPAlign64 = SQ4UniformComputeIPAlign16;
#else
    SQ4UniformComputeIPAlign16 = SQ4UniformComputeIPAlign1;
    SQ4UniformComputeIPAlign32 = SQ4UniformComputeIPAlign1;
    SQ4UniformComputeIPAlign64 = SQ4UniformComputeIPAlign1;
#endif

#ifdef ENABLE_AVX2
    SQ4UniformComputeIPAlign32 = AVX2::SQ4UniformComputeCodesIP;
    SQ4UniformComputeIPAlign64 = SQ4UniformComputeIPAlign32;
#endif

#ifdef ENABLE_AVX512
    SQ4UniformComputeIPAlign64 = AVX512::SQ4UniformComputeCodesIP;
#endif
}

}  // namespace vsag