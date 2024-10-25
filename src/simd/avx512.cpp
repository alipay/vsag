
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

#include <x86intrin.h>

#include <iostream>

namespace vsag {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

double
INT8_InnerProduct512_AVX512_impl(const void* pVect1v, const void* pVect2v, size_t qty) {
    __mmask32 mask = 0xFFFFFFFF;
    __mmask64 mask64 = 0xFFFFFFFFFFFFFFFF;

    int32_t cTmp[16];

    int8_t* pVect1 = (int8_t*)pVect1v;
    int8_t* pVect2 = (int8_t*)pVect2v;
    const int8_t* pEnd1 = pVect1 + qty;

    __m512i sum512 = _mm512_set1_epi32(0);

    while (pVect1 < pEnd1) {
        // sum512 = _mm512_dpbusd_epi32(sum512, _mm512_load_epi32(pVect1), _mm512_load_epi32(pVect2));
        __m256i v1 = _mm256_maskz_loadu_epi8(mask, pVect1);
        __m512i v1_512 = _mm512_cvtepi8_epi16(v1);
        pVect1 += 32;
        __m256i v2 = _mm256_maskz_loadu_epi8(mask, pVect2);
        __m512i v2_512 = _mm512_cvtepi8_epi16(v2);
        pVect2 += 32;
        //            _mm_prefetch(prefetch, _MM_HINT_T0);
        //            prefetch += 32;
        sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v1_512, v2_512));
    }

    _mm512_mask_storeu_epi32(cTmp, mask64, sum512);
    double res = 0;
    for (int i = 0; i < 16; i++) {
        res += cTmp[i];
    }
    return res;
}

extern int32_t
AVX2_SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim);

int32_t
AVX512_SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim) {
    if (dim == 0) {
        return 0;
    }
    alignas(512) int16_t temp[32];
    int32_t result = 0;
    uint64_t d = 0;
    __m512i sum = _mm512_setzero_si512();
    __m512i mask = _mm512_set1_epi8(0xf);
    for (; d + 127 < dim; d += 128) {
        auto xx = _mm512_loadu_si512((__m512i*)(codes1 + (d >> 1)));
        auto yy = _mm512_loadu_si512((__m512i*)(codes2 + (d >> 1)));
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
    //        result += (temp[0] + temp[1] + temp[2] + temp[3]);
    //        result += (temp[4] + temp[5] + temp[6] + temp[7]);
    //        result += (temp[8] + temp[9] + temp[10] + temp[11]);
    //        result += (temp[12] + temp[13] + temp[14] + temp[15]);
    //
    //        result += (temp[16] + temp[17] + temp[18] + temp[19]);
    //        result += (temp[20] + temp[21] + temp[22] + temp[23]);
    //        result += (temp[24] + temp[25] + temp[26] + temp[27]);
    //        result += (temp[28] + temp[29] + temp[30] + temp[31]);

    result += AVX2_SQ4UniformComputeCodesIP(codes1 + (d >> 1), codes2 + (d >> 1), dim - d);
    return result;
}

int32_t
INT4_IP_avx512_impl(const void* p1_vec, const void* p2_vec, int dim) {
    const uint8_t* x = (const uint8_t*)p1_vec;
    const uint8_t* y = (const uint8_t*)p2_vec;
    return AVX512_SQ4UniformComputeCodesIP(x, y, dim);
}

float
L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *(size_t*)qty_ptr;
    float PORTABLE_ALIGN64 TmpRes[16];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
    }

    _mm512_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return (res);
}

float
InnerProductSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;

    const float* pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];

    return sum;
}

}  // namespace vsag
