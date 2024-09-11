
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

#include "fp32_simd.h"
namespace vsag {

#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#define PORTABLE_ALIGN64 __attribute__((aligned(64)))

extern float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

extern float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);

/* L2 Distance */
float
L2SqrSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty4 = qty >> 2;

    const float* pEnd1 = pVect1 + (qty4 << 2);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

float
L2SqrSIMD4ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = L2SqrSIMD4ExtSSE(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float* pVect1 = (float*)pVect1v + qty4;
    float* pVect2 = (float*)pVect2v + qty4;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

    return (res + res_tail);
}

float
L2SqrSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);
    float PORTABLE_ALIGN32 TmpRes[8];
    size_t qty16 = qty >> 4;

    const float* pEnd1 = pVect1 + (qty16 << 4);

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
}

extern float (*L2SqrSIMD16Ext)(const void*, const void*, const void*);

float
L2SqrSIMD16ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
    float* pVect1 = (float*)pVect1v + qty16;
    float* pVect2 = (float*)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
    return (res + res_tail);
}

/* IP Distance */
float
InnerProductSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float* pEnd1 = pVect1 + 16 * qty16;
    const float* pEnd2 = pVect1 + 4 * qty4;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

float
InnerProductSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    size_t qty16 = qty / 16;

    const float* pEnd1 = pVect1 + 16 * qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

extern float (*InnerProductSIMD16Ext)(const void*, const void*, const void*);
extern float (*InnerProductSIMD4Ext)(const void*, const void*, const void*);

float
InnerProductDistanceSIMD16ExtResidualsSSE(const void* pVect1v,
                                          const void* pVect2v,
                                          const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float* pVect1 = (float*)pVect1v + qty16;
    float* pVect2 = (float*)pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

float
InnerProductDistanceSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return 1.0f - InnerProductSIMD16Ext(pVect1v, pVect2v, qty_ptr);
}

float
InnerProductDistanceSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return 1.0f - InnerProductSIMD4Ext(pVect1v, pVect2v, qty_ptr);
}

float
InnerProductDistanceSIMD4ExtResidualsSSE(const void* pVect1v,
                                         const void* pVect2v,
                                         const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float* pVect1 = (float*)pVect1v + qty4;
    float* pVect2 = (float*)pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}

void
PQDistanceSSEFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
    const float* float_centers = (const float*)single_dim_centers;
    float* float_result = (float*)result;
    for (size_t idx = 0; idx < 256; idx += 4) {
        __m128 v_centers_dim = _mm_loadu_ps(float_centers + idx);
        __m128 v_query_vec = _mm_set1_ps(single_dim_val);
        __m128 v_diff = _mm_sub_ps(v_centers_dim, v_query_vec);
        __m128 v_diff_sq = _mm_mul_ps(v_diff, v_diff);
        __m128 v_chunk_dists = _mm_loadu_ps(&float_result[idx]);
        v_chunk_dists = _mm_add_ps(v_chunk_dists, v_diff_sq);
        _mm_storeu_ps(&float_result[idx], v_chunk_dists);
    }
}

}  // namespace vsag

namespace vsag::SSE {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    const int n = dim / 4;          // process 4 floats at a time
    __m128 sum = _mm_setzero_ps();  // initialize to 0
    for (int i = 0; i < n; ++i) {
        __m128 a = _mm_loadu_ps(query + i * 4);   // load 4 floats from memory
        __m128 b = _mm_loadu_ps(codes + i * 4);   // load 4 floats from memory
        sum = _mm_add_ps(sum, _mm_mul_ps(a, b));  // accumulate the product
    }
    alignas(16) float result[4];
    _mm_store_ps(result, sum);  // store the accumulated result into an array
    float ip = result[0] + result[1] + result[2] +
               result[3];  // calculate the sum of the accumulated results
    ip += Generic::FP32ComputeIP(query + n * 4, codes + n * 4, dim - n * 4);
    return ip;
#else
    return vsag::Generic::FP32ComputeIP(query, codes, dim);
#endif
}
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_SSE)
    const uint64_t n = dim / 4;
    __m128 sum = _mm_setzero_ps();
    for (int i = 0; i < n; ++i) {
        __m128 a = _mm_loadu_ps(query + i * 4);
        __m128 b = _mm_loadu_ps(codes + i * 4);
        __m128 diff = _mm_sub_ps(a, b);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }
    alignas(16) float result[4];
    _mm_store_ps(result, sum);
    float l2 = result[0] + result[1] + result[2] + result[3];
    l2 += Generic::FP32ComputeL2Sqr(query + n * 4, codes + n * 4, dim - n * 4);
    return l2;
#else
    return vsag::Generic::FP32ComputeL2Sqr(query, codes, dim);
#endif
}

}  // namespace vsag::SSE