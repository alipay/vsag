
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

#include <iostream>

namespace vsag {

double
INT8_IP_impl(const void* pVect1, const void* pVect2, size_t qty) {
    int8_t* vec1 = (int8_t*)pVect1;
    int8_t* vec2 = (int8_t*)pVect2;
    double res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += vec1[i] * vec2[i];
    }
    return res;
}

int32_t
INT4_IP_impl(const void* p1_vec, const void* p2_vec, int dim) {
    int8_t* x = (int8_t*)p1_vec;
    int8_t* y = (int8_t*)p2_vec;
    int32_t sum = 0;
    for (int d = 0; d < dim / 2; ++d) {
        {
            int32_t xx = x[d] & 15;
            int32_t yy = y[d] & 15;
            sum += xx * yy;
        }
        {
            int32_t xx = (x[d] >> 4) & 15;
            int32_t yy = (y[d] >> 4) & 15;
            sum += xx * yy;
        }
    }
    return sum;
}

int32_t
GENERIC_SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim) {
    int32_t result = 0;

    for (uint64_t d = 0; d < dim; d += 2) {
        float x_lo = codes1[d >> 1] & 0x0f;
        float x_hi = (codes1[d >> 1] & 0xf0) >> 4;
        float y_lo = codes2[d >> 1] & 0x0f;
        float y_hi = (codes2[d >> 1] & 0xf0) >> 4;

        result += (x_lo * y_lo + x_hi * y_hi);
    }

    return result;
}

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float*)pVect1)[i] * ((float*)pVect2)[i];
    }
    return res;
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
    const float* float_centers = (const float*)single_dim_centers;
    float* float_result = (float*)result;
    for (size_t idx = 0; idx < 256; idx++) {
        double diff = float_centers[idx] - single_dim_val;
        float_result[idx] += (float)(diff * diff);
    }
}

}  // namespace vsag
