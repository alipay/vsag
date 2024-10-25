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

#include "simd.h"

#include <cpuinfo.h>

#include <iostream>

namespace vsag {

float (*L2SqrSIMD16Ext)(const void*, const void*, const void*);
float (*L2SqrSIMD16ExtResiduals)(const void*, const void*, const void*);
float (*L2SqrSIMD4Ext)(const void*, const void*, const void*);
float (*L2SqrSIMD4ExtResiduals)(const void*, const void*, const void*);

float (*InnerProductSIMD4Ext)(const void*, const void*, const void*);
float (*InnerProductSIMD16Ext)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD16Ext)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD16ExtResiduals)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD4Ext)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD4ExtResiduals)(const void*, const void*, const void*);

//double
//INT8_InnerProduct512_AVX512_impl(const void* pVect1v, const void* pVect2v, size_t qty);
//
//int32_t
//AVX512_SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim);
//
//int32_t
//INT4_IP_avx512_impl(const void* p1_vec, const void* p2_vec, int dim);
//
//double
//INT8_IP_impl(const void* pVect1, const void* pVect2, size_t qty);
//
//int32_t
//INT4_IP_impl(const void* p1_vec, const void* p2_vec, int dim);
//
//int32_t
//GENERIC_SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim);
//
//int32_t
//AVX2_SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim);
//
//int32_t
//SSE_SQ4UniformComputeCodesIP(const uint8_t* codes1, const uint8_t* codes2, uint64_t dim);

int32_t
INT4_IP(const void* p1_vec, const void* p2_vec, int dim) {
#ifdef ENABLE_AVX512
    return INT4_IP_avx512_impl(p1_vec, p2_vec, dim);
#else
    return INT4_IP_impl(p1_vec, p2_vec, dim);
#endif
}

int32_t
INT4_L2_precompute(int32_t norm1, int32_t norm2, const void* p1_vec, const void* p2_vec, int dim) {
    return norm1 + norm2 - 2 * INT4_IP(p1_vec, p2_vec, dim);
}

double
INT8_IP(const void* pVect1v, const void* pVect2v, size_t qty) {
#ifdef ENABLE_AVX512
    return INT8_InnerProduct512_AVX512_impl(pVect1v, pVect2v, qty);
#else
    return INT8_IP_impl(pVect1v, pVect2v, qty);
#endif
}

double
INT8_L2_precompute(
    int64_t norm1, double norm2, const void* pVect1v, const void* pVect2v, size_t qty) {
    return norm1 + norm2 - 2.0 * INT8_IP(pVect1v, pVect2v, qty);
}

SimdStatus
setup_simd() {
    L2SqrSIMD16Ext = L2Sqr;
    L2SqrSIMD16ExtResiduals = L2Sqr;
    L2SqrSIMD4Ext = L2Sqr;
    L2SqrSIMD4ExtResiduals = L2Sqr;

    InnerProductSIMD4Ext = InnerProduct;
    InnerProductSIMD16Ext = InnerProduct;
    InnerProductDistanceSIMD16Ext = InnerProductDistance;
    InnerProductDistanceSIMD16ExtResiduals = InnerProductDistance;
    InnerProductDistanceSIMD4Ext = InnerProductDistance;
    InnerProductDistanceSIMD4ExtResiduals = InnerProductDistance;

    SimdStatus ret;

    if (cpuinfo_has_x86_sse()) {
        ret.runtime_has_sse = true;
#ifndef ENABLE_SSE
    }
#else
        L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;
        L2SqrSIMD16ExtResiduals = L2SqrSIMD16ExtResidualsSSE;
        L2SqrSIMD4Ext = L2SqrSIMD4ExtSSE;
        L2SqrSIMD4ExtResiduals = L2SqrSIMD4ExtResidualsSSE;

        InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
        InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
        InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
        InnerProductDistanceSIMD16ExtResiduals = InnerProductDistanceSIMD16ExtResidualsSSE;
        InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;
        InnerProductDistanceSIMD4ExtResiduals = InnerProductDistanceSIMD4ExtResidualsSSE;
    }
    ret.dist_support_sse = true;
#endif

    if (cpuinfo_has_x86_avx()) {
        ret.runtime_has_avx = true;
#ifndef ENABLE_AVX
    }
#else
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
        InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
        InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
    }
    ret.dist_support_avx = true;
#endif

    if (cpuinfo_has_x86_avx2()) {
        ret.runtime_has_avx2 = true;
#ifndef ENABLE_AVX2
    }
#else
    }
    ret.dist_support_avx2 = true;
#endif

    if (cpuinfo_has_x86_avx512f() && cpuinfo_has_x86_avx512dq() && cpuinfo_has_x86_avx512bw() &&
        cpuinfo_has_x86_avx512vl()) {
        ret.runtime_has_avx512f = true;
        ret.runtime_has_avx512dq = true;
        ret.runtime_has_avx512bw = true;
        ret.runtime_has_avx512vl = true;
#ifndef ENABLE_AVX512
    }
#else
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;
        InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
    }
    ret.dist_support_avx512f = true;
    ret.dist_support_avx512dq = true;
    ret.dist_support_avx512bw = true;
    ret.dist_support_avx512vl = true;
#endif

    return ret;
}

DistanceFunc
GetInnerProductDistanceFunc(size_t dim) {
    if (dim % 16 == 0) {
        return vsag::InnerProductDistanceSIMD16Ext;
    } else if (dim % 4 == 0) {
        return vsag::InnerProductDistanceSIMD4Ext;
    } else if (dim > 16) {
        return vsag::InnerProductDistanceSIMD16ExtResiduals;
    } else if (dim > 4) {
        return vsag::InnerProductDistanceSIMD4ExtResiduals;
    } else {
        return vsag::InnerProductDistance;
    }
}

PQDistanceFunc
GetPQDistanceFunc() {
#ifdef ENABLE_AVX
    return PQDistanceAVXFloat256;
#endif
#ifdef ENABLE_SSE
    return PQDistanceSSEFloat256;
#endif
    return PQDistanceFloat256;
}

DistanceFunc
GetL2DistanceFunc(size_t dim) {
    if (dim % 16 == 0) {
        return vsag::L2SqrSIMD16Ext;
    } else if (dim % 4 == 0) {
        return vsag::L2SqrSIMD4Ext;
    } else if (dim > 16) {
        return vsag::L2SqrSIMD16ExtResiduals;
    } else if (dim > 4) {
        return vsag::L2SqrSIMD4ExtResiduals;
    } else {
        return vsag::L2Sqr;
    }
}

}  // namespace vsag
