
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

#pragma once

#include <cstdint>
#include <string>

namespace vsag {

namespace generic {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim);
}  // namespace generic

#if defined(ENABLE_SSE)
namespace sse {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim);
}  // namespace sse
#endif

#if defined(ENABLE_AVX2)
namespace avx2 {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim);
}  // namespace avx2
#endif

#if defined(ENABLE_AVX512)
namespace avx512 {
float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim);
float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim);
}  // namespace avx512
#endif

inline float
FP32ComputeIP(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::FP32ComputeIP(query, codes, dim);
#endif
#if defined(ENABLE_AVX2)
    return avx2::FP32ComputeIP(query, codes, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::FP32ComputeIP(query, codes, dim);
#endif
    return generic::FP32ComputeIP(query, codes, dim);
}

inline float
FP32ComputeL2Sqr(const float* query, const float* codes, uint64_t dim) {
#if defined(ENABLE_AVX512)
    return avx512::FP32ComputeL2Sqr(query, codes, dim);
#endif
#if defined(ENABLE_AVX2)
    return avx2::FP32ComputeL2Sqr(query, codes, dim);
#endif
#if defined(ENABLE_SSE)
    return sse::FP32ComputeL2Sqr(query, codes, dim);
#endif
    return generic::FP32ComputeL2Sqr(query, codes, dim);
}

}  // namespace vsag
