
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

#include "sq8_simd.h"

namespace vsag {
float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const std::vector<float>& lowerBound,
             const std::vector<float>& diff,
             uint64_t dim) {
#if defined(ENABLE_AVX512)
    return AVX512::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
#endif
#if defined(ENABLE_AVX22)
    return AVX2::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return SSE::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
#endif
    return Generic::SQ8ComputeIP(query, codes, lowerBound, diff, dim);
}

float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const std::vector<float>& lowerBound,
                const std::vector<float>& diff,
                uint64_t dim) {
#if defined(ENABLE_AVX512)
    return AVX512::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
#endif
#if defined(ENABLE_AVX22)
    return AVX2::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return SSE::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
#endif
    return Generic::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);
}

float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX512)
    return AVX512::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
#endif
#if defined(ENABLE_AVX22)
    return AVX2::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return SSE::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
#endif
    return Generic::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}

float
SQ8ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
#if defined(ENABLE_AVX512)
    return AVX512::SQ8ComputeCodesL2(codes1, codes2, lowerBound, diff, dim);
#endif
#if defined(ENABLE_AVX22)
    return AVX2::SQ8ComputeCodesL2(codes1, codes2, lowerBound, diff, dim);
#endif
#if defined(ENABLE_SSE)
    return SSE::SQ8ComputeCodesL2(codes1, codes2, lowerBound, diff, dim);
#endif
    return Generic::SQ8ComputeCodesL2(codes1, codes2, lowerBound, diff, dim);
}

}  // namespace vsag

namespace vsag::AVX512 {
float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const std::vector<float>& lowerBound,
             const std::vector<float>& diff,
             uint64_t dim) {
    return vsag::Generic::SQ8ComputeIP(query, codes, lowerBound, diff, dim);  // TODO
}

float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const std::vector<float>& lowerBound,
                const std::vector<float>& diff,
                uint64_t dim) {
    return vsag::Generic::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);  // TODO
}

float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
    return Generic::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}

float
SQ8ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
    return Generic::SQ8ComputeCodesL2(codes1, codes2, lowerBound, diff, dim);
}

}  // namespace vsag::AVX512

namespace vsag::AVX2 {
float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const std::vector<float>& lowerBound,
             const std::vector<float>& diff,
             uint64_t dim) {
    return vsag::Generic::SQ8ComputeIP(query, codes, lowerBound, diff, dim);  // TODO
}
float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const std::vector<float>& lowerBound,
                const std::vector<float>& diff,
                uint64_t dim) {
    return vsag::Generic::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);  // TODO
}
float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
    return Generic::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}

float
SQ8ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
    return Generic::SQ8ComputeCodesL2(codes1, codes2, lowerBound, diff, dim);
}
}  // namespace vsag::AVX2

namespace vsag::SSE {
float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const std::vector<float>& lowerBound,
             const std::vector<float>& diff,
             uint64_t dim) {
    return vsag::Generic::SQ8ComputeIP(query, codes, lowerBound, diff, dim);  // TODO
}
float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const std::vector<float>& lowerBound,
                const std::vector<float>& diff,
                uint64_t dim) {
    return vsag::Generic::SQ8ComputeL2Sqr(query, codes, lowerBound, diff, dim);  // TODO
}
float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
    return Generic::SQ8ComputeCodesIP(codes1, codes2, lowerBound, diff, dim);
}

float
SQ8ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
    return Generic::SQ8ComputeCodesL2(codes1, codes2, lowerBound, diff, dim);
}
}  // namespace vsag::SSE

namespace vsag::Generic {
float
SQ8ComputeIP(const float* query,
             const uint8_t* codes,
             const std::vector<float>& lowerBound,
             const std::vector<float>& diff,
             uint64_t dim) {
    float result = 0.;
    for (uint64_t i = 0; i < dim; ++i) {
        result += query[i] * static_cast<float>(static_cast<float>(codes[i]) / 255.0 * diff[i] +
                                                lowerBound[i]);
    }
    return result;
}
float
SQ8ComputeL2Sqr(const float* query,
                const uint8_t* codes,
                const std::vector<float>& lowerBound,
                const std::vector<float>& diff,
                uint64_t dim) {
    float result = 0.;
    for (uint64_t i = 0; i < dim; ++i) {
        auto val = (query[i] - static_cast<float>(static_cast<float>(codes[i]) / 255.0 * diff[i] +
                                                  lowerBound[i]));
        result += val * val;
    }
    return result;
}
float
SQ8ComputeCodesIP(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
    float result = 0.;
    for (uint64_t i = 0; i < dim; ++i) {
        auto val1 =
            static_cast<float>(static_cast<float>(codes1[i]) / 255.0 * diff[i] + lowerBound[i]);
        auto val2 =
            static_cast<float>(static_cast<float>(codes2[i]) / 255.0 * diff[i] + lowerBound[i]);
        result += val1 * val2;
    }
    return result;
}

float
SQ8ComputeCodesL2(const uint8_t* codes1,
                  const uint8_t* codes2,
                  const std::vector<float>& lowerBound,
                  const std::vector<float>& diff,
                  uint64_t dim) {
    float result = 0.;
    for (uint64_t i = 0; i < dim; ++i) {
        auto val1 =
            static_cast<float>(static_cast<float>(codes1[i]) / 255.0 * diff[i] + lowerBound[i]);
        auto val2 =
            static_cast<float>(static_cast<float>(codes2[i]) / 255.0 * diff[i] + lowerBound[i]);
        result += (val1 - val2) * (val1 - val2);
    }
    return result;
}
}  // namespace vsag::Generic