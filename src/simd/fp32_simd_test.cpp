
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

#include "fp32_simd.h"

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/catch_test_macros.hpp"
#include "fixtures.h"
using namespace vsag;

#ifndef ENABLE_SSE
namespace SSE = Generic;
#endif

#ifndef ENABLE_AVX2
namespace AVX2 = SSE;
#endif

#ifndef ENABLE_AVX512
namespace AVX512 = AVX2;
#endif

#define TEST_ACCURACY(Func)                                                            \
    {                                                                                  \
        auto gt = generic::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);    \
        auto sse = sse::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);       \
        auto avx2 = avx2::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);     \
        auto avx512 = avx512::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim); \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                        \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                       \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                     \
    };

TEST_CASE("FP32 SIMD Compute", "[FP32SIMD]") {
    const std::vector<int64_t> dims = {1, 8, 16, 32, 256};
    int64_t count = 100;
    for (const auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count * 2, dim);
        std::vector<float> vec2(vec1.begin() + count, vec1.end());
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(FP32ComputeIP);
            TEST_ACCURACY(FP32ComputeL2Sqr);
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                 \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                      \
        for (int i = 0; i < count; ++i) {                                  \
            Simd::Comp(vec1.data() + i * dim, vec2.data() + i * dim, dim); \
        }                                                                  \
        return;                                                            \
    }

TEST_CASE("FP32 benchmark", "[FP32SIMD benchmark]") {
    int64_t count = 500;
    int64_t dim = 128;
    auto vec1 = fixtures::generate_vectors(count * 2, dim);
    std::vector<float> vec2(vec1.begin() + count, vec1.end());
    BENCHMARK_SIMD_COMPUTE(generic, FP32ComputeIP);
    BENCHMARK_SIMD_COMPUTE(sse, FP32ComputeIP);
    BENCHMARK_SIMD_COMPUTE(avx2, FP32ComputeIP);
    BENCHMARK_SIMD_COMPUTE(avx512, FP32ComputeIP);

    BENCHMARK_SIMD_COMPUTE(generic, FP32ComputeL2Sqr);
    BENCHMARK_SIMD_COMPUTE(sse, FP32ComputeL2Sqr);
    BENCHMARK_SIMD_COMPUTE(avx2, FP32ComputeL2Sqr);
    BENCHMARK_SIMD_COMPUTE(avx512, FP32ComputeL2Sqr);
}
