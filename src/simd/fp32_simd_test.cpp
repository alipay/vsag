
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

#define TEST_RECALL(Func)                                                              \
    {                                                                                  \
        auto gt = Generic::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);    \
        auto sse = SSE::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);       \
        auto avx2 = AVX2::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim);     \
        auto avx512 = AVX512::Func(vec1.data() + i * dim, vec2.data() + i * dim, dim); \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                        \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                       \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                     \
    };

TEST_CASE("FP32Compute", "[FP32SIMD]") {
    const std::vector<int64_t> dims = {1, 8, 16, 32, 256};
    int64_t count = 100;
    for (const auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count * 2, dim);
        std::vector<float> vec2(vec1.begin() + count, vec1.end());
        for (uint64_t i = 0; i < count; ++i) {
            TEST_RECALL(FP32ComputeIP);
            TEST_RECALL(FP32ComputeL2Sqr);
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

TEST_CASE("FP32 benchmark", "[benchmark]") {
    int64_t count = 1000;
    int64_t dim = 256;
    auto vec1 = fixtures::generate_vectors(count * 2, dim);
    std::vector<float> vec2(vec1.begin() + count, vec1.end());
    BENCHMARK_SIMD_COMPUTE(Generic, FP32ComputeIP);
    BENCHMARK_SIMD_COMPUTE(SSE, FP32ComputeIP);
    BENCHMARK_SIMD_COMPUTE(AVX2, FP32ComputeIP);
    BENCHMARK_SIMD_COMPUTE(AVX512, FP32ComputeIP);

    BENCHMARK_SIMD_COMPUTE(Generic, FP32ComputeL2Sqr);
    BENCHMARK_SIMD_COMPUTE(SSE, FP32ComputeL2Sqr);
    BENCHMARK_SIMD_COMPUTE(AVX2, FP32ComputeL2Sqr);
    BENCHMARK_SIMD_COMPUTE(AVX512, FP32ComputeL2Sqr);
}
