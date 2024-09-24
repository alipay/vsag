
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

#include "simd/sq8_simd.h"

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

#define TEST_ACCURACY(Func)                                                                        \
    {                                                                                              \
        auto gt = generic::Func(                                                                   \
            vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim);            \
        auto sse =                                                                                 \
            sse::Func(vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim);  \
        auto avx2 =                                                                                \
            avx2::Func(vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
        auto avx512 = avx512::Func(                                                                \
            vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim);            \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                                    \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                                   \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                                 \
    }

TEST_CASE("SQ8 SIMD Compute Codes", "[SQ8 SIMD]") {
    const std::vector<int64_t> dims = {1, 8, 16, 32, 97, 129, 256};
    int64_t count = 100;
    std::vector<uint8_t> vec1, vec2;
    for (const auto& dim : dims) {
        auto vec = fixtures::generate_vectors(count * 2, dim);
        vec1.resize(count * dim * 2, 0);
        std::transform(vec.begin(), vec.begin() + count * dim, vec1.begin(), [](float x) {
            return static_cast<uint8_t>(x * 255.0);
        });
        vec2.resize(count * dim * 2, 0);
        std::transform(vec.begin() + count * dim, vec.end(), vec2.begin(), [](float x) {
            return static_cast<uint8_t>(x * 255.0);
        });
        auto lb = fixtures::generate_vectors(1, dim, true, 186);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(SQ8ComputeCodesIP);
            TEST_ACCURACY(SQ8ComputeCodesL2Sqr);
        }
    }
}

TEST_CASE("SQ8 SIMD Compute", "[SQ8 SIMD]") {
    const std::vector<int64_t> dims = {1, 8, 16, 32, 97, 129, 256};
    int64_t count = 100;
    for (const auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count * 2, dim);
        std::vector<uint8_t> vec2(count * dim * 2);
        std::transform(vec1.begin() + count * dim, vec1.end(), vec2.begin(), [](float x) {
            return uint64_t(x * 255.0);
        });
        auto lb = fixtures::generate_vectors(1, dim, true, 186);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_ACCURACY(SQ8ComputeIP);
            TEST_ACCURACY(SQ8ComputeL2Sqr);
        }
    }
}

#define BENCHMARK_SIMD_COMPUTE(Simd, Comp)                                                         \
    BENCHMARK_ADVANCED(#Simd #Comp) {                                                              \
        for (int i = 0; i < count; ++i) {                                                          \
            Simd::Comp(vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
        }                                                                                          \
        return;                                                                                    \
    }

TEST_CASE("SQ8 SIMD Compute Benchmark", "[SQ8 SIMD]") {
    const std::vector<int64_t> dims = {256};
    int64_t count = 200;
    int64_t dim = 256;

    auto vec1 = fixtures::generate_vectors(count * 2, dim);
    std::vector<uint8_t> vec2(count * dim * 2);
    std::transform(vec1.begin() + count * dim, vec1.end(), vec2.begin(), [](float x) {
        return uint64_t(x * 255.0);
    });
    auto lb = fixtures::generate_vectors(1, dim, true, 180);
    auto diff = fixtures::generate_vectors(1, dim, true, 6217);
    BENCHMARK_SIMD_COMPUTE(generic, SQ8ComputeIP);
    BENCHMARK_SIMD_COMPUTE(sse, SQ8ComputeIP);
    BENCHMARK_SIMD_COMPUTE(avx2, SQ8ComputeIP);
    BENCHMARK_SIMD_COMPUTE(avx512, SQ8ComputeIP);
}
