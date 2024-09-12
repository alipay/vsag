
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

#define TEST_RECALL(Func)                                                                          \
    {                                                                                              \
        auto gt = Generic::Func(                                                                   \
            vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim);            \
        auto sse =                                                                                 \
            SSE::Func(vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim);  \
        auto avx2 =                                                                                \
            AVX2::Func(vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim); \
        auto avx512 = AVX512::Func(                                                                \
            vec1.data() + i * dim, vec2.data() + i * dim, lb.data(), diff.data(), dim);            \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(sse));                                    \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx2));                                   \
        REQUIRE(fixtures::dist_t(gt) == fixtures::dist_t(avx512));                                 \
    }

TEST_CASE("SQ8 SIMD Compute Codes", "[SQ8 SIMD]") {
    const std::vector<int64_t> dims = {1, 8, 16, 32, 97, 256};
    int64_t count = 100;
    for (const auto& dim : dims) {
        auto vec = fixtures::generate_vectors(count * 2, dim);
        std::vector<uint8_t> vec1(count * dim);
        std::transform(vec.begin(), vec.begin() + count * dim, vec1.begin(), [](float x) {
            return uint64_t(x * 255.0);
        });
        std::vector<uint8_t> vec2(count * dim);
        std::transform(vec.begin() + count * dim, vec.end(), vec2.begin(), [](float x) {
            return uint64_t(x * 255.0);
        });
        auto lb = fixtures::generate_vectors(1, dim, true, 186);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_RECALL(SQ8ComputeCodesIP);
            TEST_RECALL(SQ8ComputeCodesL2Sqr);
        }
    }
}

TEST_CASE("SQ8 SIMD Compute", "[SQ8 SIMD]") {
    const std::vector<int64_t> dims = {1, 8, 16, 32, 97, 256};
    int64_t count = 100;
    for (const auto& dim : dims) {
        auto vec1 = fixtures::generate_vectors(count * 2, dim);
        std::vector<uint8_t> vec2(count * dim);
        std::transform(vec1.begin() + count * dim, vec1.end(), vec2.begin(), [](float x) {
            return uint64_t(x * 255.0);
        });
        auto lb = fixtures::generate_vectors(1, dim, true, 186);
        auto diff = fixtures::generate_vectors(1, dim, true, 657);
        for (uint64_t i = 0; i < count; ++i) {
            TEST_RECALL(SQ8ComputeIP);
            TEST_RECALL(SQ8ComputeL2Sqr);
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
    std::vector<uint8_t> vec2(count * dim);
    std::transform(vec1.begin() + count * dim, vec1.end(), vec2.begin(), [](float x) {
        return uint64_t(x * 255.0);
    });
    auto lb = fixtures::generate_vectors(1, dim, true, 180);
    auto diff = fixtures::generate_vectors(1, dim, true, 6217);
    BENCHMARK_SIMD_COMPUTE(Generic, SQ8ComputeIP);
    BENCHMARK_SIMD_COMPUTE(SSE, SQ8ComputeIP);
    BENCHMARK_SIMD_COMPUTE(AVX2, SQ8ComputeIP);
    BENCHMARK_SIMD_COMPUTE(AVX512, SQ8ComputeIP);

}
