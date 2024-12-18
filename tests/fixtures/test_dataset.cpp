
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

#include "test_dataset.h"

#include <algorithm>
#include <cstring>

#include "fixtures.h"
#include "simd/fp32_simd.h"

namespace fixtures {

struct CompareByFirst {
    constexpr bool
    operator()(std::pair<float, int64_t> const& a,
               std::pair<float, int64_t> const& b) const noexcept {
        return a.first > b.first;
    }
};

using MaxHeap = std::priority_queue<std::pair<float, int64_t>,
                                    std::vector<std::pair<float, int64_t>>,
                                    CompareByFirst>;

template <typename T>
static T*
CopyVector(const std::vector<T>& vec) {
    auto result = new T[vec.size()];
    memcpy(result, vec.data(), vec.size() * sizeof(T));
    return result;
}

static TestDataset::DatasetPtr
GenerateRandomDataset(uint64_t dim, uint64_t count, std::string metric_str = "l2") {
    auto base = vsag::Dataset::Make();
    constexpr uint64_t query_count = 100;
    bool need_normalize = (metric_str != "cosine");
    auto vecs =
        fixtures::generate_vectors(count, dim, need_normalize, fixtures::RandomValue(0, 564));
    auto vecs_int8 = fixtures::generate_int8_codes(count, dim, fixtures::RandomValue(0, 564));
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), 10086);
    base->Dim(dim)
        ->Ids(CopyVector(ids))
        ->Float32Vectors(CopyVector(vecs))
        ->Int8Vectors(CopyVector(vecs_int8))
        ->NumElements(count)
        ->Owner(true);
    return base;
}

static std::pair<float*, int64_t*>
CalDistanceFloatMetrix(const vsag::DatasetPtr query,
                       const vsag::DatasetPtr base,
                       std::string metric_str) {
    uint64_t query_count = query->GetNumElements();
    uint64_t base_count = base->GetNumElements();

    auto* result = new float[query_count * base_count];
    auto* ids = new int64_t[query_count * base_count];
    auto dist_func = vsag::FP32ComputeL2Sqr;
    if (metric_str == "ip") {
        dist_func = [](const float* query, const float* codes, uint64_t dim) -> float {
            return 1 - vsag::FP32ComputeIP(query, codes, dim);
        };
    } else if (metric_str == "cosine") {
        dist_func = [](const float* query, const float* codes, uint64_t dim) -> float {
            auto norm_query = std::unique_ptr<float[]>(new float[dim]);
            auto norm_codes = std::unique_ptr<float[]>(new float[dim]);
            vsag::Normalize(query, norm_query.get(), dim);
            vsag::Normalize(codes, norm_codes.get(), dim);
            return 1 - vsag::FP32ComputeIP(norm_query.get(), norm_codes.get(), dim);
        };
    }
    auto dim = base->GetDim();
#pragma omp parallel for schedule(dynamic)
    for (uint64_t i = 0; i < query_count; ++i) {
        MaxHeap heap;
        for (uint64_t j = 0; j < base_count; ++j) {
            auto dist = dist_func(
                query->GetFloat32Vectors() + dim * i, base->GetFloat32Vectors() + dim * j, dim);
            heap.emplace(dist, base->GetIds()[j]);
        }
        auto idx = 0;
        while (not heap.empty()) {
            auto [dist, id] = heap.top();
            result[i * base_count + idx] = dist;
            ids[i * base_count + idx] = id;
            ++idx;
            heap.pop();
        }
    }
    return {result, ids};
}

static vsag::DatasetPtr
CalTopKGroundTruth(const std::pair<float*, int64_t*>& result,
                   uint64_t top_k,
                   uint64_t base_count,
                   uint64_t query_count) {
    auto gt = vsag::Dataset::Make();
    auto* ids = new int64_t[query_count * top_k];
    auto* dists = new float[query_count * top_k];
    for (uint64_t i = 0; i < query_count; ++i) {
        for (int j = 0; j < top_k; ++j) {
            ids[i * top_k + j] = result.second[i * base_count + j];
            dists[i * top_k + j] = result.first[i * base_count + j];
        }
    }
    gt->Dim(top_k)->Ids(ids)->Distances(dists)->Owner(true)->NumElements(query_count);
    return gt;
}

static vsag::DatasetPtr
CalFilterGroundTruth(const std::pair<float*, int64_t*>& result,
                     uint64_t top_k,
                     uint64_t div,
                     uint64_t mod,
                     uint64_t base_count,
                     uint64_t query_count) {
    auto gt = vsag::Dataset::Make();
    auto* ids = new int64_t[query_count * top_k];
    auto* dists = new float[query_count * top_k];
    for (uint64_t i = 0; i < query_count; ++i) {
        auto start = 0;
        for (int j = 0; j < top_k; ++j) {
            while (start < base_count) {
                if (result.second[i * base_count + start] % div == mod) {
                    ids[i * top_k + j] = result.second[i * base_count + start];
                    dists[i * top_k + j] = result.first[i * base_count + start];
                    ++start;
                    break;
                }
                ++start;
            }
        }
    }
    gt->Dim(top_k)->Ids(ids)->Distances(dists)->Owner(true)->NumElements(query_count);
    return gt;
}

TestDataset::TestDataset(uint64_t dim, uint64_t count, std::string metric_str)
    : dim_(dim), count_(count) {
    this->base_ = GenerateRandomDataset(dim, count, metric_str);
    constexpr uint64_t query_count = 100;
    this->query_ = GenerateRandomDataset(dim, query_count, metric_str);
    this->filter_query_ = query_;
    this->range_query_ = query_;
    {
        auto result = CalDistanceFloatMetrix(query_, base_, metric_str);
        this->top_k = 10;
        this->ground_truth_ = CalTopKGroundTruth(result, top_k, count, query_count);
        this->range_ground_truth_ = this->ground_truth_;
        this->range_radius_.resize(query_count);
        for (uint64_t i = 0; i < query_count; ++i) {
            this->range_radius_[i] =
                0.5f * (result.first[i * count + top_k] + result.first[i * count + top_k - 1]);
        }
        this->filter_function_ = [](int64_t id) -> bool { return id % 7 != 5; };
        this->filter_ground_truth_ = CalFilterGroundTruth(result, top_k, 7, 5, count, query_count);
        delete[] result.first;
        delete[] result.second;
    }
}
}  // namespace fixtures
