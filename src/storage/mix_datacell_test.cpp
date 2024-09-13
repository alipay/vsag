
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

#include "mix_datacell.h"

#include "catch2/catch_template_test_macros.hpp"
#include "default_allocator.h"
#include "fixtures.h"
#include "io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/sq8_quantizer.h"

using namespace vsag;

template <typename QuantT, typename IOT>
void
TestMixDataCellBasicUsage(std::unique_ptr<MixDataCell<QuantT, IOT>>& data_cell,
                          uint64_t dim,
                          MetricType metric,
                          float error = 1e-5) {
    uint64_t baseSize = 1000;
    uint64_t querySize = 10;
    auto vectors = fixtures::generate_vectors(baseSize, dim);
    auto querys = fixtures::generate_vectors(querySize, dim);

    auto func = [&]() {
        std::vector<uint64_t> idx(baseSize);
        std::iota(idx.begin(), idx.end(), 0);
        std::shuffle(idx.begin(), idx.end(), std::mt19937(std::random_device()()));
        std::vector<float> dists(baseSize);
        for (auto i = 0; i < querySize; ++i) {
            auto computer = data_cell->FactoryComputer(querys.data() + i * dim);
            data_cell->Query(dists.data(), computer, idx.data(), baseSize);
            float gt;
            for (auto j = 0; j < baseSize; ++j) {
                if (metric == vsag::MetricType::METRIC_TYPE_IP ||
                    metric == vsag::MetricType::METRIC_TYPE_COSINE) {
                    gt = InnerProduct(vectors.data() + idx[j] * dim, querys.data() + i * dim, &dim);
                } else if (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
                    gt = L2Sqr(vectors.data() + idx[j] * dim, querys.data() + i * dim, &dim);
                }
                REQUIRE(std::fabs(gt - dists[j]) < error);
            }
        }
    };
    data_cell->TrainQuantizer(vectors.data(), baseSize);
    data_cell->BatchInsertVector(vectors.data(), baseSize);
    func();
    REQUIRE(data_cell->TotalCount() == baseSize);
}

TEST_CASE("fp32 basic usage in mix data cell", "[ut][flatten_storage]") {
    int dim = 960;
    int M = 32;
    {
        auto allocator = new DefaultAllocator();
        auto graph_io = std::make_shared<MemoryIO>(allocator);
        auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO>>(M);
        graph_data_cell->SetIO(graph_io);

        auto data_cell = std::make_unique<
            MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            graph_data_cell);
        data_cell->SetQuantizer(std::make_unique<FP32Quantizer<>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_L2SQR);
    }
    {
        auto allocator = new DefaultAllocator();
        auto graph_io = std::make_shared<MemoryIO>(allocator);
        auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO>>(M);
        graph_data_cell->SetIO(graph_io);

        auto data_cell = std::make_unique<
            MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>, MemoryIO>>(
            graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<FP32Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_IP);
    }
    {
        auto allocator = new DefaultAllocator();
        auto graph_io = std::make_shared<MemoryIO>(allocator);
        auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO>>(M);
        graph_data_cell->SetIO(graph_io);

        auto data_cell = std::make_unique<
            MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>, MemoryIO>>(
            graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<FP32Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_COSINE);
    }
}

TEST_CASE("sq8 basic usage in mix data cell", "[ut][flatten_storage]") {
    int dim = 960;
    int M = 32;
    {
        auto allocator = new DefaultAllocator();
        auto graph_io = std::make_shared<MemoryIO>(allocator);
        auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO>>(M);
        graph_data_cell->SetIO(graph_io);

        auto data_cell = std::make_unique<
            MixDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>, MemoryIO>>(
            graph_data_cell);
        data_cell->SetQuantizer(std::make_unique<SQ8Quantizer<>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_L2SQR, 0.01);
    }
    {
        auto allocator = new DefaultAllocator();
        auto graph_io = std::make_shared<MemoryIO>(allocator);
        auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO>>(M);
        graph_data_cell->SetIO(graph_io);

        auto data_cell =
            std::make_unique<MixDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>, MemoryIO>>(
                graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_IP>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_IP, 0.01);
    }
    {
        auto allocator = new DefaultAllocator();

        auto graph_io = std::make_shared<MemoryIO>(allocator);
        auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO>>(M);
        graph_data_cell->SetIO(graph_io);

        auto data_cell = std::make_unique<
            MixDataCell<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>, MemoryIO>>(
            graph_data_cell);
        data_cell->SetQuantizer(
            std::make_unique<SQ8Quantizer<vsag::MetricType::METRIC_TYPE_COSINE>>(dim));
        data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                         std::make_unique<MemoryIO>(allocator));
        TestMixDataCellBasicUsage(data_cell, dim, vsag::MetricType::METRIC_TYPE_COSINE, 0.01);
    }
}