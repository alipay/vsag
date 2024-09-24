
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

#include "basic_searcher.h"

#include "catch2/catch_template_test_macros.hpp"
#include "default_allocator.h"
#include "fixtures.h"
#include "io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/sq8_quantizer.h"
#include "storage/graph_datacell.h"

using namespace vsag;

TEST_CASE("search with alg_hnsw", "[ut][basic_searcher]") {
    // data attr
    uint32_t base_size = 1000;
    uint32_t query_size = 100;
    uint64_t dim = 960;

    // build and search attr
    uint32_t M = 32;
    uint32_t ef_construction = 100;
    uint32_t ef_search = 300;
    uint32_t k = ef_search;
    float redundant_rate = 0.5;
    uint64_t fixed_entry_point_id = 0;
    uint64_t DEFAULT_MAX_ELEMENT = 1;

    // data preparation
    auto base_vectors = fixtures::generate_vectors(base_size, dim, true);
    std::vector<int64_t> ids(base_size);
    std::iota(ids.begin(), ids.end(), 0);

    // alg_hnsw
    auto allocator = new DefaultAllocator();
    auto space = std::make_shared<hnswlib::L2Space>(dim);
    auto io = std::make_shared<MemoryIO>(allocator);
    auto alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(),
                                                   DEFAULT_MAX_ELEMENT,
                                                   allocator,
                                                   M / 2,
                                                   ef_construction,
                                                   Options::Instance().block_size_limit());
    for (int64_t i = 0; i < base_size; ++i) {
        auto successful_insert =
            alg_hnsw->addPoint((const void*)(base_vectors.data() + i * dim), ids[i]);
        REQUIRE(successful_insert == true);
    }

    // graph data cell
    auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO, true>>(alg_hnsw);
    using GraphTmpl = std::remove_pointer_t<decltype(graph_data_cell.get())>;

    // vector data cell
    auto vector_data_cell =
        std::make_shared<MixDataCell<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>,
                                     MemoryIO,
                                     GraphDataCell<MemoryIO, true>>>(graph_data_cell);
    vector_data_cell->SetQuantizer(
        std::make_unique<FP32Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(dim));
    vector_data_cell->SetIO(std::make_unique<MemoryIO>(allocator),
                            std::make_unique<MemoryIO>(allocator));

    vector_data_cell->TrainQuantizer(base_vectors.data(), base_size);
    vector_data_cell->BatchInsertVector(base_vectors.data(), base_size);
    vector_data_cell->MakeRedundant(redundant_rate);
    using VectorDataTmpl = std::remove_pointer_t<decltype(vector_data_cell.get())>;

    // searcher
    auto searcher = std::make_shared<BasicSearcher<GraphTmpl, VectorDataTmpl>>(
        allocator, graph_data_cell, vector_data_cell);

    searcher->SetEntryPoint(fixed_entry_point_id);
    //    searcher->Optimize(ef_search, k);

    for (int i = 0; i < query_size; i++) {
        std::unordered_set<uint64_t> valid_set, set;
        auto result = searcher->KNNSearch(base_vectors.data() + i * dim, ef_search, k);
        auto valid_result = alg_hnsw->searchBaseLayerST<false, false>(
            fixed_entry_point_id, base_vectors.data() + i * dim, ef_search, nullptr);
        REQUIRE(result.size() == valid_result.size());

        for (int j = 0; j < k - 1; j++) {
            valid_set.insert(valid_result.top().second);
            set.insert(result.top().second);
            result.pop();
            valid_result.pop();
        }
        for (auto id : set) {
            REQUIRE(valid_set.find(id) != valid_set.end());
        }
        for (auto id : valid_set) {
            REQUIRE(set.find(id) != set.end());
        }
        REQUIRE(result.top().second == valid_result.top().second);
        REQUIRE(result.top().second == ids[i]);
    }
}