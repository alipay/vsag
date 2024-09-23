
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

#include "graph_datacell.h"

#include "catch2/catch_template_test_macros.hpp"
#include "fixtures.h"
#include "io/memory_io.h"

using namespace vsag;

TEST_CASE("basic usage for graph data cell (original)", "[ut][GraphDataCell]") {
    uint32_t M = 32;
    uint32_t data_size = 1000;
    std::vector<size_t> neighbor_sizes(data_size);
    std::random_device rd;
    std::mt19937 gen(rd());

    auto allocator = new DefaultAllocator();
    auto io = std::make_shared<MemoryIO>(allocator);
    auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO, false>>(M);
    graph_data_cell->SetIO(io);

    for (uint32_t i = 0; i < data_size; i++) {
        neighbor_sizes[i] = gen() % (M * 2);
        std::vector<uint64_t> neighbor_ids(neighbor_sizes[i]);
        std::iota(neighbor_ids.begin(), neighbor_ids.end(), i);
        auto cur_size = graph_data_cell->InsertNode(neighbor_ids);
        REQUIRE(cur_size == i + 1);
    }

    for (uint32_t i = 0; i < data_size; i++) {
        std::vector<uint64_t> neighbor_ids;
        uint64_t neighbor_size = graph_data_cell->GetNeighborSize(i);
        graph_data_cell->GetNeighbors(i, neighbor_ids);
        if (neighbor_sizes[i] > M) {
            REQUIRE(neighbor_size == M);
        } else {
            REQUIRE(neighbor_size == neighbor_sizes[i]);
        }
        REQUIRE(neighbor_size == neighbor_ids.size());
        for (uint32_t j = 0; j < neighbor_ids.size(); j++) {
            REQUIRE(neighbor_ids[j] == i + j);
        }
    }
}

TEST_CASE("basic usage for graph data cell (adapter of hnsw)", "[ut][GraphDataCell]") {
    uint32_t M = 32;
    uint32_t data_size = 1000;
    uint32_t ef_construction = 100;
    uint64_t DEFAULT_MAX_ELEMENT = 1;
    uint64_t dim = 960;
    auto vectors = fixtures::generate_vectors(data_size, dim);
    std::vector<int64_t> ids(data_size);
    std::iota(ids.begin(), ids.end(), 0);

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
    for (int64_t i = 0; i < data_size; ++i) {
        auto successful_insert =
            alg_hnsw->addPoint((const void*)(vectors.data() + i * dim), ids[i]);
        REQUIRE(successful_insert == true);
    }

    auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO, true>>(alg_hnsw);

    for (uint32_t i = 0; i < data_size; i++) {
        auto neighbor_size = graph_data_cell->GetNeighborSize(i);
        std::vector<uint64_t> neighbor_ids(neighbor_size);
        graph_data_cell->GetNeighbors(i, neighbor_ids);

        int* data = (int*)alg_hnsw->get_linklist0(i);
        REQUIRE(neighbor_size == alg_hnsw->getListCount((hnswlib::linklistsizeint*)data));

        for (uint32_t j = 0; j < neighbor_size; j++) {
            REQUIRE(neighbor_ids[j] == *(data + j + 1));
        }
    }
}