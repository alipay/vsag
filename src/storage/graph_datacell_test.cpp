
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
#include "default_allocator.h"
#include "fixtures.h"
#include "io/memory_io.h"

using namespace vsag;

TEST_CASE("graph data cell basic usage", "[ut][GraphDataCell]") {
    uint32_t M = 32;
    uint32_t data_size = 1000;
    std::vector<size_t> neighbor_sizes(data_size);
    std::random_device rd;
    std::mt19937 gen(rd());

    auto allocator = new DefaultAllocator();
    auto io = std::make_shared<MemoryIO>(allocator);
    auto graph_data_cell = std::make_shared<GraphDataCell<MemoryIO>>(M);
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