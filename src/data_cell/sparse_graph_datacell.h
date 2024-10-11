
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

#pragma once

#include "../utils.h"
#include "graph_interface.h"
#include "io/memory_block_io.h"

namespace vsag {

class SparseGraphDataCell : public GraphInterface {
public:
    using NeighborCountsType = uint32_t;

    SparseGraphDataCell(const nlohmann::json& graph_json_params,
                        const IndexCommonParam& common_param);

    explicit SparseGraphDataCell(Allocator* allocator = nullptr, uint64_t max_degree = 32);

    void
    InsertNeighborsById(uint64_t id, const std::vector<uint64_t>& neighbor_ids) override;

    uint32_t
    GetNeighborSize(uint64_t id) override;

    void
    GetNeighbors(uint64_t id, std::vector<uint64_t>& neighbor_ids) override;

    /****
     * prefetch neighbors of a base point with id
     * @param id of base point
     * @param neighbor_i index of neighbor, 0 for neighbor size, 1 for first neighbor
     */
    void
    Prefetch(uint64_t id, uint64_t neighbor_i) override {
        // TODO(LHT): implement
    }

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

private:
    uint64_t code_line_size_{0};
    Allocator* allocator_{nullptr};
    UnorderedMap<uint64_t, std::unique_ptr<Vector<uint64_t>>> neighbors_;
};

}  // namespace vsag