
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

#include <shared_mutex>

#include "../utils.h"
#include "graph_interface.h"
#include "io/memory_block_io.h"

namespace vsag {

class SparseGraphDataCell : public GraphInterface {
public:
    using NeighborCountsType = uint32_t;

    SparseGraphDataCell(const JsonType& graph_param, const IndexCommonParam& common_param);

    explicit SparseGraphDataCell(const SafeAllocatorPtr& allocator = nullptr,
                                 uint32_t max_degree = 32);

    void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) override;

    uint32_t
    GetNeighborSize(InnerIdType id) const override;

    void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const override;

    /****
     * prefetch neighbors of a base point with id
     * @param id of base point
     * @param neighbor_i index of neighbor, 0 for neighbor size, 1 for first neighbor
     */
    void
    Prefetch(InnerIdType id, uint32_t neighbor_i) override {
        // TODO(LHT): implement
    }

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

private:
    uint32_t code_line_size_{0};
    const SafeAllocatorPtr allocator_{nullptr};
    UnorderedMap<InnerIdType, std::unique_ptr<Vector<InnerIdType>>> neighbors_;
    mutable std::shared_mutex neighbors_map_mutex_{};
};

}  // namespace vsag
