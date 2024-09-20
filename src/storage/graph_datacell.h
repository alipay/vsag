
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
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include "io/basic_io.h"
#include "quantization/quantizer.h"

namespace vsag {

/**
 * built by nn-descent or incremental insertion
 * add neighbors and pruning
 * retrieve neighbors
 */

template <typename IOTmpl>
class GraphDataCell {
public:
    GraphDataCell(uint64_t maximum_degree = 32) : maximum_degree_(maximum_degree){};

    explicit GraphDataCell(const std::string& initializeJson);  // todo

    uint64_t
    InsertNode(const std::vector<uint64_t> neighbor_ids);

    void
    InsertNeighbors(uint64_t id, const std::vector<uint64_t> neighbor_ids) {  // todo
        return;
    };

    void
    Prune(uint64_t id) {
        return;
    }

    uint32_t
    GetNeighborSize(uint64_t id);

    void
    GetNeighbors(uint64_t id, std::vector<uint64_t>& neighbor_ids);

    inline void
    SetMaxCapacity(uint64_t capacity) {
        this->maxCapacity_ = std::max(capacity, this->totalCount_);  // TODO add warning
    }

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>>& io) {
        this->io_.swap(io);
    }

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>>&& io) {
        this->io_.swap(io);
    }

    inline uint64_t
    TotalCount() {
        return this->totalCount_;
    }

    /****
     * prefetch neighbors of a base point with id
     * @param id of base point
     * @param neighbor_i index of neighbor, 0 for neighbor size, 1 for first neighbor
     */
    inline void
    Prefetch(uint64_t id, uint64_t neighbor_i) {
        io_->Prefetch(id * get_single_offset() + neighbor_i * sizeof(uint64_t));
    }

    inline uint64_t
    GetMaximumDegree() {
        return this->maximum_degree_;
    }

private:
    inline uint64_t
    get_single_offset() {
        return maximum_degree_ * sizeof(uint64_t) + sizeof(uint32_t);
    }

private:
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    uint64_t totalCount_{0};

    uint64_t maxCapacity_{1000000};

    uint64_t maximum_degree_{0};
};

template <typename IOTmpl>
uint64_t
GraphDataCell<IOTmpl>::InsertNode(const std::vector<uint64_t> neighbor_ids) {
    auto cur_offset = totalCount_ * this->get_single_offset();
    uint32_t neighbor_size = neighbor_ids.size();
    if (neighbor_size > this->maximum_degree_) {
        neighbor_size = maximum_degree_;
    }
    this->io_->Write(reinterpret_cast<uint8_t*>(&neighbor_size), sizeof(neighbor_size), cur_offset);
    cur_offset += sizeof(neighbor_size);

    this->io_->Write(reinterpret_cast<const uint8_t*>(neighbor_ids.data()),
                     neighbor_size * sizeof(uint64_t),
                     cur_offset);
    cur_offset += neighbor_size * sizeof(uint64_t);

    totalCount_++;
    return totalCount_;
}

template <typename IOTmpl>
uint32_t
GraphDataCell<IOTmpl>::GetNeighborSize(uint64_t id) {
    uint32_t size = 0;
    if (id >= totalCount_) {
        return 0;
    }

    io_->Read(reinterpret_cast<uint8_t*>(&size), sizeof(size), id * this->get_single_offset());

    return size;
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl>::GetNeighbors(uint64_t id, std::vector<uint64_t>& neighbor_ids) {
    uint32_t size = GetNeighborSize(id);
    uint64_t cur_offset = id * this->get_single_offset() + sizeof(size);
    if (size == 0) {
        return;
    }
    neighbor_ids.resize(size, 0);
    for (int i = 0; i < size; i++) {
        io_->Read(reinterpret_cast<uint8_t*>(&neighbor_ids[i]), sizeof(uint64_t), cur_offset);
        cur_offset += sizeof(uint64_t);
    }
}

}  // namespace vsag