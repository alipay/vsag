
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
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>

#include "algorithm/hnswlib/hnswalg.h"
#include "common.h"
#include "graph_interface.h"
#include "index/index_common_param.h"
#include "io/basic_io.h"
#include "vsag/constants.h"

namespace vsag {

/**
 * built by nn-descent or incremental insertion
 * add neighbors and pruning
 * retrieve neighbors
 */
template <typename IOTmpl, bool is_adapter>
class GraphDataCell;

template <typename IOTmpl>
class GraphDataCell<IOTmpl, false> : public GraphInterface {
public:
    using NeighborCountsType = uint32_t;

    GraphDataCell(const nlohmann::json& graph_json_params,
                  const nlohmann::json& io_json_params,
                  const IndexCommonParam& common_param);

    void
    InsertNeighborsById(uint64_t id, const std::vector<uint64_t>& neighbor_ids) override;

    uint32_t
    GetNeighborSize(uint64_t id) override;

    void
    GetNeighbors(uint64_t id, std::vector<uint64_t>& neighbor_ids) override;

    inline void
    SetIO(std::shared_ptr<BasicIO<IOTmpl>> io) {
        this->io_ = io;
    }

    /****
     * prefetch neighbors of a base point with id
     * @param id of base point
     * @param neighbor_i index of neighbor, 0 for neighbor size, 1 for first neighbor
     */
    void
    Prefetch(uint64_t id, uint64_t neighbor_i) override {
        io_->Prefetch(id * this->code_line_size_ + sizeof(NeighborCountsType) +
                      neighbor_i * sizeof(uint64_t));
    }

    void
    Serialize(StreamWriter& writer) override;

    void
    Deserialize(StreamReader& reader) override;

private:
    std::shared_ptr<BasicIO<IOTmpl>> io_{nullptr};

    uint64_t code_line_size_{0};
};

template <typename IOTmpl>
GraphDataCell<IOTmpl, false>::GraphDataCell(const nlohmann::json& graph_json_params,
                                            const nlohmann::json& io_json_params,
                                            const IndexCommonParam& common_param) {
    this->io_ = std::make_shared<IOTmpl>(io_json_params, common_param);
    if (graph_json_params.contains(GRAPH_PARAM_MAX_DEGREE)) {
        this->maximum_degree_ = graph_json_params[GRAPH_PARAM_MAX_DEGREE];
    }

    if (graph_json_params.contains(GRAPH_PARAM_INIT_MAX_CAPACITY)) {
        this->max_capacity_ = graph_json_params[GRAPH_PARAM_INIT_MAX_CAPACITY];
    }

    this->code_line_size_ = this->maximum_degree_ * sizeof(uint64_t) + sizeof(NeighborCountsType);
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl, false>::InsertNeighborsById(uint64_t id,
                                                  const std::vector<uint64_t>& neighbor_ids) {
    if (neighbor_ids.size() > this->maximum_degree_) {
        logger::error(fmt::format(
            "insert neighbors count {} more than {}", neighbor_ids.size(), this->maximum_degree_));
    }
    this->max_capacity_ = std::max(this->max_capacity_, id + 1);
    auto start = id * this->code_line_size_;
    NeighborCountsType neighbor_count = neighbor_ids.size();
    this->io_->Write((uint8_t*)(&neighbor_count), sizeof(neighbor_count), start);
    start += sizeof(neighbor_count);
    this->io_->Write(
        (uint8_t*)(neighbor_ids.data()), neighbor_ids.size() * sizeof(uint64_t), start);
}

template <typename IOTmpl>
uint32_t
GraphDataCell<IOTmpl, false>::GetNeighborSize(uint64_t id) {
    auto start = id * this->code_line_size_;
    NeighborCountsType result = 0;
    this->io_->Read((uint8_t*)(&result), sizeof(result), start);
    return result;
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl, false>::GetNeighbors(uint64_t id, std::vector<uint64_t>& neighbor_ids) {
    auto start = id * this->code_line_size_;
    NeighborCountsType neighbor_count = 0;
    this->io_->Read((uint8_t*)(&neighbor_count), sizeof(neighbor_count), start);
    neighbor_ids.resize(neighbor_count);
    start += sizeof(neighbor_count);
    this->io_->Read((uint8_t*)(neighbor_ids.data()), neighbor_ids.size() * sizeof(uint64_t), start);
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl, false>::Serialize(StreamWriter& writer) {
    GraphInterface::Serialize(writer);
    this->io_->Serialize(writer);
    StreamWriter::WriteObj(writer, this->code_line_size_);
}

template <typename IOTmpl>
void
GraphDataCell<IOTmpl, false>::Deserialize(StreamReader& reader) {
    GraphInterface::Deserialize(reader);
    this->io_->Deserialize(reader);
    StreamReader::ReadObj(reader, this->code_line_size_);
}

}  // namespace vsag