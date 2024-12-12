
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

#include <cstdint>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <vector>

#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "stream_reader.h"
#include "stream_writer.h"
#include "typing.h"

namespace vsag {

class GraphInterface;
using GraphInterfacePtr = std::shared_ptr<GraphInterface>;

class GraphInterface {
public:
    GraphInterface() = default;

    virtual ~GraphInterface() = default;

    static GraphInterfacePtr
    MakeInstance(const JsonType& graph_interface_param,
                 const IndexCommonParam& common_param,
                 bool is_sparse = false);

public:
    virtual void
    InsertNeighborsById(InnerIdType id, const Vector<InnerIdType>& neighbor_ids) = 0;

    virtual uint32_t
    GetNeighborSize(InnerIdType id) const = 0;

    virtual void
    GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const = 0;

    virtual void
    Resize(InnerIdType new_size) = 0;

    virtual void
    Prefetch(InnerIdType id, uint32_t neighbor_i) = 0;

public:
    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->total_count_);
        StreamWriter::WriteObj(writer, this->max_capacity_);
        StreamWriter::WriteObj(writer, this->maximum_degree_);
    }

    virtual void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->total_count_);
        StreamReader::ReadObj(reader, this->max_capacity_);
        StreamReader::ReadObj(reader, this->maximum_degree_);
    }

    virtual InnerIdType
    InsertNeighbors(const Vector<InnerIdType>& neighbor_ids) {
        this->max_capacity_ = std::max(this->max_capacity_, total_count_ + 1);
        this->InsertNeighborsById(total_count_ + 1, neighbor_ids);
        IncreaseTotalCount(1);
        return total_count_;
    }

    virtual void
    IncreaseTotalCount(InnerIdType count) {
        std::unique_lock<std::mutex> lock(global_);
        total_count_ += count;
    }

    [[nodiscard]] virtual InnerIdType
    TotalCount() const {
        return this->total_count_;
    }

    [[nodiscard]] virtual InnerIdType
    MaximumDegree() const {
        return this->maximum_degree_;
    }

    [[nodiscard]] virtual InnerIdType
    MaxCapacity() const {
        return this->max_capacity_;
    }

    virtual void
    SetMaximumDegree(uint32_t maximum_degree) {
        this->maximum_degree_ = maximum_degree;
    }

    virtual void
    SetTotalCount(InnerIdType total_count) {
        this->total_count_ = total_count;
    };

    virtual void
    SetMaxCapacity(InnerIdType capacity) {
        this->max_capacity_ = std::max(capacity, this->total_count_);
    };

public:
    InnerIdType total_count_{0};

    InnerIdType max_capacity_{1000000};
    uint32_t maximum_degree_{0};

    std::mutex global_{};
};

}  // namespace vsag
