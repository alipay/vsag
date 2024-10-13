
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

#include <nlohmann/json.hpp>
#include <string>

#include "index/index_common_param.h"
#include "quantization/computer.h"
#include "stream_reader.h"
#include "stream_writer.h"

namespace vsag {
class FlattenInterface {
public:
    FlattenInterface() = default;

    static std::shared_ptr<FlattenInterface>
    MakeInstance(const nlohmann::json& json_obj, const IndexCommonParam& common_param);

public:
    virtual void
    Query(float* result_dists,
          std::shared_ptr<ComputerInterface> computer,
          const uint64_t* idx,
          uint64_t id_count) = 0;

    virtual std::shared_ptr<ComputerInterface>
    FactoryComputer(const float* query) = 0;

    virtual void
    Train(const float* data, uint64_t count) = 0;

    virtual void
    InsertVector(const float* vector, uint64_t idx = INT64_MAX) = 0;

    virtual void
    BatchInsertVector(const float* vectors, uint64_t count, uint64_t* idx = nullptr) = 0;

    virtual float
    ComputePairVectors(uint64_t id1, uint64_t id2) = 0;

    virtual void
    Prefetch(uint64_t id) = 0;

public:
    virtual void
    SetMaxCapacity(uint64_t capacity) {
        this->max_capacity_ = capacity;
    };

    [[nodiscard]] virtual const uint8_t*
    GetCodesById(uint64_t id, bool& need_release) const {
        return nullptr;
    }

    virtual bool
    GetCodesById(uint8_t* codes, uint64_t id) const {
        return false;
    }

    [[nodiscard]] virtual uint64_t
    TotalCount() const {
        return this->total_count_;
    }

    virtual void
    Serialize(StreamWriter& writer) {
        StreamWriter::WriteObj(writer, this->total_count_);
        StreamWriter::WriteObj(writer, this->max_capacity_);
        StreamWriter::WriteObj(writer, this->code_size_);
    }

    virtual void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadObj(reader, this->total_count_);
        StreamReader::ReadObj(reader, this->max_capacity_);
        StreamReader::ReadObj(reader, this->code_size_);
    }

public:
    uint64_t total_count_{0};
    uint64_t max_capacity_{1000000};
    uint64_t code_size_{0};
};

}  // namespace vsag
