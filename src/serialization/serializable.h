
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

#include <vsag/binaryset.h>

#include <cstdint>

namespace vsag {

class Writer {
public:
    virtual void
    Write(const void* src, uint64_t length) = 0;

protected:
    Writer(uint64_t capicity) : capicity_(capicity) {
    }

    uint64_t used_ = 0;
    uint64_t capicity_ = 0;
};
using WriterPtr = std::shared_ptr<Writer>;

class DatacellSerializable {
public:
    virtual void
    InvokeWrite(const WriterPtr& writer) = 0;

    virtual uint64_t
    GetDatacellSize() = 0;

protected:
    DatacellSerializable() = default;
};
using DatacellSerializer = std::shared_ptr<DatacellSerializable>;

class Serializable {
public:
    virtual std::vector<std::pair<std::string, DatacellSerializer>>
    GetDatacellSerializers() = 0;

private:
    Serializable() = default;
};

using SerializablePtr = std::shared_ptr<Serializable>;

}  // namespace vsag
