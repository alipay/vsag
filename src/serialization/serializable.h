
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

namespace vsag {

using write_func = std::function<void(void* src, uint64_t length)>;

struct DataCellWriter {
    std::function<void(const write_func& func)> write;
    // std::function<void(std::ofstream stream)> write_to_all_in_one;
    std::function<uint64_t()> get_length;
};

struct Serializable {
public:
    virtual std::vector<std::pair<std::string, DataCellWriter>>
    GetDataCellWriter() = 0;

private:
    Serializable() = default;
};

using SerializablePtr = std::shared_ptr<Serializable>;

}  // namespace vsag
