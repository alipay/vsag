
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

#include "./kv.h"

#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "vsag/binaryset.h"

namespace vsag {

using njson = nlohmann::json;

tl::expected<BinarySet, Error>
KVSerialization::Serialize(const SerializablePtr& index) {
    BinarySet bs;
    njson metadata;
    for (auto dc_pair : index->GetDataCellWriter()) {
        uint64_t dc_size = dc_pair.second.get_length();
        Binary binary{
            .data = std::shared_ptr<int8_t[]>(new int8_t[dc_size]),
            .size = dc_size,
        };
        auto write_func = [&dc_size, &binary](void* src, uint64_t length) -> void {
            if (length >= dc_size) {
                throw std::runtime_error("write excceed buffer size");
            }
            memcpy(binary.data.get(), src, length);
        };
        dc_pair.second.write(write_func);
        bs.Set(dc_pair.first, binary);
    }

    auto metadata_string = metadata.dump();
    Binary metadata_binary{
        .data = std::shared_ptr<int8_t[]>(new int8_t[metadata_string.size()]),
        .size = metadata_string.size(),
    };
    memcpy(metadata_binary.data.get(), metadata_string.c_str(), metadata_string.size());
    bs.Set("_meta", metadata_binary);
    return bs;
}

tl::expected<IndexPtr, Error>
KVSerialization::Deserialize(const BinarySet& binary_set) {
    return nullptr;
}

tl::expected<IndexPtr, Error>
KVSerialization::Deserialize(const ReaderSet& reader_set) {
    return nullptr;
}

}  // namespace vsag
