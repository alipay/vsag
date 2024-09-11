
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

#include "./all_in_one.h"

#include <nlohmann/json.hpp>

using njson = nlohmann::json;

namespace vsag {
tl::expected<void, Error>
AllInOneSerialization::Serialize(const SerializablePtr& index, std::ostream& out_stream) {
    njson metadata;
    uint64_t pos = 0;
    for (auto dc_pair : index->GetDataCellWriter()) {
        auto& dc_name = dc_pair.first;
        uint64_t dc_size = dc_pair.second.get_length();
        auto write_func = [&dc_size, &out_stream](void* src, uint64_t length) -> void {
            if (length >= dc_size) {
                throw std::runtime_error("write excceed buffer size");
            }
            out_stream.write((char*)src, length);
        };
        dc_pair.second.write(write_func);
        metadata["data_cell_offsets"][dc_name] = pos;
        pos += dc_size;
    }
    out_stream.write("VSAG", 4);
    auto metadata_string = metadata.dump();
    out_stream.write(metadata_string.c_str(), metadata_string.size());
    return {};
}

tl::expected<IndexPtr, Error>
AllInOneSerialization::Deserialize(const std::istream& in_stream) {
    return nullptr;
}

}  // namespace vsag
