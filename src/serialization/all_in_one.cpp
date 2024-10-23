
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

class StreamWriter : public Writer {
public:
    virtual void
    Write(const void* src, uint64_t length) override {
        if (used_ + length >= capicity_) {
            throw std::runtime_error("write excceed buffer size");
        }
        out_stream_.write((const char*)src, length);
        used_ += length;
    }

public:
    explicit StreamWriter(std::ostream& out_stream, uint64_t capicity)
        : out_stream_(out_stream), Writer(capicity) {
    }

private:
    std::ostream& out_stream_;
};

tl::expected<void, Error>
AllInOneSerialization::Serialize(const SerializablePtr& index, std::ostream& out_stream) {
    njson metadata;
    uint64_t pos = 0;
    for (auto [dc_name, serializer] : index->GetDatacellSerializers()) {
        uint64_t dc_size = serializer->GetDatacellSize();
        auto writer = std::make_shared<StreamWriter>(out_stream, dc_size);
        serializer->InvokeWrite(writer);
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
