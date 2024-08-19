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

#include "vsag/binaryset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/readerset.h"

namespace vsag {

class DiskANN;

class DiskannSerialization {
public:
    static tl::expected<BinarySet, Error>
    Serialize(const DiskANN& diskann);

    // static tl::expected<void, Error>
    // Serialize(const DiskANN& diskann, std::ostream& out_stream);

    static tl::expected<void, Error>
    Deserialize(DiskANN& diskann, const BinarySet& binary_set);

    static tl::expected<void, Error>
    Deserialize(DiskANN& diskann, const ReaderSet& reader_set);

    // static tl::expected<void, Error>
    // Deserialize(DiskANN& diskann, std::istream& in_stream);

    static BinarySet
    EmptyBinaryset();

    static Binary
    ConvertStreamToBinary(const std::stringstream& stream);

    static void
    ConvertBinaryToStream(const Binary& binary, std::stringstream& stream);
};

}  // namespace vsag
