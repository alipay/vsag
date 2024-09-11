
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

#include "./serializable.h"
#include "vsag/binaryset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/index.h"

namespace vsag {

class KVSerialization {
public:
    static tl::expected<BinarySet, Error>
    Serialize(const SerializablePtr& index);

    static tl::expected<IndexPtr, Error>
    Deserialize(const BinarySet& binary_set);

    static tl::expected<IndexPtr, Error>
    Deserialize(const ReaderSet& reader_set);

private:
    KVSerialization() = delete;
};

}  // namespace vsag
