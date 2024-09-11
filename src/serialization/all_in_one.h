
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

#include <iostream>

#include "./serializable.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/index.h"

namespace vsag {

// TODO(wxyu): make sure cursor in available range
class StreamSlice {
public:
    StreamSlice(std::istream& s) {
    }
};

class AllInOneSerialization {
public:
    static tl::expected<void, Error>
    Serialize(const SerializablePtr& index, std::ostream& out_stream);

    static tl::expected<IndexPtr, Error>
    Deserialize(const std::istream& in_stream);

private:
    AllInOneSerialization() = delete;
};

}  // namespace vsag
