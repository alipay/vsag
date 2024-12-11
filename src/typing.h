
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
#include <unordered_map>
#include <unordered_set>
#include <deque>

#include "allocator_wrapper.h"
#include "nlohmann/json.hpp"

namespace vsag {

using InnerIdType = uint32_t;  // inner id's type; index's vector count may less than 2^31 - 1
using LabelType = uint64_t;    // external id's type

using JsonType = nlohmann::json;  // alias for nlohmann::json type

template <typename T>
using UnorderedSet =
    std::unordered_set<T, std::hash<T>, std::equal_to<T>, vsag::AllocatorWrapper<T>>;

template <typename T>
using Vector = std::vector<T, vsag::AllocatorWrapper<T>>;

template <typename T>
using Deque = std::deque<T, vsag::AllocatorWrapper<T>>;

template <typename KeyType, typename ValType>
using UnorderedMap = std::unordered_map<KeyType,
                                        ValType,
                                        std::hash<KeyType>,
                                        std::equal_to<KeyType>,
                                        vsag::AllocatorWrapper<std::pair<const KeyType, ValType>>>;

}  // namespace vsag
