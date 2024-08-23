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

#include "default_allocator.h"

#include <fmt/format.h>

namespace vsag {
void*
DefaultAllocator::Allocate(size_t size) {
    auto ptr = malloc(size);
    std::lock_guard<std::mutex> guard(set_mutex_);
    allocated_ptrs_.insert(ptr);
    return ptr;
}

void
DefaultAllocator::Deallocate(void* p) {
    if (!p) {
        return;
    }
    std::lock_guard<std::mutex> guard(set_mutex_);
    if (allocated_ptrs_.find(p) == allocated_ptrs_.end()) {
        throw std::runtime_error(
            fmt::format("deallocate: address {} is not allocated by {}", p, Name()));
    }
    allocated_ptrs_.erase(p);
    free(p);
}

void*
DefaultAllocator::Reallocate(void* p, size_t size) {
    if (!p) {
        return Allocate(size);
    }
    std::lock_guard<std::mutex> guard(set_mutex_);
    if (allocated_ptrs_.find(p) == allocated_ptrs_.end()) {
        throw std::runtime_error(
            fmt::format("reallocate: address {} is not allocated by {}", p, Name()));
    }
    allocated_ptrs_.erase(p);
    auto ptr = realloc(p, size);
    allocated_ptrs_.insert(ptr);
    return ptr;
}

std::string
DefaultAllocator::Name() {
    return "DefaultAllocator";
}

}  // namespace vsag
