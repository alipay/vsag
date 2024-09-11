
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
#include <xmmintrin.h>

#include <cstring>

#include "basic_io.h"
#include "vsag/allocator.h"
namespace vsag {
class MemoryIO : public BasicIO<MemoryIO> {
public:
    explicit MemoryIO(Allocator* allocator) : allocator_(allocator) {
        start_ = reinterpret_cast<uint8_t*>(allocator_->Allocate(MIN_SIZE));
        currentSize_ = MIN_SIZE;
    }

    ~MemoryIO() {
        allocator_->Deallocate(start_);
    }

    inline void
    WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset);

    inline bool
    ReadImpl(uint8_t* data, uint64_t size, uint64_t offset) const;

    [[nodiscard]] inline const uint8_t*
    ReadImpl(uint64_t size, uint64_t offset) const;

    inline bool
    MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const;

    inline void
    PrefetchImpl(uint64_t offset, uint64_t cacheLine = 64);

private:
    [[nodiscard]] inline bool
    CheckValidOffset(uint64_t size) const {
        return size <= currentSize_;
    }

    void
    CheckAndRealloc(uint64_t size) {
        if (CheckValidOffset(size)) {
            return;
        }
        start_ = reinterpret_cast<uint8_t*>(allocator_->Reallocate(start_, size));
        currentSize_ = size;
    }

    Allocator* allocator_{nullptr};

    uint8_t* start_{nullptr};

    uint64_t currentSize_ = 0;

    static const uint64_t MIN_SIZE = 1024;
};

void
MemoryIO::WriteImpl(const uint8_t* data, uint64_t size, uint64_t offset) {
    CheckAndRealloc(size + offset);
    memcpy(start_ + offset, data, size);
}

bool
MemoryIO::ReadImpl(uint8_t* data, uint64_t size, uint64_t offset) const {
    bool ret = CheckValidOffset(size + offset);
    if (ret) {
        memcpy(data, start_ + offset, size);
    }
    return ret;
}

const uint8_t*
MemoryIO::ReadImpl(uint64_t size, uint64_t offset) const {
    if (CheckValidOffset(size + offset)) {
        return start_ + offset;
    }
    return nullptr;
}
bool
MemoryIO::MultiReadImpl(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
    bool ret = true;
    for (uint64_t i = 0; i < count; ++i) {
        ret &= this->ReadImpl(datas, sizes[i], offsets[i]);
        datas += sizes[i];
    }
    return ret;
}
void
MemoryIO::PrefetchImpl(uint64_t offset, uint64_t cacheLine) {
    _mm_prefetch(this->start_ + offset, _MM_HINT_T0);  // todo
}

}  // namespace vsag