
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
#include <functional>
#include <unordered_set>

#include "algorithm/hnswlib/visited_list_pool.h"
#include "io/basic_io.h"
#include "vsag/bitset.h"

namespace vsag {

static const uint64_t SET_SIZE = 1000000;

class FilterDataCell {
public:
    FilterDataCell(Allocator* allocator,
                   int64_t* labels,
                   const std::function<bool(int64_t)>& label_filer,
                   uint64_t data_size)
        : label_filer_(label_filer), allocator_(allocator), data_size_(data_size) {
        labels_ = (int64_t*)allocator_->Allocate(data_size * sizeof(int64_t));
        memcpy(labels_, labels, data_size * sizeof(int64_t));
        visited_list_pool_ = new hnswlib::VisitedListPool(1, data_size_, allocator_);
    };

    ~FilterDataCell() {
        allocator_->Deallocate(labels_);
        delete visited_list_pool_;
    }

    inline bool
    IsValid(uint64_t id, hnswlib::VisitedList* vl) {
        int64_t label = labels_[id];
        return (label_filer_(label) and vl->mass[id] == vl->curV);
    }

    inline void
    SetVisited(uint64_t id, hnswlib::VisitedList* vl) {
        vl->mass[id] = vl->curV;
    }

    hnswlib::VisitedList*
    PopVisitedList() {
        return visited_list_pool_->getFreeVisitedList();
    }

    void
    ReleaseVisitedList(hnswlib::VisitedList* vl) {
        visited_list_pool_->releaseVisitedList(vl);
    }

    inline void
    Prefetch(uint64_t id, hnswlib::VisitedList* vl) {
        _mm_prefetch(vl->mass + id, _MM_HINT_T0);
        _mm_prefetch(labels_ + id, _MM_HINT_T0);
    }

    inline int64_t
    GetLabel(uint64_t id) {
        return labels_[id];
    }

private:
    Allocator* allocator_;

    int64_t* labels_;

    uint64_t data_size_{0};

    std::function<bool(int64_t)> label_filer_;

    std::unordered_set<uint64_t> visited_set_;

    hnswlib::VisitedListPool* visited_list_pool_{nullptr};
};

}  // namespace vsag