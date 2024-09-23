
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

#include <chrono>
#include <functional>
#include <queue>
#include <random>

#include "../logger.h"
#include "io/memory_io.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/sq8_quantizer.h"
#include "storage/filter_datacell.h"
#include "storage/graph_datacell.h"
#include "storage/mix_datacell.h"

namespace vsag {

static const uint64_t SAMPLE_SIZE = 10000;
static const uint32_t CENTROID_EF = 500;
static const uint32_t PREFETCH_DEGREE_DIVIDE = 3;
static const uint32_t PREFETCH_MAXIMAL_DEGREE = 1;
static const uint32_t PREFETCH_MAXIMAL_LINES = 1;

struct CompareByFirst {
    constexpr bool
    operator()(std::pair<float, uint64_t> const& a,
               std::pair<float, uint64_t> const& b) const noexcept {
        return a.first < b.first;
    }
};

template <typename GraphTmpl, typename VectorDataTmpl>
class BasicSearcher {
public:
    BasicSearcher(Allocator* allocator,
                  std::shared_ptr<GraphTmpl> graph,
                  std::shared_ptr<VectorDataTmpl> vector);

    ~BasicSearcher() {
        delete visited_list_pool_;  // todo: delete with allocator?
    };

    virtual void
    Optimize(uint32_t ef_search, uint32_t k);

    virtual std::priority_queue<std::pair<float, uint64_t>,
                                std::vector<std::pair<float, uint64_t>>,
                                CompareByFirst>
    KNNSearch(const float* query, uint32_t ef_search, uint32_t k) const;

    virtual std::priority_queue<std::pair<float, uint64_t>>
    RangeSearch(const float* query, float radius) const {
        throw std::runtime_error("Error: not support range search");
    };

    inline void
    SetEntryPoint(uint64_t id) {
        if (id < vector_data_cell_->TotalCount()) {
            entry_point_id_ = id;
        }
    };

private:
    uint32_t
    visit(hnswlib::VisitedList* vl,
          std::pair<float, uint64_t>& current_node_pair,
          std::vector<uint32_t>& to_be_visited_rid,
          std::vector<uint32_t>& to_be_visited_id) const;

private:
    std::shared_ptr<GraphTmpl> graph_;

    std::shared_ptr<VectorDataTmpl> vector_data_cell_;

    hnswlib::VisitedListPool* visited_list_pool_{nullptr};

    uint64_t entry_point_id_{0};

    uint32_t prefetch_neighbor_visit_num{0};

    uint32_t prefetch_neighbor_codes_num{0};

    uint32_t prefetch_cache_line{0};
};

}  // namespace vsag