
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
#include "storage/filter_datacell.h"
#include "storage/graph_datacell.h"
#include "storage/mix_datacell.h"

namespace vsag {

static const uint32_t SAMPLE_SIZE = 10000;
static const uint32_t CENTROID_EF = 500;
static const uint32_t PREFETCH_DEGREE_DIVIDE = 3;
static const uint32_t PREFETCH_DEGREE_MINIMAL = 10;

struct CompareByFirst {
    constexpr bool
    operator()(std::pair<float, uint64_t> const& a,
               std::pair<float, uint64_t> const& b) const noexcept {
        return a.first < b.first;
    }
};

template <typename T>
struct is_mix_data_cell : std::false_type {};

template <typename QuantTmpl, typename IOTmpl, typename GraphTmpl>
struct is_mix_data_cell<MixDataCell<QuantTmpl, IOTmpl, GraphTmpl>> : std::true_type {};

template <typename GraphTmpl, typename VectorDataTmpl, typename FilterTmpl>
class BasicSearcher {
public:
    BasicSearcher(std::shared_ptr<GraphTmpl> graph,
                  std::shared_ptr<VectorDataTmpl> vector,
                  std::shared_ptr<FilterTmpl> filter)
        : graph_(graph), vector_data_cell_(vector), filter_(filter) {
        // set centroid as entry point
        uint64_t sample_size = std::min(SAMPLE_SIZE, vector_data_cell_->TotalCount());
        uint32_t dim = vector_data_cell_->GetDim();

        std::vector<double> query(dim, 0);
        std::vector<float> query_fp(dim, 0);
        std::shared_ptr<float> raw_data(new float[dim], [](float* p) { delete[] p; });
        for (uint64_t i = 0; i < sample_size; ++i) {
            vector_data_cell_->DecodeById(raw_data.get(), i);
            for (int d = 0; d < dim; d++) {
                query[d] += raw_data.get()[d];
            }
        }
        for (uint32_t d = 0; d < dim; d++) {
            query_fp[d] = query[d] / (double)sample_size;
        }

        entry_point_id_ = KNNSearch(query_fp, CENTROID_EF, 1).top().second;
    };

    ~BasicSearcher() = default;

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
    /**
     * used for mix data cell
     */
    uint32_t
    visit(std::true_type,
          hnswlib::VisitedList* vl,
          std::pair<float, uint64_t>& current_node_pair,
          std::pair<float, uint64_t>& next_node_pair,
          std::vector<uint32_t>& to_be_visited) const;

    /**
     * used for flatten data cell
     */
    uint32_t
    visit(std::false_type,
          hnswlib::VisitedList* vl,
          std::pair<float, uint64_t>& current_node_pair,
          std::pair<float, uint64_t>& next_node_pair,
          std::vector<uint32_t>& to_be_visited) const;

private:
    std::shared_ptr<GraphTmpl> graph_;

    std::shared_ptr<VectorDataTmpl> vector_data_cell_;

    std::shared_ptr<FilterTmpl> filter_;

    uint64_t entry_point_id_{0};

    uint32_t prefetch_neighbor_visit_num{0};

    uint32_t prefetch_neighbor_codes_num{0};

    uint32_t prefetch_cache_line{0};
};

}  // namespace vsag