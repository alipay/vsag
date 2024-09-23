
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

#include "basic_searcher.h"

namespace vsag {

template <typename GraphTmpl, typename VectorDataStorageTmpl, typename FilterTmpl>
void
BasicSearcher<GraphTmpl, VectorDataStorageTmpl, FilterTmpl>::Optimize(uint32_t ef_search,
                                                                      uint32_t k) {
    uint64_t sample_size = std::min(SAMPLE_SIZE, vector_data_cell_->TotalCount());
    uint64_t dim = vector_data_cell_->GetDim();
    uint64_t code_size = vector_data_cell_->GetCodeSize();

    // baseline performance
    logger::info(fmt::format("=============Start optimization============="));
    logger::debug(fmt::format("====baseline evaluation===="));
    this->prefetch_neighbor_codes_num = 1;
    this->prefetch_cache_line = 1;
    std::shared_ptr<float> raw_data(new float[dim], [](float* p) { delete[] p; });
    auto st = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < sample_size; ++i) {
        vector_data_cell_->DecodeById(raw_data.get(), i);
        KNNSearch(raw_data.get(), ef_search, k);
    }
    auto ed = std::chrono::high_resolution_clock::now();
    double baseline_cost = std::chrono::duration<double>(ed - st).count();

    // configs of neighbor codes computation
    {
        logger::debug(fmt::format("====configs of codes computation evaluation===="));

        // init trying configs
        std::vector<int> try_neighbor_codes_nums(
            std::max(graph_->GetMaximumDegree() / PREFETCH_DEGREE_DIVIDE, PREFETCH_DEGREE_MINIMAL));
        std::vector<int> try_cache_lines(code_size / 64 + 2);
        std::iota(try_neighbor_codes_nums.begin(), try_neighbor_codes_nums.end(), 1);
        std::iota(try_cache_lines.begin(), try_cache_lines.end(), 1);

        // evaluation
        double min_cost = std::numeric_limits<double>::max();
        uint32_t best_neighbor_codes_num = 0, best_cache_line = 0;
        for (auto neighbor_codes_num : try_neighbor_codes_nums) {
            for (auto cache_line : try_cache_lines) {
                this->prefetch_neighbor_codes_num = neighbor_codes_num;
                this->prefetch_cache_line = cache_line;
                st = std::chrono::high_resolution_clock::now();
                for (uint32_t i = 0; i < sample_size; ++i) {
                    vector_data_cell_->DecodeById(raw_data.get(), i);
                    KNNSearch(raw_data.get(), ef_search, k);
                }
                ed = std::chrono::high_resolution_clock::now();
                auto cost = std::chrono::duration<double>(ed - st).count();
                if (cost < min_cost) {
                    min_cost = cost;
                    best_neighbor_codes_num = neighbor_codes_num;
                    best_cache_line = cache_line;
                }
                logger::info(fmt::format("try neighbor_codes_num = {}, cache_line = {}, ",
                                         "gaining {:.2f}% performance improvement",
                                         neighbor_codes_num,
                                         cache_line,
                                         100.0 * (baseline_cost / cost - 1)));
            }
        }
        logger::info(
            fmt::format("setting best neighbor_codes_num = {}, best cache_line = {}, "
                        "gaining {:.2f}% performance improvement",
                        best_neighbor_codes_num,
                        best_cache_line,
                        100.0 * (baseline_cost / min_cost - 1)));
        this->prefetch_neighbor_codes_num = best_neighbor_codes_num;
        this->prefetch_cache_line = best_cache_line;
    }

    // configs of neighbor visit check
    {
        logger::debug(fmt::format("====configs of visit map evaluation===="));

        std::vector<int> try_neighbor_visit_nums(
            std::max(graph_->GetMaximumDegree() / PREFETCH_DEGREE_DIVIDE, PREFETCH_DEGREE_MINIMAL));
        std::iota(try_neighbor_visit_nums.begin(), try_neighbor_visit_nums.end(), 1);

        double min_cost = std::numeric_limits<double>::max();
        uint32_t best_neighbor_visit_num = 0;
        for (auto neighbor_visit_num : try_neighbor_visit_nums) {
            this->neighbor_visit_num = neighbor_visit_num;
            st = std::chrono::high_resolution_clock::now();
            for (uint32_t i = 0; i < sample_size; ++i) {
                vector_data_cell_->DecodeById(raw_data.get(), i);
                KNNSearch(raw_data.get(), ef_search, k);
            }
            ed = std::chrono::high_resolution_clock::now();
            auto cost = std::chrono::duration<double>(ed - st).count();
            if (cost < min_cost) {
                min_cost = cost;
                best_neighbor_visit_num = neighbor_visit_num;
            }
            logger::info(fmt::format("try neighbor_visit_num = {}, ",
                                     "gaining {:.2f}% performance improvement",
                                     neighbor_visit_num,
                                     100.0 * (baseline_cost / cost - 1)));
        }
        logger::info(
            fmt::format("setting best neighbor_visit_num = {}, "
                        "gaining {:.2f}% performance improvement",
                        best_neighbor_visit_num,
                        100.0 * (baseline_cost / min_cost - 1)));
        this->prefetch_neighbor_visit_num = best_neighbor_visit_num;
    }

    vector_data_cell_->SetPrefetchParameters(prefetch_neighbor_codes_num, prefetch_cache_line);

    logger::info("=============Done optimization=============");
}

template <typename GraphTmpl, typename VectorDataTmpl, typename FilterTmpl>
uint32_t
BasicSearcher<GraphTmpl, VectorDataTmpl, FilterTmpl>::visit(
    std::false_type,
    hnswlib::VisitedList* vl,
    std::pair<float, uint64_t>& current_node_pair,
    std::pair<float, uint64_t>& next_node_pair,
    std::vector<uint32_t>& to_be_visited) const {
    uint32_t count_no_visited = 0;
    std::vector<uint64_t> neighbors;

    graph_->Prefetch(next_node_pair.second, 0);

    graph_->GetNeighbors(current_node_pair.second, neighbors);

    for (uint32_t i = 0; i < prefetch_neighbor_visit_num; i++) {
        filter_->Prefetch(neighbors[i]);
    }

    for (uint32_t i = 0; i < neighbors.size(); i++) {
        if (i + prefetch_neighbor_visit_num < neighbors.size()) {
            filter_->Prefetch(neighbors[i + prefetch_neighbor_visit_num]);
        }
        if (filter_->IsValid(neighbors[i])) {
            to_be_visited[count_no_visited++] = i;
        }
        filter_->SetVisited(neighbors[i], vl);
    }
    return count_no_visited;
}
template <typename GraphTmpl, typename VectorDataTmpl, typename FilterTmpl>
uint32_t
BasicSearcher<GraphTmpl, VectorDataTmpl, FilterTmpl>::visit(
    std::true_type,
    hnswlib::VisitedList* vl,
    std::pair<float, uint64_t>& current_node_pair,
    std::pair<float, uint64_t>& next_node_pair,
    std::vector<uint32_t>& to_be_visited) const {
    std::false_type false_type_instance;
    return visit(false_type_instance, vl, current_node_pair, next_node_pair, to_be_visited);
}

template <typename GraphTmpl, typename VectorDataStorageTmpl, typename FilterTmpl>
std::priority_queue<std::pair<float, uint64_t>,
                    std::vector<std::pair<float, uint64_t>>,
                    CompareByFirst>
BasicSearcher<GraphTmpl, VectorDataStorageTmpl, FilterTmpl>::KNNSearch(const float* query,
                                                                       uint32_t ef_search,
                                                                       uint32_t k) const {
    std::priority_queue<std::pair<float, uint64_t>,
                        std::vector<std::pair<float, uint64_t>>,
                        CompareByFirst>
        top_candidates;
    std::priority_queue<std::pair<float, uint64_t>,
                        std::vector<std::pair<float, uint64_t>>,
                        CompareByFirst>
        candidate_set;

    auto computer = vector_data_cell_->FactoryComputer(query);
    auto vl = filter_->PopVisitedList();

    float lower_bound;
    float dist;
    uint64_t candidate_id;
    uint32_t hops = 0;
    uint32_t dist_cmp = 0;
    uint32_t count_no_visited = 0;
    std::vector<uint32_t> to_be_visited(vector_data_cell_->TotalCount() * 2);
    std::vector<float> line_dists(vector_data_cell_->TotalCount() * 2);

    vector_data_cell_->Query(&dist, computer, entry_point_id_, 1);
    top_candidates.emplace(dist, entry_point_id_);
    candidate_set.emplace(-dist, entry_point_id_);

    while (!candidate_set.empty()) {
        hops++;
        std::pair<float, uint64_t> current_node_pair = candidate_set.top();

        if ((-current_node_pair.first) > lower_bound && (top_candidates.size() == ef_search)) {
            break;
        }
        candidate_set.pop();
        std::pair<float, uint64_t> next_node_pair = candidate_set.top();

        count_no_visited = visit(typename is_mix_data_cell<VectorDataStorageTmpl>::type(),
                                 current_node_pair,
                                 next_node_pair,
                                 vl,
                                 to_be_visited);

        dist_cmp += count_no_visited;

        vector_data_cell_->QueryLine(
            line_dists.data(), computer, current_node_pair.second, to_be_visited, count_no_visited);

        for (uint32_t i = 0; i < count_no_visited; i++) {
            dist = line_dists[i];
            candidate_id = to_be_visited[i];
            if (top_candidates.size() < ef_search || lower_bound > dist) {
                candidate_set.emplace(-dist, candidate_id);

                top_candidates.emplace(dist, candidate_id);

                if (top_candidates.size() > ef_search)
                    top_candidates.pop();

                if (!top_candidates.empty())
                    lower_bound = top_candidates.top().first;
            }
        }
    }

    filter_->ReleaseVisitedList(vl);
    return top_candidates;
}

}  // namespace vsag