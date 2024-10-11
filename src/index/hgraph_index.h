
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

#include <nlohmann/json.hpp>
#include <random>

#include "../utils.h"
#include "algorithm/hnswlib/algorithm_interface.h"
#include "algorithm/hnswlib/visited_list_pool.h"
#include "common.h"
#include "data_cell/flatten_interface.h"
#include "data_cell/graph_interface.h"
#include "index_common_param.h"
#include "vsag/index.h"

namespace vsag {
class HGraphIndex : public Index {
public:
    struct CompareByFirst {
        constexpr bool
        operator()(std::pair<float, uint64_t> const& a,
                   std::pair<float, uint64_t> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    using MaxHeap = std::priority_queue<std::pair<float, uint64_t>,
                                        std::vector<std::pair<float, uint64_t>>,
                                        CompareByFirst>;
    using LabelType = uint64_t;

    HGraphIndex(const nlohmann::json& json_obj, const IndexCommonParam& common_param) noexcept;

    void
    Init();

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        return this->build(base);
    }

    virtual tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override {
        return this->add(base);
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        auto func = [&](int64_t id) -> bool {
            int64_t bit_index = id & ROW_ID_MASK;
            return invalid->Test(bit_index);
        };
        return this->knn_search(query, k, parameters, func);
    }

    // TODO(LHT): implement
    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        return this->knn_search(query, k, parameters, filter);
    }

    // TODO(LHT): implement
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        return Dataset::Make();
    }

    // TODO(LHT): implement
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        return Dataset::Make();
    }

    // TODO(LHT): implement
    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override {
        return Dataset::Make();
    }

    int64_t
    GetNumElements() const override {
        return this->basic_flatten_codes_->TotalCount();
    }

    // TODO(LHT): implement
    int64_t
    GetMemoryUsage() const override {
        return 0;
    }

    // TODO(LHT): implement
    tl::expected<BinarySet, Error>
    Serialize() const override {
        return this->serialize();
    }

    virtual tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        IOStreamWriter writer(out_stream);
        this->serialize(writer);
        return tl::expected<void, Error>();
    }

    virtual tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override{
        IOStreamReader reader(in_stream);
        this->deserialize(reader);
        return tl::expected<void, Error>();
    }

    // TODO(LHT): implement
    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        return tl::expected<void, Error>();
    };

    // TODO(LHT): implement
    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        return tl::expected<void, Error>();
    };

    tl::expected<BinarySet, Error>
    serialize() const;

    void
    serialize(StreamWriter& writer);

    void
    deserialize(StreamReader& reader);

    void
    serialize_basic_info(StreamWriter& writer);

    void
    deserialize_basic_info(StreamReader& reader);

public:
    std::shared_ptr<FlattenInterface> basic_flatten_codes_{nullptr};
    std::shared_ptr<FlattenInterface> high_precise_codes_{nullptr};
    std::vector<std::shared_ptr<GraphInterface>> route_graphs_{};
    std::shared_ptr<GraphInterface> bottom_graph_{nullptr};

    bool use_reorder_{false};

    int64_t dim_{0};
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};

    const nlohmann::json json_obj_{};
    const IndexCommonParam common_param_{};

private:
    tl::expected<std::vector<int64_t>, Error>
    build(const DatasetPtr& base);

    tl::expected<std::vector<int64_t>, Error>
    add(const DatasetPtr& base);

    void
    hnsw_add(const DatasetPtr& base);

    inline int
    get_random_level() {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * mult_;
        return (int)r;
    }

    std::shared_ptr<GraphInterface>
    generate_one_route_graph();

    MaxHeap
    search_one_graph(const float* query,
                     const std::shared_ptr<GraphInterface>& graph,
                     const std::shared_ptr<FlattenInterface>& codes,
                     uint64_t ep,
                     uint64_t ef,
                     hnswlib::BaseFilterFunctor* isIdAllowed = nullptr) const;

    void
    select_edges_by_heuristic(MaxHeap& edges,
                              uint64_t max_size,
                              const std::shared_ptr<FlattenInterface>& flatten);

    uint64_t
    mutually_connect_new_element(uint64_t cur_c,
                                 MaxHeap& top_candidates,
                                 std::shared_ptr<GraphInterface> graph,
                                 std::shared_ptr<FlattenInterface> flatten,
                                 bool is_update);

    tl::expected<DatasetPtr, Error>
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const std::function<bool(int64_t)>& filter) const;

private:
    std::default_random_engine level_generator_{2023};
    double mult_{1.0};

    Allocator* allocator_{nullptr};

    UnorderedMap<LabelType, uint64_t> label_lookup_;
    mutable std::mutex label_lookup_mutex_{};  // lock for label_lookup_

    uint64_t enter_point_id_{UINT64_MAX};
    uint64_t max_level_{0};
    std::mutex global_mutex_;

    // Locks operations with element by label value
    mutable vsag::Vector<std::mutex> label_op_mutex_;

    static const uint64_t MAX_LABEL_OPERATION_LOCKS = 65536;

    uint64_t ef_construct_{200};

    std::shared_ptr<hnswlib::VisitedListPool> pool_{nullptr};

    mutable vsag::Vector<std::recursive_mutex> neighbors_mutex_;
};
}  // namespace vsag
