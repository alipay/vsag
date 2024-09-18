
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

#include "functional"
#include "queue"
#include "storage/graph_datacell.h"

namespace vsag {

enum class START_POINT_STRATEGY { FIXED = 0, RANDOM = 1, HIERARCHICAL = 2, LSH = 3 };

template <typename GraphTmpl, typename VectorDataStorageTmpl, typename LabelStorageTmpl>
class BasicSearcher {
public:
    BasicSearcher(std::shared_ptr<GraphTmpl> graph,
                  std::shared_ptr<VectorDataStorageTmpl> vector_storage,
                  std::shared_ptr<LabelStorageTmpl> label_storage)
        : graph_(graph), vector_storage_(vector_storage), label_storage_(label_storage){};

    ~BasicSearcher() = default;

    virtual void
    InitStartPoint(START_POINT_STRATEGY strategy);

    virtual void
    Optimize(uint32_t ef_search);

    virtual std::priority_queue<std::pair<float, uint64_t>>
    KNNSearch(uint32_t ef_search, uint32_t k, const std::function<bool(int64_t)>& filter);

    virtual void
    RangeSearch(float radius, const std::function<bool(int64_t)>& filter);

private:
    std::shared_ptr<GraphTmpl> graph_;

    std::shared_ptr<VectorDataStorageTmpl> vector_storage_;

    std::shared_ptr<LabelStorageTmpl> label_storage_;
};

}  // namespace vsag