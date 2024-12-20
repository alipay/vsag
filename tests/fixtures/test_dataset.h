
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
#include <vector>

#include "vsag/dataset.h"
namespace fixtures {

class TestDataset {
public:
    using DatasetPtr = vsag::DatasetPtr;

    TestDataset(uint64_t dim,
                uint64_t count,
                std::string metric_str = "l2",
                bool with_path = false);

    DatasetPtr base_{nullptr};

    DatasetPtr query_{nullptr};
    DatasetPtr ground_truth_{nullptr};
    int64_t top_k{10};

    DatasetPtr range_query_{nullptr};
    DatasetPtr range_ground_truth_{nullptr};
    std::vector<float> range_radius_{0.0f};

    DatasetPtr filter_query_{nullptr};
    DatasetPtr filter_ground_truth_{nullptr};
    std::function<bool(int64_t)> filter_function_{nullptr};

    const uint64_t dim_;
    const uint64_t count_;
};

using TestDatasetPtr = std::shared_ptr<TestDataset>;
}  // namespace fixtures
