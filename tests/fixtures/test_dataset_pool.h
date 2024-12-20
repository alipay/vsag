
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

#include <unordered_map>
#include <vector>

#include "test_dataset.h"

namespace fixtures {
class TestDatasetPool {
public:
    TestDatasetPtr
    GetDatasetAndCreate(uint64_t dim,
                        uint64_t count,
                        const std::string& metric_str = "l2",
                        bool with_path = false);

private:
    static std::string
    key_gen(int64_t dim,
            uint64_t count,
            const std::string& metric_str = "l2",
            bool with_path = false);

private:
    std::unordered_map<std::string, TestDatasetPtr> pool_;
    std::vector<std::pair<uint64_t, uint64_t>> dim_counts_;
};
}  // namespace fixtures
